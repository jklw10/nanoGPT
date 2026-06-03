import torch
import torch.nn as nn
import torch.nn.functional as F

from parts import quantizer

def unbroadcast(grad, target_tensor):
    """Sums out broadcasted dimensions to match the shape of target_tensor."""
    if grad.shape == target_tensor.shape:
        return grad
    # Sum along dimensions prepended during broadcasting
    ndim_diff = grad.ndim - target_tensor.ndim
    if ndim_diff > 0:
        grad = grad.sum(dim=tuple(range(ndim_diff)))
    # Sum along dimensions where the target has size 1
    for dim, size in enumerate(target_tensor.shape):
        if size == 1:
            grad = grad.sum(dim=dim, keepdim=True)
    return grad


class BinaryXOR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        # Standard logical XOR for 0/1 float tensors
        return (a != b).float()

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        
        # Gradient with respect to 'a' is grad_output * (1 - 2*b)
        grad_a = grad_output * (1.0 - 2.0 * b)
        # Gradient with respect to 'b' is grad_output * (1 - 2*a)
        grad_b = grad_output * (1.0 - 2.0 * a)
        
        return unbroadcast(grad_a, a), unbroadcast(grad_b, b)


class TopKHot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k):
        _, indices = torch.topk(x, k=k, dim=-1)
        k_hot = torch.zeros_like(x).scatter_(-1, indices, 1.0)
        ctx.save_for_backward(x)
        ctx.k = k
        return k_hot

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        k = ctx.k
        with torch.no_grad():
            # Find where we want the outputs to be 1 to minimize loss
            _, topk_indices = torch.topk(-g, k=k, dim=-1)
            hard_target = torch.zeros_like(x).scatter_(-1, topk_indices, 1.0)
        # Target propagation gradient
        grad_x = torch.sigmoid(x) - hard_target
        return grad_x, None
    
    # Helper to make it easier to call
def xor_topk(x, w, k):
    y_x = quantizer.ThresHot.apply(x, k)
    y_w = quantizer.ThresHot.apply(w, k)
    return BinaryXOR.apply(y_x, y_w)

# --- Verification script ---
if __name__ == "__main__":
    # Inputs and weights with gradients enabled
    x = torch.randn(2, 5, requires_grad=True)
    w = torch.randn(1, 5, requires_grad=True)  # Broadcastable weight
    k = 2

    # Forward Pass
    z = xor_topk(x, w, k)
    
    # Define a simple target state we want to reach
    target = torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 1.0, 0.0]])
    
    # Calculate Mean Squared Error loss
    loss = F.mse_loss(z, target)
    
    # Backward Pass
    loss.backward()
    
    print("Forward XOR output:\n", z)
    print("\nGradient on x:\n", x.grad)
    print("\nGradient on w:\n", w.grad)