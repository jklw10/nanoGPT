import torch
import torch.optim as optim
import time

# Helper function to compute the exact regularized objective function
def compute_objective(alpha, K_indef, y, lambda_reg):
    pred = torch.matmul(K_indef, alpha)
    loss_mse = 0.5 * torch.sum((y - pred) ** 2)
    # Squeezed to maintain a true scalar rather than a 1x1 matrix
    loss_reg = 0.5 * lambda_reg * torch.sum(alpha * torch.matmul(K_indef, alpha))
    return loss_mse + loss_reg

# 1. Optimized Dual-Signature DCA Optimizer
def run_dca(K_plus, K_minus, y, lambda_reg, max_iter=2, tol=1e-6, jitter=1e-6):
    n = K_plus.size(0)
    K_indef = K_plus - K_minus
    K_indef_sq = torch.matmul(K_indef, K_indef)
    A = K_indef_sq + lambda_reg * K_plus
    
    # Add minor diagonal perturbation to ensure stable Cholesky factorization
    A = A + jitter * torch.eye(n, dtype=A.dtype, device=A.device)
    b_base = torch.matmul(K_indef, y)
    
    alpha = torch.zeros((n, 1), dtype=K_plus.dtype, device=K_plus.device)
    
    start_time = time.time()
    # Compute Cholesky factorization once (O(n^3))
    L = torch.linalg.cholesky(A)
    
    for i in range(max_iter):
        b_t = b_base + lambda_reg * torch.matmul(K_minus, alpha)
        # Solve using the Cholesky factor (O(n^2) per iteration)
        alpha_next = torch.cholesky_solve(b_t, L)
        
        if torch.norm(alpha_next - alpha) < tol:
            break
        alpha = alpha_next
    duration = time.time() - start_time
    return alpha, i + 1, duration

# 2. First-Order Optimizers (SGD and AdamW)
def run_first_order(optimizer_name, K_indef, y, lambda_reg, lr, epochs=2000):
    n = K_indef.size(0)
    alpha = torch.zeros((n, 1), dtype=K_indef.dtype, device=K_indef.device, requires_grad=True)
    
    if optimizer_name == "SGD":
        optimizer = optim.SGD([alpha], lr=lr)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW([alpha], lr=lr)
        
    start_time = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = compute_objective(alpha, K_indef, y, lambda_reg)
        loss.backward()
        optimizer.step()
        
        # Monitor for divergence
        if torch.isnan(alpha).any() or torch.isinf(alpha).any():
            return None, epoch, time.time() - start_time, True
            
    duration = time.time() - start_time
    return alpha.detach(), epochs, duration, False

# =====================================================================
# Execution and Comparison
# =====================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate Synthetic Dataset
    n_samples = 250
    d_features = 5
    X = torch.randn(n_samples, d_features, device=device)
    
    # Construct Indefinite Kernel (K_plus - K_minus)
    dist_sq = torch.cdist(X, X, p=2)**2
    K_plus = torch.exp(-0.5 * dist_sq)
    K_minus = torch.exp(-2.0 * dist_sq)
    K_indef = K_plus - K_minus
    
    # Generate targets
    true_alpha = torch.randn(n_samples, 1, device=device)
    y = torch.matmul(K_indef, true_alpha) + torch.randn(n_samples, 1, device=device) * 0.1
    
    lambda_reg = 0.1
    
    print("--- STARTING BENCHMARK ---\n")
    
    # Run DCA
    alpha_dca, dca_iters, dca_time = run_dca(K_plus, K_minus, y, lambda_reg)
    dca_mse = torch.mean((y - torch.matmul(K_indef, alpha_dca))**2).item()
    print(f"[DCA Optimizer]")
    print(f"  Iterations: {dca_iters}")
    print(f"  Time taken: {dca_time:.4f} seconds")
    print(f"  Final MSE : {dca_mse:.6f}\n")
    
    # Run AdamW
    adam_lr = 1e-3
    alpha_adam, adam_epochs, adam_time, adam_div = run_first_order("AdamW", K_indef, y, lambda_reg, lr=adam_lr)
    if not adam_div:
        adam_mse = torch.mean((y - torch.matmul(K_indef, alpha_adam))**2).item()
        print(f"[AdamW Optimizer (lr={adam_lr})]")
        print(f"  Epochs    : {adam_epochs}")
        print(f"  Time taken: {adam_time:.4f} seconds")
        print(f"  Final MSE : {adam_mse:.6f}\n")
    else:
        print(f"[AdamW Optimizer] Diverged due to non-convex landscape!\n")
        
    # Run SGD
    sgd_lr = 1e-4
    alpha_sgd, sgd_epochs, sgd_time, sgd_div = run_first_order("SGD", K_indef, y, lambda_reg, lr=sgd_lr)
    if not sgd_div:
        sgd_mse = torch.mean((y - torch.matmul(K_indef, alpha_sgd))**2).item()
        print(f"[SGD Optimizer (lr={sgd_lr})]")
        print(f"  Epochs    : {sgd_epochs}")
        print(f"  Time taken: {sgd_time:.4f} seconds")
        print(f"  Final MSE : {sgd_mse:.6f}\n")
    else:
        print(f"[SGD Optimizer] Diverged due to non-convex landscape!\n")