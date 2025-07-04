
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

#surely a file with this name hasn't been done before.

#everything here is probably a candidate to be made into a kernel.

def funnyMulti(x, y):
    return torch.sign(x) * torch.sqrt(torch.abs(x * y) + 1e-10)

def unfunnyMulti(x, y):
    return torch.sign(x) * torch.abs((x ** 2) / (y + 1e-10))

@torch.compile(backend='inductor', mode='max-autotune')
def mmnorm(x, dim = -1, scale = None):  
    if(scale is None):
        scale = 1 + 2 / x.nelement() #changed recently
    w_avg = x.mean(dim=dim, keepdim=True)
    w_range = torch.abs(x.max(dim=dim, keepdim=True)[0] - x.min(dim=dim, keepdim=True)[0] + 1e-10)
    return ((x - w_avg) / w_range) * scale

def fast_sin_leakyrelu(x):
     base = F.leaky_relu(x)
     sine_mod = F.relu(torch.sign(x)) * torch.sin(1.25 * x)
     return base + sine_mod


def oozing_floor(x):
     base = F.leaky_relu(x)
     oozing_floor = F.relu(torch.sign(x)) * torch.floor(1.25 * x)
     return base + oozing_floor

def fast_sin_gelu_leaky(x, leaky=0.01):
     base = F.gelu(x)
     sine_mod = torch.where(x >= 0, torch.sin(1.25 * x), x * leaky)
     return base + sine_mod


def fast_sin_gelu(x):
     base = F.gelu(x)
     sine_mod = torch.relu(torch.sign(x)) * torch.sin(1.25 * x)
     return base + sine_mod

def fast_sin_gelu_sinleak(x, leaky=0.01):
     base = F.gelu(x)
     lh = F.leaky_relu(torch.sign(x),negative_slope=leaky)
     sine_mod = lh * torch.sin(1.25 * x)
     return base + sine_mod

def modified_sin_gelu_leaky(x, leaky=0.01):
     sqpi = torch.sqrt(torch.ones_like(x)*2/torch.pi)
     base = 0.5 * x * (1 + F.tanh(sqpi * (x + 0.044715 * x**3)))
     sine_mod = torch.where(x >= 0, torch.sin(1.25 * x), x * leaky)
     return base + sine_mod
def create_memory_causal_mask(memory_length, incoming_length):
    # Create a mask that allows attending to memory tokens and implements
    # causal attention for the incoming tokens
    total_length = memory_length + incoming_length
    mask = torch.zeros((incoming_length, total_length), dtype=torch.bool)
    
    # Allow attending to all memory tokens
    mask[:, :memory_length] = True
    
    # For incoming tokens, implement causal masking
    for i in range(incoming_length):
        mask[i, memory_length:memory_length+i+1] = True
    
    # Convert to attention mask format (False where attention is allowed)
    return ~mask
def create_memory_causal_mask4(memory_length, incoming_length):
    total_length = memory_length + incoming_length
    mask = torch.zeros((total_length,incoming_length))
    mask[incoming_length:,:] = torch.triu(
        torch.ones((incoming_length, incoming_length), dtype=torch.float) * float('-inf'),
        diagonal=1
        )
    return mask

def create_memory_causal_mask2(memory_length, incoming_length):
    total_length = memory_length + incoming_length
    mask = torch.zeros((total_length,total_length))
    mask[incoming_length:,incoming_length:] = torch.triu(
        torch.ones((incoming_length, incoming_length), dtype=torch.float) * float('-inf'),
        diagonal=1
        )
    #mask[incoming_length:,:incoming_length] = torch.triu(
    #    torch.ones((incoming_length, incoming_length), dtype=torch.float) * float('-inf'),
    #    diagonal=1
    #    )
    return mask

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)
#@torch.compile(backend='inductor', mode='max-autotune')
#def quaternion_multiply(q1, q2):
#    """
#    Performs quaternion multiplication of two quaternions or batches of quaternions.
#    Args:
#        q1 (torch.Tensor): First quaternion or batch of quaternions, shape (*, 4),
#                           where the last dimension is [real, i, j, k].
#        q2 (torch.Tensor): Second quaternion or batch of quaternions, shape (*, 4),
#                           compatible with q1's batch dimensions.
#    Returns:
#        torch.Tensor: Quaternion product q1 * q2, shape (*, 4).
#    """
#    a, b, c, d = q1.unbind(-1)
#    e, f, g, h = q2.unbind(-1)
#    real_part = a * e - b * f - c * g - d * h
#    i_part    = a * f + b * e + c * h - d * g
#    j_part    = a * g - b * h + c * e + d * f
#    k_part    = a * h + b * g - c * f + d * e
#    return torch.stack([real_part, i_part, j_part, k_part], dim=-1)

#@torch.compile(backend='inductor', mode='max-autotune')
#def quaternion_multiply_matmul(q1, q2):
#    # q1 and q2 are tensors of shape [..., 4]
#    
#    # Reshape q2 for matmul: from [..., 4] to [..., 4, 1]
#    q2_reshaped = q2.unsqueeze(-1)
#
#    # Create the multiplication matrix from q1
#    w, x, y, z = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
#    
#    # Create the 4 rows of the matrix. Each row is shape [..., 4]
#    row1 = torch.stack([ w, -x, -y, -z], dim=-1)
#    row2 = torch.stack([ x,  w, -z,  y], dim=-1)
#    row3 = torch.stack([ y,  z,  w, -x], dim=-1)
#    row4 = torch.stack([ z, -y,  x,  w], dim=-1)
#
#    # Stack the 4 rows to create the [..., 4, 4] matrix
#    # We stack along a new dimension, which becomes dim=-2
#    q1_matrix = torch.stack([row1, row2, row3, row4], dim=-2)
#
#    # Perform the multiplication
#    # [..., 4, 4] @ [..., 4, 1] -> [..., 4, 1]
#    result_reshaped = torch.matmul(q1_matrix, q2_reshaped)
#
#    # Reshape result back to [..., 4]
#    return result_reshaped.squeeze(-1)

@torch.compile(backend='inductor', mode='max-autotune')
def parallel_quaternion_scan_log_n(local_rotations):
    if local_rotations.dim() == 4 and local_rotations.shape[3] == 1:
        x = local_rotations.squeeze(-1)
    else:
        x = local_rotations.clone() # Clone to avoid in-place modification of input
    B, T, C , _= x.shape
    device = x.device
    # Pad to the next power of 2 for the algorithm to work cleanly.
    next_pow_of_2 = 2**math.ceil(math.log2(T))
    if T != next_pow_of_2:
        # Pad with identity quaternions [1, 0, 0, 0]
        padding_size = next_pow_of_2 - T
        identity_padding = torch.zeros(B, padding_size, C, 4, device=device)
        identity_padding[..., 0] = 1.0
        x = torch.cat([x, identity_padding], dim=1)
    # The main loop runs log(n) times
    num_levels = int(math.log2(x.shape[1]))
    # --- 1. Up-Sweep (Reduction) Phase ---
    # At each level d, we combine elements 2**d apart.
    for d in range(num_levels):
        stride = 2**d
        offset = 2**(d + 1)
        # We select elements to be updated. In PyTorch, it's easier to compute for all
        # and then mask the results back into the original tensor.
        # Get the left-hand side of the multiplication (from a shifted version of x)
        source = x[:, :-stride, :, :]
        # Get the right-hand side (the elements to be updated)
        # Note: We slice the original tensor, not the shifted one
        target = x[:, stride:, :, :]
        # Perform the multiplication in parallel for all selected pairs
        res = quaternion_multiply(source, target)
        # Create a mask to update only the correct elements: k where (k+1) % 2**(d+1) == 0
        # This mask identifies the right-most element in each block of size 2**(d+1)
        mask_indices = torch.arange(stride - 1, x.shape[1] - stride, offset, device=device)
        x[:, mask_indices + stride] = res[:, mask_indices]
    # --- 2. Down-Sweep (Scan) Phase ---
    # Clear the last element to identity, as it now holds the sum of the whole sequence
    x[:, -1, :, :] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    for d in range(num_levels - 1, -1, -1):
        stride = 2**d
        offset = 2**(d + 1)
        # Get elements that will be swapped and used in multiplication
        # These are the right-most elements of blocks of size 2**d
        source = x[:, :-stride, :, :]
        # Elements to be updated
        target = x[:, stride:, :, :]
        # Perform the multiplication needed for the scan
        res = quaternion_multiply(source, target)
        # Create the same mask as in the up-sweep to select the right elements
        mask_indices = torch.arange(stride - 1, x.shape[1] - stride, offset, device=device)
        # This is the tricky part of the down-sweep:
        # a) The right element of the pair gets the value from the left element (the prefix).
        # b) The left element gets the new combined value.
        # This is effectively a swap and multiply.
        temp = x[:, mask_indices + stride].clone()   # Value of right element (e.g., x[k])
        x[:, mask_indices + stride] = res[:, mask_indices] # x[k] = op(x[k-stride], x[k])
        x[:, mask_indices] = temp # x[k-stride] = original x[k]
    # The result is now an "exclusive" scan (output[i] = product of inputs 0 to i-1)
    # The original function needs an "inclusive" scan (output[i] = product of inputs 0 to i)
    # We achieve this by shifting the result and multiplying by the original input.
    exclusive_scan = x
    # Shift right and multiply by original input to get inclusive scan
    # [id, q0, q0*q1, q0*q1*q2, ...] -> multiply by [q0, q1, q2, q3, ...]
    final_result = quaternion_multiply(exclusive_scan, local_rotations)
    # Un-pad to original sequence length
    return final_result[:, :T, :]


@torch.compile(backend='inductor', mode='max-autotune')
def parallel_quaternion_scan( local_rotations):
    #todo, make it actually parallel instead of just lying in the title lmao
    seq_len = local_rotations.shape[1]
    H_t = local_rotations[:, 0, :, :]
    cumulative_outputs = [H_t]
    for t in range(1, seq_len):
        H_t = quaternion_multiply(H_t, local_rotations[:, t, :, :])
        cumulative_outputs.append(H_t)
    return torch.stack(cumulative_outputs, dim=1)

@torch.compile(backend='inductor', mode='max-autotune')
def quaternion_multiply(q, p):
    # q and p are tensors of shape [..., 4]
    # Ensure they are contiguous, which is good practice anyway
    q = q.contiguous()
    p = p.contiguous()

    # Extract components
    w1, x1, y1, z1 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    w2, x2, y2, z2 = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    
    # Calculate the product components directly
    # This is the "Hamilton product"
    w_new = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x_new = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y_new = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z_new = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w_new, x_new, y_new, z_new], dim=-1)

@torch.compile(backend='inductor', mode='max-autotune')
def zca_newton_schulz(G, epsilon=1e-5, steps=5, power_iters=10):
    if G.ndim < 2:
        return G
    G = G - G.mean(dim=0, keepdim=True)
    n, d = G.shape
    device, dtype = G.device, G.dtype
    
    # Compute covariance matrix with regularization
    Cov = (1/n) * torch.matmul(G.T, G) + epsilon * torch.eye(d, device=device, dtype=dtype)
    
    # Estimate spectral norm using power iteration
    v = torch.randn(d, 1, device=device, dtype=dtype)
    for _ in range(power_iters):
        u = torch.matmul(Cov, v)
        u = u / u.norm()
        v = torch.matmul(Cov.T, u)
        v = v / v.norm()
    s = torch.matmul(u.T, torch.matmul(Cov, v))#.squeeze()
    
    # Scale Cov to ensure convergence
    B = Cov / s
    Y = torch.eye(d, device=device, dtype=dtype)
    
    # Newton-Schulz iterations for inverse square root of B
    for _ in range(steps):
        Y = 0.5 * torch.matmul(Y, 3 * torch.eye(d, device=device, dtype=dtype) - torch.matmul(Y, torch.matmul(B, Y)))
    
    # Rescale to get Cov^{-1/2}
    W = Y / (s ** 0.5)
    
    return torch.matmul(G, W)


@torch.compile(backend='inductor', mode='max-autotune')
def GradOrth(param, adjusted_grad):
    w = param.view(-1)
    g = adjusted_grad.view(-1)
    w_norm_sq = torch.dot(w, w) + 1e-30
    proj = torch.dot(w, g) / w_norm_sq
    g_orth = g - proj * w
    g_norm = g.norm(2)
    g_orth_norm = g_orth.norm(2) + 1e-30
    g_orth_scaled = g_orth * (g_norm / g_orth_norm)
    adjusted_grad = (g_orth_scaled.view_as(adjusted_grad))
    return adjusted_grad

def fftnorm(grad,center=0.5,sigma=0.5, dim=None):
    if(grad.ndim>=2):
        adjusted_grad = torch.fft.fft2(grad,dim=dim) 
        #bump width centered on 0 (phase), width of sigma 0.1
        gk = gaussian_kernel(adjusted_grad.real, center, sigma, dim = dim)
        adjusted_grad = torch.complex(gk*(adjusted_grad.real),gk*(adjusted_grad.imag))
        return torch.fft.ifft2(adjusted_grad,dim=dim).real
    else:
        adjusted_grad = torch.fft.fft(grad,dim=dim) 
        #bump width centered on 0 (phase), width of sigma 0.1
        gk = gaussian_kernel(adjusted_grad.real, center, sigma)
        #print(gk)
        adjusted_grad = torch.complex(gk*(adjusted_grad.real),gk*(adjusted_grad.imag))
        return torch.fft.ifft(adjusted_grad).real

def fft_bmean_csquish_add(x, x2):
    b,t,c = x.size()
    halfft2 = torch.fft.fft(x2,dim=-1)
    halfft = torch.fft.fft(x,dim=-1)
    xc = halfft.mean(dim=0).unsqueeze(0)
    xc = torch.complex(xc.real[:,:,c//4:-c//4],xc.imag[:,:,c//4:-c//4])
    if((x2 != 0).any()):
        xc += halfft2
        xc /= 2
    
    return torch.fft.ifft(xc,dim=-1).real


#def fft_bmean_csquish_add(x, x2):
#    b,t,c = x.size()
#    halfft2 = torch.fft.fft(x2,dim=-1)
#    halfft = torch.fft.fft(x,dim=-1)
#    xc = halfft
#    if((x2 != 0).any()):
#        xc += halfft2
#        xc /= 2
#    xc = torch.complex(xc.real[:,:,c//4:-c//4],xc.imag[:,:,c//4:-c//4])
#    
#    return torch.fft.ifft(xc,dim=-1).real

def fft_bmean_tsquish_add(x, x2):
    b,t,c = x.size()
    halfft2 = torch.fft.fft(x2,dim=1)
    halfft = torch.fft.fft(x,dim=1)
    xc = halfft.mean(dim=0).unsqueeze(0)
    xc = torch.complex(xc.real[:,t//4:-t//4,:],xc.imag[:,t//4:-t//4,:])
    if((x2 != 0).any()):
        xc += halfft2
        xc /= 2
    
    return torch.fft.ifft(xc,dim=1).real


def fft_trunc_tsquish(x):
    b,t,c = x.size()
    halfft = torch.fft.fft(x,dim=1)
    xc = torch.complex(halfft.real[:,t//4:-t//4,:],halfft.imag[:,t//4:-t//4,:])
    return torch.fft.ifft(xc,dim=1).real

def fft_trunc_csquish(x):
    b,t,c = x.size()
    halfft = torch.fft.fft(x,dim=-1)
    xc = torch.complex(halfft.real[:,:,c//4:-c//4],halfft.imag[:,:,c//4:-c//4])
    #if(xc.isnan().any):
    #    print("nan complx")
    o = torch.fft.ifft(xc,dim=-1).real
    #if(o.isnan().any):
    #    print("nan out")
    return o 
    
def gaussian_kernel(grad, center_offset: float, sigma = 3.0, dim=None) -> torch.Tensor:
    size = grad.size(dim)
    device= grad.device
    dtype= grad.dtype
    
    # Calculate total elements and validate input
    num_elements = torch.prod(torch.tensor(size, dtype=dtype, device=device))
    # Generate position indices
    indices = torch.arange(num_elements, dtype=dtype, device=device)
    
    # Calculate Gaussian parameters
    center = center_offset * (num_elements - 1)
    #max_distance = max(center, (num_elements - 1) - center)
    sigma = sigma * (num_elements - 1)
    
    # Compute Gaussian values and reshape
    kernel = torch.exp(-(indices - center).pow(2) / (2 * sigma**2))
    return kernel.view(size)



def mmdiem( a :torch.Tensor,b :torch.Tensor):
    numel = a.numel()# torch.sqrt(torch.ones(1,device=a.device,dtype=a.dtype)* a.numel())
    afl = a.flatten()
    bfl = b.flatten()
    arange = afl.max() - afl.min() + 1e-10
    brange = bfl.max() - bfl.min() + 1e-10
    anorm = (afl - afl.mean()) / (arange)
    bnorm = (bfl - bfl.mean()) / (brange)
    variance = torch.sqrt(a.var()*b.var()+1e-10)
    return torch.dot(anorm, bnorm) /  numel / variance # + (torch.abs(arange-brange)) + torch.abs(a.var() - b.var())

def fast_sin_relu(x):
     base = F.relu(x)
     sine_mod = F.relu(torch.sign(x)) * torch.sin(1.25 * x)
     return base + sine_mod


def quaternionize(x):
    b,t,n=x.shape
    x4d = x.view(b, t, n//4, 4)
    x_norm = x4d.norm(dim=-1, keepdim=True) 
    x_normalized = x4d / x_norm             
    return x_normalized