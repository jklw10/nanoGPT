
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

#surely a file with this name hasn't been done before.

#everything here is probably a candidate to be made into a kernel.

def funnyMulti(x, y):
    return torch.sign(x) * torch.sqrt(torch.abs(x * y))

def unfunnyMulti(x, y):
    return torch.sign(x) * torch.abs((x ** 2) / y)

def mmnorm(x, dim = -1, scale = None):  
    if(scale is None):
        input_size = x.size(dim)
        scale = (1 + 2 / input_size)
    w_avg = x.mean(dim=dim, keepdim=True)
    w_range = torch.abs(x.max(dim=dim, keepdim=True)[0] - x.min(dim=dim, keepdim=True)[0] + 1e-10)
    return ((x - w_avg) / w_range) * scale


def create_memory_causal_mask(memory_length, incoming_length):
    total_length = memory_length + incoming_length
    mask = torch.tril(torch.full((total_length, total_length), float('-inf')))    
    
    # Memory can attend to all memory tokens
    mask[:total_length, :memory_length] = 0
    mask[:memory_length, :total_length] = 0
    
    
    return mask
def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)
def quaternion_multiply(q1, q2):
    """
    Performs quaternion multiplication of two quaternions or batches of quaternions.
    Args:
        q1 (torch.Tensor): First quaternion or batch of quaternions, shape (*, 4),
                           where the last dimension is [real, i, j, k].
        q2 (torch.Tensor): Second quaternion or batch of quaternions, shape (*, 4),
                           compatible with q1's batch dimensions.
    Returns:
        torch.Tensor: Quaternion product q1 * q2, shape (*, 4).
    """
    a, b, c, d = q1.unbind(-1)
    e, f, g, h = q2.unbind(-1)
    real_part = a * e - b * f - c * g - d * h
    i_part    = a * f + b * e + c * h - d * g
    j_part    = a * g - b * h + c * e + d * f
    k_part    = a * h + b * g - c * f + d * e
    return torch.stack([real_part, i_part, j_part, k_part], dim=-1)

def zca_newton_schulz(G, epsilon=1e-5, steps=2, power_iters=2):
    if G.ndim < 3:
        return G
    original_shape = G.shape
    G = G.view(-1, original_shape[-1])  # Flatten to (batch*seq_len, features)
            
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
    s = torch.matmul(u.T, torch.matmul(Cov, v))
    
    # Scale Cov to ensure convergence
    B = Cov / s
    Y = torch.eye(d, device=device, dtype=dtype)
    
    # Newton-Schulz iterations for inverse square root of B
    for _ in range(steps):
        Y = 0.5 * torch.matmul(Y, 3 * torch.eye(d, device=device, dtype=dtype) - torch.matmul(Y, torch.matmul(B, Y)))
    
    # Rescale to get Cov^{-1/2}
    W = Y / (s ** 0.5)
    
    return torch.matmul(G, W).view(original_shape)

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
