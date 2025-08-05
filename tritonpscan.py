import math
import time
import torch
import torch.nn as nn
import triton
import triton.language as tl
import model
import utils
import torch.nn.functional as F
from triton.ops.matmul_perf_model import early_config_prune

from torch.amp import custom_fwd, custom_bwd
# --- Triton Kernels ---
@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1
@triton.jit
def gelu(x):
    return 0.5 * x * (1 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
@triton.jit
def gelu_backward(x, dy):
    """
    Backward pass for GeLU.
    """
    cdf = 0.5 * (1.0 + tanh((0.7978845608 * (x + 0.044715 * x * x * x))))
    deriv = cdf + x * 0.5 * (1.0 - tanh((0.7978845608 * (x + 0.044715 * x * x * x)))**2) * (0.7978845608 * (1.0 + 0.134145 * x * x))
    return dy * deriv

@triton.jit
def fused_scanner(x, y, ln_weight, ln_bias, w1, b1, w2, b2):
    xy = tl.cat(y, x, can_reorder=True)
    mean = tl.sum(xy, axis=1) / xy.shape[1]
    var = tl.sum((xy - mean[:, None]) * (xy - mean[:, None]), axis=1) / xy.shape[1]
    inv_std = tl.rsqrt(var + 1e-5)
    normed = (xy - mean[:, None]) * inv_std[:, None]

    normed = normed * ln_weight + ln_bias
    h = tl.dot(normed, w1) + b1
    h = gelu(h)
    return tl.dot(h, w2) + b2

@triton.jit
def blelloch_scan_kernel(
    x_ptr, out_ptr,
    ln_weight_ptr, ln_bias_ptr,
    w1_ptr, b1_ptr,
    w2_ptr, b2_ptr,
    identity_ptr,
    B, C,
    stride_b, stride_t, stride_c,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr
):
    b_idx = tl.program_id(0)
    t_idx = tl.arange(0, BLOCK_T)
    c_idx = tl.arange(0, BLOCK_C)

    # compute offsets
    offsets = b_idx * stride_b + t_idx[:, None] * stride_t + c_idx[None, :]
    x = tl.load(x_ptr + offsets, mask=c_idx[None, :] < C)
    acc = tl.zeros_like(x)

    # store identity (e.g. input) to acc
    tl.store(out_ptr + offsets, x, mask=c_idx[None, :] < C)

    # --- Upsweep ---
    d = 1
    while d < BLOCK_T:
        mask = (t_idx % (2 * d) == (2 * d - 1)) & (t_idx >= d)

        left_off = b_idx * stride_b + (t_idx - d)[:, None] * stride_t + c_idx[None, :]
        right_off = b_idx * stride_b + t_idx[:, None] * stride_t + c_idx[None, :]

        left = tl.load(out_ptr + left_off, mask=mask[:, None] & (c_idx[None, :] < C))
        right = tl.load(out_ptr + right_off, mask=mask[:, None] & (c_idx[None, :] < C))

        fused = fused_scanner(left, right, ln_weight_ptr, ln_bias_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr)
        tl.store(out_ptr + right_off, fused, mask=mask[:, None] & (c_idx[None, :] < C))

        d *= 2

    # set last element to identity (optional — already initialized above)

    # --- Downsweep ---
    d = BLOCK_T // 2
    while d >= 1:
        mask = (t_idx % (2 * d) == (2 * d - 1)) & (t_idx >= d)

        left_off = b_idx * stride_b + (t_idx - d)[:, None] * stride_t + c_idx[None, :]
        right_off = b_idx * stride_b + t_idx[:, None] * stride_t + c_idx[None, :]

        left = tl.load(out_ptr + left_off, mask=mask[:, None] & (c_idx[None, :] < C))
        right = tl.load(out_ptr + right_off, mask=mask[:, None] & (c_idx[None, :] < C))

        tl.store(out_ptr + left_off, right, mask=mask[:, None] & (c_idx[None, :] < C))
        fused = fused_scanner(right, left, ln_weight_ptr, ln_bias_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr)
        tl.store(out_ptr + right_off, fused, mask=mask[:, None] & (c_idx[None, :] < C))

        d //= 2

    # --- Final inclusive scan ---
    #y = fused_scanner(acc, x, ln_weight_ptr, ln_bias_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr)
    
    x = tl.load(x_ptr + offsets, mask=c_idx[None, :] < C)
    acc = tl.load(out_ptr + offsets, mask=c_idx[None, :] < C)

    y = fused_scanner(acc, x, ln_weight_ptr, ln_bias_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr)

    tl.store(out_ptr + offsets, y, mask=c_idx[None, :] < C)
    # Store result
    #tl.store(out_ptr + x_offsets, y, mask=c_idx[None, :] < C)

def pscan_triton(x, scanner, identity):
    B, T, C = x.shape
    BLOCK_T = 2 ** math.ceil(math.log2(T))
    BLOCK_C = 2 ** math.ceil(math.log2(C))

    # Pad x and output buffer
    x_padded = F.pad(x, (0, 0, 0, BLOCK_T - T), value=0).contiguous()
    out = torch.empty_like(x_padded)

    # Extract weights from scanner
    ln_weight = scanner.ln.weight.data.contiguous()
    ln_bias = scanner.ln.bias.data.contiguous()
    w1 = scanner.linear_1.weight.data.contiguous()
    b1 = scanner.linear_1.bias.data.contiguous()
    w2 = scanner.linear_2.weight.data.contiguous()
    b2 = scanner.linear_2.bias.data.contiguous()

    # Ensure identity is [C]
    identity = identity.expand(C).contiguous()

    blelloch_scan_kernel[(B,)](
        x_padded, out,
        ln_weight, ln_bias,
        w1, b1,
        w2, b2,
        identity,
        B, C,
        x_padded.stride(0), x_padded.stride(1), x_padded.stride(2),
        BLOCK_T=BLOCK_T,
        BLOCK_C=BLOCK_C
    )

    return out[:, :T]

# --- Verification and Benchmark ---
if __name__ == '__main__':
    # Config, small first
    B, T, C = 10, 32, 128
    device = 'cuda'
    dtype = torch.bfloat16

    # Inputs
    print(" Verifying correctness and Warmup start")
    x_in = torch.randn(B, T, C, device=device, dtype=dtype, requires_grad=True)
    identity = nn.Parameter(torch.randn(C, device=device, dtype=dtype))
    scanner = model.Scanner(d_model=C).to(device=device).to(dtype=dtype) 
    
    t0 = time.time()
    y_triton_for_bwd = pscan_triton(x_in, scanner, identity)#warmup autotune
    t1 = time.time()
    y_torch_for_bwd = utils.pscan(x_in, scanner, identity)#warmup autotune
    t2 =time.time()
    
    print(f" Warmup time, triton: {(t1-t0)*1000:.2f} ms, torch: {(t2-t1)*1000:.2f} ms")
    print(f" difference: {torch.nn.functional.mse_loss(y_triton_for_bwd,y_torch_for_bwd)}")
    print(" Benchmarking ")
    
    t0 = time.time()
    y_triton_for_bwd = pscan_triton(x_in, scanner, identity)
    t1 = time.time()
    y_triton_for_bwd.sum().backward(retain_graph=True)
    t2 =time.time()
    print(f" Backward Pass: {(t1-t0):.2f} ns, bkw {(t2-t1):.2f} ns")
    
    t0 = time.time()
    y_torch_for_bwd = utils.pscan(x_in, scanner, identity)
    t1 = time.time()
    y_torch_for_bwd.sum().backward(retain_graph=True)
    t2 =time.time()
    print(f"torch Forward Pass: {(t1-t0):.2f}ns, bkw {(t2-t1):.2f} ns")
    