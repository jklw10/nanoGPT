import torch
import time
from typing import Tuple

# Ensure reproducible results
torch.manual_seed(42)

# Global configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
B, D = 2048, 1024  # Batch size and Feature dimension
GROUPS = 8         # Number of groups for Sliced ZCA
ITERS = 5          # Number of iterations for Newton-Schulz ZCA
BETA = 0.05        # Decay rate for online update

print(f"Running benchmark on: {DEVICE.upper()}")
print(f"Matrix Dimensions: Batch Size = {B}, Features = {D}")

# =========================================================================
# 1. Whitening Algorithms
# =========================================================================

def zca_whitening_svd(X: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Baseline ZCA Whitening using standard Eigenvalue Decomposition (EVD)."""
    X_centered = X - X.mean(dim=0, keepdim=True)
    N = X_centered.shape[0]
    cov = (X_centered.T @ X_centered) / (N - 1) + eps * torch.eye(D, device=X.device, dtype=DTYPE)
    # EVD / eigh is an O(D^3) bottleneck
    L, U = torch.linalg.eigh(cov)
    L_inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.clamp(L, min=1e-7)))
    W = U @ L_inv_sqrt @ U.T
    return X_centered @ W.T

def zca_whitening_sliced(X: torch.Tensor, groups: int = 8, eps: float = 1e-5) -> torch.Tensor:
    """Sliced ZCA Whitening operating independently on groups of channels."""
    B_sz, D_sz = X.shape
    m = D_sz // groups
    X_centered = X - X.mean(dim=0, keepdim=True)
    
    # Reshape features to [groups, B, m]
    X_reshaped = X_centered.view(B_sz, groups, m).transpose(0, 1)
    
    # Batch covariance computation: [groups, m, m]
    cov = torch.bmm(X_reshaped.transpose(1, 2), X_reshaped) / (B_sz - 1)
    cov = cov + eps * torch.eye(m, device=X.device, dtype=DTYPE).unsqueeze(0)
    
    # Batch EVD
    L, U = torch.linalg.eigh(cov)
    L_inv_sqrt = 1.0 / torch.sqrt(torch.clamp(L, min=1e-7))
    
    # Batch reconstruction of the projection matrices
    W = torch.bmm(U * L_inv_sqrt.unsqueeze(1), U.transpose(1, 2))
    
    # Batch projection
    X_whitened = torch.bmm(X_reshaped, W)
    return X_whitened.transpose(0, 1).contiguous().view(B_sz, D_sz)

def zca_whitening_newton_schulz(X: torch.Tensor, iters: int = 5, eps: float = 1e-5) -> torch.Tensor:
    """SVD-Free ZCA Whitening using iterative matrix multiplications (Newton-Schulz)."""
    B_sz, D_sz = X.shape
    X_centered = X - X.mean(dim=0, keepdim=True)
    cov = (X_centered.T @ X_centered) / (B_sz - 1) + eps * torch.eye(D_sz, device=X.device, dtype=DTYPE)
    
    # Trace normalization to ensure the spectral radius is strictly less than 1
    norm = cov.trace()
    Y = cov / norm
    
    # Coupled Newton-Schulz iteration for A^(-1/2)
    P = Y.clone()
    Q = torch.eye(D_sz, device=X.device, dtype=DTYPE)
    I3 = 3.0 * torch.eye(D_sz, device=X.device, dtype=DTYPE)
    
    for _ in range(iters):
        QP = Q @ P
        P = 0.5 * P @ (I3 - QP)
        Q = 0.5 * (I3 - QP) @ Q
        
    # Scale back by the normalization factor
    W = Q / torch.sqrt(norm)
    return X_centered @ W.T

# =========================================================================
# 2. Online Update Algorithms
# =========================================================================

def update_inv_cov_recompute(cov_prev: torch.Tensor, x_new: torch.Tensor, beta: float, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
    """Incremental covariance update followed by a full matrix inversion (O(D^3))."""
    cov_new = (1.0 - beta) * cov_prev + beta * torch.outer(x_new, x_new)
    inv_cov_new = torch.linalg.inv(cov_new + eps * torch.eye(D, device=cov_prev.device, dtype=DTYPE))
    return cov_new, inv_cov_new

def update_inv_cov_sherman_morrison(inv_cov_prev: torch.Tensor, x_new: torch.Tensor, beta: float) -> torch.Tensor:
    """Direct inverse covariance update via the Sherman-Morrison identity (O(D^2))."""
    inv_u = inv_cov_prev @ x_new
    denom = (1.0 - beta) + beta * torch.dot(x_new, inv_u)
    num = beta * torch.outer(inv_u, inv_u)
    return (inv_cov_prev - num / denom) / (1.0 - beta)


# =========================================================================
# 3. Compilation Wrapper
# =========================================================================

# Compile all functions using PyTorch Inductor
compiled_svd = torch.compile(zca_whitening_svd)
compiled_sliced = torch.compile(zca_whitening_sliced)
compiled_newton_schulz = torch.compile(zca_whitening_newton_schulz)

compiled_recompute = torch.compile(update_inv_cov_recompute)
compiled_sherman_morrison = torch.compile(update_inv_cov_sherman_morrison)

# =========================================================================
# 4. Benchmark Execution Harness
# =========================================================================

def run_timing(fn, *args, num_warmup: int = 25, num_runs: int = 100) -> float:
    """Helper to warm up and precisely measure the runtime of a function."""
    # Compilation & Warmup phase
    for _ in range(num_warmup):
        _ = fn(*args)
        
    if DEVICE == "cuda":
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(num_runs):
            _ = fn(*args)
        end_event.record()
        torch.cuda.synchronize()
        return start_event.elapsed_time(end_event) / num_runs  # Returned in ms
    else:
        start = time.perf_counter()
        for _ in range(num_runs):
            _ = fn(*args)
        end = time.perf_counter()
        return (end - start) * 1000 / num_runs  # Returned in ms


if __name__ == "__main__":
    # Generate mock inputs
    X_data = torch.randn(B, D, device=DEVICE, dtype=DTYPE)
    cov_init = (X_data.T @ X_data) / (B - 1) + 1e-5 * torch.eye(D, device=DEVICE, dtype=DTYPE)
    inv_cov_init = torch.linalg.inv(cov_init)
    x_step = torch.randn(D, device=DEVICE, dtype=DTYPE)

    print("\n--- Phase 1: Batch Whitening Methods Comparison ---")
    
    # 1. Measure SVD ZCA
    time_svd = run_timing(compiled_svd, X_data)
    print(f"Compiled Baseline SVD ZCA:         {time_svd:8.3f} ms")
    
    # 2. Measure Sliced ZCA
    time_sliced = run_timing(compiled_sliced, X_data, GROUPS)
    print(f"Compiled Sliced ZCA (G={GROUPS}):      {time_sliced:8.3f} ms  ({time_svd / time_sliced:.2f}x speedup)")
    
    # 3. Measure Newton-Schulz ZCA
    time_ns = run_timing(compiled_newton_schulz, X_data, ITERS)
    print(f"Compiled Newton-Schulz (I={ITERS}):   {time_ns:8.3f} ms  ({time_svd / time_ns:.2f}x speedup)")

    print("\n--- Phase 2: Online Covariance Update Methods Comparison ---")
    
    # 1. Measure Covariance Update + Re-Inversion
    time_recomp = run_timing(compiled_recompute, cov_init, x_step, BETA)
    print(f"Compiled Covariance Re-Inversion:   {time_recomp:8.3f} ms")
    
    # 2. Measure Sherman-Morrison Update
    time_sm = run_timing(compiled_sherman_morrison, inv_cov_init, x_step, BETA)
    print(f"Compiled Sherman-Morrison Update:   {time_sm:8.3f} ms  ({time_recomp / time_sm:.2f}x speedup)")

    # =========================================================================
    # 5. Numerical Validation
    # =========================================================================
    print("\n--- Numerical Sanity Check ---")
    with torch.no_grad():
        out_svd = zca_whitening_svd(X_data)
        out_ns = zca_whitening_newton_schulz(X_data, iters=10)
        
        # Verify the features have been successfully decorrelated (Covariance ≈ Identity)
        cov_svd = (out_svd.T @ out_svd) / (B - 1)
        cov_ns = (out_ns.T @ out_ns) / (B - 1)
        identity = torch.eye(D, device=DEVICE, dtype=DTYPE)
        
        err_svd = torch.norm(cov_svd - identity).item()
        err_ns = torch.norm(cov_ns - identity).item()
        
        print(f"L2 distance of SVD Whitened Covariance to Identity: {err_svd:.5f}")
        print(f"L2 distance of N-S Whitened Covariance to Identity: {err_ns:.5f} (with 10 iterations)")