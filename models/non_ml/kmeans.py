import torch
import time
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch._inductor.config.disable_cpp_codegen = True
torch._dynamo.config.cache_size_limit = 128

# =====================================================================
# 1. JIT Kernels (Atomic-Free & Strictly Static Shapes)
# =====================================================================

def _power_iteration_step(X: torch.Tensor, v: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    z = torch.mv(X, v) - torch.dot(mu, v)
    v_next = torch.mv(X.T, z) - z.sum() * mu
    return v_next / (torch.linalg.norm(v_next) + 1e-9)

def safe_compile(fn):
    return torch.compile(fn)

power_iteration_step_compiled = safe_compile(_power_iteration_step)

def get_normal_power_iteration(X_static: torch.Tensor, iters: int = 2) -> torch.Tensor:
    d = X_static.shape[1]
    mu = X_static.mean(dim=0)
    v = torch.randn(d, device=device, dtype=X_static.dtype)
    v = v / (torch.linalg.norm(v) + 1e-9)
    for _ in range(iters):
        v = power_iteration_step_compiled(X_static, v, mu)
    return v

# =====================================================================
# 2. Ultra-Fast cuBLAS Lloyd & e-means Steps
# =====================================================================

def _lloyd_step_cublas(X: torch.Tensor, C: torch.Tensor, chunk_size: int = 65536) -> torch.Tensor:
    n, d = X.shape
    k = C.shape[0]
    c_half_norm = 0.5 * (C * C).sum(dim=1)
    
    new_C = torch.zeros_like(C)
    counts = torch.zeros(k, 1, device=X.device, dtype=X.dtype)
    
    for i in range(0, n, chunk_size):
        X_chunk = X[i : i + chunk_size]
        scores = torch.addmm(-c_half_norm, X_chunk, C.T)
        assign = torch.argmax(scores, dim=1)
        
        O_chunk = torch.zeros(X_chunk.shape[0], k, device=X.device, dtype=X.dtype)
        O_chunk.scatter_(1, assign.unsqueeze(1), 1.0)
        
        new_C += O_chunk.T @ X_chunk
        counts += O_chunk.sum(dim=0, keepdim=True).T
        
    dead_mask = (counts == 0)
    updated_C = new_C / torch.clamp(counts, min=1.0)
    return torch.where(dead_mask, C, updated_C)

def _emeans_step_cublas(B_X: torch.Tensor, C: torch.Tensor, S: torch.Tensor, N_stat: torch.Tensor, alpha: float):
    c_half_norm = 0.5 * (C * C).sum(dim=1)
    scores = torch.addmm(-c_half_norm, B_X, C.T)
    assign = torch.argmax(scores, dim=1)
    
    k = C.shape[0]
    O = torch.zeros(B_X.shape[0], k, device=B_X.device, dtype=B_X.dtype)
    O.scatter_(1, assign.unsqueeze(1), 1.0)
    
    s_batch = O.T @ B_X
    n_batch = O.sum(dim=0, keepdim=True).T
    
    S = (1.0 - alpha) * S + alpha * s_batch
    N_stat = (1.0 - alpha) * N_stat + alpha * n_batch
    C = S / torch.clamp(N_stat, min=1e-6)
    return C, S, N_stat, scores

# =====================================================================
# 3. Island Slicing Primitives
# =====================================================================

def split_dense_island_fast(proj: torch.Tensor, num_bins: int = 64) -> torch.Tensor:
    n_pts = proj.shape[0]
    sample = proj[torch.randint(0, n_pts, (min(4096, n_pts),), device=proj.device)] if n_pts > 4096 else proj
    min_v, max_v = sample.aminmax()
    span = max_v - min_v
    if span <= 1e-6:
        return min_v
        
    norm_proj = ((sample - min_v) / (span + 1e-7) * (num_bins - 1)).clamp(0, num_bins - 1).long()
    counts = torch.bincount(norm_proj, minlength=num_bins).float()
    
    is_dense = counts > (0.35 * counts.mean())
    dense_idxs = torch.where(is_dense)[0]
    
    if dense_idxs.numel() < 2:
        return min_v + 0.5 * span
        
    first = dense_idxs[0]
    last = dense_idxs[-1]
    
    bin_idxs = torch.arange(num_bins, device=proj.device)
    valid_mask = (bin_idxs >= first) & (bin_idxs <= last)
    empty_between = (~is_dense) & valid_mask
    bin_width = span / float(num_bins)
    
    padded = torch.nn.functional.pad(empty_between.float(), (1, 1), value=0.0)
    diff = padded[1:] - padded[:-1]
    starts = torch.where(diff == 1)[0]
    ends = torch.where(diff == -1)[0]
    
    if starts.numel() > 0:
        lengths = ends - starts
        max_idx = torch.argmax(lengths)
        best_center = (ends[max_idx] - 1).float() - (lengths[max_idx].float() / 2.0)
        return min_v + best_center * bin_width
        
    return min_v + 0.5 * span

# =====================================================================
# 4. Algorithm Implementations
# =====================================================================

# 1. Lloyd's k-means
def run_lloyd_kmeans(X: torch.Tensor, k: int, iters: int = 15) -> torch.Tensor:
    n = X.shape[0]
    C = X[torch.randint(0, n, (k,), device=device)].clone()
    for _ in range(iters):
        C = _lloyd_step_cublas(X, C)
    return C

# 2-4. Divisive Tree Family (Annoy, RP-Tree, Engine v3)
def run_divisive_tree(X: torch.Tensor, k: int, split_rule: str = "dense_island") -> torch.Tensor:
    n, d = X.shape
    clusters = [torch.arange(n, device=device)]
    
    while len(clusters) < k:
        clusters.sort(key=lambda idxs: idxs.shape[0], reverse=True)
        idxs = clusters.pop(0)
        m = idxs.shape[0]
        if m <= 4:
            clusters.append(idxs)
            break
            
        sample_idx = idxs[torch.randint(0, m, (4096,), device=device)]
        w = get_normal_power_iteration(X[sample_idx])
        proj = torch.mv(X, w)[idxs]
            
        if split_rule == "dense_island":
            thresh = split_dense_island_fast(proj)
        elif split_rule == "median":
            sub_p = proj[torch.randint(0, m, (min(m, 4096),), device=device)] if m > 4096 else proj
            thresh = torch.median(sub_p)
        else:  # midpoint
            min_v, max_v = proj.aminmax()
            thresh = 0.5 * (min_v + max_v)
            
        l_mask = proj < thresh
        n_l = l_mask.sum().item()
        
        if n_l < int(0.10 * m) or n_l > int(0.90 * m):
            sub_p = proj[torch.randint(0, m, (min(m, 4096),), device=device)]
            thresh = torch.median(sub_p)
            l_mask = proj < thresh
            n_l = l_mask.sum().item()
            if n_l == 0 or n_l == m:
                half = m // 2
                l_mask = torch.zeros(m, dtype=torch.bool, device=device)
                l_mask[:half] = True
                
        r_mask = ~l_mask
        clusters.append(idxs[l_mask])
        clusters.append(idxs[r_mask])
        
    centroids = torch.empty(k, d, device=device, dtype=X.dtype)
    for i in range(min(k, len(clusters))):
        idxs = clusters[i]
        m = idxs.shape[0]
        if m > 2048:
            sub = idxs[torch.randint(0, m, (2048,), device=device)]
            centroids[i] = X[sub].mean(dim=0)
        elif m > 0:
            centroids[i] = X[idxs].mean(dim=0)
        else:
            centroids[i] = X[torch.randint(0, n, (1,), device=device)].squeeze(0)
            
    return centroids

# =====================================================================
# e-means (Robustized Paper Implementation)
# =====================================================================

def get_emeans_hyperparameters(n: int, k: int, epochs: float):
    B = 0.035 * (k ** 0.13) * (n ** 0.88) * (epochs ** 0.81)
    B = max(k * 2, min(int(round(B)), n))
    B = ((B + 127) // 128) * 128
    B = min(B, n)

    a0 = 0.824 * (k ** -0.46) * (B ** 0.90) * (n ** -0.45) * (epochs ** -0.11)
    a0 = min(max(float(a0), 0.01), 1.0)
    return B, a0

def run_emeans(X: torch.Tensor, k: int, epochs: float = 3.0, init_C: torch.Tensor = None) -> torch.Tensor:
    n, d = X.shape
    B, a0 = get_emeans_hyperparameters(n, k, epochs)
    n_batches = max(1, int((n * epochs) // B))

    C = init_C.clone() if init_C is not None else X[torch.randperm(n, device=device)[:k]].clone()
    N_stat = torch.full((k, 1), float(B) / float(k), device=device, dtype=X.dtype)
    S = C * N_stat

    for step in range(n_batches):
        progress = float(step) / float(n_batches)
        alpha = 0.1 * a0 + 0.5 * (1.0 - 0.1) * a0 * (1.0 + math.cos(math.pi * progress))

        batch_idx = torch.randint(0, n, (B,), device=device)
        B_X = X[batch_idx]

        C, S, N_stat, scores = _emeans_step_cublas(B_X, C, S, N_stat, alpha)

        # Revival is active ONLY in the first 50% of batches to prevent late disruption
        if (step + 1) % 5 == 0 and step < int(0.5 * n_batches):
            mean_n = N_stat.mean().item()
            dead_mask = (N_stat.squeeze(1) < 0.1 * mean_n)
            dead_indices = torch.where(dead_mask)[0]

            if len(dead_indices) > 0:
                # Compute batch point residual distances reusing scores
                max_score, _ = torch.max(scores, dim=1)
                bx_sq = (B_X * B_X).sum(dim=1)
                residual_sq = torch.clamp(bx_sq - 2.0 * max_score, min=0.0)

                # Filter extreme outliers (>6x median) to prevent hijacking centroids
                med_res = torch.median(residual_sq)
                valid_res = residual_sq < (6.0 * med_res + 1e-4)
                weights = torch.where(valid_res, residual_sq, torch.zeros_like(residual_sq))

                if weights.sum() > 0:
                    num_to_sample = min(len(dead_indices), int((weights > 0).sum().item()))
                    chosen_idx = torch.multinomial(weights, num_to_sample, replacement=False)
                    new_pts = B_X[chosen_idx]

                    init_n = float(B) / float(k)
                    target_dead = dead_indices[:num_to_sample]
                    N_stat[target_dead] = init_n
                    S[target_dead] = new_pts * init_n
                    C[target_dead] = new_pts

    # Fast cuBLAS finalization step
    C = _lloyd_step_cublas(X, C)
    return C

def run_hybrid_v3(X: torch.Tensor, k: int) -> torch.Tensor:
    seed_C = run_divisive_tree(X, k=k, split_rule="dense_island")
    return run_emeans(X, k=k, epochs=1.0, init_C=seed_C)

# 8-9. Engine v5 & HYBRID v5 (Residual Power Seeding)
def run_engine_v5_residual(X: torch.Tensor, k: int) -> torch.Tensor:
    n, d = X.shape
    centroids = [X.mean(dim=0)]
    
    sub_n = min(16384, n)
    sub_idx = torch.randint(0, n, (sub_n,), device=device)
    X_sub = X[sub_idx]
    
    while len(centroids) < k:
        C_curr = torch.stack(centroids)
        c_half = 0.5 * (C_curr * C_curr).sum(dim=1)
        scores = torch.addmm(-c_half, X_sub, C_curr.T)
        assign = torch.argmax(scores, dim=1)
        
        counts = torch.bincount(assign, minlength=len(centroids))
        target_cluster = torch.argmax(counts).item()
        
        cluster_mask = (assign == target_cluster)
        pts = X_sub[cluster_mask]
        
        if pts.shape[0] < 8:
            centroids.append(X[torch.randint(0, n, (1,), device=device)].squeeze(0))
            continue
            
        pts_static = pts[torch.randint(0, pts.shape[0], (4096,), device=device)]
        v = get_normal_power_iteration(pts_static, iters=2)
            
        proj = torch.mv(pts, v)
        thresh = split_dense_island_fast(proj)
        
        l_mask = proj < thresh
        n_l = l_mask.sum().item()
        if n_l == 0 or n_l == pts.shape[0]:
            thresh = torch.median(proj)
            l_mask = proj < thresh
            if l_mask.sum().item() == 0 or l_mask.sum().item() == pts.shape[0]:
                half = pts.shape[0] // 2
                l_mask = torch.zeros(pts.shape[0], dtype=torch.bool, device=device)
                l_mask[:half] = True
                
        r_mask = ~l_mask
        centroids[target_cluster] = pts[l_mask].mean(dim=0)
        centroids.append(pts[r_mask].mean(dim=0))
        
    return torch.stack(centroids)

def run_hybrid_v5(X: torch.Tensor, k: int) -> torch.Tensor:
    seed_C = run_engine_v5_residual(X, k=k)
    return run_emeans(X, k=k, epochs=1.0, init_C=seed_C)

# =====================================================================
# 5. Multi-Scale Benchmark Runner
# =====================================================================

def evaluate_mse(X: torch.Tensor, centroids: torch.Tensor) -> float:
    chunk_size = 32768
    total_mse = 0.0
    n = X.shape[0]
    for i in range(0, n, chunk_size):
        chunk = X[i:i + chunk_size]
        dists = torch.cdist(chunk, centroids, p=2)
        total_mse += torch.min(dists * dists, dim=1).values.sum().item()
    return total_mse / n

def run_benchmark():
    configs = [
        {"N": 640000, "D": 1024, "K": 16},
        {"N": 640000, "D": 1024, "K": 64},
    ]
    
    print(f"Hardware: {device.type.upper()} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    algos = [
        ("1. Classic Lloyd k-means (15 iters)", lambda x, k: run_lloyd_kmeans(x, k, iters=15)),
        ("2. Annoy Divisive (Power-Iter + Midpoint)", lambda x, k: run_divisive_tree(x, k, split_rule="midpoint")),
        ("3. RP-Tree Divisive (Power-Iter + Median)", lambda x, k: run_divisive_tree(x, k, split_rule="median")),
        ("4. Engine v3 (Sequential Island Slicing)", lambda x, k: run_divisive_tree(x, k, split_rule="dense_island")),
        ("5. HYBRID v3 (Engine v3 -> e-means 1.0ep)", lambda x, k: run_hybrid_v3(x, k)),
        ("6. Raw e-means (1.0 epochs)", lambda x, k: run_emeans(x, k, epochs=1.0)),
        ("7. Raw e-means (2.0 epochs)", lambda x, k: run_emeans(x, k, epochs=2.0)),
        ("8. Raw e-means (3.0 epochs)", lambda x, k: run_emeans(x, k, epochs=3.0)),
        ("9. Raw e-means (5.0 epochs)", lambda x, k: run_emeans(x, k, epochs=5.0)),
        ("10. Engine v5 (Residual Power Seeding)", lambda x, k: run_engine_v5_residual(x, k)),
        ("11. HYBRID v5 (Engine v5 -> e-means 1.0ep)", lambda x, k: run_hybrid_v5(x, k)),
    ]

    for cfg in configs:
        N, D, K = cfg["N"], cfg["D"], cfg["K"]
        print(f"\nData scale: N={N:,} points | Dimension={D} | Target Clusters k={K}")
        print(f"{'Algorithm':<44} | {'Time (ms)':<10} | {'MSE (Distortion)':<16} | {'Speedup'}")
        print("-" * 84)

        torch.manual_seed(42)
        total_n = N + 20
        X = torch.empty((total_n, D), device=device, dtype=torch.float32)
        pts_per_k = N // K
        for i in range(K):
            center = torch.randn(D, device=device) * 4.0
            torch.randn(pts_per_k, D, device=device, out=X[i * pts_per_k : (i + 1) * pts_per_k])
            X[i * pts_per_k : (i + 1) * pts_per_k].add_(center)
        torch.randn(20, D, device=device, out=X[N:]).mul_(35.0)

        _ = power_iteration_step_compiled(X[:4096], torch.randn(D, device=device, dtype=X.dtype), X[:4096].mean(dim=0))

        if device.type == "cuda":
            torch.cuda.synchronize()

        lloyd_time = None

        for name, fn in algos:
            if device.type == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            centroids = fn(X, K)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if "Lloyd" in name:
                lloyd_time = elapsed_ms

            mse = evaluate_mse(X, centroids)
            speedup_str = f"{lloyd_time / elapsed_ms:.1f}x" if (lloyd_time and elapsed_ms > 0) else "1.0x"
            print(f"{name:<44} | {elapsed_ms:<10.2f} | {mse:<16.3f} | {speedup_str}")

        del X
        if device.type == "cuda":
            torch.cuda.empty_cache()

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    run_benchmark()