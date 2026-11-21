import torch
import time
import math
import os
import numpy as np
import matplotlib.pyplot as plt
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
    O = torch.zeros(B_X.shape[0], k, device=B_X.device, dtype=torch.float32)
    O.scatter_(1, assign.unsqueeze(1), 1.0)
    
    s_batch = O.T @ B_X.to(torch.float32)
    n_batch = O.sum(dim=0, keepdim=True).T
    
    S = (1.0 - alpha) * S + alpha * s_batch
    N_stat = (1.0 - alpha) * N_stat + alpha * n_batch
    
    # Inactive centroids hold coordinates instead of collapsing to zero
    valid = (N_stat.squeeze(1) > 1e-6)
    new_C = (S / torch.clamp(N_stat, min=1e-6)).to(C.dtype)
    C = torch.where(valid.unsqueeze(1), new_C, C)
    return C, S, N_stat, assign, scores

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
# e-means (Modified vs Paper-Faithful)
# =====================================================================

def get_emeans_hyperparameters(n: int, k: int, epochs: float):
    B = 0.035 * (k ** 0.13) * (n ** 0.88) * (epochs ** 0.81)
    B = max(k * 2, min(int(round(B)), n))
    B = ((B + 127) // 128) * 128
    B = min(B, n)

    a0 = 0.824 * (k ** -0.46) * (B ** 0.90) * (n ** -0.45) * (epochs ** -0.11)
    a0 = min(max(float(a0), 0.01), 1.0)
    return B, a0

def run_modified_emeans(X: torch.Tensor, k: int, epochs: float = 3.0, init_C: torch.Tensor = None) -> torch.Tensor:
    n, d = X.shape
    B, a0 = get_emeans_hyperparameters(n, k, epochs)
    n_batches = max(1, int((n * epochs) // B))

    C = init_C.clone() if init_C is not None else X[torch.randperm(n, device=device)[:k]].clone()
    
    N_stat = torch.full((k, 1), 1e-10, device=device, dtype=torch.float32)
    S = C.to(torch.float32) * 1e-10

    for step in range(n_batches):
        progress = float(step) / float(n_batches)
        alpha = a0 * (0.1 + (1.0 - 0.1) * 0.5 * (1.0 + math.cos(math.pi * progress)))

        batch_idx = torch.randint(0, n, (B,), device=device)
        B_X = X[batch_idx]

        C, S, N_stat, assign, scores = _emeans_step_cublas(B_X, C, S, N_stat, alpha)

        # Dead-cluster revival during the first 75% of batches
        if (step + 1) % 5 == 0 and step < int(0.75 * n_batches):
            mean_n = N_stat.mean().item()
            dead_mask = (N_stat.squeeze(1) < 0.1 * mean_n)
            dead_indices = torch.where(dead_mask)[0]

            if len(dead_indices) > 0:
                max_score, _ = torch.max(scores, dim=1)
                bx_sq = (B_X * B_X).sum(dim=1)
                residual_sq = torch.clamp(bx_sq - 2.0 * max_score, min=0.0)

                cap_val = torch.quantile(residual_sq, 0.99)
                weights = torch.clamp(residual_sq, max=cap_val)

                if weights.sum() > 0:
                    num_dead = len(dead_indices)
                    num_to_sample = min(num_dead, int((weights > 0).sum().item()))
                    chosen_idx = torch.multinomial(weights, num_to_sample, replacement=False)
                    new_pts = B_X[chosen_idx]

                    target_dead = dead_indices[:num_to_sample]
                    N_stat[target_dead] = mean_n
                    S[target_dead] = new_pts.to(torch.float32) * mean_n
                    C[target_dead] = new_pts

    C = _lloyd_step_cublas(X, C)
    return C

def run_original_emeans(X: torch.Tensor, k: int, epochs: float = 3.0, init_C: torch.Tensor = None) -> torch.Tensor:
    """Paper-faithful implementation of Bessa et al. (arXiv:2611.09501).
    Samples donors WITHOUT replacement (Appendix H.1) and vectorizes splitting."""
    n, d = X.shape
    B, a0 = get_emeans_hyperparameters(n, k, epochs)
    n_batches = max(1, int((n * epochs) // B))

    C = init_C.clone() if init_C is not None else X[torch.randperm(n, device=device)[:k]].clone()
    
    N_stat = torch.full((k, 1), 1e-10, device=device, dtype=torch.float32)
    S = C.to(torch.float32) * 1e-10

    fac = torch.ones(d, device=device, dtype=torch.float32)
    fac[::2] += 1.0 / 1024.0
    fac[1::2] -= 1.0 / 1024.0

    for step in range(n_batches):
        progress = float(step) / float(n_batches)
        alpha = a0 * (0.1 + (1.0 - 0.1) * 0.5 * (1.0 + math.cos(math.pi * progress)))

        batch_idx = torch.randint(0, n, (B,), device=device)
        B_X = X[batch_idx]

        C, S, N_stat, assign, scores = _emeans_step_cublas(B_X, C, S, N_stat, alpha)

        # Dead-cluster revival (Section 3.2 & Appendix H.1): without replacement
        if (step + 1) % 5 == 0:
            mean_n = N_stat.mean().item()
            dead_mask = (N_stat.squeeze(1) < 0.1 * mean_n)
            dead_indices = torch.where(dead_mask)[0]

            if len(dead_indices) > 0:
                active_weights = torch.clamp(N_stat.squeeze(1), min=0.0).clone()
                active_weights[dead_mask] = 0.0
                num_active = int((active_weights > 0).sum().item())

                if num_active > 0:
                    num_revive = min(len(dead_indices), num_active)
                    donors = torch.multinomial(active_weights, num_revive, replacement=False)
                    target_dead = dead_indices[:num_revive]

                    half_n = 0.5 * N_stat[donors]
                    c_donor = C[donors].to(torch.float32)
                    c1 = c_donor * fac
                    c2 = c_donor * (2.0 - fac)

                    N_stat[donors] = half_n
                    N_stat[target_dead] = half_n
                    S[donors] = c1 * half_n
                    S[target_dead] = c2 * half_n
                    C[donors] = c1.to(X.dtype)
                    C[target_dead] = c2.to(X.dtype)

    C = _lloyd_step_cublas(X, C)
    return C

def run_hybrid_v3(X: torch.Tensor, k: int) -> torch.Tensor:
    seed_C = run_divisive_tree(X, k=k, split_rule="dense_island")
    return run_modified_emeans(X, k=k, epochs=1.0, init_C=seed_C)

# 14-15. Engine v5 & HYBRID v5 (Residual Power Seeding)
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
    return run_modified_emeans(X, k=k, epochs=1.0, init_C=seed_C)

# =====================================================================
# Experiment Branches (v6, v7, v8-Spherical) — Non-invasive Add-ons
# =====================================================================

def _masked_power_iteration_syncfree(
    X_sub: torch.Tensor,
    mask_f: torch.Tensor,
    mask_sum: torch.Tensor,
    iters: int = 2,
) -> torch.Tensor:
    d = X_sub.shape[1]
    denom = torch.clamp(mask_sum, min=1.0)
    mu = torch.mv(X_sub.T, mask_f) / denom
    v = torch.randn(d, device=X_sub.device, dtype=X_sub.dtype)
    v = v / (torch.linalg.norm(v) + 1e-9)
    for _ in range(iters):
        z = (torch.mv(X_sub, v) - torch.dot(mu, v)) * mask_f
        v_next = torch.mv(X_sub.T, z) - z.sum() * mu
        v = v_next / (torch.linalg.norm(v_next) + 1e-9)
    return v


def _split_dense_island_syncfree(
    proj: torch.Tensor,
    mask_bool: torch.Tensor,
    bin_idxs: torch.Tensor,
    tri: torch.Tensor,
    num_bins: int = 64,
) -> torch.Tensor:
    large = 1e30
    min_v = torch.where(mask_bool, proj, large).min()
    max_v = torch.where(mask_bool, proj, -large).max()
    safe_span = torch.clamp(max_v - min_v, min=1e-6)

    norm_proj = ((proj - min_v) / safe_span * (num_bins - 1)).clamp(0, num_bins - 1).long()
    counts = torch.zeros(num_bins, device=proj.device, dtype=torch.float32)
    counts.scatter_add_(0, norm_proj, mask_bool.float())

    is_dense = counts > (0.35 * counts.sum() / float(num_bins))
    first = torch.where(is_dense, bin_idxs, num_bins).min()
    last = torch.where(is_dense, bin_idxs, -1).max()

    valid_mask = (bin_idxs >= first) & (bin_idxs <= last)
    empty_between = (~is_dense) & valid_mask

    # Longest contiguous empty gap via 64x64 prefix-sum table (zero syncs)
    not_empty_cs = torch.cumsum((~empty_between).long(), dim=0)
    cs_pad = torch.nn.functional.pad(not_empty_cs, (1, 0), value=0)
    all_empty = ((cs_pad[1:].unsqueeze(0) - cs_pad[:-1].unsqueeze(1)) == 0) & tri

    lengths = (bin_idxs.unsqueeze(0) - bin_idxs.unsqueeze(1) + 1) * all_empty.long()
    flat_idx = torch.argmax(lengths.view(-1))
    best_s = flat_idx // num_bins
    best_e = flat_idx % num_bins
    max_len = lengths.view(-1)[flat_idx]

    best_center = 0.5 * (best_s + best_e).float()
    gap_thresh = min_v + best_center * (safe_span / float(num_bins - 1))
    mid_thresh = min_v + 0.5 * safe_span
    return torch.where(max_len > 0, gap_thresh, mid_thresh).to(proj.dtype)


# --- Branch v6: Sync-Free SSE + Dense-Island Seeding ---
def run_engine_v6_sse(X: torch.Tensor, k: int, robust_cap_q: float = None) -> torch.Tensor:
    n, d = X.shape
    sub_n = min(16384, n)
    sub_idx = torch.randint(0, n, (sub_n,), device=X.device)
    X_sub = X[sub_idx]
    x_sub_sq = (X_sub * X_sub).sum(dim=1)

    C_buf = torch.empty((k, d), device=X.device, dtype=X.dtype)
    C_buf[0] = X_sub.mean(dim=0)
    cluster_sse = torch.zeros(k, device=X.device, dtype=torch.float32)

    num_bins = 64
    bin_idxs = torch.arange(num_bins, device=X.device)
    tri = torch.triu(torch.ones(num_bins, num_bins, dtype=torch.bool, device=X.device))

    for m in range(1, k):
        C_curr = C_buf[:m]
        c_half = 0.5 * (C_curr * C_curr).sum(dim=1)
        scores = torch.addmm(-c_half, X_sub, C_curr.T)
        max_score, assign = torch.max(scores, dim=1)

        pt_sse = torch.clamp(x_sub_sq - 2.0 * max_score, min=0.0).float()
        if robust_cap_q is not None:
            pt_sse = torch.clamp(pt_sse, max=torch.quantile(pt_sse, robust_cap_q))

        cluster_sse.zero_()
        cluster_sse[:m].scatter_add_(0, assign, pt_sse)
        target_cluster = torch.argmax(cluster_sse[:m])

        mask_bool = (assign == target_cluster)
        mask_cnt = mask_bool.sum()
        mask_f = mask_bool.to(X_sub.dtype)
        mask_sum = mask_f.sum()

        v = _masked_power_iteration_syncfree(X_sub, mask_f, mask_sum, iters=2)
        proj = torch.mv(X_sub, v)
        thresh = _split_dense_island_syncfree(proj, mask_bool, bin_idxs, tri, num_bins=num_bins)

        l_mask = (proj < thresh) & mask_bool
        l_cnt = l_mask.sum()
        r_cnt = mask_cnt - l_cnt
        mean_proj = (proj * mask_f).sum() / torch.clamp(mask_sum, min=1.0)
        thresh = torch.where((l_cnt < 1) | (r_cnt < 1), mean_proj, thresh)

        l_mask = (proj < thresh) & mask_bool
        r_mask = mask_bool & (~l_mask)
        l_f = l_mask.to(X_sub.dtype)
        r_f = r_mask.to(X_sub.dtype)
        l_sum = l_f.sum()
        r_sum = r_f.sum()

        c_left = torch.mv(X_sub.T, l_f) / torch.clamp(l_sum, min=1.0)
        c_right = torch.mv(X_sub.T, r_f) / torch.clamp(r_sum, min=1.0)

        valid_split = (l_sum >= 1.0) & (r_sum >= 1.0)
        C_buf[target_cluster] = torch.where(valid_split, c_left, C_buf[target_cluster])
        C_buf[m] = torch.where(valid_split, c_right, X_sub[m % sub_n])

    return C_buf


def run_hybrid_v6(X: torch.Tensor, k: int, epochs: float = 1.0, robust_cap_q: float = None) -> torch.Tensor:
    seed_C = run_engine_v6_sse(X, k=k, robust_cap_q=robust_cap_q)
    return run_modified_emeans(X, k=k, epochs=epochs, init_C=seed_C)


# --- Branch v7: Batched log2(k)-Round PDDP Seeding ---
def run_engine_v7_batched_pddp(X: torch.Tensor, k: int, power_iters: int = 2) -> torch.Tensor:
    n, d = X.shape
    sub_n = min(16384, n)
    X_sub = X[torch.randint(0, n, (sub_n,), device=X.device)]
    x_sub_sq = (X_sub * X_sub).sum(dim=1)

    C_buf = torch.empty((k, d), device=X.device, dtype=X.dtype)
    C_buf[0] = X_sub.mean(dim=0)
    m = 1

    while m < k:
        b = min(m, k - m)
        C_curr = C_buf[:m]
        c_half = 0.5 * (C_curr * C_curr).sum(dim=1)
        scores = torch.addmm(-c_half, X_sub, C_curr.T)
        max_score, assign = torch.max(scores, dim=1)

        if b < m:
            pt_sse = torch.clamp(x_sub_sq - 2.0 * max_score, min=0.0).float()
            cluster_sse = torch.zeros(m, device=X.device, dtype=torch.float32)
            cluster_sse.scatter_add_(0, assign, pt_sse)
            _, split_ids = torch.topk(cluster_sse, b, largest=True)
        else:
            split_ids = torch.arange(m, device=X.device)

        O = torch.zeros(sub_n, m, device=X.device, dtype=X.dtype)
        O.scatter_(1, assign.unsqueeze(1), 1.0)
        counts = O.sum(dim=0).clamp(min=1.0).unsqueeze(1)
        M = (O.T @ X_sub) / counts

        V = torch.randn(m, d, device=X.device, dtype=X.dtype)
        V = V / (V.norm(dim=1, keepdim=True) + 1e-9)

        for _ in range(power_iters):
            dot_x_v = (X_sub @ V.T).gather(1, assign.unsqueeze(1)).squeeze(1)
            dot_mu_v = (M * V).sum(dim=1)
            z = dot_x_v - dot_mu_v[assign]
            Z_mat = O * z.unsqueeze(1)
            V_next = Z_mat.T @ X_sub - (Z_mat.sum(dim=0).unsqueeze(1)) * M
            V = V_next / (V_next.norm(dim=1, keepdim=True) + 1e-9)

        proj = (X_sub @ V.T).gather(1, assign.unsqueeze(1)).squeeze(1)
        thresh_per_cluster = (M * V).sum(dim=1)
        is_left = proj < thresh_per_cluster[assign]

        O_left = O * is_left.to(X.dtype).unsqueeze(1)
        O_right = O - O_left
        cnt_left = O_left.sum(dim=0)
        cnt_right = O_right.sum(dim=0)

        C_left = (O_left.T @ X_sub) / cnt_left.clamp(min=1.0).unsqueeze(1)
        C_right = (O_right.T @ X_sub) / cnt_right.clamp(min=1.0).unsqueeze(1)

        valid = (cnt_left >= 1.0) & (cnt_right >= 1.0)
        C_left_safe = torch.where(valid.unsqueeze(1), C_left, C_curr)
        fallback_pts = X_sub[torch.arange(m, device=X.device) % sub_n]
        C_right_safe = torch.where(valid.unsqueeze(1), C_right, fallback_pts)

        C_buf[split_ids] = C_left_safe[split_ids]
        C_buf[m : m + b] = C_right_safe[split_ids]
        m += b

    return C_buf


def run_hybrid_v7(X: torch.Tensor, k: int, epochs: float = 1.0) -> torch.Tensor:
    seed_C = run_engine_v7_batched_pddp(X, k=k)
    return run_modified_emeans(X, k=k, epochs=epochs, init_C=seed_C)


# --- Branch v8: Spherical (Cosine) Gradient-Direction Clustering ---
def run_spherical_v8_grad(G: torch.Tensor, k: int, polish_iters: int = 3) -> torch.Tensor:
    """Clusters gradients by cosine direction (immune to outlier gradient norms).
    Returns unit-normalized centroids."""
    G_unit = torch.nn.functional.normalize(G, p=2, dim=1)
    C = run_engine_v7_batched_pddp(G_unit, k=k)
    C = torch.nn.functional.normalize(C, p=2, dim=1)

    n = G_unit.shape[0]
    chunk_size = 65536
    for _ in range(polish_iters):
        new_C = torch.zeros_like(C)
        counts = torch.zeros(k, 1, device=G.device, dtype=G.dtype)
        for i in range(0, n, chunk_size):
            chunk = G_unit[i : i + chunk_size]
            assign = torch.argmax(chunk @ C.T, dim=1)
            O = torch.zeros(chunk.shape[0], k, device=G.device, dtype=G.dtype)
            O.scatter_(1, assign.unsqueeze(1), 1.0)
            new_C += O.T @ chunk
            counts += O.sum(dim=0, keepdim=True).T
        valid = (counts.squeeze(1) > 0).unsqueeze(1)
        C = torch.where(valid, torch.nn.functional.normalize(new_C, p=2, dim=1), C)
    return C
# =====================================================================
# Experiment Branch v9: Batched Priority-SSE + Batched Dense-Island
# =====================================================================

def run_engine_v9_priority_island(
    X: torch.Tensor,
    k: int,
    power_iters: int = 2,
    robust_cap_q: float = 0.95,
) -> torch.Tensor:
    n, d = X.shape
    sub_n = min(16384, n)
    X_sub = X[torch.randint(0, n, (sub_n,), device=X.device)]
    x_sub_sq = (X_sub * X_sub).sum(dim=1)

    C_buf = torch.empty((k, d), device=X.device, dtype=X.dtype)
    C_buf[0] = X_sub.mean(dim=0)

    num_bins = 64
    bin_idxs = torch.arange(num_bins, device=X.device)
    tri = torch.triu(torch.ones(num_bins, num_bins, dtype=torch.bool, device=X.device))

    m = 1
    while m < k:
        # Split top ~67% highest-SSE clusters each round (8 rounds for k=64)
        b = min(m, max(1, (2 * m + 2) // 3), k - m)

        C_curr = C_buf[:m]
        c_half = 0.5 * (C_curr * C_curr).sum(dim=1)
        scores = torch.addmm(-c_half, X_sub, C_curr.T)
        max_score, assign = torch.max(scores, dim=1)

        pt_sse = torch.clamp(x_sub_sq - 2.0 * max_score, min=0.0).float()
        if robust_cap_q is not None:
            pt_sse = torch.clamp(pt_sse, max=torch.quantile(pt_sse, robust_cap_q))

        cluster_sse = torch.zeros(m, device=X.device, dtype=torch.float32)
        cluster_sse.scatter_add_(0, assign, pt_sse)
        _, split_ids = torch.topk(cluster_sse, b, largest=True)

        # Map points into 0..b-1 for selected clusters, b for unselected
        cluster_to_sel = torch.full((m,), b, device=X.device, dtype=torch.long)
        cluster_to_sel[split_ids] = torch.arange(b, device=X.device)
        sel_pos = cluster_to_sel[assign]
        safe_pos = sel_pos.clamp(max=b - 1)
        in_sel = (sel_pos < b).to(X.dtype)

        O_full = torch.zeros(sub_n, b + 1, device=X.device, dtype=X.dtype)
        O_full.scatter_(1, sel_pos.unsqueeze(1), 1.0)
        O = O_full[:, :b]  # (sub_n, b)

        counts = O.sum(dim=0).clamp(min=1.0).unsqueeze(1)
        M = (O.T @ X_sub) / counts  # (b, d)

        V = torch.randn(b, d, device=X.device, dtype=X.dtype)
        V = V / (V.norm(dim=1, keepdim=True) + 1e-9)

        for _ in range(power_iters):
            dot_x_v = (X_sub @ V.T).gather(1, safe_pos.unsqueeze(1)).squeeze(1) * in_sel
            dot_mu_v = (M * V).sum(dim=1)
            z = (dot_x_v - dot_mu_v[safe_pos]) * in_sel
            Z_mat = O * z.unsqueeze(1)
            V_next = Z_mat.T @ X_sub - (Z_mat.sum(dim=0).unsqueeze(1)) * M
            V = V_next / (V_next.norm(dim=1, keepdim=True) + 1e-9)

        proj = (X_sub @ V.T).gather(1, safe_pos.unsqueeze(1)).squeeze(1)

        # Batched 64-bin dense-island gap finder across all b selected clusters
        O_bool = O > 0.5
        min_v = torch.where(O_bool, proj.unsqueeze(1), 1e30).min(dim=0).values
        max_v = torch.where(O_bool, proj.unsqueeze(1), -1e30).max(dim=0).values
        safe_span = torch.clamp(max_v - min_v, min=1e-6)

        bin_idx = ((proj - min_v[safe_pos]) / safe_span[safe_pos] * (num_bins - 1)).clamp(0, num_bins - 1).long()
        flat_bin = sel_pos * num_bins + bin_idx
        hist = torch.zeros((b + 1) * num_bins, device=X.device, dtype=torch.float32)
        hist.scatter_add_(0, flat_bin, in_sel.float())
        counts_hist = hist[: b * num_bins].view(b, num_bins)

        is_dense = counts_hist > (0.35 * counts_hist.sum(dim=1, keepdim=True) / float(num_bins))
        first = torch.where(is_dense, bin_idxs.unsqueeze(0), num_bins).min(dim=1, keepdim=True).values
        last = torch.where(is_dense, bin_idxs.unsqueeze(0), -1).max(dim=1, keepdim=True).values

        valid_mask = (bin_idxs.unsqueeze(0) >= first) & (bin_idxs.unsqueeze(0) <= last)
        empty_between = (~is_dense) & valid_mask

        not_empty_cs = torch.cumsum((~empty_between).long(), dim=1)
        cs_pad = torch.nn.functional.pad(not_empty_cs, (1, 0), value=0)
        all_empty = ((cs_pad[:, 1:].unsqueeze(1) - cs_pad[:, :-1].unsqueeze(2)) == 0) & tri.unsqueeze(0)

        lengths = (bin_idxs.view(1, 1, num_bins) - bin_idxs.view(1, num_bins, 1) + 1) * all_empty.long()
        max_len, flat_idx = torch.max(lengths.view(b, -1), dim=1)
        best_s = flat_idx // num_bins
        best_e = flat_idx % num_bins

        best_center = 0.5 * (best_s + best_e).float()
        gap_thresh = min_v + best_center * (safe_span / float(num_bins - 1))
        mid_thresh = min_v + 0.5 * safe_span
        thresh = torch.where(max_len > 0, gap_thresh, mid_thresh).to(X.dtype)

        dot_mu_v = (M * V).sum(dim=1)
        is_left = (proj < thresh[safe_pos]) & (sel_pos < b)
        O_left = O * is_left.to(X.dtype).unsqueeze(1)
        O_right = O - O_left
        cnt_left = O_left.sum(dim=0)
        cnt_right = O_right.sum(dim=0)

        # Branchless fallback to cluster mean if gap-split left either side empty
        bad = (cnt_left < 1.0) | (cnt_right < 1.0)
        thresh_safe = torch.where(bad, dot_mu_v, thresh)
        is_left = (proj < thresh_safe[safe_pos]) & (sel_pos < b)
        O_left = O * is_left.to(X.dtype).unsqueeze(1)
        O_right = O - O_left
        cnt_left = O_left.sum(dim=0)
        cnt_right = O_right.sum(dim=0)

        C_left = (O_left.T @ X_sub) / cnt_left.clamp(min=1.0).unsqueeze(1)
        C_right = (O_right.T @ X_sub) / cnt_right.clamp(min=1.0).unsqueeze(1)

        valid = (cnt_left >= 1.0) & (cnt_right >= 1.0)
        C_left_safe = torch.where(valid.unsqueeze(1), C_left, C_curr[split_ids])
        fallback_pts = X_sub[torch.arange(b, device=X.device) % sub_n]
        C_right_safe = torch.where(valid.unsqueeze(1), C_right, fallback_pts)

        C_buf[split_ids] = C_left_safe
        C_buf[m : m + b] = C_right_safe
        m += b

    return C_buf


def _warm_emeans_refine(
    X: torch.Tensor,
    init_C: torch.Tensor,
    epochs: float = 0.5,
    a0: float = 0.35,
) -> torch.Tensor:
    """Refines seeded centroids without wiping them out on batch 0."""
    n, d = X.shape
    k = init_C.shape[0]
    B, _ = get_emeans_hyperparameters(n, k, epochs)
    n_batches = max(1, int((n * epochs) // B))

    C = init_C.clone()
    prior_n = max(1.0, 0.25 * (float(B) / float(k)))
    N_stat = torch.full((k, 1), prior_n, device=X.device, dtype=torch.float32)
    S = C.to(torch.float32) * prior_n

    for step in range(n_batches):
        progress = float(step) / float(n_batches)
        alpha = a0 * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))
        batch_idx = torch.randint(0, n, (B,), device=X.device)
        C, S, N_stat, _, _ = _emeans_step_cublas(X[batch_idx], C, S, N_stat, alpha)

    return _lloyd_step_cublas(X, C)


def run_hybrid_v9_warm(X: torch.Tensor, k: int, epochs: float = 0.5) -> torch.Tensor:
    seed_C = run_engine_v9_priority_island(X, k=k, robust_cap_q=0.95)
    return _warm_emeans_refine(X, seed_C, epochs=epochs, a0=0.35)


def run_hybrid_v9_2lloyd(X: torch.Tensor, k: int) -> torch.Tensor:
    C = run_engine_v9_priority_island(X, k=k, robust_cap_q=0.95)
    C = _lloyd_step_cublas(X, C)
    return _lloyd_step_cublas(X, C)

def run_engine_v9_ablation(X: torch.Tensor, k: int, use_island: bool = False, verbose: bool = False) -> torch.Tensor:
    n, d = X.shape
    sub_n = min(16384, n)
    X_sub = X[torch.randint(0, n, (sub_n,), device=X.device)]
    x_sub_sq = (X_sub * X_sub).sum(dim=1)

    C_buf = torch.empty((k, d), device=X.device, dtype=X.dtype)
    C_buf[0] = X_sub.mean(dim=0)

    num_bins = 64
    bin_idxs = torch.arange(num_bins, device=X.device)
    tri = torch.triu(torch.ones(num_bins, num_bins, dtype=torch.bool, device=X.device))

    m = 1
    while m < k:
        b = min(m, max(1, (2 * m + 2) // 3), k - m)
        C_curr = C_buf[:m]
        c_half = 0.5 * (C_curr * C_curr).sum(dim=1)
        scores = torch.addmm(-c_half, X_sub, C_curr.T)
        max_score, assign = torch.max(scores, dim=1)

        pt_sse = torch.clamp(x_sub_sq - 2.0 * max_score, min=0.0).float()
        pt_sse = torch.clamp(pt_sse, max=torch.quantile(pt_sse, 0.95))

        cluster_sse = torch.zeros(m, device=X.device, dtype=torch.float32)
        cluster_sse.scatter_add_(0, assign, pt_sse)
        _, split_ids = torch.topk(cluster_sse, b, largest=True)

        cluster_to_sel = torch.full((m,), b, device=X.device, dtype=torch.long)
        cluster_to_sel[split_ids] = torch.arange(b, device=X.device)
        sel_pos = cluster_to_sel[assign]
        safe_pos = sel_pos.clamp(max=b - 1)
        in_sel = (sel_pos < b).to(X.dtype)

        O_full = torch.zeros(sub_n, b + 1, device=X.device, dtype=X.dtype)
        O_full.scatter_(1, sel_pos.unsqueeze(1), 1.0)
        O = O_full[:, :b]

        counts = O.sum(dim=0).clamp(min=1.0).unsqueeze(1)
        M = (O.T @ X_sub) / counts

        V = torch.randn(b, d, device=X.device, dtype=X.dtype)
        V = V / (V.norm(dim=1, keepdim=True) + 1e-9)

        for _ in range(2):
            dot_x_v = (X_sub @ V.T).gather(1, safe_pos.unsqueeze(1)).squeeze(1) * in_sel
            dot_mu_v = (M * V).sum(dim=1)
            z = (dot_x_v - dot_mu_v[safe_pos]) * in_sel
            Z_mat = O * z.unsqueeze(1)
            V_next = Z_mat.T @ X_sub - (Z_mat.sum(dim=0).unsqueeze(1)) * M
            V = V_next / (V_next.norm(dim=1, keepdim=True) + 1e-9)

        proj = (X_sub @ V.T).gather(1, safe_pos.unsqueeze(1)).squeeze(1)
        dot_mu_v = (M * V).sum(dim=1)

        if use_island:
            O_bool = O > 0.5
            min_v = torch.where(O_bool, proj.unsqueeze(1), 1e30).min(dim=0).values
            max_v = torch.where(O_bool, proj.unsqueeze(1), -1e30).max(dim=0).values
            safe_span = torch.clamp(max_v - min_v, min=1e-6)
            bin_idx = ((proj - min_v[safe_pos]) / safe_span[safe_pos] * (num_bins - 1)).clamp(0, num_bins - 1).long()
            flat_bin = sel_pos * num_bins + bin_idx
            hist = torch.zeros((b + 1) * num_bins, device=X.device, dtype=torch.float32)
            hist.scatter_add_(0, flat_bin, in_sel.float())
            counts_hist = hist[: b * num_bins].view(b, num_bins)
            is_dense = counts_hist > (0.35 * counts_hist.sum(dim=1, keepdim=True) / float(num_bins))
            first = torch.where(is_dense, bin_idxs.unsqueeze(0), num_bins).min(dim=1, keepdim=True).values
            last = torch.where(is_dense, bin_idxs.unsqueeze(0), -1).max(dim=1, keepdim=True).values
            valid_mask = (bin_idxs.unsqueeze(0) >= first) & (bin_idxs.unsqueeze(0) <= last)
            empty_between = (~is_dense) & valid_mask
            not_empty_cs = torch.cumsum((~empty_between).long(), dim=1)
            cs_pad = torch.nn.functional.pad(not_empty_cs, (1, 0), value=0)
            all_empty = ((cs_pad[:, 1:].unsqueeze(1) - cs_pad[:, :-1].unsqueeze(2)) == 0) & tri.unsqueeze(0)
            lengths = (bin_idxs.view(1, 1, num_bins) - bin_idxs.view(1, num_bins, 1) + 1) * all_empty.long()
            max_len, flat_idx = torch.max(lengths.view(b, -1), dim=1)
            best_s, best_e = flat_idx // num_bins, flat_idx % num_bins
            gap_thresh = min_v + 0.5 * (best_s + best_e).float() * (safe_span / float(num_bins - 1))
            mid_thresh = min_v + 0.5 * safe_span
            thresh = torch.where(max_len > 0, gap_thresh, mid_thresh).to(X.dtype)
        else:
            thresh = dot_mu_v

        is_left = (proj < thresh[safe_pos]) & (sel_pos < b)
        O_left = O * is_left.to(X.dtype).unsqueeze(1)
        O_right = O - O_left
        cnt_left, cnt_right = O_left.sum(dim=0), O_right.sum(dim=0)

        bad = (cnt_left < 1.0) | (cnt_right < 1.0)
        thresh_safe = torch.where(bad, dot_mu_v, thresh)
        is_left = (proj < thresh_safe[safe_pos]) & (sel_pos < b)
        O_left = O * is_left.to(X.dtype).unsqueeze(1)
        O_right = O - O_left
        cnt_left, cnt_right = O_left.sum(dim=0), O_right.sum(dim=0)

        if verbose:
            print(f"m={m:2d} b={b:2d} | min_cnt={(cnt_left.min().item(), cnt_right.min().item())} | bad={bad.sum().item()}")

        C_left = (O_left.T @ X_sub) / cnt_left.clamp(min=1.0).unsqueeze(1)
        C_right = (O_right.T @ X_sub) / cnt_right.clamp(min=1.0).unsqueeze(1)
        valid = (cnt_left >= 1.0) & (cnt_right >= 1.0)
        C_buf[split_ids] = torch.where(valid.unsqueeze(1), C_left, C_curr[split_ids])
        C_buf[m : m + b] = torch.where(valid.unsqueeze(1), C_right, X_sub[torch.arange(b, device=X.device) % sub_n])
        m += b

    return C_buf


def run_engine_v10_fast(
    X: torch.Tensor,
    k: int,
    use_guarded_island: bool = False,
    robust_cap_q: float = 0.95,
) -> torch.Tensor:
    n, d = X.shape
    sub_n = min(16384, n)
    X_sub = X[torch.randint(0, n, (sub_n,), device=X.device)]
    x_sub_sq = (X_sub * X_sub).sum(dim=1)

    C_buf = torch.empty((k, d), device=X.device, dtype=X.dtype)
    C_buf[0] = X_sub.mean(dim=0)

    if use_guarded_island:
        num_bins = 64
        bin_idxs = torch.arange(num_bins, device=X.device)
        tri = torch.triu(torch.ones(num_bins, num_bins, dtype=torch.bool, device=X.device))

    m = 1
    while m < k:
        b = min(m, max(1, (2 * m + 2) // 3), k - m)
        C_curr = C_buf[:m]
        c_half = 0.5 * (C_curr * C_curr).sum(dim=1)
        scores = torch.addmm(-c_half, X_sub, C_curr.T)
        max_score, assign = torch.max(scores, dim=1)

        pt_sse = torch.clamp(x_sub_sq - 2.0 * max_score, min=0.0).float()
        if robust_cap_q is not None:
            pt_sse = torch.clamp(pt_sse, max=torch.quantile(pt_sse, robust_cap_q))

        cluster_sse = torch.zeros(m, device=X.device, dtype=torch.float32)
        cluster_sse.scatter_add_(0, assign, pt_sse)
        _, split_ids = torch.topk(cluster_sse, b, largest=True)

        cluster_to_sel = torch.full((m,), b, device=X.device, dtype=torch.long)
        cluster_to_sel[split_ids] = torch.arange(b, device=X.device)
        sel_pos = cluster_to_sel[assign]
        safe_pos = sel_pos.clamp(max=b - 1)
        in_sel = (sel_pos < b).to(X.dtype)

        O_full = torch.zeros(sub_n, b + 1, device=X.device, dtype=X.dtype)
        O_full.scatter_(1, sel_pos.unsqueeze(1), 1.0)
        O = O_full[:, :b]

        total_cnt = O.sum(dim=0)
        M = (O.T @ X_sub) / total_cnt.clamp(min=1.0).unsqueeze(1)

        V = torch.randn(b, d, device=X.device, dtype=X.dtype)
        V = V / (V.norm(dim=1, keepdim=True) + 1e-9)

        for _ in range(2):
            dot_x_v = (X_sub @ V.T).gather(1, safe_pos.unsqueeze(1)).squeeze(1) * in_sel
            dot_mu_v = (M * V).sum(dim=1)
            z = (dot_x_v - dot_mu_v[safe_pos]) * in_sel
            Z_mat = O * z.unsqueeze(1)
            V_next = Z_mat.T @ X_sub - (Z_mat.sum(dim=0).unsqueeze(1)) * M
            V = V_next / (V_next.norm(dim=1, keepdim=True) + 1e-9)

        proj = (X_sub @ V.T).gather(1, safe_pos.unsqueeze(1)).squeeze(1)
        dot_mu_v = (M * V).sum(dim=1)

        if use_guarded_island:
            O_bool = O > 0.5
            min_v = torch.where(O_bool, proj.unsqueeze(1), 1e30).min(dim=0).values
            max_v = torch.where(O_bool, proj.unsqueeze(1), -1e30).max(dim=0).values
            safe_span = torch.clamp(max_v - min_v, min=1e-6)

            bin_idx = ((proj - min_v[safe_pos]) / safe_span[safe_pos] * (num_bins - 1)).clamp(0, num_bins - 1).long()
            flat_bin = sel_pos * num_bins + bin_idx
            hist = torch.zeros((b + 1) * num_bins, device=X.device, dtype=torch.float32)
            hist.scatter_add_(0, flat_bin, in_sel.float())
            counts_hist = hist[: b * num_bins].view(b, num_bins)

            is_dense = counts_hist > (0.35 * counts_hist.sum(dim=1, keepdim=True) / float(num_bins))
            first = torch.where(is_dense, bin_idxs.unsqueeze(0), num_bins).min(dim=1, keepdim=True).values
            last = torch.where(is_dense, bin_idxs.unsqueeze(0), -1).max(dim=1, keepdim=True).values
            valid_mask = (bin_idxs.unsqueeze(0) >= first) & (bin_idxs.unsqueeze(0) <= last)
            empty_between = (~is_dense) & valid_mask

            not_empty_cs = torch.cumsum((~empty_between).long(), dim=1)
            cs_pad = torch.nn.functional.pad(not_empty_cs, (1, 0), value=0)
            all_empty = ((cs_pad[:, 1:].unsqueeze(1) - cs_pad[:, :-1].unsqueeze(2)) == 0) & tri.unsqueeze(0)
            lengths = (bin_idxs.view(1, 1, num_bins) - bin_idxs.view(1, num_bins, 1) + 1) * all_empty.long()
            max_len, flat_idx = torch.max(lengths.view(b, -1), dim=1)
            best_s, best_e = flat_idx // num_bins, flat_idx % num_bins

            gap_thresh = min_v + 0.5 * (best_s + best_e).float() * (safe_span / float(num_bins - 1))
            mid_thresh = min_v + 0.5 * safe_span
            thresh = torch.where(max_len > 0, gap_thresh, mid_thresh).to(X.dtype)

            is_left = (proj < thresh[safe_pos]) & (sel_pos < b)
            O_left = O * is_left.to(X.dtype).unsqueeze(1)
            cnt_left = O_left.sum(dim=0)
            cnt_right = total_cnt - cnt_left

            # 10% balance guard: reject edge-peeling cuts and fall back to mean-split
            min_side = 0.10 * total_cnt
            imbalanced = (cnt_left < min_side) | (cnt_right < min_side)
            thresh = torch.where(imbalanced, dot_mu_v, thresh)
        else:
            thresh = dot_mu_v

        is_left = (proj < thresh[safe_pos]) & (sel_pos < b)
        O_left = O * is_left.to(X.dtype).unsqueeze(1)
        O_right = O - O_left
        cnt_left, cnt_right = O_left.sum(dim=0), O_right.sum(dim=0)

        C_left = (O_left.T @ X_sub) / cnt_left.clamp(min=1.0).unsqueeze(1)
        C_right = (O_right.T @ X_sub) / cnt_right.clamp(min=1.0).unsqueeze(1)
        valid = (cnt_left >= 1.0) & (cnt_right >= 1.0)
        C_buf[split_ids] = torch.where(valid.unsqueeze(1), C_left, C_curr[split_ids])
        C_buf[m : m + b] = torch.where(valid.unsqueeze(1), C_right, X_sub[torch.arange(b, device=X.device) % sub_n])
        m += b

    return C_buf


def run_hybrid_v10_1lloyd(X: torch.Tensor, k: int, use_guarded_island: bool = False) -> torch.Tensor:
    C = run_engine_v10_fast(X, k=k, use_guarded_island=use_guarded_island)
    return _lloyd_step_cublas(X, C)


def run_hybrid_v10_warm(X: torch.Tensor, k: int, epochs: float = 0.5, use_guarded_island: bool = False) -> torch.Tensor:
    seed_C = run_engine_v10_fast(X, k=k, use_guarded_island=use_guarded_island)
    return _warm_emeans_refine(X, seed_C, epochs=epochs, a0=0.35)


def _warm_emeans_refine_revive(
    X: torch.Tensor,
    init_C: torch.Tensor,
    epochs: float = 0.5,
    a0: float = 0.35,
    revive_thresh: float = 0.15,
    revive_cap_q: float = 0.96,
) -> torch.Tensor:
    n, d = X.shape
    k = init_C.shape[0]
    B, _ = get_emeans_hyperparameters(n, k, epochs)
    n_batches = max(1, int((n * epochs) // B))

    C = init_C.clone()
    prior_n = max(1.0, 0.25 * (float(B) / float(k)))
    N_stat = torch.full((k, 1), prior_n, device=X.device, dtype=torch.float32)
    S = C.to(torch.float32) * prior_n

    for step in range(n_batches):
        progress = float(step) / float(n_batches)
        alpha = a0 * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))
        batch_idx = torch.randint(0, n, (B,), device=X.device)
        B_X = X[batch_idx]

        C, S, N_stat, _, scores = _emeans_step_cublas(B_X, C, S, N_stat, alpha)

        if (step + 1) % 5 == 0 and step < int(0.75 * n_batches):
            mean_n = N_stat.mean().item()
            dead_indices = torch.where(N_stat.squeeze(1) < revive_thresh * mean_n)[0]
            if len(dead_indices) > 0:
                max_score, _ = torch.max(scores, dim=1)
                bx_sq = (B_X * B_X).sum(dim=1)
                residual_sq = torch.clamp(bx_sq - 2.0 * max_score, min=0.0)
                cap_val = torch.quantile(residual_sq, revive_cap_q)
                weights = torch.clamp(residual_sq, max=cap_val)
                if weights.sum() > 0:
                    num_to_sample = min(len(dead_indices), int((weights > 0).sum().item()))
                    chosen_idx = torch.multinomial(weights, num_to_sample, replacement=False)
                    new_pts = B_X[chosen_idx]
                    target_dead = dead_indices[:num_to_sample]
                    N_stat[target_dead] = mean_n
                    S[target_dead] = new_pts.to(torch.float32) * mean_n
                    C[target_dead] = new_pts

    return _lloyd_step_cublas(X, C)


def run_hybrid_v11(X: torch.Tensor, k: int, epochs: float = 0.5, revive_thresh: float = 0.15) -> torch.Tensor:
    seed_C = run_engine_v10_fast(X, k=k, use_guarded_island=False)
    return _warm_emeans_refine_revive(X, seed_C, epochs=epochs, a0=0.35, revive_thresh=revive_thresh, revive_cap_q=0.96)


def run_engine_v12_pddp_d2(
    X: torch.Tensor,
    k: int,
    tail_r: int = 8,
    robust_cap_q: float = 0.95,
) -> torch.Tensor:
    n, d = X.shape
    r = min(max(0, tail_r), max(0, k - 1))
    k_pddp = k - r

    sub_n = min(16384, n)
    X_sub = X[torch.randint(0, n, (sub_n,), device=X.device)]
    x_sub_sq = (X_sub * X_sub).sum(dim=1)

    C_buf = torch.empty((k, d), device=X.device, dtype=X.dtype)
    C_buf[0] = X_sub.mean(dim=0)

    # Phase 1: Fast Batched Priority-SSE PDDP for k - r clusters
    m = 1
    while m < k_pddp:
        b = min(m, max(1, (2 * m + 2) // 3), k_pddp - m)
        C_curr = C_buf[:m]
        c_half = 0.5 * (C_curr * C_curr).sum(dim=1)
        scores = torch.addmm(-c_half, X_sub, C_curr.T)
        max_score, assign = torch.max(scores, dim=1)

        pt_sse = torch.clamp(x_sub_sq - 2.0 * max_score, min=0.0).float()
        if robust_cap_q is not None:
            pt_sse = torch.clamp(pt_sse, max=torch.quantile(pt_sse, robust_cap_q))

        cluster_sse = torch.zeros(m, device=X.device, dtype=torch.float32)
        cluster_sse.scatter_add_(0, assign, pt_sse)
        _, split_ids = torch.topk(cluster_sse, b, largest=True)

        cluster_to_sel = torch.full((m,), b, device=X.device, dtype=torch.long)
        cluster_to_sel[split_ids] = torch.arange(b, device=X.device)
        sel_pos = cluster_to_sel[assign]
        safe_pos = sel_pos.clamp(max=b - 1)
        in_sel = (sel_pos < b).to(X.dtype)

        O_full = torch.zeros(sub_n, b + 1, device=X.device, dtype=X.dtype)
        O_full.scatter_(1, sel_pos.unsqueeze(1), 1.0)
        O = O_full[:, :b]

        total_cnt = O.sum(dim=0)
        M = (O.T @ X_sub) / total_cnt.clamp(min=1.0).unsqueeze(1)

        V = torch.randn(b, d, device=X.device, dtype=X.dtype)
        V = V / (V.norm(dim=1, keepdim=True) + 1e-9)

        for _ in range(2):
            dot_x_v = (X_sub @ V.T).gather(1, safe_pos.unsqueeze(1)).squeeze(1) * in_sel
            dot_mu_v = (M * V).sum(dim=1)
            z = (dot_x_v - dot_mu_v[safe_pos]) * in_sel
            Z_mat = O * z.unsqueeze(1)
            V_next = Z_mat.T @ X_sub - (Z_mat.sum(dim=0).unsqueeze(1)) * M
            V = V_next / (V_next.norm(dim=1, keepdim=True) + 1e-9)

        proj = (X_sub @ V.T).gather(1, safe_pos.unsqueeze(1)).squeeze(1)
        dot_mu_v = (M * V).sum(dim=1)

        is_left = (proj < dot_mu_v[safe_pos]) & (sel_pos < b)
        O_left = O * is_left.to(X.dtype).unsqueeze(1)
        O_right = O - O_left
        cnt_left, cnt_right = O_left.sum(dim=0), O_right.sum(dim=0)

        C_left = (O_left.T @ X_sub) / cnt_left.clamp(min=1.0).unsqueeze(1)
        C_right = (O_right.T @ X_sub) / cnt_right.clamp(min=1.0).unsqueeze(1)
        valid = (cnt_left >= 1.0) & (cnt_right >= 1.0)
        C_buf[split_ids] = torch.where(valid.unsqueeze(1), C_left, C_curr[split_ids])
        C_buf[m : m + b] = torch.where(valid.unsqueeze(1), C_right, X_sub[torch.arange(b, device=X.device) % sub_n])
        m += b

    # Phase 2: Capped-D2 Residual Tail for remaining r centroids
    if r > 0:
        C_pddp = C_buf[:k_pddp]
        c_half = 0.5 * (C_pddp * C_pddp).sum(dim=1)
        max_score = torch.max(torch.addmm(-c_half, X_sub, C_pddp.T), dim=1).values
        min_sq = torch.clamp(x_sub_sq - 2.0 * max_score, min=0.0).float()

        for j in range(k_pddp, k):
            cap_val = torch.quantile(min_sq, robust_cap_q if robust_cap_q is not None else 0.99)
            w = torch.clamp(min_sq, max=cap_val) + 1e-12
            idx = torch.multinomial(w, 1).squeeze(0)
            new_c = X_sub[idx]
            C_buf[j] = new_c
            new_sq = torch.clamp(x_sub_sq + (new_c * new_c).sum() - 2.0 * torch.mv(X_sub, new_c), min=0.0).float()
            min_sq = torch.minimum(min_sq, new_sq)

    return C_buf


def run_hybrid_v12(X: torch.Tensor, k: int, tail_frac: float = 0.125, epochs: float = 0.5) -> torch.Tensor:
    r = max(2, int(round(k * tail_frac)))
    seed_C = run_engine_v12_pddp_d2(X, k=k, tail_r=r, robust_cap_q=0.95)
    return _warm_emeans_refine(X, seed_C, epochs=epochs, a0=0.35)


def _damped_warm_emeans_refine(
    X: torch.Tensor,
    init_C: torch.Tensor,
    epochs: float = 0.5,
    a0: float = 0.35,
    damp_quantile_scale: float = 2.0,
) -> torch.Tensor:
    n, d = X.shape
    k = init_C.shape[0]
    B, _ = get_emeans_hyperparameters(n, k, epochs)
    n_batches = max(1, int((n * epochs) // B))

    C = init_C.clone()
    prior_n = max(1.0, 0.25 * (float(B) / float(k)))
    N_stat = torch.full((k, 1), prior_n, device=X.device, dtype=torch.float32)
    S = C.to(torch.float32) * prior_n

    # Estimate scale λ once from a small probe so damping adapts to task variance
    probe = X[torch.randint(0, n, (min(4096, n),), device=X.device)]
    c_half0 = 0.5 * (C * C).sum(dim=1)
    max_sc0 = torch.max(torch.addmm(-c_half0, probe, C.T), dim=1).values
    probe_sq = torch.clamp((probe * probe).sum(dim=1) - 2.0 * max_sc0, min=0.0)
    lam = torch.clamp(damp_quantile_scale * torch.median(probe_sq), min=1e-4)

    for step in range(n_batches):
        progress = float(step) / float(n_batches)
        alpha = a0 * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))
        B_X = X[torch.randint(0, n, (B,), device=X.device)]

        c_half = 0.5 * (C * C).sum(dim=1)
        scores = torch.addmm(-c_half, B_X, C.T)
        max_score, assign = torch.max(scores, dim=1)

        # Globally bounded force weight: w_i = λ / (||x - c||² + λ)
        # Ensures |w_i * (x - c)| ≤ √λ / 2 (damped_force_globally_bounded)
        dist_sq = torch.clamp((B_X * B_X).sum(dim=1) - 2.0 * max_score, min=0.0).float()
        w = (lam / (dist_sq + lam)).unsqueeze(1)

        O = torch.zeros(B_X.shape[0], k, device=X.device, dtype=torch.float32)
        O.scatter_(1, assign.unsqueeze(1), 1.0)
        O_w = O * w

        s_batch = O_w.T @ B_X.to(torch.float32)
        n_batch = O_w.sum(dim=0, keepdim=True).T

        S = (1.0 - alpha) * S + alpha * s_batch
        N_stat = (1.0 - alpha) * N_stat + alpha * n_batch
        valid = (N_stat.squeeze(1) > 1e-6)
        C = torch.where(valid.unsqueeze(1), (S / torch.clamp(N_stat, min=1e-6)).to(C.dtype), C)

    return _lloyd_step_cublas(X, C)


def run_hybrid_v13_damped(X: torch.Tensor, k: int, epochs: float = 0.5) -> torch.Tensor:
    seed_C = run_engine_v10_fast(X, k=k, use_guarded_island=False)
    return _damped_warm_emeans_refine(X, seed_C, epochs=epochs, a0=0.35)


def _multiscale_damped_emeans_refine(
    X: torch.Tensor,
    init_C: torch.Tensor,
    epochs: float = 0.5,
    num_branches: int = 4,
    damp_quantile_scale: float = 2.0,
) -> torch.Tensor:
    n, d = X.shape
    k = init_C.shape[0]
    B, _ = get_emeans_hyperparameters(n, k, epochs)
    n_batches = max(1, int((n * epochs) // B))

    C = init_C.clone()
    prior_n = max(1.0, 0.25 * (float(B) / float(k)))

    # Multi-scale decay rates mirroring LinearMetricSimplexAttention (slow memory -> fast tracker)
    rates = torch.logspace(-2.0, -0.35, steps=num_branches, device=X.device, dtype=torch.float32)  # (M,)
    N_branches = torch.full((num_branches, k, 1), prior_n, device=X.device, dtype=torch.float32)
    S_branches = C.to(torch.float32).unsqueeze(0).expand(num_branches, k, d).clone() * prior_n

    probe = X[torch.randint(0, n, (min(4096, n),), device=X.device)]
    c_half0 = 0.5 * (C * C).sum(dim=1)
    max_sc0 = torch.max(torch.addmm(-c_half0, probe, C.T), dim=1).values
    probe_sq = torch.clamp((probe * probe).sum(dim=1) - 2.0 * max_sc0, min=0.0)
    lam = torch.clamp(damp_quantile_scale * torch.median(probe_sq), min=1e-4)

    for step in range(n_batches):
        progress = float(step) / float(n_batches)
        cosine_scale = 0.2 + 0.8 * 0.5 * (1.0 + math.cos(math.pi * progress))
        B_X = X[torch.randint(0, n, (B,), device=X.device)]

        c_half = 0.5 * (C * C).sum(dim=1)
        scores = torch.addmm(-c_half, B_X, C.T)
        max_score, assign = torch.max(scores, dim=1)

        dist_sq = torch.clamp((B_X * B_X).sum(dim=1) - 2.0 * max_score, min=0.0).float()
        w = (lam / (dist_sq + lam)).unsqueeze(1)

        O = torch.zeros(B_X.shape[0], k, device=X.device, dtype=torch.float32)
        O.scatter_(1, assign.unsqueeze(1), 1.0)
        O_w = O * w

        s_batch = O_w.T @ B_X.to(torch.float32)     # (k, d)
        n_batch = O_w.sum(dim=0, keepdim=True).T     # (k, 1)

        # Update all M decay timescales simultaneously
        alphas = (rates * cosine_scale).view(num_branches, 1, 1)
        S_branches = (1.0 - alphas) * S_branches + alphas * s_batch.unsqueeze(0)
        N_branches = (1.0 - alphas) * N_branches + alphas * n_batch.unsqueeze(0)

        # Activity-gated branch simplex: busy clusters use fast branches, rare clusters lean on slow memory
        activity = n_batch / torch.clamp(n_batch.mean(), min=1e-6)  # (k, 1)
        branch_logits = (activity.squeeze(1).unsqueeze(0) - 1.0) * torch.linspace(-1.5, 1.5, num_branches, device=X.device).unsqueeze(1)
        branch_w = torch.softmax(branch_logits, dim=0).unsqueeze(-1)  # (M, k, 1)

        C_branches = S_branches / torch.clamp(N_branches, min=1e-6)
        C = (branch_w * C_branches).sum(dim=0).to(C.dtype)

    return _lloyd_step_cublas(X, C)


def run_hybrid_v14_multiscale(X: torch.Tensor, k: int, epochs: float = 0.5) -> torch.Tensor:
    seed_C = run_engine_v10_fast(X, k=k, use_guarded_island=False)
    return _multiscale_damped_emeans_refine(X, seed_C, epochs=epochs)

# =====================================================================
# Experiment Branch v15: Chunked Damped-Lloyd (Benchmarked on (N, D))
# =====================================================================

def _estimate_damping_lam(X: torch.Tensor, C: torch.Tensor, damp_scale: float = 2.0) -> torch.Tensor:
    n = X.shape[0]
    probe = X[torch.randint(0, n, (min(4096, n),), device=X.device)].float()
    C_f32 = C.float()
    c_half = 0.5 * (C_f32 * C_f32).sum(dim=1)
    max_score = torch.max(torch.addmm(-c_half, probe, C_f32.T), dim=1).values
    dist_sq = torch.clamp((probe * probe).sum(dim=1) - 2.0 * max_score, min=0.0)
    return torch.clamp(damp_scale * torch.median(dist_sq), min=1e-4)


def _damped_lloyd_step_cublas(
    X: torch.Tensor,
    C: torch.Tensor,
    damp_scale: float = 2.0,
    lam: torch.Tensor = None,
    chunk_size: int = 65536,
) -> torch.Tensor:
    n, d = X.shape
    k = C.shape[0]
    if lam is None:
        lam = _estimate_damping_lam(X, C, damp_scale=damp_scale)

    C_f32 = C.float()
    c_half_norm = 0.5 * (C_f32 * C_f32).sum(dim=1)

    new_C = torch.zeros(k, d, device=X.device, dtype=torch.float32)
    counts = torch.zeros(k, 1, device=X.device, dtype=torch.float32)

    for i in range(0, n, chunk_size):
        X_chunk = X[i : i + chunk_size]
        X_f32 = X_chunk.float() if X_chunk.dtype != torch.float32 else X_chunk
        scores = torch.addmm(-c_half_norm, X_f32, C_f32.T)
        max_score, assign = torch.max(scores, dim=1)

        x_sq = (X_f32 * X_f32).sum(dim=1)
        dist_sq = torch.clamp(x_sq - 2.0 * max_score, min=0.0)
        w = (lam / (dist_sq + lam)).to(X_chunk.dtype)

        O_chunk = torch.zeros(X_chunk.shape[0], k, device=X.device, dtype=X_chunk.dtype)
        O_chunk.scatter_(1, assign.unsqueeze(1), w.unsqueeze(1))

        new_C += (O_chunk.T @ X_chunk).float()
        counts += O_chunk.float().sum(dim=0, keepdim=True).T

    dead_mask = (counts < 1e-6)
    updated_C = new_C / torch.clamp(counts, min=1e-6)
    return torch.where(dead_mask, C_f32, updated_C).to(C.dtype)


def run_damped_lloyd_kmeans(X: torch.Tensor, k: int, iters: int = 15, damp_scale: float = 2.0) -> torch.Tensor:
    n = X.shape[0]
    C = X[torch.randint(0, n, (k,), device=X.device)].clone()
    for _ in range(iters):
        C = _damped_lloyd_step_cublas(X, C, damp_scale=damp_scale)
    return C


def run_hybrid_v15_damped_2lloyd(X: torch.Tensor, k: int, damp_scale: float = 2.0) -> torch.Tensor:
    C = run_engine_v10_fast(X, k=k, use_guarded_island=False)
    C = _damped_lloyd_step_cublas(X, C, damp_scale=damp_scale)
    return _damped_lloyd_step_cublas(X, C, damp_scale=damp_scale)



def run_hybrid_v13_damped_lloyd(X: torch.Tensor, k: int, damp_scale: float = 2.0) -> torch.Tensor:
    seed_C = run_engine_v10_fast(X, k=k, use_guarded_island=False)
    return _damped_lloyd_step_cublas(X, seed_C, damp_scale=damp_scale)


def run_engine_v10_float_damped(
    X: torch.Tensor,
    k: int,
    split_mode: str = "otsu",   # "otsu" or "valley"
    smooth: bool = False,
    robust_cap_q: float = 0.95,
) -> torch.Tensor:
    n, d = X.shape
    sub_n = min(16384, n)
    X_sub = X[torch.randint(0, n, (sub_n,), device=X.device)]
    x_sub_sq = (X_sub * X_sub).sum(dim=1)

    C_buf = torch.empty((k, d), device=X.device, dtype=X.dtype)
    C_buf[0] = X_sub.mean(dim=0)

    num_bins = 64
    centers = torch.linspace(0.0, 1.0, num_bins, device=X.device, dtype=torch.float32)

    m = 1
    while m < k:
        b = min(m, max(1, (2 * m + 2) // 3), k - m)
        C_curr = C_buf[:m]
        c_half = 0.5 * (C_curr * C_curr).sum(dim=1)
        scores = torch.addmm(-c_half, X_sub, C_curr.T)
        max_score, assign = torch.max(scores, dim=1)

        raw_sse = torch.clamp(x_sub_sq - 2.0 * max_score, min=0.0).float()
        if robust_cap_q is not None:
            cap = torch.quantile(raw_sse, robust_cap_q)
            pt_sse = torch.clamp(raw_sse, max=cap)
            # 1.0 for 95% of points; bounds outlier 2nd-moment contribution to `cap`
            w_pt = torch.clamp(cap / torch.clamp(raw_sse, min=1e-6), max=1.0).to(X.dtype)
        else:
            pt_sse = raw_sse
            w_pt = torch.ones(sub_n, device=X.device, dtype=X.dtype)

        cluster_sse = torch.zeros(m, device=X.device, dtype=torch.float32)
        cluster_sse.scatter_add_(0, assign, pt_sse)
        _, split_ids = torch.topk(cluster_sse, b, largest=True)

        cluster_to_sel = torch.full((m,), b, device=X.device, dtype=torch.long)
        cluster_to_sel[split_ids] = torch.arange(b, device=X.device)
        sel_pos = cluster_to_sel[assign]
        safe_pos = sel_pos.clamp(max=b - 1)
        in_sel = (sel_pos < b).to(X.dtype)

        O_full = torch.zeros(sub_n, b + 1, device=X.device, dtype=X.dtype)
        O_full.scatter_(1, sel_pos.unsqueeze(1), 1.0)
        O = O_full[:, :b]
        O_w = O * w_pt.unsqueeze(1)

        total_cnt = O.sum(dim=0)
        w_sum = O_w.sum(dim=0).clamp(min=1e-6)
        M = (O_w.T @ X_sub) / w_sum.unsqueeze(1)

        V = torch.randn(b, d, device=X.device, dtype=X.dtype)
        V = V / (V.norm(dim=1, keepdim=True) + 1e-9)

        for _ in range(2):
            dot_x_v = (X_sub @ V.T).gather(1, safe_pos.unsqueeze(1)).squeeze(1) * in_sel
            dot_mu_v = (M * V).sum(dim=1)
            z = (dot_x_v - dot_mu_v[safe_pos]) * in_sel
            Z_mat = O_w * z.unsqueeze(1)
            V_next = Z_mat.T @ X_sub - (Z_mat.sum(dim=0).unsqueeze(1)) * M
            V = V_next / (V_next.norm(dim=1, keepdim=True) + 1e-9)

        proj = (X_sub @ V.T).gather(1, safe_pos.unsqueeze(1)).squeeze(1)
        dot_mu_v = (M * V).sum(dim=1)

        # Weighted float bounds & soft binning
        diff = (proj - dot_mu_v[safe_pos]) * in_sel
        var_buf = torch.zeros(b + 1, device=X.device, dtype=torch.float32)
        var_buf.scatter_add_(0, sel_pos, (w_pt * diff * diff).float())
        var = var_buf[:b] / w_sum.float()
        half_span = (4.0 * torch.sqrt(var.clamp(min=1e-12))).clamp(min=1e-4)
        lo_v = dot_mu_v.float() - half_span
        span = 2.0 * half_span

        pos = ((proj.float() - lo_v[safe_pos]) / span[safe_pos] * (num_bins - 1)).clamp(0.0, float(num_bins - 1) - 1e-4)
        b0 = pos.floor().long()
        b1 = (b0 + 1).clamp(max=num_bins - 1)
        frac = pos - b0.float()
        in_sel_w = (in_sel * w_pt).float()

        hist = torch.zeros((b + 1) * num_bins, device=X.device, dtype=torch.float32)
        hist.scatter_add_(0, sel_pos * num_bins + b0, (1.0 - frac) * in_sel_w)
        hist.scatter_add_(0, sel_pos * num_bins + b1, frac * in_sel_w)
        density = hist[: b * num_bins].view(b, num_bins)

        if smooth:
            d_pad = torch.nn.functional.pad(density, (1, 1), mode="replicate")
            density = 0.25 * d_pad[:, :-2] + 0.5 * d_pad[:, 1:-1] + 0.25 * d_pad[:, 2:]

        if split_mode == "otsu":
            w_L = torch.cumsum(density, dim=1)[:, :-1]
            w_tot = density.sum(dim=1, keepdim=True)
            w_R = w_tot - w_L
            m_L = torch.cumsum(density * centers, dim=1)[:, :-1]
            m_tot = (density * centers).sum(dim=1, keepdim=True)
            m_R = m_tot - m_L
            otsu_score = w_L * w_R * (m_L / w_L.clamp(min=1e-6) - m_R / w_R.clamp(min=1e-6)) ** 2
            best_idx = torch.argmax(otsu_score, dim=1)
            thresh = (lo_v + ((best_idx.float() + 0.5) / (num_bins - 1)) * span).to(X.dtype)
        else:
            left_peak = torch.cummax(density, dim=1).values
            right_peak = torch.cummax(density.flip(1), dim=1).values.flip(1)
            valley_depth = torch.clamp(torch.minimum(left_peak, right_peak) - density, min=0.0)
            cdf = torch.cumsum(density, dim=1) / density.sum(dim=1, keepdim=True).clamp(min=1e-6)
            score = valley_depth * (4.0 * cdf * (1.0 - cdf))
            max_score, best_idx = torch.max(score, dim=1)
            valley_thresh = lo_v + (best_idx.float() / (num_bins - 1)) * span
            thresh = torch.where(max_score > (0.02 * density.max(dim=1).values), valley_thresh, dot_mu_v.float()).to(X.dtype)

        is_left = (proj < thresh[safe_pos]) & (sel_pos < b)
        O_left = O * is_left.to(X.dtype).unsqueeze(1)
        cnt_left = O_left.sum(dim=0)
        cnt_right = total_cnt - cnt_left
        thresh = torch.where((cnt_left < 1.0) | (cnt_right < 1.0), dot_mu_v, thresh)

        is_left = (proj < thresh[safe_pos]) & (sel_pos < b)
        left_f = is_left.to(X.dtype).unsqueeze(1)
        Ow_left = O_w * left_f
        Ow_right = O_w - Ow_left
        cnt_left = (O * left_f).sum(dim=0)
        cnt_right = total_cnt - cnt_left

        C_left = (Ow_left.T @ X_sub) / Ow_left.sum(dim=0).clamp(min=1e-6).unsqueeze(1)
        C_right = (Ow_right.T @ X_sub) / Ow_right.sum(dim=0).clamp(min=1e-6).unsqueeze(1)
        valid = (cnt_left >= 1.0) & (cnt_right >= 1.0)
        C_buf[split_ids] = torch.where(valid.unsqueeze(1), C_left, C_curr[split_ids])
        C_buf[m : m + b] = torch.where(valid.unsqueeze(1), C_right, X_sub[torch.arange(b, device=X.device) % sub_n])
        m += b

    return C_buf

def run_hybrid_v10_float_damped(X: torch.Tensor, k: int, split_mode: str = "otsu", lloyd_steps: int = 2) -> torch.Tensor:
    C = run_engine_v10_float_damped(X, k=k, split_mode=split_mode, smooth=False)
    for _ in range(lloyd_steps):
        C = _damped_lloyd_step_cublas(X, C, damp_scale=2.0)
    return C

def run_engine_v10_kde(
    X: torch.Tensor,
    k: int,
    bw_scale: float = 0.65,
    robust_cap_q: float = 0.95,
    use_sse_gate: bool = True,
    refresh_final: bool = True,
) -> torch.Tensor:
    n, d = X.shape
    sub_n = min(16384, n)
    X_sub = X[torch.randint(0, n, (sub_n,), device=X.device)]
    x_sub_sq = (X_sub * X_sub).sum(dim=1)

    C_buf = torch.empty((k, d), device=X.device, dtype=X.dtype)
    C_buf[0] = X_sub.mean(dim=0)

    num_bins = 64
    idx = torch.arange(num_bins, device=X.device, dtype=torch.float32)
    dist2 = (idx.unsqueeze(0) - idx.unsqueeze(1)) ** 2
    sigma_bins = float(num_bins - 1) / 8.0

    m = 1
    while m < k:
        b_sched = min(m, max(1, (2 * m + 2) // 3), k - m)
        C_curr = C_buf[:m]
        c_half = 0.5 * (C_curr * C_curr).sum(dim=1)
        scores = torch.addmm(-c_half, X_sub, C_curr.T)
        max_score, assign = torch.max(scores, dim=1)

        raw_sse = torch.clamp(x_sub_sq - 2.0 * max_score, min=0.0).float()
        if robust_cap_q is not None:
            cap = torch.quantile(raw_sse, robust_cap_q)
            pt_sse = torch.clamp(raw_sse, max=cap)
            w_pt = torch.clamp(cap / torch.clamp(raw_sse, min=1e-6), max=1.0).to(X.dtype)
        else:
            pt_sse = raw_sse
            w_pt = torch.ones(sub_n, device=X.device, dtype=X.dtype)

        cluster_sse = torch.zeros(m, device=X.device, dtype=torch.float32)
        cluster_sse.scatter_add_(0, assign, pt_sse)

        if use_sse_gate:
            active_cnt = int((cluster_sse >= 0.15 * cluster_sse.max()).sum().item())
            b = max(1, min(b_sched, active_cnt))
        else:
            b = b_sched

        _, split_ids = torch.topk(cluster_sse, b, largest=True)

        cluster_to_sel = torch.full((m,), b, device=X.device, dtype=torch.long)
        cluster_to_sel[split_ids] = torch.arange(b, device=X.device)
        sel_pos = cluster_to_sel[assign]
        safe_pos = sel_pos.clamp(max=b - 1)
        in_sel = (sel_pos < b).to(X.dtype)

        O_full = torch.zeros(sub_n, b + 1, device=X.device, dtype=X.dtype)
        O_full.scatter_(1, sel_pos.unsqueeze(1), 1.0)
        O = O_full[:, :b]
        O_w = O * w_pt.unsqueeze(1)

        total_cnt = O.sum(dim=0)
        w_sum = O_w.sum(dim=0).clamp(min=1e-6)
        M = (O_w.T @ X_sub) / w_sum.unsqueeze(1)

        V = torch.randn(b, d, device=X.device, dtype=X.dtype)
        V = V / (V.norm(dim=1, keepdim=True) + 1e-9)
        for _ in range(2):
            dot_x_v = (X_sub @ V.T).gather(1, safe_pos.unsqueeze(1)).squeeze(1) * in_sel
            dot_mu_v = (M * V).sum(dim=1)
            z = (dot_x_v - dot_mu_v[safe_pos]) * in_sel
            Z_mat = O_w * z.unsqueeze(1)
            V_next = Z_mat.T @ X_sub - (Z_mat.sum(dim=0).unsqueeze(1)) * M
            V = V_next / (V_next.norm(dim=1, keepdim=True) + 1e-9)

        proj = (X_sub @ V.T).gather(1, safe_pos.unsqueeze(1)).squeeze(1)
        dot_mu_v = (M * V).sum(dim=1)

        diff = (proj - dot_mu_v[safe_pos]) * in_sel
        var_buf = torch.zeros(b + 1, device=X.device, dtype=torch.float32)
        var_buf.scatter_add_(0, sel_pos, (w_pt * diff * diff).float())
        var = var_buf[:b] / w_sum.float()
        half_span = (4.0 * torch.sqrt(var.clamp(min=1e-12))).clamp(min=1e-4)
        lo_v = dot_mu_v.float() - half_span
        span = 2.0 * half_span

        pos = ((proj.float() - lo_v[safe_pos]) / span[safe_pos] * (num_bins - 1)).clamp(0.0, float(num_bins - 1) - 1e-4)
        b0 = pos.floor().long()
        b1 = (b0 + 1).clamp(max=num_bins - 1)
        frac = pos - b0.float()
        in_sel_w = (in_sel * w_pt).float()

        hist = torch.zeros((b + 1) * num_bins, device=X.device, dtype=torch.float32)
        hist.scatter_add_(0, sel_pos * num_bins + b0, (1.0 - frac) * in_sel_w)
        hist.scatter_add_(0, sel_pos * num_bins + b1, frac * in_sel_w)
        raw_linear = hist[: b * num_bins].view(b, num_bins)

        # FFT-free per-cluster Silverman Gaussian KDE via batched 64x64 GEMM
        h_bins = bw_scale * sigma_bins * total_cnt.float().clamp(min=2.0).pow(-0.2)
        h2 = torch.clamp(h_bins * h_bins, min=0.05).view(b, 1, 1)
        K = torch.exp(-0.5 * dist2.unsqueeze(0) / h2)
        K = K / K.sum(dim=1, keepdim=True).clamp(min=1e-6)
        density = torch.bmm(K, raw_linear.unsqueeze(2)).squeeze(2)

        left_peak = torch.cummax(density, dim=1).values
        right_peak = torch.cummax(density.flip(1), dim=1).values.flip(1)
        valley_depth = torch.clamp(torch.minimum(left_peak, right_peak) - density, min=0.0)
        cdf = torch.cumsum(density, dim=1) / density.sum(dim=1, keepdim=True).clamp(min=1e-6)
        v_max, v_idx = torch.max(valley_depth * (4.0 * cdf * (1.0 - cdf)), dim=1)
        v_cand = lo_v + (v_idx.float() / (num_bins - 1)) * span
        thresh = torch.where(v_max > (0.02 * density.max(dim=1).values), v_cand, dot_mu_v.float()).to(X.dtype)

        is_left = (proj < thresh[safe_pos]) & (sel_pos < b)
        cnt_left = (O * is_left.to(X.dtype).unsqueeze(1)).sum(dim=0)
        cnt_right = total_cnt - cnt_left
        thresh = torch.where((cnt_left < 1.0) | (cnt_right < 1.0), dot_mu_v, thresh)

        is_left = (proj < thresh[safe_pos]) & (sel_pos < b)
        left_f = is_left.to(X.dtype).unsqueeze(1)
        Ow_left = O_w * left_f
        Ow_right = O_w - Ow_left
        cnt_left = (O * left_f).sum(dim=0)
        cnt_right = total_cnt - cnt_left

        C_left = (Ow_left.T @ X_sub) / Ow_left.sum(dim=0).clamp(min=1e-6).unsqueeze(1)
        C_right = (Ow_right.T @ X_sub) / Ow_right.sum(dim=0).clamp(min=1e-6).unsqueeze(1)
        valid = (cnt_left >= 1.0) & (cnt_right >= 1.0)
        C_buf[split_ids] = torch.where(valid.unsqueeze(1), C_left, C_curr[split_ids])
        C_buf[m : m + b] = torch.where(valid.unsqueeze(1), C_right, X_sub[torch.arange(b, device=X.device) % sub_n])
        m += b

    if refresh_final:
        c_half = 0.5 * (C_buf * C_buf).sum(dim=1)
        max_sc, assign_fin = torch.max(torch.addmm(-c_half, X_sub, C_buf.T), dim=1)
        raw_sse_fin = torch.clamp(x_sub_sq - 2.0 * max_sc, min=0.0).float()
        w_fin = torch.clamp(torch.quantile(raw_sse_fin, robust_cap_q) / torch.clamp(raw_sse_fin, min=1e-6), max=1.0).to(X.dtype) if robust_cap_q is not None else torch.ones(sub_n, device=X.device, dtype=X.dtype)
        O_fin = torch.zeros(sub_n, k, device=X.device, dtype=X.dtype)
        O_fin.scatter_(1, assign_fin.unsqueeze(1), 1.0)
        Ow_fin = O_fin * w_fin.unsqueeze(1)
        w_tot = Ow_fin.sum(dim=0)
        C_new = (Ow_fin.T @ X_sub) / w_tot.clamp(min=1e-6).unsqueeze(1)
        C_buf = torch.where((w_tot > 1e-6).unsqueeze(1), C_new, C_buf)

    return C_buf


def run_engine_v10_aware(
    X: torch.Tensor,
    k: int,
    robust_cap_q: float = 0.95,
    bw_scale: float = 0.65,
    refresh_final: bool = True,
) -> torch.Tensor:
    n, d = X.shape
    sub_n = min(16384, n)
    X_sub = X[torch.randint(0, n, (sub_n,), device=X.device)]
    x_sub_sq = (X_sub * X_sub).sum(dim=1)
    is_B = (torch.arange(sub_n, device=X.device) & 1) == 1

    C_buf = torch.empty((k, d), device=X.device, dtype=X.dtype)
    C_buf[0] = X_sub.mean(dim=0)
    prev_V = torch.zeros((k, d), device=X.device, dtype=X.dtype)

    num_bins = 64
    idx = torch.arange(num_bins, device=X.device, dtype=torch.float32)
    dist2 = (idx.unsqueeze(0) - idx.unsqueeze(1)) ** 2
    sigma_bins = float(num_bins - 1) / 8.0

    m = 1
    while m < k:
        b_sched = min(m, max(1, (2 * m + 2) // 3), k - m)
        C_curr = C_buf[:m]
        c_half = 0.5 * (C_curr * C_curr).sum(dim=1)
        max_score, assign = torch.max(torch.addmm(-c_half, X_sub, C_curr.T), dim=1)

        raw_sse = torch.clamp(x_sub_sq - 2.0 * max_score, min=0.0).float()
        if robust_cap_q is not None:
            cap = torch.quantile(raw_sse, robust_cap_q)
            w_pt = torch.clamp(cap / torch.clamp(raw_sse, min=1e-6), max=1.0).to(X.dtype)
        else:
            w_pt = torch.ones(sub_n, device=X.device, dtype=X.dtype)

        O = torch.zeros(sub_n, m, device=X.device, dtype=X.dtype)
        O.scatter_(1, assign.unsqueeze(1), 1.0)
        O_w = O * w_pt.unsqueeze(1)

        total_cnt = O.sum(dim=0)
        w_sum = O_w.sum(dim=0).clamp(min=1e-6)
        M = (O_w.T @ X_sub) / w_sum.unsqueeze(1)

        # Rotation-aware probe: orthogonalize against parent cut axis prev_V[:m]
        V = torch.randn(m, d, device=X.device, dtype=X.dtype)
        pv = prev_V[:m]
        V = V - (V * pv).sum(dim=1, keepdim=True) * pv
        V = V / (V.norm(dim=1, keepdim=True) + 1e-9)

        for _ in range(3):
            dot_x_v = (X_sub @ V.T).gather(1, assign.unsqueeze(1)).squeeze(1)
            dot_mu_v = (M * V).sum(dim=1)
            z = dot_x_v - dot_mu_v[assign]
            Z_mat = O_w * z.unsqueeze(1)
            V_next = Z_mat.T @ X_sub - (Z_mat.sum(dim=0).unsqueeze(1)) * M
            V = V_next / (V_next.norm(dim=1, keepdim=True) + 1e-9)

        proj = (X_sub @ V.T).gather(1, assign.unsqueeze(1)).squeeze(1)
        dot_mu_v = (M * V).sum(dim=1)

        # Split-half (A vs B) batched bmm Gaussian KDE
        diff = proj - dot_mu_v[assign]
        var_buf = torch.zeros(m, device=X.device, dtype=torch.float32)
        var_buf.scatter_add_(0, assign, (w_pt * diff * diff).float())
        var = var_buf / w_sum.float()
        half_span = (4.0 * torch.sqrt(var.clamp(min=1e-12))).clamp(min=1e-4)
        lo_v = dot_mu_v.float() - half_span
        span = 2.0 * half_span

        pos = ((proj.float() - lo_v[assign]) / span[assign] * (num_bins - 1)).clamp(0.0, float(num_bins - 1) - 1e-4)
        b0 = pos.floor().long()
        b1 = (b0 + 1).clamp(max=num_bins - 1)
        frac = pos - b0.float()

        flat_base = is_B.long() * (m * num_bins) + assign * num_bins
        hist_ab = torch.zeros(2 * m * num_bins, device=X.device, dtype=torch.float32)
        hist_ab.scatter_add_(0, flat_base + b0, (1.0 - frac) * w_pt.float())
        hist_ab.scatter_add_(0, flat_base + b1, frac * w_pt.float())
        raw_ab = hist_ab.view(2, m, num_bins).permute(1, 2, 0)  # (m, 64, 2)

        h_bins = bw_scale * sigma_bins * (0.5 * total_cnt.float()).clamp(min=2.0).pow(-0.2)
        h2 = torch.clamp(h_bins * h_bins, min=0.05).view(m, 1, 1)
        K = torch.exp(-0.5 * dist2.unsqueeze(0) / h2)
        K = K / K.sum(dim=1, keepdim=True).clamp(min=1e-6)
        dens_ab = torch.bmm(K, raw_ab)  # (m, 64, 2)
        dens_A, dens_B = dens_ab[:, :, 0], dens_ab[:, :, 1]

        lp_A = torch.cummax(dens_A, dim=1).values
        rp_A = torch.cummax(dens_A.flip(1), dim=1).values.flip(1)
        lp_B = torch.cummax(dens_B, dim=1).values
        rp_B = torch.cummax(dens_B.flip(1), dim=1).values.flip(1)

        val_persist = torch.minimum(
            torch.clamp(torch.minimum(lp_A, rp_A) - dens_A, min=0.0),
            torch.clamp(torch.minimum(lp_B, rp_B) - dens_B, min=0.0),
        )
        dens_mean = 0.5 * (dens_A + dens_B)
        cdf = torch.cumsum(dens_mean, dim=1) / dens_mean.sum(dim=1, keepdim=True).clamp(min=1e-6)
        v_max, v_idx = torch.max(val_persist * (4.0 * cdf * (1.0 - cdf)), dim=1)

        df_A = dens_A[:, 1:] - dens_A[:, :-1]
        df_B = dens_B[:, 1:] - dens_B[:, :-1]
        agree = (df_A * df_B).sum(dim=1) / ((df_A.norm(dim=1) * df_B.norm(dim=1)) + 1e-8)

        has_valley = (v_max > 0.03 * torch.minimum(dens_A, dens_B).max(dim=1).values) & (agree > 0.35)
        v_cand = lo_v + (v_idx.float() / (num_bins - 1)) * span
        thresh = torch.where(has_valley, v_cand, dot_mu_v.float()).to(X.dtype)

        is_left = proj < thresh[assign]
        cnt_left = (O * is_left.to(X.dtype).unsqueeze(1)).sum(dim=0)
        cnt_right = total_cnt - cnt_left
        thresh = torch.where((cnt_left < 1.0) | (cnt_right < 1.0), dot_mu_v, thresh)

        left_f = (proj < thresh[assign]).to(X.dtype).unsqueeze(1)
        Ow_left = O_w * left_f
        Ow_right = O_w - Ow_left
        cnt_left = (O * left_f).sum(dim=0)
        cnt_right = total_cnt - cnt_left

        wl = Ow_left.sum(dim=0)
        wr = Ow_right.sum(dim=0)
        C_left = (Ow_left.T @ X_sub) / wl.clamp(min=1e-6).unsqueeze(1)
        C_right = (Ow_right.T @ X_sub) / wr.clamp(min=1e-6).unsqueeze(1)

        valid = (cnt_left >= 1.0) & (cnt_right >= 1.0)
        delta_sse = (wl * wr / (wl + wr).clamp(min=1e-6)) * ((C_left - C_right) ** 2).sum(dim=1)
        delta_sse = torch.where(valid, delta_sse, torch.zeros_like(delta_sse))

        # Peak / split-gain aware budget: skip already-isolated single peaks & 1024D Giant slices
        active_cnt = int((delta_sse >= 0.05 * delta_sse.max()).sum().item())
        b = max(1, min(b_sched, active_cnt))
        _, top_ids = torch.topk(delta_sse, b, largest=True)

        ok = valid[top_ids]
        C_buf[top_ids] = torch.where(ok.unsqueeze(1), C_left[top_ids], C_curr[top_ids])
        C_buf[m : m + b] = torch.where(ok.unsqueeze(1), C_right[top_ids], X_sub[torch.arange(b, device=X.device) % sub_n])
        prev_V[top_ids] = V[top_ids]
        prev_V[m : m + b] = V[top_ids]
        m += b

    if refresh_final:
        c_half = 0.5 * (C_buf * C_buf).sum(dim=1)
        max_sc, assign_fin = torch.max(torch.addmm(-c_half, X_sub, C_buf.T), dim=1)
        raw_sse_fin = torch.clamp(x_sub_sq - 2.0 * max_sc, min=0.0).float()
        w_fin = torch.clamp(torch.quantile(raw_sse_fin, robust_cap_q) / torch.clamp(raw_sse_fin, min=1e-6), max=1.0).to(X.dtype) if robust_cap_q is not None else torch.ones(sub_n, device=X.device, dtype=X.dtype)
        O_fin = torch.zeros(sub_n, k, device=X.device, dtype=X.dtype).scatter_(1, assign_fin.unsqueeze(1), 1.0)
        Ow_fin = O_fin * w_fin.unsqueeze(1)
        w_tot = Ow_fin.sum(dim=0)
        C_new = (Ow_fin.T @ X_sub) / w_tot.clamp(min=1e-6).unsqueeze(1)
        C_buf = torch.where((w_tot > 1e-6).unsqueeze(1), C_new, C_buf)

    return C_buf


@torch.no_grad()
def run_hybrid_v10_aware_2dlloyd(
    X: torch.Tensor,
    k: int,
    damp_scale: float = 2.0,
) -> torch.Tensor:
    C = run_engine_v10_aware(X, k)
    C = _damped_lloyd_step_cublas(X, C, damp_scale=damp_scale)
    C = _damped_lloyd_step_cublas(X, C, damp_scale=damp_scale)
    return C

@torch.no_grad()
def run_engine_v10_aware_low_vram(
    X: torch.Tensor,
    k: int,
    robust_cap_q: float = 0.95,
    bw_scale: float = 0.65,
    refresh_final: bool = True,
    sub_chunk: int = 4096,
) -> torch.Tensor:
    n, d = X.shape
    dev = X.device
    sub_n = min(16384, n)
    X_sub = X[torch.randint(0, n, (sub_n,), device=dev)]
    x_sub_sq = torch.linalg.vector_norm(X_sub.float(), dim=1).square()
    is_B = (torch.arange(sub_n, device=dev) & 1) == 1

    C_buf = torch.empty((k, d), device=dev, dtype=X.dtype)
    C_buf[0] = X_sub.mean(dim=0)
    prev_V = torch.zeros((k, d), device=dev, dtype=X.dtype)

    num_bins = 64
    idx = torch.arange(num_bins, device=dev, dtype=torch.float32)
    dist2 = (idx.unsqueeze(0) - idx.unsqueeze(1)) ** 2
    sigma_bins = float(num_bins - 1) / 8.0

    # Reusable 1D buffers across all rounds (avoids allocator staircase)
    max_score = torch.empty(sub_n, device=dev, dtype=torch.float32)
    assign = torch.empty(sub_n, device=dev, dtype=torch.long)
    proj = torch.empty(sub_n, device=dev, dtype=torch.float32)

    m = 1
    while m < k:
        b_sched = min(m, max(1, (2 * m + 2) // 3), k - m)
        C_curr = C_buf[:m]
        c_half = 0.5 * torch.linalg.vector_norm(C_curr.float(), dim=1).square().to(X.dtype)

        # 1. Chunked assignment (caps scores at sub_chunk x m)
        for s in range(0, sub_n, sub_chunk):
            e = min(s + sub_chunk, sub_n)
            sc = torch.addmm(-c_half, X_sub[s:e], C_curr.T)
            ms, asg = torch.max(sc, dim=1)
            max_score[s:e] = ms.float()
            assign[s:e] = asg
            del sc

        raw_sse = torch.clamp(x_sub_sq - 2.0 * max_score, min=0.0)
        if robust_cap_q is not None:
            cap = torch.quantile(raw_sse, robust_cap_q)
            w_pt = torch.clamp(cap / torch.clamp(raw_sse, min=1e-6), max=1.0)
        else:
            w_pt = torch.ones(sub_n, device=dev, dtype=torch.float32)

        total_cnt = torch.bincount(assign, minlength=m).float()
        w_sum = torch.zeros(m, device=dev, dtype=torch.float32)
        w_sum.index_add_(0, assign, w_pt)
        w_sum_safe = w_sum.clamp(min=1e-6)

        # 2. Weighted cluster means without building (sub_n, m) O_w
        M_sum = torch.zeros(m, d, device=dev, dtype=torch.float32)
        for s in range(0, sub_n, sub_chunk):
            e = min(s + sub_chunk, sub_n)
            M_sum.index_add_(0, assign[s:e], X_sub[s:e].float() * w_pt[s:e].unsqueeze(1))
        M = M_sum / w_sum_safe.unsqueeze(1)

        # 3. Rotation-aware probe orthogonalized against parent cut axis
        V = torch.randn(m, d, device=dev, dtype=torch.float32)
        pv = prev_V[:m].float()
        V = V - (V * pv).sum(dim=1, keepdim=True) * pv
        V = V / (V.norm(dim=1, keepdim=True) + 1e-9)

        # 4. Power iteration without (sub_n, m) X_sub @ V.T or Z_mat
        V_next = torch.empty((m, d), device=dev, dtype=torch.float32)
        zw_sum = torch.empty(m, device=dev, dtype=torch.float32)
        for _ in range(3):
            V_next.zero_()
            zw_sum.zero_()
            dot_mu_v = (M * V).sum(dim=1)
            for s in range(0, sub_n, sub_chunk):
                e = min(s + sub_chunk, sub_n)
                a_c = assign[s:e]
                X_c = X_sub[s:e].float()
                dot_x_v_c = torch.einsum("nd,nd->n", X_c, V[a_c])
                zw_c = w_pt[s:e] * (dot_x_v_c - dot_mu_v[a_c])
                V_next.index_add_(0, a_c, X_c * zw_c.unsqueeze(1))
                zw_sum.index_add_(0, a_c, zw_c)
            V_next -= zw_sum.unsqueeze(1) * M
            V = V_next / (V_next.norm(dim=1, keepdim=True) + 1e-9)

        for s in range(0, sub_n, sub_chunk):
            e = min(s + sub_chunk, sub_n)
            a_c = assign[s:e]
            proj[s:e] = torch.einsum("nd,nd->n", X_sub[s:e].float(), V[a_c])
        dot_mu_v = (M * V).sum(dim=1)

        # 5. Split-half (A vs B) Gaussian KDE (chunked over m if m > 512)
        diff = proj - dot_mu_v[assign]
        var_buf = torch.zeros(m, device=dev, dtype=torch.float32)
        var_buf.index_add_(0, assign, w_pt * diff * diff)
        var = var_buf / w_sum_safe
        half_span = (4.0 * torch.sqrt(var.clamp(min=1e-12))).clamp(min=1e-4)
        lo_v = dot_mu_v - half_span
        span = 2.0 * half_span

        pos = ((proj - lo_v[assign]) / span[assign] * (num_bins - 1)).clamp(0.0, float(num_bins - 1) - 1e-4)
        b0 = pos.floor().long()
        b1 = (b0 + 1).clamp(max=num_bins - 1)
        frac = pos - b0.float()

        flat_base = is_B.long() * (m * num_bins) + assign * num_bins
        hist_ab = torch.zeros(2 * m * num_bins, device=dev, dtype=torch.float32)
        hist_ab.scatter_add_(0, flat_base + b0, (1.0 - frac) * w_pt)
        hist_ab.scatter_add_(0, flat_base + b1, frac * w_pt)
        raw_ab = hist_ab.view(2, m, num_bins).permute(1, 2, 0)  # (m, 64, 2)

        h_bins = bw_scale * sigma_bins * (0.5 * total_cnt).clamp(min=2.0).pow(-0.2)
        h2 = torch.clamp(h_bins * h_bins, min=0.05)

        dens_ab = torch.empty_like(raw_ab)
        for ms in range(0, m, 512):
            me = min(ms + 512, m)
            K_blk = torch.exp(-0.5 * dist2.unsqueeze(0) / h2[ms:me].view(-1, 1, 1))
            K_blk = K_blk / K_blk.sum(dim=1, keepdim=True).clamp(min=1e-6)
            dens_ab[ms:me] = torch.bmm(K_blk, raw_ab[ms:me])
            del K_blk

        dens_A, dens_B = dens_ab[:, :, 0], dens_ab[:, :, 1]
        lp_A = torch.cummax(dens_A, dim=1).values
        rp_A = torch.cummax(dens_A.flip(1), dim=1).values.flip(1)
        lp_B = torch.cummax(dens_B, dim=1).values
        rp_B = torch.cummax(dens_B.flip(1), dim=1).values.flip(1)

        val_persist = torch.minimum(
            torch.clamp(torch.minimum(lp_A, rp_A) - dens_A, min=0.0),
            torch.clamp(torch.minimum(lp_B, rp_B) - dens_B, min=0.0),
        )
        dens_mean = 0.5 * (dens_A + dens_B)
        cdf = torch.cumsum(dens_mean, dim=1) / dens_mean.sum(dim=1, keepdim=True).clamp(min=1e-6)
        v_max, v_idx = torch.max(val_persist * (4.0 * cdf * (1.0 - cdf)), dim=1)

        df_A = dens_A[:, 1:] - dens_A[:, :-1]
        df_B = dens_B[:, 1:] - dens_B[:, :-1]
        agree = (df_A * df_B).sum(dim=1) / ((df_A.norm(dim=1) * df_B.norm(dim=1)) + 1e-8)

        has_valley = (v_max > 0.03 * torch.minimum(dens_A, dens_B).max(dim=1).values) & (agree > 0.35)
        v_cand = lo_v + (v_idx.float() / (num_bins - 1)) * span
        thresh = torch.where(has_valley, v_cand, dot_mu_v)

        is_left = proj < thresh[assign]
        cnt_left = torch.bincount(assign[is_left], minlength=m).float()
        cnt_right = total_cnt - cnt_left
        thresh = torch.where((cnt_left < 1.0) | (cnt_right < 1.0), dot_mu_v, thresh)

        # 6. Left/Right centroids via single-pass subtraction (no Ow_left/Ow_right)
        is_left = proj < thresh[assign]
        cnt_left = torch.bincount(assign[is_left], minlength=m).float()
        cnt_right = total_cnt - cnt_left

        wl = torch.zeros(m, device=dev, dtype=torch.float32)
        wl.index_add_(0, assign[is_left], w_pt[is_left])
        wr = w_sum - wl

        C_left = torch.zeros(m, d, device=dev, dtype=torch.float32)
        for s in range(0, sub_n, sub_chunk):
            e = min(s + sub_chunk, sub_n)
            mask_c = is_left[s:e]
            if mask_c.any():
                C_left.index_add_(
                    0,
                    assign[s:e][mask_c],
                    X_sub[s:e][mask_c].float() * w_pt[s:e][mask_c].unsqueeze(1),
                )

        C_right = (M_sum - C_left) / wr.clamp(min=1e-6).unsqueeze(1)
        C_left = C_left / wl.clamp(min=1e-6).unsqueeze(1)

        valid = (cnt_left >= 1.0) & (cnt_right >= 1.0)
        delta_sse = (wl * wr / (wl + wr).clamp(min=1e-6)) * torch.linalg.vector_norm(C_left - C_right, dim=1).square()
        delta_sse = torch.where(valid, delta_sse, torch.zeros_like(delta_sse))

        active_cnt = int((delta_sse >= 0.05 * delta_sse.max()).sum().item())
        b = max(1, min(b_sched, active_cnt))
        _, top_ids = torch.topk(delta_sse, b, largest=True)

        ok = valid[top_ids]
        C_buf[top_ids] = torch.where(ok.unsqueeze(1), C_left[top_ids].to(X.dtype), C_curr[top_ids])
        C_buf[m : m + b] = torch.where(ok.unsqueeze(1), C_right[top_ids].to(X.dtype), X_sub[torch.arange(b, device=dev) % sub_n])
        V_cast = V.to(X.dtype)
        prev_V[top_ids] = V_cast[top_ids]
        prev_V[m : m + b] = V_cast[top_ids]
        m += b

    if refresh_final:
        c_half = 0.5 * torch.linalg.vector_norm(C_buf.float(), dim=1).square().to(X.dtype)
        for s in range(0, sub_n, sub_chunk):
            e = min(s + sub_chunk, sub_n)
            sc = torch.addmm(-c_half, X_sub[s:e], C_buf.T)
            ms, asg = torch.max(sc, dim=1)
            max_score[s:e] = ms.float()
            assign[s:e] = asg
            del sc

        raw_sse_fin = torch.clamp(x_sub_sq - 2.0 * max_score, min=0.0)
        w_fin = (
            torch.clamp(torch.quantile(raw_sse_fin, robust_cap_q) / torch.clamp(raw_sse_fin, min=1e-6), max=1.0)
            if robust_cap_q is not None
            else torch.ones(sub_n, device=dev, dtype=torch.float32)
        )
        w_tot = torch.zeros(k, device=dev, dtype=torch.float32)
        w_tot.index_add_(0, assign, w_fin)

        C_new = torch.zeros(k, d, device=dev, dtype=torch.float32)
        for s in range(0, sub_n, sub_chunk):
            e = min(s + sub_chunk, sub_n)
            C_new.index_add_(0, assign[s:e], X_sub[s:e].float() * w_fin[s:e].unsqueeze(1))
        C_new = (C_new / w_tot.clamp(min=1e-6).unsqueeze(1)).to(X.dtype)
        C_buf = torch.where((w_tot > 1e-6).unsqueeze(1), C_new, C_buf)

    return C_buf


@torch.no_grad()
def _damped_lloyd_step_low_vram(
    X: torch.Tensor,
    C: torch.Tensor,
    damp_scale: float = 2.0,
    lam: torch.Tensor = None,
) -> torch.Tensor:
    n, d = X.shape
    k = C.shape[0]
    dev = X.device

    C_f32 = C.float()
    c_half_norm = 0.5 * torch.linalg.vector_norm(C_f32, dim=1).square()

    if lam is None:
        probe = X[torch.randint(0, n, (min(4096, n),), device=dev)].float()
        max_sc = torch.max(torch.addmm(-c_half_norm, probe, C_f32.T), dim=1).values
        probe_sq = torch.linalg.vector_norm(probe, dim=1).square()
        lam = torch.clamp(damp_scale * torch.median(torch.clamp(probe_sq - 2.0 * max_sc, min=0.0)), min=1e-4)
        del probe, max_sc, probe_sq

    # Cap chunk_size so scores/O_chunk stay small even at large k
    chunk_size = max(2048, min(16384, (8 * 1024 * 1024) // max(k, 1)))
    use_gemm = (k <= 512)
    O_buf = torch.empty((chunk_size, k), device=dev, dtype=X.dtype) if use_gemm else None

    new_C = torch.zeros(k, d, device=dev, dtype=torch.float32)
    counts = torch.zeros(k, 1, device=dev, dtype=torch.float32)

    for i in range(0, n, chunk_size):
        X_chunk = X[i : i + chunk_size]
        cur = X_chunk.shape[0]
        X_f32 = X_chunk.float() if X_chunk.dtype != torch.float32 else X_chunk

        scores = torch.addmm(-c_half_norm, X_f32, C_f32.T)
        max_score, assign = torch.max(scores, dim=1)
        del scores

        x_sq = torch.linalg.vector_norm(X_f32, dim=1).square()
        dist_sq = torch.clamp(x_sq - 2.0 * max_score, min=0.0)
        w = lam / (dist_sq + lam)

        if use_gemm:
            O_cur = O_buf[:cur]
            O_cur.zero_()
            O_cur.scatter_(1, assign.unsqueeze(1), w.to(X.dtype).unsqueeze(1))
            new_C += (O_cur.T @ X_chunk).float()
            counts += O_cur.float().sum(dim=0, keepdim=True).T
        else:
            new_C.index_add_(0, assign, X_f32 * w.unsqueeze(1))
            counts[:, 0].index_add_(0, assign, w)

    dead_mask = (counts < 1e-6)
    updated_C = new_C / torch.clamp(counts, min=1e-6)
    return torch.where(dead_mask, C_f32, updated_C).to(C.dtype)


@torch.no_grad()
def run_hybrid_v10_aware_2dlloyd_low(
    X: torch.Tensor,
    k: int,
    damp_scale: float = 2.0,
) -> torch.Tensor:
    C = run_engine_v10_aware_low_vram(X, k)
    C = _damped_lloyd_step_low_vram(X, C, damp_scale=damp_scale)
    C = _damped_lloyd_step_low_vram(X, C, damp_scale=damp_scale)
    return C


@torch.no_grad()
def run_engine_v10_aware_fast_low(
    X: torch.Tensor,
    k: int,
    robust_cap_q: float = 0.95,
    bw_scale: float = 0.65,
    refresh_final: bool = True,
) -> torch.Tensor:
    n, d = X.shape
    sub_n = min(16384, n)
    X_sub = X[torch.randint(0, n, (sub_n,), device=X.device)]
    # Zero (sub_n, d) temporary allocation vs (X_sub * X_sub).sum(dim=1)
    x_sub_sq = torch.linalg.vector_norm(X_sub.float(), dim=1).square()
    is_B = (torch.arange(sub_n, device=X.device) & 1) == 1

    C_buf = torch.empty((k, d), device=X.device, dtype=X.dtype)
    C_buf[0] = X_sub.mean(dim=0)
    prev_V = torch.zeros((k, d), device=X.device, dtype=X.dtype)

    num_bins = 64
    idx = torch.arange(num_bins, device=X.device, dtype=torch.float32)
    dist2 = (idx.unsqueeze(0) - idx.unsqueeze(1)) ** 2
    sigma_bins = float(num_bins - 1) / 8.0

    m = 1
    while m < k:
        b_sched = min(m, max(1, (2 * m + 2) // 3), k - m)
        C_curr = C_buf[:m]
        c_half = 0.5 * torch.linalg.vector_norm(C_curr, dim=1).square()
        max_score, assign = torch.max(torch.addmm(-c_half, X_sub, C_curr.T), dim=1)

        raw_sse = torch.clamp(x_sub_sq - 2.0 * max_score.float(), min=0.0)
        if robust_cap_q is not None:
            cap = torch.quantile(raw_sse, robust_cap_q)
            w_pt = torch.clamp(cap / torch.clamp(raw_sse, min=1e-6), max=1.0).to(X.dtype)
        else:
            w_pt = torch.ones(sub_n, device=X.device, dtype=X.dtype)

        O = torch.zeros(sub_n, m, device=X.device, dtype=X.dtype)
        O.scatter_(1, assign.unsqueeze(1), 1.0)
        O_w = O * w_pt.unsqueeze(1)

        total_cnt = O.sum(dim=0)
        w_sum = O_w.sum(dim=0).clamp(min=1e-6)
        M = (O_w.T @ X_sub) / w_sum.unsqueeze(1)

        V = torch.randn(m, d, device=X.device, dtype=X.dtype)
        pv = prev_V[:m]
        V = V - (V * pv).sum(dim=1, keepdim=True) * pv
        V = V / (V.norm(dim=1, keepdim=True) + 1e-9)

        for _ in range(3):
            dot_x_v = (X_sub @ V.T).gather(1, assign.unsqueeze(1)).squeeze(1)
            dot_mu_v = (M * V).sum(dim=1)
            z = dot_x_v - dot_mu_v[assign]
            Z_mat = O_w * z.unsqueeze(1)
            V_next = Z_mat.T @ X_sub - (Z_mat.sum(dim=0).unsqueeze(1)) * M
            V = V_next / (V_next.norm(dim=1, keepdim=True) + 1e-9)

        proj = (X_sub @ V.T).gather(1, assign.unsqueeze(1)).squeeze(1)
        dot_mu_v = (M * V).sum(dim=1)

        diff = proj - dot_mu_v[assign]
        var_buf = torch.zeros(m, device=X.device, dtype=torch.float32)
        var_buf.scatter_add_(0, assign, (w_pt * diff * diff).float())
        var = var_buf / w_sum.float()
        half_span = (4.0 * torch.sqrt(var.clamp(min=1e-12))).clamp(min=1e-4)
        lo_v = dot_mu_v.float() - half_span
        span = 2.0 * half_span

        pos = ((proj.float() - lo_v[assign]) / span[assign] * (num_bins - 1)).clamp(0.0, float(num_bins - 1) - 1e-4)
        b0 = pos.floor().long()
        b1 = (b0 + 1).clamp(max=num_bins - 1)
        frac = pos - b0.float()

        flat_base = is_B.long() * (m * num_bins) + assign * num_bins
        hist_ab = torch.zeros(2 * m * num_bins, device=X.device, dtype=torch.float32)
        hist_ab.scatter_add_(0, flat_base + b0, (1.0 - frac) * w_pt.float())
        hist_ab.scatter_add_(0, flat_base + b1, frac * w_pt.float())
        raw_ab = hist_ab.view(2, m, num_bins).permute(1, 2, 0)

        h_bins = bw_scale * sigma_bins * (0.5 * total_cnt.float()).clamp(min=2.0).pow(-0.2)
        h2 = torch.clamp(h_bins * h_bins, min=0.05).view(m, 1, 1)
        K = torch.exp(-0.5 * dist2.unsqueeze(0) / h2)
        K = K / K.sum(dim=1, keepdim=True).clamp(min=1e-6)
        dens_ab = torch.bmm(K, raw_ab)
        dens_A, dens_B = dens_ab[:, :, 0], dens_ab[:, :, 1]

        lp_A = torch.cummax(dens_A, dim=1).values
        rp_A = torch.cummax(dens_A.flip(1), dim=1).values.flip(1)
        lp_B = torch.cummax(dens_B, dim=1).values
        rp_B = torch.cummax(dens_B.flip(1), dim=1).values.flip(1)

        val_persist = torch.minimum(
            torch.clamp(torch.minimum(lp_A, rp_A) - dens_A, min=0.0),
            torch.clamp(torch.minimum(lp_B, rp_B) - dens_B, min=0.0),
        )
        dens_mean = 0.5 * (dens_A + dens_B)
        cdf = torch.cumsum(dens_mean, dim=1) / dens_mean.sum(dim=1, keepdim=True).clamp(min=1e-6)
        v_max, v_idx = torch.max(val_persist * (4.0 * cdf * (1.0 - cdf)), dim=1)

        df_A = dens_A[:, 1:] - dens_A[:, :-1]
        df_B = dens_B[:, 1:] - dens_B[:, :-1]
        agree = (df_A * df_B).sum(dim=1) / ((df_A.norm(dim=1) * df_B.norm(dim=1)) + 1e-8)

        has_valley = (v_max > 0.03 * torch.minimum(dens_A, dens_B).max(dim=1).values) & (agree > 0.35)
        v_cand = lo_v + (v_idx.float() / (num_bins - 1)) * span
        thresh = torch.where(has_valley, v_cand, dot_mu_v.float()).to(X.dtype)

        is_left = proj < thresh[assign]
        cnt_left = (O * is_left.to(X.dtype).unsqueeze(1)).sum(dim=0)
        cnt_right = total_cnt - cnt_left
        thresh = torch.where((cnt_left < 1.0) | (cnt_right < 1.0), dot_mu_v, thresh)

        left_f = (proj < thresh[assign]).to(X.dtype).unsqueeze(1)
        Ow_left = O_w * left_f
        Ow_right = O_w - Ow_left
        cnt_left = (O * left_f).sum(dim=0)
        cnt_right = total_cnt - cnt_left

        wl = Ow_left.sum(dim=0)
        wr = Ow_right.sum(dim=0)
        C_left = (Ow_left.T @ X_sub) / wl.clamp(min=1e-6).unsqueeze(1)
        C_right = (Ow_right.T @ X_sub) / wr.clamp(min=1e-6).unsqueeze(1)

        valid = (cnt_left >= 1.0) & (cnt_right >= 1.0)
        delta_sse = (wl * wr / (wl + wr).clamp(min=1e-6)) * torch.linalg.vector_norm(C_left - C_right, dim=1).square()
        delta_sse = torch.where(valid, delta_sse, torch.zeros_like(delta_sse))

        active_cnt = int((delta_sse >= 0.05 * delta_sse.max()).sum().item())
        b = max(1, min(b_sched, active_cnt))
        _, top_ids = torch.topk(delta_sse, b, largest=True)

        ok = valid[top_ids]
        C_buf[top_ids] = torch.where(ok.unsqueeze(1), C_left[top_ids], C_curr[top_ids])
        C_buf[m : m + b] = torch.where(ok.unsqueeze(1), C_right[top_ids], X_sub[torch.arange(b, device=X.device) % sub_n])
        prev_V[top_ids] = V[top_ids]
        prev_V[m : m + b] = V[top_ids]
        m += b

    if refresh_final:
        c_half = 0.5 * torch.linalg.vector_norm(C_buf, dim=1).square()
        max_sc, assign_fin = torch.max(torch.addmm(-c_half, X_sub, C_buf.T), dim=1)
        raw_sse_fin = torch.clamp(x_sub_sq - 2.0 * max_sc.float(), min=0.0)
        w_fin = (
            torch.clamp(torch.quantile(raw_sse_fin, robust_cap_q) / torch.clamp(raw_sse_fin, min=1e-6), max=1.0).to(X.dtype)
            if robust_cap_q is not None
            else torch.ones(sub_n, device=X.device, dtype=X.dtype)
        )
        O_fin = torch.zeros(sub_n, k, device=X.device, dtype=X.dtype).scatter_(1, assign_fin.unsqueeze(1), 1.0)
        Ow_fin = O_fin * w_fin.unsqueeze(1)
        w_tot = Ow_fin.sum(dim=0)
        C_new = (Ow_fin.T @ X_sub) / w_tot.clamp(min=1e-6).unsqueeze(1)
        C_buf = torch.where((w_tot > 1e-6).unsqueeze(1), C_new, C_buf)

    return C_buf


@torch.no_grad()
def _damped_lloyd_step_fast_low(
    X: torch.Tensor,
    C: torch.Tensor,
    damp_scale: float = 2.0,
    lam: torch.Tensor = None,
    chunk_size: int = 65536,
) -> torch.Tensor:
    n, d = X.shape
    k = C.shape[0]
    C_f32 = C.float()
    c_half_norm = 0.5 * torch.linalg.vector_norm(C_f32, dim=1).square()

    if lam is None:
        probe = X[torch.randint(0, n, (min(4096, n),), device=X.device)].float()
        max_sc = torch.max(torch.addmm(-c_half_norm, probe, C_f32.T), dim=1).values
        probe_sq = torch.linalg.vector_norm(probe, dim=1).square()
        lam = torch.clamp(damp_scale * torch.median(torch.clamp(probe_sq - 2.0 * max_sc, min=0.0)), min=1e-4)
        del probe, max_sc, probe_sq

    # Keep 65536 chunks for fast cuBLAS GEMMs when k <= 512, auto-cap only for very large k
    eff_chunk = min(chunk_size, max(4096, (32 * 1024 * 1024) // max(k, 1)))
    O_buf = torch.empty((min(eff_chunk, n), k), device=X.device, dtype=X.dtype)

    new_C = torch.zeros(k, d, device=X.device, dtype=torch.float32)
    counts = torch.zeros(k, 1, device=X.device, dtype=torch.float32)

    for i in range(0, n, eff_chunk):
        X_chunk = X[i : i + eff_chunk]
        cur = X_chunk.shape[0]
        X_f32 = X_chunk.float() if X_chunk.dtype != torch.float32 else X_chunk

        scores = torch.addmm(-c_half_norm, X_f32, C_f32.T)
        max_score, assign = torch.max(scores, dim=1)
        del scores  # free 16 MiB before writing O_buf

        # Zero (chunk_size, d) temporary allocation vs (X_f32 * X_f32).sum(dim=1)
        x_sq = torch.linalg.vector_norm(X_f32, dim=1).square()
        dist_sq = torch.clamp(x_sq - 2.0 * max_score, min=0.0)
        w = (lam / (dist_sq + lam)).to(X_chunk.dtype)

        O_cur = O_buf[:cur]
        O_cur.zero_()
        O_cur.scatter_(1, assign.unsqueeze(1), w.unsqueeze(1))

        new_C += (O_cur.T @ X_chunk).float()
        counts += O_cur.float().sum(dim=0, keepdim=True).T

    dead_mask = (counts < 1e-6)
    updated_C = new_C / torch.clamp(counts, min=1e-6)
    return torch.where(dead_mask, C_f32, updated_C).to(C.dtype)


@torch.no_grad()
def run_hybrid_v10_aware_2dlloyd_fast(X: torch.Tensor, k: int, damp_scale: float = 2.0) -> torch.Tensor:
    C = run_engine_v10_aware_fast_low(X, k)
    C = _damped_lloyd_step_fast_low(X, C, damp_scale=damp_scale)
    C = _damped_lloyd_step_fast_low(X, C, damp_scale=damp_scale)
    return C



def run_damped_lloyd_kmeans_fast(X: torch.Tensor, k: int, iters: int = 15, damp_scale: float = 2.0) -> torch.Tensor:
    n = X.shape[0]
    C = X[torch.randint(0, n, (k,), device=X.device)].clone()
    for _ in range(iters):
        C = _damped_lloyd_step_fast_low(X, C, damp_scale=damp_scale)
    return C

# =====================================================================
# 5. Diverse Task Generators
# =====================================================================

def make_task_isolated_blobs(N: int, D: int, K: int):
    """Task 1: Separated isotropic Gaussian islands + 20 extreme outliers."""
    total_n = N + 20
    X = torch.empty((total_n, D), device=device, dtype=torch.float32)
    pts_per_k = N // K
    for i in range(K):
        center = torch.randn(D, device=device) * 4.0
        torch.randn(pts_per_k, D, device=device, out=X[i * pts_per_k : (i + 1) * pts_per_k])
        X[i * pts_per_k : (i + 1) * pts_per_k].add_(center)
    if pts_per_k * K < N:
        X[pts_per_k * K : N] = X[: N - pts_per_k * K]
    torch.randn(20, D, device=device, out=X[N:]).mul_(35.0)
    core_mask = torch.zeros(total_n, dtype=torch.bool, device=device)
    core_mask[:N] = True
    return X, core_mask

def make_task_l2_manifold(N: int, D: int, K: int):
    """Task 2: Continuous L2-normalized embedding manifold with Zipfian hubs (DINO/CLIP-like)."""
    latent_rank = 64
    num_topics = 256
    # Orthonormal projection from 64D latent manifold into D=1024
    Q, _ = torch.linalg.qr(torch.randn(D, latent_rank, device=device))
    U = Q.T  # (64, D)

    topic_centers = torch.randn(num_topics, latent_rank, device=device)
    topic_centers /= topic_centers.norm(dim=1, keepdim=True)

    # Zipfian topic popularity
    probs = (torch.arange(num_topics, device=device, dtype=torch.float32) + 2.0) ** -1.1
    probs /= probs.sum()
    topic_ids = torch.multinomial(probs, N, replacement=True)

    # Overlapping manifold spread + slight ambient noise, then L2-normalized
    Z = topic_centers[topic_ids] + torch.randn(N, latent_rank, device=device) * 0.42
    X = Z @ U + torch.randn(N, D, device=device) * 0.015
    X /= torch.clamp(X.norm(dim=1, keepdim=True), min=1e-8)
    return X, None

def make_task_anisotropic_subspace(N: int, D: int, K: int):
    """Task 3: Rotated high-condition-number subspace ('pancakes & needles')."""
    Q, _ = torch.linalg.qr(torch.randn(D, D, device=device))
    eig_scales = 10.0 * (0.985 ** torch.arange(D, device=device, dtype=torch.float32)) + 0.15

    X = torch.empty((N, D), device=device, dtype=torch.float32)
    pts_per_k = N // K
    for i in range(K):
        center = (torch.randn(D, device=device) * eig_scales) @ Q
        local_scales = eig_scales * 0.25
        noise = (torch.randn(pts_per_k, D, device=device) * local_scales) @ Q
        X[i * pts_per_k : (i + 1) * pts_per_k] = noise + center
    if pts_per_k * K < N:
        X[pts_per_k * K : N] = X[: N - pts_per_k * K]

    perm = torch.randperm(N, device=device)
    return X[perm], None

def make_task_heavy_tail_trap(N: int, D: int, K: int):
    """Task 4: 96% tight core clusters + 4% diffuse noise (exceeds 99th-percentile cap)."""
    n_core = int(0.96 * N)
    n_noise = N - n_core

    X = torch.empty((N, D), device=device, dtype=torch.float32)
    pts_per_k = n_core // K
    for i in range(K):
        center = torch.randn(D, device=device) * 3.0
        torch.randn(pts_per_k, D, device=device, out=X[i * pts_per_k : (i + 1) * pts_per_k])
        X[i * pts_per_k : (i + 1) * pts_per_k].add_(center)
    if pts_per_k * K < n_core:
        X[pts_per_k * K : n_core] = X[: n_core - pts_per_k * K]

    # 4% diffuse background clutter
    torch.randn(n_noise, D, device=device, out=X[n_core:]).mul_(10.0)

    core_mask = torch.zeros(N, dtype=torch.bool, device=device)
    core_mask[:n_core] = True

    perm = torch.randperm(N, device=device)
    return X[perm], core_mask[perm]

def make_task_extreme_imbalance(N: int, D: int, K: int):
    """Task 5: 75% of points in 2 wide giants + 25% in (K-2) tight dwarfs."""
    n_giants = int(0.75 * N)
    n_dwarfs = N - n_giants
    X = torch.empty((N, D), device=device, dtype=torch.float32)

    # 2 giant wide clouds
    g_pts = n_giants // 2
    for g in range(2):
        center = torch.randn(D, device=device) * 4.0
        start = g * g_pts
        end = (g + 1) * g_pts if g == 0 else n_giants
        torch.randn(end - start, D, device=device, out=X[start:end]).mul_(3.2).add_(center)

    # (K - 2) small tight clusters
    num_dwarfs = max(1, K - 2)
    d_pts = n_dwarfs // num_dwarfs
    for i in range(num_dwarfs):
        center = torch.randn(D, device=device) * 4.0
        start = n_giants + i * d_pts
        end = n_giants + (i + 1) * d_pts if i < num_dwarfs - 1 else N
        torch.randn(end - start, D, device=device, out=X[start:end]).mul_(0.6).add_(center)

    perm = torch.randperm(N, device=device)
    return X[perm], None

def make_task_uniform(N: int, D: int, K: int):
    """Task 6: Uniform hypercube [-1, 1]^D (diffuse, zero density valleys)."""
    X = torch.empty((N, D), device=device, dtype=torch.float32).uniform_(-1.0, 1.0)
    return X, None

# =====================================================================
# 6. Multi-Task Benchmark Runner
# =====================================================================

def evaluate_mse(X: torch.Tensor, centroids: torch.Tensor, core_mask: torch.Tensor = None):
    chunk_size = 32768
    total_sq = 0.0
    core_sq = 0.0
    n = X.shape[0]
    for i in range(0, n, chunk_size):
        chunk = X[i:i + chunk_size]
        dists = torch.cdist(chunk, centroids, p=2)
        min_sq = torch.min(dists * dists, dim=1).values
        total_sq += min_sq.sum().item()
        if core_mask is not None:
            core_sq += min_sq[core_mask[i:i + chunk_size]].sum().item()
    total_mse = total_sq / n
    core_mse = (core_sq / core_mask.sum().item()) if core_mask is not None else None
    return total_mse, core_mse

def run_benchmark():
    tasks = [
        {"title": "Task 1a: Isolated Gaussian Blobs + 20 Outliers", "fn": make_task_isolated_blobs, "N": 640000, "D": 1024, "K": 16},
        {"title": "Task 1b: Isolated Gaussian Blobs + 20 Outliers", "fn": make_task_isolated_blobs, "N": 640000, "D": 1024, "K": 64},
        {"title": "Task 2: L2-Normalized Manifold (Zipfian Hubs)",   "fn": make_task_l2_manifold,      "N": 640000, "D": 1024, "K": 64},
        {"title": "Task 3: Rotated Anisotropic Subspace",            "fn": make_task_anisotropic_subspace, "N": 640000, "D": 1024, "K": 64},
        {"title": "Task 4: 96% Tight Core + 4% Heavy-Tail Clutter",  "fn": make_task_heavy_tail_trap,  "N": 640000, "D": 1024, "K": 64},
        {"title": "Task 5: Extreme Imbalance (2 Giants + 62 Dwarfs)","fn": make_task_extreme_imbalance,"N": 640000, "D": 1024, "K": 64},
    ]

    print(f"Hardware: {device.type.upper()} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    algos = [
        ("1. Classic Lloyd k-means (15 iters)",         lambda x, k: run_lloyd_kmeans(x, k, iters=15)),
        #("2. Annoy Divisive (Power-Iter + Midpoint)", lambda x, k: run_divisive_tree(x, k, split_rule="midpoint")),
        #("3. RP-Tree Divisive (Power-Iter + Median)", lambda x, k: run_divisive_tree(x, k, split_rule="median")),
        ##("4. Engine v3 (Sequential Island Slicing)",  lambda x, k: run_divisive_tree(x, k, split_rule="dense_island")),
        ##("5. HYBRID v3 (Engine v3 -> e-means 1.0ep)", lambda x, k: run_hybrid_v3(x, k)),
        #("10. og e-means (1.0 epochs)",               lambda x, k: run_original_emeans(x, k, epochs=1.0)),#unstable over more than 1 epoch
        #("11. og e-means (2.0 epochs)",               lambda x, k: run_original_emeans(x, k, epochs=2.0)),
        #("12. og e-means (3.0 epochs)",               lambda x, k: run_original_emeans(x, k, epochs=3.0)),
        #("13. og e-means (5.0 epochs)",               lambda x, k: run_original_emeans(x, k, epochs=5.0)),
        #("6. mod e-means (1.0 epochs)",               lambda x, k: run_modified_emeans(x, k, epochs=1.0)),#more stable
        #("7. mod e-means (2.0 epochs)",               lambda x, k: run_modified_emeans(x, k, epochs=2.0)),#more stable
        #("8. mod e-means (3.0 epochs)",               lambda x, k: run_modified_emeans(x, k, epochs=3.0)),#more stable
        #("9. mod e-means (5.0 epochs)",               lambda x, k: run_modified_emeans(x, k, epochs=5.0)),#more stable
        #("14. Engine v5 (Residual Power Seeding)",      lambda x, k: run_engine_v5_residual(x, k)), #best fast
        #("15. HYBRID v5 (Engine v5 -> e-means 1.0ep)",  lambda x, k: run_hybrid_v5(x, k)),          #fast and most robust, half the mse in task3 vs anything else.
        #("16. Engine v6 (Sync-Free SSE + Island)",      lambda x, k: run_engine_v6_sse(x, k)),
        #("17. HYBRID v6 (v6 -> e-means 1.0ep)",         lambda x, k: run_hybrid_v6(x, k, epochs=1.0)),
        #("18. HYBRID v6-rob (q95 SSE -> e-means 1ep)",lambda x, k: run_hybrid_v6(x, k, epochs=1.0, robust_cap_q=0.95)),
        #("19. Engine v7 (Batched log2 PDDP)",         lambda x, k: run_engine_v7_batched_pddp(x, k)),
        #("20. HYBRID v7 (v7 -> e-means 1.0ep)",       lambda x, k: run_hybrid_v7(x, k, epochs=1.0)),
        #("21. HYBRID v7-fast (v7 -> e-means 0.5ep)",  lambda x, k: run_hybrid_v7(x, k, epochs=0.5)),
        #("22. Engine v9 (Batched Priority + Island)", lambda x, k: run_engine_v9_priority_island(x, k)),
        #("23. HYBRID v9-warm (v9 -> warm 0.5ep)",     lambda x, k: run_hybrid_v9_warm(x, k, epochs=0.5)),
        #("24. HYBRID v9-2lloyd (v9 -> 2x Lloyd)",     lambda x, k: run_hybrid_v9_2lloyd(x, k)),
        #("24. HYBRID abl islands",                      lambda x, k: run_engine_v9_ablation(x, k,use_island=True,verbose=True)),
        #("24. HYBRID abl no islands",                   lambda x, k: run_engine_v9_ablation(x, k,verbose=True)),
        #("25. Engine v10 (No-Island Priority PDDP)",  lambda x, k: run_engine_v10_fast(x, k, use_guarded_island=False)),
        #("26. HYBRID v10-1lloyd (No-Island -> 1xL)",  lambda x, k: run_hybrid_v10_1lloyd(x, k, use_guarded_island=False)),
        #("27. HYBRID v10-warm (No-Island -> 0.5ep)",  lambda x, k: run_hybrid_v10_warm(x, k, epochs=0.5, use_guarded_island=False)),
        #("28. HYBRID v10-guard (10%-Guard -> 0.5ep)", lambda x, k: run_hybrid_v10_warm(x, k, epochs=0.5, use_guarded_island=True)),
        #("29. HYBRID v11 (v10 -> warm+rev0.15 0.5ep)", lambda x, k: run_hybrid_v11(x, k, epochs=0.5, revive_thresh=0.15)),
        #("30. HYBRID v11-fast (v10 -> warm+rev 0.35ep)",lambda x, k: run_hybrid_v11(x, k, epochs=0.35, revive_thresh=0.15)),
        #("31. HYBRID v12-1/8 (PDDP+q95 D2 -> 0.5ep)",   lambda x, k: run_hybrid_v12(x, k, tail_frac=0.125, epochs=0.5)),
        #("32. HYBRID v12-1/4 (PDDP+q95 D2 -> 0.5ep)",   lambda x, k: run_hybrid_v12(x, k, tail_frac=0.25, epochs=0.5)),
        #("33. HYBRID v13",                               lambda x, k: run_hybrid_v13_damped(x, k, epochs=0.5)),
        #("34. HYBRID v14",                               lambda x, k: run_hybrid_v14_multiscale(x, k, epochs=0.5)),
        ("35. Damped Lloyd k-means (15 iters)",          lambda x, k: run_damped_lloyd_kmeans_fast(x, k, iters=15)),
        #("36. HYBRID v15 (v10 -> 2x Damped Lloyd)",      lambda x, k: run_hybrid_v15_damped_2lloyd(x, k)),
        #("36. HYBRID v13 (v10 -> 1x Damped Lloyd)",      lambda x, k: run_hybrid_v13_damped_lloyd(x, k)),
        ("25. Engine v10 (No-Island)",                  lambda x, k: run_engine_v10_fast(x, k, use_guarded_island=False)),
        #("25f. Engine v10-float-damped (Raw Otsu)",     lambda x, k: run_engine_v10_float_damped(x, k, split_mode="otsu")),
        #("25g. Engine v10-float-damped (Valley)",       lambda x, k: run_engine_v10_float_damped(x, k, split_mode="valley")),
        #("36a. HYBRID v15 (v10 -> 2x Damped Lloyd)",    lambda x, k: run_hybrid_v15_damped_2lloyd(x, k)),
        #("36b. HYBRID float-otsu -> 2x Damped Lloyd",   lambda x, k: run_hybrid_v10_float_damped(x, k, split_mode="otsu", lloyd_steps=2)),
        #("36c. HYBRID float-valley -> 2x Damped Lloyd", lambda x, k: run_hybrid_v10_float_damped(x, k, split_mode="valley", lloyd_steps=2)),
        ("37. v10 kde",                    lambda x, k: run_engine_v10_kde(x, k)),
        ("38. kde sse-less",                    lambda x, k: run_engine_v10_kde(x, k, use_sse_gate=False)),
        #("40. HYBRID v10-aware -> 2x Damped Lloyd",          lambda x, k: _damped_lloyd_step_cublas(x, _damped_lloyd_step_cublas(x, run_engine_v10_aware(x, k), 2.0), 2.0)),
        ("41. v10-aware ",          lambda x, k: run_engine_v10_aware(x, k)),
        ("42. HYBRID v10-aware -> 2x Damped Lloyd", lambda x, k: run_hybrid_v10_aware_2dlloyd(x, k)),
        ("43. HYBRID v10-aware_fast -> 2x Damped Lloyd_fast", lambda x, k: run_hybrid_v10_aware_2dlloyd_fast(x, k)),
        ("44. v10-aware_fast", lambda x, k: run_engine_v10_aware_fast_low(x, k)),
    ]

    compiled_warmup = False

    for cfg in tasks:
        N, D, K = cfg["N"], cfg["D"], cfg["K"]
        torch.manual_seed(42)
        X, core_mask = cfg["fn"](N, D, K)
        has_core = core_mask is not None

        if not compiled_warmup:
            _ = power_iteration_step_compiled(
                X[:4096],
                torch.randn(D, device=device, dtype=X.dtype),
                X[:4096].mean(dim=0),
            )
            compiled_warmup = True

        print(f"\n=== {cfg['title']} (N={X.shape[0]:,} | D={D} | k={K}) ===")
        if has_core:
            print(f"{'Algorithm':<44} | {'Time (ms)':<10} | {'Total MSE':<13} | {'Core MSE':<13} | {'Speedup'}")
            print("-" * 100)
        else:
            print(f"{'Algorithm':<44} | {'Time (ms)':<10} | {'MSE (Distortion)':<16} | {'Speedup'}")
            print("-" * 84)

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

            total_mse, core_mse = evaluate_mse(X, centroids, core_mask)
            speedup_str = f"{lloyd_time / elapsed_ms:.1f}x" if (lloyd_time and elapsed_ms > 0) else "1.0x"

            if has_core:
                print(f"{name:<44} | {elapsed_ms:<10.2f} | {total_mse:<13.4f} | {core_mse:<13.4f} | {speedup_str}")
            else:
                print(f"{name:<44} | {elapsed_ms:<10.2f} | {total_mse:<16.4f} | {speedup_str}")

        del X, core_mask
        if device.type == "cuda":
            torch.cuda.empty_cache()




@torch.no_grad()
def evaluate_full_with_counts(X: torch.Tensor, centroids: torch.Tensor, core_mask: torch.Tensor = None):
    chunk_size = 32768
    n = X.shape[0]
    k = centroids.shape[0]
    total_sq = 0.0
    core_sq = 0.0
    counts = torch.zeros(k, device=X.device, dtype=torch.long)

    for i in range(0, n, chunk_size):
        chunk = X[i : i + chunk_size]
        dists = torch.cdist(chunk, centroids, p=2)
        min_d, assign = torch.min(dists, dim=1)
        min_sq = min_d * min_d
        total_sq += min_sq.sum().item()
        counts += torch.bincount(assign, minlength=k)
        if core_mask is not None:
            core_sq += min_sq[core_mask[i : i + chunk_size]].sum().item()

    total_mse = total_sq / n
    core_mse = (core_sq / core_mask.sum().item()) if core_mask is not None else None
    return total_mse, core_mse, counts.cpu().numpy()


@torch.no_grad()
def prepare_task_pca_view(X: torch.Tensor, core_mask: torch.Tensor = None, n_vis: int = 8000):
    """Builds a shared 2D PCA projection & fixed point subsample for one task."""
    n = X.shape[0]
    gen = torch.Generator(device=X.device).manual_seed(123)

    # Fit PCA on core points (when present) so extreme outliers don't collapse the axes
    if core_mask is not None:
        core_idx = torch.where(core_mask)[0]
        pca_sel = core_idx[torch.randint(0, core_idx.numel(), (min(16384, core_idx.numel()),), generator=gen, device=X.device)]
        out_idx = torch.where(~core_mask)[0]
        n_out_vis = min(200, out_idx.numel())
        n_core_vis = min(n_vis - n_out_vis, core_idx.numel())
        vis_idx = torch.cat([
            core_idx[torch.randint(0, core_idx.numel(), (n_core_vis,), generator=gen, device=X.device)],
            out_idx[torch.randint(0, out_idx.numel(), (n_out_vis,), generator=gen, device=X.device)] if n_out_vis > 0 else out_idx,
        ])
    else:
        pca_sel = torch.randint(0, n, (min(16384, n),), generator=gen, device=X.device)
        vis_idx = torch.randint(0, n, (min(n_vis, n),), generator=gen, device=X.device)

    X_pca = X[pca_sel]
    mean_x = X_pca.mean(dim=0, keepdim=True)
    _, _, V = torch.pca_lowrank(X_pca - mean_x, q=2, center=False)  # V: (D, 2)

    X_vis = X[vis_idx]
    X_vis_2d = ((X_vis - mean_x) @ V).cpu().numpy()
    vis_is_core = core_mask[vis_idx].cpu().numpy() if core_mask is not None else np.ones(X_vis.shape[0], dtype=bool)
    return X_vis, X_vis_2d, vis_is_core, mean_x, V


def _plot_cluster_panel(ax, X_vis_2d, vis_assign, C_2d, counts, xlim, ylim, title, subtitle):
    k = C_2d.shape[0]
    cmap = plt.get_cmap("turbo")
    # Prime-stride permutation separates adjacent cluster IDs visually
    color_ids = (vis_assign * 29) % max(1, k)
    pt_colors = cmap(color_ids / max(1, k - 1))

    ax.scatter(
        X_vis_2d[:, 0],
        X_vis_2d[:, 1],
        c=pt_colors,
        s=4,
        alpha=0.35,
        linewidths=0,
        rasterized=True,
    )

    in_bounds = (
        (C_2d[:, 0] >= xlim[0]) & (C_2d[:, 0] <= xlim[1]) &
        (C_2d[:, 1] >= ylim[0]) & (C_2d[:, 1] <= ylim[1])
    )
    active = counts > 0

    if np.any(in_bounds & active):
        c_ids = (np.arange(k) * 29) % max(1, k)
        c_colors = cmap(c_ids / max(1, k - 1))
        sizes = 35 + 75 * np.sqrt(counts / max(1, counts.max()))
        ax.scatter(
            C_2d[in_bounds & active, 0],
            C_2d[in_bounds & active, 1],
            c=c_colors[in_bounds & active],
            s=sizes[in_bounds & active],
            marker="X",
            edgecolors="black",
            linewidths=0.7,
            zorder=5,
        )

    if np.any(in_bounds & (~active)):
        ax.scatter(
            C_2d[in_bounds & (~active), 0],
            C_2d[in_bounds & (~active), 1],
            c="none",
            s=55,
            marker="o",
            edgecolors="red",
            linewidths=1.2,
            zorder=6,
            label="Dead (0 pts)",
        )

    out_cnt = int((~in_bounds).sum())
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=4)

    badge = subtitle + (f"\n[{out_cnt} centroids off-frame]" if out_cnt > 0 else "")
    ax.text(
        0.02, 0.98, badge,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.82, edgecolor="#cccccc"),
        zorder=10,
    )


def _plot_balance_panel(ax, results_for_task, n_total, k):
    algo_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    x_ranks = np.arange(1, k + 1)

    for idx, res in enumerate(results_for_task):
        sorted_cnts = np.sort(res["counts"])[::-1]
        ax.plot(
            x_ranks,
            np.maximum(sorted_cnts, 0.5),
            label=f"{res['short_name']} (MSE={res['total_mse']:.2f}, {res['ms']:.0f}ms)",
            color=algo_colors[idx % len(algo_colors)],
            linewidth=1.6,
        )

    ax.axhline(n_total / float(k), color="black", linestyle="--", linewidth=1.0, alpha=0.7, label=f"Balanced ({n_total // k:,})")
    ax.set_yscale("log")
    ax.set_xlabel("Centroid Rank (sorted by size)", fontsize=8)
    ax.set_ylabel("Assigned Points (log)", fontsize=8)
    ax.set_title("Cluster Population & MSE", fontsize=9.5, fontweight="bold", pad=4)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=6.8, loc="lower left", framealpha=0.85)


def run_visualizer(out_dir: str = "viz_outputs", show: bool = False):
    os.makedirs(out_dir, exist_ok=True)

    tasks = [
        {"slug": "task1a_blobs_k16",   "title": "Task 1a: Isolated Blobs + 20 Outliers",          "fn": make_task_isolated_blobs,      "N": 640000, "D": 1024, "K": 16},
        {"slug": "task1b_blobs_k64",   "title": "Task 1b: Isolated Blobs + 20 Outliers",          "fn": make_task_isolated_blobs,      "N": 640000, "D": 1024, "K": 64},
        {"slug": "task2_l2_manifold",  "title": "Task 2: L2-Normalized Manifold (Zipfian Hubs)",  "fn": make_task_l2_manifold,         "N": 640000, "D": 1024, "K": 64},
        {"slug": "task3_anisotropic",  "title": "Task 3: Rotated Anisotropic Subspace",           "fn": make_task_anisotropic_subspace,"N": 640000, "D": 1024, "K": 64},
        {"slug": "task4_heavy_tail",   "title": "Task 4: 96% Tight Core + 4% Heavy-Tail Clutter", "fn": make_task_heavy_tail_trap,     "N": 640000, "D": 1024, "K": 64},
        {"slug": "task5_imbalance",    "title": "Task 5: Extreme Imbalance (2 Giants + 62 Dwarfs)","fn": make_task_extreme_imbalance,  "N": 640000, "D": 1024, "K": 64},
        {"slug": "task6_uniform",      "title": "Task 6: Uniform Hypercube [-1, 1]^D",           "fn": make_task_uniform,             "N": 640000, "D": 1024, "K": 64},
    ]

    algos = [
        ("25. Engine v10",                "v10 (No-Island)",   lambda x, k: run_engine_v10_fast(x, k, use_guarded_island=False)),
        ("37. v10 kde",                   "v10 KDE",           lambda x, k: run_engine_v10_kde(x, k)),
        ("38. kde sse-less",              "KDE SSE-less",      lambda x, k: run_engine_v10_kde(x, k, use_sse_gate=False)),
        ("40. HYBRID v10-aware + 2xDL",   "v10-aware + 2xDL",  lambda x, k: _damped_lloyd_step_cublas(x, _damped_lloyd_step_cublas(x, run_engine_v10_aware(x, k), 2.0), 2.0)),
        ("41. v10-aware",                 "v10-aware",         lambda x, k: run_engine_v10_aware(x, k)),
    ]

    n_tasks = len(tasks)
    n_algos = len(algos)
    fig_all, axes_all = plt.subplots(n_tasks, n_algos + 1, figsize=(23, 3.2 * n_tasks), dpi=150)

    compiled_warmup = False

    for t_idx, cfg in enumerate(tasks):
        N, D, K = cfg["N"], cfg["D"], cfg["K"]
        torch.manual_seed(42)
        X, core_mask = cfg["fn"](N, D, K)

        if not compiled_warmup:
            _ = power_iteration_step_compiled(
                X[:4096],
                torch.randn(D, device=device, dtype=X.dtype),
                X[:4096].mean(dim=0),
            )
            compiled_warmup = True

        X_vis, X_vis_2d, vis_is_core, mean_x, V = prepare_task_pca_view(X, core_mask, n_vis=8000)

        # Robust axis limits based on core/inlier 2D points
        core_pts_2d = X_vis_2d[vis_is_core] if np.any(vis_is_core) else X_vis_2d
        lo = np.percentile(core_pts_2d, 0.3, axis=0)
        hi = np.percentile(core_pts_2d, 99.7, axis=0)
        pad = 0.14 * np.maximum(hi - lo, 1e-4)
        xlim = (lo[0] - pad[0], hi[0] + pad[0])
        ylim = (lo[1] - pad[1], hi[1] + pad[1])

        fig_t, axes_t = plt.subplots(2, 3, figsize=(15.5, 9.5), dpi=160)
        axes_t_flat = axes_t.ravel()
        task_results = []

        print(f"\n=== {cfg['title']} (N={X.shape[0]:,} | D={D} | k={K}) ===")

        for a_idx, (full_name, short_name, fn) in enumerate(algos):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            centroids = fn(X, K)
            if device.type == "cuda":
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000.0

            total_mse, core_mse, counts = evaluate_full_with_counts(X, centroids, core_mask)
            vis_assign = torch.argmin(torch.cdist(X_vis, centroids, p=2), dim=1).cpu().numpy()
            C_2d = ((centroids - mean_x) @ V).cpu().numpy()

            dead_cnt = int((counts == 0).sum())
            min_nz = int(counts[counts > 0].min()) if np.any(counts > 0) else 0
            max_cnt = int(counts.max())

            core_line = f"\nCore MSE: {core_mse:.2f}" if core_mse is not None else ""
            subtitle = (
                f"{ms:.1f} ms | MSE: {total_mse:.3f}{core_line}\n"
                f"Dead: {dead_cnt}/{K} | Sz: [{min_nz:,}, {max_cnt:,}]"
            )

            print(f"  {full_name:<28} | {ms:7.2f} ms | MSE={total_mse:10.4f} | Dead={dead_cnt}")

            _plot_cluster_panel(axes_t_flat[a_idx], X_vis_2d, vis_assign, C_2d, counts, xlim, ylim, full_name, subtitle)
            _plot_cluster_panel(axes_all[t_idx, a_idx], X_vis_2d, vis_assign, C_2d, counts, xlim, ylim,
                                full_name if t_idx == 0 else "", subtitle)

            task_results.append({
                "short_name": short_name,
                "ms": ms,
                "total_mse": total_mse,
                "core_mse": core_mse,
                "counts": counts,
            })

        _plot_balance_panel(axes_t_flat[5], task_results, X.shape[0], K)
        _plot_balance_panel(axes_all[t_idx, n_algos], task_results, X.shape[0], K)

        axes_all[t_idx, 0].set_ylabel(
            f"{cfg['title']}\n(N={X.shape[0]:,}, k={K})",
            fontsize=8.5, fontweight="bold", labelpad=8,
        )

        fig_t.suptitle(f"{cfg['title']}  (N={X.shape[0]:,}, D={D}, k={K}) — 2D PCA Projection", fontsize=12.5, fontweight="bold")
        fig_t.tight_layout(rect=[0, 0, 1, 0.96])
        task_path = os.path.join(out_dir, f"viz_{cfg['slug']}.png")
        fig_t.savefig(task_path)
        if not show:
            plt.close(fig_t)

        del X, core_mask, X_vis
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fig_all.suptitle("Multi-Task Clustering Comparison (2D PCA Projections + Cluster Population Balance)", fontsize=14, fontweight="bold")
    fig_all.tight_layout(rect=[0, 0, 1, 0.985])
    overview_path = os.path.join(out_dir, "viz_overview_grid.png")
    fig_all.savefig(overview_path)
    print(f"\nSaved overview grid -> {overview_path}")
    print(f"Saved per-task figures -> {out_dir}/viz_task*.png")

    if show:
        plt.show()
    else:
        plt.close(fig_all)

import gc
import argparse


def _fmt_mib(num_bytes: int) -> str:
    return f"{num_bytes / (1024.0 ** 2):8.1f} MiB"



def _summarize_peak_snapshot(snap: dict, dev_idx: int = 0, top_k: int = 8):
    """Replays PyTorch's CUDA memory trace to show what was live at peak allocation."""
    traces = snap.get("device_traces", [])
    if dev_idx >= len(traces) or not traces[dev_idx]:
        print("  [trace] No device trace events captured.")
        return

    live = {}          # addr -> (size, frames)
    cur_bytes = 0
    peak_bytes = 0
    peak_live = {}

    for ev in traces[dev_idx]:
        act = ev.get("action")
        addr = ev.get("addr")
        if act == "alloc":
            sz = ev.get("size", 0)
            live[addr] = (sz, ev.get("frames", []))
            cur_bytes += sz
            if cur_bytes > peak_bytes:
                peak_bytes = cur_bytes
                peak_live = dict(live)
        elif act in ("free_requested", "free_completed"):
            if addr in live:
                cur_bytes -= live.pop(addr)[0]

    by_site = {}
    for sz, frames in peak_live.values():
        site = "<unknown>"
        for fr in frames:
            fn = fr.get("filename", "")
            if "torch" not in fn or "site-packages" not in fn:
                fname = os.path.basename(fn)
                site = f"{fname}:{fr.get('line')} ({fr.get('name')})"
                break
        if site == "<unknown>" and frames:
            fr = frames[0]
            fname = os.path.basename(fr.get("filename", ""))
            site = f"{fname}:{fr.get('line')} ({fr.get('name')})"
        by_site[site] = by_site.get(site, 0) + sz

    print(f"  [Peak Allocation Breakdown | Peak Extra ~ {peak_bytes / (1024.0 ** 2):.1f} MiB]")
    for site, sz in sorted(by_site.items(), key=lambda kv: kv[1], reverse=True)[:top_k]:
        print(f"    {sz / (1024.0 ** 2):8.2f} MiB  <-  {site}")

def run_vram_profile(trace_peak_allocs: bool = True, trace_first_task_only: bool = True):
    if device.type != "cuda":
        print("VRAM profiler requires CUDA.")
        return

    tasks = [
        {"title": "Task 1a: Isolated Blobs (k=16)",      "fn": make_task_isolated_blobs,       "N": 640000, "D": 1024, "K": 16},
        {"title": "Task 1b: Isolated Blobs (k=64)",      "fn": make_task_isolated_blobs,       "N": 640000, "D": 1024, "K": 64},
        {"title": "Task 2: L2 Manifold",                 "fn": make_task_l2_manifold,          "N": 640000, "D": 1024, "K": 64},
        {"title": "Task 3: Rotated Anisotropic",         "fn": make_task_anisotropic_subspace, "N": 640000, "D": 1024, "K": 64},
        {"title": "Task 4: Heavy-Tail Clutter",          "fn": make_task_heavy_tail_trap,      "N": 640000, "D": 1024, "K": 64},
        {"title": "Task 5: Extreme Imbalance",           "fn": make_task_extreme_imbalance,    "N": 640000, "D": 1024, "K": 64},
        # {"title": "Task 6: Uniform [-1, 1]^D",         "fn": make_task_uniform,              "N": 640000, "D": 1024, "K": 64},
    ]

    algos = [
        ("1. Classic Lloyd k-means (15 iters)",         lambda x, k: run_lloyd_kmeans(x, k, iters=15)),
        ("35. Damped Lloyd k-means (15 iters)",          lambda x, k: run_damped_lloyd_kmeans_fast(x, k, iters=15)),
        ("25. Engine v10 (No-Island)",              lambda x, k: run_engine_v10_fast(x, k, use_guarded_island=False)),
        ("37. v10 kde",                             lambda x, k: run_engine_v10_kde(x, k)),
        ("38. kde sse-less",                        lambda x, k: run_engine_v10_kde(x, k, use_sse_gate=False)),
        ("41. v10-aware",                           lambda x, k: run_engine_v10_aware(x, k)),
        ("42. HYBRID v10-aware -> 2x Damped Lloyd", lambda x, k: _damped_lloyd_step_cublas(x, _damped_lloyd_step_cublas(x, run_engine_v10_aware(x, k), 2.0), 2.0)),
        ("43. v10-aware_low",                lambda x, k:  run_engine_v10_aware_low_vram(x, k)),
        ("43. HYBRID v10-aware_low -> 2x Damped Lloyd_low", lambda x, k:  run_hybrid_v10_aware_2dlloyd_low(x, k)),
        ("43. HYBRID v10-aware_fast -> 2x Damped Lloyd_fast", lambda x, k:  run_hybrid_v10_aware_2dlloyd_fast(x, k)),
    ]

    dev_idx = torch.cuda.current_device()
    _, total_vram = torch.cuda.mem_get_info(dev_idx)
    print(f"GPU: {torch.cuda.get_device_name(dev_idx)} | Total VRAM: {total_vram / (1024.0 ** 3):.2f} GiB")

    compiled_warmup = False

    for t_idx, cfg in enumerate(tasks):
        N, D, K = cfg["N"], cfg["D"], cfg["K"]
        gc.collect()
        torch.cuda.empty_cache()

        torch.manual_seed(42)
        X, core_mask = cfg["fn"](N, D, K)
        del core_mask

        if not compiled_warmup:
            _ = power_iteration_step_compiled(
                X[:4096],
                torch.randn(D, device=device, dtype=X.dtype),
                X[:4096].mean(dim=0),
            )
            compiled_warmup = True

        x_bytes = X.numel() * X.element_size()
        print(f"\n=== {cfg['title']} (N={X.shape[0]:,} | D={D} | k={K} | X={x_bytes / (1024.0 ** 2):.1f} MiB) ===")
        print(
            f"{'Algorithm':<42} | {'Peak Extra':<11} | {'Peak Total':<11} | "
            f"{'Peak Rsvd':<11} | {'Retained':<10} | {'Drv Free':<11} | {'Retries'}"
        )
        print("-" * 122)

        for name, fn in algos:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(dev_idx)

            base_alloc = torch.cuda.memory_allocated(dev_idx)
            do_trace = trace_peak_allocs and (not trace_first_task_only or t_idx == 0)

            if do_trace:
                try:
                    torch.cuda.memory._record_memory_history(max_entries=50000)
                except Exception:
                    do_trace = False

            centroids = fn(X, K)
            torch.cuda.synchronize()

            snap = None
            if do_trace:
                try:
                    snap = torch.cuda.memory._snapshot()
                    torch.cuda.memory._record_memory_history(enabled=None)
                except Exception:
                    snap = None

            peak_alloc = torch.cuda.max_memory_allocated(dev_idx)
            peak_rsvd = torch.cuda.max_memory_reserved(dev_idx)
            after_alloc = torch.cuda.memory_allocated(dev_idx)
            drv_free, _ = torch.cuda.mem_get_info(dev_idx)
            retries = torch.cuda.memory_stats(dev_idx).get("num_alloc_retries", 0)

            peak_extra = max(0, peak_alloc - base_alloc)
            retained = after_alloc - base_alloc

            print(
                f"{name:<42} | {_fmt_mib(peak_extra):<11} | {_fmt_mib(peak_alloc):<11} | "
                f"{_fmt_mib(peak_rsvd):<11} | {_fmt_mib(retained):<10} | {_fmt_mib(drv_free):<11} | {retries}"
            )

            if snap is not None:
                _summarize_peak_snapshot(snap, dev_idx=dev_idx, top_k=6)

            del centroids

        del X
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    parser = argparse.ArgumentParser()
    parser.add_argument("--vram", action="store_true", help="Run separate CUDA VRAM profiler instead of speed benchmark")
    parser.add_argument("--visual", action="store_true", help="Run visualizer")
    parser.add_argument("--trace-all", action="store_true", help="Trace peak call stacks on every task (not just Task 1a)")
    args, _ = parser.parse_known_args()

    if args.vram:
        run_vram_profile(trace_peak_allocs=True, trace_first_task_only=not args.trace_all)
    elif args.visual:
        run_visualizer()
    else:
        run_benchmark()
#results

#Hardware: CUDA (NVIDIA GeForce RTX 3060)
#
#=== Task 1a: Isolated Gaussian Blobs + 20 Outliers (N=640,020 | D=1024 | k=16) ===
#Algorithm                                    | Time (ms)  | Total MSE     | Core MSE      | Speedup
#----------------------------------------------------------------------------------------------------
#1. Classic Lloyd k-means (15 iters)          | 378.13     | 3041.6110     | 3001.9409     | 1.0x
#2. Annoy Divisive (Power-Iter + Midpoint)    | 186.84     | 1106.6349     | 1066.8946     | 2.0x
#3. RP-Tree Divisive (Power-Iter + Median)    | 178.47     | 1097.6849     | 1057.9424     | 2.1x
#4. Engine v3 (Sequential Island Slicing)     | 194.74     | 1099.2204     | 1059.4727     | 1.9x
#5. HYBRID v3 (Engine v3 -> e-means 1.0ep)    | 272.34     | 1063.0791     | 1023.3272     | 1.4x
#10. og e-means (1.0 epochs)                  | 120.63     | 3040.0420     | 3000.3691     | 3.1x
#11. og e-means (2.0 epochs)                  | 116.66     | 2040.8099     | 2001.1078     | 3.2x
#12. og e-means (3.0 epochs)                  | 165.39     | 1063.0791     | 1023.3272     | 2.3x
#13. og e-means (5.0 epochs)                  | 218.58     | 2975.1917     | 2935.5485     | 1.7x
#6. mod e-means (1.0 epochs)                  | 134.01     | 1063.0791     | 1023.3272     | 2.8x
#7. mod e-means (2.0 epochs)                  | 113.68     | 1063.0791     | 1023.3272     | 3.3x
#8. mod e-means (3.0 epochs)                  | 163.08     | 1063.0791     | 1023.3272     | 2.3x
#9. mod e-means (5.0 epochs)                  | 241.17     | 2988.6904     | 2949.0220     | 1.6x
#14. Engine v5 (Residual Power Seeding)       | 86.61      | 1066.3125     | 1026.5592     | 4.4x
#15. HYBRID v5 (Engine v5 -> e-means 1.0ep)   | 160.80     | 1063.0791     | 1023.3272     | 2.4x
#16. Engine v6 (Sync-Free SSE + Island)       | 80.73      | 1063.5214     | 1023.7722     | 4.7x
#17. HYBRID v6 (v6 -> e-means 1.0ep)          | 128.48     | 1063.0791     | 1023.3272     | 2.9x
#18. HYBRID v6-rob (q95 SSE -> e-means 1ep)   | 138.63     | 1063.0791     | 1023.3272     | 2.7x
#19. Engine v7 (Batched log2 PDDP)            | 11.00      | 3143.7210     | 3104.0673     | 34.4x
#20. HYBRID v7 (v7 -> e-means 1.0ep)          | 91.57      | 1063.0791     | 1023.3272     | 4.1x
#21. HYBRID v7-fast (v7 -> e-means 0.5ep)     | 72.32      | 1063.0791     | 1023.3272     | 5.2x
#22. Engine v9 (Batched Priority + Island)    | 17.23      | 7071.8569     | 7032.3531     | 22.0x
#23. HYBRID v9-warm (v9 -> warm 0.5ep)        | 67.88      | 9121.4248     | 9083.6752     | 5.6x
#24. HYBRID v9-2lloyd (v9 -> 2x Lloyd)        | 55.49      | 6076.3200     | 6040.8466     | 1.0x
#24. HYBRID abl islands                       | 17.52      | 6135.8884     | 6096.3444     | 3.2x
#24. HYBRID abl no islands                    | 14.56      | 1065.9065     | 1026.1533     | 3.8x
#25. Engine v10 (No-Island Priority PDDP)     | 14.25      | 1124.9178     | 1085.1676     | 3.9x
#26. HYBRID v10-1lloyd (No-Island -> 1xL)     | 34.62      | 1063.1051     | 1023.3529     | 1.6x
#27. HYBRID v10-warm (No-Island -> 0.5ep)     | 65.37      | 1063.0791     | 1023.3272     | 0.8x
#28. HYBRID v10-guard (10%-Guard -> 0.5ep)    | 75.27      | 5058.8175     | 5019.2358     | 0.7x
#29. HYBRID v11 (v10 -> warm+rev0.15 0.5ep)   | 77.71      | 1063.0791     | 1023.3272     | 0.7x
#30. HYBRID v11-fast (v10 -> warm+rev 0.35ep) | 76.21      | 1063.0791     | 1023.3272     | 0.7x
#31. HYBRID v12-1/8 (PDDP+q95 D2 -> 0.5ep)    | 73.28      | 1063.0791     | 1023.3272     | 0.8x
#32. HYBRID v12-1/4 (PDDP+q95 D2 -> 0.5ep)    | 69.07      | 2048.0626     | 2010.2307     | 0.8x
#33. HYBRID v13                               | 75.69      | 1063.0791     | 1023.3272     | 0.7x
#33. HYBRID v14                               | 98.44      | 1063.0791     | 1023.3272     | 0.6x
#
#=== Task 1b: Isolated Gaussian Blobs + 20 Outliers (N=640,020 | D=1024 | k=64) ===
#Algorithm                                    | Time (ms)  | Total MSE     | Core MSE      | Speedup
#----------------------------------------------------------------------------------------------------
#1. Classic Lloyd k-means (15 iters)          | 382.30     | 3758.4617     | 3722.7886     | 1.0x
#2. Annoy Divisive (Power-Iter + Midpoint)    | 652.19     | 1652.4726     | 1612.6413     | 0.6x
#3. RP-Tree Divisive (Power-Iter + Median)    | 573.78     | 2552.7824     | 2512.9937     | 0.7x
#4. Engine v3 (Sequential Island Slicing)     | 666.86     | 1582.1265     | 1542.2817     | 0.6x
#5. HYBRID v3 (Engine v3 -> e-means 1.0ep)    | 758.41     | 1064.0215     | 1024.1551     | 0.5x
#10. og e-means (1.0 epochs)                  | 84.63      | 1800.6818     | 1760.8436     | 4.5x
#11. og e-means (2.0 epochs)                  | 136.21     | 1806.7110     | 1766.8753     | 2.8x
#12. og e-means (3.0 epochs)                  | 166.12     | 1825.6972     | 1785.8607     | 2.3x
#13. og e-means (5.0 epochs)                  | 239.09     | 2273.3734     | 2233.5403     | 1.6x
#6. mod e-means (1.0 epochs)                  | 87.72      | 1064.0215     | 1024.1551     | 4.4x
#7. mod e-means (2.0 epochs)                  | 127.82     | 1064.0215     | 1024.1551     | 3.0x
#8. mod e-means (3.0 epochs)                  | 169.88     | 1064.0215     | 1024.1551     | 2.3x
#9. mod e-means (5.0 epochs)                  | 253.77     | 1064.0215     | 1024.1551     | 1.5x
#14. Engine v5 (Residual Power Seeding)       | 257.72     | 1335.6925     | 1297.7965     | 1.5x
#15. HYBRID v5 (Engine v5 -> e-means 1.0ep)   | 328.20     | 1064.0215     | 1024.1551     | 1.2x
#16. Engine v6 (Sync-Free SSE + Island)       | 213.26     | 1067.3763     | 1027.5016     | 1.8x
#17. HYBRID v6 (v6 -> e-means 1.0ep)          | 291.14     | 1064.0215     | 1024.1551     | 1.3x
#18. HYBRID v6-rob (q95 SSE -> e-means 1ep)   | 314.24     | 1064.0271     | 1024.1607     | 1.2x
#19. Engine v7 (Batched log2 PDDP)            | 16.04      | 3872.1659     | 3832.4051     | 23.8x
#20. HYBRID v7 (v7 -> e-means 1.0ep)          | 100.99     | 1064.0271     | 1024.1607     | 3.8x
#21. HYBRID v7-fast (v7 -> e-means 0.5ep)     | 78.46      | 1064.0215     | 1024.1551     | 4.9x
#22. Engine v9 (Batched Priority + Island)    | 27.08      | 13084.8307    | 13045.2330    | 14.1x
#23. HYBRID v9-warm (v9 -> warm 0.5ep)        | 78.65      | 13609.0989    | 13581.0844    | 4.9x
#24. HYBRID v9-2lloyd (v9 -> 2x Lloyd)        | 81.26      | 14072.7642    | 14044.0030    | 1.0x
#24. HYBRID abl islands                       | 26.06      | 14332.5983    | 14293.0381    | 3.1x
#24. HYBRID abl no islands                    | 23.25      | 1077.8967     | 1038.0261     | 3.5x
#25. Engine v10 (No-Island Priority PDDP)     | 23.40      | 1086.4295     | 1046.5583     | 3.5x
#26. HYBRID v10-1lloyd (No-Island -> 1xL)     | 49.95      | 1064.0057     | 1024.1387     | 1.6x
#27. HYBRID v10-warm (No-Island -> 0.5ep)     | 75.30      | 1064.0215     | 1024.1551     | 1.1x
#28. HYBRID v10-guard (10%-Guard -> 0.5ep)    | 79.67      | 3607.6144     | 3567.8434     | 1.0x
#29. HYBRID v11 (v10 -> warm+rev0.15 0.5ep)   | 81.31      | 1064.0271     | 1024.1607     | 1.0x
#30. HYBRID v11-fast (v10 -> warm+rev 0.35ep) | 78.97      | 1064.0271     | 1024.1607     | 1.0x
#31. HYBRID v12-1/8 (PDDP+q95 D2 -> 0.5ep)    | 80.05      | 1817.5128     | 1779.4492     | 1.0x
#32. HYBRID v12-1/4 (PDDP+q95 D2 -> 0.5ep)    | 86.23      | 2560.4578     | 2522.5154     | 0.9x
#33. HYBRID v13                               | 90.27      | 1064.0016     | 1024.1351     | 0.9x
#33. HYBRID v14                               | 102.20     | 1064.0215     | 1024.1551     | 0.8x
#
#=== Task 2: L2-Normalized Manifold (Zipfian Hubs) (N=640,000 | D=1024 | k=64) ===
#Algorithm                                    | Time (ms)  | MSE (Distortion) | Speedup
#------------------------------------------------------------------------------------
#1. Classic Lloyd k-means (15 iters)          | 380.60     | 0.8976           | 1.0x
#2. Annoy Divisive (Power-Iter + Midpoint)    | 574.58     | 0.9141           | 0.7x
#3. RP-Tree Divisive (Power-Iter + Median)    | 574.77     | 0.9147           | 0.7x
#4. Engine v3 (Sequential Island Slicing)     | 655.96     | 0.9139           | 0.6x
#5. HYBRID v3 (Engine v3 -> e-means 1.0ep)    | 753.84     | 0.8956           | 0.5x
#10. og e-means (1.0 epochs)                  | 79.87      | 0.8956           | 4.8x
#11. og e-means (2.0 epochs)                  | 120.68     | 0.8948           | 3.2x
#12. og e-means (3.0 epochs)                  | 158.97     | 0.8944           | 2.4x
#13. og e-means (5.0 epochs)                  | 233.16     | 0.8942           | 1.6x
#6. mod e-means (1.0 epochs)                  | 78.97      | 0.8956           | 4.8x
#7. mod e-means (2.0 epochs)                  | 117.77     | 0.8948           | 3.2x
#8. mod e-means (3.0 epochs)                  | 159.65     | 0.8945           | 2.4x
#9. mod e-means (5.0 epochs)                  | 232.34     | 0.8941           | 1.6x
#14. Engine v5 (Residual Power Seeding)       | 233.61     | 0.9179           | 1.6x
#15. HYBRID v5 (Engine v5 -> e-means 1.0ep)   | 335.72     | 0.8956           | 1.1x
#16. Engine v6 (Sync-Free SSE + Island)       | 229.12     | 0.9198           | 1.7x
#17. HYBRID v6 (v6 -> e-means 1.0ep)          | 299.65     | 0.8953           | 1.3x
#18. HYBRID v6-rob (q95 SSE -> e-means 1ep)   | 311.88     | 0.8954           | 1.2x
#19. Engine v7 (Batched log2 PDDP)            | 16.05      | 0.9079           | 23.7x
#20. HYBRID v7 (v7 -> e-means 1.0ep)          | 93.30      | 0.8956           | 4.1x
#21. HYBRID v7-fast (v7 -> e-means 0.5ep)     | 74.72      | 0.8968           | 5.1x
#22. Engine v9 (Batched Priority + Island)    | 27.20      | 0.9184           | 14.0x
#23. HYBRID v9-warm (v9 -> warm 0.5ep)        | 78.82      | 0.8975           | 4.8x
#24. HYBRID v9-2lloyd (v9 -> 2x Lloyd)        | 77.80      | 0.9042           | 1.0x
#24. HYBRID abl islands                       | 26.52      | 0.9178           | 2.9x
#24. HYBRID abl no islands                    | 23.54      | 0.9077           | 3.3x
#25. Engine v10 (No-Island Priority PDDP)     | 23.96      | 0.9078           | 3.2x
#26. HYBRID v10-1lloyd (No-Island -> 1xL)     | 49.85      | 0.9057           | 1.6x
#27. HYBRID v10-warm (No-Island -> 0.5ep)     | 75.66      | 0.8973           | 1.0x
#28. HYBRID v10-guard (10%-Guard -> 0.5ep)    | 78.54      | 0.8974           | 1.0x
#29. HYBRID v11 (v10 -> warm+rev0.15 0.5ep)   | 82.00      | 0.8976           | 0.9x
#30. HYBRID v11-fast (v10 -> warm+rev 0.35ep) | 78.45      | 0.8975           | 1.0x
#31. HYBRID v12-1/8 (PDDP+q95 D2 -> 0.5ep)    | 80.50      | 0.9000           | 1.0x
#32. HYBRID v12-1/4 (PDDP+q95 D2 -> 0.5ep)    | 86.27      | 0.9039           | 0.9x
#33. HYBRID v13                               | 90.38      | 0.8974           | 0.9x
#33. HYBRID v14                               | 94.54      | 0.9004           | 0.8x
#
#=== Task 3: Rotated Anisotropic Subspace (N=640,000 | D=1024 | k=64) ===
#Algorithm                                    | Time (ms)  | MSE (Distortion) | Speedup
#------------------------------------------------------------------------------------
#1. Classic Lloyd k-means (15 iters)          | 381.45     | 830.6502         | 1.0x
#2. Annoy Divisive (Power-Iter + Midpoint)    | 575.71     | 452.7610         | 0.7x
#3. RP-Tree Divisive (Power-Iter + Median)    | 575.82     | 810.5841         | 0.7x
#4. Engine v3 (Sequential Island Slicing)     | 666.56     | 471.5156         | 0.6x
#5. HYBRID v3 (Engine v3 -> e-means 1.0ep)    | 761.17     | 278.3763         | 0.5x
#10. og e-means (1.0 epochs)                  | 81.29      | 808.6921         | 4.7x
#11. og e-means (2.0 epochs)                  | 120.11     | 731.8621         | 3.2x
#12. og e-means (3.0 epochs)                  | 162.04     | 831.1373         | 2.4x
#13. og e-means (5.0 epochs)                  | 233.92     | 747.8395         | 1.6x
#6. mod e-means (1.0 epochs)                  | 80.57      | 538.8471         | 4.7x
#7. mod e-means (2.0 epochs)                  | 118.95     | 520.9680         | 3.2x
#8. mod e-means (3.0 epochs)                  | 159.39     | 696.1522         | 2.4x
#9. mod e-means (5.0 epochs)                  | 233.16     | 431.9281         | 1.6x
#14. Engine v5 (Residual Power Seeding)       | 233.21     | 233.6015         | 1.6x
#15. HYBRID v5 (Engine v5 -> e-means 1.0ep)   | 311.22     | 224.0490         | 1.2x
#16. Engine v6 (Sync-Free SSE + Island)       | 214.88     | 224.7501         | 1.8x
#17. HYBRID v6 (v6 -> e-means 1.0ep)          | 295.18     | 224.0490         | 1.3x
#18. HYBRID v6-rob (q95 SSE -> e-means 1ep)   | 299.20     | 224.0490         | 1.3x
#19. Engine v7 (Batched log2 PDDP)            | 16.02      | 652.4964         | 23.8x
#20. HYBRID v7 (v7 -> e-means 1.0ep)          | 94.12      | 772.4771         | 4.1x
#21. HYBRID v7-fast (v7 -> e-means 0.5ep)     | 75.19      | 565.8873         | 5.1x
#22. Engine v9 (Batched Priority + Island)    | 27.15      | 2288.3363        | 14.1x
#23. HYBRID v9-warm (v9 -> warm 0.5ep)        | 78.59      | 2341.9356        | 4.9x
#24. HYBRID v9-2lloyd (v9 -> 2x Lloyd)        | 78.44      | 2477.5437        | 1.0x
#24. HYBRID abl islands                       | 27.81      | 2191.5099        | 2.8x
#24. HYBRID abl no islands                    | 23.57      | 265.3283         | 3.3x
#25. Engine v10 (No-Island Priority PDDP)     | 23.88      | 226.2767         | 3.3x
#26. HYBRID v10-1lloyd (No-Island -> 1xL)     | 50.15      | 224.0490         | 1.6x
#27. HYBRID v10-warm (No-Island -> 0.5ep)     | 75.71      | 224.0490         | 1.0x
#28. HYBRID v10-guard (10%-Guard -> 0.5ep)    | 78.43      | 702.1773         | 1.0x
#29. HYBRID v11 (v10 -> warm+rev0.15 0.5ep)   | 81.46      | 224.0490         | 1.0x
#30. HYBRID v11-fast (v10 -> warm+rev 0.35ep) | 79.01      | 224.0490         | 1.0x
#31. HYBRID v12-1/8 (PDDP+q95 D2 -> 0.5ep)    | 80.28      | 397.3358         | 1.0x
#32. HYBRID v12-1/4 (PDDP+q95 D2 -> 0.5ep)    | 85.83      | 524.0550         | 0.9x
#33. HYBRID v13                               | 90.50      | 224.0490         | 0.9x
#33. HYBRID v14                               | 95.07      | 224.0490         | 0.8x
#
#=== Task 4: 96% Tight Core + 4% Heavy-Tail Clutter (N=640,000 | D=1024 | k=64) ===
#Algorithm                                    | Time (ms)  | Total MSE     | Core MSE      | Speedup
#----------------------------------------------------------------------------------------------------
#1. Classic Lloyd k-means (15 iters)          | 383.55     | 7124.4656     | 3159.3019     | 1.0x
#2. Annoy Divisive (Power-Iter + Midpoint)    | 571.67     | 5567.2200     | 1479.1834     | 0.7x
#3. RP-Tree Divisive (Power-Iter + Median)    | 575.97     | 6005.8590     | 1956.4526     | 0.7x
#4. Engine v3 (Sequential Island Slicing)     | 673.09     | 5566.3277     | 1509.3250     | 0.6x
#5. HYBRID v3 (Engine v3 -> e-means 1.0ep)    | 761.80     | 5359.0160     | 1315.6774     | 0.5x
#10. og e-means (1.0 epochs)                  | 85.86      | 5475.8609     | 1437.2246     | 4.5x
#11. og e-means (2.0 epochs)                  | 124.63     | 5593.4384     | 1559.7034     | 3.1x
#12. og e-means (3.0 epochs)                  | 164.87     | 5612.9800     | 1586.1772     | 2.3x
#13. og e-means (5.0 epochs)                  | 243.14     | 5984.3224     | 1970.5811     | 1.6x
#6. mod e-means (1.0 epochs)                  | 89.94      | 5341.5785     | 1297.5102     | 4.3x
#7. mod e-means (2.0 epochs)                  | 138.13     | 5427.5293     | 1357.9196     | 2.8x
#8. mod e-means (3.0 epochs)                  | 179.20     | 5200.7462     | 1150.6325     | 2.1x
#9. mod e-means (5.0 epochs)                  | 261.03     | 5315.5823     | 1240.4467     | 1.5x
#14. Engine v5 (Residual Power Seeding)       | 274.27     | 6328.1127     | 2317.9993     | 1.4x
#15. HYBRID v5 (Engine v5 -> e-means 1.0ep)   | 338.12     | 5483.5851     | 1445.4321     | 1.1x
#16. Engine v6 (Sync-Free SSE + Island)       | 223.98     | 6634.2581     | 2636.3545     | 1.7x
#17. HYBRID v6 (v6 -> e-means 1.0ep)          | 309.79     | 5762.9465     | 1736.7685     | 1.2x
#18. HYBRID v6-rob (q95 SSE -> e-means 1ep)   | 309.41     | 5492.3508     | 1454.5676     | 1.2x
#19. Engine v7 (Batched log2 PDDP)            | 15.99      | 6574.8395     | 2488.0144     | 24.0x
#20. HYBRID v7 (v7 -> e-means 1.0ep)          | 108.48     | 5704.0355     | 1641.2879     | 3.5x
#21. HYBRID v7-fast (v7 -> e-means 0.5ep)     | 83.40      | 5697.0095     | 1630.8148     | 4.6x
#22. Engine v9 (Batched Priority + Island)    | 26.70      | 11015.9013    | 7199.4084     | 14.4x
#23. HYBRID v9-warm (v9 -> warm 0.5ep)        | 78.50      | 11981.4977    | 8206.7373     | 4.9x
#24. HYBRID v9-2lloyd (v9 -> 2x Lloyd)        | 78.45      | 11689.7293    | 7902.3886     | 1.0x
#24. HYBRID abl islands                       | 27.75      | 11573.1809    | 7780.7832     | 2.8x
#24. HYBRID abl no islands                    | 23.19      | 5302.8123     | 1105.2206     | 3.4x
#25. Engine v10 (No-Island Priority PDDP)     | 24.25      | 5298.2623     | 1107.1407     | 3.2x
#26. HYBRID v10-1lloyd (No-Island -> 1xL)     | 50.53      | 5239.8568     | 1059.9437     | 1.6x
#27. HYBRID v10-warm (No-Island -> 0.5ep)     | 75.80      | 5175.1992     | 1096.6964     | 1.0x
#28. HYBRID v10-guard (10%-Guard -> 0.5ep)    | 78.67      | 6490.8700     | 2457.2787     | 1.0x
#29. HYBRID v11 (v10 -> warm+rev0.15 0.5ep)   | 82.09      | 5182.3268     | 1102.1594     | 1.0x
#30. HYBRID v11-fast (v10 -> warm+rev 0.35ep) | 79.01      | 5175.4473     | 1096.9541     | 1.0x
#31. HYBRID v12-1/8 (PDDP+q95 D2 -> 0.5ep)    | 80.66      | 5749.6010     | 1722.3728     | 1.0x
#32. HYBRID v12-1/4 (PDDP+q95 D2 -> 0.5ep)    | 89.54      | 5878.1100     | 1856.5742     | 0.9x
#33. HYBRID v13                               | 90.78      | 5256.0061     | 1033.6168     | 0.9x
#33. HYBRID v14                               | 95.23      | 5256.3858     | 1034.4922     | 0.8x
#
#=== Task 5: Extreme Imbalance (2 Giants + 62 Dwarfs) (N=640,000 | D=1024 | k=64) ===
#Algorithm                                    | Time (ms)  | MSE (Distortion) | Speedup
#------------------------------------------------------------------------------------
#1. Classic Lloyd k-means (15 iters)          | 383.69     | 10888.7694       | 1.0x
#2. Annoy Divisive (Power-Iter + Midpoint)    | 572.69     | 11017.1746       | 0.7x
#3. RP-Tree Divisive (Power-Iter + Median)    | 572.61     | 10991.9225       | 0.7x
#4. Engine v3 (Sequential Island Slicing)     | 660.34     | 10740.6822       | 0.6x
#5. HYBRID v3 (Engine v3 -> e-means 1.0ep)    | 746.09     | 9045.8929        | 0.5x
#10. og e-means (1.0 epochs)                  | 85.48      | 9799.0886        | 4.5x
#11. og e-means (2.0 epochs)                  | 123.73     | 10250.8893       | 3.1x
#12. og e-means (3.0 epochs)                  | 162.98     | 10567.1745       | 2.4x
#13. og e-means (5.0 epochs)                  | 236.38     | 10394.0012       | 1.6x
#6. mod e-means (1.0 epochs)                  | 102.49     | 8791.0535        | 3.7x
#7. mod e-means (2.0 epochs)                  | 134.80     | 9240.6504        | 2.8x
#8. mod e-means (3.0 epochs)                  | 179.62     | 9598.2402        | 2.1x
#9. mod e-means (5.0 epochs)                  | 263.11     | 9983.4797        | 1.5x
#14. Engine v5 (Residual Power Seeding)       | 230.44     | 10027.3399       | 1.7x
#15. HYBRID v5 (Engine v5 -> e-means 1.0ep)   | 342.55     | 9121.2601        | 1.1x
#16. Engine v6 (Sync-Free SSE + Island)       | 217.86     | 10547.2951       | 1.8x
#17. HYBRID v6 (v6 -> e-means 1.0ep)          | 306.06     | 9150.0017        | 1.3x
#18. HYBRID v6-rob (q95 SSE -> e-means 1ep)   | 310.04     | 8909.9975        | 1.2x
#19. Engine v7 (Batched log2 PDDP)            | 16.03      | 10941.3298       | 23.9x
#20. HYBRID v7 (v7 -> e-means 1.0ep)          | 105.39     | 9243.2547        | 3.6x
#21. HYBRID v7-fast (v7 -> e-means 0.5ep)     | 84.22      | 9279.0798        | 4.6x
#22. Engine v9 (Batched Priority + Island)    | 26.70      | 11478.2692       | 14.4x
#23. HYBRID v9-warm (v9 -> warm 0.5ep)        | 78.30      | 11261.4052       | 4.9x
#24. HYBRID v9-2lloyd (v9 -> 2x Lloyd)        | 77.90      | 11346.9463       | 1.0x
#24. HYBRID abl islands                       | 25.93      | 11417.8217       | 3.0x
#24. HYBRID abl no islands                    | 24.50      | 10739.3337       | 3.2x
#25. Engine v10 (No-Island Priority PDDP)     | 23.88      | 10740.8417       | 3.3x
#26. HYBRID v10-1lloyd (No-Island -> 1xL)     | 49.95      | 10704.7045       | 1.6x
#27. HYBRID v10-warm (No-Island -> 0.5ep)     | 75.56      | 10725.4398       | 1.0x
#28. HYBRID v10-guard (10%-Guard -> 0.5ep)    | 79.29      | 10370.9170       | 1.0x
#29. HYBRID v11 (v10 -> warm+rev0.15 0.5ep)   | 90.27      | 10649.8447       | 0.9x
#30. HYBRID v11-fast (v10 -> warm+rev 0.35ep) | 87.66      | 9887.4050        | 0.9x
#31. HYBRID v12-1/8 (PDDP+q95 D2 -> 0.5ep)    | 80.58      | 10650.4379       | 1.0x
#32. HYBRID v12-1/4 (PDDP+q95 D2 -> 0.5ep)    | 85.59      | 10671.7108       | 0.9x
#33. HYBRID v13                               | 89.95      | 10593.6522       | 0.9x
#33. HYBRID v14                               | 96.06      | 10776.0818       | 0.8x