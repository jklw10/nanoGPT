import argparse
import gc
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch._inductor.config.disable_cpp_codegen = True
torch._dynamo.config.cache_size_limit = 128


# =====================================================================
# 1. Shared Primitives & Compiled Kernels
# =====================================================================

def _power_iteration_step(X: torch.Tensor, v: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    z = torch.mv(X, v) - torch.dot(mu, v)
    v_next = torch.mv(X.T, z) - z.sum() * mu
    return v_next / (torch.linalg.norm(v_next) + 1e-9)


power_iteration_step_compiled = torch.compile(_power_iteration_step)


def get_normal_power_iteration(X_static: torch.Tensor, iters: int = 2) -> torch.Tensor:
    d = X_static.shape[1]
    mu = X_static.mean(dim=0)
    v = torch.randn(d, device=device, dtype=X_static.dtype)
    v = v / (torch.linalg.norm(v) + 1e-9)
    for _ in range(iters):
        v = power_iteration_step_compiled(X_static, v, mu)
    return v


def _lloyd_step_cublas(X: torch.Tensor, C: torch.Tensor, chunk_size: int = 65536) -> torch.Tensor:
    """Single full-dataset Lloyd update using cuBLAS addmm scores: x^T c - 0.5 ||c||^2."""
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

    dead_mask = counts == 0
    updated_C = new_C / torch.clamp(counts, min=1.0)
    return torch.where(dead_mask, C, updated_C)


def _emeans_step_cublas(B_X: torch.Tensor, C: torch.Tensor, S: torch.Tensor, N_stat: torch.Tensor, alpha: float):
    """Single mini-batch EMA update of sufficient statistics (S, N_stat) and centroids C."""
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

    valid = N_stat.squeeze(1) > 1e-6
    new_C = (S / torch.clamp(N_stat, min=1e-6)).to(C.dtype)
    C = torch.where(valid.unsqueeze(1), new_C, C)
    return C, S, N_stat, assign, scores


def get_emeans_hyperparameters(n: int, k: int, epochs: float):
    """Scaling laws for batch size B and initial learning rate a0 from Bessa et al."""
    B = 0.035 * (k ** 0.13) * (n ** 0.88) * (epochs ** 0.81)
    B = max(k * 2, min(int(round(B)), n))
    B = ((B + 127) // 128) * 128
    B = min(B, n)

    a0 = 0.824 * (k ** -0.46) * (B ** 0.90) * (n ** -0.45) * (epochs ** -0.11)
    a0 = min(max(float(a0), 0.01), 1.0)
    return B, a0


# =====================================================================
# 2. Older SOTA / Standard Baselines
# =====================================================================

def run_lloyd_kmeans(X: torch.Tensor, k: int, iters: int = 15) -> torch.Tensor:
    """
    Classic Lloyd's k-means:
      - Random Forgy initialization (k uniform samples from X).
      - `iters` exact full-dataset cuBLAS assignment + mean-update passes.
      - Dead centroids retain their previous coordinates.
    """
    n = X.shape[0]
    C = X[torch.randint(0, n, (k,), device=device)].clone()
    for _ in range(iters):
        C = _lloyd_step_cublas(X, C)
    return C


def run_divisive_tree(X: torch.Tensor, k: int, split_rule: str = "median") -> torch.Tensor:
    """
    Sequential Divisive Tree Baselines (Annoy-style / RP-Tree):
      - Repeatedly pops the largest cluster by point count until k leaves exist.
      - Estimates top principal direction via 2-step compiled power iteration on a 4096-pt sample.
      - split_rule="midpoint" (Annoy-like): splits at 0.5 * (min + max) along projection.
      - split_rule="median"   (RP-Tree):    splits at median of a 4096-pt subsample.
      - Falls back to median / 50-50 split if a cut produces <10% or >90% of cluster points.
    """
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

        if split_rule == "median":
            sub_p = proj[torch.randint(0, m, (min(m, 4096),), device=device)] if m > 4096 else proj
            thresh = torch.median(sub_p)
        else:  # "midpoint"
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

        clusters.append(idxs[l_mask])
        clusters.append(idxs[~l_mask])

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


def run_original_emeans(X: torch.Tensor, k: int, epochs: float = 3.0, init_C: torch.Tensor = None) -> torch.Tensor:
    """
    Original e-means (Bessa et al., arXiv:2611.09501):
      - Mini-batch EMA updates of (S, N_stat) using heuristic batch size B and cosine-annealed
        learning rate alpha in [0.1*a0, a0].
      - Every 5 steps, revives dead clusters (N_stat < 0.1 * mean(N_stat)) by sampling active
        donors WITHOUT replacement proportional to N_stat and splitting each donor into
        c * (1 +/- 1/1024) alternating by parity with halved weight (Section 3.2, App. H.1).
      - Finishes with 1 exact full-dataset Lloyd pass.
    """
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

        C, S, N_stat, _, _ = _emeans_step_cublas(B_X, C, S, N_stat, alpha)

        if (step + 1) % 5 == 0:
            mean_n = N_stat.mean().item()
            dead_mask = N_stat.squeeze(1) < 0.1 * mean_n
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

    return _lloyd_step_cublas(X, C)


# =====================================================================
# 3. Fast Divisive Seeding & Damped-Lloyd Methods
# =====================================================================

@torch.no_grad()
def run_engine_v11_aware(
    X: torch.Tensor,
    k: int,
    use_sse: bool = True,
    valley_boost: float = 2.5,
    robust_cap_q: float = 0.95,
    bw_scale: float = 0.65,
) -> torch.Tensor:
    n, d = X.shape
    sub_n = min(16384, n)
    X_sub = X[torch.randint(0, n, (sub_n,), device=X.device)]
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

        O = torch.zeros(sub_n, m, device=X.device, dtype=X.dtype).scatter_(1, assign.unsqueeze(1), 1.0)
        O_w = O * w_pt.unsqueeze(1)
        total_cnt = O.sum(dim=0)
        w_sum = O_w.sum(dim=0).clamp(min=1e-6)
        M = (O_w.T @ X_sub) / w_sum.unsqueeze(1)

        # 1. Dual-Axis Initialization
        V1 = torch.randn(m, d, device=X.device, dtype=X.dtype)
        pv = prev_V[:m]
        V1 = V1 - (V1 * pv).sum(dim=1, keepdim=True) * pv
        V1 = V1 / (V1.norm(dim=1, keepdim=True) + 1e-9)

        V2 = torch.randn(m, d, device=X.device, dtype=X.dtype)
        V2 = V2 - (V2 * V1).sum(dim=1, keepdim=True) * V1
        V2 = V2 / (V2.norm(dim=1, keepdim=True) + 1e-9)

        # 2. Deflated Dual Power Iteration (stacked 2m GEMM)
        for _ in range(3):
            V_both = torch.cat([V1, V2], dim=0)
            dots = X_sub @ V_both.T
            dot1 = dots[:, :m].gather(1, assign.unsqueeze(1)).squeeze(1)
            dot2 = dots[:, m:].gather(1, assign.unsqueeze(1)).squeeze(1)

            mu1 = (M * V1).sum(dim=1)
            mu2 = (M * V2).sum(dim=1)

            z1 = dot1 - mu1[assign]
            z2 = dot2 - mu2[assign]

            Z1_mat = O_w * z1.unsqueeze(1)
            Z2_mat = O_w * z2.unsqueeze(1)
            Z_both = torch.cat([Z1_mat, Z2_mat], dim=1)

            V_next_both = Z_both.T @ X_sub - torch.cat([
                Z1_mat.sum(dim=0).unsqueeze(1) * M,
                Z2_mat.sum(dim=0).unsqueeze(1) * M,
            ], dim=0)

            V1 = V_next_both[:m]
            V1 = V1 / (V1.norm(dim=1, keepdim=True) + 1e-9)

            V2 = V_next_both[m:]
            V2 = V2 - (V2 * V1).sum(dim=1, keepdim=True) * V1
            V2 = V2 / (V2.norm(dim=1, keepdim=True) + 1e-9)

        # 3. 2D Coordinates & Mode Tracking
        V_both = torch.cat([V1, V2], dim=0)
        dots = X_sub @ V_both.T
        u1 = dots[:, :m].gather(1, assign.unsqueeze(1)).squeeze(1) - (M * V1).sum(dim=1)[assign]
        u2 = dots[:, m:].gather(1, assign.unsqueeze(1)).squeeze(1) - (M * V2).sum(dim=1)[assign]

        var1 = torch.zeros(m, device=X.device, dtype=torch.float32).scatter_add_(0, assign, (w_pt * u1 * u1).float()) / w_sum.float()
        sig1 = torch.sqrt(var1.clamp(min=1e-12))
        cA_1 = -0.75 * sig1
        cA_2 = torch.zeros_like(sig1)
        cB_1 = +0.75 * sig1
        cB_2 = torch.zeros_like(sig1)

        for _ in range(3):
            dA2 = (u1 - cA_1[assign]) ** 2 + (u2 - cA_2[assign]) ** 2
            dB2 = (u1 - cB_1[assign]) ** 2 + (u2 - cB_2[assign]) ** 2
            is_B_pt = dB2 < dA2

            w_B_pt = w_pt * is_B_pt.to(X.dtype)
            w_A_pt = w_pt * (~is_B_pt).to(X.dtype)

            sum_wA = torch.zeros(m, device=X.device, dtype=torch.float32).scatter_add_(0, assign, w_A_pt.float()).clamp(min=1e-6)
            sum_wB = torch.zeros(m, device=X.device, dtype=torch.float32).scatter_add_(0, assign, w_B_pt.float()).clamp(min=1e-6)

            cA_1 = torch.zeros(m, device=X.device, dtype=torch.float32).scatter_add_(0, assign, (w_A_pt * u1).float()) / sum_wA
            cA_2 = torch.zeros(m, device=X.device, dtype=torch.float32).scatter_add_(0, assign, (w_A_pt * u2).float()) / sum_wA
            cB_1 = torch.zeros(m, device=X.device, dtype=torch.float32).scatter_add_(0, assign, (w_B_pt * u1).float()) / sum_wB
            cB_2 = torch.zeros(m, device=X.device, dtype=torch.float32).scatter_add_(0, assign, (w_B_pt * u2).float()) / sum_wB

        del_1 = cB_1 - cA_1
        del_2 = cB_2 - cA_2
        norm_del = torch.sqrt(del_1 * del_1 + del_2 * del_2).clamp(min=1e-9)
        dir_1 = (del_1 / norm_del).to(X.dtype).unsqueeze(1)
        dir_2 = (del_2 / norm_del).to(X.dtype).unsqueeze(1)

        V_star = dir_1 * V1 + dir_2 * V2
        V_star = V_star / (V_star.norm(dim=1, keepdim=True) + 1e-9)

        # 4. Project along V* & evaluate Split-Half KDE consensus
        proj = (X_sub @ V_star.T).gather(1, assign.unsqueeze(1)).squeeze(1)
        dot_mu_v = (M * V_star).sum(dim=1)

        diff = proj - dot_mu_v[assign]
        var_buf = torch.zeros(m, device=X.device, dtype=torch.float32).scatter_add_(0, assign, (w_pt * diff * diff).float())
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
        K_kernel = torch.exp(-0.5 * dist2.unsqueeze(0) / h2)
        K_kernel = K_kernel / K_kernel.sum(dim=1, keepdim=True).clamp(min=1e-6)
        dens_ab = torch.bmm(K_kernel, raw_ab)
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

        min_dens_max = torch.minimum(dens_A, dens_B).max(dim=1).values
        has_valley = (v_max > 0.03 * min_dens_max) & (agree > 0.35)
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

        # 5. Split Ranking: With SSE vs Pure Valley (Without SSE)
        if use_sse:
            delta_sse = (wl * wr / (wl + wr).clamp(min=1e-6)) * torch.linalg.vector_norm(C_left - C_right, dim=1).square()
            delta_sse = torch.where(valid, delta_sse, torch.zeros_like(delta_sse))
            # Valley Boost: penalize unimodal clouds from cutting
            ranking_score = delta_sse * torch.where(has_valley, valley_boost, 1.0)
        else:
            # Pure Valley: Normalized persistence depth (v_max / min_dens_max).
            # When no valleys exist anywhere, falls back to normalized cluster size (total_cnt).
            valley_rel_depth = torch.where(has_valley, v_max / min_dens_max.clamp(min=1e-6), torch.zeros_like(v_max))
            ranking_score = torch.where(valid, valley_rel_depth, torch.zeros_like(valley_rel_depth))
            if ranking_score.max() == 0:
                ranking_score = torch.where(valid, total_cnt.float(), torch.zeros_like(total_cnt.float()))

        active_cnt = int((ranking_score >= 0.05 * ranking_score.max()).sum().item())
        b = max(1, min(b_sched, active_cnt))
        _, top_ids = torch.topk(ranking_score, b, largest=True)

        ok = valid[top_ids]
        C_buf[top_ids] = torch.where(ok.unsqueeze(1), C_left[top_ids], C_curr[top_ids])
        C_buf[m : m + b] = torch.where(ok.unsqueeze(1), C_right[top_ids], X_sub[torch.arange(b, device=X.device) % sub_n])
        prev_V[top_ids] = V_star[top_ids]
        prev_V[m : m + b] = V_star[top_ids]
        m += b

    return C_buf

@torch.no_grad()
def run_hybrid_v11_aware_emeans(X: torch.Tensor, k: int, epochs: float = 0.5) -> torch.Tensor:
    """Hybrid: Engine v11-aware seeding -> 0.5 epoch e-means polish."""
    C = run_engine_v11_aware(X, k)
    return run_original_emeans(X, k, epochs=epochs, init_C=C)

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
    Q, _ = torch.linalg.qr(torch.randn(D, latent_rank, device=device))
    U = Q.T

    topic_centers = torch.randn(num_topics, latent_rank, device=device)
    topic_centers /= topic_centers.norm(dim=1, keepdim=True)

    probs = (torch.arange(num_topics, device=device, dtype=torch.float32) + 2.0) ** -1.1
    probs /= probs.sum()
    topic_ids = torch.multinomial(probs, N, replacement=True)

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

    g_pts = n_giants // 2
    for g in range(2):
        center = torch.randn(D, device=device) * 4.0
        start = g * g_pts
        end = (g + 1) * g_pts if g == 0 else n_giants
        torch.randn(end - start, D, device=device, out=X[start:end]).mul_(3.2).add_(center)

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

def make_task_two_uniform_blocks(N: int, D: int, K: int, gap: float = 2.0, split_ratio: float = 0.5):
 
    n_a = int(split_ratio * N)

    # Generate points uniformly in [-1, 1]^D
    X = torch.empty((N, D), device=device, dtype=torch.float32).uniform_(-1.0, 1.0)

    # Anisotropic dimensions: stretch random dimensions to make realistic hyper-rectangles
    box_scales = 1.0 + 3.0 * torch.rand(D, device=device, dtype=torch.float32)
    X.mul_(box_scales)

    # Separate Block A and Block B along axis 0 by (box_span + gap)
    half_shift = box_scales[0].item() + 0.5 * gap
    X[:n_a, 0] -= half_shift
    X[n_a:, 0] += half_shift

    # Randomly rotate the whole dataset in D dimensions so the cut isn't trivially axis-aligned
    Q, _ = torch.linalg.qr(torch.randn(D, D, device=device))
    X = X @ Q

    # Shuffle row order
    perm = torch.randperm(N, device=device)
    return X[perm], None

# =====================================================================
# 6. Speed / Distortion Benchmark Runner
# =====================================================================

@torch.no_grad()
def evaluate_mse(X: torch.Tensor, centroids: torch.Tensor, core_mask: torch.Tensor = None):
    chunk_size = 32768
    total_sq = 0.0
    core_sq = 0.0
    n = X.shape[0]
    for i in range(0, n, chunk_size):
        chunk = X[i : i + chunk_size]
        dists = torch.cdist(chunk, centroids, p=2)
        min_sq = torch.min(dists * dists, dim=1).values
        total_sq += min_sq.sum().item()
        if core_mask is not None:
            core_sq += min_sq[core_mask[i : i + chunk_size]].sum().item()
    total_mse = total_sq / n
    core_mse = (core_sq / core_mask.sum().item()) if core_mask is not None else None
    return total_mse, core_mse


def run_benchmark():
    tasks = [
        {"title": "Task 1a: Isolated Gaussian Blobs + 20 Outliers",  "fn": make_task_isolated_blobs,     "N": 640000, "D": 1024, "K": 16},
        {"title": "Task 1b: Isolated Gaussian Blobs + 20 Outliers",  "fn": make_task_isolated_blobs,     "N": 640000, "D": 1024, "K": 64},
        {"title": "Task 2: L2-Normalized Manifold (Zipfian Hubs)",   "fn": make_task_l2_manifold,        "N": 640000, "D": 1024, "K": 64},
        {"title": "Task 3: Rotated Anisotropic Subspace",            "fn": make_task_anisotropic_subspace,"N": 640000, "D": 1024, "K": 64},
        {"title": "Task 4: 96% Tight Core + 4% Heavy-Tail Clutter",  "fn": make_task_heavy_tail_trap,    "N": 640000, "D": 1024, "K": 64},
        {"title": "Task 5: Extreme Imbalance (2 Giants + 62 Dwarfs)","fn": make_task_extreme_imbalance,  "N": 640000, "D": 1024, "K": 64},
       #{"title": "Task 6: uniform distribution",                    "fn": make_task_uniform,  "N": 640000, "D": 1024, "K": 64},
       #{"title": "Task 7: Two Rotated Uniform Blocks (Flat Density)","fn": make_task_two_uniform_blocks, "N": 640000, "D": 1024, "K": 64},
    ]

    algos = [
        # Older SOTA / Standard Baselines
        ("Classic Lloyd k-means (15 iters)",            lambda x, k: run_lloyd_kmeans(x, k, iters=15)),
        ("Annoy Divisive (Power-Iter + Midpoint)",      lambda x, k: run_divisive_tree(x, k, split_rule="midpoint")),
        ("RP-Tree Divisive (Power-Iter + Median)",      lambda x, k: run_divisive_tree(x, k, split_rule="median")),
        ("Original e-means (1.0 epoch)",                lambda x, k: run_original_emeans(x, k, epochs=1.0)),
        ("Original e-means (3.0 epochs)",               lambda x, k: run_original_emeans(x, k, epochs=3.0)),
        # My Methods    
        ("v11-aware",                            lambda x, k: run_engine_v11_aware(x, k)),
        ("v11-aware emeans",                     lambda x, k: run_hybrid_v11_aware_emeans(x, k)),
    ]

    print(f"Hardware: {device.type.upper()} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
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
            print(f"{'Algorithm':<46} | {'Time (ms)':<10} | {'Total MSE':<13} | {'Core MSE':<13} | {'Speedup'}")
            print("-" * 102)
        else:
            print(f"{'Algorithm':<46} | {'Time (ms)':<10} | {'MSE (Distortion)':<16} | {'Speedup'}")
            print("-" * 86)

        if device.type == "cuda":
            torch.cuda.synchronize()
        baseline_time = None

        for idx, (name, fn) in enumerate(algos):
            if device.type == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            centroids = fn(X, K)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if idx == 0:
                baseline_time = elapsed_ms

            total_mse, core_mse = evaluate_mse(X, centroids, core_mask)
            speedup_str = f"{baseline_time / elapsed_ms:.1f}x" if (baseline_time and elapsed_ms > 0) else "1.0x"

            if has_core:
                print(f"{name:<46} | {elapsed_ms:<10.2f} | {total_mse:<13.4f} | {core_mse:<13.4f} | {speedup_str}")
            else:
                print(f"{name:<46} | {elapsed_ms:<10.2f} | {total_mse:<16.4f} | {speedup_str}")

        del X, core_mask
        if device.type == "cuda":
            torch.cuda.empty_cache()


# =====================================================================
# 7. 2D PCA & Cluster-Balance Visualizer
# =====================================================================

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
    _, _, V = torch.pca_lowrank(X_pca - mean_x, q=2, center=False)

    X_vis = X[vis_idx]
    X_vis_2d = ((X_vis - mean_x) @ V).cpu().numpy()
    vis_is_core = core_mask[vis_idx].cpu().numpy() if core_mask is not None else np.ones(X_vis.shape[0], dtype=bool)
    return X_vis, X_vis_2d, vis_is_core, mean_x, V


def _plot_cluster_panel(ax, X_vis_2d, vis_assign, C_2d, counts, xlim, ylim, title, subtitle):
    k = C_2d.shape[0]
    cmap = plt.get_cmap("turbo")
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
    algo_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
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
    ax.legend(fontsize=6.6, loc="lower left", framealpha=0.85)


def run_visualizer(out_dir: str = "viz_outputs", show: bool = False):
    os.makedirs(out_dir, exist_ok=True)

    tasks = [
        {"slug": "task1a_blobs_k16",  "title": "Task 1a: Isolated Blobs + 20 Outliers",          "fn": make_task_isolated_blobs,      "N": 640000, "D": 1024, "K": 16},
        {"slug": "task1b_blobs_k64",  "title": "Task 1b: Isolated Blobs + 20 Outliers",          "fn": make_task_isolated_blobs,      "N": 640000, "D": 1024, "K": 64},
        {"slug": "task2_l2_manifold", "title": "Task 2: L2-Normalized Manifold (Zipfian Hubs)",  "fn": make_task_l2_manifold,         "N": 640000, "D": 1024, "K": 64},
        {"slug": "task3_anisotropic", "title": "Task 3: Rotated Anisotropic Subspace",           "fn": make_task_anisotropic_subspace,"N": 640000, "D": 1024, "K": 64},
        {"slug": "task4_heavy_tail",  "title": "Task 4: 96% Tight Core + 4% Heavy-Tail Clutter", "fn": make_task_heavy_tail_trap,     "N": 640000, "D": 1024, "K": 64},
        {"slug": "task5_imbalance",   "title": "Task 5: Extreme Imbalance (2 Giants + 62 Dwarfs)","fn": make_task_extreme_imbalance,  "N": 640000, "D": 1024, "K": 64},
        {"slug": "task6_uniform",     "title": "Task 6: Uniform Hypercube [-1, 1]^D",           "fn": make_task_uniform,             "N": 640000, "D": 1024, "K": 64},
    ]

    algos = [
        ("Classic Lloyd (15 iters)",          "Classic Lloyd",   lambda x, k: run_lloyd_kmeans(x, k, iters=15)),
        ("Original e-means (1.0 ep)",         "og e-means 1ep",  lambda x, k: run_original_emeans(x, k, epochs=1.0)),
    ]

    n_tasks = len(tasks)
    n_algos = len(algos)
    n_panels = n_algos + 1
    ncols_t = 4
    nrows_t = math.ceil(n_panels / ncols_t)

    fig_all, axes_all = plt.subplots(n_tasks, n_panels, figsize=(3.9 * n_panels, 3.2 * n_tasks), dpi=150)
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

        core_pts_2d = X_vis_2d[vis_is_core] if np.any(vis_is_core) else X_vis_2d
        lo = np.percentile(core_pts_2d, 0.3, axis=0)
        hi = np.percentile(core_pts_2d, 99.7, axis=0)
        pad = 0.14 * np.maximum(hi - lo, 1e-4)
        xlim = (lo[0] - pad[0], hi[0] + pad[0])
        ylim = (lo[1] - pad[1], hi[1] + pad[1])

        fig_t, axes_t = plt.subplots(nrows_t, ncols_t, figsize=(4.8 * ncols_t, 4.6 * nrows_t), dpi=160)
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

            print(f"  {full_name:<30} | {ms:7.2f} ms | MSE={total_mse:10.4f} | Dead={dead_cnt}")

            _plot_cluster_panel(axes_t_flat[a_idx], X_vis_2d, vis_assign, C_2d, counts, xlim, ylim, full_name, subtitle)
            _plot_cluster_panel(
                axes_all[t_idx, a_idx], X_vis_2d, vis_assign, C_2d, counts, xlim, ylim,
                full_name if t_idx == 0 else "", subtitle,
            )

            task_results.append({
                "short_name": short_name,
                "ms": ms,
                "total_mse": total_mse,
                "core_mse": core_mse,
                "counts": counts,
            })

        _plot_balance_panel(axes_t_flat[n_algos], task_results, X.shape[0], K)
        _plot_balance_panel(axes_all[t_idx, n_algos], task_results, X.shape[0], K)

        for extra_idx in range(n_panels, len(axes_t_flat)):
            axes_t_flat[extra_idx].set_visible(False)

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


# =====================================================================
# 8. CUDA VRAM Profiler
# =====================================================================

def _fmt_mib(num_bytes: int) -> str:
    return f"{num_bytes / (1024.0 ** 2):8.1f} MiB"


def _summarize_peak_snapshot(snap: dict, dev_idx: int = 0, top_k: int = 8):
    """Replays PyTorch's CUDA memory trace to show what was live at peak allocation."""
    traces = snap.get("device_traces", [])
    if dev_idx >= len(traces) or not traces[dev_idx]:
        print("  [trace] No device trace events captured.")
        return

    live = {}
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
        {"title": "Task 1a: Isolated Blobs (k=16)", "fn": make_task_isolated_blobs,       "N": 640000, "D": 1024, "K": 16},
        {"title": "Task 1b: Isolated Blobs (k=64)", "fn": make_task_isolated_blobs,       "N": 640000, "D": 1024, "K": 64},
        {"title": "Task 2: L2 Manifold",            "fn": make_task_l2_manifold,          "N": 640000, "D": 1024, "K": 64},
        {"title": "Task 3: Rotated Anisotropic",    "fn": make_task_anisotropic_subspace, "N": 640000, "D": 1024, "K": 64},
        {"title": "Task 4: Heavy-Tail Clutter",     "fn": make_task_heavy_tail_trap,      "N": 640000, "D": 1024, "K": 64},
        {"title": "Task 5: Extreme Imbalance",      "fn": make_task_extreme_imbalance,    "N": 640000, "D": 1024, "K": 64},
    ]

    algos = [
        ("Classic Lloyd k-means (15 iters)",            lambda x, k: run_lloyd_kmeans(x, k, iters=15)),
        ("Original e-means (1.0 epoch)",                lambda x, k: run_original_emeans(x, k, epochs=1.0)),
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
            f"{'Algorithm':<46} | {'Peak Extra':<11} | {'Peak Total':<11} | "
            f"{'Peak Rsvd':<11} | {'Retained':<10} | {'Drv Free':<11} | {'Retries'}"
        )
        print("-" * 126)

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
                f"{name:<46} | {_fmt_mib(peak_extra):<11} | {_fmt_mib(peak_alloc):<11} | "
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
    parser.add_argument("--vram", action="store_true", help="Run CUDA VRAM profiler instead of speed benchmark")
    parser.add_argument("--visual", action="store_true", help="Run 2D PCA & balance visualizer")
    parser.add_argument("--trace-all", action="store_true", help="Trace peak call stacks on every task (not just Task 1a)")
    args, _ = parser.parse_known_args()

    if args.vram:
        run_vram_profile(trace_peak_allocs=True, trace_first_task_only=not args.trace_all)
    elif args.visual:
        run_visualizer()
    else:
        run_benchmark()


#   =====================================================================
#   9. Recorded Benchmark Results (NVIDIA GeForce RTX 3060)
#   =====================================================================
#
#   === Task 1a: Isolated Gaussian Blobs + 20 Outliers (N=640,020 | D=1024 | k=16) ===
#   Algorithm                                      | Time (ms)  | Total MSE     | Core MSE      | Speedup
#   ------------------------------------------------------------------------------------------------------
#   Classic Lloyd k-means (15 iters)               | 438.83     | 3041.6110     | 3001.9409     | 1.0x
#   Annoy Divisive (Power-Iter + Midpoint)         | 219.14     | 1106.6349     | 1066.8946     | 2.0x
#   RP-Tree Divisive (Power-Iter + Median)         | 180.27     | 1097.6849     | 1057.9424     | 2.4x
#   Original e-means (1.0 epoch)                   | 217.78     | 2015.2855     | 1975.5860     | 2.0x
#   Original e-means (3.0 epochs)                  | 149.88     | 2023.9491     | 1984.2651     | 2.9x
#   Damped Lloyd k-means (15 iters)                | 410.10     | 5654.7840     | 5615.1749     | 1.1x
#   Engine v10-aware                               | 182.87     | 1064.5972     | 1024.8434     | 2.4x
#   HYBRID v10-aware -> 2x Damped Lloyd            | 85.35      | 1064.8239     | 1025.0701     | 5.1x
#   
#   === Task 1b: Isolated Gaussian Blobs + 20 Outliers (N=640,020 | D=1024 | k=64) ===
#   Algorithm                                      | Time (ms)  | Total MSE     | Core MSE      | Speedup
#   ------------------------------------------------------------------------------------------------------
#   Classic Lloyd k-means (15 iters)               | 384.51     | 3758.4617     | 3722.7886     | 1.0x
#   Annoy Divisive (Power-Iter + Midpoint)         | 572.69     | 1652.4726     | 1612.6413     | 0.7x
#   RP-Tree Divisive (Power-Iter + Median)         | 572.10     | 2552.7824     | 2512.9937     | 0.7x
#   Original e-means (1.0 epoch)                   | 86.48      | 2279.6719     | 2239.8428     | 4.4x
#   Original e-means (3.0 epochs)                  | 165.35     | 2028.2930     | 1988.4669     | 2.3x
#   Damped Lloyd k-means (15 iters)                | 508.36     | 5369.9818     | 5332.1759     | 0.8x
#   Engine v10-aware                               | 70.39      | 1067.7843     | 1027.9086     | 5.5x
#   HYBRID v10-aware -> 2x Damped Lloyd            | 113.30     | 1064.4451     | 1024.5707     | 3.4x
#   
#   === Task 2: L2-Normalized Manifold (Zipfian Hubs) (N=640,000 | D=1024 | k=64) ===
#   Algorithm                                      | Time (ms)  | MSE (Distortion) | Speedup
#   --------------------------------------------------------------------------------------
#   Classic Lloyd k-means (15 iters)               | 380.46     | 0.8976           | 1.0x
#   Annoy Divisive (Power-Iter + Midpoint)         | 573.52     | 0.9141           | 0.7x
#   RP-Tree Divisive (Power-Iter + Median)         | 590.32     | 0.9147           | 0.6x
#   Original e-means (1.0 epoch)                   | 80.54      | 0.8956           | 4.7x
#   Original e-means (3.0 epochs)                  | 158.72     | 0.8945           | 2.4x
#   Damped Lloyd k-means (15 iters)                | 508.02     | 0.8976           | 0.7x
#   Engine v10-aware                               | 39.96      | 0.9053           | 9.5x
#   HYBRID v10-aware -> 2x Damped Lloyd            | 107.91     | 0.9019           | 3.5x
#   
#   === Task 3: Rotated Anisotropic Subspace (N=640,000 | D=1024 | k=64) ===
#   Algorithm                                      | Time (ms)  | MSE (Distortion) | Speedup
#   --------------------------------------------------------------------------------------
#   Classic Lloyd k-means (15 iters)               | 379.70     | 830.6502         | 1.0x
#   Annoy Divisive (Power-Iter + Midpoint)         | 574.74     | 452.7610         | 0.7x
#   RP-Tree Divisive (Power-Iter + Median)         | 576.01     | 810.5841         | 0.7x
#   Original e-means (1.0 epoch)                   | 83.01      | 688.8532         | 4.6x
#   Original e-means (3.0 epochs)                  | 159.88     | 765.1186         | 2.4x
#   Damped Lloyd k-means (15 iters)                | 507.13     | 948.2379         | 0.7x
#   Engine v10-aware                               | 45.02      | 224.6109         | 8.4x
#   HYBRID v10-aware -> 2x Damped Lloyd            | 116.46     | 224.0022         | 3.3x
#   
#   === Task 4: 96% Tight Core + 4% Heavy-Tail Clutter (N=640,000 | D=1024 | k=64) ===
#   Algorithm                                      | Time (ms)  | Total MSE     | Core MSE      | Speedup
#   ------------------------------------------------------------------------------------------------------
#   Classic Lloyd k-means (15 iters)               | 379.02     | 7124.4656     | 3159.3019     | 1.0x
#   Annoy Divisive (Power-Iter + Midpoint)         | 580.78     | 5567.2200     | 1479.1834     | 0.7x
#   RP-Tree Divisive (Power-Iter + Median)         | 581.80     | 6005.8590     | 1956.4526     | 0.7x
#   Original e-means (1.0 epoch)                   | 85.95      | 5599.1499     | 1565.6543     | 4.4x
#   Original e-means (3.0 epochs)                  | 164.26     | 5436.8494     | 1366.0159     | 2.3x
#   Damped Lloyd k-means (15 iters)                | 508.38     | 7422.9238     | 3432.1962     | 0.7x
#   Engine v10-aware                               | 51.29      | 5271.1078     | 1028.6087     | 7.4x
#   HYBRID v10-aware -> 2x Damped Lloyd            | 112.38     | 5266.3449     | 1024.1359     | 3.4x
#   
#   === Task 5: Extreme Imbalance (2 Giants + 62 Dwarfs) (N=640,000 | D=1024 | k=64) ===
#   Algorithm                                      | Time (ms)  | MSE (Distortion) | Speedup
#   --------------------------------------------------------------------------------------
#   Classic Lloyd k-means (15 iters)               | 381.29     | 10888.7694       | 1.0x
#   Annoy Divisive (Power-Iter + Midpoint)         | 583.66     | 11017.1746       | 0.7x
#   RP-Tree Divisive (Power-Iter + Median)         | 586.17     | 10991.9225       | 0.7x
#   Original e-means (1.0 epoch)                   | 88.12      | 9606.7910        | 4.3x
#   Original e-means (3.0 epochs)                  | 164.65     | 10849.9219       | 2.3x
#   Damped Lloyd k-means (15 iters)                | 510.76     | 11019.3253       | 0.7x
#   Engine v10-aware                               | 63.74      | 8086.1209        | 6.0x
#   HYBRID v10-aware_ -> 2x Damped Lloyd           | 127.72     | 8083.5966        | 3.0x
#   
#   ============================================================================================
#    Task 1b: Isolated Blobs + Outliers (Averaged across 4 Seeds)
#   ============================================================================================
#   Algorithm                        | Time (ms)        | Total MSE            | Core MSE        
#   --------------------------------------------------------------------------------------------
#   Classic Lloyd (15 iters)         |  434.7 ± 23.9    |  4972.433 ± 704.688  |  4936.00 ± 704.32
#   Original e-means (1.0 ep)        |  129.3 ± 43.7    |  2028.272 ± 385.504  |  1989.21 ± 385.07
#   Engine v10-aware                 |   93.1 ± 63.6    |  1067.053 ± 0.496    |  1027.95 ± 0.05 
#   Engine v11-aware                 |   96.4 ±  7.6    |  1067.053 ± 0.495    |  1027.95 ± 0.05 
#   HYBRID v11-aware -> e-means      |  167.8 ± 15.2    |  1063.029 ± 0.510    |  1023.94 ± 0.05 
#   
#   ============================================================================================
#    Task 2: L2-Normalized Manifold (Averaged across 4 Seeds)
#   ============================================================================================
#   Algorithm                        | Time (ms)        | Total MSE            | Core MSE        
#   --------------------------------------------------------------------------------------------
#   Classic Lloyd (15 iters)         |  425.5 ±  1.7    |     0.898 ± 0.000    | N/A             
#   Original e-means (1.0 ep)        |   99.9 ±  4.9    |     0.895 ± 0.000    | N/A             
#   Engine v10-aware                 |   52.2 ±  3.3    |     0.906 ± 0.000    | N/A             
#   Engine v11-aware                 |   71.2 ± 11.3    |     0.907 ± 0.000    | N/A             
#   HYBRID v11-aware -> e-means      |  135.9 ±  4.7    |     0.897 ± 0.000    | N/A             
#   
#   ============================================================================================
#    Task 3: Rotated Anisotropic (Averaged across 4 Seeds)
#   ============================================================================================
#   Algorithm                        | Time (ms)        | Total MSE            | Core MSE        
#   --------------------------------------------------------------------------------------------
#   Classic Lloyd (15 iters)         |  425.7 ±  1.0    |   711.408 ± 69.051   | N/A             
#   Original e-means (1.0 ep)        |   97.7 ±  1.6    |   670.179 ± 98.113   | N/A             
#   Engine v10-aware                 |   54.0 ±  4.7    |   224.688 ± 0.044    | N/A             
#   Engine v11-aware                 |   84.2 ± 10.7    |   224.689 ± 0.044    | N/A             
#   HYBRID v11-aware -> e-means      |  142.8 ±  6.7    |   223.809 ± 0.031    | N/A             
#   
#   ============================================================================================
#    Task 4: 96% Core + 4% Clutter (Averaged across 4 Seeds)
#   ============================================================================================
#   Algorithm                        | Time (ms)        | Total MSE            | Core MSE        
#   --------------------------------------------------------------------------------------------
#   Classic Lloyd (15 iters)         |  417.1 ±  0.3    |  7046.293 ± 87.979   |  3081.55 ± 90.00
#   Original e-means (1.0 ep)        |   97.5 ±  1.3    |  5702.704 ± 155.655  |  1647.56 ± 156.73
#   Engine v11-aware                 |   77.5 ±  4.0    |  5269.140 ± 1.856    |  1029.10 ± 0.39 
#   HYBRID v11-aware -> e-means      |  135.0 ±  6.6    |  5172.610 ± 2.135    |  1094.90 ± 1.54 
#   
#   ============================================================================================
#    Task 5: Extreme Imbalance (2G + 62D) (Averaged across 4 Seeds)
#   ============================================================================================
#   Algorithm                        | Time (ms)        | Total MSE            | Core MSE        
#   --------------------------------------------------------------------------------------------
#   Classic Lloyd (15 iters)         |  415.8 ±  0.4    | 10937.710 ± 211.988  | N/A             
#   Original e-means (1.0 ep)        |   96.2 ±  0.5    |  9717.994 ± 141.995  | N/A             
#   Engine v10-aware                 |   61.4 ±  4.8    |  8103.561 ± 22.517   | N/A             
#   Engine v11-aware                 |   93.7 ± 11.9    |  7973.746 ± 25.345   | N/A             
#   HYBRID v11-aware -> e-means      |  163.4 ± 15.6    |  7970.973 ± 25.191   | N/A    
#   ---------------------------------------------------------------------
#   C. CUDA VRAM Profile (RTX 3060, X = 2500.0 - 2500.1 MiB)
#   ---------------------------------------------------------------------
#   === Task 1b: Isolated Blobs (k=64) (N=640,020 | D=1024 | k=64 | X=2500.1 MiB) ===
#   Algorithm                                      | Peak Extra  | Peak Total  | Peak Rsvd   | Retained   | Drv Free    | Retries
#   ------------------------------------------------------------------------------------------------------------------------------
#   Classic Lloyd k-means (15 iters)               |     49.0 MiB |   2558.2 MiB |   2574.0 MiB |      0.2 MiB |   8641.0 MiB | 0
#   Original e-means (1.0 epoch)                   |     81.8 MiB |   2591.0 MiB |   2632.0 MiB |      0.2 MiB |   8583.0 MiB | 0
#   Damped Lloyd k-means (15 iters)                |     34.8 MiB |   2544.0 MiB |   2560.0 MiB |      0.2 MiB |   8655.0 MiB | 0
#   Engine v10-aware                               |     98.1 MiB |   2607.3 MiB |   2618.0 MiB |      0.2 MiB |   8597.0 MiB | 0
#   HYBRID v10-aware -> 2x Damped Lloyd            |     96.6 MiB |   2605.8 MiB |   2618.0 MiB |      0.2 MiB |   8597.0 MiB | 0
#   
#   === Task 2: L2 Manifold (N=640,000 | D=1024 | k=64 | X=2500.0 MiB) ===
#   Algorithm                                      | Peak Extra  | Peak Total  | Peak Rsvd   | Retained   | Drv Free    | Retries
#   ------------------------------------------------------------------------------------------------------------------------------
#   Classic Lloyd k-means (15 iters)               |     49.0 MiB |   2558.1 MiB |   2572.0 MiB |      0.2 MiB |   8607.0 MiB | 0
#   Original e-means (1.0 epoch)                   |     81.7 MiB |   2590.8 MiB |   2628.0 MiB |      0.2 MiB |   8551.0 MiB | 0
#   Damped Lloyd k-means (15 iters)                |     34.8 MiB |   2543.9 MiB |   2558.0 MiB |      0.2 MiB |   8621.0 MiB | 0
#   Engine v10-aware                               |     97.5 MiB |   2606.6 MiB |   2616.0 MiB |      0.2 MiB |   8563.0 MiB | 0
#   HYBRID v10-aware -> 2x Damped Lloyd            |     97.5 MiB |   2606.6 MiB |   2616.0 MiB |      0.2 MiB |   8563.0 MiB | 0
#   
#   === Task 3: Rotated Anisotropic (N=640,000 | D=1024 | k=64 | X=2500.0 MiB) ===
#   Algorithm                                      | Peak Extra  | Peak Total  | Peak Rsvd   | Retained   | Drv Free    | Retries
#   ------------------------------------------------------------------------------------------------------------------------------
#   Classic Lloyd k-means (15 iters)               |     49.0 MiB |   2558.1 MiB |   2572.0 MiB |      0.2 MiB |   8607.0 MiB | 0
#   Original e-means (1.0 epoch)                   |     81.8 MiB |   2590.9 MiB |   2628.0 MiB |      0.2 MiB |   8551.0 MiB | 0
#   Damped Lloyd k-means (15 iters)                |     34.8 MiB |   2543.9 MiB |   2558.0 MiB |      0.2 MiB |   8621.0 MiB | 0
#   Engine v10-aware                               |     98.4 MiB |   2607.5 MiB |   2616.0 MiB |      0.2 MiB |   8563.0 MiB | 0
#   HYBRID v10-aware -> 2x Damped Lloyd            |     98.5 MiB |   2607.6 MiB |   2616.0 MiB |      0.2 MiB |   8563.0 MiB | 0
#   
#   === Task 4: Heavy-Tail Clutter (N=640,000 | D=1024 | k=64 | X=2500.0 MiB) ===
#   Algorithm                                      | Peak Extra  | Peak Total  | Peak Rsvd   | Retained   | Drv Free    | Retries
#   ------------------------------------------------------------------------------------------------------------------------------
#   Classic Lloyd k-means (15 iters)               |     49.0 MiB |   2558.1 MiB |   2572.0 MiB |      0.2 MiB |   8607.0 MiB | 0
#   Original e-means (1.0 epoch)                   |     81.7 MiB |   2590.9 MiB |   2630.0 MiB |      0.2 MiB |   8549.0 MiB | 0
#   Damped Lloyd k-means (15 iters)                |     34.8 MiB |   2543.9 MiB |   2558.0 MiB |      0.2 MiB |   8621.0 MiB | 0
#   Engine v10-aware                               |     98.5 MiB |   2607.6 MiB |   2618.0 MiB |      0.2 MiB |   8561.0 MiB | 0
#   HYBRID v10-aware -> 2x Damped Lloyd            |     98.7 MiB |   2607.8 MiB |   2618.0 MiB |      0.2 MiB |   8561.0 MiB | 0
#   
#   === Task 5: Extreme Imbalance (N=640,000 | D=1024 | k=64 | X=2500.0 MiB) ===
#   Algorithm                                      | Peak Extra  | Peak Total  | Peak Rsvd   | Retained   | Drv Free    | Retries
#   ------------------------------------------------------------------------------------------------------------------------------
#   Classic Lloyd k-means (15 iters)               |     49.0 MiB |   2558.1 MiB |   2572.0 MiB |      0.2 MiB |   8607.0 MiB | 0
#   Original e-means (1.0 epoch)                   |     81.7 MiB |   2590.9 MiB |   2630.0 MiB |      0.2 MiB |   8549.0 MiB | 0
#   Damped Lloyd k-means (15 iters)                |     34.8 MiB |   2543.9 MiB |   2558.0 MiB |      0.2 MiB |   8621.0 MiB | 0
#   Engine v10-aware                               |     98.6 MiB |   2607.7 MiB |   2616.0 MiB |      0.2 MiB |   8563.0 MiB | 0
#   HYBRID v10-aware -> 2x Damped Lloyd            |     98.7 MiB |   2607.9 MiB |   2616.0 MiB |      0.2 MiB |   8563.0 MiB | 0