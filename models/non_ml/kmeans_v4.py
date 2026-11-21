import math
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================
# 1. Generators with Ground-Truth Centers & Theoretical Floors
# =====================================================================

def make_task_1a_truth(N=640000, D=1024, K=16):
    torch.manual_seed(42)
    X = torch.empty((N + 20, D), device=device, dtype=torch.float32)
    labels = torch.empty(N + 20, device=device, dtype=torch.long)
    true_c = torch.randn(K, D, device=device) * 4.0
    pts_per_k = N // K
    for i in range(K):
        torch.randn(pts_per_k, D, device=device, out=X[i * pts_per_k : (i + 1) * pts_per_k])
        X[i * pts_per_k : (i + 1) * pts_per_k].add_(true_c[i])
        labels[i * pts_per_k : (i + 1) * pts_per_k] = i
    torch.randn(20, D, device=device, out=X[N:]).mul_(35.0)
    labels[N:] = -1
    return X, labels, true_c, 1063.45, "Task 1a: Isolated Blobs + Outliers (K=16)"


def make_task_1b_truth(N=640000, D=1024, K=64):
    torch.manual_seed(42)
    X = torch.empty((N + 20, D), device=device, dtype=torch.float32)
    labels = torch.empty(N + 20, device=device, dtype=torch.long)
    true_c = torch.randn(K, D, device=device) * 4.0
    pts_per_k = N // K
    for i in range(K):
        torch.randn(pts_per_k, D, device=device, out=X[i * pts_per_k : (i + 1) * pts_per_k])
        X[i * pts_per_k : (i + 1) * pts_per_k].add_(true_c[i])
        labels[i * pts_per_k : (i + 1) * pts_per_k] = i
    torch.randn(20, D, device=device, out=X[N:]).mul_(35.0)
    labels[N:] = -1
    return X, labels, true_c, 1063.45, "Task 1b: Isolated Blobs + Outliers (K=64)"


def make_task_2_truth(N=640000, D=1024, K=64):
    torch.manual_seed(42)
    latent_rank = 64
    Q, _ = torch.linalg.qr(torch.randn(D, latent_rank, device=device))
    U = Q.T
    topic_centers = torch.randn(K, latent_rank, device=device)
    topic_centers /= topic_centers.norm(dim=1, keepdim=True)
    probs = (torch.arange(K, device=device, dtype=torch.float32) + 2.0) ** -1.1
    probs /= probs.sum()
    labels = torch.multinomial(probs, N, replacement=True)
    Z = topic_centers[labels] + torch.randn(N, latent_rank, device=device) * 0.42
    X = Z @ U + torch.randn(N, D, device=device) * 0.015
    X /= torch.clamp(X.norm(dim=1, keepdim=True), min=1e-8)
    true_c = (topic_centers @ U)
    true_c /= true_c.norm(dim=1, keepdim=True)
    return X, labels, true_c, 0.8940, "Task 2: L2-Normalized Manifold (K=64)"


def make_task_3_truth(N=640000, D=1024, K=64):
    torch.manual_seed(42)
    Q, _ = torch.linalg.qr(torch.randn(D, D, device=device))
    eig_scales = 10.0 * (0.985 ** torch.arange(D, device=device, dtype=torch.float32)) + 0.15
    X = torch.empty((N, D), device=device, dtype=torch.float32)
    labels = torch.empty(N, device=device, dtype=torch.long)
    true_c = torch.empty((K, D), device=device, dtype=torch.float32)
    pts_per_k = N // K
    for i in range(K):
        c = (torch.randn(D, device=device) * eig_scales) @ Q
        true_c[i] = c
        local_scales = eig_scales * 0.25
        noise = (torch.randn(pts_per_k, D, device=device) * local_scales) @ Q
        X[i * pts_per_k : (i + 1) * pts_per_k] = noise + c
        labels[i * pts_per_k : (i + 1) * pts_per_k] = i
    return X, labels, true_c, 224.05, "Task 3: Rotated Anisotropic Subspace (K=64)"


def make_task_4_truth(N=640000, D=1024, K=64):
    torch.manual_seed(42)
    n_core = int(0.96 * N)
    n_noise = N - n_core
    X = torch.empty((N, D), device=device, dtype=torch.float32)
    labels = torch.empty(N, device=device, dtype=torch.long)
    true_c = torch.randn(K, D, device=device) * 3.0
    pts_per_k = n_core // K
    for i in range(K):
        torch.randn(pts_per_k, D, device=device, out=X[i * pts_per_k : (i + 1) * pts_per_k])
        X[i * pts_per_k : (i + 1) * pts_per_k].add_(true_c[i])
        labels[i * pts_per_k : (i + 1) * pts_per_k] = i
    torch.randn(n_noise, D, device=device, out=X[n_core:]).mul_(10.0)
    labels[n_core:] = -1
    return X, labels, true_c, 5447.68, "Task 4: 96% Core + 4% Clutter (K=64)"


def make_task_5_truth(N=640000, D=1024, K=64):
    torch.manual_seed(42)
    n_giants = int(0.75 * N)
    n_dwarfs = N - n_giants
    X = torch.empty((N, D), device=device, dtype=torch.float32)
    labels = torch.empty(N, device=device, dtype=torch.long)
    true_c = torch.empty((K, D), device=device, dtype=torch.float32)
    g_pts = n_giants // 2
    for g in range(2):
        c = torch.randn(D, device=device) * 4.0
        true_c[g] = c
        start = g * g_pts
        end = (g + 1) * g_pts if g == 0 else n_giants
        torch.randn(end - start, D, device=device, out=X[start:end]).mul_(3.2).add_(c)
        labels[start:end] = g
    num_dwarfs = K - 2
    d_pts = n_dwarfs // num_dwarfs
    for i in range(num_dwarfs):
        c = torch.randn(D, device=device) * 4.0
        cid = 2 + i
        true_c[cid] = c
        start = n_giants + i * d_pts
        end = n_giants + (i + 1) * d_pts if i < num_dwarfs - 1 else N
        torch.randn(end - start, D, device=device, out=X[start:end]).mul_(0.6).add_(c)
        labels[start:end] = cid
    return X, labels, true_c, 7956.48, "Task 5: Extreme Imbalance (2 Giants + 62 Dwarfs)"


def make_task_7_truth(N=640000, D=1024, K=64):
    torch.manual_seed(42)
    n_a = N // 2
    X = torch.empty((N, D), device=device, dtype=torch.float32).uniform_(-1.0, 1.0)
    box_scales = 1.0 + 3.0 * torch.rand(D, device=device, dtype=torch.float32)
    X.mul_(box_scales)
    half_shift = box_scales[0].item() + 1.0
    X[:n_a, 0] -= half_shift
    X[n_a:, 0] += half_shift
    Q, _ = torch.linalg.qr(torch.randn(D, D, device=device))
    X = X @ Q
    true_c = torch.zeros((2, D), device=device, dtype=torch.float32)
    true_c[0, 0] = -half_shift
    true_c[1, 0] = +half_shift
    true_c = true_c @ Q
    labels = torch.zeros(N, device=device, dtype=torch.long)
    labels[n_a:] = 1
    return X, labels, true_c, 2392.50, "Task 7: Two Rotated Uniform Blocks (K=64)"


# =====================================================================
# 2. Production Engine v11-aware (Valley-Prioritized)
# =====================================================================

@torch.no_grad()
def run_engine_v11_production(X: torch.Tensor, k: int, valley_boost: float = 2.5) -> torch.Tensor:
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
        cap = torch.quantile(raw_sse, 0.95)
        w_pt = torch.clamp(cap / torch.clamp(raw_sse, min=1e-6), max=1.0).to(X.dtype)

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

        # 2. Deflated Dual Power Iteration
        for _ in range(3):
            V_both = torch.cat([V1, V2], dim=0)
            dots = X_sub @ V_both.T
            dot1 = dots[:, :m].gather(1, assign.unsqueeze(1)).squeeze(1)
            dot2 = dots[:, m:].gather(1, assign.unsqueeze(1)).squeeze(1)
            mu1, mu2 = (M * V1).sum(dim=1), (M * V2).sum(dim=1)
            z1, z2 = dot1 - mu1[assign], dot2 - mu2[assign]
            Z1_mat, Z2_mat = O_w * z1.unsqueeze(1), O_w * z2.unsqueeze(1)
            Z_both = torch.cat([Z1_mat, Z2_mat], dim=1)

            V_next = Z_both.T @ X_sub - torch.cat([Z1_mat.sum(dim=0).unsqueeze(1) * M, Z2_mat.sum(dim=0).unsqueeze(1) * M], dim=0)
            V1 = V_next[:m] / (V_next[:m].norm(dim=1, keepdim=True) + 1e-9)
            V2 = V_next[m:]
            V2 = V2 - (V2 * V1).sum(dim=1, keepdim=True) * V1
            V2 = V2 / (V2.norm(dim=1, keepdim=True) + 1e-9)

        # 3. 2D Modal Pursuit & Optimal Rotation theta*
        dots = X_sub @ torch.cat([V1, V2], dim=0).T
        u1 = dots[:, :m].gather(1, assign.unsqueeze(1)).squeeze(1) - (M * V1).sum(dim=1)[assign]
        u2 = dots[:, m:].gather(1, assign.unsqueeze(1)).squeeze(1) - (M * V2).sum(dim=1)[assign]

        var1 = torch.zeros(m, device=X.device, dtype=torch.float32).scatter_add_(0, assign, (w_pt * u1 * u1).float()) / w_sum.float()
        sig1 = torch.sqrt(var1.clamp(min=1e-12))
        cA_1, cA_2 = -0.75 * sig1, torch.zeros_like(sig1)
        cB_1, cB_2 = +0.75 * sig1, torch.zeros_like(sig1)

        for _ in range(3):
            dA2 = (u1 - cA_1[assign]) ** 2 + (u2 - cA_2[assign]) ** 2
            dB2 = (u1 - cB_1[assign]) ** 2 + (u2 - cB_2[assign]) ** 2
            is_B_pt = dB2 < dA2
            w_B_pt, w_A_pt = w_pt * is_B_pt.to(X.dtype), w_pt * (~is_B_pt).to(X.dtype)
            s_wA = torch.zeros(m, device=X.device).scatter_add_(0, assign, w_A_pt.float()).clamp(min=1e-6)
            s_wB = torch.zeros(m, device=X.device).scatter_add_(0, assign, w_B_pt.float()).clamp(min=1e-6)
            cA_1 = torch.zeros(m, device=X.device).scatter_add_(0, assign, (w_A_pt * u1).float()) / s_wA
            cA_2 = torch.zeros(m, device=X.device).scatter_add_(0, assign, (w_A_pt * u2).float()) / s_wA
            cB_1 = torch.zeros(m, device=X.device).scatter_add_(0, assign, (w_B_pt * u1).float()) / s_wB
            cB_2 = torch.zeros(m, device=X.device).scatter_add_(0, assign, (w_B_pt * u2).float()) / s_wB

        del_1, del_2 = cB_1 - cA_1, cB_2 - cA_2
        norm_del = torch.sqrt(del_1 * del_1 + del_2 * del_2).clamp(min=1e-9)
        V_star = (del_1 / norm_del).unsqueeze(1) * V1 + (del_2 / norm_del).unsqueeze(1) * V2
        V_star = V_star / (V_star.norm(dim=1, keepdim=True) + 1e-9)

        # 4. Split-Half Consensus along V*
        proj = (X_sub @ V_star.T).gather(1, assign.unsqueeze(1)).squeeze(1)
        dot_mu_v = (M * V_star).sum(dim=1)
        diff = proj - dot_mu_v[assign]
        var_buf = torch.zeros(m, device=X.device, dtype=torch.float32).scatter_add_(0, assign, (w_pt * diff * diff).float())
        span = 2.0 * (4.0 * torch.sqrt((var_buf / w_sum.float()).clamp(min=1e-12))).clamp(min=1e-4)
        lo_v = dot_mu_v.float() - 0.5 * span

        pos = ((proj.float() - lo_v[assign]) / span[assign] * (num_bins - 1)).clamp(0.0, float(num_bins - 1) - 1e-4)
        b0 = pos.floor().long()
        b1 = (b0 + 1).clamp(max=num_bins - 1)
        frac = pos - b0.float()

        flat_base = is_B.long() * (m * num_bins) + assign * num_bins
        hist_ab = torch.zeros(2 * m * num_bins, device=X.device, dtype=torch.float32)
        hist_ab.scatter_add_(0, flat_base + b0, (1.0 - frac) * w_pt.float())
        hist_ab.scatter_add_(0, flat_base + b1, frac * w_pt.float())

        h_bins = 0.65 * sigma_bins * (0.5 * total_cnt.float()).clamp(min=2.0).pow(-0.2)
        h2 = torch.clamp(h_bins * h_bins, min=0.05).view(m, 1, 1)
        K_kernel = torch.exp(-0.5 * dist2.unsqueeze(0) / h2)
        K_kernel = K_kernel / K_kernel.sum(dim=1, keepdim=True).clamp(min=1e-6)
        dens_ab = torch.bmm(K_kernel, hist_ab.view(2, m, num_bins).permute(1, 2, 0))
        dens_A, dens_B = dens_ab[:, :, 0], dens_ab[:, :, 1]

        lp_A = torch.cummax(dens_A, dim=1).values
        rp_A = torch.cummax(dens_A.flip(1), dim=1).values.flip(1)
        lp_B = torch.cummax(dens_B, dim=1).values
        rp_B = torch.cummax(dens_B.flip(1), dim=1).values.flip(1)
        val_persist = torch.minimum(torch.clamp(torch.minimum(lp_A, rp_A) - dens_A, min=0.0), torch.clamp(torch.minimum(lp_B, rp_B) - dens_B, min=0.0))
        dens_mean = 0.5 * (dens_A + dens_B)
        cdf = torch.cumsum(dens_mean, dim=1) / dens_mean.sum(dim=1, keepdim=True).clamp(min=1e-6)
        v_max, v_idx = torch.max(val_persist * (4.0 * cdf * (1.0 - cdf)), dim=1)

        df_A = dens_A[:, 1:] - dens_A[:, :-1]
        df_B = dens_B[:, 1:] - dens_B[:, :-1]
        agree = (df_A * df_B).sum(dim=1) / ((df_A.norm(dim=1) * df_B.norm(dim=1)) + 1e-8)
        has_valley = (v_max > 0.03 * torch.minimum(dens_A, dens_B).max(dim=1).values) & (agree > 0.35)
        thresh = torch.where(has_valley, lo_v + (v_idx.float() / (num_bins - 1)) * span, dot_mu_v.float()).to(X.dtype)

        left_f = (proj < thresh[assign]).to(X.dtype).unsqueeze(1)
        Ow_left = O_w * left_f
        Ow_right = O_w - Ow_left
        wl, wr = Ow_left.sum(dim=0), Ow_right.sum(dim=0)
        C_left = (Ow_left.T @ X_sub) / wl.clamp(min=1e-6).unsqueeze(1)
        C_right = (Ow_right.T @ X_sub) / wr.clamp(min=1e-6).unsqueeze(1)

        valid = ((O * left_f).sum(dim=0) >= 1.0) & ((total_cnt - (O * left_f).sum(dim=0)) >= 1.0)
        delta_sse = (wl * wr / (wl + wr).clamp(min=1e-6)) * torch.linalg.vector_norm(C_left - C_right, dim=1).square()
        ranking_score = torch.where(valid, delta_sse * torch.where(has_valley, valley_boost, 1.0), torch.zeros_like(delta_sse))

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


# =====================================================================
# 3. Master Multi-Task Diagnostic Visualizer
# =====================================================================

@torch.no_grad()
def generate_all_diagnostics(out_dir="diagnostics_all_tasks"):
    os.makedirs(out_dir, exist_ok=True)
    task_generators = [
        make_task_1a_truth,
        make_task_1b_truth,
        make_task_2_truth,
        make_task_3_truth,
        make_task_4_truth,
        make_task_5_truth,
        make_task_7_truth,
    ]

    print(f"Hardware: {device.type.upper()} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Generating full diagnostic suites to -> {out_dir}/ ...\n")

    summary_rows = []

    for fn in task_generators:
        X, labels, true_c, floor_mse, title = fn()
        K = 16 if "1a" in title else 64
        K_truth = true_c.shape[0]

        # Run Engine v11-aware
        C_learned = run_engine_v11_production(X, K)

        # Exact Full-Dataset MSE
        chunk_size = 32768
        sq_err = 0.0
        for i in range(0, X.shape[0], chunk_size):
            sq_err += torch.cdist(X[i : i + chunk_size], C_learned).min(dim=1).values.square().sum().item()
        actual_mse = sq_err / X.shape[0]

        # Ground-Truth Center-Space Projection (Invariant to Point Cloud Clutter)
        mu_c = true_c.mean(dim=0, keepdim=True)
        _, _, V_space = torch.pca_lowrank(true_c - mu_c, q=2, center=False)

        truth_2d = ((true_c - mu_c) @ V_space).cpu().numpy()
        learned_2d = ((C_learned - mu_c) @ V_space).cpu().numpy()

        # Point subsample for backdrop
        sub_idx = torch.randint(0, X.shape[0], (min(4000, X.shape[0]),), device=device)
        pts_2d = ((X[sub_idx] - mu_c) @ V_space).cpu().numpy()

        # Capture Analysis (Voronoi Matching)
        dists = torch.cdist(C_learned, true_c)
        nearest = dists.argmin(dim=1).cpu().numpy()
        captures = np.bincount(nearest, minlength=K_truth)

        orphans = np.where(captures == 0)[0]
        n_orphans = len(orphans)
        gap_pct = ((actual_mse - floor_mse) / floor_mse) * 100.0

        summary_rows.append({
            "title": title, "actual": actual_mse, "floor": floor_mse, "gap": gap_pct, "orphans": n_orphans,
        })

        # Render Diagnostics
        fig = plt.figure(figsize=(16, 9), dpi=160)
        gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.2], hspace=0.32)

        ax_map = fig.add_subplot(gs[0])
        ax_bar = fig.add_subplot(gs[1])

        # 1. Map Panel
        ax_map.scatter(pts_2d[:, 0], pts_2d[:, 1], c="#e0e0e0", s=6, alpha=0.3, label="Data Points (Sample)", rasterized=True)
        ax_map.scatter(truth_2d[:, 0], truth_2d[:, 1], s=90, marker="o", facecolors="none", edgecolors="#2b5c8f", linewidths=1.8, label=f"True Centers ({K_truth})", zorder=5)

        if n_orphans > 0:
            ax_map.scatter(truth_2d[orphans, 0], truth_2d[orphans, 1], s=210, marker="o", facecolors="none", edgecolors="red", linewidths=2.2, label=f"Orphaned Clusters ({n_orphans})", zorder=7)

        ax_map.scatter(learned_2d[:, 0], learned_2d[:, 1], s=110, marker="X", c="gold", edgecolors="black", linewidths=1.1, label=f"Learned Centroids ({K})", zorder=10)

        badge = (
            f"Actual MSE: {actual_mse:.3f}\n"
            f"Bayes Floor: {floor_mse:.3f}\n"
            f"Gap: +{gap_pct:.2f}%\n"
            f"Orphaned Modes: {n_orphans}/{K_truth}"
        )
        ax_map.text(
            0.02, 0.96, badge, transform=ax_map.transAxes, va="top", ha="left", fontsize=9,
            fontweight="bold", bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.92, edgecolor="#999999"), zorder=12,
        )

        ax_map.set_title(f"{title} — Center-Space 2D Subspace & Capture Geometry", fontsize=11, fontweight="bold", pad=8)
        ax_map.set_xticks([])
        ax_map.set_yticks([])
        ax_map.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax_map.grid(True, linestyle=":", alpha=0.3)

        # 2. Bar Panel
        x_bars = np.arange(K_truth)
        ax_bar.bar(x_bars, captures, color="#4a90e2", width=0.65, label="Allocated Centroids", alpha=0.85)
        ideal_h = 1.0 if K == K_truth else (K / K_truth)
        ax_bar.axhline(ideal_h, color="red", linestyle="--", linewidth=1.2, label=f"Ideal Bayes Allocation ({ideal_h:.1f})")

        ax_bar.set_xlim(-1, K_truth)
        ax_bar.set_ylim(0, max(max(captures), int(ideal_h) + 1) + 0.8)
        ax_bar.set_xticks(x_bars)
        tick_labels = [f"C{i+1}" for i in range(K_truth)]
        ax_bar.set_xticklabels(tick_labels, fontsize=6.5, rotation=90)
        ax_bar.set_xlabel("Ground-Truth Cluster ID", fontsize=8.5, fontweight="bold")
        ax_bar.set_ylabel("Centroids Assigned", fontsize=8.5, fontweight="bold")
        ax_bar.set_title("Centroid Allocation per Ground-Truth Cluster", fontsize=9.5, fontweight="bold")
        ax_bar.grid(True, axis="y", linestyle=":", alpha=0.4)
        ax_bar.legend(loc="upper right", fontsize=8)

        slug = title.split(":")[0].lower().replace(" ", "_")
        out_fig = os.path.join(out_dir, f"diag_{slug}.png")
        fig.savefig(out_fig, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_fig}")

    # Print Summary Table
    print("\n" + "=" * 98)
    print(f"{'Task':<46} | {'Actual MSE':<12} | {'Floor MSE':<12} | {'Gap (%)':<10} | {'Orphans'}")
    print("=" * 98)
    for r in summary_rows:
        print(f"{r['title']:<46} | {r['actual']:<12.3f} | {r['floor']:<12.3f} | {r['gap']:+9.2f}% | {r['orphans']}")
    print("=" * 98)


if __name__ == "__main__":
    generate_all_diagnostics()