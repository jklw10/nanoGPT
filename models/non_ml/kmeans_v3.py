import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync_time(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


# =====================================================================
# 1. Naive vs. Aware (Split-Half + prev_V + Delta-SSE) Implementations
# =====================================================================

def run_v10_aware(
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


def run_v10_naive(X: torch.Tensor, k: int) -> torch.Tensor:
    n, d = X.shape
    sub_n = min(16384, n)
    X_sub = X[torch.randint(0, n, (sub_n,), device=X.device)]
    x_sub_sq = (X_sub * X_sub).sum(dim=1)
    C_buf = torch.empty((k, d), device=X.device, dtype=X.dtype)
    C_buf[0] = X_sub.mean(dim=0)
    m = 1
    while m < k:
        b = min(m, max(1, (2 * m + 2) // 3), k - m)
        C_curr = C_buf[:m]
        c_half = 0.5 * (C_curr * C_curr).sum(dim=1)
        max_score, assign = torch.max(torch.addmm(-c_half, X_sub, C_curr.T), dim=1)
        pt_sse = torch.clamp(x_sub_sq - 2.0 * max_score, min=0.0).float()
        pt_sse = torch.clamp(pt_sse, max=torch.quantile(pt_sse, 0.95))
        csse = torch.zeros(m, device=X.device).scatter_add_(0, assign, pt_sse)
        _, split_ids = torch.topk(csse, b, largest=True)
        c2s = torch.full((m,), b, device=X.device, dtype=torch.long)
        c2s[split_ids] = torch.arange(b, device=X.device)
        sel = c2s[assign]; safe = sel.clamp(max=b - 1); in_s = (sel < b).to(X.dtype)
        O = torch.zeros(sub_n, b + 1, device=X.device, dtype=X.dtype).scatter_(1, sel.unsqueeze(1), 1.0)[:, :b]
        cnt = O.sum(dim=0)
        M = (O.T @ X_sub) / cnt.clamp(min=1.0).unsqueeze(1)
        V = torch.randn(b, d, device=X.device, dtype=X.dtype); V /= V.norm(dim=1, keepdim=True) + 1e-9
        for _ in range(2):
            z = ((X_sub @ V.T).gather(1, safe.unsqueeze(1)).squeeze(1) - (M * V).sum(dim=1)[safe]) * in_s
            Z = O * z.unsqueeze(1)
            V = Z.T @ X_sub - Z.sum(dim=0).unsqueeze(1) * M
            V /= V.norm(dim=1, keepdim=True) + 1e-9
        proj = (X_sub @ V.T).gather(1, safe.unsqueeze(1)).squeeze(1)
        thresh = (M * V).sum(dim=1)
        left_f = ((proj < thresh[safe]) & (sel < b)).to(X.dtype).unsqueeze(1)
        Ol, Or = O * left_f, O - (O * left_f)
        cl, cr = Ol.sum(dim=0), Or.sum(dim=0)
        valid = (cl >= 1.0) & (cr >= 1.0)
        C_buf[split_ids] = torch.where(valid.unsqueeze(1), (Ol.T @ X_sub) / cl.clamp(min=1.0).unsqueeze(1), C_curr[split_ids])
        C_buf[m : m + b] = torch.where(valid.unsqueeze(1), (Or.T @ X_sub) / cr.clamp(min=1.0).unsqueeze(1), X_sub[torch.arange(b, device=X.device) % sub_n])
        m += b
    return C_buf


# =====================================================================
# 2. Plot 1: 1D Split-Half Persistence vs. Naive on Uniform & Gaussians
# =====================================================================

def plot_1d_persistence_vs():
    torch.manual_seed(7)
    u_proj = torch.cat([torch.rand(1500) * 7.0 - 9.0, torch.rand(1500) * 7.0 + 2.0])
    g_centers = torch.tensor([-12.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0])
    g_proj = torch.cat([c + torch.randn(400) * 0.42 for c in g_centers])

    fig, axes = plt.subplots(2, 2, figsize=(14, 7.2))
    fig.suptitle("Resampling Awareness: Naive Single-Sample vs. Split-Half Persistent Peaks & Pits", fontsize=13, fontweight="bold")

    for row, (name, proj) in enumerate([("Two Uniform Clusters U[-9,-2] & U[2,9]", u_proj), ("8 Skewed Gaussian Blobs", g_proj)]):
        # Naive full-sample uniform-x derivative
        p, _ = torch.sort(proj)
        n = p.numel()
        r = 8
        idx = torch.arange(n)
        avg_d = (p[torch.clamp(idx + r, max=n - 1)] - p[torch.clamp(idx - r, min=0)]).clamp(min=1e-4) / (torch.clamp(idx + r, max=n - 1) - torch.clamp(idx - r, min=0)).float()
        pdf_pts = (1.0 / avg_d) / n

        M = 512
        xg = torch.linspace(p[0], p[-1], M)
        dx = (xg[-1] - xg[0]) / (M - 1)
        ir = torch.searchsorted(p, xg).clamp(1, n - 1); il = ir - 1
        alpha = ((xg - p[il]) / (p[ir] - p[il]).clamp(min=1e-9)).clamp(0, 1)
        fg = (1 - alpha) * pdf_pts[il] + alpha * pdf_pts[ir]
        kx = torch.arange(-12, 13, dtype=torch.float32)
        kg = (torch.exp(-0.5 * (kx / 4.0) ** 2)); kg = (kg / kg.sum()).view(1, 1, -1)
        fg_sm = F.conv1d(fg.view(1, 1, -1), kg, padding=12).view(-1)
        d2 = torch.zeros_like(fg_sm); d2[1:-1] = (fg_sm[2:] - fg_sm[:-2]) / (2 * dx)
        sig = fg_sm > 0.05 * fg_sm.max()
        naive_peaks = torch.where((d2[:-1] > 0) & (d2[1:] <= 0) & sig[:-1])[0]
        naive_pits = torch.where((d2[:-1] < 0) & (d2[1:] >= 0) & sig[:-1])[0]

        ax_n = axes[row, 0]
        ax_n.fill_between(xg.numpy(), fg_sm.numpy(), alpha=0.25, color="gray")
        ax_n.plot(xg.numpy(), fg_sm.numpy(), lw=1.6, color="black")
        ax_n.scatter(xg[naive_peaks].numpy(), fg_sm[naive_peaks].numpy(), s=45, color="limegreen", edgecolors="black", zorder=5, label=f"Peaks ({len(naive_peaks)})")
        ax_n.scatter(xg[naive_pits].numpy(), fg_sm[naive_pits].numpy(), s=45, marker="D", color="crimson", edgecolors="black", zorder=5, label=f"Pits ({len(naive_pits)})")
        ax_n.set_title(f"{name}\nNaive: {len(naive_peaks)} peaks, {len(naive_pits)} pits", fontsize=10, fontweight="bold")
        ax_n.legend(fontsize=8); ax_n.grid(True, alpha=0.2)

        # Split-half persistent KDE on 64 bins
        perm = torch.randperm(n)
        half_A, half_B = proj[perm[: n // 2]], proj[perm[n // 2 :]]
        lo, span = proj.mean() - 4 * proj.std(), 8 * proj.std()
        num_bins = 64
        idx_b = torch.arange(num_bins, dtype=torch.float32)
        dist2 = (idx_b.unsqueeze(0) - idx_b.unsqueeze(1)) ** 2
        h_bins = 0.65 * (63.0 / 8.0) * ((n / 2) ** -0.2)
        K = torch.exp(-0.5 * dist2 / (h_bins ** 2)); K /= K.sum(dim=1, keepdim=True)

        def bin_kde(s):
            pos = ((s - lo) / span * 63.0).clamp(0.0, 62.9999)
            b0 = pos.floor().long(); frac = pos - b0.float()
            lin = torch.zeros(num_bins)
            lin.scatter_add_(0, b0, 1.0 - frac)
            lin.scatter_add_(0, (b0 + 1).clamp(max=63), frac)
            return K @ lin

        dA, dB = bin_kde(half_A), bin_kde(half_B)
        lpA, rpA = torch.cummax(dA, 0).values, torch.cummax(dA.flip(0), 0).values.flip(0)
        lpB, rpB = torch.cummax(dB, 0).values, torch.cummax(dB.flip(0), 0).values.flip(0)
        val_p = torch.minimum((torch.minimum(lpA, rpA) - dA).clamp(min=0), (torch.minimum(lpB, rpB) - dB).clamp(min=0))
        xs_b = torch.linspace(lo, lo + span, num_bins)
        dfA, dfB = dA[1:] - dA[:-1], dB[1:] - dB[:-1]
        agree = ((dfA * dfB).sum() / (dfA.norm() * dfB.norm() + 1e-8)).item()

        # Only accept pits where BOTH halves have > 3% depth
        thresh_v = 0.03 * torch.minimum(dA, dB).max()
        persist_pits = torch.where((val_p[1:-1] > thresh_v) & (val_p[1:-1] >= val_p[:-2]) & (val_p[1:-1] >= val_p[2:]))[0] + 1

        ax_a = axes[row, 1]
        ax_a.plot(xs_b.numpy(), dA.numpy(), lw=1.6, color="steelblue", label="Half A")
        ax_a.plot(xs_b.numpy(), dB.numpy(), lw=1.6, ls="--", color="darkorange", label="Half B")
        ax_a.fill_between(xs_b.numpy(), torch.minimum(dA, dB).numpy(), alpha=0.20, color="teal", label="Persistent overlap")
        if len(persist_pits) > 0:
            ax_a.scatter(xs_b[persist_pits].numpy(), torch.minimum(dA, dB)[persist_pits].numpy(),
                         s=70, marker="D", color="crimson", edgecolors="black", zorder=6, label=f"Persistent pits ({len(persist_pits)})")
        ax_a.set_title(f"Split-Half Aware (df_A·df_B corr={agree:.2f})\nPersistent pits = {len(persist_pits)} (ripples rejected!)", fontsize=10, fontweight="bold")
        ax_a.legend(fontsize=8); ax_a.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig("vis_aware_1_split_half_persistence.png", dpi=140)
    print("Saved vis_aware_1_split_half_persistence.png")


# =====================================================================
# 3. Plot 2: 2D Versus (Naive v10 vs. Aware v10) with Timings
# =====================================================================

def eval_2d(X, C, core_mask):
    c_half = 0.5 * (C * C).sum(dim=1)
    x_sq = (X * X).sum(dim=1)
    max_sc, assign = torch.max(torch.addmm(-c_half, X, C.T), dim=1)
    mse = torch.clamp(x_sq - 2.0 * max_sc, min=0.0)[core_mask].mean().item()
    return mse, assign


def plot_2d_versus():
    torch.manual_seed(7)
    # Task A: Skewed Blobs
    cA = torch.tensor([[-12.0, 0], [1, 2.2], [3, -2.2], [5, 2.2], [7, -2.2], [9, 2.2], [11, -2.2], [13, 0]])
    XA = torch.cat([c + torch.randn(400, 2) * 0.42 for c in cA])
    mA = torch.ones(XA.shape[0], dtype=torch.bool)

    # Task B: Grid + Outliers
    cB = torch.tensor([[-6, -4.], [-6, 4], [-2, -4], [-2, 4], [2, -4], [2, 4], [6, -4], [6, 4]])
    XB = torch.cat([c + torch.randn(350, 2) * 0.55 for c in cB] + [torch.tensor([[110., 90.], [114., 86.]])])
    mB = torch.cat([torch.ones(2800, dtype=torch.bool), torch.zeros(2, dtype=torch.bool)])

    # Task C: Uniform Squares (2 clusters, k=2)
    XC = torch.cat([torch.rand(1200, 2) * 5.0 - torch.tensor([7.0, 2.5]),
                    torch.rand(1200, 2) * 5.0 + torch.tensor([2.0, -2.5])])
    mC = torch.ones(XC.shape[0], dtype=torch.bool)

    tasks = [
        ("Skewed 8 Blobs (k=8)", XA, mA, 8, (-15, 16), (-6, 6)),
        ("8 Blobs + 2 Outliers (k=8)", XB, mB, 8, (-10, 14), (-8, 8)),
        ("2 Uniform Squares (k=2)", XC, mC, 2, (-9, 9), (-5, 5)),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(12.5, 11.5))
    fig.suptitle("2D Versus: Naive v10 vs. Aware v10 (Split-Half + prev_V Rotation + ΔSSE)", fontsize=13, fontweight="bold")
    cmap = cm.get_cmap("tab10")

    for row, (title, X, cmask, k, xl, yl) in enumerate(tasks):
        for col, (aname, fn) in enumerate([("Naive v10", run_v10_naive), ("Aware v10", run_v10_aware)]):
            torch.manual_seed(123)
            t0 = sync_time(X.device)
            C = fn(X, k)
            dt_ms = (sync_time(X.device) - t0) * 1000.0
            mse, asg = eval_2d(X, C, cmask)

            ax = axes[row, col]
            for cid in range(k):
                pts = X[(asg == cid) & cmask]
                if pts.numel() > 0:
                    ax.scatter(pts[:, 0], pts[:, 1], s=7, alpha=0.55, color=cmap(cid % 10))
            ax.scatter(C[:, 0], C[:, 1], s=110, marker="*", color="yellow", edgecolors="black", zorder=6)
            ax.set_title(f"{title} — {aname}\nCore MSE={mse:.3f} | Time={dt_ms:.2f} ms", fontsize=10, fontweight="bold")
            ax.set_xlim(xl); ax.set_ylim(yl); ax.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("vis_aware_2_2d_versus.png", dpi=140)
    print("Saved vis_aware_2_2d_versus.png")


# =====================================================================
# 4. 1024D Timing & MSE Comparison (N=64,000, D=1024, K=64)
# =====================================================================

def bench_1024d():
    print(f"\n=== 1024D Benchmark (N=64,000 | D=1024 | K=64 | device={device}) ===")
    N, D, K = 64000, 1024, 64

    def make_t1():
        X = torch.randn(N, D, device=device)
        centers = torch.randn(K, D, device=device) * 4.0
        ids = torch.randint(0, K, (N,), device=device)
        X.add_(centers[ids])
        outs = torch.randn(20, D, device=device) * 35.0
        return torch.cat([X, outs], dim=0), torch.cat([torch.ones(N, dtype=torch.bool, device=device), torch.zeros(20, dtype=torch.bool, device=device)])

    def make_t5():
        X = torch.empty((N, D), device=device)
        ng = int(0.75 * N); g_pts = ng // 2
        for g in range(2):
            c = torch.randn(D, device=device) * 4.0
            s, e = g * g_pts, (g + 1) * g_pts if g == 0 else ng
            torch.randn(e - s, D, device=device, out=X[s:e]).mul_(3.2).add_(c)
        nd = (N - ng) // (K - 2)
        for i in range(K - 2):
            c = torch.randn(D, device=device) * 4.0
            s = ng + i * nd; e = ng + (i + 1) * nd if i < K - 3 else N
            torch.randn(e - s, D, device=device, out=X[s:e]).mul_(0.6).add_(c)
        return X[torch.randperm(N, device=device)], torch.ones(N, dtype=torch.bool, device=device)

    for tname, maker in [("Task 1b (Blobs + Outliers)", make_t1), ("Task 5 (2 Giants + 62 Dwarfs)", make_t5)]:
        torch.manual_seed(42)
        X, cmask = maker()
        # Warmup
        _ = run_v10_naive(X, K)
        _ = run_v10_aware(X, K)

        print(f"\n--- {tname} ---")
        for aname, fn in [("v10 Naive (No-Island)", run_v10_naive), ("v10 Aware (SplitHalf+prevV+dSSE)", run_v10_aware)]:
            torch.manual_seed(99)
            t0 = sync_time(device)
            C = fn(X, K)
            dt_ms = (sync_time(device) - t0) * 1000.0
            c_half = 0.5 * (C * C).sum(dim=1)
            x_sq = (X * X).sum(dim=1)
            max_sc = torch.max(torch.addmm(-c_half, X, C.T), dim=1).values
            core_mse = torch.clamp(x_sq - 2.0 * max_sc, min=0.0)[cmask].mean().item()
            print(f"  {aname:<34} | {dt_ms:7.2f} ms | Core MSE = {core_mse:10.2f}")


if __name__ == "__main__":
    plot_1d_persistence_vs()
    plot_2d_versus()
    bench_1024d()
    plt.show()