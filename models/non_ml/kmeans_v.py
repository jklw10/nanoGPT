import math
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm

torch.manual_seed(7)


# =====================================================================
# 1. Synthetic 2D Tasks
# =====================================================================

def make_skewed_blobs_2d(pts_per_blob: int = 400, std: float = 0.42):
    centers = torch.tensor([
        [-12.0,  0.0],
        [  1.0,  2.2], [  3.0, -2.2], [  5.0,  2.2],
        [  7.0, -2.2], [  9.0,  2.2], [ 11.0, -2.2], [ 13.0,  0.0],
    ], dtype=torch.float32)
    X_list, y_list = [], []
    for cid, c in enumerate(centers):
        X_list.append(c + torch.randn(pts_per_blob, 2) * std)
        y_list.append(torch.full((pts_per_blob,), cid, dtype=torch.long))
    return torch.cat(X_list, dim=0), torch.cat(y_list, dim=0)


def make_outlier_blobs_2d(pts_per_blob: int = 350, std: float = 0.55):
    centers = torch.tensor([
        [-6.0, -4.0], [-6.0,  4.0], [-2.0, -4.0], [-2.0,  4.0],
        [ 2.0, -4.0], [ 2.0,  4.0], [ 6.0, -4.0], [ 6.0,  4.0],
    ], dtype=torch.float32)
    X_list, y_list = [], []
    for cid, c in enumerate(centers):
        X_list.append(c + torch.randn(pts_per_blob, 2) * std)
        y_list.append(torch.full((pts_per_blob,), cid, dtype=torch.long))
    outliers = torch.tensor([[110.0, 90.0], [114.0, 86.0]], dtype=torch.float32)
    X_list.append(outliers)
    y_list.append(torch.full((2,), -1, dtype=torch.long))
    return torch.cat(X_list, dim=0), torch.cat(y_list, dim=0)


# =====================================================================
# 2. FFT-Free Gaussian KDE (bmm) & Orthonormal DCT-II Helpers
# =====================================================================

def build_dct_matrix(num_bins: int, device=None, dtype=torch.float32) -> torch.Tensor:
    k = torch.arange(num_bins, device=device, dtype=dtype).unsqueeze(1)
    n = torch.arange(num_bins, device=device, dtype=dtype).unsqueeze(0)
    C = torch.cos(math.pi * k * (n + 0.5) / float(num_bins))
    C[0, :] *= math.sqrt(1.0 / float(num_bins))
    C[1:, :] *= math.sqrt(2.0 / float(num_bins))
    return C  # (N, N), C @ C.T == I


def apply_bmm_gaussian_kde(
    density: torch.Tensor,     # (b, num_bins)
    dist2: torch.Tensor,       # (num_bins, num_bins)
    h_bins: torch.Tensor,      # (b,)
) -> torch.Tensor:
    h2 = torch.clamp(h_bins * h_bins, min=0.05).view(-1, 1, 1)
    K = torch.exp(-0.5 * dist2.unsqueeze(0) / h2)                 # (b, N, N)
    K = K / K.sum(dim=1, keepdim=True).clamp(min=1e-6)            # preserve mass
    return torch.bmm(K, density.unsqueeze(2)).squeeze(2)          # (b, N)


def apply_dct_gaussian_filter(
    density: torch.Tensor,     # (b, num_bins)
    C_dct: torch.Tensor,       # (num_bins, num_bins)
    h_bins: torch.Tensor,      # (b,)
):
    num_bins = density.shape[1]
    spec = density @ C_dct.T                                      # (b, N)
    ks = torch.arange(num_bins, device=density.device, dtype=density.dtype)
    omega = math.pi * ks / float(num_bins)                        # rad / bin
    filt = torch.exp(-0.5 * (omega.unsqueeze(0) ** 2) * (h_bins.unsqueeze(1) ** 2))
    filtered = (spec * filt) @ C_dct
    return filtered.clamp(min=0.0), spec


# =====================================================================
# 3. Traced v10 with FFT-Free KDE + SSE Gate + Subsample Refresh
# =====================================================================

def run_v10_kde_traced(
    X: torch.Tensor,
    y_true: torch.Tensor,
    k: int,
    kde_type: str = "bmm_kde",   # "bmm_kde", "dct_kde", "linear"
    bw_scale: float = 0.65,
    use_sse_gate: bool = True,
    refresh_final: bool = True,
    seed: int = 123,
):
    torch.manual_seed(seed)
    n, d = X.shape
    x_sq = (X * X).sum(dim=1)
    C_buf = torch.empty((k, d), dtype=X.dtype)
    C_buf[0] = X.mean(dim=0)

    num_bins = 64
    idx = torch.arange(num_bins, dtype=torch.float32)
    dist2 = (idx.unsqueeze(0) - idx.unsqueeze(1)) ** 2
    C_dct = build_dct_matrix(num_bins)
    sigma_bins = float(num_bins - 1) / 8.0

    history = []
    m = 1
    round_idx = 0
    while m < k:
        round_idx += 1
        b_sched = min(m, max(1, (2 * m + 2) // 3), k - m)

        C_curr = C_buf[:m]
        c_half = 0.5 * (C_curr * C_curr).sum(dim=1)
        scores = torch.addmm(-c_half, X, C_curr.T)
        max_score, assign = torch.max(scores, dim=1)

        raw_sse = torch.clamp(x_sq - 2.0 * max_score, min=0.0)
        cap = torch.quantile(raw_sse, 0.95)
        pt_sse = torch.clamp(raw_sse, max=cap)
        w_pt = torch.clamp(cap / torch.clamp(raw_sse, min=1e-6), max=1.0)

        cluster_sse = torch.zeros(m)
        cluster_sse.scatter_add_(0, assign, pt_sse)

        if use_sse_gate:
            active_cnt = int((cluster_sse >= 0.15 * cluster_sse.max()).sum().item())
            b = max(1, min(b_sched, active_cnt))
        else:
            b = b_sched

        _, split_ids = torch.topk(cluster_sse, b, largest=True)

        cluster_to_sel = torch.full((m,), b, dtype=torch.long)
        cluster_to_sel[split_ids] = torch.arange(b)
        sel_pos = cluster_to_sel[assign]
        safe_pos = sel_pos.clamp(max=b - 1)
        in_sel = (sel_pos < b).float()

        O_full = torch.zeros(n, b + 1)
        O_full.scatter_(1, sel_pos.unsqueeze(1), 1.0)
        O = O_full[:, :b]
        O_w = O * w_pt.unsqueeze(1)

        total_cnt = O.sum(dim=0)
        w_sum = O_w.sum(dim=0).clamp(min=1e-6)
        M = (O_w.T @ X) / w_sum.unsqueeze(1)

        V = torch.randn(b, d)
        V = V / (V.norm(dim=1, keepdim=True) + 1e-9)
        for _ in range(2):
            dot_x_v = (X @ V.T).gather(1, safe_pos.unsqueeze(1)).squeeze(1) * in_sel
            dot_mu_v = (M * V).sum(dim=1)
            z = (dot_x_v - dot_mu_v[safe_pos]) * in_sel
            Z_mat = O_w * z.unsqueeze(1)
            V_next = Z_mat.T @ X - (Z_mat.sum(dim=0).unsqueeze(1)) * M
            V = V_next / (V_next.norm(dim=1, keepdim=True) + 1e-9)

        proj = (X @ V.T).gather(1, safe_pos.unsqueeze(1)).squeeze(1)
        dot_mu_v = (M * V).sum(dim=1)

        diff = (proj - dot_mu_v[safe_pos]) * in_sel
        var_buf = torch.zeros(b + 1)
        var_buf.scatter_add_(0, sel_pos, w_pt * diff * diff)
        var = var_buf[:b] / w_sum
        half_span = (4.0 * torch.sqrt(var.clamp(min=1e-12))).clamp(min=1e-4)
        lo_v = dot_mu_v - half_span
        span = 2.0 * half_span

        # Linear binning
        pos = ((proj - lo_v[safe_pos]) / span[safe_pos] * (num_bins - 1)).clamp(0.0, float(num_bins - 1) - 1e-4)
        b0 = pos.floor().long()
        b1 = (b0 + 1).clamp(max=num_bins - 1)
        frac = pos - b0.float()
        in_sel_w = in_sel * w_pt

        hist = torch.zeros((b + 1) * num_bins)
        hist.scatter_add_(0, sel_pos * num_bins + b0, (1.0 - frac) * in_sel_w)
        hist.scatter_add_(0, sel_pos * num_bins + b1, frac * in_sel_w)
        raw_linear = hist[: b * num_bins].view(b, num_bins)

        # Per-cluster Silverman bandwidth in bin units
        h_bins = bw_scale * sigma_bins * total_cnt.clamp(min=2.0).pow(-0.2)
        dct_density, spec = apply_dct_gaussian_filter(raw_linear, C_dct, h_bins)
        bmm_density = apply_bmm_gaussian_kde(raw_linear, dist2, h_bins)

        if kde_type == "bmm_kde":
            density = bmm_density
        elif kde_type == "dct_kde":
            density = dct_density
        else:
            density = raw_linear

        # Valley cut on smooth KDE density (fallback to mean when unimodal)
        left_peak = torch.cummax(density, dim=1).values
        right_peak = torch.cummax(density.flip(1), dim=1).values.flip(1)
        valley_depth = torch.clamp(torch.minimum(left_peak, right_peak) - density, min=0.0)
        cdf = torch.cumsum(density, dim=1) / density.sum(dim=1, keepdim=True).clamp(min=1e-6)
        v_score = valley_depth * (4.0 * cdf * (1.0 - cdf))
        v_max, v_idx = torch.max(v_score, dim=1)
        v_cand = lo_v + (v_idx.float() / (num_bins - 1)) * span
        thresh = torch.where(v_max > (0.02 * density.max(dim=1).values), v_cand, dot_mu_v)

        is_left = (proj < thresh[safe_pos]) & (sel_pos < b)
        cnt_left = (O * is_left.float().unsqueeze(1)).sum(dim=0)
        cnt_right = total_cnt - cnt_left
        thresh = torch.where((cnt_left < 1.0) | (cnt_right < 1.0), dot_mu_v, thresh)

        is_left = (proj < thresh[safe_pos]) & (sel_pos < b)
        left_f = is_left.float().unsqueeze(1)
        Ow_left = O_w * left_f
        Ow_right = O_w - Ow_left
        cnt_left = (O * left_f).sum(dim=0)
        cnt_right = total_cnt - cnt_left

        history.append({
            "round": round_idx, "m": m, "b": b,
            "C_curr": C_curr.clone(), "assign": assign.clone(), "split_ids": split_ids.clone(),
            "M": M.clone(), "V": V.clone(), "lo_v": lo_v.clone(), "span": span.clone(),
            "raw_linear": raw_linear.clone(), "bmm_density": bmm_density.clone(),
            "dct_density": dct_density.clone(), "spec": spec.clone(), "h_bins": h_bins.clone(),
            "thresh": thresh.clone(), "dot_mu_v": dot_mu_v.clone(),
            "cnt_left": cnt_left.clone(), "cnt_right": cnt_right.clone(),
        })

        C_left = (Ow_left.T @ X) / Ow_left.sum(dim=0).clamp(min=1e-6).unsqueeze(1)
        C_right = (Ow_right.T @ X) / Ow_right.sum(dim=0).clamp(min=1e-6).unsqueeze(1)
        valid = (cnt_left >= 1.0) & (cnt_right >= 1.0)
        C_buf[split_ids] = torch.where(valid.unsqueeze(1), C_left, C_curr[split_ids])
        C_buf[m : m + b] = torch.where(valid.unsqueeze(1), C_right, X[torch.arange(b) % n])
        m += b

    C_final = C_buf[:k]

    # End-of-seeder centroid refresh on X_sub (fixes stranded centroids like Demo 2b!)
    if refresh_final:
        c_half = 0.5 * (C_final * C_final).sum(dim=1)
        max_sc, assign_fin = torch.max(torch.addmm(-c_half, X, C_final.T), dim=1)
        raw_sse_fin = torch.clamp(x_sq - 2.0 * max_sc, min=0.0)
        cap_fin = torch.quantile(raw_sse_fin, 0.95)
        w_fin = torch.clamp(cap_fin / torch.clamp(raw_sse_fin, min=1e-6), max=1.0)
        O_fin = torch.zeros(n, k)
        O_fin.scatter_(1, assign_fin.unsqueeze(1), 1.0)
        Ow_fin = O_fin * w_fin.unsqueeze(1)
        w_tot = Ow_fin.sum(dim=0)
        C_new = (Ow_fin.T @ X) / w_tot.clamp(min=1e-6).unsqueeze(1)
        C_final = torch.where((w_tot > 1e-6).unsqueeze(1), C_new, C_final)

    core_mask = y_true >= 0
    dists = torch.cdist(X, C_final)
    min_sq, final_assign = torch.min(dists * dists, dim=1)
    core_mse = min_sq[core_mask].mean().item()

    blob_to_centroids = {
        bid: torch.unique(final_assign[y_true == bid]).tolist()
        for bid in torch.unique(y_true[core_mask]).tolist()
    }
    merged = [b_id for b_id, cs in blob_to_centroids.items()
              if len(cs) == 1 and sum(1 for cs2 in blob_to_centroids.values() if cs[0] in cs2) > 1]
    return history, C_final, final_assign, core_mse, merged


# =====================================================================
# 4. Visualizations
# =====================================================================

def plot_1d_density_comparison(X1, y1):
    """Compare Hard Hist, Linear, 3-tap Blur, bmm Gaussian KDE, DCT KDE + DCT Spectrum."""
    hist_res, *_ = run_v10_kde_traced(X1, y1, k=8, kde_type="bmm_kde")
    snap = hist_res[0]  # Round 1: all 8 blobs in C0
    lo, span = snap["lo_v"][0].item(), snap["span"][0].item()
    xs = torch.linspace(lo, lo + span, 64).numpy()

    # Recompute hard histogram & 3-tap binomial blur on the same projection for comparison
    proj = (X1 @ snap["V"][0])
    bin_idx = ((proj - lo) / span * 63).clamp(0, 63).long()
    hard_hist = torch.bincount(bin_idx, minlength=64).float()
    lin = snap["raw_linear"][0]
    d_pad = torch.nn.functional.pad(lin.unsqueeze(0), (1, 1), mode="replicate")
    blur3 = (0.25 * d_pad[:, :-2] + 0.5 * d_pad[:, 1:-1] + 0.25 * d_pad[:, 2:]).squeeze(0)
    bmm_kde = snap["bmm_density"][0]
    dct_kde = snap["dct_density"][0]
    spec = snap["spec"][0]

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    fig.suptitle(f"1D Density Estimators (Round 1, n=3200, Silverman h={snap['h_bins'][0]:.2f} bins)", fontsize=13, fontweight="bold")

    curves = [
        (axes[0, 0], hard_hist, "Hard 64-bin Histogram (v3/v5/v9)", "gray"),
        (axes[0, 1], lin,       "Linear Binning (25e/25f/25g)",    "steelblue"),
        (axes[0, 2], blur3,     "Linear + [0.25, 0.5, 0.25] (25c)", "darkorange"),
        (axes[1, 0], bmm_kde,   "FFT-Free Batched Gaussian KDE (bmm)", "crimson"),
        (axes[1, 1], dct_kde,   "FFT-Free DCT Gaussian Low-Pass",   "forestgreen"),
    ]
    for ax, y, title, col in curves:
        ax.fill_between(xs, y.numpy(), alpha=0.30, color=col)
        ax.plot(xs, y.numpy(), lw=1.8, color=col)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_yticks([])
        ax.grid(True, alpha=0.2)

    # Bottom-right: DCT Spectrum Magnitude |spec_k| for k=1..32
    ax_sp = axes[1, 2]
    ks = torch.arange(1, 33).numpy()
    ax_sp.bar(ks, spec[1:33].abs().numpy(), color="purple", alpha=0.7)
    ax_sp.set_title("DCT Spectrum |spec[1..32]| (Multi-blob Harmonics)", fontsize=10, fontweight="bold")
    ax_sp.set_xlabel("DCT Frequency k")
    ax_sp.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig("vis_kde_1_estimators.png", dpi=140)
    print("Saved vis_kde_1_estimators.png")


def plot_spectral_signatures():
    """Show how DCT spectrum distinguishes 1 Gaussian vs 2 Blobs vs 8 Blobs."""
    num_bins = 64
    C_dct = build_dct_matrix(num_bins)
    idx = torch.arange(num_bins, dtype=torch.float32)
    dist2 = (idx.unsqueeze(0) - idx.unsqueeze(1)) ** 2

    cases = [
        ("Unimodal (1 Blob)",        torch.randn(1000) * 1.0),
        ("Bimodal (2 Blobs)",        torch.cat([torch.randn(500) - 3.0, torch.randn(500) + 3.0])),
        ("Multimodal (5 Blobs)",     torch.cat([torch.randn(200) + c for c in [-6, -3, 0, 3, 6]])),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 6.8))
    fig.suptitle("Spectral Signature in DCT Space (Zero FFT!)", fontsize=13, fontweight="bold")

    for col, (name, p) in enumerate(cases):
        mu = p.mean()
        half_span = 4.0 * p.std()
        lo, span = mu - half_span, 2.0 * half_span
        pos = ((p - lo) / span * (num_bins - 1)).clamp(0.0, 62.9999)
        b0 = pos.floor().long()
        frac = pos - b0.float()
        lin = torch.zeros(num_bins)
        lin.scatter_add_(0, b0, 1.0 - frac)
        lin.scatter_add_(0, (b0 + 1).clamp(max=63), frac)

        h_bins = torch.tensor([0.65 * (63.0 / 8.0) * (len(p) ** -0.2)])
        kde = apply_bmm_gaussian_kde(lin.unsqueeze(0), dist2, h_bins).squeeze(0)
        spec = (lin @ C_dct.T).abs()
        spec_norm = spec[1:25] / spec[0].clamp(min=1e-6)

        xs = torch.linspace(lo, lo + span, num_bins).numpy()
        axes[0, col].fill_between(xs, kde.numpy(), alpha=0.35, color="teal")
        axes[0, col].plot(xs, kde.numpy(), lw=1.8, color="teal")
        axes[0, col].set_title(f"{name}\n(h={h_bins.item():.2f} bins)", fontsize=10, fontweight="bold")
        axes[0, col].set_yticks([])

        axes[1, col].bar(range(1, 25), spec_norm.numpy(), color="indigo", alpha=0.75)
        axes[1, col].set_title("Normalized DCT Energy |spec[k]| / DC", fontsize=9)
        axes[1, col].set_ylim(0, 0.9)
        axes[1, col].set_xlabel("Frequency k")
        axes[1, col].grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("vis_kde_2_spectral_signatures.png", dpi=140)
    print("Saved vis_kde_2_spectral_signatures.png")


def plot_step_trace(X, y_true, history, C_final, final_assign, title, save_path, xlim, ylim):
    rounds = history[:4]
    n_cols = len(rounds) + 1
    fig, axes = plt.subplots(2, n_cols, figsize=(4.3 * n_cols, 7.6), gridspec_kw={"height_ratios": [1.35, 1.0]})
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    cmap = cm.get_cmap("tab10")
    core_mask, out_mask = y_true >= 0, y_true < 0

    for col, snap in enumerate(rounds):
        ax2d, ax1d = axes[0, col], axes[1, col]
        for cid in range(snap["m"]):
            pts = X[(snap["assign"] == cid) & core_mask]
            if pts.numel() > 0:
                ax2d.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.55, color=cmap(cid % 10))
        if out_mask.any():
            ax2d.scatter(X[out_mask, 0], X[out_mask, 1], s=45, marker="^", color="black", zorder=6)
        ax2d.scatter(snap["C_curr"][:, 0], snap["C_curr"][:, 1], s=110, marker="*", color="white", edgecolors="black", zorder=7)

        for s_idx in range(snap["b"]):
            mu, v = snap["M"][s_idx], snap["V"][s_idx]
            th, dot_mu = snap["thresh"][s_idx], snap["dot_mu_v"][s_idx]
            p_anchor = mu + (th - dot_mu) * v
            tang = torch.tensor([-v[1], v[0]])
            ax2d.annotate("", xy=(mu[0] + 4.0 * v[0], mu[1] + 4.0 * v[1]), xytext=(mu[0], mu[1]),
                          arrowprops=dict(arrowstyle="->", color="black", lw=1.8), zorder=8)
            ax2d.plot([p_anchor[0] - 25 * tang[0], p_anchor[0] + 25 * tang[0]],
                      [p_anchor[1] - 25 * tang[1], p_anchor[1] + 25 * tang[1]], ls="--", lw=1.6, color="crimson", zorder=8)

        ax2d.set_title(f"Round {snap['round']}: m={snap['m']}, b={snap['b']}", fontsize=10)
        ax2d.set_xlim(xlim); ax2d.set_ylim(ylim); ax2d.grid(True, alpha=0.2)

        xs = torch.linspace(snap["lo_v"][0], snap["lo_v"][0] + snap["span"][0], 64).numpy()
        ax1d.plot(xs, snap["raw_linear"][0].numpy(), lw=1.0, alpha=0.45, color="gray", label="Linear bin")
        ax1d.fill_between(xs, snap["bmm_density"][0].numpy(), alpha=0.35, color="crimson")
        ax1d.plot(xs, snap["bmm_density"][0].numpy(), lw=1.8, color="crimson", label="bmm KDE")
        ax1d.axvline(snap["dot_mu_v"][0].item(), color="navy", ls=":", lw=1.5, label="Mean")
        ax1d.axvline(snap["thresh"][0].item(), color="black", ls="--", lw=2.0, label="Cut")
        ax1d.set_title(f"C{snap['split_ids'][0].item()} KDE (h={snap['h_bins'][0]:.2f} bins)", fontsize=9)
        ax1d.set_yticks([])
        if col == 0:
            ax1d.legend(fontsize=7, loc="upper right")

    ax_fin = axes[0, -1]
    for cid in range(C_final.shape[0]):
        pts = X[(final_assign == cid) & core_mask]
        if pts.numel() > 0:
            ax_fin.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.55, color=cmap(cid % 10))
    ax_fin.scatter(C_final[:, 0], C_final[:, 1], s=120, marker="*", color="yellow", edgecolors="black", zorder=7)
    ax_fin.set_title(f"Final k={C_final.shape[0]} (Refreshed)", fontsize=10, fontweight="bold")
    ax_fin.set_xlim(xlim); ax_fin.set_ylim(ylim); ax_fin.grid(True, alpha=0.2)

    ax_sum = axes[1, -1]
    ax_sum.axis("off")
    lines = ["Round summary (with SSE Gate):"]
    for snap in history:
        s = ", ".join(f"C{c}->{int(l)}/{int(r)}" for c, l, r in zip(snap["split_ids"].tolist(), snap["cnt_left"].tolist(), snap["cnt_right"].tolist()))
        lines.append(f"R{snap['round']} (m={snap['m']}, b={snap['b']}): {s}")
    ax_sum.text(0.02, 0.95, "\n".join(lines), va="top", ha="left", fontsize=7.5, family="monospace")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=140)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    X1, y1 = make_skewed_blobs_2d()
    X2, y2 = make_outlier_blobs_2d()

    plot_1d_density_comparison(X1, y1)
    plot_spectral_signatures()

    h1, Cf1, fa1, mse1, m1 = run_v10_kde_traced(X1, y1, k=8, kde_type="bmm_kde", use_sse_gate=True, refresh_final=True)
    plot_step_trace(X1, y1, h1, Cf1, fa1,
                    f"Demo 1 Solved (bmm KDE + SSE Gate + Refresh) | Core MSE={mse1:.2f} | Merged={m1}",
                    "vis_kde_3_demo1_solved.png", (-15, 16), (-6, 6))

    h2, Cf2, fa2, mse2, m2 = run_v10_kde_traced(X2, y2, k=8, kde_type="bmm_kde", use_sse_gate=True, refresh_final=True)
    plot_step_trace(X2, y2, h2, Cf2, fa2,
                    f"Demo 2 Solved (q95-Damped + bmm KDE + Refresh) | Core MSE={mse2:.2f} | Merged={m2}",
                    "vis_kde_4_demo2_solved.png", (-10, 14), (-8, 8))

    plt.show()