import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def plot_cumsum_derivatives(proj: torch.Tensor = None, half_w: int = 8, save_path: str = "vis_cumsum_derivatives.png"):
    if proj is None:
        torch.manual_seed(7)
        centers = torch.tensor([-12.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0])
        proj = torch.cat([c + torch.randn(400) * 0.42 for c in centers])

    # 1. Bin-free centered m-spacing & both cumsums
    p, _ = torch.sort(proj)
    n = p.numel()
    r = max(1, min(half_w, n // 6))
    idx = torch.arange(n, device=p.device)
    left = torch.clamp(idx - r, min=0)
    right = torch.clamp(idx + r, max=n - 1)
    span_idx = (right - left).float().clamp(min=1.0)
    eps = 1e-5 * (p[-1] - p[0]).clamp(min=1e-6)
    avg_dist = (p[right] - p[left]).clamp(min=eps) / span_idx

    cdf_pts = (idx.float() + 0.5) / float(n)
    csum_pts = torch.cumsum(1.0 / avg_dist, dim=0)
    csum_pts = csum_pts / csum_pts[-1].clamp(min=1e-9)

    # 2. Resample both cumsums onto a uniform x-grid
    M = 512
    xg = torch.linspace(p[0], p[-1], M, device=p.device)
    dx = (xg[-1] - xg[0]) / float(M - 1)
    idx_r = torch.searchsorted(p, xg).clamp(1, n - 1)
    idx_l = idx_r - 1
    alpha = ((xg - p[idx_l]) / (p[idx_r] - p[idx_l]).clamp(min=1e-9)).clamp(0.0, 1.0)

    cdf_grid = (1.0 - alpha) * cdf_pts[idx_l] + alpha * cdf_pts[idx_r]
    csum_grid = (1.0 - alpha) * csum_pts[idx_l] + alpha * csum_pts[idx_r]

    # Smooth in x-space with replicate padding so monotonically rising edges stay monotonic
    kx = torch.arange(-12, 13, device=p.device, dtype=torch.float32)
    kg = torch.exp(-0.5 * (kx / 4.0) ** 2)
    kg = (kg / kg.sum()).view(1, 1, -1)

    cdf_sm = F.conv1d(F.pad(cdf_grid.view(1, 1, -1), (12, 12), mode="replicate"), kg).view(-1)
    csum_sm = F.conv1d(F.pad(csum_grid.view(1, 1, -1), (12, 12), mode="replicate"), kg).view(-1)

    # 3. Differentiate both cumsums wrt x
    def diff_x(y):
        dy = torch.zeros_like(y)
        dy[1:-1] = (y[2:] - y[:-2]) / (2.0 * dx)
        dy[0] = (y[1] - y[0]) / dx
        dy[-1] = (y[-1] - y[-2]) / dx
        return dy.clamp(min=0.0)

    d_cdf_dx = diff_x(cdf_sm)     # = f(x)
    d_csum_dx = diff_x(csum_sm)   # ~ f(x)^2 (normalized integral)

    # Find peaks & pits for guide lines
    d2 = torch.zeros_like(d_cdf_dx)
    d2[1:-1] = (d_cdf_dx[2:] - d_cdf_dx[:-2]) / (2.0 * dx)
    peaks_idx = torch.where((d2[:-1] > 0) & (d2[1:] <= 0) & (d_cdf_dx[:-1] > 0.15 * d_cdf_dx.max()))[0]
    peaks_x = xg[peaks_idx]

    pits_x = []
    for j in range(len(peaks_idx) - 1):
        p0, p1 = int(peaks_idx[j].item()), int(peaks_idx[j + 1].item())
        seg = d_cdf_dx[p0 : p1 + 1]
        m_val = seg.min()
        min_pos = torch.where(seg <= m_val + 1e-4 * d_cdf_dx.max())[0].float().mean()
        pits_x.append((xg[p0] + min_pos * dx).item())

    # Normalized copies for visual comparison of pit depth
    f_norm = d_cdf_dx / d_cdf_dx.max().clamp(min=1e-9)
    f2_norm = d_csum_dx / d_csum_dx.max().clamp(min=1e-9)

    # 4. Plot
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 8.2), sharex=True)
    fig.suptitle("Differentiating Cumsums wrt x:  d(CDF)/dx = f(x)   vs.   d(Cumsum(m/dist))/dx ∝ f(x)²", fontsize=12.5, fontweight="bold")
    x_np = xg.cpu().numpy()

    axes[0].plot(x_np, cdf_sm.cpu().numpy(), lw=2.0, color="navy", label="Ordinary CDF  (∫ f(x) dx)")
    axes[0].plot(x_np, csum_sm.cpu().numpy(), lw=2.0, ls="--", color="teal", label="Cumsum(m/dist)  (∝ ∫ f(x)² dx)")
    axes[0].set_ylabel("Cumsum")
    axes[0].set_title("1. Both Cumsums (Monotonically Non-Decreasing)", fontsize=10, fontweight="bold")
    axes[0].legend(fontsize=8.5, loc="upper left"); axes[0].grid(True, alpha=0.25)

    axes[1].fill_between(x_np, d_cdf_dx.cpu().numpy(), alpha=0.28, color="navy")
    axes[1].plot(x_np, d_cdf_dx.cpu().numpy(), lw=1.8, color="navy", label="d(CDF)/dx  =  f(x)")
    axes[1].scatter(peaks_x.cpu().numpy(), d_cdf_dx[peaks_idx].cpu().numpy(), s=55, color="limegreen", edgecolors="black", zorder=6)
    axes[1].set_ylabel("d(CDF)/dx")
    axes[1].set_title("2. Derivative of Ordinary CDF wrt x  (= Density f(x), always ≥ 0)", fontsize=10, fontweight="bold")
    axes[1].legend(fontsize=8.5, loc="upper left"); axes[1].grid(True, alpha=0.25)

    axes[2].fill_between(x_np, f2_norm.cpu().numpy(), alpha=0.30, color="teal")
    axes[2].plot(x_np, f2_norm.cpu().numpy(), lw=2.0, color="teal", label="d(Cumsum(m/dist))/dx  ∝ f(x)²  (normalized)")
    axes[2].plot(x_np, f_norm.cpu().numpy(), lw=1.3, ls=":", color="navy", alpha=0.75, label="f(x) (normalized, for comparison)")
    axes[2].set_xlabel("1D Projection x")
    axes[2].set_ylabel("Normalized")
    axes[2].set_title("3. Derivative of Cumsum(m/dist) wrt x  (Sharper Peaks, Much Deeper Pits)", fontsize=10, fontweight="bold")
    axes[2].legend(fontsize=8.5, loc="upper left"); axes[2].grid(True, alpha=0.25)

    for px in peaks_x.cpu().tolist():
        for ax in axes:
            ax.axvline(px, color="limegreen", ls=":", lw=0.9, alpha=0.7)
    for vx in pits_x:
        for ax in axes:
            ax.axvline(vx, color="crimson", ls="--", lw=1.0, alpha=0.75)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=140)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    plot_cumsum_derivatives()
    plt.show()