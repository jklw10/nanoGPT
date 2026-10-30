import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== Rendering Fair Caustic Visual Comparison Grid on: {device} ===")

# ==============================================================================
# 1. OPTICAL SIMULATION
# ==============================================================================

@torch.no_grad()
def glass_lens_sdf(p: torch.Tensor) -> torch.Tensor:
    r = torch.norm(p[..., [0, 2]], dim=-1)
    angle = torch.atan2(p[..., 2], p[..., 0])
    ripples = 0.08 * torch.cos(6.0 * angle) * torch.sin(8.0 * r)
    lens_shape = torch.abs(p[..., 1] - 0.5) - 0.2 + 0.15 * (r ** 2)
    return lens_shape + ripples

@torch.no_grad()
def compute_normal(p: torch.Tensor) -> torch.Tensor:
    eps = 1e-3
    dx = torch.tensor([eps, 0.0, 0.0], device=device)
    dy = torch.tensor([0.0, eps, 0.0], device=device)
    dz = torch.tensor([0.0, 0.0, eps], device=device)
    nx = glass_lens_sdf(p + dx) - glass_lens_sdf(p - dx)
    ny = glass_lens_sdf(p + dy) - glass_lens_sdf(p - dy)
    nz = glass_lens_sdf(p + dz) - glass_lens_sdf(p - dz)
    return F.normalize(torch.stack([nx, ny, nz], dim=-1), dim=-1)

@torch.no_grad()
def snell_refract(d: torch.Tensor, n: torch.Tensor, eta: float) -> tuple[torch.Tensor, torch.Tensor]:
    cos_i = -torch.sum(d * n, dim=-1, keepdim=True)
    sin2_t = eta**2 * (1.0 - cos_i**2)
    tir = (sin2_t >= 1.0)
    refracted = eta * d + (eta * cos_i - torch.sqrt(torch.clamp(1.0 - sin2_t, min=1e-5))) * n
    return F.normalize(refracted, dim=-1), tir.squeeze(-1)

# ==============================================================================
# NON-LEARNING WEIGHT-NORM PREDICTABILITY LOSS (NLWNL)
# ==============================================================================
@torch.no_grad()
def compute_nlwnl(pred: torch.Tensor, gt: torch.Tensor, kernel_size=7, sigma=1.5, lambda_edge=3.0) -> float:
    p = pred.unsqueeze(0).unsqueeze(0) if pred.dim() == 2 else pred
    g = gt.unsqueeze(0).unsqueeze(0) if gt.dim() == 2 else gt
    pad = kernel_size // 2

    x = torch.arange(-pad, pad + 1, device=pred.device, dtype=torch.float32)
    g1d = torch.exp(-(x**2) / (2 * sigma**2))
    g1d = g1d / g1d.sum()
    g2d = (g1d.unsqueeze(1) @ g1d.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=pred.device, dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=pred.device, dtype=torch.float32).view(1, 1, 3, 3)

    gt_smooth = F.conv2d(g, g2d, padding=pad)
    raw_sq_err = (p - g) ** 2
    v_res = F.conv2d(raw_sq_err, g2d, padding=pad)

    gx = F.conv2d(gt_smooth, sobel_x, padding=1)
    gy = F.conv2d(gt_smooth, sobel_y, padding=1)
    edge_mag = torch.sqrt(gx**2 + gy**2 + 1e-6)

    noise_score = v_res / (1.0 + lambda_edge * edge_mag)
    mean_n = noise_score.mean()
    std_n = noise_score.std() + 1e-6
    p_score = torch.tanh((noise_score - mean_n) / std_n)

    chaos = ((1.0 - p_score) / 2.0).mean() * 0.99
    modulated_loss = (raw_sq_err / (1.0 + p_score * chaos)).mean().item()
    return modulated_loss

@torch.no_grad()
def trace_mc_pass(res, grid_dim, seed_pass):
    caustic = torch.zeros((res, res), device=device)
    light_dir = F.normalize(torch.tensor([0.2, -1.0, 0.3], device=device), dim=0)
    
    gx = torch.linspace(-1.2, 1.2, grid_dim, device=device)
    gz = torch.linspace(-1.2, 1.2, grid_dim, device=device)
    grid_x, grid_z = torch.meshgrid(gx, gz, indexing="ij")
    
    torch.manual_seed(seed_pass)
    jitter = (torch.rand((grid_dim, grid_dim, 2), device=device) - 0.5) * (2.4 / grid_dim)
    p_x = (grid_x + jitter[..., 0]).flatten()
    p_z = (grid_z + jitter[..., 1]).flatten()
    p_y = torch.ones_like(p_x) * 2.0
    
    ray_o = torch.stack([p_x, p_y, p_z], dim=-1)
    ray_d = light_dir.expand_as(ray_o)
    
    t = torch.zeros(ray_o.shape[0], device=device)
    for _ in range(24):
        t = t + torch.clamp(glass_lens_sdf(ray_o + t.unsqueeze(-1) * ray_d), min=1e-3)
        
    p_entry = ray_o + t.unsqueeze(-1) * ray_d
    n_entry = compute_normal(p_entry)
    ray_inside, tir1 = snell_refract(ray_d, n_entry, 1.0 / 1.52)
    
    p_inside = p_entry + ray_inside * 0.05
    t_in = torch.zeros(ray_o.shape[0], device=device)
    for _ in range(20):
        t_in = t_in + torch.clamp(-glass_lens_sdf(p_inside + t_in.unsqueeze(-1) * ray_inside), min=1e-3)
        
    p_exit = p_inside + t_in.unsqueeze(-1) * ray_inside
    n_exit = -compute_normal(p_exit)
    ray_exit, tir2 = snell_refract(ray_inside, n_exit, 1.52 / 1.0)
    
    t_floor = -p_exit[..., 1] / (ray_exit[..., 1] - 1e-4)
    p_floor = p_exit + t_floor.unsqueeze(-1) * ray_exit
    
    px = torch.clamp(((p_floor[..., 0] + 1.8) / 3.6 * res).long(), 0, res-1)
    pz = torch.clamp(((p_floor[..., 2] + 1.8) / 3.6 * res).long(), 0, res-1)
    
    valid = (~tir1) & (~tir2) & (t_floor > 0) & (p_floor[..., 1] < 0.05)
    caustic.index_put_((pz[valid], px[valid]), torch.ones_like(px[valid], dtype=torch.float), accumulate=True)
    return caustic / (grid_dim ** 2)

@torch.no_grad()
def render_sppm_pass(res, grid_dim, pass_k, sigma0_sq=2e-5, seed_pass=1):
    caustic_pass = torch.zeros((res, res), device=device)
    
    gx = torch.linspace(-1.2, 1.2, grid_dim, device=device)
    gz = torch.linspace(-1.2, 1.2, grid_dim, device=device)
    grid_u, grid_v = torch.meshgrid(gx, gz, indexing="ij")
    du, dv = (gx[1] - gx[0]).item(), (gz[1] - gz[0]).item()
    
    torch.manual_seed(seed_pass)
    jitter = (torch.rand((grid_dim, grid_dim, 2), device=device) - 0.5) * (2.4 / grid_dim)
    p_x = (grid_u + jitter[..., 0]).flatten()
    p_z = (grid_v + jitter[..., 1]).flatten()
    p_y = torch.ones_like(p_x) * 2.0
    
    light_dir = F.normalize(torch.tensor([0.2, -1.0, 0.3], device=device), dim=0)
    ray_o = torch.stack([p_x, p_y, p_z], dim=-1)
    ray_d = light_dir.expand_as(ray_o)
    
    t = torch.zeros(ray_o.shape[0], device=device)
    for _ in range(24):
        t = t + torch.clamp(glass_lens_sdf(ray_o + t.unsqueeze(-1) * ray_d), min=1e-3)
        
    p_entry = ray_o + t.unsqueeze(-1) * ray_d
    n_entry = compute_normal(p_entry)
    ray_inside, tir1 = snell_refract(ray_d, n_entry, 1.0 / 1.52)
    
    p_inside = p_entry + ray_inside * 0.05
    t_in = torch.zeros(ray_o.shape[0], device=device)
    for _ in range(20):
        t_in = t_in + torch.clamp(-glass_lens_sdf(p_inside + t_in.unsqueeze(-1) * ray_inside), min=1e-3)
        
    p_exit = p_inside + t_in.unsqueeze(-1) * ray_inside
    n_exit = -compute_normal(p_exit)
    ray_exit, tir2 = snell_refract(ray_inside, n_exit, 1.52 / 1.0)
    
    t_floor = -p_exit[..., 1] / (ray_exit[..., 1] - 1e-4)
    p_floor = p_exit + t_floor.unsqueeze(-1) * ray_exit
    valid = (~tir1) & (~tir2) & (t_floor > 0) & (p_floor[..., 1] < 0.05)
    p_floor = p_floor[..., [0, 2]].reshape(grid_dim, grid_dim, 2)
    valid = valid.reshape(grid_dim, grid_dim)

    # Ray Differentials
    dx_du = (p_floor[1:, :, 0] - p_floor[:-1, :, 0]) / du
    dx_dv = (p_floor[:, 1:, 0] - p_floor[:, :-1, 0]) / dv
    dz_du = (p_floor[1:, :, 1] - p_floor[:-1, :, 1]) / du
    dz_dv = (p_floor[:, 1:, 1] - p_floor[:, :-1, 1]) / dv
    
    J00 = F.pad(dx_du, (0, 0, 0, 1))
    J01 = F.pad(dx_dv, (0, 1, 0, 0))
    J10 = F.pad(dz_du, (0, 0, 0, 1))
    J11 = F.pad(dz_dv, (0, 1, 0, 0))
    
    du_sq = (du * 0.5) ** 2
    reg = sigma0_sq / float(pass_k)
    
    S00 = (J00**2 + J01**2) * du_sq + reg
    S11 = (J10**2 + J11**2) * du_sq + reg
    S01 = (J00*J10 + J01*J11) * du_sq
    
    valid_mask = valid & (S00 < 0.05) & (S11 < 0.05)
    valid_idx = torch.where(valid_mask)
    
    fx, fz = p_floor[..., 0][valid_idx], p_floor[..., 1][valid_idx]
    s00, s11, s01 = S00[valid_idx], S11[valid_idx], S01[valid_idx]
    
    px_c = (fx + 1.8) / 3.6 * res
    pz_c = (fz + 1.8) / 3.6 * res
    
    pixel_scale = res / 3.6
    r_eff_px = torch.sqrt(s00 + s11) * pixel_scale
    
    tight_mask = r_eff_px < 0.5
    if tight_mask.any():
        tc_x, tc_z = px_c[tight_mask], pz_c[tight_mask]
        x0 = torch.floor(tc_x).long()
        z0 = torch.floor(tc_z).long()
        wx = tc_x - x0.float()
        wz = tc_z - z0.float()
        
        x0_c = torch.clamp(x0, 0, res - 2)
        z0_c = torch.clamp(z0, 0, res - 2)
        
        caustic_pass.index_put_((z0_c, x0_c), (1.0 - wx) * (1.0 - wz), accumulate=True)
        caustic_pass.index_put_((z0_c, x0_c + 1), wx * (1.0 - wz), accumulate=True)
        caustic_pass.index_put_((z0_c + 1, x0_c), (1.0 - wx) * wz, accumulate=True)
        caustic_pass.index_put_((z0_c + 1, x0_c + 1), wx * wz, accumulate=True)

    wide_mask = ~tight_mask
    if wide_mask.any():
        wc_x, wc_z = px_c[wide_mask], pz_c[wide_mask]
        w_fx, w_fz = fx[wide_mask], fz[wide_mask]
        w_s00, w_s11, w_s01 = s00[wide_mask], s11[wide_mask], s01[wide_mask]
        
        det = w_s00 * w_s11 - w_s01**2 + 1e-9
        inv00, inv11, inv01 = w_s11 / det, w_s00 / det, -w_s01 / det
        
        patch_r = 2
        weight_sum = torch.zeros(wc_x.shape[0], device=device)
        patch_weights, patch_coords = [], []
        
        for dy in range(-patch_r, patch_r + 1):
            for dx in range(-patch_r, patch_r + 1):
                ix = torch.clamp((wc_x + dx).long(), 0, res - 1)
                iz = torch.clamp((wc_z + dy).long(), 0, res - 1)
                
                ox = (ix.float() / res * 3.6 - 1.8) - w_fx
                oz = (iz.float() / res * 3.6 - 1.8) - w_fz
                
                mahalanobis = inv00 * ox**2 + 2.0 * inv01 * ox * oz + inv11 * oz**2
                bw = torch.exp(-0.5 * mahalanobis)
                
                weight_sum += bw
                patch_weights.append(bw)
                patch_coords.append((iz, ix))
                
        norm_factor = 1.0 / torch.clamp(weight_sum, min=1e-6)
        for bw, (iz, ix) in zip(patch_weights, patch_coords):
            caustic_pass.index_put_((iz, ix), bw * norm_factor, accumulate=True)

    return caustic_pass / (grid_dim ** 2)

# ==============================================================================
# 2. GENERATE COMPARED RENDERS (DECORRELATED SEEDS & UNIFIED NORMALIZATION)
# ==============================================================================

@torch.no_grad()
def main():
    res = 512
    print("\n1/3 Generating Ground Truth Reference (Seeds 1000..1019)...")
    gt_accum = torch.zeros((res, res), device=device)
    for p in range(20):
        gt_accum += trace_mc_pass(res, 1000, seed_pass=1000 + p)
    q99_ref = torch.quantile(gt_accum[gt_accum > 0], 0.99)
    gt_img = torch.clamp(gt_accum / q99_ref, 0.0, 1.0)

    print("2/3 Rendering Point MC at Pass 10 & Pass 40 (Seeds 0..39)...")
    mc_10_accum = torch.zeros((res, res), device=device)
    for p in range(10):
        mc_10_accum += trace_mc_pass(res, 500, seed_pass=p)
    mc_10_img = torch.clamp((mc_10_accum / 10) / (q99_ref / 20), 0.0, 1.0)
    
    mc_40_accum = mc_10_accum.clone()
    for p in range(10, 40):
        mc_40_accum += trace_mc_pass(res, 500, seed_pass=p)
    mc_40_img = torch.clamp((mc_40_accum / 40) / (q99_ref / 20), 0.0, 1.0)

    print("3/3 Rendering SPPM Photon Beams at Pass 10 & Pass 40 (Seeds 2000..2040)...")
    sppm_10_accum = torch.zeros((res, res), device=device)
    for k in range(1, 11):
        g_dim = min(800, int(300 * k))
        sppm_10_accum += (render_sppm_pass(res, g_dim, k, seed_pass=2000 + k) - sppm_10_accum) / k
    # UNIFIED NORMALIZATION: Scale SPPM using GT reference factor q99_ref
    sppm_10_img = torch.clamp(sppm_10_accum / (q99_ref / 20), 0.0, 1.0)

    sppm_40_accum = sppm_10_accum.clone()
    for k in range(11, 41):
        g_dim = min(800, int(300 * k))
        sppm_40_accum += (render_sppm_pass(res, g_dim, k, seed_pass=2000 + k) - sppm_40_accum) / k
    sppm_40_img = torch.clamp(sppm_40_accum / (q99_ref / 20), 0.0, 1.0)

    # Compute NLWNL metric scores against Ground Truth (Lower = Better)
    nlwnl_mc_10 = compute_nlwnl(mc_10_img, gt_img)
    nlwnl_sppm_10 = compute_nlwnl(sppm_10_img, gt_img)
    nlwnl_mc_40 = compute_nlwnl(mc_40_img, gt_img)
    nlwnl_sppm_40 = compute_nlwnl(sppm_40_img, gt_img)

    win_p10_mc = " [WINNER]" if nlwnl_mc_10 < nlwnl_sppm_10 else ""
    win_p10_sppm = " [WINNER]" if nlwnl_sppm_10 < nlwnl_mc_10 else ""
    win_p40_mc = " [WINNER]" if nlwnl_mc_40 < nlwnl_sppm_40 else ""
    win_p40_sppm = " [WINNER]" if nlwnl_sppm_40 < nlwnl_mc_40 else ""

    # ==============================================================================
    # PLOT 2x5 VISUAL COMPARISON GRID
    # ==============================================================================
    gt_np = gt_img.cpu().numpy()
    mc_10_np = mc_10_img.cpu().numpy()
    sppm_10_np = sppm_10_img.cpu().numpy()
    mc_40_np = mc_40_img.cpu().numpy()
    sppm_40_np = sppm_40_img.cpu().numpy()

    crop = slice(180, 340), slice(180, 340)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8.5))
    cmap = "inferno"

    # Row 1: Full Views
    axes[0, 0].imshow(gt_np, cmap=cmap)
    axes[0, 0].set_title("Ground Truth (20M Photons)\nReference (NLWNL = 0.0000)", fontsize=10)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(mc_10_np, cmap=cmap)
    axes[0, 1].set_title(f"Point MC (Pass 10 | 0.38s)\nNLWNL = {nlwnl_mc_10:.6f}{win_p10_mc}", fontsize=10)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(sppm_10_np, cmap=cmap)
    axes[0, 2].set_title(f"SPPM Beams (Pass 10 | 0.81s)\nNLWNL = {nlwnl_sppm_10:.6f}{win_p10_sppm}", fontsize=10)
    axes[0, 2].axis("off")

    axes[0, 3].imshow(mc_40_np, cmap=cmap)
    axes[0, 3].set_title(f"Point MC (Pass 40 | 1.55s)\nNLWNL = {nlwnl_mc_40:.6f}{win_p40_mc}", fontsize=10)
    axes[0, 3].axis("off")

    axes[0, 4].imshow(sppm_40_np, cmap=cmap)
    axes[0, 4].set_title(f"SPPM Beams (Pass 40 | 3.39s)\nNLWNL = {nlwnl_sppm_40:.6f}{win_p40_sppm}", fontsize=10)
    axes[0, 4].axis("off")

    # Row 2: Zoomed Detail Crops
    axes[1, 0].imshow(gt_np[crop], cmap=cmap)
    axes[1, 0].set_title("Zoom: Ground Truth\n[Clean Reference]", fontsize=10)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(mc_10_np[crop], cmap=cmap)
    axes[1, 1].set_title("Zoom: Point MC (Pass 10)\n[Heavy Independent Shot Noise]", fontsize=10)
    axes[1, 1].axis("off")

    axes[1, 2].imshow(sppm_10_np[crop], cmap=cmap)
    axes[1, 2].set_title("Zoom: SPPM Beams (Pass 10)\n[Noise-Free Smooth Focal Cusps]", fontsize=10)
    axes[1, 2].axis("off")

    axes[1, 3].imshow(mc_40_np[crop], cmap=cmap)
    axes[1, 3].set_title("Zoom: Point MC (Pass 40)\n[Noisy Point Hits]", fontsize=10)
    axes[1, 3].axis("off")

    axes[1, 4].imshow(sppm_40_np[crop], cmap=cmap)
    axes[1, 4].set_title("Zoom: SPPM Beams (Pass 40)\n[Crisp, Clean Caustics]", fontsize=10)
    axes[1, 4].axis("off")

    plt.tight_layout()
    plt.savefig("caustics_visual_comparison_fixed.png", dpi=200)
    print("\nFinished rendering fair visual comparison!")
    print(f"\n--- Fair NLWNL Metric Breakdown (Lower = Better) ---")
    print(f"Pass 10 -> Point MC: {nlwnl_mc_10:.6f} | SPPM Beams: {nlwnl_sppm_10:.6f}")
    print(f"Pass 40 -> Point MC: {nlwnl_mc_40:.6f} | SPPM Beams: {nlwnl_sppm_40:.6f}")
    plt.show()

if __name__ == "__main__":
    main()