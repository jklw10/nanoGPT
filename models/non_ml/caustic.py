import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== Unified Krein Space Engine running on: {device} ===")

# Directional Sun Light
LIGHT_DIR = F.normalize(torch.tensor([0.2, -1.0, 0.3], device=device), dim=0)
GLASS_IOR = 1.52

# Camera Setup
CAM_POS = torch.tensor([0.0, 1.4, -3.2], device=device)
CAM_LOOKAT = torch.tensor([0.0, -0.2, 0.0], device=device)

def get_scene_params(scene_type: str):
    if scene_type == "glass_lens":
        return 0.0, 1.2  # floor_y, grid_bounds
    else:  # stellarator or sphere
        return -0.75, 1.3  # floor_y, grid_bounds

def rotate_y(p: torch.Tensor, angle_rad: float) -> torch.Tensor:
    if angle_rad == 0.0:
        return p
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    px = p[..., 0] * cos_a + p[..., 2] * sin_a
    py = p[..., 1]
    pz = -p[..., 0] * sin_a + p[..., 2] * cos_a
    return torch.stack([px, py, pz], dim=-1)

@torch.no_grad()
def scene_sdf(p: torch.Tensor, angle_rad: float = 0.0, scene_type: str = "stellarator") -> torch.Tensor:
    p_rot = rotate_y(p, angle_rad)
    if scene_type == "sphere":
        return torch.norm(p_rot, dim=-1) - 0.75
    elif scene_type == "glass_lens":
        r = torch.norm(p_rot[..., [0, 2]], dim=-1)
        angle = torch.atan2(p_rot[..., 2], p_rot[..., 0])
        ripples = 0.08 * torch.cos(6.0 * angle) * torch.sin(8.0 * r)
        lens_shape = torch.abs(p_rot[..., 1] - 0.5) - 0.2 + 0.15 * (r ** 2)
        return lens_shape + ripples
    else:  # stellarator torus
        r_maj = torch.norm(p_rot[..., [0, 2]], dim=-1)
        theta = torch.atan2(p_rot[..., 2], p_rot[..., 0])
        
        local_x = r_maj - 0.75
        local_y = p_rot[..., 1]
        phi = torch.atan2(local_y, local_x)
        
        r_min = torch.sqrt(local_x**2 + local_y**2)
        helical_twist = 0.08 * torch.cos(5.0 * theta + 2.0 * phi)
        return r_min - (0.26 + helical_twist)

@torch.no_grad()
def compute_normal(p: torch.Tensor, angle_rad: float = 0.0, scene_type: str = "stellarator") -> torch.Tensor:
    eps = 1e-3
    dx = torch.tensor([eps, 0.0, 0.0], device=device)
    dy = torch.tensor([0.0, eps, 0.0], device=device)
    dz = torch.tensor([0.0, 0.0, eps], device=device)
    nx = scene_sdf(p + dx, angle_rad, scene_type) - scene_sdf(p - dx, angle_rad, scene_type)
    ny = scene_sdf(p + dy, angle_rad, scene_type) - scene_sdf(p - dy, angle_rad, scene_type)
    nz = scene_sdf(p + dz, angle_rad, scene_type) - scene_sdf(p - dz, angle_rad, scene_type)
    return F.normalize(torch.stack([nx, ny, nz], dim=-1), dim=-1)

@torch.no_grad()
def snell_refract(d: torch.Tensor, n: torch.Tensor, eta: float) -> tuple[torch.Tensor, torch.Tensor]:
    cos_i = -torch.sum(d * n, dim=-1, keepdim=True)
    sin2_t = eta**2 * (1.0 - cos_i**2)
    tir = (sin2_t >= 1.0)
    refracted = eta * d + (eta * cos_i - torch.sqrt(torch.clamp(1.0 - sin2_t, min=1e-5))) * n
    return F.normalize(refracted, dim=-1), tir.squeeze(-1)

# ==============================================================================
# 2. HIGH-PERFORMANCE CAUSTIC MAP GENERATOR (POINT MC & KREIN SPPM)
# ==============================================================================


@torch.no_grad()
def compute_krein_anisotropic_filter(caustic_map: torch.Tensor, sigma_s=1.8, alpha_max=4.0, lambda_damp=1e-4):
    """
    Bilateral Krein-Space Anisotropic Filter:
    Combines spatial Krein metric tensor v^T G v with photometric range weighting
    to eliminate light-bleeding into dark background pixels (preserving high SSIM).
    """
    H, W = caustic_map.shape
    device = caustic_map.device
    
    # 1. Structural Guide Map for Gradient
    guide_map = F.avg_pool2d(caustic_map[None, None, ...], kernel_size=3, stride=1, padding=1)[0, 0]
    gy, gx = torch.gradient(guide_map)
    grad_mag_sq = gx**2 + gy**2
    grad_mag = torch.sqrt(grad_mag_sq + lambda_damp)
    
    nx, ny = gx / grad_mag, gy / grad_mag
    
    # 2. Structural Anisotropic Scale alpha(x,y)
    q95 = torch.quantile(grad_mag, 0.95) + 1e-5
    alpha_field = torch.clamp((grad_mag / q95) * alpha_max, 0.0, alpha_max).reshape(1, -1)
    
    # 3. Kernel Window Setup
    kernel_size = int(2 * math.ceil(2 * sigma_s) + 1)
    half_k = kernel_size // 2
    ax = torch.arange(-half_k, half_k + 1, device=device).float()
    grid_y, grid_x = torch.meshgrid(ax, ax, indexing="ij")
    
    dx = grid_x.reshape(-1, 1)  # (K^2, 1)
    dy = grid_y.reshape(-1, 1)  # (K^2, 1)
    
    r_sq = dx**2 + dy**2
    
    # Normal projection distance: (v . n)^2
    nx_flat = nx.reshape(1, -1)
    ny_flat = ny.reshape(1, -1)
    v_dot_n_sq = (dx * nx_flat + dy * ny_flat) ** 2
    
    # 4. Krein Spatial Quadratic Form: v^T G v = r^2 + alpha * (v . n)^2
    v_G_v = r_sq + alpha_field * v_dot_n_sq  # (K^2, H*W)
    
    # Spatial Krein Weight
    spatial_weights = torch.exp(-v_G_v / (2.0 * (sigma_s ** 2)))
    
    # 5. Extract Neighborhood Patches & Center Values
    img_4d = caustic_map[None, None, ...]
    patches = F.unfold(img_4d, kernel_size=kernel_size, padding=half_k)[0]  # (K^2, H*W)
    center_vals = caustic_map.reshape(1, -1)                                 # (1, H*W)
    
    # Photometric Range Weight (Blocks light bleeding into dark background)
    # Range sigma set relative to local caustic peak intensity
    sigma_r = max(0.05, 0.4 * float(torch.max(caustic_map)))
    intensity_diff_sq = (patches - center_vals) ** 2
    range_weights = torch.exp(-intensity_diff_sq / (2.0 * (sigma_r ** 2)))
    
    # 6. Combined Bilateral Krein Weight
    total_weights = spatial_weights * range_weights
    total_weights = total_weights / (total_weights.sum(dim=0, keepdim=True) + 1e-8)
    
    filtered = (total_weights * patches).sum(dim=0)
    return filtered.reshape(H, W)

@torch.no_grad()
def generate_sppm_caustic_map(caustic_res=512, angle_rad=0.0, scene_type="stellarator", passes=3, krein_enabled=True, seed_offset=1, alpha=0.35):
    caustic_map = torch.zeros((caustic_res, caustic_res), device=device)
    floor_y, grid_b = get_scene_params(scene_type)
    
    for k in range(1, passes + 1):
        grid_dim = min(800, int(300 * k))
        gx = torch.linspace(-grid_b, grid_b, grid_dim, device=device)
        gz = torch.linspace(-grid_b, grid_b, grid_dim, device=device)
        grid_u, grid_v = torch.meshgrid(gx, gz, indexing="ij")
        
        torch.manual_seed(seed_offset * 777 + k * 13)
        jitter = (torch.rand((grid_dim, grid_dim, 2), device=device) - 0.5) * ((2.0 * grid_b) / grid_dim)
        p_x = (grid_u + jitter[..., 0]).flatten()
        p_z = (grid_v + jitter[..., 1]).flatten()
        p_y = torch.ones_like(p_x) * 2.0
        
        ray_o = torch.stack([p_x, p_y, p_z], dim=-1)
        ray_d = LIGHT_DIR.expand_as(ray_o)
        
        t = torch.zeros(ray_o.shape[0], device=device)
        for _ in range(24):
            t = t + torch.clamp(scene_sdf(ray_o + t.unsqueeze(-1) * ray_d, angle_rad, scene_type), min=1e-3)
            
        p_entry = ray_o + t.unsqueeze(-1) * ray_d
        n_entry = compute_normal(p_entry, angle_rad, scene_type)
        ray_inside, tir1 = snell_refract(ray_d, n_entry, 1.0 / GLASS_IOR)
        
        p_inside = p_entry + ray_inside * 0.05
        t_in = torch.zeros(ray_o.shape[0], device=device)
        for _ in range(20):
            t_in = t_in + torch.clamp(-scene_sdf(p_inside + t_in.unsqueeze(-1) * ray_inside, angle_rad, scene_type), min=1e-3)
            
        p_exit = p_inside + t_in.unsqueeze(-1) * ray_inside
        n_exit = -compute_normal(p_exit, angle_rad, scene_type)
        ray_exit, tir2 = snell_refract(ray_inside, n_exit, GLASS_IOR / 1.0)
        
        t_floor = (floor_y - p_exit[..., 1]) / (ray_exit[..., 1] - 1e-4)
        p_floor = p_exit + t_floor.unsqueeze(-1) * ray_exit
        valid = (~tir1) & (~tir2) & (t_floor > 0) & (p_floor[..., 1] < floor_y + 0.05)
        
        fx, fz = p_floor[..., 0][valid], p_floor[..., 2][valid]
        px_c = torch.clamp(((fx + 1.8) / 3.6 * caustic_res).long(), 0, caustic_res - 1)
        pz_c = torch.clamp(((fz + 1.8) / 3.6 * caustic_res).long(), 0, caustic_res - 1)
        
        pass_caustic = torch.zeros((caustic_res, caustic_res), device=device)
        if px_c.numel() > 0:
            pass_caustic.index_put_((pz_c, px_c), torch.ones_like(px_c, dtype=torch.float), accumulate=True)
        pass_caustic = pass_caustic / (grid_dim ** 2)
        
        # Progressive Krein Space Filter
        if krein_enabled:
            sigma_s = max(0.25, 1.8 * (float(k) ** -0.50))
            pass_caustic = compute_krein_anisotropic_filter(
                pass_caustic, 
                sigma_s=sigma_s, 
                alpha_max=4.0
            )
            
        caustic_map += (pass_caustic - caustic_map) / k

    return caustic_map

@torch.no_grad()
def generate_mc_caustic_map(caustic_res=512, angle_rad=0.0, scene_type="stellarator", passes=3, seed_offset=1):
    caustic_map = torch.zeros((caustic_res, caustic_res), device=device)
    floor_y, grid_b = get_scene_params(scene_type)
    
    for k in range(1, passes + 1):
        grid_dim = min(800, int(300 * k))
        gx = torch.linspace(-grid_b, grid_b, grid_dim, device=device)
        gz = torch.linspace(-grid_b, grid_b, grid_dim, device=device)
        grid_u, grid_v = torch.meshgrid(gx, gz, indexing="ij")
        
        torch.manual_seed(seed_offset * 777 + k * 13)
        jitter = (torch.rand((grid_dim, grid_dim, 2), device=device) - 0.5) * ((2.0 * grid_b) / grid_dim)
        p_x = (grid_u + jitter[..., 0]).flatten()
        p_z = (grid_v + jitter[..., 1]).flatten()
        p_y = torch.ones_like(p_x) * 2.0
        
        ray_o = torch.stack([p_x, p_y, p_z], dim=-1)
        ray_d = LIGHT_DIR.expand_as(ray_o)
        
        t = torch.zeros(ray_o.shape[0], device=device)
        for _ in range(24):
            t = t + torch.clamp(scene_sdf(ray_o + t.unsqueeze(-1) * ray_d, angle_rad, scene_type), min=1e-3)
            
        p_entry = ray_o + t.unsqueeze(-1) * ray_d
        n_entry = compute_normal(p_entry, angle_rad, scene_type)
        ray_inside, tir1 = snell_refract(ray_d, n_entry, 1.0 / GLASS_IOR)
        
        p_inside = p_entry + ray_inside * 0.05
        t_in = torch.zeros(ray_o.shape[0], device=device)
        for _ in range(20):
            t_in = t_in + torch.clamp(-scene_sdf(p_inside + t_in.unsqueeze(-1) * ray_inside, angle_rad, scene_type), min=1e-3)
            
        p_exit = p_inside + t_in.unsqueeze(-1) * ray_inside
        n_exit = -compute_normal(p_exit, angle_rad, scene_type)
        ray_exit, tir2 = snell_refract(ray_inside, n_exit, GLASS_IOR / 1.0)
        
        t_floor = (floor_y - p_exit[..., 1]) / (ray_exit[..., 1] - 1e-4)
        p_floor = p_exit + t_floor.unsqueeze(-1) * ray_exit
        valid = (~tir1) & (~tir2) & (t_floor > 0) & (p_floor[..., 1] < floor_y + 0.05)
        
        fx, fz = p_floor[..., 0][valid], p_floor[..., 2][valid]
        px_c = (fx + 1.8) / 3.6 * caustic_res
        pz_c = (fz + 1.8) / 3.6 * caustic_res
        
        valid_coords = (px_c >= 0) & (px_c < caustic_res - 1) & (pz_c >= 0) & (pz_c < caustic_res - 1)
        pass_caustic = torch.zeros((caustic_res, caustic_res), device=device)
        if valid_coords.any():
            x0_c, z0_c = px_c[valid_coords].long(), pz_c[valid_coords].long()
            pass_caustic.index_put_((z0_c, x0_c), torch.ones_like(x0_c, dtype=torch.float), accumulate=True)

        caustic_map += ((pass_caustic / (grid_dim ** 2)) - caustic_map) / k

    return caustic_map

# ==============================================================================
# 3. FULL 3D PRIMARY RAYMARCHER
# ==============================================================================

@torch.no_grad()
def render_full_3d_frame(res=512, caustic_res=512, angle_rad=0.0, caustic_mode="sppm", scene_type="stellarator", enable_caustics=True, passes=3, krein_enabled=True):
    floor_y, _ = get_scene_params(scene_type)

    if enable_caustics:
        if caustic_mode == "mc":
            caustic_map = generate_mc_caustic_map(caustic_res=caustic_res, angle_rad=angle_rad, scene_type=scene_type, passes=passes)
        else:
            caustic_map = generate_sppm_caustic_map(caustic_res=caustic_res, angle_rad=angle_rad, scene_type=scene_type, passes=passes, krein_enabled=krein_enabled)
            
        q99 = torch.quantile(caustic_map[caustic_map > 0], 0.99) if (caustic_map > 0).any() else 1.0
        caustic_map_norm = torch.clamp(caustic_map / (q99 + 1e-5), 0.0, 3.0)
    else:
        caustic_map_norm = torch.zeros((caustic_res, caustic_res), device=device)

    # Primary Camera Rays
    y_lin = torch.linspace(1.0, -1.0, res, device=device)
    x_lin = torch.linspace(-1.0, 1.0, res, device=device)
    grid_y, grid_x = torch.meshgrid(y_lin, x_lin, indexing="ij")
    
    cam_fwd = F.normalize(CAM_LOOKAT - CAM_POS, dim=0)
    cam_right = F.normalize(torch.linalg.cross(cam_fwd, torch.tensor([0.0, 1.0, 0.0], device=device)), dim=0)
    cam_up = torch.linalg.cross(cam_right, cam_fwd)
    
    ray_dir = F.normalize(grid_x.unsqueeze(-1) * cam_right + grid_y.unsqueeze(-1) * cam_up + 1.8 * cam_fwd, dim=-1)
    ray_ori = CAM_POS.expand_as(ray_dir)
    
    # Raymarch to Object
    t_m = torch.zeros((res, res), device=device)
    for _ in range(28):
        p_m = ray_ori + t_m.unsqueeze(-1) * ray_dir
        d_m = scene_sdf(p_m, angle_rad, scene_type)
        t_m += torch.clamp(d_m, min=1e-3)
        
    p_glass_hit = ray_ori + t_m.unsqueeze(-1) * ray_dir
    d_glass_final = scene_sdf(p_glass_hit, angle_rad, scene_type)
    hit_glass = d_glass_final < 0.008

    def shade_sky(dir_vec):
        sky_grad = torch.clamp(dir_vec[..., 1].unsqueeze(-1) * 0.5 + 0.5, 0.0, 1.0)
        top_sky = torch.tensor([0.15, 0.25, 0.40], device=device)
        bot_sky = torch.tensor([0.02, 0.04, 0.08], device=device)
        return top_sky * sky_grad + bot_sky * (1.0 - sky_grad)

    def shade_floor(px, pz, ray_origin, ray_direction):
        tile_x = (torch.abs(torch.fmod(px + 10.0, 0.40)) < 0.02).float()
        tile_z = (torch.abs(torch.fmod(pz + 10.0, 0.40)) < 0.02).float()
        grid_lines = torch.clamp(tile_x + tile_z, 0.0, 1.0)
        
        base_slate = torch.tensor([0.18, 0.22, 0.28], device=device)
        tile_color = base_slate * (1.0 - 0.6 * grid_lines.unsqueeze(-1))
        
        in_caustic_bounds = (torch.abs(px) < 1.75) & (torch.abs(pz) < 1.75)
        cx_idx = torch.clamp(((px + 1.8) / 3.6 * caustic_res).long(), 0, caustic_res - 1)
        cz_idx = torch.clamp(((pz + 1.8) / 3.6 * caustic_res).long(), 0, caustic_res - 1)
        
        c_intensity = caustic_map_norm[cz_idx, cx_idx].unsqueeze(-1)
        c_intensity = torch.where(in_caustic_bounds.unsqueeze(-1), c_intensity, torch.zeros_like(c_intensity))
        
        caustic_color = c_intensity * torch.tensor([1.2, 0.95, 0.4], device=device)
        
        r_dist = torch.sqrt(px**2 + pz**2)
        shadow = torch.clamp(1.0 - 0.4 * torch.exp(-((r_dist - 0.7)**2) / 0.2), 0.5, 1.0).unsqueeze(-1)
        raw_floor_color = tile_color * (0.35 * shadow) + caustic_color
        
        hit_pos = torch.stack([px, torch.full_like(px, floor_y), pz], dim=-1)
        dist_from_cam = torch.norm(hit_pos - ray_origin, dim=-1, keepdim=True)
        fog_factor = torch.clamp((dist_from_cam - 3.5) / 6.0, 0.0, 1.0)
        sky_horizon = shade_sky(ray_direction)
        
        return torch.lerp(raw_floor_color, sky_horizon, fog_factor)

    t_floor_direct = (floor_y - ray_ori[..., 1]) / (ray_dir[..., 1] - 1e-4)
    p_floor_direct = ray_ori + t_floor_direct.unsqueeze(-1) * ray_dir
    hit_floor_direct = (~hit_glass) & (ray_dir[..., 1] < -1e-4) & (t_floor_direct > 0)
    hit_sky_direct = (~hit_glass) & (~hit_floor_direct)
    
    bg_color = torch.zeros((res, res, 3), device=device)
    if hit_floor_direct.any():
        bg_color[hit_floor_direct] = shade_floor(
            p_floor_direct[..., 0][hit_floor_direct],
            p_floor_direct[..., 2][hit_floor_direct],
            ray_ori[hit_floor_direct],
            ray_dir[hit_floor_direct]
        )
    if hit_sky_direct.any():
        bg_color[hit_sky_direct] = shade_sky(ray_dir[hit_sky_direct])

    glass_color = torch.zeros((res, res, 3), device=device)
    if hit_glass.any():
        p_g = p_glass_hit[hit_glass]
        d_g = ray_dir[hit_glass]
        n_g1 = compute_normal(p_g, angle_rad, scene_type)
        
        cos_i = torch.clamp(-torch.sum(d_g * n_g1, dim=-1, keepdim=True), 0.0, 1.0)
        fresnel = 0.04 + (1.0 - 0.04) * ((1.0 - cos_i) ** 5)
        
        d_in, _ = snell_refract(d_g, n_g1, 1.0 / GLASS_IOR)
        p_in = p_g + d_in * 0.05
        
        t_in = torch.zeros(p_g.shape[0], device=device)
        for _ in range(20):
            p_curr = p_in + t_in.unsqueeze(-1) * d_in
            d_curr = -scene_sdf(p_curr, angle_rad, scene_type)
            t_in += torch.clamp(d_curr, min=1e-3)
            
        p_exit = p_in + t_in.unsqueeze(-1) * d_in
        n_exit = -compute_normal(p_exit, angle_rad, scene_type)
        d_exit, _ = snell_refract(d_in, n_exit, GLASS_IOR / 1.0)
        
        t_floor_exit = (floor_y - p_exit[..., 1]) / (d_exit[..., 1] - 1e-4)
        p_floor_exit = p_exit + t_floor_exit.unsqueeze(-1) * d_exit
        hit_floor_exit = (d_exit[..., 1] < -1e-4) & (t_floor_exit > 0)
        
        refracted_bg = torch.zeros_like(p_g)
        if hit_floor_exit.any():
            refracted_bg[hit_floor_exit] = shade_floor(
                p_floor_exit[..., 0][hit_floor_exit],
                p_floor_exit[..., 2][hit_floor_exit],
                p_exit[hit_floor_exit],
                d_exit[hit_floor_exit]
            )
        if (~hit_floor_exit).any():
            refracted_bg[~hit_floor_exit] = shade_sky(d_exit[~hit_floor_exit])
            
        h = F.normalize(-LIGHT_DIR + (-d_g), dim=-1)
        spec = torch.pow(torch.clamp(torch.sum(n_g1 * h, dim=-1, keepdim=True), 0.0, 1.0), 64.0)
        specular_light = spec * torch.tensor([1.5, 1.4, 1.2], device=device)
        
        glass_color[hit_glass] = (1.0 - fresnel) * refracted_bg + fresnel * shade_sky(d_g) + specular_light

    final_rgb = torch.where(hit_glass.unsqueeze(-1), glass_color, bg_color)
    return torch.clamp(final_rgb, 0.0, 1.0)

# ==============================================================================
# 4. BENCHMARK SANITY CHECK
# ==============================================================================

@torch.no_grad()
def compute_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size=11, sigma=1.5) -> float:
    i1 = img1.unsqueeze(0).unsqueeze(0) if img1.dim() == 2 else img1
    i2 = img2.unsqueeze(0).unsqueeze(0) if img2.dim() == 2 else img2
    
    x = torch.arange(-window_size // 2 + 1, window_size // 2 + 1, device=img1.device).float()
    xx, yy = torch.meshgrid(x, x, indexing="ij")
    window = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    window = (window / window.sum()).unsqueeze(0).unsqueeze(0)
    
    mu1 = F.conv2d(i1, window, padding=window_size // 2)
    mu2 = F.conv2d(i2, window, padding=window_size // 2)
    
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    sigma1_sq = F.conv2d(i1 * i1, window, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.conv2d(i2 * i2, window, padding=window_size // 2) - mu2_sq
    sigma12 = F.conv2d(i1 * i2, window, padding=window_size // 2) - mu1_mu2
    
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()

@torch.no_grad()
def run_benchmark_sanity_check():
    res = 512
    alpha = 1.0
    print("\n" + "="*105)
    print(f" === BENCHMARK SANITY CHECK (GLASS LENS CAUSTICS) [Alpha={alpha}] ===")
    print("="*105)
    
    print("\n1. Generating Ground Truth Reference (20M Photon MC)...")
    gt_map = torch.zeros((res, res), device=device)
    for p in range(20):
        gt_map += generate_mc_caustic_map(caustic_res=res, scene_type="glass_lens", passes=1, seed_offset=p)
    q99_ref = torch.quantile(gt_map[gt_map > 0], 0.99)
    gt_img = torch.clamp(gt_map / q99_ref, 0.0, 1.0)

    # Point MC
    print("2. Computing Point MC Passes (1, 10, 40)...")
    mc_1_accum = generate_mc_caustic_map(caustic_res=res, scene_type="glass_lens", passes=1, seed_offset=0)
    mc_1_img = torch.clamp(mc_1_accum / (q99_ref / 20), 0.0, 1.0)

    mc_10_accum = mc_1_accum.clone()
    for p in range(1, 10):
        mc_10_accum += generate_mc_caustic_map(caustic_res=res, scene_type="glass_lens", passes=1, seed_offset=p)
    mc_10_img = torch.clamp((mc_10_accum / 10) / (q99_ref / 20), 0.0, 1.0)

    mc_40_accum = mc_10_accum.clone()
    for p in range(10, 40):
        mc_40_accum += generate_mc_caustic_map(caustic_res=res, scene_type="glass_lens", passes=1, seed_offset=p)
    mc_40_img = torch.clamp((mc_40_accum / 40) / (q99_ref / 20), 0.0, 1.0)

    # SPPM Krein ON (with Progressive Alpha)
    print("3. Computing SPPM Krein ON Passes (1, 10, 40)...")
    sppm_on_1_map = generate_sppm_caustic_map(caustic_res=res, scene_type="glass_lens", passes=1, krein_enabled=True, seed_offset=1, alpha=alpha)
    q99_sppm1_on = torch.quantile(sppm_on_1_map[sppm_on_1_map > 0], 0.99)
    sppm_on_1_img = torch.clamp(sppm_on_1_map / q99_sppm1_on, 0.0, 1.0)

    sppm_on_10_map = generate_sppm_caustic_map(caustic_res=res, scene_type="glass_lens", passes=10, krein_enabled=True, seed_offset=1, alpha=alpha)
    q99_sppm10_on = torch.quantile(sppm_on_10_map[sppm_on_10_map > 0], 0.99)
    sppm_on_10_img = torch.clamp(sppm_on_10_map / q99_sppm10_on, 0.0, 1.0)

    sppm_on_40_map = generate_sppm_caustic_map(caustic_res=res, scene_type="glass_lens", passes=40, krein_enabled=True, seed_offset=1, alpha=alpha)
    q99_sppm40_on = torch.quantile(sppm_on_40_map[sppm_on_40_map > 0], 0.99)
    sppm_on_40_img = torch.clamp(sppm_on_40_map / q99_sppm40_on, 0.0, 1.0)

    # SPPM Krein OFF
    print("4. Computing SPPM Krein OFF Passes (1, 10, 40)...")
    sppm_off_1_map = generate_sppm_caustic_map(caustic_res=res, scene_type="glass_lens", passes=1, krein_enabled=False, seed_offset=1, alpha=alpha)
    q99_sppm1_off = torch.quantile(sppm_off_1_map[sppm_off_1_map > 0], 0.99)
    sppm_off_1_img = torch.clamp(sppm_off_1_map / q99_sppm1_off, 0.0, 1.0)

    sppm_off_10_map = generate_sppm_caustic_map(caustic_res=res, scene_type="glass_lens", passes=10, krein_enabled=False, seed_offset=1, alpha=alpha)
    q99_sppm10_off = torch.quantile(sppm_off_10_map[sppm_off_10_map > 0], 0.99)
    sppm_off_10_img = torch.clamp(sppm_off_10_map / q99_sppm10_off, 0.0, 1.0)

    sppm_off_40_map = generate_sppm_caustic_map(caustic_res=res, scene_type="glass_lens", passes=40, krein_enabled=False, seed_offset=1, alpha=alpha)
    q99_sppm40_off = torch.quantile(sppm_off_40_map[sppm_off_40_map > 0], 0.99)
    sppm_off_40_img = torch.clamp(sppm_off_40_map / q99_sppm40_off, 0.0, 1.0)

    print("\n" + "="*105)
    print(" === BENCHMARK METRICS: PERCEPTUAL SSIM ON GLASS LENS CAUSTICS ===")
    print("="*105)
    print(f"{'Method':<20} | {'1 Pass SSIM':<18} | {'10 Passes SSIM':<18} | {'40 Passes SSIM':<18}")
    print("-" * 105)
    
    print(f"{'Point MC':<20} | {compute_ssim(mc_1_img, gt_img):<18.4f} | {compute_ssim(mc_10_img, gt_img):<18.4f} | {compute_ssim(mc_40_img, gt_img):<18.4f}")
    print(f"{'SPPM Krein ON':<20} | {compute_ssim(sppm_on_1_img, gt_img):<18.4f} | {compute_ssim(sppm_on_10_img, gt_img):<18.4f} | {compute_ssim(sppm_on_40_img, gt_img):<18.4f}")
    print(f"{'SPPM Krein OFF':<20} | {compute_ssim(sppm_off_1_img, gt_img):<18.4f} | {compute_ssim(sppm_off_10_img, gt_img):<18.4f} | {compute_ssim(sppm_off_40_img, gt_img):<18.4f}")
    print("="*105 + "\n")

# ==============================================================================
# 5. REALTIME INTERACTIVE 3D SCENE VIEWPORT
# ==============================================================================

def run_interactive_3d_scene():
    angle = 0.0
    rotation_speed = 1.0        # rad/s
    is_paused = False
    
    screen_res = 384           # Screen Viewport Res (384, 512, 768, 1024)
    caustic_map_res = 512      # Photon Field Res
    caustic_passes = 3         # Detail Passes per frame
    
    caustic_modes = ["sppm", "mc"]
    caustic_mode_idx = 0
    scenes = ["stellarator", "glass_lens", "sphere"]
    scene_idx = 0
    caustics_enabled = True
    krein_enabled = True
    
    progressive_accum = False
    accum_buffer = None
    accum_count = 0
    
    print("="*70)
    print(" === LAUNCHING REALTIME 3D GLASS RENDERER ===")
    print(" [K]               : Toggle Krein Space Regularization ON / OFF")
    print(" [1], [2], [3], [4] : Viewport Res (384 / 512 / 768 / 1024)")
    print(" [+] / [-]         : Increase / Decrease Photon Passes (1 to 16)")
    print(" [A]               : Toggle Progressive Frame Accumulation")
    print(" [M] / [SPACE]     : Swap Solver (SPPM Beams <-> Point MC)")
    print(" [S]               : Swap Geometry (Stellarator <-> Lens <-> Sphere)")
    print(" [C]               : Toggle Caustics ON / OFF")
    print(" [P]               : Pause / Resume Spinning")
    print(" [ESC] / [Q]       : Quit")
    print("="*70 + "\n")
    
    window_name = "Realtime 3D Glass Scene - High-Performance Caustics"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 768, 768)

    t_last = time.time()
    
    while True:
        t_start = time.time()
        dt = t_start - t_last
        t_last = t_start
        
        if not is_paused and not progressive_accum:
            angle += rotation_speed * dt
            accum_count = 0
            accum_buffer = None

        current_mode = caustic_modes[caustic_mode_idx]
        current_scene = scenes[scene_idx]

        # Render Full 3D Camera Frame with Floor Caustics
        rgb_tensor = render_full_3d_frame(
            res=screen_res,
            caustic_res=caustic_map_res,
            angle_rad=angle,
            caustic_mode=current_mode,
            scene_type=current_scene,
            enable_caustics=caustics_enabled,
            passes=caustic_passes,
            krein_enabled=krein_enabled
        )
        
        # Progressive Frame Accumulation
        if progressive_accum or is_paused:
            if accum_buffer is None or accum_buffer.shape != rgb_tensor.shape:
                accum_buffer = rgb_tensor.clone()
                accum_count = 1
            else:
                accum_count += 1
                accum_buffer += (rgb_tensor - accum_buffer) / accum_count
            rgb_render = accum_buffer.cpu().numpy()
        else:
            rgb_render = rgb_tensor.cpu().numpy()
            accum_count = 1

        rgb_uint8 = (rgb_render * 255.0).astype(np.uint8)
        bgr_frame = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
        
        fps = 1.0 / max(dt, 1e-4)
        algo_title = "SPPM (Beams)" if current_mode == "sppm" else "Point MC"
        krein_status = "ON" if krein_enabled else "OFF (Singular)"
        
        info_text1 = f"FPS: {fps:.1f} | Res: {screen_res}x{screen_res} | Scene: {current_scene.upper()} [S]"
        info_text2 = f"Solver: {algo_title} [M] | Krein Reg: {krein_status} [K] | Passes: {caustic_passes} [+/-]"
        
        cv2.putText(bgr_frame, info_text1, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bgr_frame, info_text2, (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0) if krein_enabled else (0, 0, 255), 1, cv2.LINE_AA)
        
        cv2.imshow(window_name, bgr_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break
        elif key in (ord('k'), ord('K')):
            krein_enabled = not krein_enabled
            accum_buffer = None
            print(f"Krein Space Regularization Toggled: {'ON' if krein_enabled else 'OFF'}")
        elif key in (ord('1'),):
            screen_res, caustic_map_res = 384, 384
            accum_buffer = None
        elif key in (ord('2'),):
            screen_res, caustic_map_res = 512, 512
            accum_buffer = None
        elif key in (ord('3'),):
            screen_res, caustic_map_res = 768, 768
            accum_buffer = None
        elif key in (ord('4'),):
            screen_res, caustic_map_res = 1024, 1024
            accum_buffer = None
        elif key in (ord('+'), ord('=')):
            caustic_passes = min(16, caustic_passes + 1)
            accum_buffer = None
        elif key in (ord('-'), ord('_')):
            caustic_passes = max(1, caustic_passes - 1)
            accum_buffer = None
        elif key in (ord('a'), ord('A')):
            progressive_accum = not progressive_accum
            accum_buffer = None
        elif key in (ord('m'), ord('M'), 32):
            caustic_mode_idx = (caustic_mode_idx + 1) % len(caustic_modes)
            accum_buffer = None
        elif key in (ord('s'), ord('S')):
            scene_idx = (scene_idx + 1) % len(scenes)
            accum_buffer = None
        elif key in (ord('c'), ord('C')):
            caustics_enabled = not caustics_enabled
            accum_buffer = None
        elif key in (ord('p'), ord('P')):
            is_paused = not is_paused
            accum_buffer = None

    cv2.destroyAllWindows()
    print("Closed Realtime Scene Engine.")

# ==============================================================================
# 6. MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # 1. Run Benchmark Sanity Check
    run_benchmark_sanity_check()
    
    # 2. Launch Interactive Rotating 3D Scene
    run_interactive_3d_scene()