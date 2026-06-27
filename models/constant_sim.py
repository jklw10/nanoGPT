import torch
import pyray as pr
import math
import numpy as np

# =========================================================================
# Simulation Parameters (24x24x24 Grid, 16 Channels)
# =========================================================================
GRID_SIZE = 24          # Voxel grid size
CHANNELS = 16           # Dimensionality of local "shape" (M = 16)
SPACING = 0.55          # World spacing between voxels
HALF_GRID = (GRID_SIZE - 1) / 2.0

# Physical Constants of Imperfect Diffusion
D = 0.12                # Diffusion rate (D)
KAPPA_SQ = 0.05         # Informational absorption/decay (κ² - entropy)
NOISE_STRENGTH = 0.20   # Stochastic coordinate noise (entropy injection)
DT = 0.1                # Step size

# Temporal Homeostasis tracking
ALPHA = 0.15            # EMA decay rate (slow tracking)
DENSITY_SCALE = 0.015   # Normalization factor for the holographic mapping
PERCENTILE = 85.0       # Render only the top 15% most stable, low-error shapes

# 1. Initialize the 16-dimensional continuous state vector S and its EMA
S = torch.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE, CHANNELS), dtype=torch.float32)
ema = torch.zeros_like(S)

# =========================================================================
# Raylib UI Initialization
# =========================================================================
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800

pr.init_window(SCREEN_WIDTH, SCREEN_HEIGHT, "Un-Scuffed 3D Holographic Field Solver")
pr.set_target_fps(60)

camera = pr.Camera3D()
camera.position = pr.Vector3(15.0, 10.0, 15.0)
camera.target = pr.Vector3(0.0, 0.0, 0.0)
camera.up = pr.Vector3(0.0, 1.0, 0.0)
camera.fovy = 45.0
camera.projection = pr.CAMERA_PERSPECTIVE

# =========================================================================
# Main Loop
# =========================================================================
step = 0
while not pr.window_should_close():
    
    # 1. Update the Non-Linear Imperfect Diffusion (PyTorch)
    with torch.no_grad():
        # Inject periodic localized coordinate "shocks" (stochastic vacuum fluctuations)
        if step % 8 == 0:
            rx = torch.randint(0, GRID_SIZE, (1,)).item()
            ry = torch.randint(0, GRID_SIZE, (1,)).item()
            rz = torch.randint(0, GRID_SIZE, (1,)).item()
            S[rx, ry, rz, :] += torch.randn(CHANNELS) * 2.0
            
        # Vectorized 3D Laplacian over all 16 channels
        padded = torch.zeros((GRID_SIZE + 2, GRID_SIZE + 2, GRID_SIZE + 2, CHANNELS), dtype=torch.float32)
        padded[1:-1, 1:-1, 1:-1, :] = S
        
        # Mirror boundary conditions at the edges
        padded[0, :, :, :] = padded[2, :, :, :]
        padded[-1, :, :, :] = padded[-3, :, :, :]
        padded[:, 0, :, :] = padded[:, 2, :, :]
        padded[:, -1, :, :] = padded[:, -3, :, :]
        padded[:, :, 0, :] = padded[:, :, 2, :]
        padded[:, :, -1, :] = padded[:, :, -3, :]
        
        laplacian = (
            padded[2:, 1:-1, 1:-1, :] + padded[:-2, 1:-1, 1:-1, :] +
            padded[1:-1, 2:, 1:-1, :] + padded[1:-1, :-2, 1:-1, :] +
            padded[1:-1, 1:-1, 2:, :] + padded[1:-1, 1:-1, :-2, :] -
            6.0 * S
        )
        
        # Calculate local temporal prediction error BEFORE updating the EMA
        error = torch.sum((S - ema) ** 2, dim=3)
        
        # Map error non-linearly to emergent mass-energy density
        density = torch.clamp(DENSITY_SCALE / (error + 1e-4), max=1.0)
        
        # Update S: dS/dt = D * ∇²S - κ² * S + noise
        S += DT * (D * laplacian - KAPPA_SQ * S)
        S += torch.randn_like(S) * NOISE_STRENGTH * math.sqrt(DT)
        
        # Update local EMA memory
        ema = (1.0 - ALPHA) * ema + ALPHA * S
        
        # Extract active voxels above a soft threshold to start with
        active_coords = torch.nonzero(density > 0.01)
        active_num = active_coords.shape[0]
        
        coords_np = active_coords.numpy()
        values_np = density[active_coords[:, 0], active_coords[:, 1], active_coords[:, 2]].numpy()

    # 2. Camera Update (Slow automatic orbit)
    time = pr.get_time()
    cam_x = 18.0 * math.cos(time * 0.1)
    cam_z = 18.0 * math.sin(time * 0.1)
    cam_y = 4.0 * math.sin(time * 0.03) + 7.0
    camera.position = pr.Vector3(cam_x, cam_y, cam_z)

    # 3. Fast Numpy Depth-Sorting & Percentile Filtering (The "Un-Scuffing")
    if active_num > 0:
        # Convert voxel grid coordinates to centered 3D world space
        x_world = (coords_np[:, 0] - HALF_GRID) * SPACING
        y_world = (coords_np[:, 1] - HALF_GRID) * SPACING
        z_world = (coords_np[:, 2] - HALF_GRID) * SPACING
        
        # Calculate dynamic percentile threshold (render only the top most stable elements)
        cutoff = np.percentile(values_np, PERCENTILE)
        filter_mask = values_np >= cutoff
        
        # Apply the filter mask
        x_world = x_world[filter_mask]
        y_world = y_world[filter_mask]
        z_world = z_world[filter_mask]
        values_filtered = values_np[filter_mask]
        
        # Calculate distance-squared of each filtered voxel to the camera
        dist_sq = (x_world - cam_x)**2 + (y_world - cam_y)**2 + (z_world - cam_z)**2
        
        # Sort indices descending (farthest first) to guarantee correct alpha-blending
        sort_indices = np.argsort(-dist_sq)
        
        # Re-index the coordinates and values based on depth sorting
        x_render = x_world[sort_indices]
        y_render = y_world[sort_indices]
        z_render = z_world[sort_indices]
        vals_render = values_filtered[sort_indices]
        render_num = len(x_render)
    else:
        render_num = 0

    # 4. Rendering
    pr.begin_drawing()
    pr.clear_background(pr.BLACK)
    
    pr.begin_mode_3d(camera)
    
    # Draw reference bounding cage
    pr.draw_cube_wires(pr.Vector3(0.0, 0.0, 0.0), GRID_SIZE * SPACING, GRID_SIZE * SPACING, GRID_SIZE * SPACING, pr.DARKGRAY)
    
    # Render the emergent stable "holographic" density fields as a clean point-cloud
    for idx in range(render_num):
        x, y, z = float(x_render[idx]), float(y_render[idx]), float(z_render[idx])
        val = float(vals_render[idx])
        
        # Glowing color scheme
        r_val = int(min(255, val * 350))
        g_val = int(min(255, val * val * 200))
        b_val = int(min(255, (1.0 - val) * 160 + val * 40))
        alpha = int(min(255, val * 180)) 
        
        color = pr.Color(r_val, g_val, b_val, alpha)
        
        # Un-scuffed: Render as smaller, non-colliding cubes (creating a beautiful 3D matrix/dot display)
        # Using a size of SPACING * 0.35 leaves plenty of empty space to look "through" the cloud
        pr.draw_cube(pr.Vector3(x, y, z), SPACING * 0.35, SPACING * 0.35, SPACING * 0.35, color)
        
    pr.end_mode_3d()
    
    # Draw HUD Overlay
    pr.draw_rectangle(10, 10, 480, 150, pr.Color(0, 0, 0, 200))
    pr.draw_fps(20, 20)
    pr.draw_text(f"Total Voxel States (16D): {GRID_SIZE**3}", 20, 45, 16, pr.LIGHTGRAY)
    pr.draw_text(f"Depth-Sorted Rendered Voxels: {render_num}", 20, 65, 16, pr.GREEN)
    pr.draw_text(f"Stochastic Jitter (Vacuum Noise): {NOISE_STRENGTH:.2f}", 20, 85, 16, pr.BLUE)
    pr.draw_text(f"Display Filter: Top {100.0 - PERCENTILE:.0f}% Most Stable Shapes", 20, 105, 16, pr.YELLOW)
    pr.draw_text(f"Transparency Engine: 60FPS Back-to-Front Depth Sorted", 20, 128, 14, pr.LIGHTGRAY)
    
    pr.end_drawing()
    step += 1

pr.close_window()