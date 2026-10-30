import pygame
import torch
import numpy as np
import sys

# Initialize Pygame
pygame.init()
GRID_SIZE = 200        # Resolution of the continuous field grid
SCALE = 4              # Upscale factor for rendering
WIDTH = GRID_SIZE * SCALE
HEIGHT = GRID_SIZE * SCALE

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Continuous Field Physics: Emergent Solitons")
clock = pygame.time.Clock()

# -------------------------------------------------------------------------
# Continuous Field Parameters (Gray-Scott Reaction-Diffusion with Noise)
# -------------------------------------------------------------------------
# Du and Dv are the continuous diffusion rates of the background & active fields
Du = 0.16
Dv = 0.08

# Feed (F) and Kill (k) rates governing the stability of localized spots
F = 0.035
k = 0.060
DT = 1.0

# Noise levels (Space-time white noise added to the field)
NOISE_SIGMA = 0.025

# Initialize fields
# U is the feedstock field (starts at 1.0 everywhere)
# V is the "active/coherence" field (starts at 0.0 everywhere)
U = torch.ones((GRID_SIZE, GRID_SIZE), dtype=torch.float32)
V = torch.zeros((GRID_SIZE, GRID_SIZE), dtype=torch.float32)

# Seed a tiny initial region in the center to start with some structures
center = GRID_SIZE // 2
U[center-10:center+10, center-10:center+10] = 0.50
V[center-10:center+10, center-10:center+10] = 0.25

# -------------------------------------------------------------------------
# Discrete 5-Point Laplacian Stencil (Circular Boundary Conditions)
# -------------------------------------------------------------------------
def laplacian(A):
    # Circular shift allows for smooth, infinite space boundary conditions
    return (torch.roll(A, 1, 0) + torch.roll(A, -1, 0) +
            torch.roll(A, 1, 1) + torch.roll(A, -1, 1) - 4.0 * A)

# -------------------------------------------------------------------------
# Main Simulation Loop
# -------------------------------------------------------------------------
running = True
noise_active = True

while running:
    # Event Handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_n:
                noise_active = not noise_active  # Toggle stochastic noise
            elif event.key == pygame.K_c:
                # Clear all spots (Reset field)
                U = torch.ones((GRID_SIZE, GRID_SIZE), dtype=torch.float32)
                V = torch.zeros((GRID_SIZE, GRID_SIZE), dtype=torch.float32)

    # Manual Intervention: Left-click to paint "active field" (seeding particles)
    if pygame.mouse.get_pressed()[0]:
        mx, my = pygame.mouse.get_pos()
        gx, gy = mx // SCALE, my // SCALE
        # Apply a Gaussian localized seed of the active field V
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                nx, ny = (gx + dx) % GRID_SIZE, (gy + dy) % GRID_SIZE
                dist_sq = dx*dx + dy*dy
                if dist_sq < 16:
                    weight = np.exp(-dist_sq / 4.0)
                    U[ny, nx] = U[ny, nx] * (1.0 - 0.5 * weight)
                    V[ny, nx] = torch.clamp(V[ny, nx] + 0.5 * weight, max=1.0)

    # -------------------------------------------------------------------------
    # Field Evolution (Solving the Continuous PDEs)
    # -------------------------------------------------------------------------
    # 1. Calculate Spatial Diffusion gradients using Laplacians
    lap_U = laplacian(U)
    lap_V = laplacian(V)
    
    # 2. Local autocatalytic reaction term: U * V^2
    reaction = U * (V ** 2)
    
    # 3. Update field states (PDE finite difference steps)
    dU = (Du * lap_U - reaction + F * (1.0 - U)) * DT
    dV = (Dv * lap_V + reaction - (F + k) * V) * DT
    
    U = torch.clamp(U + dU, min=0.0, max=1.0)
    V = torch.clamp(V + dV, min=0.0, max=1.0)
    
    # 4. Inject Space-Time White Noise (Stochastic forcing)
    if noise_active:
        # Clamping prevents concentrations from drifting outside physical bounds [0, 1]
        U = torch.clamp(U + torch.randn_like(U) * NOISE_SIGMA, min=0.0, max=1.0)
        V = torch.clamp(V + torch.randn_like(V) * NOISE_SIGMA, min=0.0, max=1.0)

    # -------------------------------------------------------------------------
    # Render Field to screen
    # -------------------------------------------------------------------------
    # Map the 2D active field V to an RGB array
    # Pygame's surfarray expects shape (W, H, 3), so we transpose the PyTorch tensor
    v_np = V.t().cpu().numpy()
    
    # Build beautiful color-mapped output
    rgb = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)
    
    # Map high active field V to bright glowing cyan, and background U to deep blue
    u_np = U.t().cpu().numpy()
    rgb[:, :, 0] = (v_np * 0).astype(np.uint8)                         # Red
    rgb[:, :, 1] = (v_np * 180 + (1.0 - u_np) * 40).astype(np.uint8)   # Green
    rgb[:, :, 2] = (v_np * 255 + (1.0 - u_np) * 100).astype(np.uint8)  # Blue

    # Create a Pygame surface and scale it up smoothly to fit the window
    surf = pygame.surfarray.make_surface(rgb)
    scaled_surf = pygame.transform.scale(surf, (WIDTH, HEIGHT))
    screen.blit(scaled_surf, (0, 0))

    # Render simple HUD instructions
    font = pygame.font.SysFont(None, 24)
    instructions = [
        "Left-Click & Drag: Manually seed active field V",
        "Key [N]: Toggle Space-Time White Noise Injection",
        "Key [C]: Clear Field",
        f"Noise Active: {noise_active} | Grid: {GRID_SIZE}x{GRID_SIZE}"
    ]
    for i, text in enumerate(instructions):
        img = font.render(text, True, (200, 210, 230))
        # Draw a semi-transparent background box for readability
        pygame.draw.rect(screen, (10, 15, 25), (15, 15 + i * 25, img.get_width() + 10, 22))
        screen.blit(img, (20, 18 + i * 25))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()