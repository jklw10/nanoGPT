import pygame
import torch
import torch.nn as nn
import math
import random
import sys

def get_rank1_approximation(g_2d, steps=3):
    """Computes a rank-1 approximation (u, v) of a 2D gradient matrix."""
    m, n = g_2d.shape
    device = g_2d.device
    u = torch.ones(m, 1, device=device)
    for _ in range(steps):
        v = torch.matmul(g_2d.t(), u)
        v_norm = torch.norm(v) + 1e-8
        v = v / v_norm
        u = torch.matmul(g_2d, v)
        u_norm = torch.norm(u) + 1e-8
        u = u / u_norm
    scale = torch.sqrt(torch.norm(g_2d) + 1e-8)
    return u * scale, v * scale

def sherman_morrison_update(inv_cov, u, beta, inv_init, leak):
    inv_M = (1.0 / (beta + leak)) * inv_cov
    
    alpha = 1.0 - beta
    
    inv_M_u = torch.matmul(inv_M, u)
    
    denominator = 1.0 + alpha * torch.matmul(u.t(), inv_M_u)
    
    inv_cov_new = inv_M - (alpha / denominator) * torch.matmul(inv_M_u, inv_M_u.t())
    
    return inv_cov_new

class KreinKroneckerSM(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta_cov=0.99, eps=1e-8,
                 damping=1e-5, gamma=1e-3, weight_decay=1e-2, 
                 R=1.5, kernel='softsign'):
        defaults = dict(lr=lr, beta_cov=beta_cov, eps=eps, damping=damping,
                        gamma=gamma, weight_decay=weight_decay, R=R,
                        kernel=kernel)
        super(KreinKroneckerSM, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta_cov = group['beta_cov']
            damping = group['damping']
            gamma = group['gamma']
            weight_decay = group['weight_decay']
            R = group['R']
            kernel = group['kernel']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                eps_geom = 1e-12

                orig_shape = p.shape
                is_multidim = p.dim() > 1
                
                if is_multidim:
                    m = orig_shape[0]
                    n = p.numel() // m
                else:
                    m, n = p.numel(), 1

                if len(state) == 0:
                    state['step'] = 0
                    state['p_plus'] = torch.clamp(p.data, min=0.0).add_(1e-4)
                    state['p_minus'] = torch.clamp(-p.data, min=0.0).add_(1e-4)
                    
                    scale_factor = 1.0 / math.sqrt(max(damping, eps_geom))
                    state['L_inv_init'] = scale_factor * torch.eye(m, device=p.device)
                    state['R_inv_init'] = scale_factor * torch.eye(n, device=p.device)
                    
                    for ch in ['plus', 'minus']:
                        state[f'L_inv_{ch}'] = state['L_inv_init'].clone()
                        state[f'R_inv_{ch}'] = state['R_inv_init'].clone()

                state['step'] += 1

                p_plus = state['p_plus'].to(torch.float32)
                p_minus = state['p_minus'].to(torch.float32)
                grad_fp32 = grad.to(torch.float32)

                # Apply weight decay to all parameters (including 1D biases) to prevent parameter drift
                if weight_decay != 0:
                    decay_factor = 1.0 - lr * weight_decay
                    p_plus.mul_(decay_factor)
                    p_minus.mul_(decay_factor)

                g_plus = torch.clamp(grad_fp32, min=0.0)
                g_minus = torch.clamp(-grad_fp32, min=0.0)
                g_plus_2d = g_plus.view(m, n)
                g_minus_2d = g_minus.view(m, n)

                if is_multidim:
                    leak = 1.05 * (1.0 - beta_cov)
                    
                    # Plus Channel
                    u_plus, v_plus = get_rank1_approximation(g_plus_2d)
                    state['L_inv_plus'] = sherman_morrison_update(
                        state['L_inv_plus'], u_plus, beta_cov, state['L_inv_init'], leak
                    )
                    state['R_inv_plus'] = sherman_morrison_update(
                        state['R_inv_plus'], v_plus, beta_cov, state['R_inv_init'], leak
                    )
                    
                    # Minus Channel
                    u_minus, v_minus = get_rank1_approximation(g_minus_2d)
                    state['L_inv_minus'] = sherman_morrison_update(
                        state['L_inv_minus'], u_minus, beta_cov, state['L_inv_init'], leak
                    )
                    state['R_inv_minus'] = sherman_morrison_update(
                        state['R_inv_minus'], v_minus, beta_cov, state['R_inv_init'], leak
                    )

                channels = {
                    'plus': (p_plus, g_plus, g_plus_2d, state['L_inv_plus'], state['R_inv_plus']),
                    'minus': (p_minus, g_minus, g_minus_2d, state['L_inv_minus'], state['R_inv_minus'])
                }

                for ch_name, (p_ch, g_ch, g_ch_2d, L_inv, R_inv) in channels.items():
                    step_size = lr
                    if not is_multidim:
                        p_ch.add_(g_ch, alpha=-step_size)
                        p_ch.clamp_(min=1e-6)
                        continue

                    r = torch.abs(p_ch)
                    r_safe = torch.clamp(r, min=eps_geom, max=R - eps_geom)

                    x_norm = r_safe / R
                    if kernel == 'softsign':
                        K_norm = x_norm / (x_norm + 1.0)
                        K_prime = 1.0 / (x_norm + 1.0).square()
                    elif kernel == 'tanh':
                        K_norm = torch.tanh(x_norm)
                        K_prime = 1.0 - K_norm.square()

                    one_minus_K2 = torch.clamp(1.0 - K_norm.square(), min=eps_geom)
                    scale = one_minus_K2.square() / (K_prime + damping * one_minus_K2.square())

                    step_precond_2d = torch.matmul(L_inv, torch.matmul(g_ch_2d, R_inv))
                    step_precond = step_precond_2d.view_as(p_ch)
                    step_natural = step_precond * scale

                    gamma_stable = min(gamma, 1.9 / step_size) if step_size > 0 else 0.0
                    coeff_num = K_norm * K_prime * one_minus_K2
                    coeff_denom = torch.clamp(K_prime.square() + damping * one_minus_K2.square(), min=eps_geom)
                    
                    entropic_force = (gamma_stable * (coeff_num / coeff_denom)) * (p_ch / r_safe)
                    proposed_update = step_natural + entropic_force

                    max_step = 0.9 * (R - r_safe)
                    final_update = torch.clamp(proposed_update, min=-max_step, max=max_step)

                    p_ch.add_(final_update, alpha=-step_size)
                    p_ch.clamp_(min=1e-6)

                state['p_plus'].copy_(p_plus)
                state['p_minus'].copy_(p_minus)
                p.copy_((p_plus - p_minus).to(p.dtype))

        return loss

class PlanarQuadrotor:
    def __init__(self, x=400.0, y=300.0):
        # State vector: [x, y, theta, dx, dy, dtheta]
        self.x = x
        self.y = y
        self.theta = 0.0  
        self.dx = 0.0
        self.dy = 0.0
        self.dtheta = 0.0

        # Physical Constants
        self.m = 1.0         # Mass (kg)
        self.I = 0.08        # Inertia (kg m^2)
        self.L = 0.4         # Arm half-length (m)
        self.g = 9.81        # Gravity (m/s^2)
        self.dt = 1/60.0     # Physics timestep

        # Thrust bounds
        self.T_max = 12.0    
        self.T_min = 0.0

    def step(self, T1, T2, wind_x, wind_y):
        T1 = max(self.T_min, min(self.T_max, T1))
        T2 = max(self.T_min, min(self.T_max, T2))

        ax = ((T1 + T2) * math.sin(self.theta) + wind_x) / self.m
        ay = ((T1 + T2) * math.cos(self.theta) - self.m * self.g + wind_y) / self.m
        alpha = (self.L * (T2 - T1)) / self.I

        self.dx += ax * self.dt
        self.dy += ay * self.dt
        self.dtheta += alpha * self.dt

        self.x += self.dx * 50.0 * self.dt  
        self.y += self.dy * 50.0 * self.dt

        self.theta += self.dtheta * self.dt

        self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi


def get_pid_control(drone, target_x, target_y):
    x_err = (target_x - drone.x) / 100.0
    y_err = (target_y - drone.y) / 100.0

    kp_x = 0.8
    kd_x = 1.2
    target_theta = kp_x * x_err - kd_x * drone.dx
    target_theta = max(-0.6, min(0.6, target_theta))  

    kp_theta = 15.0
    kd_theta = 3.0
    theta_err = target_theta - drone.theta
    torque = kp_theta * theta_err - kd_theta * drone.dtheta

    kp_y = 5.0
    kd_y = 2.5
    cos_theta = math.cos(drone.theta)
    hover_thrust = (drone.m * drone.g) / max(0.5, cos_theta)
    thrust_y = kp_y * y_err - kd_y * drone.dy + hover_thrust

    T1_base = (thrust_y - torque) / 2.0
    T2_base = (thrust_y + torque) / 2.0

    # Also return target_theta so the neural network's loss function can align with it
    return max(0.0, min(12.0, T1_base)), max(0.0, min(12.0, T2_base)), target_theta

class DroneController(nn.Module):
    def __init__(self):
        super(DroneController, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 2)
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                nn.init.constant_(m.bias, 0.0)
        nn.init.constant_(self.net[-1].weight, 0.0)
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, state_error):
        out = self.net(state_error)
        thrust_adj = torch.tanh(out) * 4.0
        return thrust_adj[0], thrust_adj[1]

def draw_drone(screen, drone, T1, T2):
    drone_width = 80
    drone_height = 8
    cx, cy = int(drone.x), int(600 - drone.y)
    cos_t = math.cos(drone.theta)
    sin_t = math.sin(drone.theta)
    
    lx = cx - (drone_width // 2) * cos_t
    ly = cy + (drone_width // 2) * sin_t
    rx = cx + (drone_width // 2) * cos_t
    ry = cy - (drone_width // 2) * sin_t
    
    pygame.draw.line(screen, (150, 150, 150), (lx, ly), (rx, ry), drone_height)
    pygame.draw.circle(screen, (200, 50, 50), (int(lx), int(ly)), 12, 2)
    pygame.draw.circle(screen, (50, 200, 50), (int(rx), int(ry)), 12, 2)
    
    # Draw T1 (Left) Vector
    t1_len = int(T1 * 5)
    t1_end_x = lx - t1_len * sin_t
    t1_end_y = ly - t1_len * cos_t
    pygame.draw.line(screen, (255, 0, 0), (lx, ly), (t1_end_x, t1_end_y), 3)
    
    # Draw T2 (Right) Vector
    t2_len = int(T2 * 5)
    t2_end_x = rx - t2_len * sin_t
    t2_end_y = ry - t2_len * cos_t
    pygame.draw.line(screen, (0, 255, 0), (rx, ry), (t2_end_x, t2_end_y), 3)

def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Safe Online Continual Learning - Planar Quadrotor")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 18)

    drone = PlanarQuadrotor(400.0, 300.0)
    controller = DroneController()

    # Create both optimizers
    adam_opt = torch.optim.Adam(controller.parameters(), lr=1e-3)
    krein_opt = KreinKroneckerSM(controller.parameters(), lr=1e-3, R=1.5)

    active_opt_name = "KreinKroneckerSM (Safe Mode)"
    optimizer = krein_opt

    target_x, target_y = 400.0, 400.0
    time_tick = 0

    base_wind_x = 0.0

    running = True
    while running:
        screen.fill((20, 24, 30))
        time_tick += 1

        # 1. Handle keyboard inputs
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    active_opt_name = "Adam (Standard Mode)"
                    optimizer = adam_opt
                elif event.key == pygame.K_2:
                    active_opt_name = "KreinKroneckerSM (Safe Mode)"
                    optimizer = krein_opt

        # Check if Spacebar is held down
        keys = pygame.key.get_pressed()
        space_held = keys[pygame.K_SPACE]

        # 2. Update target path (orbital trajectory)
        target_x = 400.0 + 150.0 * math.cos(time_tick * 0.01)
        target_y = 350.0 + 80.0 * math.sin(time_tick * 0.02)
        pygame.draw.circle(screen, (0, 180, 255), (int(target_x), int(600 - target_y)), 8)

        # 3. Simulate wind
        if space_held:
            current_wind_x = 5.0  
            current_wind_y = random.uniform(-1.0, 1.0)
        else:
            base_wind_x += random.uniform(-0.5, 0.5)
            base_wind_x = max(-3.0, min(3.0, base_wind_x))
            current_wind_x = base_wind_x
            current_wind_y = random.uniform(-0.5, 0.5)

        # Draw wind vectors
        if abs(current_wind_x) > 0.1:
            wind_color = (200, 150, 0) if space_held else (100, 100, 120)
            for j in range(5):
                wy = 100 + j * 100
                wx_start = 50 if -current_wind_x > 0 else 750
                wx_end = wx_start + int(-current_wind_x * 15)
                pygame.draw.line(screen, wind_color, (wx_start, wy), (wx_end, wy), 2)
                pygame.draw.polygon(screen, wind_color, [(wx_end, wy-4), (wx_end, wy+4), (wx_end + (8 if -current_wind_x > 0 else -8), wy)])

        # 4. Formulate the controller tracking error
        x_err = (drone.x - target_x) / 100.0
        y_err = (drone.y - target_y) / 100.0
        state_tensor = torch.tensor([x_err, y_err, drone.dx, drone.dy, drone.theta, drone.dtheta], dtype=torch.float32)

        # 5. Get Robust Base PID Output (for physical flight safety)
        T1_base, T2_base, target_theta = get_pid_control(drone, target_x, target_y)

        # 6. Online Neural Controller Adaptation
        optimizer.zero_grad()
        
        # Forward pass: Get NN's residual adaptive correction
        T1_nn_tensor, T2_nn_tensor = controller(state_tensor)
        T1_nn, T2_nn = T1_nn_tensor.item(), T2_nn_tensor.item()

        # Combine Base PID and NN Correction
        T1_total_tensor = T1_base + T1_nn_tensor
        T2_total_tensor = T2_base + T2_nn_tensor
        T1 = T1_base + T1_nn
        T2 = T2_base + T2_nn

        # Differentiable 1-step prediction of the next physical state
        ax = ((T1_total_tensor + T2_total_tensor) * math.sin(drone.theta) + current_wind_x) / drone.m
        ay = ((T1_total_tensor + T2_total_tensor) * math.cos(drone.theta) - drone.m * drone.g + current_wind_y) / drone.m
        alpha = (drone.L * (T2_total_tensor - T1_total_tensor)) / drone.I

        pred_dx = drone.dx + ax * drone.dt
        pred_dy = drone.dy + ay * drone.dt
        pred_dtheta = drone.dtheta + alpha * drone.dt

        pred_x = drone.x + pred_dx * 50.0 * drone.dt
        pred_y = drone.y + pred_dy * 50.0 * drone.dt
        pred_theta = drone.theta + pred_dtheta * drone.dt

        # Compute predicted errors (used to run analytical backpropagation)
        pred_x_err = (pred_x - target_x) / 100.0
        pred_y_err = (pred_y - target_y) / 100.0
        
        # Align the neural network's attitude loss with the physical target_theta required for compensation
        pred_theta_err = pred_theta - target_theta

        # Compute 1-step predicted loss
        loss = (pred_x_err**2 + pred_y_err**2) * 5.0 + (pred_theta_err**2) * 2.0
        loss.backward()

        # Optimize NN parameters online
        optimizer.step()

        # Update physical simulation
        drone.step(T1, T2, current_wind_x, current_wind_y)

        # 7. Render & Calculate parameters
        draw_drone(screen, drone, T1, T2)

        max_param_norm = 0.0
        for p in controller.parameters():
            max_param_norm = max(max_param_norm, torch.norm(p).item())

        tilt_deg = math.degrees(drone.theta)
        is_safe = (0 < drone.x < 800) and (0 < drone.y < 600)

        # 8. Render HUD
        hud_color = (100, 255, 100) if is_safe else (255, 100, 100)
        status_text = "SYSTEM STATUS: SECURE" if is_safe else "SYSTEM STATUS: OUT OF BOUNDS"
        
        screen.blit(font.render(status_text, True, hud_color), (20, 20))
        screen.blit(font.render(f"Active Optimizer : {active_opt_name}", True, (255, 255, 255)), (20, 45))
        screen.blit(font.render(f"Max Weight Norm  : {max_param_norm:.4f}  (Limit R=1.5)", True, (255, 255, 255)), (20, 70))
        screen.blit(font.render(f"Aerodynamic Wind : {current_wind_x:.2f} N", True, (200, 180, 100)), (20, 95))
        screen.blit(font.render(f"Drone Tilt Angle : {tilt_deg:.1f} deg", True, (255, 255, 255)), (20, 120))
        screen.blit(font.render(f"NN Corrections   : T1={T1_nn:+.2f}N, T2={T2_nn:+.2f}N", True, (0, 255, 255)), (20, 145))
        
        screen.blit(font.render("[1] Switch to Adam (Unstable Under Turbulence)", True, (150, 150, 150)), (20, 500))
        screen.blit(font.render("[2] Switch to KreinKroneckerSM (Safe Bounded)", True, (150, 150, 150)), (20, 525))
        screen.blit(font.render("[SPACEBAR] Hold to Trigger Severe Wind Gust Blast", True, (255, 150, 0)), (20, 550))

        if not is_safe:
            drone = PlanarQuadrotor(400.0, 300.0)
            controller = DroneController()
            adam_opt = torch.optim.Adam(controller.parameters(), lr=1e-3)
            krein_opt = KreinKroneckerSM(controller.parameters(), lr=1e-3, R=1.5)
            optimizer = krein_opt if "KreinKroneckerSM" in active_opt_name else adam_opt

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()