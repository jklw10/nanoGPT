
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import parts.utils as utils
import math
import training_tools.environments.arcenv as arcenv
import matplotlib.colors as mcolors
import matplotlib.patches as patches
# ==========================================
# 0. SETUP & SWEEP
# ==========================================
torch.set_float32_matmul_precision('high')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware: {DEVICE}")
torch.manual_seed(42)


class Sweep2D:
    def __init__(self, param_x_name, param_x_vals, param_y_name, param_y_vals, num_seeds=2, agents_per_env=5, device=DEVICE):
        self.p_x_name = param_x_name
        self.p_y_name = param_y_name
        self.num_seeds = num_seeds
        self.agents_per_env = agents_per_env
        self.vals_x = torch.tensor(param_x_vals, dtype=torch.float32, device=device)
        self.vals_y = torch.tensor(param_y_vals, dtype=torch.float32, device=device)
        self.grid_x = len(self.vals_x)
        self.grid_y = len(self.vals_y)
        self.num_configs = self.grid_x * self.grid_y
        
        self.num_envs = self.num_configs * self.num_seeds
        self.batch_size = self.num_envs * self.agents_per_env  # Total autonomous brains
        
        grid_y_mat, grid_x_mat = torch.meshgrid(self.vals_y, self.vals_x, indexing='ij')
        
        # Ensure all agents inside the same environment share the identical HP config
        repeats = self.num_seeds * self.agents_per_env
        self.grid_x_flat = grid_x_mat.flatten().repeat_interleave(repeats)
        self.grid_y_flat = grid_y_mat.flatten().repeat_interleave(repeats)
        
    def get_agent_kwargs(self):
        return {self.p_x_name: self.grid_x_flat, self.p_y_name: self.grid_y_flat}
        
    def get_params(self, config_idx):
        w_idx = config_idx // self.grid_x
        s_idx = config_idx % self.grid_x
        return self.vals_x[s_idx].item(), self.vals_y[w_idx].item()
        
    def to_2d_grid(self, config_flat_tensor):
        return config_flat_tensor.view(self.grid_y, self.grid_x)
        
    def get_extent(self):
        return[self.vals_x.min().item(), self.vals_x.max().item(),
                self.vals_y.min().item(), self.vals_y.max().item()]


# ==========================================
# 1. SWAPPABLE ENVIRONMENT INTERFACE
# ==========================================
class BatchedCAEnv:
    """
    Batched Cellular Automata Environment.
    Hazards strictly spawn from the borders and wander inwards.
    Optimal strategy is to build a wall perimeter to protect the inner food zone.
    """
    def __init__(self, num_envs, agents_per_env, grid_size=32, dim=16, device=DEVICE):
        self.B = num_envs
        self.A = agents_per_env
        self.N = self.B * self.A  # Total global agents
        self.grid_size = grid_size
        self.dim = dim
        self.device = device
        
        # 5 Input Nodes: Food, Hazards, Walls, Peers, Hunger (Noise)
        self.num_inputs = 5  
        
        # Environment Spatial Grids (B, H, W)
        self.walls = torch.zeros((self.B, grid_size, grid_size), device=device)
        self.food = torch.zeros((self.B, grid_size, grid_size), device=device)
        self.hazards = torch.zeros((self.B, grid_size, grid_size), device=device)
        
        # Agents spawn clustered in the center 50% of the map to give them time to build outward
        self.agent_pos = torch.rand((self.B, self.A, 2), device=device) * (grid_size * 0.5) + (grid_size * 0.25)
        self.hunger = torch.zeros((self.B, self.A), device=device)
        
        # Initialize sparse ecology
        self._spawn_random(self.food, 0.01)
        self._spawn_hazards_edges(0.1) # Hazards only start at the perimeter
        
    def _spawn_random(self, grid, prob):
        mask = torch.rand_like(grid) < prob
        grid[mask] = 1.0
        
    def _spawn_hazards_edges(self, prob):
        mask = torch.rand_like(self.hazards) < prob
        # Clear inner area so hazards ONLY spawn on the absolute outer edge
        mask[:, 1:-1, 1:-1] = False
        self.hazards[mask] = 1.0
        
    def get_vision(self):
        """ 
        Agents extract local 4x4 surrounding patches from the CA.
        Perfectly maps to dim=16.
        """
        coords = self.agent_pos.long().clamp(0, self.grid_size - 1)
        bx = torch.arange(self.B, device=self.device).view(-1, 1).expand(self.B, self.A)
        
        # 4x4 Meshgrid offsets
        dx_vals = torch.arange(-1, 3, device=self.device)
        dy_vals = torch.arange(-1, 3, device=self.device)
        gy, gx = torch.meshgrid(dy_vals, dx_vals, indexing='ij')
        dx = gx.flatten()
        dy = gy.flatten()
        
        cx = (coords[:, :, 0].unsqueeze(2) + dx).clamp(0, self.grid_size - 1)
        cy = (coords[:, :, 1].unsqueeze(2) + dy).clamp(0, self.grid_size - 1)
        b_idx = bx.unsqueeze(2)
        
        peer_grid = torch.zeros((self.B, self.grid_size, self.grid_size), device=self.device)
        peer_grid[bx, coords[:,:,1], coords[:,:,0]] = 1.0
        
        # Grab CA state patches (Shape of each: B, A, 16)
        v_food = self.food[b_idx, cy, cx]       
        v_hazards = self.hazards[b_idx, cy, cx]
        v_walls = self.walls[b_idx, cy, cx]
        v_peers = peer_grid[b_idx, cy, cx]
        
        # Hunger acts as a noise-generator eye.
        v_hunger_noise = torch.randn(self.B, self.A, self.dim, device=self.device) * self.hunger.unsqueeze(-1)
        
        inputs = torch.stack([v_food, v_hazards, v_walls, v_peers, v_hunger_noise], dim=2)
        
        return inputs.view(self.N, self.num_inputs, self.dim)
        
    def step(self, agent_outputs):
        out = agent_outputs.view(self.B, self.A, -1, self.dim)
        
        motor_node = self.num_inputs
        build_node = self.num_inputs + 1
        
        bx = torch.arange(self.B, device=self.device).view(-1, 1).expand(self.B, self.A)
        
        # -------------------------------------
        # 1. MOVEMENT & WALL COLLISIONS
        # -------------------------------------
        move_out = out[:, :, motor_node, 0:2]
        move_dir = torch.tanh(move_out) * 0.6  
        
        proposed_pos = self.agent_pos + move_dir
        proposed_coords = proposed_pos.long().clamp(0, self.grid_size - 1)
        
        hit_wall = self.walls[bx, proposed_coords[:, :, 1], proposed_coords[:, :, 0]] > 0
        
        self.agent_pos = torch.where(hit_wall.unsqueeze(-1), self.agent_pos, proposed_pos)
        self.agent_pos = self.agent_pos.clamp(0, self.grid_size - 1.001)
        
        # -------------------------------------
        # 2. ADJACENT WALL BUILDING
        # -------------------------------------
        coords = self.agent_pos.long()
        
        build_action = out[:, :, build_node, 0] 
        build_dx = out[:, :, build_node, 1].clamp(-1.0, 1.0).round().long()
        build_dy = out[:, :, build_node, 2].clamp(-1.0, 1.0).round().long()
        
        target_x = (coords[:, :, 0] + build_dx).clamp(0, self.grid_size - 1)
        target_y = (coords[:, :, 1] + build_dy).clamp(0, self.grid_size - 1)
        
        build_mask = build_action > 0.5
        destroy_mask = build_action < -0.5
        
        self.walls[bx[build_mask], target_y[build_mask], target_x[build_mask]] = 1.0
        self.walls[bx[destroy_mask], target_y[destroy_mask], target_x[destroy_mask]] = 0.0
        
        # -------------------------------------
        # 3. INTERACTING WITH CA GRID
        # -------------------------------------
        food_mask = self.food[bx, coords[:,:,1], coords[:,:,0]] > 0
        self.hunger[food_mask] -= 2.0  # Eat food
        self.food[bx[food_mask], coords[:,:,1][food_mask], coords[:,:,0][food_mask]] = 0.0
        
        hazard_mask = self.hazards[bx, coords[:,:,1], coords[:,:,0]] > 0
        self.hunger[hazard_mask] += 2.0  # Hit by hazard
        
        self.hunger = (self.hunger + 0.02).clamp(0, 10.0) # Baseline starvation over time
        
        # -------------------------------------
        # 4. CA WORLD DYNAMICS
        # -------------------------------------
        self._spawn_random(self.food, 0.002) # Sparser food trickle
        
        # Hazards random walk
        shift_x = torch.randint(-1, 2, (1,)).item()
        shift_y = torch.randint(-1, 2, (1,)).item()
        shifted_hazards = torch.roll(self.hazards, shifts=(shift_y, shift_x), dims=(1, 2))
        
        # Prevent wrapping (toroidal teleporting) so outside edges act as the only entry points
        if shift_x == 1: shifted_hazards[:, :, 0] = 0.0
        elif shift_x == -1: shifted_hazards[:, :, -1] = 0.0
        if shift_y == 1: shifted_hazards[:, 0, :] = 0.0
        elif shift_y == -1: shifted_hazards[:, -1, :] = 0.0
        
        # Hazards are physically blocked by walls
        shifted_hazards = shifted_hazards * (1.0 - self.walls)  
        
        # Blending shifts the 1s into the new cells and clears the old ones
        self.hazards = torch.clamp(self.hazards * 0.4 + shifted_hazards, 0, 1).round()
        
        # Trickle new hazards ONLY from the outside edges
        if torch.rand(1).item() < 0.2:
            self._spawn_hazards_edges(0.05) 

        return self.hunger


# ==========================================
# 2. BRAIN DEFINITION (Dynamic Arbitrary Inputs)
# ==========================================
class TriadPiece(nn.Module):
    def __init__(self, dim, batch_size, defaults, num_nodes, **hp_grids):
        super().__init__() 
        self.D = dim
        hps = defaults
        hps.update(hp_grids)
        
        for k, v in hps.items():
            if isinstance(v, torch.Tensor):
                self.register_buffer(k, v.flatten())
            else:
                setattr(self, k, v)
                
        base_weight = torch.randn(1, num_nodes, dim, dim, device=DEVICE) / math.sqrt(self.D)
        self.weight = nn.Parameter(base_weight.expand(batch_size, -1, -1, -1).clone())

        self.register_buffer('dynamic_w_scale', torch.ones(1, device=DEVICE)*self.get_dyn_scale())
        w_scale = self.hp('w_scale', 4)
        self.weight.copy_(utils.rms_norm(self.weight,dim=[-1,-2]) * w_scale*self.dynamic_w_scale) 

        self.register_buffer('g_ema', torch.randn_like(self.weight) * 0.01) 
        self.register_buffer('E_baseline', 0.01 + 0.01 * torch.randn(batch_size, num_nodes, dim, device=DEVICE))
        
    def hp(self, name, ndim):
        val = getattr(self, name)
        if isinstance(val, torch.Tensor):
            return val.view(-1, *([1] * (ndim - 1)))
        return val
    
    def get_dyn_scale(self):
        untanh = self.hp('UNTANH', 4)
        if not isinstance(untanh, torch.Tensor):
            untanh = torch.tensor(untanh, device=DEVICE)
        diff = torch.abs(untanh - 1.0) + 1e-4 
        optimal_curve = 2.5 * torch.pow(diff, 1.0/3.0)
        return optimal_curve
    
    def forward(self, x):
        untanh = self.hp('UNTANH', 3)
        raw_pred = torch.einsum('bnj,bnjk->bnk', x, self.weight)
        prediction = utils.buntanh(raw_pred, untanh, math.sqrt(2 + math.sqrt(2)))
        return prediction
    
    def get_gradient(self, layer_input, error_signal):
        ema_speed = self.hp('EMA_SPEED', 3)
        rew_sens = self.hp('FIXED_REWARD_SENSITIVITY', 4)
        n_scale = self.hp('NOISE_SCALE', 3)
        
        E_curr = F.softmax(error_signal, dim=-1)
        self.E_baseline.mul_(1.0 - ema_speed).add_(ema_speed * E_curr)
        advantage = E_curr - self.E_baseline
        plasticity = 1.0 + utils.rms_norm(advantage.unsqueeze(3) * rew_sens)  
        
        noisy_state = utils.noised_input(layer_input, error_signal, n_scale)
        lg1 = utils.rms_norm(-torch.einsum('bni,bnj->bnij', error_signal, noisy_state))
        
        return (plasticity * lg1) 
    
    def step(self, grad, step_counter):
        wd = self.hp('weight_decay', 4)
        grad = grad - (wd * self.weight)
        
        speed = self.hp('speed', 4)
        width = self.hp('width', 4)
        center = (step_counter * speed) % self.D
        g_mask = utils.wrapping_gaussian_kernel2(grad, center, width)
        grad = grad * g_mask
        
        momentum = self.hp('MOMENTUM', 4)
        lr = self.hp('LR', 4)

        self.g_ema.mul_(momentum).add_((1.0 - momentum) * grad)
        self.weight.add_(lr * self.g_ema)
        
        w_scale = self.hp('w_scale', 4)
        dyn_scale = self.dynamic_w_scale.view(-1, 1, 1, 1)
        self.weight.copy_(utils.rms_norm(self.weight,dim=[-1,-2]) * w_scale * dyn_scale)


class PureTriadicBrain(nn.Module):
    def __init__(self, dim, batch_size, num_nodes=16, num_inputs=5, **hp_grids):
        super().__init__()
        self.N = num_nodes
        self.D = dim
        self.batch_size = batch_size
        self.num_inputs = num_inputs
        
        hps = {
            'EMA_SPEED': 0.05, 'FIXED_REWARD_SENSITIVITY': 1.0,
            'LR': 0.006, 'MOMENTUM': 0.4, 'UNTANH': 0.6, 
            'speed': 0.5, 'width': 0.01, 'weight_decay': 0.01,
            'NOISE_SCALE': 0.8, 'c_moment': 0.5, 'w_scale': 1.0,
            'sharpness': 1.0, 'K_connect': num_nodes/2
        }
        hps.update(hp_grids)
        for k, v in hps.items():
            if isinstance(v, torch.Tensor): self.register_buffer(k, v.flatten())
            else: setattr(self, k, v)
        
        self.w1 = TriadPiece(dim, batch_size, hps, num_nodes, **hp_grids)
        self.w2 = TriadPiece(dim, batch_size, hps, num_nodes, **hp_grids)
        self.w3 = TriadPiece(dim, batch_size, hps, num_nodes, **hp_grids)
        
        self.register_buffer('state', 1e-5*torch.randn(batch_size, self.N, dim, device=DEVICE))
        self.register_buffer('output', 1e-5*torch.randn(batch_size, self.N, dim, device=DEVICE))
        self.register_buffer('step_counter', torch.tensor(0.0, device=DEVICE))
        
        self.register_buffer('A', torch.randn(batch_size, self.N, self.N, device=DEVICE))
        self.register_buffer('A_ema', torch.zeros(batch_size, self.N, self.N, device=DEVICE))
        
        ## Simple Innate routing bias: Motor nodes inherently listen more to sensors
        #for action_node in range(self.num_inputs, self.num_inputs + 2):
        #    for sensor_node in range(self.num_inputs):
        #        self.A_ema[:, action_node, sensor_node] += 1.0  

        self.A = F.softmax(self.A, dim=2)  
        self.register_buffer('node_pos', torch.randn(batch_size, self.N, 3)) 

    def hp(self, name, ndim):
        val = getattr(self, name)
        if isinstance(val, torch.Tensor): return val.view(-1, *([1] * (ndim - 1)))
        return val

    def step(self, env_inputs):
        self.step_counter += 1.0
        
        queries    = self.w3(self.state)
        keys       = self.w2(self.state) 
        prediction = self.w1(self.state) 

        raw_A = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(self.D)
        raw_A = utils.rms_norm(raw_A,dim=[-1,-2])
        routing_momentum = self.hp('c_moment', 3)
        
        if self.step_counter.item() == 1.0: self.A_ema.copy_(raw_A)  
        else: self.A_ema.copy_(self.A_ema * routing_momentum + raw_A * (1.0 - routing_momentum))
        
        raw_A = self.A_ema
        self.A[:, 0:self.num_inputs, :] = 0.0 # Sensors are deaf to internal thoughts
        
        raw_A = utils.sinkhorn_knopp(raw_A)
        uniform_baseline = 1.0 / (self.N*self.N+1.0)
        above_avg_mask = raw_A > uniform_baseline
        sparse_A = torch.where(above_avg_mask, raw_A, torch.tensor(float(0), device=DEVICE))
        
        A_try = utils.sinkhorn_knopp(sparse_A)
        self.A.copy_(A_try)
        self.A[:, 0:self.num_inputs, :] = 0.0

        target = torch.bmm(self.A, self.output)
        
        # ---------------------------------------------
        # Inject Swappable Environment State
        # ---------------------------------------------
        target[:, 0:self.num_inputs, :] = env_inputs

        err1_base = prediction - target
        err2 = keys - err1_base.detach()
        err3 = queries - err2.detach()
        err1 = prediction - (target + err3.detach())

        grad1 = self.w1.get_gradient(self.state, err1)
        grad2 = self.w2.get_gradient(self.state, err2)
        grad3 = self.w3.get_gradient(self.state, err3)

        self.w1.step(grad1, self.step_counter)
        self.w2.step(grad2, self.step_counter)
        self.w3.step(grad3, self.step_counter)

        self.output.copy_(prediction)
        self.state.copy_(target)

# ==========================================
# 3. RUNNERS & VISUALIZERS
# ==========================================

def run_ca_sweep(sweep, dim=16):
    print(f"Initializing Multi-Agent CA Batch...\nEnvs: {sweep.num_envs} | Agents/Env: {sweep.agents_per_env} | Total Agents: {sweep.batch_size}")
    
    # Init Env
    env = arcenv.BatchedARCEnv(num_envs=sweep.num_envs, agents_per_env=sweep.agents_per_env, dim=dim, device=DEVICE)
    
    # Init Brain
    brain = PureTriadicBrain(dim=dim, batch_size=sweep.batch_size, num_inputs=env.num_inputs, **sweep.get_agent_kwargs()).to(DEVICE)
    brain = torch.compile(brain)
    initial_snapshots = copy.deepcopy(brain)
    
    total_steps = 10000
    eval_start = 9000 
    eval_steps = total_steps - eval_start
    
    hunger_history = torch.zeros((eval_steps, env.B, env.A), device=DEVICE)
    
    for step in range(total_steps):
        inputs = env.get_vision()
        brain.step(inputs)
        hunger = env.step(brain.output)
        
        if step >= eval_start:
            hunger_history[step - eval_start] = hunger

    # ---------------------------------------------
    # Aggregate Fitness (Stability of Hunger)
    # ---------------------------------------------
    # Lower hunger mean + lower variance over time = Higher Fitness
    mean_h_time = hunger_history.mean(dim=0) # (B, A)
    var_h_time = hunger_history.var(dim=0)   # (B, A)
    
    fitness_agents = torch.exp(-(mean_h_time + var_h_time)) # (B, A)
    fitness_envs = fitness_agents.mean(dim=1).cpu()         # Average agent success per environment (B,)
    
    fitness_matrix = fitness_envs.view(sweep.num_configs, sweep.num_seeds)
    mean_fitness = fitness_matrix.mean(dim=1)
    var_fitness = fitness_matrix.var(dim=1)
    penalized_fitness = mean_fitness - var_fitness

    # Find the top 1 specific environment instance across all configurations for playback later
    best_overall_env_idx = torch.argmax(fitness_envs).item()

    return fitness_envs, penalized_fitness, mean_fitness, var_fitness, fitness_matrix, brain, initial_snapshots, env, best_overall_env_idx

def diagnose_batch_health(brain, initial_brain, fitness_envs):
    print("\n" + "="*80)
    print("🧠 RUNNING BATCH AUTOPSY & FEATURE CORRELATION")
    print("="*80)
    # Align fitness mapping with raw brain states (B*A).
    # Since fitness_envs is (B), we expand it to (B*A) purely for metric correlation analysis.
    A_per_env = brain.batch_size // len(fitness_envs)
    fitness_agents = fitness_envs.repeat_interleave(A_per_env).to(DEVICE)
    
    metrics = {}
    metrics['Output_Saturation_Pct'] = (brain.output.abs()).mean(dim=(1,2))
    metrics['Avg_Self_Attention'] = torch.diagonal(brain.A, dim1=1, dim2=2).mean(dim=1)
    
    eps = 1e-7
    results =[]
    for name, vals in metrics.items():
        vx = vals - vals.mean()
        vy = fitness_agents - fitness_agents.mean()
        corr = (vx * vy).sum() / (vx.norm() * vy.norm() + eps)
        results.append((corr.item(), name))
        
    for corr, name in sorted(results, key=lambda x: abs(x[0]), reverse=True):
        print(f" {'🔥' if abs(corr)>0.3 else '  '} {name:<25} | Corr: {corr:>7.4f}")

def visualize_sweep(sweep, penalized_fitness, var_fitness):
    perf_grid = sweep.to_2d_grid(penalized_fitness).numpy()
    var_grid = sweep.to_2d_grid(var_fitness).numpy()
    best_config_idx = torch.argmax(penalized_fitness).item()
    best_x, best_y = sweep.get_params(best_config_idx)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    im1 = ax1.imshow(perf_grid, origin='lower', extent=sweep.get_extent(), aspect='auto', cmap='viridis')
    ax1.set_title("Variance-Penalized Env Fitness (Stable Low Hunger)")
    ax1.scatter(best_x, best_y, color='cyan', marker='*', s=150, edgecolor='black', label='Best Config')
    fig.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(var_grid, origin='lower', extent=sweep.get_extent(), aspect='auto', cmap='inferno')
    ax2.set_title("Fitness Instability (Variance Across Seeds)")
    ax2.scatter(best_x, best_y, color='cyan', marker='*', s=150, edgecolor='black')
    fig.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.show()

def play_best_environment(brain, env, best_env_idx, steps=200):
    """ Post-run dynamic visualization of the best batch interacting with peers """
    print(f"\n🎬 Playing Multi-Agent Environment Simulation (Env Index: {best_env_idx})")
    print("Close the visualization window to end the script.")
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for step in range(steps):
        inputs = env.get_vision()
        brain.step(inputs)
        env.step(brain.output)
        
        # Extract CA state maps strictly for the single best environment
        walls = env.walls[best_env_idx].cpu().numpy()
        food = env.food[best_env_idx].cpu().numpy()
        hazards = env.hazards[best_env_idx].cpu().numpy()
        
        # Build RGB rendering (R=Hazard, G=Food, B=Walls)
        img = np.zeros((env.grid_size, env.grid_size, 3))
        img[..., 0] = hazards  
        img[..., 1] = food     
        img[..., 2] = walls    
        
        ax.clear()
        ax.imshow(img, origin='lower', extent=[0, env.grid_size, 0, env.grid_size])
        
        # Render interacting peers
        agents_x = env.agent_pos[best_env_idx, :, 0].cpu().numpy()
        agents_y = env.agent_pos[best_env_idx, :, 1].cpu().numpy()
        ax.scatter(agents_x, agents_y, c='white', edgecolors='black', s=80, label='Agents', zorder=5)
        
        ax.set_title(f"🏆 Best Batch Survival CA | Time: {step}/{steps}")
        ax.set_xlim(0, env.grid_size)
        ax.set_ylim(0, env.grid_size)
        ax.axis('off')
        plt.pause(0.01)
        
        if not plt.fignum_exists(fig.number):
            break
            
    plt.ioff()
    plt.close()

#if __name__ == "__main__":
#    torch.set_grad_enabled(False) 
#    
#    # Total Agents running simultaneously = 10 x 10 x 2 x 5 = 1,000 Agents! 
#    # Perfectly fits your "1k 10-node agent" fast-test constraint.
#    sweep = Sweep2D(
#        param_x_name='UNTANH', param_x_vals=np.linspace(0.1, 3.0, 10),
#        param_y_name='w_scale', param_y_vals=np.linspace(0.3, 1.5, 10),
#        num_seeds=2, agents_per_env=5
#    )
#    
#    (fitness_envs, penalized_fitness, mean_fitness, var_fitness, 
#     fitness_matrix, brain, initial, final_env, best_idx) = run_ca_sweep(sweep, dim=16)
#    
#    diagnose_batch_health(brain, initial, fitness_envs)
#    
#    # 1. Plot the Hyperparameter Landscapes
#    visualize_sweep(sweep, penalized_fitness, var_fitness)
#    
#    # 2. Animate the best resulting environment live!
#    play_best_environment(brain, final_env, best_idx, steps=1000)


# ==========================================
# 4. ARC VISUALIZATION & LIVE PLAYER
# ==========================================

# Standard ARC Colormap (0-9)
ARC_COLORS =[
    '#000000', # 0: Black
    '#0074D9', # 1: Blue
    '#FF4136', # 2: Red
    '#2ECC40', # 3: Green
    '#FFDC00', # 4: Yellow
    '#AAAAAA', # 5: Grey
    '#F012BE', # 6: Magenta
    '#FF851B', # 7: Orange
    '#7FDBFF', # 8: Teal
    '#870C25'  # 9: Maroon
]
arc_cmap = mcolors.ListedColormap(ARC_COLORS)
arc_norm = mcolors.Normalize(vmin=0, vmax=9)

def play_live_arc_learning(best_untanh, best_w_scale, dim=64, num_nodes=32):
    print(f"\n🎬 Launching Live ARC Global Painter")
    print(f"Optimal Params -> UNTANH: {best_untanh:.4f} | w_scale: {best_w_scale:.4f}")
    
    env = arcenv.BatchedARCEnv(num_envs=1, agents_per_env=1, grid_size=16, dim=dim)
    
    best_kwargs = {
        'UNTANH': torch.tensor([best_untanh], device=DEVICE),
        'w_scale': torch.tensor([best_w_scale], device=DEVICE)
    }
    
    brain = PureTriadicBrain(
        dim=dim, batch_size=1, num_nodes=num_nodes, num_inputs=env.num_inputs, **best_kwargs
    ).to(DEVICE)
    
    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    total_steps = 1500
    demo_steps = env.demo_duration
    
    for step in range(total_steps):
        inputs = env.get_vision()
        brain.step(inputs)
        env.step(brain.output)
        
        if step % 15 == 0 or step == demo_steps - 1 or step == total_steps - 1:
            for ax in axes: ax.clear()
            
            if step < demo_steps:
                phase_title = f"PHASE 1: DEMO (Learning Rule) | Step {step}/{demo_steps}"
                fig.patch.set_facecolor('#ffffff') 
            else:
                phase_title = f"PHASE 2: TEST (Target Blinded!) | Step {step}/{total_steps}"
                fig.patch.set_facecolor('#fff0f0') 
                
            fig.suptitle(f"Live Embodied ARC Solver (Global Painter)\n{phase_title}\nError: {env.hunger[0,0].item():.2f}", fontsize=16, fontweight='bold')
            
            axes[0].imshow(env.input_grid[0].cpu().numpy(), cmap=arc_cmap, norm=arc_norm)
            axes[0].set_title("Input Task", fontsize=14)
            axes[0].axis('off')
            
            # The Canvas
            axes[1].imshow(env.canvas_grid[0].cpu().numpy(), cmap=arc_cmap, norm=arc_norm)
            axes[1].set_title("Agent's Canvas", fontsize=14)
            axes[1].axis('off')
            
            # Draw the Agent's "Cursor" (Red box where it just painted)
            if env.last_paint_mask[0, 0]:
                px = env.last_paint_x[0, 0].item()
                py = env.last_paint_y[0, 0].item()
                rect = patches.Rectangle((px - 0.5, py - 0.5), 1, 1, linewidth=3, edgecolor='red', facecolor='none')
                axes[1].add_patch(rect)
            
            axes[2].imshow(env.target_grid[0].cpu().numpy(), cmap=arc_cmap, norm=arc_norm)
            if step < demo_steps:
                axes[2].set_title("Target (Visible to Agent)", fontsize=14)
            else:
                axes[2].set_title("Target (HIDDEN FROM AGENT)", fontsize=14, color='red')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.pause(0.01)
            
            if not plt.fignum_exists(fig.number):
                break
                
    plt.ioff()
    plt.close()

if __name__ == "__main__":
    torch.set_grad_enabled(False) 
    
    # Updated Sweep Configs
    # Note: agents_per_env=1, so num_envs=batch_size
    sweep = Sweep2D(
        param_x_name='UNTANH', param_x_vals=np.linspace(0.1, 3.0, 10),
        param_y_name='w_scale', param_y_vals=np.linspace(0.3, 1.5, 10),
        num_seeds=2, agents_per_env=1 
    )
    
    # We now pass dim=64 to hold the 4x4 spatial patches cleanly!
    (fitness_envs, penalized_fitness, mean_fitness, var_fitness, 
     fitness_matrix, brain, initial, final_env, best_idx) = run_ca_sweep(sweep, dim=64)
    
    diagnose_batch_health(brain, initial, fitness_envs)
    visualize_sweep(sweep, penalized_fitness, var_fitness)
    
    best_config_idx = torch.argmax(penalized_fitness).item()
    best_untanh, best_w_scale = sweep.get_params(best_config_idx)
    
    # Launch Live Viewer with new 32 Node architecture
    play_live_arc_learning(best_untanh, best_w_scale, dim=64, num_nodes=32)