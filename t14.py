
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math

import utils

# ==========================================
# 0. SETUP
# ==========================================
torch.set_float32_matmul_precision('high')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware: {DEVICE}")

class Sweep2D:
    def __init__(self, param_x_name, param_x_vals, param_y_name, param_y_vals, device=DEVICE):
        self.p_x_name = param_x_name
        self.p_y_name = param_y_name
        
        self.vals_x = torch.tensor(param_x_vals, dtype=torch.float32, device=device)
        self.vals_y = torch.tensor(param_y_vals, dtype=torch.float32, device=device)
        
        self.grid_x = len(self.vals_x)
        self.grid_y = len(self.vals_y)
        self.batch_size = self.grid_x * self.grid_y
        
        grid_y_mat, grid_x_mat = torch.meshgrid(self.vals_y, self.vals_x, indexing='ij')
        
        # Keep them as flat 1D arrays [batch_size]. The agent will dynamically expand them!
        self.grid_x_flat = grid_x_mat.flatten()
        self.grid_y_flat = grid_y_mat.flatten()
        
    def get_agent_kwargs(self):
        return {
            self.p_x_name: self.grid_x_flat,
            self.p_y_name: self.grid_y_flat
        }
        
    def get_params(self, flat_idx):
        w_idx = flat_idx // self.grid_x
        s_idx = flat_idx % self.grid_x
        return self.vals_x[s_idx].item(), self.vals_y[w_idx].item()
        
    def to_2d_grid(self, flat_tensor):
        return flat_tensor.view(self.grid_y, self.grid_x)
        
    def get_extent(self):
        return[
            self.vals_x.min().item(), self.vals_x.max().item(),
            self.vals_y.min().item(), self.vals_y.max().item()
        ]

# ==========================================
# 2. BRAIN DEFINITION
# ==========================================
phi = 1.61803398875
class PureTriadicBrain(nn.Module):
    def __init__(self, dim, batch_size, num_nodes=80, **hp_grids):
        super().__init__()
        self.N = num_nodes
        self.D = dim
        self.batch_size = batch_size
        
        # Complete dictionary of all sweepable default parameters
        hps = {
            'EMA_SPEED': 0.5,
            'FIXED_REWARD_SENSITIVITY': 1.0,
            'LR': 1e-3,
            'MOMENTUM': 0.42,
            'UNTANH': phi,#0.99,#0.8 #why?
            #'UNTANH': math.sqrt(2+math.sqrt(2)),#0.99, #why?
            'speed': 1.5,
            'width': 0.1,
            'weight_decay': 0.01,
            'lateral_decay': 0.01,
            'w_scale': 1.4,
            'G_NORM': 0.0 ,      # 1.0 = True, 0.0 = False
            'conn_count': 400,
            'c_moment': 0.99
        }
        
        hps.update(hp_grids)
        
        for k, v in hps.items():
            if isinstance(v, torch.Tensor):
                # Flatten ensures it's 1D, completely dodging PyTorch broadcast traps
                self.register_buffer(k, v.flatten())
            else:
                setattr(self, k, v)
        
        w_scale = self.hp('w_scale', 4)
        self.W1 = nn.Parameter(torch.randn(batch_size, self.N, dim, dim, device=DEVICE) / math.sqrt(dim))*w_scale
        self.W2 = nn.Parameter(torch.randn(batch_size, self.N, dim, dim, device=DEVICE) / math.sqrt(dim))*w_scale
        self.W3 = nn.Parameter(torch.randn(batch_size, self.N, dim, dim, device=DEVICE) / math.sqrt(dim))*w_scale
        
        self.register_buffer('M1', torch.randn_like(self.W1)*0.2)
        self.register_buffer('M2', torch.randn_like(self.W2)*0.2)
        self.register_buffer('M3', torch.randn_like(self.W3)*0.2)
        
        self.register_buffer('E_baseline', torch.zeros(batch_size, self.N, dim, device=DEVICE))
        self.register_buffer('state', torch.zeros(batch_size, self.N, dim, device=DEVICE))
        self.register_buffer('output', torch.zeros(batch_size, self.N, dim, device=DEVICE))
        self.register_buffer('target', torch.zeros(batch_size, self.N, dim, device=DEVICE))
        self.register_buffer('step_counter', torch.tensor(0.0, device=DEVICE))
        #self.A = nn.Parameter(torch.zeros(batch_size, self.N, self.N, device=DEVICE), requires_grad=False)
        self.register_buffer('A', torch.zeros(batch_size, self.N, self.N, device=DEVICE))
        
        conn_count = self.hp('conn_count', 1)
        for b in range(batch_size):
            if isinstance(conn_count, torch.Tensor):
                cc = conn_count[b].int()
            else:
                cc = conn_count
            for _ in range(cc):
                src = torch.randint(0, self.N, (1,)).item()
                dst = torch.randint(2, self.N, (1,)).item() 
                self.A[b, dst, src] = 1.0
        self.A_sums = self.A.sum(dim=2, keepdim=True) + 1e-5   
        
    def hp(self, name, ndim):
        """Safely fetches a hyperparameter and auto-aligns it for `ndim` broadcasting."""
        val = getattr(self, name)
        if isinstance(val, torch.Tensor):
            # Reshape from [B] to[B, 1, 1...] dynamically matching the operation
            return val.view(-1, *([1] * (ndim - 1)))
        return val

    def step(self, eye_env, stomach_env):
        self.step_counter += 1.0
        # 1. Project state through W3 and W2 to get Queries and Keys
        # state: [B, N, D], W: [B, N, D, D] -> Output:[B, N, D]
        queries = torch.einsum('bnj,bnjk->bnk', self.state, self.W3) 
        keys = torch.einsum('bnj,bnjk->bnk', self.state, self.W2)    

        # Compute raw affinities [B, N, N]
        raw_A = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(self.D)

        topk_vals, topk_indices = torch.topk(raw_A, k=5, dim=2)
        mask = torch.full_like(raw_A, float('-inf'))
        mask.scatter_(2, topk_indices, topk_vals)

        P = F.softmax(mask, dim=2)

        routing_momentum = self.hp('c_moment', 3)
        self.A.copy_(self.A * routing_momentum + P * (1.0 - routing_momentum))


        # 3. Protect your sensory nodes (Eyes and Stomach shouldn't receive signals)
        # Sinkhorn makes them receive 1.0, so we forcefully zero them out after.
        self.A[:, 0:2, :] = 0.0

        # 4. Route the signals
        total_in = torch.bmm(self.A, self.output)
        # We don't need self.A_sums division anymore because Sinkhorn already normalized the rows!
        self.target = total_in 

        self.target[:, 0, :] = eye_env
        self.target[:, 1, :] = stomach_env

        raw_pred = torch.einsum('bnj,bnjk->bnk', self.state, self.W1)
        
        untanh = self.hp('UNTANH', 3)
        raw = raw_pred - torch.tanh(raw_pred) * untanh
        prediction = torch.tanh(raw) * math.sqrt(2+math.sqrt(2))
        
        error_signal = prediction - self.target
        E_curr = F.softmax(error_signal,dim=-1)
        baseline_mask = (self.E_baseline == 0).float()
        self.E_baseline.copy_(self.E_baseline * (1 - baseline_mask) + E_curr * baseline_mask)
        
        ema_speed = self.hp('EMA_SPEED', 3)
        self.E_baseline.mul_(1.0 - ema_speed).add_(ema_speed * E_curr)
        
        advantage = E_curr - self.E_baseline
        
        # SENSITIVITY applies to advantage.unsqueeze(3) (which is a 4D tensor)
        rew_sens = self.hp('FIXED_REWARD_SENSITIVITY', 4)
        r_pre = advantage.unsqueeze(3) * rew_sens
        R = -(utils.rms_norm(r_pre))#,dim=-1) #TODO check viability
        mask = (self.E_baseline > 0).float().unsqueeze(3)
        
        local_grad = -torch.einsum('bni,bnj->bnij', error_signal, self.state)
        
        g_norm_flag = self.hp('G_NORM', 4)
        if (g_norm_flag > 0.5): 
           # rms_scale = torch.rsqrt(local_grad.pow(2).mean(dim=(2,3), keepdim=True) + 1e-6)
            local_grad = utils.rms_norm(local_grad) 
            #s= local_grad.shape
            #make this one sane for the task: no batch mixing atleast for sweeps
            #local_grad = utils.zca_newton_schulz(local_grad.view(-1,s[-1]),steps=2,power_iters=2).view(s)
        #local_grad = local_grad-local_grad.mean()
        wd = self.hp('weight_decay', 4)
        lat_decay = self.hp('lateral_decay', 4)
        plasticity = 1.0 - R  
        
        grad1 = (plasticity * local_grad * mask) + (lat_decay * self.W3) - (wd * self.W1)
        grad2 = (plasticity * local_grad * mask) + (lat_decay * self.W1) - (wd * self.W2)
        grad3 = (plasticity * local_grad * mask) + (lat_decay * self.W2) - (wd * self.W3)
        
        speed = self.hp('speed', 4)
        width = self.hp('width', 4)
        
        indices = torch.arange(self.D, device=DEVICE, dtype=torch.float32).view(1, 1, 1, self.D)
        center = (self.step_counter * speed) % self.D
        
        diff = torch.abs(indices - center)
        diff = torch.minimum(diff, self.D - diff)
        
        g_mask = torch.exp(-(diff ** 2) / (2 * width ** 2 + 1e-6))
        
        grad1 = grad1 * g_mask
        grad2 = grad2 * g_mask
        grad3 = grad3 * g_mask

        # Momentum and LR apply to matrices (which are 4D tensors)
        momentum = self.hp('MOMENTUM', 4)
        lr = self.hp('LR', 4)

        self.M1.mul_(momentum).add_((1.0 - momentum) * grad1)
        self.M2.mul_(momentum).add_((1.0 - momentum) * grad2)
        self.M3.mul_(momentum).add_((1.0 - momentum) * grad3)
        
        self.W1.add_(lr * self.M1)
        self.W2.add_(lr * self.M2)
        self.W3.add_(lr * self.M3)
        
        self.output.copy_(prediction)
        self.state.copy_(self.target)
# ==========================================
# 3. TASK RUNNER
# ==========================================
def run_tracking_task(sweep, dim=16):
    batch_size = sweep.batch_size
    print(f"Initializing {batch_size} Agents on {DEVICE}...")
    
    # Auto-unpacks the sweep variables and handles dodges automatically
    brain = PureTriadicBrain(dim=dim, batch_size=batch_size, **sweep.get_agent_kwargs()).to(DEVICE)
    
    agent_pos = torch.full((batch_size, 2), -3.0, device=DEVICE)
    agent_vel = torch.zeros(batch_size, 2, device=DEVICE)
    
    total_steps = 3000
    eval_start = 1000 
    eval_steps = total_steps - eval_start
    
    agent_pos_history = torch.zeros((total_steps, batch_size, 2), device=DEVICE)
    food_pos_history = torch.zeros((total_steps, 2), device=DEVICE)
    dist_history = torch.zeros((eval_steps, batch_size), device=DEVICE)
    
    print(f"Running Task (Total: {total_steps} steps, Eval: {eval_steps} steps)...")
    for step in range(total_steps):
        t = step / 100.0
        food_x = math.sin(t) * 2.0
        food_y = math.sin(t * 2.0) * 1.5
        food_pos = torch.tensor([[food_x, food_y]], device=DEVICE).expand(batch_size, 2)
        
        diffs = food_pos - agent_pos
        dists = torch.norm(diffs, dim=1, keepdim=True)
        direction = diffs / (dists + 1e-5)
        
        eye_env = torch.zeros(batch_size, brain.D, device=DEVICE)
        eye_env[:, 0:2] = direction
        
        starvation = torch.clamp(dists / 4.0, 0.0, 1.0)
        stomach_env = torch.randn(batch_size, brain.D, device=DEVICE) * starvation * 2.0

        brain.step(eye_env, stomach_env)
        
        motor_out = brain.output[:, 2, 0:2].clone()
        
        agent_vel = (agent_vel * 0.85) + (motor_out * 0.2)
        agent_pos += agent_vel
        agent_pos = torch.clamp(agent_pos, -4.0, 4.0)
        
        agent_pos_history[step] = agent_pos
        food_pos_history[step] = food_pos[0]
        
        if step >= eval_start:
            dist_history[step - eval_start] = dists.squeeze()

    mean_dist = dist_history.mean(dim=0).cpu()
    var_dist = dist_history.var(dim=0).cpu()
    fitness = torch.exp(-(mean_dist + var_dist))
    return fitness, mean_dist, var_dist, agent_pos_history, food_pos_history, eval_start

# ==========================================
# 4. RENDER VISUALIZATION
# ==========================================
def visualize_sweep(sweep, fitness, mean_dist, var_dist, agent_pos_history, food_pos_history, eval_start):
    perf_grid = sweep.to_2d_grid(fitness).numpy()
    
    best_idx = torch.argmax(fitness).item()
    valid_mask = ~torch.isnan(fitness)
    if not valid_mask.any():
        print("CRITICAL ERROR: All agents collapsed to NaN! The network exploded.")
        return
        
    valid_fitness = fitness[valid_mask]
    threshold = torch.quantile(valid_fitness, 0.99)
    
    # Safe top indices calculation
    top_indices = torch.nonzero(torch.nan_to_num(fitness, nan=-1.0) >= threshold).squeeze()
    if top_indices.dim() == 0: 
        top_indices = top_indices.unsqueeze(0)
    
    if len(top_indices) == 0:
        rep_idx = best_idx # Fallback
    else:
        top_indices = top_indices[torch.argsort(fitness[top_indices])]
        rep_idx = top_indices[len(top_indices) // 2].item()
    
    best_x, best_y = sweep.get_params(best_idx)
    rep_x, rep_y = sweep.get_params(rep_idx)

    print(f"\nBest Model: Mean Dist = {mean_dist[best_idx]:.3f} | Variance = {var_dist[best_idx]:.3f}")

    fig = plt.figure(figsize=(20, 6))
    
    ax1 = plt.subplot(1, 3, 1)
    im1 = ax1.imshow(perf_grid, origin='lower', extent=sweep.get_extent(), aspect='auto', cmap='viridis', vmin=0, vmax=1.0)
    ax1.set_title("Variance-Optimized Fitness Landscape\n(1.0 = Perfect Overlap)")
    ax1.set_xlabel(f"{sweep.p_x_name}")
    ax1.set_ylabel(f"{sweep.p_y_name}")
    
    ax1.scatter(best_x, best_y, color='cyan', marker='*', s=150, edgecolor='black', label='Best Outlier')
    ax1.scatter(rep_x, rep_y, color='magenta', marker='o', s=80, edgecolor='white', label='Top 1% Rep')
    ax1.legend(loc="upper right")
    fig.colorbar(im1, ax=ax1)

    eval_steps = agent_pos_history.shape[0] - eval_start
    food_path = food_pos_history.cpu().numpy()
    time_colors = np.linspace(0, 1, eval_steps)
    
    plots =[
        (plt.subplot(1, 3, 2), best_idx, best_x, best_y, "Absolute Best Run"),
        (plt.subplot(1, 3, 3), rep_idx, rep_x, rep_y, "Top 1% Representative Run")
    ]
    
    for plot_idx, (ax, idx, val_x, val_y, title) in enumerate(plots):
        agent_path = agent_pos_history[:, idx].cpu().numpy()
        
        ax.plot(agent_path[:eval_start, 0], agent_path[:eval_start, 1], 
                color='red', alpha=0.5, linewidth=1.5, label='Initial Exploration (Untracked)')
        
        sc = ax.scatter(agent_path[eval_start:, 0], agent_path[eval_start:, 1], 
                        c=time_colors, cmap='plasma', s=10, alpha=0.8, label='Autonomous Tracking')
        
        ax.plot(food_path[eval_start:, 0], food_path[eval_start:, 1], 
                color='black', linestyle='--', linewidth=2, alpha=0.6, label='Target Path')
        
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        
        score_text = f"Mean Dist: {mean_dist[idx]:.2f} | Var: {var_dist[idx]:.2f}"
        ax.set_title(f"{title}\n{sweep.p_x_name}: {val_x:.4f} | {sweep.p_y_name}: {val_y:.4f}\n{score_text}")
        ax.legend(loc="upper right")
        
        if plot_idx == 1: 
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label('Time Steps (Evaluation)')

    plt.suptitle("Agent Trajectories", fontsize=16)
    plt.tight_layout()
    plt.show()

# ==========================================
# 5. EXECUTION SCRIPT
# ==========================================
if __name__ == "__main__":
    torch.set_grad_enabled(False) 
    
    #hps = {
    #    'EMA_SPEED': 0.05,
    #    'FIXED_REWARD_SENSITIVITY': 1.0,
    #    'LR': 1e-2,
    #    'MOMENTUM': 0.8,
    #    'UNTANH': 0.99,
    #    'speed': 0.5,    
    #    'width': 0.001  
    #    'weight_decay': 0.01,
    #    'lateral_decay': 0.01 
    #}
    
    #sweep = Sweep2D(
    #    param_x_name='FIXED_REWARD_SENSITIVITY', param_x_vals=np.linspace(0.1, 2.0, 32),
    #    param_y_name='EMA_SPEED', param_y_vals=np.linspace(0.01, 1.0, 32)
    #)
    sweep = Sweep2D(
        param_x_name='conn_count', param_x_vals=np.linspace(100, 3000, 32),
        param_y_name='c_moment', param_y_vals=np.linspace(0.00005, 1.0, 32)
    )
    # Example: Look how easy it is to sweep MOMENTUM and LR now!
    #sweep = Sweep2D(
    #    param_x_name='LR', param_x_vals=np.linspace(0.0001, 0.1, 32),
    #    param_y_name='MOMENTUM', param_y_vals=np.linspace(0.1, 0.99, 32)
    #)
    #
    #fitness, mean_dist, var_dist, agent_pos_hist, food_pos_hist, eval_start = run_tracking_task(sweep, dim=16)
    #visualize_sweep(sweep, fitness, mean_dist, var_dist, agent_pos_hist, food_pos_hist, eval_start)

    #sweep = Sweep2D(
    #    param_x_name='EMA_SPEED', param_x_vals=np.linspace(0.1, 0.999, 32),
    #    param_y_name='LR', param_y_vals=np.linspace(0.0001, 1.0, 32),
    #)
    #
    #fitness, mean_dist, var_dist, agent_pos_hist, food_pos_hist, eval_start = run_tracking_task(sweep, dim=16)
    #visualize_sweep(sweep, fitness, mean_dist, var_dist, agent_pos_hist, food_pos_hist, eval_start)

    
    
    fitness, mean_dist, var_dist, agent_pos_hist, food_pos_hist, eval_start = run_tracking_task(sweep, dim=16)
    visualize_sweep(sweep, fitness, mean_dist, var_dist, agent_pos_hist, food_pos_hist, eval_start)