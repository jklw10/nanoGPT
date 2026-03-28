
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import standard_nodenet

import copy
import math

from types import SimpleNamespace

from dataclasses import dataclass

@dataclass
class HyperParamConfig:
    UNTANH: torch.Tensor
    EMA_SPEED: torch.Tensor
    FIXED_REWARD_SENSITIVITY: torch.Tensor
    LR: torch.Tensor
    MOMENTUM: torch.Tensor
    weight_decay: torch.Tensor
    NOISE_SCALE: torch.Tensor
    c_moment: torch.Tensor
    w_scale: torch.Tensor
    sharpness: torch.Tensor
    a_variance: torch.Tensor
    speed: torch.Tensor
    width: torch.Tensor
    lateral_decay: torch.Tensor

param_space = {
    'UNTANH': {'low': 0.1, 'high': 3.0},
    'EMA_SPEED': {'low': 0.01, 'high': 0.5},
    'FIXED_REWARD_SENSITIVITY': {'low': 0.1, 'high': 2.0},
    'LR': {'low': 0.001, 'high': 0.05},
    'MOMENTUM': {'low': 0.1, 'high': 0.9},
    'weight_decay': {'low': 0.001, 'high': 0.1},
    'NOISE_SCALE': {'low': 0.1, 'high': 1.5},
    'c_moment': {'low': 0.1, 'high': 0.99},
    'w_scale': {'low': 0.1, 'high': 2.0},
    'sharpness': {'low': 0.5, 'high': 5.0},
    'a_variance': {'low': 0.0, 'high': 1.0}
}

defaults = {
    "speed": 0.5,
    "width": 0.01,
    "lateral_decay": 0.01,
    "EMA_SPEED": 0.20310950837532799,
    "FIXED_REWARD_SENSITIVITY": 1.360218046607915,
    "LR": 0.005720721430830338,
    "MOMENTUM": 0.999,
    "weight_decay": 0.021698750753204772,
    "NOISE_SCALE": 1.5,
    "c_moment": 0.8021549684435622,
    "w_scale": 0.6340345603476942,
    "sharpness": 1.0390895413424572,
    "a_variance": 0.002633467302632888
}


class Sweep2D:
    def __init__(self, param_x_name, param_x_vals, param_y_name, param_y_vals, device):
        self.p_x_name = param_x_name
        self.p_y_name = param_y_name
        
        self.vals_x = torch.tensor(param_x_vals, dtype=torch.float32, device=device)
        self.vals_y = torch.tensor(param_y_vals, dtype=torch.float32, device=device)
        
        self.grid_x = len(self.vals_x)
        self.grid_y = len(self.vals_y)
        self.batch_size = self.grid_x * self.grid_y
        
        grid_y_mat, grid_x_mat = torch.meshgrid(self.vals_y, self.vals_x, indexing='ij')
        
        self.grid_x_flat = grid_x_mat.flatten()
        self.grid_y_flat = grid_y_mat.flatten()
        
    def get_agent_kwargs(self):
        return {
            self.p_x_name: self.grid_x_flat,
            self.p_y_name: self.grid_y_flat
        }

    def get_config(self, defaults, param_space):
        """
        Generates a batched HyperParamConfig matching the sweep grid.
        Swept parameters use the grid, while others use the scalar from defaults.
        """
        config_dict = {}
        all_keys = set(defaults.keys()).union(set(param_space.keys()))
        
        for key in all_keys:
            if key == self.p_x_name:
                val_tensor = self.grid_x_flat.clone()
            elif key == self.p_y_name:
                val_tensor = self.grid_y_flat.clone()
            else:
                # Fallback sequentially: defaults -> middle of param_space -> 0.0
                if key in defaults:
                    val = defaults[key]
                elif key in param_space:
                    val = (param_space[key]['low'] + param_space[key]['high']) / 2.0
                else:
                    val = 0.0
                val_tensor = torch.full((self.batch_size,), float(val), dtype=torch.float32, device=self.vals_x.device)
            
            config_dict[key] = val_tensor
            
        try:
            # Try returning the actual Dataclass so it perfectly fits type hints
            from models.node_agent.nodenet import HyperParamConfig
            import dataclasses
            valid_keys = {f.name for f in dataclasses.fields(HyperParamConfig)}
            filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
            return HyperParamConfig(**filtered_dict)
        except ImportError:
            # Fallback to SimpleNamespace if the file is moved / unavailable
            return SimpleNamespace(**config_dict)
        
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
        
    def visualize(self, mean_fitness, var_fitness):
        perf_grid = self.to_2d_grid(mean_fitness).numpy()
        var_grid = self.to_2d_grid(var_fitness).numpy()

        best_idx = torch.argmax(mean_fitness).item()
        valid_mask = ~torch.isnan(mean_fitness)
        if not valid_mask.any():
            print("CRITICAL ERROR: All agents collapsed to NaN!")
            return

        valid_fitness = mean_fitness[valid_mask]
        threshold = torch.quantile(valid_fitness, 0.99)

        top_indices = torch.nonzero(torch.nan_to_num(mean_fitness, nan=-1.0) >= threshold).squeeze()
        if top_indices.dim() == 0: 
            top_indices = top_indices.unsqueeze(0)

        if len(top_indices) == 0:
            rep_idx = best_idx 
        else:
            top_indices = top_indices[torch.argsort(mean_fitness[top_indices])]
            rep_idx = top_indices[len(top_indices) // 2].item()

        best_x, best_y = self.get_params(best_idx)
        rep_x, rep_y = self.get_params(rep_idx)

        fig = plt.figure(figsize=(16, 12))

        # --- PLOT 1: Mean Fitness ---
        ax1 = plt.subplot(2, 2, 1)
        im1 = ax1.imshow(perf_grid, origin='lower', extent=self.get_extent(), aspect='auto', cmap='viridis', vmin=0, vmax=1.0)
        ax1.set_title("Mean Fitness Landscape\n(1.0 = Perfect Overlap, Averaged across Seeds)")
        ax1.set_xlabel(f"{self.p_x_name}")
        ax1.set_ylabel(f"{self.p_y_name}")
        ax1.scatter(best_x, best_y, color='cyan', marker='*', s=150, edgecolor='black', label='Best Outlier')
        ax1.scatter(rep_x, rep_y, color='magenta', marker='o', s=80, edgecolor='white', label='Top 1% Rep')
        ax1.legend(loc="upper right")
        fig.colorbar(im1, ax=ax1, label='Fitness')

        # --- PLOT 2: Variance ---
        ax2 = plt.subplot(2, 2, 2)
        im2 = ax2.imshow(var_grid, origin='lower', extent=self.get_extent(), aspect='auto', cmap='inferno')
        ax2.set_title("Seed Instability Landscape\n(Variance of Fitness Across 8 Seeds)")
        ax2.set_xlabel(f"{self.p_x_name}")
        ax2.set_ylabel(f"{self.p_y_name}")
        fig.colorbar(im2, ax=ax2, label='Cross-Seed Variance')

        plt.tight_layout()
        plt.show()

def run_tracking_task(model_constructor, optimizer, device, batch_size, total_steps = 2000, eval_start = 1000 ,  **model_args):
    print(f"Initializing {batch_size}  Agents on {device}...")
    
    brain = model_constructor(batch_size=batch_size, **model_args).to(device)
    brain = torch.compile(brain)
    optimizer = optimizer(
        brain.parameters(), 
        lr=0.005, 
        weight_decay=0.02, 
        betas=(0.9, 0.999)
    )
    initial_snapshots = copy.deepcopy(brain)
    agent_pos = torch.full((batch_size, 2), -3.0, device=device)
    agent_vel = torch.zeros(batch_size, 2, device=device)
    
    eval_steps = total_steps - eval_start
    
    agent_pos_history = torch.zeros((total_steps, batch_size, 2), device=device)
    food_pos_history = torch.zeros((total_steps, 2), device=device)
    dist_history = torch.zeros((eval_steps, batch_size), device=device)
    
    for step in range(total_steps):
        t = step / 100.0
        food_pos = torch.tensor([[math.sin(t) * 2.0, math.sin(t * 2.0) * 1.5]], device=device).expand(batch_size, 2)
        
        diffs = food_pos - agent_pos
        dists = torch.norm(diffs, dim=1, keepdim=True)
        direction = diffs / (dists + 1e-5)
        
        starvation = dists 
        environment = torch.randn(batch_size, 2, device=device) * starvation * 1e-5
        environment[:, 0:2] = direction
        
        output, loss = brain(environment)
        loss.backward()
        optimizer.step()

        motor_out = output[:, 2, 0:2].clone()
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
    return fitness, mean_dist, var_dist, agent_pos_history, food_pos_history, eval_start, brain, initial_snapshots


torch.set_float32_matmul_precision('high')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware: {DEVICE}")
torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class BrainWrapper(nn.Module):
    def __init__(self, model_class, config, num_nodes, dim, batch_size, device):
        super().__init__()
        self.core = model_class(batch_size=batch_size, num_nodes=num_nodes, dim=dim, device=device)
        self.core.apply_hyperparams(config)
        
    def forward(self, x):
        output, loss = self.core(x)
        return output, loss
    
def get_model_constructor(model_class, config, num_nodes, dim):
    """
    Returns a constructor function matching figure8.env's expected signature:
    `constructor(batch_size, **model_args)`
    """
    def constructor(batch_size, **kwargs):
        # We ignore kwargs because all parameters are successfully loaded inside `config`
        return BrainWrapper(model_class, config, num_nodes, dim, batch_size, config.UNTANH.device)
    return constructor

def visualize_best(sweep, fitness, mean_dist, var_dist, agent_pos_history, food_pos_history, eval_start):
    perf_grid = sweep.to_2d_grid(fitness).numpy()
    best_idx = torch.argmax(fitness).item()
    
    threshold = torch.quantile(fitness, 0.99)
    top_indices = torch.nonzero(fitness >= threshold).squeeze()
    if top_indices.dim() == 0:
        top_indices = top_indices.unsqueeze(0)
    top_indices = top_indices[torch.argsort(fitness[top_indices])]
    rep_idx = top_indices[len(top_indices) // 2].item()
    
    best_x, best_y = sweep.get_params(best_idx)
    rep_x, rep_y = sweep.get_params(rep_idx)

    fig = plt.figure(figsize=(20, 6))
    
    ax1 = plt.subplot(1, 3, 1)
    im1 = ax1.imshow(perf_grid, origin='lower', extent=sweep.get_extent(), aspect='auto', cmap='viridis', vmin=0, vmax=1.0)
    ax1.set_title("Variance-Optimized Fitness Landscape")
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
        ax.plot(agent_path[:eval_start, 0], agent_path[:eval_start, 1], color='red', alpha=0.5, linewidth=1.5, label='Initial Exploration')
        sc = ax.scatter(agent_path[eval_start:, 0], agent_path[eval_start:, 1], c=time_colors, cmap='plasma', s=10, alpha=0.8, label='Tracking')
        ax.plot(food_path[eval_start:, 0], food_path[eval_start:, 1], color='black', linestyle='--', linewidth=2, alpha=0.6, label='Target Path')
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
        ax.set_title(f"{title}\n{sweep.p_x_name}: {val_x:.4f} | {sweep.p_y_name}: {val_y:.4f}\nDist: {mean_dist[idx]:.2f} | Var: {var_dist[idx]:.2f}")
        ax.legend(loc="upper right")
        if plot_idx == 1: fig.colorbar(sc, ax=ax, label='Time Steps (Evaluation)')

    plt.suptitle("Agent Trajectories", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #torch.set_grad_enabled(False)
    print(f"Executing on: {DEVICE}")
    
    # 1. Setup Sweep Space
    sweep = Sweep2D(
        param_x_name='UNTANH', param_x_vals=np.linspace(0.5, 1.7, 8),
        param_y_name='w_scale', param_y_vals=np.linspace(0.1, 1.0, 8),
        device=DEVICE
    )
    
    # 2. Automatically generate a full batched HyperParamConfig based on Nodenet's space
    config = sweep.get_config(defaults, param_space)
    
    # 3. Choose the model from nodenet, wrap it in our life-fitting adapter
    model_class = standard_nodenet.MultiHeadTriadNodeNet # Swap with NodeNet, TriadNodeNet, etc
    constructor = get_model_constructor(model_class, config, num_nodes=16, dim=16)
    
    fitness, mean_dist, var_dist, agent_pos_hist, food_pos_hist, eval_start, brain, initial = run_tracking_task(
        model_constructor=constructor,
        optimizer = torch.optim.AdamW,
        device=DEVICE,
        batch_size=sweep.batch_size,
        total_steps=2000,
        eval_start=1000
    )
    visualize_best(sweep,fitness,mean_dist,var_dist,agent_pos_hist, food_pos_hist,eval_start)
    #sweep.visualize(fitness, fitness)
    
