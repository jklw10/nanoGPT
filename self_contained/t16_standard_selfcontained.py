import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import copy
import math
from types import SimpleNamespace
from dataclasses import dataclass

class CustomHebbianLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, w_scale, untanh_slope, error_ema, ema_speed, rew_sens, n_scale, step_counter, speed, width):
        norm_weight = F.normalize(weight, p=2, dim=-2) * w_scale
        #todo, try different norms.
        raw_pred = torch.einsum('bnd,bnde->bne', x, norm_weight)
        
        bound = math.sqrt(2.0 + math.sqrt(2.0))
        out = buntanh(raw_pred, untanh_slope, bound)
    
        ctx.save_for_backward(x, weight, error_ema, ema_speed, rew_sens, n_scale, step_counter, speed, width)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, error_ema, ema_speed, rew_sens, n_scale, step_counter, speed, width = ctx.saved_tensors
        
        error_signal = grad_output

        with torch.no_grad():
            new_ema = error_ema * (1.0 - ema_speed) + (error_signal * ema_speed)
            error_ema.copy_(new_ema)

            E_curr = F.softmax(error_signal, dim=-1)
            advantage = E_curr - error_ema
            plasticity = 1.0 + rms_norm(advantage.unsqueeze(3) * rew_sens)

            noisy_state = noised_input(x, error_signal, n_scale)
            
            lg1 = rms_norm(torch.einsum('bni,bnj->bnij', error_signal, noisy_state))
            grad_weight = plasticity * lg1
            
            center = (step_counter * speed) % weight.shape[-1]
            g_mask = wrapping_gaussian_kernel2(grad_weight, center, width)
            grad_weight = grad_weight * g_mask

        return None, grad_weight, None, None, None, None, None, None, None, None, None

class NodeParams(nn.Module):
    def __init__(self, batch_size, num_nodes, dim, device):
        super().__init__() 
        self.D = dim
        self.device = device
        
        base_weight = torch.randn(batch_size, num_nodes, dim, dim, device=device) / math.sqrt(self.D)
        self.weight = nn.Parameter(base_weight)
        self.register_buffer('error_ema', 0.01 + 0.01 * torch.randn(batch_size, num_nodes, dim, device=device))
        
        # Defaults
        self.untanh_slope = torch.tensor(1.0, device=device)
        self.w_scale = torch.tensor(1.0, device=device)
        self.ema_speed = torch.tensor(0.2, device=device)
        self.rew_sens = torch.tensor(1.36, device=device)
        self.n_scale = torch.tensor(1.5, device=device)
        self.speed = torch.tensor(0.5, device=device)
        self.width = torch.tensor(0.01, device=device)
     
    def forward(self, x, step_counter):
        # Apply the custom Autograd rule
        return CustomHebbianLinear.apply(
            x, self.weight, self.w_scale, self.untanh_slope,
            self.error_ema, self.ema_speed, self.rew_sens, self.n_scale,
            step_counter, self.speed, self.width
        )
        
    def apply_hyperparams(self, config):
        self.untanh_slope = config.UNTANH.view(-1, 1, 1)
        self.w_scale = config.w_scale.view(-1, 1, 1, 1)
        
        if hasattr(config, 'EMA_SPEED'):
            self.ema_speed = config.EMA_SPEED.view(-1, 1, 1)
            self.rew_sens = config.FIXED_REWARD_SENSITIVITY.view(-1, 1, 1, 1)
            self.n_scale = config.NOISE_SCALE.view(-1, 1, 1)
            self.speed = config.speed.view(-1, 1, 1, 1)
            self.width = config.width.view(-1, 1, 1, 1)

class MultiHeadTriadNodeNet(nn.Module):
    def __init__(self, batch_size, num_nodes, dim, device):
        super().__init__()
        self.N = num_nodes
        self.D = dim
        self.H = dim // 4
        self.head_dim = 4
        self.device = device
        
        self.w1 = NodeParams(batch_size, num_nodes, dim, device)
        self.w2 = NodeParams(batch_size, num_nodes, dim, device)
        self.w3 = NodeParams(batch_size, num_nodes, dim, device)
        
        self.register_buffer('state', 1e-5 * torch.randn(batch_size, self.N, dim, device=device))
        self.register_buffer('output', 1e-5 * torch.randn(batch_size, self.N, dim, device=device))
        self.register_buffer('A_ema', torch.zeros(batch_size, self.H, self.N, self.N, device=device))
        self.register_buffer('A_mask', torch.ones(batch_size, self.H, self.N, self.N, device=device))
        self.current_input_nodes = 0
        self.register_buffer('step_counter', torch.tensor(0.0, device=device))
        
        self.c_moment = 0.8
        self.a_variance = 0.0026

    def apply_hyperparams(self, config):
        if hasattr(config, 'c_moment'):
            self.c_moment = config.c_moment.view(-1, 1, 1, 1)
        if hasattr(config, 'a_variance'):
            self.a_variance = config.a_variance.view(-1, 1, 1, 1)
            
        self.w1.apply_hyperparams(config)
        self.w2.apply_hyperparams(config)
        self.w3.apply_hyperparams(config)

    def forward(self, env_input):
        self.step_counter.add_(1.0)

        batch_size, input_dim = env_input.shape
        num_input_nodes = math.ceil(input_dim / self.D)
        
        # 1. OUT-OF-PLACE Mask Updates (Fixes Inductor slicing bugs)
        if num_input_nodes != self.current_input_nodes:
            new_mask = torch.ones(batch_size, self.H, self.N, self.N, device=self.device)
            new_mask[:, :, :num_input_nodes, :] = 0.0 
            self.A_mask = new_mask # Reassigns the buffer
            
            # Clone avoids modifying the old A_ema memory saved in Autograd
            new_ema = self.A_ema.clone()
            new_ema[:, :, num_input_nodes, :num_input_nodes] += 3.0 / num_input_nodes
            self.A_ema = new_ema 
            
            self.current_input_nodes = num_input_nodes

        queries    = self.w3(self.state, self.step_counter)
        keys       = self.w2(self.state, self.step_counter) 
        prediction = self.w1(self.state, self.step_counter) 

        Q = queries.view(batch_size, self.N, self.H, self.head_dim).transpose(1, 2)
        K = keys.view(batch_size, self.N, self.H, self.head_dim).transpose(1, 2)

        raw_A = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        raw_A = rms_norm(raw_A, dim=[-1, -2]) 
        
        # 2. OUT-OF-PLACE EMA Update. 
        # Detaching raw_A here intentionally breaks Backprop-Through-Time (BPTT), 
        # ensuring Q and K only learn from their proxy targets exactly as you designed.
        self.A_ema = (self.A_ema * self.c_moment + raw_A.detach() * (1.0 - self.c_moment)) * self.A_mask
        
        B_H_shape = (batch_size * self.H, self.N, self.N)
        raw_A_3D = self.A_ema.reshape(B_H_shape)
        raw_A_3D = sinkhorn_knopp(raw_A_3D)
        raw_A_norm = raw_A_3D.reshape(batch_size, self.H, self.N, self.N)

        uniform_baseline = 1.0 / (self.N + 1.0)
        sparse_A = torch.where(raw_A_norm > uniform_baseline, raw_A_norm, 0.0)
        
        sparse_A_3D = sparse_A.reshape(B_H_shape)
        A = (sinkhorn_knopp(sparse_A_3D).reshape(batch_size, self.H, self.N, self.N)) * self.A_mask
        
        current_var = A.var(dim=[-1, -2], keepdim=True)
        vd = F.relu(self.a_variance - current_var)
        
        V = self.output.view(batch_size, self.N, self.H, self.head_dim).transpose(1, 2)
        boredom_noise = vd * torch.randn_like(V)
        V = V + boredom_noise 
        
        target = torch.matmul(A, V) 
        target = target.transpose(1, 2).reshape(batch_size, self.N, self.D)
        target = softsign(target)

        # 3. PURELY FUNCTIONAL Environment Injection
        pad_len = num_input_nodes * self.D - input_dim
        if pad_len > 0:
            env_input_padded = F.pad(env_input, (0, pad_len))
        else:
            env_input_padded = env_input
            
        env_input_reshaped = env_input_padded.view(batch_size, num_input_nodes, self.D)
        
        # Using torch.cat instead of `target[:, :] = env_input` strictly prevents inplace errors
        if num_input_nodes < self.N:
            target = torch.cat([env_input_reshaped, target[:, num_input_nodes:, :]], dim=1)
        else:
            target = env_input_reshaped

        # --- Calculate Proxy Target Losses ---
        err1_base = prediction - target.detach()
        loss_w2 = F.mse_loss(keys, err1_base.detach(), reduction='none') 

        err2 = keys.detach() - err1_base.detach()
        loss_w3 = F.mse_loss(queries, err2, reduction='none')

        err3 = queries.detach() - err2
        loss_w1 = F.mse_loss(prediction, target.detach() + err3, reduction='none')

        total_loss = (loss_w1 + loss_w2 + loss_w3).sum()
        
        # 4. OUT-OF-PLACE Buffer Update
        # Re-assigning completely bypasses the inplace Autograd protections!
        self.output = prediction.detach()
        self.state = target.detach()

        return self.output, total_loss
def softsign(x):
    return x / (1 + abs(x))
torch.compile(backend='inductor', mode='max-autotune')
def wrapping_gaussian_kernel2(activation_tensor: torch.Tensor, 
                                     center_offset, 
                                     sigma) -> torch.Tensor:
    
    features = activation_tensor.shape[-1]
    device = activation_tensor.device
    dtype = activation_tensor.dtype
    
    indices = torch.arange(features, dtype=dtype, device=device)
    
    center = center_offset * features
    sigma = sigma * features
    
    # Calculate absolute difference
    diff = torch.abs(indices - center)
    diff = torch.min(diff, features - diff)
    
    kernel = torch.exp(-(diff).pow(2) / (2 * sigma**2))
    return kernel

def rms_norm(x, dim=-1):
    return x * torch.rsqrt(x.pow(2).mean(dim=dim, keepdim=True) + 1e-8)

def buntanh(x, slope, bound):
    return torch.tanh(x - torch.tanh(x)*slope)*bound
def sinkhorn_knopp(log_alpha,iter=5,dim=[-1,-2]):
    device = log_alpha.device
    d1, d2 = dim
    u = torch.zeros(log_alpha.shape[:d1] + (1,), device=device) # (batch, M, 1)
    v = torch.zeros(log_alpha.shape[:d2] + (1, log_alpha.shape[d1]), device=device) # (batch, 1, N)
    for _ in range(iter):
        u = -torch.logsumexp(log_alpha + v, dim=d1, keepdim=True)
        v = -torch.logsumexp(log_alpha + u, dim=d2, keepdim=True)
    return torch.exp(log_alpha + u + v)

def noised_input(input, grad, scale=0.1):
    state_mag = input.abs().sum(dim=[-2,-1], keepdim=True) 
    noise1 = torch.randn_like(input) / (1.0 + state_mag)
    noise2 = torch.randn_like(input) * grad.std() * scale
    return input + noise1 + noise2
phi = 1.61803398875


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
    model_class = MultiHeadTriadNodeNet # Swap with NodeNet, TriadNodeNet, etc
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
    
