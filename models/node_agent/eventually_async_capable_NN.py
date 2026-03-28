
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
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

FULL_SPACE = {
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

class HPManager:
    def __init__(self, active_tensors, relations, N, D, batch_size, device=DEVICE):
        self.N_val = float(N)
        self.D_val = float(D)
        self.batch_size = batch_size
        self.device = device
        
        self.defaults = {
            'speed': 0.5,
            'width': 0.01,
            'lateral_decay': 0.01,
            "EMA_SPEED": 0.20310950837532799,
            "FIXED_REWARD_SENSITIVITY": 1.360218046607915,
            "LR": 0.005720721430830338,
            "MOMENTUM": 0.2521688557418596,
            "weight_decay": 0.021698750753204772,
            "NOISE_SCALE": 1.5,
            "c_moment": 0.8021549684435622,
            "w_scale": 0.6340345603476942,
            "sharpness": 1.0390895413424572,
            "a_variance": 0.002633467302632888
        }
        self.update(active_tensors, relations)

    def update(self, active_tensors, relations):
        self.tensors = active_tensors
        self.relations = relations
        self.cache = {
            'N': torch.full((self.batch_size,), self.N_val, device=self.device),
            'D': torch.full((self.batch_size,), self.D_val, device=self.device)
        }
    def resolve_all(self):
        res = {}
        for k in FULL_SPACE.keys():
            res[k] = self.get(k, 1)
        for k in self.defaults.keys():
            res[k] = self.get(k, 1)
        return res
    def get(self, name, ndim):
        if name in self.cache:
            val = self.cache[name]
        elif name in self.tensors:
            val = self.tensors[name]
            self.cache[name] = val
        elif name in self.relations:
            rel = self.relations[name]
            eq_type = rel.get('type', 'linear')
            
            if isinstance(rel['master'], list):
                m1_val = self.get(rel['master'][0], 1).flatten()
                m2_val = self.get(rel['master'][1], 1).flatten()
                
                if eq_type == 'multi_power':
                    val = rel['a'] * torch.pow(m1_val + 1e-8, rel['b']) * torch.pow(m2_val + 1e-8, rel['c'])
                else: 
                    val = rel['a'] * m1_val + rel['b'] * m2_val + rel['c']
            else:
                master_val = self.get(rel['master'], 1).flatten() 
                if eq_type == 'power':
                    val = rel['a'] * torch.pow(master_val + 1e-8, rel['b'])
                elif eq_type == 'quadratic':
                    val = rel['a'] * (master_val**2) + rel['b'] * master_val + rel['c']
                else: 
                    val = rel['a'] * master_val + rel['b']
                
            if name in FULL_SPACE:
                val = torch.clamp(val, FULL_SPACE[name]['low'], FULL_SPACE[name]['high'])
            self.cache[name] = val
        else:
            val = torch.tensor(self.defaults.get(name, 1.0), device=self.device).expand(self.cache['N'].shape[0])

        return val.view(-1, *([1] * (ndim - 1)))

class TriadPiece(nn.Module):
    def __init__(self, batch_size, num_nodes, dim):
        super().__init__() 
        self.D = dim
        
        base_weight = torch.randn(1, num_nodes, dim, dim, device=DEVICE) / math.sqrt(self.D)
        self.register_buffer('weight', base_weight.expand(batch_size, -1, -1, -1).clone())
        
        self.register_buffer('g_ema', torch.randn_like(self.weight) * 0.01) 
        self.register_buffer('E_baseline', 0.01 + 0.01 * torch.randn(batch_size, num_nodes, dim, device=DEVICE))
        
    def reset(self, hps):
        self.hp_untanh = hps['UNTANH'].view(-1, 1, 1)
        self.hp_ema_speed = hps['EMA_SPEED'].view(-1, 1, 1)
        self.hp_noise_scale = hps['NOISE_SCALE'].view(-1, 1, 1)

        self.hp_rew_sens = hps['FIXED_REWARD_SENSITIVITY'].view(-1, 1, 1, 1)
        self.hp_wd = hps['weight_decay'].view(-1, 1, 1, 1)
        self.hp_speed = hps['speed'].view(-1, 1, 1, 1)
        self.hp_width = hps['width'].view(-1, 1, 1, 1)
        self.hp_momentum = hps['MOMENTUM'].view(-1, 1, 1, 1)
        self.hp_lr = hps['LR'].view(-1, 1, 1, 1)
        self.hp_w_scale = hps['w_scale'].view(-1, 1, 1, 1)

        base_weight = torch.randn(1, self.weight.shape[1], self.D, self.D, device=DEVICE) / math.sqrt(self.D)
        self.weight.data.copy_(base_weight.expand_as(self.weight))
       
        
        self.weight.data.copy_(rms_norm(self.weight.data, dim=[-1,-2]) * self.hp_w_scale) 
        
        self.g_ema.normal_().mul_(0.01)
        self.E_baseline.normal_().mul_(0.01).add_(0.01)

    def forward(self, x):
        raw_pred = (x.unsqueeze(2) @ self.weight).squeeze(2)
        return buntanh(raw_pred, self.hp_untanh, math.sqrt(2 + math.sqrt(2)))
    
    def get_gradient(self, layer_input, error_signal):
        ema_speed = self.hp_ema_speed
        rew_sens = self.hp_rew_sens
        n_scale = self.hp_noise_scale
        
        E_curr = F.softmax(error_signal, dim=-1)
        self.E_baseline.mul_(1.0 - ema_speed).add_(ema_speed * E_curr)
        advantage = E_curr - self.E_baseline
        plasticity = 1.0 + rms_norm(advantage.unsqueeze(3) * rew_sens)  
        
        noisy_state = noised_input(layer_input, error_signal, n_scale)
        lg1 = rms_norm(-torch.einsum('bni,bnj->bnij', error_signal, noisy_state))
        return (plasticity * lg1) 
    
    def step(self, grad, step_counter):
        wd = self.hp_wd
        grad = grad - (wd * self.weight)
        
        speed = self.hp_speed 
        width = self.hp_width 
        center = (step_counter * speed) % self.D
        g_mask = wrapping_gaussian_kernel2(grad, center, width)
        grad = grad * g_mask
        
        momentum = self.hp_momentum 
        lr = self.hp_lr 

        self.g_ema.mul_(momentum).add_((1.0 - momentum) * grad)
        self.weight.add_(lr * self.g_ema)
        
        w_scale = self.hp_w_scale
        self.weight.data.copy_(rms_norm(self.weight.data, dim=[-1,-2]) * w_scale )

class PureTriadicBrain(nn.Module):
    def __init__(self, batch_size, num_nodes, dim, hp_manager):
        super().__init__()
        self.N = num_nodes
        self.D = dim
        self.batch_size = batch_size
        self.hpm = hp_manager
        
        self.w1 = TriadPiece(batch_size, num_nodes,dim)
        self.w2 = TriadPiece(batch_size, num_nodes,dim)
        self.w3 = TriadPiece(batch_size, num_nodes,dim)
        
        self.register_buffer('state', 1e-5*torch.randn(batch_size, self.N, dim, device=DEVICE))
        self.register_buffer('output', 1e-5*torch.randn(batch_size, self.N, dim, device=DEVICE))
        self.register_buffer('step_counter', torch.tensor(0.0, device=DEVICE))
        
        self.register_buffer('A', torch.randn(batch_size, self.N, self.N, device=DEVICE))
        self.register_buffer('A_ema', torch.zeros(batch_size, self.N, self.N, device=DEVICE))
        self.A_ema[:, 2, 0] += 3.0  
        self.A_ema[:, 2, 1] -= 2.0 
        self.A.data.copy_(F.softmax(self.A, dim=2))
        self.register_buffer('A_mask', torch.ones_like(self.A))
        self.A_mask[:, 0:2, :] = 0.0

    def reset(self, active_tensors, relations):
        # Resolve all parameters to tensors ONCE
        self.hpm.update(active_tensors, relations)
        hps = self.hpm.resolve_all()
        
        U = hps['UNTANH']
        LR = hps['LR']
        FRS = hps['FIXED_REWARD_SENSITIVITY']
        N = torch.full_like(U, self.N) # Number of nodes
        
        if 'w_scale' not in active_tensors:
            w_scale =  0.95 / (U**2)
            w_scale_sub = 1.08 * U + 0.28
            w_scale = torch.where(U<1.0,w_scale_sub,w_scale)
            hps['w_scale'] = torch.clamp(w_scale, 0.001, 10.0)
            
        #if 'weight_decay' not in active_tensors:
        #    wd = 0.024396 * U - 0.580620 * LR + 0.043600
        #    hps['weight_decay'] = torch.clamp(wd, 0.001, 0.1)
        #   
        #if 'EMA_SPEED' not in active_tensors:
        #    ema = 0.051465 * U - 0.112347 * FRS + 0.360587
        #    hps['EMA_SPEED'] = torch.clamp(ema, 0.01, 0.5)
        #    
        #if 'MOMENTUM' not in active_tensors:
        #    mom = -4.031251 * hps['weight_decay'] - 0.024813 * N + 1.018012
        #    hps['MOMENTUM'] = torch.clamp(mom, 0.1, 0.9)
        #    
        #if 'NOISE_SCALE' not in active_tensors:
        #    ns = -6.602821 * LR + 0.737121 * hps['w_scale'] + 0.507005
        #    hps['NOISE_SCALE'] = torch.clamp(ns, 0.1, 1.5)
        #    
        #if 'sharpness' not in active_tensors:
        #    shp = -55.710849 * LR - 24.207488 * hps['weight_decay'] + 5.888201
        #    hps['sharpness'] = torch.clamp(shp, 0.5, 5.0)

        self.hp_c_moment = hps['c_moment'].view(-1, 1, 1)
        self.hp_a_variance = hps['a_variance'].view(-1, 1, 1)
        self.hp_untanh = hps['UNTANH'].view(-1, 1, 1)

        self.w1.reset(hps)
        self.w2.reset(hps)
        self.w3.reset(hps)
        
        self.state.normal_().mul_(1e-5)
        self.output.normal_().mul_(1e-5)
        self.step_counter.fill_(0.0)
        
        self.A.normal_()
        self.A_ema.zero_()
        self.A_ema[:, 2, 0] += 3.0  
        self.A_ema[:, 2, 1] -= 2.0 
        self.A.data.copy_(F.softmax(self.A, dim=2))
        
        self.A_mask.fill_(1.0)
        self.A_mask[:, 0:2, :] = 0.0

    def forward(self, eye_env, stomach_env):
        self.step_counter.add_(1.0)
        
        queries    = self.w3(self.state)
        keys       = self.w2(self.state) 
        prediction = self.w1(self.state) 

        raw_A = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(self.D)
        raw_A = rms_norm(raw_A, dim=[-1,-2])
        
        is_first = (self.step_counter == 1.0).float().view(1, 1, 1)
        new_A_ema = raw_A * is_first + (self.A_ema * self.hp_c_moment + raw_A * (1.0 - self.hp_c_moment)) * (1.0 - is_first)
        self.A_ema.copy_(new_A_ema * self.A_mask)
        
        raw_A = self.A_ema 
        raw_A = sinkhorn_knopp(raw_A)
        uniform_baseline = 0.5 / self.N
        
        sparse_A = torch.where(raw_A > uniform_baseline, raw_A, 0.0)
        
        self.A.copy_(sinkhorn_knopp(sparse_A) * self.A_mask)
        
        current_var = self.A.var(dim=[-1, -2], keepdim=True)
        vd = F.relu(self.hp_a_variance - current_var)
        boredom_noise = vd * torch.randn_like(self.output)

        target = torch.bmm(self.A, self.output + boredom_noise)
        #TODO test thoroughly:
        #target = buntanh(target, self.hp_untanh, math.sqrt(2 + math.sqrt(2)))
        #target = rms_norm(target)
        #target = torch.tanh(target/5)*5
        target = softsign(target)
        target[:, 0, :] = eye_env
        target[:, 1, :] = stomach_env

        err1_base = prediction - target
        err2 = keys - err1_base.detach()
        err3 = queries - err2.detach()
        err1 = prediction - (target + err3.detach())

        self.w1.step(self.w1.get_gradient(self.state, err1), self.step_counter)
        self.w2.step(self.w2.get_gradient(self.state, err2), self.step_counter)
        self.w3.step(self.w3.get_gradient(self.state, err3), self.step_counter)

        self.output.copy_(prediction)
        self.state.copy_(target)
# ==========================================
# 3. TASK RUNNER
# ==========================================
def run_tracking_task(sweep, dim=16, num_seeds=8):
    batch_size = sweep.batch_size
    print(f"Initializing {batch_size} Agents across {num_seeds} global seeds on {DEVICE}...")
    
    active_tensors = sweep.get_agent_kwargs()
    hpm = HPManager(active_tensors, relations={}, N=16, D=dim, batch_size=batch_size, device=DEVICE)
    brain = PureTriadicBrain(batch_size=batch_size, num_nodes=16, dim=dim, hp_manager=hpm).to(DEVICE)
    
    total_steps = 3000
    eval_start = 1000 
    eval_steps = total_steps - eval_start
    
    # Store histories for ALL seeds:[num_seeds, total_steps, batch_size, 2]
    agent_pos_history_all = torch.zeros((num_seeds, total_steps, batch_size, 2), device=DEVICE)
    food_pos_history = torch.zeros((total_steps, 2), device=DEVICE)
    
    all_seed_fitness =[]
    
    print(f"Running {num_seeds} distinct environment initializations...")
    for seed in range(num_seeds):
        torch.manual_seed(seed + 42) # Lock random state so initial weights/noise match perfectly
        
        brain.reset(active_tensors, relations={})
        
        agent_pos = torch.full((batch_size, 2), -3.0, device=DEVICE)
        agent_vel = torch.zeros(batch_size, 2, device=DEVICE)
        dist_history = torch.zeros((eval_steps, batch_size), device=DEVICE)
        
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
            
            base_noise = torch.randn(1, brain.D, device=DEVICE).expand(batch_size, brain.D)
            base_noise=base_noise*wrapping_gaussian_kernel2(base_noise,1.0,0.1)
            stomach_env = base_noise * starvation * 1e-2

            brain(eye_env, stomach_env)
            
            motor_out = brain.output[:, 2, 0:2].clone()
            
            agent_vel = (agent_vel * 0.85) + (motor_out * 0.15)
            agent_pos += agent_vel
            agent_pos = torch.clamp(agent_pos, -4.0, 4.0)
            
            agent_pos_history_all[seed, step] = agent_pos
            if seed == 0:
                food_pos_history[step] = food_pos[0]
            
            if step >= eval_start:
                dist_history[step - eval_start] = dists.squeeze()

        # Original bounded fitness mapping per seed (0 to 1)
        time_mean = dist_history.mean(dim=0)
        time_var = dist_history.var(dim=0)
        seed_fitness = torch.exp(-(time_mean + time_var)) 
        all_seed_fitness.append(seed_fitness)

    # Calculate Mean and Variance of Fitness across the 8 global seeds
    stacked_fitness = torch.stack(all_seed_fitness)  # [num_seeds, batch_size]
    mean_fitness = stacked_fitness.mean(dim=0).cpu() # [batch_size] -> Higher is better
    var_fitness = stacked_fitness.var(dim=0).cpu()   # [batch_size] -> Higher is unstable

    return mean_fitness, var_fitness, agent_pos_history_all, food_pos_history, eval_start# ==========================================
def visualize_sweep(sweep, mean_fitness, var_fitness, agent_pos_history_all, food_pos_history, eval_start):
    perf_grid = sweep.to_2d_grid(mean_fitness).numpy()
    var_grid = sweep.to_2d_grid(var_fitness).numpy()
    
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
    
    best_x, best_y = sweep.get_params(best_idx)
    rep_x, rep_y = sweep.get_params(rep_idx)

    fig = plt.figure(figsize=(16, 12))
    
    # --- PLOT 1: Mean Fitness ---
    ax1 = plt.subplot(2, 2, 1)
    im1 = ax1.imshow(perf_grid, origin='lower', extent=sweep.get_extent(), aspect='auto', cmap='viridis', vmin=0, vmax=1.0)
    ax1.set_title("Mean Fitness Landscape\n(1.0 = Perfect Overlap, Averaged across Seeds)")
    ax1.set_xlabel(f"{sweep.p_x_name}")
    ax1.set_ylabel(f"{sweep.p_y_name}")
    ax1.scatter(best_x, best_y, color='cyan', marker='*', s=150, edgecolor='black', label='Best Outlier')
    ax1.scatter(rep_x, rep_y, color='magenta', marker='o', s=80, edgecolor='white', label='Top 1% Rep')
    ax1.legend(loc="upper right")
    fig.colorbar(im1, ax=ax1, label='Fitness')

    # --- PLOT 2: Variance ---
    ax2 = plt.subplot(2, 2, 2)
    im2 = ax2.imshow(var_grid, origin='lower', extent=sweep.get_extent(), aspect='auto', cmap='inferno')
    ax2.set_title("Seed Instability Landscape\n(Variance of Fitness Across 8 Seeds)")
    ax2.set_xlabel(f"{sweep.p_x_name}")
    ax2.set_ylabel(f"{sweep.p_y_name}")
    fig.colorbar(im2, ax=ax2, label='Cross-Seed Variance')

    # --- PLOT 3 & 4: Trajectories (All Seeds Overlay) ---
    num_seeds = agent_pos_history_all.shape[0]
    eval_steps = agent_pos_history_all.shape[1] - eval_start
    food_path = food_pos_history.cpu().numpy()
    time_colors = np.linspace(0, 1, eval_steps)
    
    plots =[
        (plt.subplot(2, 2, 3), best_idx, best_x, best_y, "Absolute Best Run (All Seeds Overlay)"),
        (plt.subplot(2, 2, 4), rep_idx, rep_x, rep_y, "Top 1% Representative (All Seeds Overlay)")
    ]
    
    for plot_idx, (ax, idx, val_x, val_y, title) in enumerate(plots):
        time_mean = agent_pos_history_all[:, eval_start:, idx, :].cpu()
        food_path_eval = food_pos_history[eval_start:].cpu()
        dists = torch.norm(time_mean - food_path_eval.unsqueeze(0), dim=-1)
        best_seed = torch.argmin(dists.mean(dim=1)).item()

        for seed in range(num_seeds):
            agent_path = agent_pos_history_all[seed, :, idx].cpu().numpy()
            
            if seed == best_seed:
                ax.plot(agent_path[:eval_start, 0], agent_path[:eval_start, 1], color='red', alpha=0.3, linewidth=1.5)
                sc = ax.scatter(agent_path[eval_start:, 0], agent_path[eval_start:, 1], c=time_colors, cmap='plasma', s=10, alpha=0.8)
            else:
                ax.plot(agent_path[eval_start:, 0], agent_path[eval_start:, 1], color='teal', alpha=0.3, linewidth=1.0)
        
        ax.plot(food_path[eval_start:, 0], food_path[eval_start:, 1], color='black', linestyle='--', linewidth=2, alpha=0.6, label='Target Path')
        
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        
        score_text = f"Mean Fit: {mean_fitness[idx]:.3f} | Fit Variance: {var_fitness[idx]:.3f}"
        ax.set_title(f"{title}\n{sweep.p_x_name}: {val_x:.4f} | {sweep.p_y_name}: {val_y:.4f}\n{score_text}")
        
        if plot_idx == 1: 
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label('Time Steps (Seed 0 Evaluation)')

    plt.tight_layout()
    plt.show()

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
        param_x_name='w_scale', param_x_vals=np.linspace(0.01, 0.7, 16),
        param_y_name='UNTANH', param_y_vals=np.linspace(1.0, 3.0, 16)
    )
    
    mean_fit, var_fit, pos_history_all, food_pos, eval_start = run_tracking_task(sweep, dim=24, num_seeds=8)
    visualize_sweep(sweep, mean_fit, var_fit, pos_history_all, food_pos, eval_start)