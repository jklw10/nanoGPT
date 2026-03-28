import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import parts.utils as utils
import optuna

# ==========================================
# 0. SETUP
# ==========================================
torch.set_float32_matmul_precision('high')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware: {DEVICE}")
torch.manual_seed(42)

class OptunaBatchEvaluator:
    def __init__(self, study, num_configs, num_seeds, hp_spaces, device=DEVICE):
        self.study = study
        self.num_configs = num_configs
        self.num_seeds = num_seeds
        self.batch_size = num_configs * num_seeds
        self.hp_spaces = hp_spaces
        self.device = device
        
        # Ask Optuna for 'num_configs' different hyperparameter trials at once
        self.trials = [study.ask() for _ in range(num_configs)]
        
        self.hp_tensors = {}
        for hp_name, space in hp_spaces.items():
            vals = []
            for t in self.trials:
                # Generate suggestions based on the specified space
                if space['type'] == 'float':
                    vals.append(t.suggest_float(hp_name, space['low'], space['high'], log=space.get('log', False)))
                elif space['type'] == 'int':
                    vals.append(t.suggest_int(hp_name, space['low'], space['high']))
            
            # Convert to a tensor and repeat for the number of seeds
            val_tensor = torch.tensor(vals, dtype=torch.float32, device=self.device)
            self.hp_tensors[hp_name] = val_tensor.repeat_interleave(self.num_seeds)
            
    def get_agent_kwargs(self):
        return self.hp_tensors


def run_optuna_search():
    # 1. Create the study. We want to maximize the penalized_fitness.
    # Optuna uses Tree-structured Parzen Estimator (TPE) under the hood.
    sampler = optuna.samplers.TPESampler(multivariate=True, constant_liar=True)
    study = optuna.create_study(
        direction="maximize", 
        study_name="TriadicBrain_Optimization",
        sampler=sampler,
    )
    
    
    #hps = {
    #    'EMA_SPEED': 0.05, 
    #    'FIXED_REWARD_SENSITIVITY': 1.0, 
    #    'LR': 0.006, 
    #    'MOMENTUM': 0.4, 
    #    'UNTANH': 0.6, 
    #    'speed': 0.5, 
    #    'width': 0.01, 
    #    'weight_decay': 0.01, 
    #    'lateral_decay': 0.01, 
    #    'NOISE_SCALE': 0.8, 
    #    'G_NORM': 1.0, 
    #    'conn_count': 1500,
    #    'c_moment': 0.5,
    #    'w_scale': 1.0,
    #    'sharpness': 1.0,
    #    'a_variance': 0.01,
    #    'K_connect': num_nodes/2
    #}

    # 2. Define the N-Dimensional search space (Add as many as you want!)
    hp_spaces = {
        #'G_NORM': {'type': 'int', 'low': 0, 'high': 1}, 
        #'lateral_decay': {'type': 'float', 'low': 0.001, 'high': 1.0,'log':True}, 
        
        #'LR': {'type': 'float', 'low': 0.001, 'high': 1.0,'log':True}, 
        #'UNTANH_m': {'type': 'float', 'low': 1.0, 'high': 3.0},
        #'NOISE_SCALE': {'type': 'float', 'low': 0.1, 'high': 1.0,'log':True}, 
        'UNTANH_s': {'type': 'float', 'low': 0.6, 'high': 2.0},
        'EMA_SPEED': {'type': 'float', 'low': 0.001, 'high': 1.0,'log':True}, 
        'FIXED_REWARD_SENSITIVITY': {'type': 'float', 'low': 0.1, 'high': 2.0},
        'MOMENTUM': {'type': 'float', 'low': 0.1, 'high': 1.0,'log':True}, 
        
        'weight_decay': {'type': 'float', 'low': 0.001, 'high': 1.0,'log':True}, 
        'c_moment': {'type': 'float', 'low': 0.1, 'high': 1.0, 'log':True}, 
        'w_scale': {'type': 'float', 'low': 0.01, 'high': 1.0, 'log':True}, 
        'sharpness': {'type': 'float', 'low': 0.1, 'high': 10.0, }, 
        'a_variance': {'type': 'float', 'low': 0.01, 'high': 1.0, 'log':True}, 
    }
    
    # Adjust these based on your VRAM
    num_configs_per_batch = 64  # Number of unique configs per generation
    num_seeds = 16               # Seeds per config
    num_generations = 20         # How many loops of evolutionary search to do
    
    for generation in range(num_generations):
        print("\n" + "="*50)
        print(f" GENERATION {generation + 1}/{num_generations} (Optuna TPE Model)")
        print("="*50)
        
        # Initialize our drop-in replacement
        opt_batch = OptunaBatchEvaluator(study, num_configs_per_batch, num_seeds, hp_spaces)
        
        # Run your exact existing simulation!
        (fitness_runs, penalized_fitness, mean_fitness, var_fitness, 
         fitness_matrix, agent_pos_hist, food_pos_hist, eval_start, brain, initial) = run_tracking_task(opt_batch, dim=16)
        
        # Tell Optuna how each trial did so it learns the landscape
        for i, trial in enumerate(opt_batch.trials):
            # We explicitly pass the scalar that penalizes variance
            study.tell(trial, penalized_fitness[i].item())
            
        print(f"\n🏆 Best Config So Far: {study.best_params}")
        print(f"📈 Best Penalized Fitness: {study.best_value:.5f}")

    print("\nSearch Complete! Generating Importance Plot...")
    
    # Optuna has amazing built-in analysis. This will tell you mathematically 
    # which hyperparameter actually mattered most to survival.
    try:
        import matplotlib.pyplot as plt
        optuna.visualization.matplotlib.plot_param_importances(study)
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Could not plot parameter importances. (Requires matplotlib/plotly)")

    return study
class Sweep2D:
    def __init__(self, param_x_name, param_x_vals, param_y_name, param_y_vals, num_seeds=5, device=DEVICE):
        self.p_x_name = param_x_name
        self.p_y_name = param_y_name
        self.num_seeds = num_seeds
        self.vals_x = torch.tensor(param_x_vals, dtype=torch.float32, device=device)
        self.vals_y = torch.tensor(param_y_vals, dtype=torch.float32, device=device)
        self.grid_x = len(self.vals_x)
        self.grid_y = len(self.vals_y)
        self.num_configs = self.grid_x * self.grid_y
        
        # Batch size is now Grid X * Grid Y * Num Seeds
        self.batch_size = self.num_configs * self.num_seeds
        grid_y_mat, grid_x_mat = torch.meshgrid(self.vals_y, self.vals_x, indexing='ij')
        
        # Repeat each configuration num_seeds times sequentially
        self.grid_x_flat = grid_x_mat.flatten().repeat_interleave(self.num_seeds)
        self.grid_y_flat = grid_y_mat.flatten().repeat_interleave(self.num_seeds)
        
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
# 1. BRAIN DEFINITION
# ==========================================
class TriadPiece(nn.Module):
    def __init__(self, dim, batch_size, defaults, num_nodes=10, **hp_grids):
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

        self.register_buffer('dynamic_w_scale', torch.ones(1, device=DEVICE))#*self.get_dyn_scale())
        w_scale = self.hp('w_scale', 4)
        self.weight.copy_(utils.rms_norm(self.weight,dim=[-1,-2]) * w_scale*self.dynamic_w_scale) 

        self.register_buffer('g_ema', torch.randn_like(self.weight) * 0.01) 
        self.register_buffer('E_baseline', 0.01 + 0.01 * torch.randn(batch_size, num_nodes, dim, device=DEVICE))
        
        #self.calibrate_w_scale()
        
    def hp(self, name, ndim):
        val = getattr(self, name)
        if isinstance(val, torch.Tensor):
            return val.view(-1, *([1] * (ndim - 1)))
        return val
    
    def get_dyn_scale(self):
        untanh = self.hp('UNTANH_s', 4)
        
        if not isinstance(untanh, torch.Tensor):
            untanh = torch.tensor(untanh, device=DEVICE)
        diff = torch.abs(untanh - 1.0) + 1e-4 
        optimal_curve = 2.5 * torch.pow(diff, 1.0/3.0)
        return optimal_curve# 1.0+abs(1- untanh**2)
    
    def forward(self, x):
        untanh_s = self.hp('UNTANH_s', 3)
        untanh_m = self.hp('UNTANH_m', 3)
        raw_pred = torch.einsum('bnj,bnjk->bnk', x, self.weight)
        prediction = utils.buntanh(raw_pred, untanh_s, untanh_m)
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
    def __init__(self, dim, batch_size, num_nodes=32, **hp_grids):
        super().__init__()
        self.N = num_nodes
        self.D = dim
        self.batch_size = batch_size
        
        hps = {
            'EMA_SPEED': 0.05, 
            'FIXED_REWARD_SENSITIVITY': 1.0, 
            'LR': 0.015, 
            'MOMENTUM': 0.4, 
            'UNTANH_s': 0.6, 
            'UNTANH_m': math.sqrt(2+math.sqrt(2)), 
            'speed': 0.5, 
            'width': 0.01, 
            'weight_decay': 0.01, 
            'lateral_decay': 0.01, 
            'NOISE_SCALE': 0.8, 
            'G_NORM': 1.0, 
            'conn_count': 1500,
            'c_moment': 0.21,
            'w_scale': 1.0,
            'sharpness': 2.0,
            'a_variance': 0.05,
            'K_connect': num_nodes/2
        }

        hps.update(hp_grids)
        
        hps['LR'] = hps['w_scale'] * 0.024
        
        reward_sens = hps['FIXED_REWARD_SENSITIVITY']

        hps['untanh_m'] = 0.287 / reward_sens 

        hps['noise_scale'] = hps['untanh_m'] * 0.582

        for k, v in hps.items():
            if isinstance(v, torch.Tensor):
                self.register_buffer(k, v.flatten())
            else:
                setattr(self, k, v)
        
        self.w1 = TriadPiece(dim, batch_size,hps, num_nodes, **hp_grids)
        self.w2 = TriadPiece(dim, batch_size,hps, num_nodes, **hp_grids)
        self.w3 = TriadPiece(dim, batch_size,hps, num_nodes, **hp_grids)
        
        self.register_buffer('state', 1e-5*torch.randn(batch_size, self.N, dim, device=DEVICE))
        self.register_buffer('output', 1e-5*torch.randn(batch_size, self.N, dim, device=DEVICE))
        self.register_buffer('target', 1e-5*torch.randn(batch_size, self.N, dim, device=DEVICE))
        self.register_buffer('step_counter', torch.tensor(0.0, device=DEVICE))
        
        self.register_buffer('A', torch.randn(batch_size, self.N, self.N, device=DEVICE))
        self.register_buffer('A_ema', torch.zeros(batch_size, self.N, self.N, device=DEVICE))
        self.A_ema[:, 2, 0] += 3.0  
        self.A_ema[:, 2, 1] -= 2.0 
        self.A = F.softmax(self.A, dim=2)  
        self.register_buffer('node_pos',torch.randn(batch_size, self.N, 3)) 

    def get_A(self):
        pos = self.node_pos 
        diff = pos.unsqueeze(2) - pos.unsqueeze(1) 
        dist = diff.norm(dim=-1) 
        attention_sharpness = self.hp('sharpness', 3)
        return utils.sinkhorn_knopp(-dist * attention_sharpness)

    def hp(self, name, ndim):
        val = getattr(self, name)
        if isinstance(val, torch.Tensor):
            return val.view(-1, *([1] * (ndim - 1)))
        return val

    def step(self, eye_env, stomach_env):
        self.step_counter += 1.0
        
        queries    = self.w3(self.state)
        keys       = self.w2(self.state) 
        prediction = self.w1(self.state) 

        raw_A = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(self.D)
        raw_A = utils.rms_norm(raw_A,dim=[-1,-2])
        routing_momentum = self.hp('c_moment', 3)
        
        if self.step_counter.item() == 1.0:
            self.A_ema.copy_(raw_A)  
        else:
            self.A_ema.copy_(self.A_ema * routing_momentum + raw_A * (1.0 - routing_momentum))
        
        raw_A = self.A_ema
        self.A[:, 0:2, :] = 0.0
        
        raw_A = utils.sinkhorn_knopp(raw_A)
        uniform_baseline = 1.0 / (self.N+1.0)
        above_avg_mask = raw_A > uniform_baseline
        sparse_A = torch.where(above_avg_mask, raw_A, torch.tensor(float(0), device=DEVICE))
        
        A_try = sparse_A
        #A_try = utils.sinkhorn_knopp(sparse_A)
        self.A.copy_(A_try)
        self.A[:, 0:2, :] = 0.0
        a_var_goal = self.hp('a_variance', 3)
        current_var = self.A.var(dim=-1, keepdim=True) 
        vd = F.relu(a_var_goal - current_var)
        noise = vd * torch.randn_like(self.output)
        target = torch.bmm(self.A, self.output+noise)
        target[:, 0, :] = eye_env
        target[:, 1, :] = stomach_env

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
# 2. RUNNER & RENDERER
# ==========================================

def run_tracking_task(sweep, dim=16):
    batch_size = sweep.batch_size
    #print(f"Initializing {batch_size} Agents ({sweep.grid_x}x{sweep.grid_y} Grid, {sweep.num_seeds} Seeds/Point) on {DEVICE}...")
    
    brain = PureTriadicBrain(dim=dim, batch_size=batch_size, **sweep.get_agent_kwargs()).to(DEVICE)
    brain = torch.compile(brain)
    initial_snapshots = copy.deepcopy(brain)
    agent_pos = torch.full((batch_size, 2), -3.0, device=DEVICE)
    agent_vel = torch.zeros(batch_size, 2, device=DEVICE)
    
    total_steps = 2000
    eval_start = 1000 
    eval_steps = total_steps - eval_start
    
    agent_pos_history = torch.zeros((total_steps, batch_size, 2), device=DEVICE)
    food_pos_history = torch.zeros((total_steps, 2), device=DEVICE)
    dist_history = torch.zeros((eval_steps, batch_size), device=DEVICE)
    
    for step in range(total_steps):
        t = step / 100.0
        food_pos = torch.tensor([[math.sin(t) * 2.0, math.sin(t * 2.0) * 1.5]], device=DEVICE).expand(batch_size, 2)
        
        diffs = food_pos - agent_pos
        dists = torch.norm(diffs, dim=1, keepdim=True)
        direction = diffs / (dists + 1e-5)
        
        eye_env = torch.zeros(batch_size, brain.D, device=DEVICE)
        eye_env[:, 0:2] = direction
        
        starvation = dists 
        stomach_env = torch.randn(batch_size, brain.D, device=DEVICE) * starvation * 1e-5
        
        brain.step(eye_env, stomach_env)
        
        motor_out = brain.output[:, 2, 0:2].clone()
        agent_vel = (agent_vel * 0.85) + (motor_out * 0.2)
        agent_pos += agent_vel
        agent_pos = torch.clamp(agent_pos, -4.0, 4.0)
        
        agent_pos_history[step] = agent_pos
        food_pos_history[step] = food_pos[0]
        
        if step >= eval_start:
            dist_history[step - eval_start] = dists.squeeze()

    # Time-based mean and variance for each run
    mean_dist_time = dist_history.mean(dim=0).cpu()
    var_dist_time = dist_history.var(dim=0).cpu()
    fitness_runs = torch.exp(-(mean_dist_time + var_dist_time)) # Shape: (B,)
    
    # Reshape fitness array to (num_configs, num_seeds)
    fitness_matrix = fitness_runs.view(sweep.num_configs, sweep.num_seeds)
    
    # Aggregated stats across seeds
    mean_fitness = fitness_matrix.mean(dim=1)
    var_fitness = fitness_matrix.var(dim=1)
    
    # Variance Penalized Fitness: Higher means consistently good performance.
    penalized_fitness = mean_fitness - var_fitness

    return fitness_runs, penalized_fitness, mean_fitness, var_fitness, fitness_matrix, agent_pos_history, food_pos_history, eval_start, brain, initial_snapshots


def visualize_sweep(sweep, penalized_fitness, mean_fitness, var_fitness, fitness_matrix, agent_pos_history, food_pos_history, eval_start):
    perf_grid = sweep.to_2d_grid(penalized_fitness).numpy()
    var_grid = sweep.to_2d_grid(var_fitness).numpy()
    
    best_config_idx = torch.argmax(penalized_fitness).item()
    
    threshold = torch.quantile(penalized_fitness, 0.99)
    top_indices = torch.nonzero(penalized_fitness >= threshold).squeeze()
    if top_indices.dim() == 0: top_indices = top_indices.unsqueeze(0)
    top_indices = top_indices[torch.argsort(penalized_fitness[top_indices])]
    rep_config_idx = top_indices[len(top_indices) // 2].item()
    
    best_x, best_y = sweep.get_params(best_config_idx)
    rep_x, rep_y = sweep.get_params(rep_config_idx)

    fig = plt.figure(figsize=(24, 6))
    
    # 1. Variance-Penalized Fitness Landscape
    ax1 = plt.subplot(1, 4, 1)
    im1 = ax1.imshow(perf_grid, origin='lower', extent=sweep.get_extent(), aspect='auto', cmap='viridis')
    ax1.set_title("Variance-Penalized Fitness Landscape")
    ax1.set_xlabel(f"{sweep.p_x_name}")
    ax1.set_ylabel(f"{sweep.p_y_name}")
    ax1.scatter(best_x, best_y, color='cyan', marker='*', s=150, edgecolor='black', label='Best Config')
    ax1.scatter(rep_x, rep_y, color='magenta', marker='o', s=80, edgecolor='white', label='Top 1% Rep')
    ax1.legend(loc="upper right")
    fig.colorbar(im1, ax=ax1)

    # 2. Instability (Variance) Heatmap
    ax2 = plt.subplot(1, 4, 2)
    im2 = ax2.imshow(var_grid, origin='lower', extent=sweep.get_extent(), aspect='auto', cmap='inferno')
    ax2.set_title("Instability (Fitness Variance Across Seeds)")
    ax2.set_xlabel(f"{sweep.p_x_name}")
    ax2.set_ylabel(f"{sweep.p_y_name}")
    ax2.scatter(best_x, best_y, color='cyan', marker='*', s=150, edgecolor='black')
    fig.colorbar(im2, ax=ax2)

    eval_steps = agent_pos_history.shape[0] - eval_start
    food_path = food_pos_history.cpu().numpy()
    time_colors = np.linspace(0, 1, eval_steps)
    
    # Function to grab the "Median Seed" from a particular config for plotting
    def get_median_seed_idx(config_idx):
        config_fitnesses = fitness_matrix[config_idx]
        median_val = config_fitnesses.median()
        seed_offset = torch.argmin((config_fitnesses - median_val).abs()).item()
        return config_idx * sweep.num_seeds + seed_offset

    best_run_idx = get_median_seed_idx(best_config_idx)
    rep_run_idx = get_median_seed_idx(rep_config_idx)

    plots =[
        (plt.subplot(1, 4, 3), best_run_idx, best_x, best_y, "Best Config (Median Seed)"),
        (plt.subplot(1, 4, 4), rep_run_idx, rep_x, rep_y, "Top 1% Config (Median Seed)")
    ]
    
    for plot_idx, (ax, idx, val_x, val_y, title) in enumerate(plots):
        agent_path = agent_pos_history[:, idx].cpu().numpy()
        ax.plot(agent_path[:eval_start, 0], agent_path[:eval_start, 1], color='red', alpha=0.5, linewidth=1.5, label='Initial Exploration')
        sc = ax.scatter(agent_path[eval_start:, 0], agent_path[eval_start:, 1], c=time_colors, cmap='plasma', s=10, alpha=0.8, label='Tracking')
        ax.plot(food_path[eval_start:, 0], food_path[eval_start:, 1], color='black', linestyle='--', linewidth=2, alpha=0.6, label='Target Path')
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
        
        # Display the real fitness for the chosen seed vs penalized config average
        seed_fitness = fitness_matrix.flatten()[idx].item()
        ax.set_title(f"{title}\n{sweep.p_x_name}: {val_x:.4f} | {sweep.p_y_name}: {val_y:.4f}\nSeed Fitness: {seed_fitness:.2f}")
        ax.legend(loc="upper right")
        if plot_idx == 1: fig.colorbar(sc, ax=ax, label='Time Steps (Evaluation)')

    plt.suptitle("Hyperparameter Sweep with Multi-Seed Variance Analysis", fontsize=16)
    plt.tight_layout()
    plt.show()

def diagnose_batch_health(brain, initial_brain, fitness_runs):
    print("\n" + "="*80)
    print("🧠 RUNNING BATCH AUTOPSY & FEATURE CORRELATION")
    print("="*80)
    
    fitness_runs = fitness_runs.to(brain.w1.weight.device)
    B = brain.batch_size
    metrics = {}
    eps = 1e-7
    
    if initial_brain is not None:
        metrics['init_motor_state_mag'] = initial_brain.state[:, 2, :].norm(dim=1)
        metrics['init_eye_state_mag'] = initial_brain.state[:, 0, :].norm(dim=1)

        w1_init = initial_brain.w1.weight.view(B, -1)
        w1_final = brain.w1.weight.view(B, -1)
        metrics['w1_travel_dist'] = (w1_final - w1_init).norm(dim=1)
        metrics['w1_init_cosim_final'] = F.cosine_similarity(w1_init, w1_final, dim=1)

        pos = initial_brain.node_pos
        diff = pos.unsqueeze(2) - pos.unsqueeze(1)
        dist = diff.norm(dim=-1)
        metrics['init_motor_eye_dist'] = dist[:, 2, 0]
        metrics['init_motor_stomach_dist'] = dist[:, 2, 1]
        metrics['init_min_nn_dist'] = dist.topk(2, dim=-1, largest=False).values[:,:,1].min(dim=1).values

        motor_init = initial_brain.state[:, 2, :]
        eye_init = initial_brain.state[:, 0, :]
        metrics['init_motor_eye_alignment'] = F.cosine_similarity(motor_init, eye_init, dim=1)

        init_pred = initial_brain.w1(initial_brain.state)
        metrics['init_prediction_mag'] = init_pred.abs().mean(dim=(1,2))
        metrics['init_prediction_variance'] = init_pred.var(dim=(1,2))

    metrics['Motor_Listens_To_Eye'] = brain.A[:, 2, 0]
    metrics['Motor_Listens_To_Stomach'] = brain.A[:, 2, 1]
    metrics['Avg_Self_Attention'] = torch.diagonal(brain.A, dim1=1, dim2=2).mean(dim=1)
    
    A_norm = brain.A / (brain.A.sum(dim=-1, keepdim=True) + eps)
    metrics['Routing_Entropy'] = -(A_norm * torch.log(A_norm + eps)).sum(dim=-1).mean(dim=-1)
    metrics['Route_Variance'] = brain.A.view(B,-1).var(dim=-1)

    w1_flat = brain.w1.weight.view(B, -1)
    w2_flat = brain.w2.weight.view(B, -1)
    w3_flat = brain.w3.weight.view(B, -1)
    
    metrics['CosSim_W1_W2'] = F.cosine_similarity(w1_flat, w2_flat, dim=1)
    metrics['CosSim_W2_W3'] = F.cosine_similarity(w2_flat, w3_flat, dim=1)

    metrics['Output_Saturation_Pct'] = (brain.output.abs()).mean(dim=(1,2))
    metrics['Mean_State_Magnitude'] = brain.state[:, 2:, :].abs().mean(dim=(1,2))
    
    metrics['w1_E_Baseline_Mag'] = brain.w1.E_baseline.abs().mean(dim=(1,2))
    metrics['w2_E_Baseline_Mag'] = brain.w2.E_baseline.abs().mean(dim=(1,2))
    metrics['w3_E_Baseline_Mag'] = brain.w3.E_baseline.abs().mean(dim=(1,2))
    metrics['W1_Norm'] = w1_flat.norm(dim=1)
    metrics['W2_Norm'] = w2_flat.norm(dim=1)

    threshold = torch.quantile(fitness_runs, 0.95)
    top_mask = fitness_runs >= threshold
    bot_mask = fitness_runs < threshold
    
    results =[]
    for name, vals in metrics.items():
        vx = vals - vals.mean()
        vy = fitness_runs - fitness_runs.mean()
        corr = (vx * vy).sum() / (vx.norm() * vy.norm() + eps)
        
        top_mean = vals[top_mask].mean().item()
        bot_mean = vals[bot_mask].mean().item()
        
        results.append((corr.item(), name, top_mean, bot_mean))
        
    results.sort(key=lambda x: abs(x[0]), reverse=True)
    
    print(f"{'Metric Name':<28} | {'Corr w/ Fitness':<17} | {'Top 5% Mean':<12} | {'Bottom 95% Mean':<12}")
    print("-" * 78)
    for corr, name, top_mean, bot_mean in results:
        marker = "🔥" if abs(corr) > 0.4 else "  "
        print(f"{marker} {name:<25} | {corr:>15.4f}   | {top_mean:>12.4f} | {bot_mean:>12.4f}")
        
    return metrics

def search2d():
    torch.set_grad_enabled(False) 
    
    #hps = {
    #    'EMA_SPEED': 0.05,
    #    'FIXED_REWARD_SENSITIVITY': 1.0,
    #    'LR': 0.006,
    #    'MOMENTUM': 0.4,
    #    'UNTANH': 0.6, 
    #    'speed': 0.5,
    #    'width': 0.1,
    #    'weight_decay': 0.01,
    #    'lateral_decay': 0.01,
    #    'NOISE_SCALE': 0.8, 
    #    'G_NORM': 1.0,       
    #    'conn_count': 1500,
    #    'c_moment': 0.99,
    #    'w_scale': 1.0,
    #    'sharpness': 1.0,
    #    'K_connect': num_nodes/2
    #     a_variance
    #}
    sweep = Sweep2D(
        param_x_name='UNTANH', param_x_vals=np.linspace(0.5, 1.7, 16),
        param_y_name='LR', param_y_vals=np.linspace(0.0, 0.1, 16),
        num_seeds=5 
    )
    
    # Run the sweep and capture aggregated data
    (fitness_runs, penalized_fitness, mean_fitness, var_fitness, 
     fitness_matrix, agent_pos_hist, food_pos_hist, eval_start, brain, initial) = run_tracking_task(sweep, dim=16)
    
    # Autopsy checks standard `fitness_runs` (batch flattened individual run performance)
    diagnose_batch_health(brain, initial, fitness_runs)
    
    # Visualize mapping the Instability vs Penalized performance over spatial domains
    visualize_sweep(sweep, penalized_fitness, mean_fitness, var_fitness, fitness_matrix, agent_pos_hist, food_pos_hist, eval_start)


def analyze_hyperparam_relations(study):
    print("\n" + "="*60)
    print("🧠 EXTRACTING HYPERPARAMETER RELATIONSHIPS")
    print("="*60)
    
    # Get all trials as a Pandas DataFrame
    df = study.trials_dataframe()
    
    # Filter only the trials that actually finished and were in the Top 10% of performance
    df = df[df['state'] == 'COMPLETE']
    threshold = df['value'].quantile(0.90)
    top_df = df[df['value'] >= threshold]
    
    # Extract just the hyperparameter columns
    param_cols = [c for c in top_df.columns if c.startswith('params_')]
    clean_names = {c: c.replace('params_', '') for c in param_cols}
    top_params = top_df[param_cols].rename(columns=clean_names)
    
    # Calculate how strongly every parameter correlates with every other parameter
    corr_matrix = top_params.corr()
    
    # Find the strongest relationships to help you reduce HP count to 0!
    print("\n🔥 STRONGEST MATHEMATICAL COUPLINGS (Top 10% Trials):")
    links =[]
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.4:  # Threshold for a significant relationship
                links.append((abs(corr_val), corr_val, col1, col2))
                
    links.sort(reverse=True)
    for _, corr_val, col1, col2 in links:
        direction = "PROPORTIONAL (+)" if corr_val > 0 else "INVERSELY PROPORTIONAL (-)"
        print(f"[{corr_val:>5.2f}] {col1:<25} <---> {col2:<25} ({direction})")
        
    print("\n💡 Tip: If two variables are highly coupled (e.g. > 0.7), you can hardcode one as a multiple of the other and delete it from Optuna!")

    # Plot a beautiful heatmap of these relationships
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
    plt.title("Hyperparameter Dependency Matrix (Top 10% Performing Models)")
    plt.tight_layout()
    plt.show()

    # Optuna's built-in contour plots for the top 3 most important parameters
    try:
        optuna.visualization.matplotlib.plot_contour(study)
        plt.show()
    except Exception as e:
        pass

if __name__ == "__main__":
    torch.set_grad_enabled(False) 
    
    # Run Optuna Search
    study = run_optuna_search()
    analyze_hyperparam_relations(study)