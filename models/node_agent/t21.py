import os
import json
import random
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import optuna
import parts.utils as utils
from scipy.stats import linregress, mannwhitneyu
import time
from sklearn.linear_model import RANSACRegressor, LinearRegression
import warnings
import itertools
from optuna.importance import MeanDecreaseImpurityImportanceEvaluator
# ==========================================
# 0. SETUP & DYNAMIC HP MANAGER
# ==========================================
torch.set_float32_matmul_precision('high')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware: {DEVICE}")

RELATIONS_FILE = "hp_relations.json"

# The original 11-dimensional search space
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
    """
    Acts as the central nervous system for hyperparameters. 
    If a parameter is active, it pulls from the Optuna tensor.
    If it was collapsed, it calculates the polynomial curve on the fly.
    """
    def __init__(self, active_tensors, relations, N, D, batch_size, device=DEVICE):
        self.tensors = active_tensors
        self.relations = relations
        
        # Base variables that can act as "Masters"
        self.cache = {
            'N': torch.full((batch_size,), float(N), device=device),
            'D': torch.full((batch_size,), float(D), device=device)
        }
        
        # Hardcoded structural defaults
        self.defaults = {
            'speed': 0.5,
            'width': 0.01,
            'lateral_decay': 0.01,
        }

    def get(self, name, ndim):
        if name in self.cache:
            val = self.cache[name]
        elif name in self.tensors:
            val = self.tensors[name]
            self.cache[name] = val
        elif name in self.relations:
            rel = self.relations[name]
            eq_type = rel.get('type', 'linear')
            
            # --- MULTI-VARIATE LAWS ---
            if isinstance(rel['master'], list):
                m1_val = self.get(rel['master'][0], 1).flatten()
                m2_val = self.get(rel['master'][1], 1).flatten()
                
                if eq_type == 'multi_power':
                    # y = a * (m1^b) * (m2^c)
                    val = rel['a'] * torch.pow(m1_val + 1e-8, rel['b']) * torch.pow(m2_val + 1e-8, rel['c'])
                else: # multi_linear
                    # y = a*m1 + b*m2 + c
                    val = rel['a'] * m1_val + rel['b'] * m2_val + rel['c']
            
            # --- SINGLE-VARIATE LAWS ---
            else:
                master_val = self.get(rel['master'], 1).flatten() 
                if eq_type == 'power':
                    val = rel['a'] * torch.pow(master_val + 1e-8, rel['b'])
                elif eq_type == 'quadratic':
                    val = rel['a'] * (master_val**2) + rel['b'] * master_val + rel['c']
                else: # linear
                    val = rel['a'] * master_val + rel['b']
                
            if name in FULL_SPACE:
                val = torch.clamp(val, FULL_SPACE[name]['low'], FULL_SPACE[name]['high'])
            self.cache[name] = val
        else:
            val = torch.tensor(self.defaults.get(name, 1.0), device=DEVICE).expand(self.cache['N'].shape[0])

        return val.view(-1, *([1] * (ndim - 1)))

# ==========================================
# 1. BRAIN DEFINITION
# ==========================================
class TriadPiece(nn.Module):
    def __init__(self, dim, batch_size, num_nodes, hp_manager):
        super().__init__() 
        self.D = dim
        self.hpm = hp_manager
        
        base_weight = torch.randn(1, num_nodes, dim, dim, device=DEVICE) / math.sqrt(self.D)
        self.weight = nn.Parameter(base_weight.expand(batch_size, -1, -1, -1).clone())
        self.register_buffer('dynamic_w_scale', torch.ones(1, device=DEVICE))
        
        # Initialize scaled
        w_scale = self.hpm.get('w_scale', 4)
        self.weight.copy_(utils.rms_norm(self.weight, dim=[-1,-2]) * w_scale * self.dynamic_w_scale) 

        self.register_buffer('g_ema', torch.randn_like(self.weight) * 0.01) 
        self.register_buffer('E_baseline', 0.01 + 0.01 * torch.randn(batch_size, num_nodes, dim, device=DEVICE))
        
    def forward(self, x):
        untanh = self.hpm.get('UNTANH', 3)
        raw_pred = (x.unsqueeze(2) @ self.weight).squeeze(2)
        return utils.buntanh(raw_pred, untanh, math.sqrt(2 + math.sqrt(2)))
    
    def get_gradient(self, layer_input, error_signal):
        ema_speed = self.hpm.get('EMA_SPEED', 3)
        rew_sens = self.hpm.get('FIXED_REWARD_SENSITIVITY', 4)
        n_scale = self.hpm.get('NOISE_SCALE', 3)
        
        E_curr = F.softmax(error_signal, dim=-1)
        self.E_baseline.mul_(1.0 - ema_speed).add_(ema_speed * E_curr)
        advantage = E_curr - self.E_baseline
        plasticity = 1.0 + utils.rms_norm(advantage.unsqueeze(3) * rew_sens)  
        
        noisy_state = utils.noised_input(layer_input, error_signal, n_scale)
        lg1 = utils.rms_norm(-torch.einsum('bni,bnj->bnij', error_signal, noisy_state))
        return (plasticity * lg1) 
    
    def step(self, grad, step_counter):
        wd = self.hpm.get('weight_decay', 4)
        grad = grad - (wd * self.weight)
        
        speed = self.hpm.get('speed', 4)
        width = self.hpm.get('width', 4)
        center = (step_counter * speed) % self.D
        g_mask = utils.wrapping_gaussian_kernel2(grad, center, width)
        grad = grad * g_mask
        
        momentum = self.hpm.get('MOMENTUM', 4)
        lr = self.hpm.get('LR', 4)

        self.g_ema.mul_(momentum).add_((1.0 - momentum) * grad)
        self.weight.add_(lr * self.g_ema)
        
        w_scale = self.hpm.get('w_scale', 4)
        dyn_scale = self.dynamic_w_scale.view(-1, 1, 1, 1)
        self.weight.copy_(utils.rms_norm(self.weight, dim=[-1,-2]) * w_scale * dyn_scale)

class PureTriadicBrain(nn.Module):
    def __init__(self, dim, batch_size, num_nodes, hp_manager):
        super().__init__()
        self.N = num_nodes
        self.D = dim
        self.batch_size = batch_size
        self.hpm = hp_manager
        
        self.w1 = TriadPiece(dim, batch_size, num_nodes, hp_manager)
        self.w2 = TriadPiece(dim, batch_size, num_nodes, hp_manager)
        self.w3 = TriadPiece(dim, batch_size, num_nodes, hp_manager)
        
        self.register_buffer('state', 1e-5*torch.randn(batch_size, self.N, dim, device=DEVICE))
        self.register_buffer('output', 1e-5*torch.randn(batch_size, self.N, dim, device=DEVICE))
        self.register_buffer('target', 1e-5*torch.randn(batch_size, self.N, dim, device=DEVICE))
        self.register_buffer('step_counter', torch.tensor(0.0, device=DEVICE))
        
        self.register_buffer('A', torch.randn(batch_size, self.N, self.N, device=DEVICE))
        self.register_buffer('A_ema', torch.zeros(batch_size, self.N, self.N, device=DEVICE))
        self.A_ema[:, 2, 0] += 3.0  
        self.A_ema[:, 2, 1] -= 2.0 
        self.A = F.softmax(self.A, dim=2)
        self.A_mask = torch.ones_like(self.A)
        
        self.A_mask[:, 0:2, :] = 0.0
        

    def step(self, eye_env, stomach_env):
        self.step_counter += 1.0
        
        queries    = self.w3(self.state)
        keys       = self.w2(self.state) 
        prediction = self.w1(self.state) 

        raw_A = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(self.D)
        raw_A = utils.rms_norm(raw_A, dim=[-1,-2])
        routing_momentum = self.hpm.get('c_moment', 3)
        
        if self.step_counter == 1.0:
            self.A_ema.copy_(raw_A)  
        else:
            self.A_ema.copy_(self.A_ema * routing_momentum + raw_A * (1.0 - routing_momentum))
        
        # 1. Sparse Sinkhorn Routing (Scales safely with N)
        raw_A = self.A_ema
        raw_A = raw_A * self.A_mask
        
        raw_A = utils.sinkhorn_knopp(raw_A)
        uniform_baseline = 0.5 / self.N  # Fixed mask logic!
        above_avg_mask = raw_A > uniform_baseline
        sparse_A = torch.where(above_avg_mask, raw_A, torch.tensor(0.0, device=DEVICE))
        
        self.A.copy_(utils.sinkhorn_knopp(sparse_A))
        self.A = self.A * self.A_mask
        
        # 2. "Boredom" Mechanism (Variance-Penalty Noise)
        a_var_goal = self.hpm.get('a_variance', 3)
        current_var = self.A.var(dim=[-1, -2], keepdim=True)
        vd = F.relu(a_var_goal - current_var)
        boredom_noise = vd * torch.randn_like(self.output)

        target = torch.bmm(self.A, self.output + boredom_noise)
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
# 2. BATCH EVALUATOR & TASK RUNNER
# ==========================================
def run_tracking_task(hp_tensors, relations, N, D, num_configs, num_seeds):
    batch_size = num_configs * num_seeds
    fitness_total = 0
    
    phase_x = torch.rand(batch_size, device=DEVICE) * 2 * math.pi
    phase_y = torch.rand(batch_size, device=DEVICE) * 2 * math.pi
    # Run the same batch through TWO distinct environments to prevent task-overfitting
    hpm = HPManager(hp_tensors, relations, N, D, batch_size)
    for task_type in ["smooth", "chaotic"]:
        brain = PureTriadicBrain(dim=D, batch_size=batch_size, num_nodes=N, hp_manager=hpm).to(DEVICE)
        brain = torch.compile(brain, dynamic=True, mode="reduce-overhead")
        agent_pos = torch.full((batch_size, 2), -3.0, device=DEVICE)
        agent_vel = torch.zeros(batch_size, 2, device=DEVICE)
        
        total_steps = 1500
        eval_start = 500 
        eval_steps = total_steps - eval_start
        dist_history = torch.zeros((eval_steps, batch_size), device=DEVICE)
        
        for step in range(total_steps):
            t = step / 100.0
            
            if task_type == "smooth":
                food_x = torch.sin(t + phase_x) * 2.0
                food_y = torch.sin(t * 2.0 + phase_y) * 1.5
            else:
                # Chaos with random phases
                food_x = torch.sin(t * 1.3 + phase_x) * 2.0 + torch.cos(t * 2.7 + phase_x) * 1.0
                food_y = torch.sin(t * 0.9 + phase_y) * 1.5 + torch.cos(t * 1.5 + phase_y) * 1.5
                
            food_pos = torch.stack([food_x, food_y], dim=1)

            diffs = food_pos - agent_pos
            dists = torch.norm(diffs, dim=1, keepdim=True)
            direction = diffs / (dists + 1e-5)
            
            eye_env = torch.zeros(batch_size, D, device=DEVICE)
            eye_env[:, 0:2] = direction
            stomach_env = torch.randn(batch_size, D, device=DEVICE) * dists * 1e-5
            
            brain.step(eye_env, stomach_env)
            
            motor_out = brain.output[:, 2, 0:2].clone()
            agent_vel = (agent_vel * 0.85) + (motor_out * 0.2)
            agent_pos += agent_vel
            agent_pos = torch.clamp(agent_pos, -4.0, 4.0)
            
            if step >= eval_start:
                dist_history[step - eval_start] = dists.squeeze()

        mean_dist_time = dist_history.mean(dim=0)
        var_dist_time = dist_history.var(dim=0)
        fitness_runs = torch.exp(-(mean_dist_time + var_dist_time)) 
        
        fitness_matrix = fitness_runs.view(num_configs, num_seeds)
        penalized_fitness = fitness_matrix.mean(dim=1) - fitness_matrix.var(dim=1)
        fitness_total += penalized_fitness

    # Return the average penalized fitness across all task environments
    return (fitness_total / 2.0).cpu().numpy()
# ==========================================
# 3. THE OPTUNA ARCHITECT & AUTO-REDUCER
# ==========================================
def load_file(file):
    if os.path.exists(file):
        with open(file, 'r') as f:
            return json.load(f)
    return {}

def save_relation(slave, master, a, b, c, eq_type):
    rels = load_file(RELATIONS_FILE)
    rels[slave] = {
        "master": master, 
        "a": float(a), 
        "b": float(b), 
        "c": float(c), 
        "type": str(eq_type)
    }
    with open(RELATIONS_FILE, 'w') as f:
        json.dump(rels, f, indent=4)
    print(f"Saved {eq_type.upper()} law for {slave} to {RELATIONS_FILE}")

def get_active_space(relations):
    active = {}
    for k, v in FULL_SPACE.items():
        if k not in relations:
            active[k] = v
    return active

def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    if ss_tot == 0: 
        return 0.0
    return 1.0 - (ss_res / ss_tot)
def analyze_and_collapse_dimension(study, active_space, profile = True):
    print(" INITIATING MULTI-VARIATE DIMENSIONAL COLLAPSE...")
    t_start = time.perf_counter()
    
    try:
        importances = optuna.importance.get_param_importances(
            study, 
            evaluator=MeanDecreaseImpurityImportanceEvaluator()
        )
    except Exception:
        return None

    df = study.trials_dataframe()
    df = df[df['state'] == 'COMPLETE']
    
    if len(df) < 20: 
        print(" Not enough data for robust statistical collapse. Waiting...")
        return None
    
    # Filter to top 90th percentile to focus on the "golden path"
    top_df = df[df['value'] >= df['value'].quantile(0.90)].copy()
    rename_map = {c: c.replace('params_', '') for c in top_df.columns}
    top_df = top_df.rename(columns=rename_map)
    
    cols =[c for c in list(active_space.keys()) + ['N', 'D'] if c in top_df.columns]

    best_collapse = None
    highest_score = 0.0  
    best_raw_r2 = 0.0    
    n_samples = len(top_df)

    def get_adj_r2(r2, n, k):
        if n <= k + 1: 
            return 0.0
        return 1 - ((1 - r2) * (n - 1) / (n - k - 1))

    # --- RANSAC Helper Function ---
    def fit_ransac(X, y_target):
        """Fits RANSAC and returns (coefficients, intercept). Returns None if fails."""
        try:
            # random_state ensures reproducibility, residual_threshold can be tuned if needed
            ransac = RANSACRegressor(LinearRegression(fit_intercept=True), random_state=42)
            ransac.fit(X, y_target)
            return ransac.estimator_.coef_, ransac.estimator_.intercept_
        except Exception:
            return None, None

    # Identify potential masters to heavily prune the loop
    valid_masters = [c for c in cols if c in ['N', 'D'] or importances.get(c, 0.0) > 0.05]
    
    t_setup_end = time.perf_counter()

    # ==================================================
    # 1. TEST 1D LAWS
    # ==================================================
    for col1, col2 in itertools.combinations(cols, 2):
        imp1 = importances.get(col1, 1.0 if col1 in ['N','D'] else 0.0)
        imp2 = importances.get(col2, 1.0 if col2 in ['N','D'] else 0.0)
        
        if imp1 >= imp2 and col1 in valid_masters and col2 not in ['N', 'D']:
            master, slave = col1, col2
        elif imp2 > imp1 and col2 in valid_masters and col1 not in ['N', 'D']:
            master, slave = col2, col1
        else:
            continue

        x, y = top_df[master].values, top_df[slave].values
        
        # 1. 1D Linear: y = a*x + b
        X_lin = x.reshape(-1, 1)
        coef, intercept = fit_ransac(X_lin, y)
        if coef is not None:
            r2_lin = calculate_r2(y, coef[0] * x + intercept)
            adj_r2_lin = get_adj_r2(r2_lin, n_samples, 1)
            if adj_r2_lin > highest_score: 
                highest_score, best_raw_r2 = adj_r2_lin, r2_lin
                best_collapse = (slave, master, coef[0], intercept, 0.0, 'linear')
        
        # 2. 1D Quadratic: y = a*x^2 + b*x + c
        X_quad = np.column_stack((x**2, x))
        coef, intercept = fit_ransac(X_quad, y)
        if coef is not None:
            r2_quad = calculate_r2(y, coef[0] * (x**2) + coef[1] * x + intercept)
            adj_r2_quad = get_adj_r2(r2_quad, n_samples, 2)
            if adj_r2_quad > highest_score: 
                highest_score, best_raw_r2 = adj_r2_quad, r2_quad
                best_collapse = (slave, master, coef[0], coef[1], intercept, 'quadratic')
        
        # 3. 1D Power Law: y = a * x^b => log(y) = log(a) + b*log(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if (x > 0).all() and (y > 0).all():
                X_pow = np.log(x).reshape(-1, 1)
                y_log = np.log(y)
                coef, intercept = fit_ransac(X_pow, y_log)
                if coef is not None:
                    a_pow, b_pow = np.exp(intercept), coef[0]
                    # Calculate true R2 on actual values, not log values
                    y_pred = a_pow * (x ** b_pow)
                    r2_pow = calculate_r2(y, y_pred)
                    adj_r2_pow = get_adj_r2(r2_pow, n_samples, 1)
                    if adj_r2_pow > highest_score: 
                        highest_score, best_raw_r2 = adj_r2_pow, r2_pow
                        best_collapse = (slave, master, a_pow, b_pow, 0.0, 'power')

    t_1d_end = time.perf_counter()

    # ==================================================
    # 2. TEST 2D LAWS
    # ==================================================
    for slave in cols:
        if slave in['N', 'D']: 
            continue 
        imp_s = importances.get(slave, 0.0)
        
        for m1, m2 in itertools.combinations(valid_masters, 2):
            if slave == m1 or slave == m2: 
                continue
            
            imp_m1 = importances.get(m1, 1.0 if m1 in ['N','D'] else 0.0)
            imp_m2 = importances.get(m2, 1.0 if m2 in ['N','D'] else 0.0)
            
            # Slave must be less important than both masters to be subsumed
            if imp_s >= imp_m1 or imp_s >= imp_m2: 
                continue
            
            x1, x2, y = top_df[m1].values, top_df[m2].values, top_df[slave].values
            
            # 1. 2D Linear: y = a*x1 + b*x2 + c
            X_mlin = np.column_stack((x1, x2))
            coef, intercept = fit_ransac(X_mlin, y)
            if coef is not None:
                r2_mlin = calculate_r2(y, coef[0]*x1 + coef[1]*x2 + intercept)
                adj_r2_mlin = get_adj_r2(r2_mlin, n_samples, 2)
                if adj_r2_mlin > highest_score: 
                    highest_score, best_raw_r2 = adj_r2_mlin, r2_mlin
                    best_collapse = (slave,[m1, m2], coef[0], coef[1], intercept, 'multi_linear')

            # 2. 2D Power Law: y = a * x1^b * x2^c => log(y) = log(a) + b*log(x1) + c*log(x2)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if (x1 > 0).all() and (x2 > 0).all() and (y > 0).all():
                    X_mpow = np.column_stack((np.log(x1), np.log(x2)))
                    y_log = np.log(y)
                    coef, intercept = fit_ransac(X_mpow, y_log)
                    if coef is not None:
                        a_mpow = np.exp(intercept)
                        b_mpow, c_mpow = coef[0], coef[1]
                        
                        y_pred = a_mpow * (x1**b_mpow) * (x2**c_mpow)
                        r2_mpow = calculate_r2(y, y_pred)
                        adj_r2_mpow = get_adj_r2(r2_mpow, n_samples, 2)
                        
                        if adj_r2_mpow > highest_score: 
                            highest_score, best_raw_r2 = adj_r2_mpow, r2_mpow
                            best_collapse = (slave, [m1, m2], a_mpow, b_mpow, c_mpow, 'multi_power')

    t_2d_end = time.perf_counter()

    # --- Profiling Output ---
    if profile:
        print(f"[Profiler] Tree-Parzen Importances & Setup: {t_setup_end - t_start:.4f}s")
        print(f"[Profiler] 1D Combinatorics & RANSAC:     {t_1d_end - t_setup_end:.4f}s")
        print(f"[Profiler] 2D Combinatorics & RANSAC:     {t_2d_end - t_1d_end:.4f}s")
        print(f"[Profiler] Total Collapse Analytics Time:  {t_2d_end - t_start:.4f}s")

    if best_collapse and highest_score > 0.25: 
        slave, master, a, b, c, eq_type = best_collapse
        print(f"\n💡 COUPLING DETECTED: {slave} is bound to {master}")
        print(f"   (Raw R²: {best_raw_r2:.3f} | Adjusted R²: {highest_score:.3f})")
        
        if eq_type == 'multi_power':
            print(f"   Discovered 2D POWER Law:   {slave} = {a:.4f} * ({master[0]}^{b:.4f}) * ({master[1]}^{c:.4f})")
        elif eq_type == 'multi_linear':
            print(f"   Discovered 2D LINEAR Law:  {slave} = ({a:.4f} * {master[0]}) + ({b:.4f} * {master[1]}) + {c:.4f}")
        elif eq_type == 'quadratic':
            print(f"   Discovered 1D QUADRATIC:   {slave} = ({a:.4f} * {master}²) + ({b:.4f} * {master}) + {c:.4f}")
        elif eq_type == 'power':
            print(f"   Discovered 1D POWER Law:   {slave} = {a:.4f} * ({master}^{b:.4f})")
        else:
            print(f"   Discovered 1D LINEAR Law:  {slave} = ({a:.4f} * {master}) + {b:.4f}")
            
        return {
            "slave": slave, "master": master,
            "a": a, "b": b, "c": c, "type": eq_type,
            "strikes": 0 
        }
    
    return None
def get_fresh_study(name, g_par, active_space,warmup):
    # TPE with a warm-up phase!
    # It will randomly explore for the first 256 trials (2 generations), 
    # then aggressively exploit the best regions it found.
    fresh_sampler = optuna.samplers.TPESampler(
        multivariate=True, 
        constant_liar=True, 
        n_startup_trials=warmup, 
        warn_independent_sampling=False
    )
    study = optuna.create_study(direction="maximize", study_name=name, sampler=fresh_sampler)
    
    if g_par is not None:
        active_golden_params = {k: v for k, v in g_par.items() if k in active_space}
        study.enqueue_trial(active_golden_params)
        
    return study

def run_optuna_search():
    global_best_fitness = -float('inf')
    CHECKPOINT_FILE = "best_blueprint.json"
    last_peak_generation = 0 

    num_configs = 128
    num_seeds = 8
    
    relations = load_file(RELATIONS_FILE)
    active_space = get_active_space(relations)
    
    global_best_params = None
    if os.path.exists(CHECKPOINT_FILE):
        ckpt = load_file(CHECKPOINT_FILE)
        global_best_params = ckpt.get("active_hyperparameters", None)
        global_best_fitness = ckpt.get("global_best_fitness", -float('inf'))

    
    choices =[8, 16, 24]
    nd_combinations = list(itertools.product(choices, choices))
    gen_count = len(nd_combinations) 
    #first gencount*configs as warmup.
    study = get_fresh_study(
        "TriadicCollapse", 
        global_best_params, 
        active_space,
        num_configs * gen_count
        )
    
    # Store EMA of yield instead of max yield, to account for standard RL variance
    scale_baselines = {nd: {'yield_ema': 0.0, 'best': -float('inf'), 'top_fits':[]} for nd in nd_combinations}
    
    pending_hypothesis = None
    current_hypothesis_gen = 0
    blacklist = {} 
    suspended_rule = None 
    generation = 0
    profiled = False
    while True:
        N, D = nd_combinations[generation % gen_count]
        generation += 1
        active_space = get_active_space(relations)
        
        if len(active_space) == 0:
            print("\n🚀 ALL HYPERPARAMETERS ELIMINATED. ARCHITECTURE IS FULLY SELF-DERIVED!")
            break
            
        print("\n" + "="*70)
        status = f"🧪 TESTING RULE ON {N}x{D}" if pending_hypothesis else "🌍 STANDARD EXPLORATION"
        print(f"GENERATION {generation} | Scale: N={N}, D={D} | Active HPs: {len(active_space)} | {status}")
        if pending_hypothesis:
            # Display current strike count if retrying
            strike_text = f" (Strikes: {pending_hypothesis.get('strikes', 0)}/2)" if pending_hypothesis.get('strikes', 0) > 0 else ""
            print(f"Hypothesis: {pending_hypothesis['slave']} -> {pending_hypothesis['type']} (Gauntlet: {generation - current_hypothesis_gen}/{gen_count}){strike_text}")
        print("="*70)
        
        for _ in range(num_configs):
            study.enqueue_trial({"N": int(N), "D": int(D)})
            
        trials =[study.ask() for _ in range(num_configs)]
        
        for t in trials:
            t.suggest_categorical("N", choices)
            t.suggest_categorical("D", choices)
            
        hp_tensors = {}
        for hp_name, space in active_space.items():
            vals = [t.suggest_float(hp_name, space['low'], space['high']) for t in trials]
            hp_tensors[hp_name] = torch.tensor(vals, device=DEVICE, dtype=torch.float32).repeat_interleave(num_seeds)
            
        penalized_fitness = run_tracking_task(hp_tensors, relations, N, D, num_configs, num_seeds)
        
        for i, trial in enumerate(trials):
            study.tell(trial, penalized_fitness[i])
            
        sorted_fitness = np.sort(penalized_fitness)[::-1]
        top_20_percent_idx = int(num_configs * 0.20)
        current_top_fits = sorted_fitness[:top_20_percent_idx]
        current_yield = np.mean(current_top_fits)
        
        scale_best = study.best_value
        ema_yield = scale_baselines[(N,D)]['yield_ema']
        print(f" Scale Best Fitness: {scale_best:.5f} (Historical: {scale_baselines[(N,D)]['best']:.5f})")
        print(f" Scale Yield:        {current_yield:.5f} (Historical EMA: {ema_yield:.5f})")
        
        # Track baselines only when not testing a newly applied hypothesis
        if not pending_hypothesis and not suspended_rule:
            if ema_yield == 0.0:
                scale_baselines[(N,D)]['yield_ema'] = current_yield
            else:
                # 80/20 Exponential Moving Average smooths out lucky/unlucky generations
                scale_baselines[(N,D)]['yield_ema'] = 0.8 * ema_yield + 0.2 * current_yield
                
            if scale_best > scale_baselines[(N,D)]['best']:
                scale_baselines[(N,D)]['best'] = scale_best
                
            scale_baselines[(N,D)]['top_fits'].extend(current_top_fits.tolist())
            scale_baselines[(N,D)]['top_fits'] = scale_baselines[(N,D)]['top_fits'][-500:]

        if study.best_value > global_best_fitness:
            global_best_fitness = study.best_value
            last_peak_generation = generation  
            global_best_params = study.best_trial.params.copy()
            best_params_clean = {k: v for k, v in study.best_trial.params.items() if k not in ['N', 'D']}
            print(f"🌟 NEW GLOBAL BEST! Fitness: {global_best_fitness:.5f}. Saving...")
            
            with open(CHECKPOINT_FILE, 'w') as f:
                json.dump({
                    "global_best_fitness": global_best_fitness,
                    "structural_scale": {"N": N, "D": D},
                    "active_hyperparameters": best_params_clean,
                    "active_laws": relations 
                }, f, indent=4)
        
        # --- HYPOTHESIS EVALUATION PHASE (The Gauntlet) ---
        if pending_hypothesis:
            gens_testing = generation - current_hypothesis_gen
            historical_fits = scale_baselines[(N,D)]['top_fits']
            
            if len(historical_fits) > 0:
                stat, p_val = mannwhitneyu(current_top_fits, historical_fits, alternative='less')
                
                # Failure Condition: Statistically significant (p < 0.01) AND severely depressed yield (< 60% of EMA)
                if p_val < 0.01 and current_yield < (0.6 * scale_baselines[(N,D)]['yield_ema']):
                    strikes = pending_hypothesis.get('strikes', 0) + 1
                    pending_hypothesis['strikes'] = strikes
                    
                    if strikes >= 2:
                        slave = pending_hypothesis['slave']
                        print(f"\n FATAL FAILURE ON SCALE N={N}, D={D}!")
                        print(f"Rule failed twice. Yield {current_yield:.5f} vs EMA {scale_baselines[(N,D)]['yield_ema']:.5f} (p={p_val:.4f}). YEETING RULE!")
                        
                        if slave in relations:
                            del relations[slave]
                        
                        blacklist[slave] = generation + gen_count * 3 
                        pending_hypothesis = None
                        continue 
                    else:
                        print(f"\n ANOMALY DETECTED ON N={N}, D={D}! (Strike {strikes}/2)")
                        print("Yield dropped. It might be RL variance. Retrying this exact scale combination...")
                        generation -= 1 # Decrementing generation forces the loop to re-test the exact same N,D!
                        continue

            if gens_testing >= gen_count:
                slave = pending_hypothesis['slave']
                if 'slave' in relations[slave]:
                    del relations[slave]['slave']
                    
                with open(RELATIONS_FILE, 'w') as f:
                    json.dump(relations, f, indent=4)
                
                pending_hypothesis = None
                blacklist.clear()
                last_peak_generation = generation
                print(f"\n🎓 HYPOTHESIS UNIVERSALLY ACCEPTED! '{slave}' survived all {gen_count} scale combinations.")
            
        # --- HYPOTHESIS GENERATION PHASE ---
        active_blacklist = [k for k, expire_gen in blacklist.items() if generation < expire_gen]
        
        if generation > (gen_count * 2) and generation % gen_count == 0 and not pending_hypothesis and not suspended_rule:
            search_space_for_math = {k: v for k, v in active_space.items() if k not in active_blacklist}
            proposed_rule = analyze_and_collapse_dimension(study, search_space_for_math, not profiled)
            #profiled=True
            if proposed_rule:
                pending_hypothesis = proposed_rule
                slave = pending_hypothesis['slave']
                
                relations[slave] = pending_hypothesis.copy()
                if 'slave' in relations[slave]:
                    del relations[slave]['slave']
                
                current_hypothesis_gen = generation
if __name__ == "__main__":
    torch.set_grad_enabled(False)
    # Ensure relations file starts clean on fresh runs (comment out to resume previous state)
    if os.path.exists(RELATIONS_FILE):
        os.remove(RELATIONS_FILE)
        
    run_optuna_search()