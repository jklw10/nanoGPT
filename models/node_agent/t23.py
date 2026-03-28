import os
import json
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import optuna
import parts.utils as utils
from scipy.stats import linregress, mannwhitneyu
import time
from sklearn.linear_model import RANSACRegressor, LinearRegression
import warnings
import itertools
from optuna.importance import MeanDecreaseImpurityImportanceEvaluator
import concurrent.futures

# ==========================================
# 0. SETUP & DYNAMIC HP MANAGER
# ==========================================
torch.set_float32_matmul_precision('high')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware: {DEVICE}")

RELATIONS_FILE = "hp_relations.json"

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
    Central nervous system for hyperparameters. 
    """
    def __init__(self, active_tensors, relations, N, D, batch_size, device=DEVICE):
        self.N_val = float(N)
        self.D_val = float(D)
        self.batch_size = batch_size
        self.device = device
        
        self.defaults = {
            'speed': 0.5,
            'width': 0.01,
            'lateral_decay': 0.01,
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

# ==========================================
# 1. BRAIN DEFINITION
# ==========================================
class TriadPiece(nn.Module):
    def __init__(self, dim, batch_size, num_nodes, hp_manager):
        super().__init__() 
        self.D = dim
        #self.hpm = hp_manager
        
        base_weight = torch.randn(1, num_nodes, dim, dim, device=DEVICE) / math.sqrt(self.D)
        self.register_buffer('weight', base_weight.expand(batch_size, -1, -1, -1).clone())
        
        #w_scale = self.hpm.get('w_scale', 4)
        #self.weight.data.copy_(utils.rms_norm(self.weight.data, dim=[-1,-2]) * w_scale * self.dynamic_w_scale) 

        self.register_buffer('g_ema', torch.randn_like(self.weight) * 0.01) 
        self.register_buffer('E_baseline', 0.01 + 0.01 * torch.randn(batch_size, num_nodes, dim, device=DEVICE))
        
    def reset(self, hps):
        # 1. Mount static tensors for graph execution (No Dicts!)
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

        """In-place reset avoids memory re-allocation and saves torch.compile traces."""
        base_weight = torch.randn(1, self.weight.shape[1], self.D, self.D, device=DEVICE) / math.sqrt(self.D)
        self.weight.data.copy_(base_weight.expand_as(self.weight))
       
        
        self.weight.data.copy_(utils.rms_norm(self.weight.data, dim=[-1,-2]) * self.hp_w_scale) 
        
        self.g_ema.normal_().mul_(0.01)
        self.E_baseline.normal_().mul_(0.01).add_(0.01)

    def forward(self, x):
        raw_pred = (x.unsqueeze(2) @ self.weight).squeeze(2)
        return utils.buntanh(raw_pred, self.hp_untanh, math.sqrt(2 + math.sqrt(2)))
    
    def get_gradient(self, layer_input, error_signal):
        ema_speed = self.hp_ema_speed
        rew_sens = self.hp_rew_sens
        n_scale = self.hp_noise_scale
        
        self.E_baseline.mul_(1.0 - ema_speed).add_(ema_speed * error_signal)
        E_curr = F.softmax(error_signal, dim=-1)
        advantage = E_curr - self.E_baseline
        plasticity = 1.0 + utils.rms_norm(advantage.unsqueeze(3) * rew_sens)  
        
        noisy_state = utils.noised_input(layer_input, error_signal, n_scale)
        lg1 = utils.rms_norm(-torch.einsum('bni,bnj->bnij', error_signal, noisy_state))
        return (plasticity * lg1) 
    
    def step(self, grad, step_counter):
        wd = self.hp_wd
        grad = grad - (wd * self.weight)
        
        speed = self.hp_speed 
        width = self.hp_width 
        center = (step_counter * speed) % self.D
        g_mask = utils.wrapping_gaussian_kernel2(grad, center, width)
        grad = grad * g_mask
        
        momentum = self.hp_momentum 
        lr = self.hp_lr 

        self.g_ema.mul_(momentum).add_((1.0 - momentum) * grad)
        self.weight.add_(lr * self.g_ema)
        
        w_scale = self.hp_w_scale
        self.weight.data.copy_(utils.rms_norm(self.weight.data, dim=[-1,-2]) * w_scale )
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
        raw_A = utils.rms_norm(raw_A, dim=[-1,-2])
        
        is_first = (self.step_counter == 1.0).float().view(1, 1, 1)
        new_A_ema = raw_A * is_first + (self.A_ema * self.hp_c_moment + raw_A * (1.0 - self.hp_c_moment)) * (1.0 - is_first)
        self.A_ema.copy_(new_A_ema * self.A_mask)
        
        raw_A = self.A_ema 
        raw_A = utils.sinkhorn_knopp(raw_A)
        uniform_baseline = 0.5 / self.N
        
        sparse_A = torch.where(raw_A > uniform_baseline, raw_A, 0.0)
        
        self.A.copy_(utils.sinkhorn_knopp(sparse_A) * self.A_mask)
        
        current_var = self.A.var(dim=[-1, -2], keepdim=True)
        vd = F.relu(self.hp_a_variance - current_var)
        boredom_noise = vd * torch.randn_like(self.output)

        target = torch.bmm(self.A, self.output + boredom_noise)
        #TODO test thoroughly:
        #target = utils.buntanh(target, self.hp_untanh, math.sqrt(2 + math.sqrt(2)))
        
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
# 2. VECTORIZED BATCH EVALUATOR 
# ==========================================
def run_tracking_task(brain, hp_tensors, relations, N, D, num_configs, num_seeds, stream):
    """Executes on isolated CUDA streams. Fully vectorized trajectory pre-calculation."""
    with torch.cuda.stream(stream):
        batch_size = num_configs * num_seeds
        fitness_total = 0
        
        phase_x = (torch.rand(batch_size, device=DEVICE) * 2 * math.pi).unsqueeze(0)
        phase_y = (torch.rand(batch_size, device=DEVICE) * 2 * math.pi).unsqueeze(0)
        
        total_steps = 1500
        eval_start = 500 
        eval_steps = total_steps - eval_start

        # PRE-COMPUTE ENTIRE TRAJECTORIES ONCE
        T = torch.arange(total_steps, device=DEVICE).float().view(total_steps, 1) / 100.0
        
        food_x_smooth = torch.sin(T + phase_x) * 2.0
        food_y_smooth = torch.sin(T * 2.0 + phase_y) * 1.5
        food_pos_smooth = torch.stack([food_x_smooth, food_y_smooth], dim=-1)
        
        food_x_chaos = torch.sin(T * 1.3 + phase_x) * 2.0 + torch.cos(T * 2.7 + phase_x) * 1.0
        food_y_chaos = torch.sin(T * 0.9 + phase_y) * 1.5 + torch.cos(T * 1.5 + phase_y) * 1.5
        food_pos_chaos = torch.stack([food_x_chaos, food_y_chaos], dim=-1)

        for task_type in ["smooth", "chaotic"]:
            brain.reset(hp_tensors, relations)
            
            agent_pos = torch.full((batch_size, 2), -3.0, device=DEVICE)
            agent_vel = torch.zeros(batch_size, 2, device=DEVICE)
            dist_history = torch.empty((eval_steps, batch_size), device=DEVICE)
            
            # Pre-allocate inputs for perfect CUDA-Graph tracing
            eye_env = torch.zeros(batch_size, D, device=DEVICE)
            stomach_env = torch.empty(batch_size, D, device=DEVICE)
            
            all_food_pos = food_pos_smooth if task_type == "smooth" else food_pos_chaos

            for step in range(total_steps):
                diffs = all_food_pos[step] - agent_pos
                dists = torch.norm(diffs, dim=1, keepdim=True)
                
                # In-place environment update
                eye_env[:, 0:2] = diffs / (dists + 1e-5)
                stomach_env.normal_().mul_(dists * 1e-5) 
                
                brain(eye_env, stomach_env)
                
                # In-place physics update
                motor_out = brain.output[:, 2, 0:2]
                agent_vel.mul_(0.85).add_(motor_out, alpha=0.2)
                agent_pos.add_(agent_vel).clamp_(-4.0, 4.0)
                
                if step >= eval_start:
                    dist_history[step - eval_start] = dists.squeeze(1)

            mean_dist_time = dist_history.mean(dim=0)
            var_dist_time = dist_history.var(dim=0)
            fitness_matrix = torch.exp(-(mean_dist_time + var_dist_time)).view(num_configs, num_seeds)
            
            penalized_fitness = fitness_matrix.mean(dim=1) - fitness_matrix.var(dim=1)
            fitness_total += penalized_fitness

    stream.synchronize()
    return (fitness_total / 2.0).cpu().numpy()

# ==========================================
# 3. THE OPTUNA ARCHITECT & AUTO-REDUCER
# ==========================================
def load_file(file):
    if os.path.exists(file):
        with open(file, 'r') as f:
            return json.load(f)
    return {}

def suspend_rule_to_file(slave, law, reason=""):
    suspend_path = "RL/suspended_rules.json"
    data = {}
    if os.path.exists(suspend_path):
        try:
            with open(suspend_path, 'r') as f:
                data = json.load(f)
        except Exception: 
            pass
    key = f"{slave}_{int(time.time())}"
    data[key] = {
        "parameter": slave,
        "law": law,
        "reason": reason,
        "generation_pruned": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(suspend_path, 'w') as f:
        json.dump(data, f, indent=4)
        
def save_relations(rels, path):
    with open(path, 'w') as f:
        json.dump(rels, f, indent=4)

def get_active_space(relations):
    active = {}
    for k, v in FULL_SPACE.items():
        if k not in relations:
            active[k] = v
    return active

def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    if ss_tot == 0: return 0.0
    return 1.0 - (ss_res / ss_tot)
def analyze_and_collapse_dimension(study, active_space, profile=True):
    print(" INITIATING MULTI-VARIATE DIMENSIONAL COLLAPSE...")
    t_start = time.perf_counter()
    
    try:
        importances = optuna.importance.get_param_importances(study, evaluator=MeanDecreaseImpurityImportanceEvaluator())
    except Exception:
        return None

    df = study.trials_dataframe()
    df = df[df['state'] == 'COMPLETE']
    
    if len(df) < 20: 
        return None
    
    # Take the top 10% of trials
    top_df = df[df['value'] >= df['value'].quantile(0.90)].copy()
    rename_map = {c: c.replace('params_', '') for c in top_df.columns}
    top_df = top_df.rename(columns=rename_map)
    cols = [c for c in list(active_space.keys()) +['N', 'D'] if c in top_df.columns]

    best_collapse = None
    highest_score = 0.0  
    best_raw_r2 = 0.0    
    n_samples = len(top_df)

    def get_adj_r2(r2, n, k):
        if n <= k + 1: return 0.0
        return 1 - ((1 - r2) * (n - 1) / (n - k - 1))

    def fit_ransac(X, y_target):
        try:
            # We enforce min_samples to ensure it doesn't overfit to a tiny subset
            min_s = max(int(len(y_target) * 0.25), 3) 
            ransac = RANSACRegressor(LinearRegression(fit_intercept=True), min_samples=min_s, random_state=42)
            ransac.fit(X, y_target)
            return ransac.estimator_.coef_, ransac.estimator_.intercept_, ransac.inlier_mask_
        except Exception:
            return None, None, None

    valid_masters = [c for c in cols if c in ['N', 'D'] or importances.get(c, 0.0) > 0.05]
    
    # 1. TEST 1D LAWS
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
        
        # 1A. LINEAR
        X_lin = x.reshape(-1, 1)
        coef, intercept, inlier_mask = fit_ransac(X_lin, y)
        if coef is not None and inlier_mask is not None:
            n_inliers = np.sum(inlier_mask)
            if n_inliers > 2:
                r2_lin = calculate_r2(y[inlier_mask], coef[0] * x[inlier_mask] + intercept)
                adj_r2_lin = get_adj_r2(r2_lin, n_inliers, 1)
                score = adj_r2_lin * (n_inliers / n_samples) # Penalize for dropping data
                if score > highest_score: 
                    highest_score, best_raw_r2 = score, r2_lin
                    best_collapse = (slave, master, coef[0], intercept, 0.0, 'linear')
        
        # 1B. QUADRATIC
        X_quad = np.column_stack((x**2, x))
        coef, intercept, inlier_mask = fit_ransac(X_quad, y)
        if coef is not None and inlier_mask is not None:
            n_inliers = np.sum(inlier_mask)
            if n_inliers > 3:
                x_in, y_in = x[inlier_mask], y[inlier_mask]
                r2_quad = calculate_r2(y_in, coef[0] * (x_in**2) + coef[1] * x_in + intercept)
                adj_r2_quad = get_adj_r2(r2_quad, n_inliers, 2)
                score = adj_r2_quad * (n_inliers / n_samples)
                if score > highest_score: 
                    highest_score, best_raw_r2 = score, r2_quad
                    best_collapse = (slave, master, coef[0], coef[1], intercept, 'quadratic')
        
        # 1C. POWER LAW
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if (x > 0).all() and (y > 0).all():
                X_pow = np.log(x).reshape(-1, 1)
                y_log = np.log(y)
                coef, intercept, inlier_mask = fit_ransac(X_pow, y_log)
                if coef is not None and inlier_mask is not None:
                    n_inliers = np.sum(inlier_mask)
                    if n_inliers > 2:
                        a_pow, b_pow = np.exp(intercept), coef[0]
                        r2_pow = calculate_r2(y[inlier_mask], a_pow * (x[inlier_mask] ** b_pow))
                        adj_r2_pow = get_adj_r2(r2_pow, n_inliers, 1)
                        score = adj_r2_pow * (n_inliers / n_samples)
                        if score > highest_score: 
                            highest_score, best_raw_r2 = score, r2_pow
                            best_collapse = (slave, master, a_pow, b_pow, 0.0, 'power')

    # 2. TEST 2D LAWS
    for slave in cols:
        if slave in ['N', 'D']: continue 
        imp_s = importances.get(slave, 0.0)
        
        for m1, m2 in itertools.combinations(valid_masters, 2):
            if slave == m1 or slave == m2: continue
            
            imp_m1 = importances.get(m1, 1.0 if m1 in['N','D'] else 0.0)
            imp_m2 = importances.get(m2, 1.0 if m2 in ['N','D'] else 0.0)
            if imp_s >= imp_m1 or imp_s >= imp_m2: continue
            
            x1, x2, y = top_df[m1].values, top_df[m2].values, top_df[slave].values
            
            # 2A. MULTI-LINEAR
            X_mlin = np.column_stack((x1, x2))
            coef, intercept, inlier_mask = fit_ransac(X_mlin, y)
            if coef is not None and inlier_mask is not None:
                n_inliers = np.sum(inlier_mask)
                if n_inliers > 3:
                    x1_in, x2_in, y_in = x1[inlier_mask], x2[inlier_mask], y[inlier_mask]
                    r2_mlin = calculate_r2(y_in, coef[0]*x1_in + coef[1]*x2_in + intercept)
                    adj_r2_mlin = get_adj_r2(r2_mlin, n_inliers, 2)
                    score = adj_r2_mlin * (n_inliers / n_samples)
                    if score > highest_score: 
                        highest_score, best_raw_r2 = score, r2_mlin
                        best_collapse = (slave, [m1, m2], coef[0], coef[1], intercept, 'multi_linear')

            # 2B. MULTI-POWER LAW
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if (x1 > 0).all() and (x2 > 0).all() and (y > 0).all():
                    X_mpow = np.column_stack((np.log(x1), np.log(x2)))
                    coef, intercept, inlier_mask = fit_ransac(X_mpow, np.log(y))
                    if coef is not None and inlier_mask is not None:
                        n_inliers = np.sum(inlier_mask)
                        if n_inliers > 3:
                            a_mpow, b_mpow, c_mpow = np.exp(intercept), coef[0], coef[1]
                            x1_in, x2_in, y_in = x1[inlier_mask], x2[inlier_mask], y[inlier_mask]
                            r2_mpow = calculate_r2(y_in, a_mpow * (x1_in**b_mpow) * (x2_in**c_mpow))
                            adj_r2_mpow = get_adj_r2(r2_mpow, n_inliers, 2)
                            score = adj_r2_mpow * (n_inliers / n_samples)
                            if score > highest_score: 
                                highest_score, best_raw_r2 = score, r2_mpow
                                best_collapse = (slave, [m1, m2], a_mpow, b_mpow, c_mpow, 'multi_power')

    if best_collapse and highest_score > 0.25: 
        slave, master, a, b, c, eq_type = best_collapse
        print(f"\n💡 COUPLING DETECTED: {slave} is bound to {master}")
        print(f"   (Inlier R²: {best_raw_r2:.3f} | Confidence Score: {highest_score:.3f})")
        return {"slave": slave, "master": master, "a": a, "b": b, "c": c, "type": eq_type, "strikes": 0 }
    
    return None

def get_fresh_study(name, g_par, active_space, warmup):
    fresh_sampler = optuna.samplers.TPESampler(multivariate=True, constant_liar=True, n_startup_trials=warmup, warn_independent_sampling=False)
    study = optuna.create_study(direction="maximize", study_name=name, sampler=fresh_sampler)
    if g_par is not None:
        active_golden_params = {k: v for k, v in g_par.items() if k in active_space}
        study.enqueue_trial(active_golden_params)
    return study

def run_optuna_search():
    os.makedirs("RL", exist_ok=True)
    RELATIONS_FILE = "RL/hp_relations.json"
    CHECKPOINT_FILE = "RL/best_blueprint.json"
    
    for old_file in ["hp_relations.json", "best_blueprint.json"]:
        if os.path.exists(old_file) and not os.path.exists(f"RL/{old_file}"):
            os.rename(old_file, f"RL/{old_file}")

    global_best_fitness = -float('inf')
    num_configs = 128
    num_seeds = 8
    batch_size = num_configs * num_seeds
    
    relations = load_file(RELATIONS_FILE)
    active_space = get_active_space(relations)
    
    global_best_params = None
    if os.path.exists(CHECKPOINT_FILE):
        ckpt = load_file(CHECKPOINT_FILE)
        global_best_params = ckpt.get("active_hyperparameters", None)
        global_best_fitness = ckpt.get("global_best_fitness", -float('inf'))

    choices = [8, 16, 24]
    nd_combinations = list(itertools.product(choices, choices))
    gen_count = len(nd_combinations) 
    
    study = get_fresh_study("TriadicCollapse", global_best_params, active_space, gen_count*num_configs)
    scale_baselines = {nd: {'yield_ema': 0.0, 'best': -float('inf'), 'top_fits':[]} for nd in nd_combinations}
    
    # --- PRE-ALLOCATE AND COMPILE ALL BRAINS ONCE ---
    print("\n[INIT] Compiling static computation graphs for all combinations...")
    BRAIN_CACHE = {}
    STREAMS = {}
    for N, D in nd_combinations:
        hpm_dummy = HPManager({}, relations, N, D, batch_size)
        brain = PureTriadicBrain(dim=D, batch_size=batch_size, num_nodes=N, hp_manager=hpm_dummy).to(DEVICE)
        brain = torch.compile(brain, mode="reduce-overhead")
        
        # Warm-up compile (Forces PyTorch to build the graph now instead of mid-run)
        eye_dummy = torch.zeros(batch_size, D, device=DEVICE)
        stom_dummy = torch.zeros(batch_size, D, device=DEVICE)
        
        hp_tensors = {}
        for hp_name, space in active_space.items():
            vals = [ 0 for _ in range(num_configs)]
            hp_tensors[hp_name] = torch.tensor(vals, device=DEVICE, dtype=torch.float32).repeat_interleave(num_seeds)
            
        brain.reset(hp_tensors, relations)
        brain(eye_dummy, stom_dummy) 
        
        BRAIN_CACHE[(N, D)] = brain
        STREAMS[(N, D)] = torch.cuda.Stream()
    print("[INIT] Compilations Complete! Launching Parallel Multi-Scale Engine...\n")

    engine = {
        "phase": "WARMUP", 
        "tested_rule": None,
        "suspended_rule": None,
        "suspended_law": None,
        "start_gen": 0,
        "start_best": -float('inf'),
        "yields":[],
        "emas":[],
        "strikes": 0
    }
    
    blacklist = {} 
    generation = 0
    
    while True:
        active_space = get_active_space(relations)
        if len(active_space) == 0:
            print("\n🚀 ALL HYPERPARAMETERS ELIMINATED. ARCHITECTURE IS FULLY SELF-DERIVED!")
            break
            
        print("\n" + "="*80)
        if engine["phase"] == "TESTING":
            status = f"🧪 PHASE 1: GAUNTLET '{engine['tested_rule']}' (Strikes: {engine['strikes']}/2)"
        elif engine["phase"] == "SUSPENDING":
            status = f"🔪 PHASE 2: ABLATING '{engine['suspended_rule']}'"
        else:
            status = f"🌍 MAPPING LANDSCAPE ({engine['phase']})"
            
        print(f"MACRO-GENERATION {generation//gen_count + 1} | Active HPs: {len(active_space)} | {status}")
        print("="*80)
        
        # 1. Ask Optuna for all parameter combinations
        macro_tasks = {}
        for N, D in nd_combinations:
            trials =[study.ask() for _ in range(num_configs)]
            for t in trials:
                t.suggest_categorical("N", choices)
                t.suggest_categorical("D", choices)
                
            hp_tensors = {}
            for hp_name, space in active_space.items():
                vals =[t.suggest_float(hp_name, space['low'], space['high']) for t in trials]
                hp_tensors[hp_name] = torch.tensor(vals, device=DEVICE, dtype=torch.float32).repeat_interleave(num_seeds)
                
            macro_tasks[(N, D)] = (trials, hp_tensors)
            
        # 2. Parallel Hardware Execution 
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=gen_count) as executor:
            futures = {}
            for (N, D), (_, hp_tensors) in macro_tasks.items():
                f = executor.submit(run_tracking_task, BRAIN_CACHE[(N, D)], hp_tensors, relations, N, D, num_configs, num_seeds, STREAMS[(N, D)])
                futures[f] = (N, D)
                
            for future in concurrent.futures.as_completed(futures):
                N, D = futures[future]
                results[(N, D)] = future.result()
                
        # 3. Synchronous State Processing
        for N, D in nd_combinations:
            generation += 1
            trials, _ = macro_tasks[(N, D)]
            penalized_fitness = results[(N, D)]
            
            for i, trial in enumerate(trials):
                study.tell(trial, penalized_fitness[i])
                
            sorted_fitness = np.sort(penalized_fitness)[::-1]
            current_top_fits = sorted_fitness[:int(num_configs * 0.20)]
            current_yield = np.mean(current_top_fits)
            
            scale_best = study.best_value
            ema_yield = scale_baselines[(N,D)]['yield_ema']
            print(f"[{N}x{D}] Best: {scale_best:.4f} (Hist: {scale_baselines[(N,D)]['best']:.4f}) | Yield: {current_yield:.4f} (EMA: {ema_yield:.4f})")
            
            if engine["phase"] not in ["TESTING", "SUSPENDING"]:
                if ema_yield == 0.0:
                    scale_baselines[(N,D)]['yield_ema'] = current_yield
                else:
                    scale_baselines[(N,D)]['yield_ema'] = 0.8 * ema_yield + 0.2 * current_yield
                if scale_best > scale_baselines[(N,D)]['best']:
                    scale_baselines[(N,D)]['best'] = scale_best
                scale_baselines[(N,D)]['top_fits'].extend(current_top_fits.tolist())
                scale_baselines[(N,D)]['top_fits'] = scale_baselines[(N,D)]['top_fits'][-500:]

            if engine["phase"] == "SUSPENDING":
                engine["yields"].append(current_yield)
                engine["emas"].append(scale_baselines[(N,D)]['yield_ema'])

            if study.best_value > global_best_fitness:
                global_best_fitness = study.best_value
                best_params_clean = {k: v for k, v in study.best_trial.params.items() if k not in ['N', 'D']}
                print(f"🌟 NEW GLOBAL BEST! Fitness: {global_best_fitness:.5f}. Saving to RL/...")
                
                ckpt_data = {
                    "global_best_fitness": global_best_fitness,
                    "generation": generation,
                    "structural_scale": {"N": N, "D": D},
                    "active_hyperparameters": best_params_clean,
                    "active_laws": relations 
                }
                with open(CHECKPOINT_FILE, 'w') as f:
                    json.dump(ckpt_data, f, indent=4)
                with open(f"RL/peak_gen{generation}_fit{global_best_fitness:.5f}.json", 'w') as f:
                    json.dump(ckpt_data, f, indent=4)
            
            if engine["phase"] == "TESTING":
                historical_fits = scale_baselines[(N,D)]['top_fits']
                if len(historical_fits) > 0:
                    stat, p_val = mannwhitneyu(current_top_fits, historical_fits, alternative='less')
                    if p_val < 0.01 and current_yield < (0.6 * scale_baselines[(N,D)]['yield_ema']):
                        engine['strikes'] += 1
                        if engine['strikes'] >= 2:
                            print(f"\n📉 FATAL FAILURE! '{engine['tested_rule']}' failed the Gauntlet.")
                            suspend_rule_to_file(engine['tested_rule'], relations[engine['tested_rule']], "Failed Gauntlet")
                            del relations[engine['tested_rule']]
                            save_relations(relations, RELATIONS_FILE)
                            blacklist[engine['tested_rule']] = generation + gen_count * 3 
                            engine["phase"] = "IDLE"
                            engine["tested_rule"] = None
                        else:
                            print(f"\n⚠️ ANOMALY DETECTED ON N={N}, D={D}! (Strike {engine['strikes']}/2)")

            # The Engine Logic Loop
            if generation >= (gen_count * 2) and generation % gen_count == 0:
                if engine["phase"] == "TESTING":
                    if global_best_fitness > engine["start_best"]:
                        print(f"\n🎓 HYPOTHESIS ACCEPTED! Phase 1 Rule '{engine['tested_rule']}' pushed the peak.")
                        engine["tested_rule"] = None
                        engine["phase"] = "IDLE"
                    else:
                        avail =[k for k in relations.keys() if k != engine["tested_rule"]]
                        if avail:
                            engine["suspended_rule"] = random.choice(avail)
                            engine["suspended_law"] = relations[engine["suspended_rule"]].copy()
                            del relations[engine["suspended_rule"]]
                            
                            engine["phase"] = "SUSPENDING"
                            engine["start_gen"] = generation
                            engine["start_best"] = global_best_fitness
                            engine["yields"], engine["emas"] = [],[]
                            print(f"\n🤔 Rule survived but didn't improve peak. Initiating Phase 2 Ablation on '{engine['suspended_rule']}'")
                            save_relations(relations, RELATIONS_FILE)
                        else:
                            print(f"\n📉 Rule didn't improve peak, no others to ablate. Pruning.")
                            suspend_rule_to_file(engine["tested_rule"], relations[engine["tested_rule"]], "Failed to push peak")
                            del relations[engine["tested_rule"]]
                            engine["tested_rule"] = None
                            engine["phase"] = "IDLE"
                            save_relations(relations, RELATIONS_FILE)

                elif engine["phase"] == "SUSPENDING":
                    avg_yield = np.mean(engine["yields"]) if engine["yields"] else 0
                    avg_ema = np.mean(engine["emas"]) if engine["emas"] else 0
                    
                    if global_best_fitness > engine["start_best"] or avg_yield > (avg_ema * 1.05):
                        print(f"\n💥 ABLATION SUCCESS! Removing '{engine['suspended_rule']}' unleashed performance!")
                        suspend_rule_to_file(engine["suspended_rule"], engine["suspended_law"], "Ablation increased yield")
                    else:
                        print(f"\n🛡️ RULE VINDICATED! '{engine['suspended_rule']}' is structural. Reinstating.")
                        relations[engine["suspended_rule"]] = engine["suspended_law"]
                        if engine["tested_rule"] and engine["tested_rule"] in relations:
                            suspend_rule_to_file(engine["tested_rule"], relations[engine["tested_rule"]], "Failed Phase 2")
                            del relations[engine["tested_rule"]]
                            blacklist[engine["tested_rule"]] = generation + (gen_count * 4)
                            
                    engine["suspended_rule"], engine["suspended_law"], engine["tested_rule"] = None, None, None
                    engine["phase"] = "IDLE"
                    save_relations(relations, RELATIONS_FILE)

                if engine["phase"] in ["IDLE", "WARMUP"]:
                    active_blacklist =[k for k, exp in blacklist.items() if generation < exp]
                    search_space = {k: v for k, v in active_space.items() if k not in active_blacklist}
                    proposed = analyze_and_collapse_dimension(study, search_space, False)
                    
                    if proposed:
                        engine["tested_rule"] = proposed['slave']
                        relations[engine["tested_rule"]] = proposed.copy()
                        del relations[engine["tested_rule"]]['slave']
                        
                        engine["phase"], engine["start_gen"], engine["start_best"], engine["strikes"] = "TESTING", generation, global_best_fitness, 0
                        print(f"\n💡 Initiating Phase 1 Testing for '{engine['tested_rule']}'")
                        save_relations(relations, RELATIONS_FILE)
                    elif len(relations) > 0:
                        engine["suspended_rule"] = random.choice(list(relations.keys()))
                        engine["suspended_law"] = relations[engine["suspended_rule"]].copy()
                        del relations[engine["suspended_rule"]]
                        
                        engine["phase"], engine["start_gen"], engine["start_best"], engine["yields"], engine["emas"] = "SUSPENDING", generation, global_best_fitness, [], []
                        print(f"\n🧪 No new rules detected. Ablating '{engine['suspended_rule']}' to test landscape.")
                        save_relations(relations, RELATIONS_FILE)

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    # if os.path.exists(RELATIONS_FILE): os.remove(RELATIONS_FILE)
    run_optuna_search()