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
from scipy.stats import linregress
import warnings
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
            master_val = self.get(rel['master'], 1).flatten() 
            
            # Apply the winning physics equation
            eq_type = rel.get('type', 'linear')
            if eq_type == 'power':
                val = rel['a'] * torch.pow(master_val + 1e-8, rel['b'])
            elif eq_type == 'quadratic':
                val = rel['a'] * (master_val**2) + rel['b'] * master_val + rel['c']
            else: # linear
                val = rel['a'] * master_val + rel['b']
                
            # Clamping is crucial here to prevent parabolas from exploding out of bounds!
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
        raw_pred = torch.einsum('bnj,bnjk->bnk', x, self.weight)
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

    def step(self, eye_env, stomach_env):
        self.step_counter += 1.0
        
        queries    = self.w3(self.state)
        keys       = self.w2(self.state) 
        prediction = self.w1(self.state) 

        raw_A = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(self.D)
        raw_A = utils.rms_norm(raw_A, dim=[-1,-2])
        routing_momentum = self.hpm.get('c_moment', 3)
        
        if self.step_counter.item() == 1.0:
            self.A_ema.copy_(raw_A)  
        else:
            self.A_ema.copy_(self.A_ema * routing_momentum + raw_A * (1.0 - routing_momentum))
        
        # 1. Sparse Sinkhorn Routing (Scales safely with N)
        raw_A = self.A_ema
        self.A[:, 0:2, :] = 0.0
        
        raw_A = utils.sinkhorn_knopp(raw_A)
        uniform_baseline = 0.5 / self.N  # Fixed mask logic!
        above_avg_mask = raw_A > uniform_baseline
        sparse_A = torch.where(above_avg_mask, raw_A, torch.tensor(0.0, device=DEVICE))
        
        self.A.copy_(utils.sinkhorn_knopp(sparse_A))
        self.A[:, 0:2, :] = 0.0
        
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
    hpm = HPManager(hp_tensors, relations, N, D, batch_size)
    brain = PureTriadicBrain(dim=D, batch_size=batch_size, num_nodes=N, hp_manager=hpm).to(DEVICE)
    
    agent_pos = torch.full((batch_size, 2), -3.0, device=DEVICE)
    agent_vel = torch.zeros(batch_size, 2, device=DEVICE)
    
    total_steps = 1500
    eval_start = 500 
    eval_steps = total_steps - eval_start
    dist_history = torch.zeros((eval_steps, batch_size), device=DEVICE)
    
    for step in range(total_steps):
        t = step / 100.0
        food_pos = torch.tensor([[math.sin(t) * 2.0, math.sin(t * 2.0) * 1.5]], device=DEVICE).expand(batch_size, 2)
        
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

    return penalized_fitness.cpu().numpy()

# ==========================================
# 3. THE OPTUNA ARCHITECT & AUTO-REDUCER
# ==========================================
def load_relations():
    if os.path.exists(RELATIONS_FILE):
        with open(RELATIONS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_relation(slave, master, a, b, c, eq_type):
    rels = load_relations()
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

def analyze_and_collapse_dimension(study, active_space):
    print("\n" + "🌀"*25)
    print("🔍 INITIATING TRI-MATH DIMENSIONAL COLLAPSE (R² SCORING)...")
    
    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception:
        return False

    df = study.trials_dataframe()
    df = df[df['state'] == 'COMPLETE']
    if len(df) < 20: return False
    
    # Top 10% of trials
    top_df = df[df['value'] >= df['value'].quantile(0.90)].copy()
    rename_map = {c: c.replace('params_', '').replace('user_attrs_', '') for c in top_df.columns}
    top_df = top_df.rename(columns=rename_map)
    
    cols =[c for c in list(active_space.keys()) + ['N', 'D'] if c in top_df.columns]
    
    best_collapse = None
    highest_r2 = 0.0

    # Test every pair
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            col1, col2 = cols[i], cols[j]
            x, y = top_df[col1].values, top_df[col2].values
            
            # Figure out who is master vs slave
            imp1 = importances.get(col1, 1.0 if col1 in ['N','D'] else 0.0)
            imp2 = importances.get(col2, 1.0 if col2 in['N','D'] else 0.0)
            
            if imp1 >= imp2 and col2 not in['N', 'D']:
                master_name, slave_name, master_vals, slave_vals = col1, col2, x, y
            elif imp2 > imp1 and col1 not in['N', 'D']:
                master_name, slave_name, master_vals, slave_vals = col2, col1, y, x
            else:
                continue

            # --- MATH CONTEST ---
            models =[]
            
            # 1. Linear (y = ax + b)
            c_lin = np.polyfit(master_vals, slave_vals, 1)
            y_lin = c_lin[0] * master_vals + c_lin[1]
            models.append(('linear', calculate_r2(slave_vals, y_lin), c_lin[0], c_lin[1], 0.0))
            
            # 2. Quadratic/Parabola (y = ax^2 + bx + c)
            c_quad = np.polyfit(master_vals, slave_vals, 2)
            y_quad = c_quad[0] * (master_vals**2) + c_quad[1] * master_vals + c_quad[2]
            models.append(('quadratic', calculate_r2(slave_vals, y_quad), c_quad[0], c_quad[1], c_quad[2]))
            
            # 3. Power Law (y = a * x^b) --> via log-log
            with warnings.catch_warnings():
                warnings.simplefilter("ignore") # Ignore log(0) warnings
                valid = (master_vals > 0) & (slave_vals > 0)
                if valid.all():
                    log_x, log_y = np.log(master_vals), np.log(slave_vals)
                    c_pow = np.polyfit(log_x, log_y, 1)
                    a_pow = np.exp(c_pow[1])
                    b_pow = c_pow[0]
                    y_pow = a_pow * (master_vals ** b_pow)
                    models.append(('power', calculate_r2(slave_vals, y_pow), a_pow, b_pow, 0.0))

            # Find the best fitting math for this pair
            models.sort(key=lambda m: m[1], reverse=True)
            best_model = models[0]
            eq_type, r2, a, b, c = best_model
            
            # We trigger a collapse if R² > 0.50 (Explains 50% of the variance!)
            if r2 > highest_r2 and r2 > 0.250:
                highest_r2 = r2
                best_collapse = (slave_name, master_name, a, b, c, eq_type, r2)

    if best_collapse:
        slave, master, a, b, c, eq_type, r2 = best_collapse
        print(f"🔥 COUPLING DETECTED: {slave} is bound to {master} (R² Score: {r2:.3f})")
        
        if eq_type == 'quadratic':
            print(f"📐 Discovered QUADRATIC Law: {slave} = ({a:.4f} * {master}²) + ({b:.4f} * {master}) + {c:.4f}")
        elif eq_type == 'power':
            print(f"📐 Discovered POWER Law: {slave} = {a:.4f} * ({master}^{b:.4f})")
        else:
            print(f"📐 Discovered LINEAR Law: {slave} = ({a:.4f} * {master}) + {b:.4f}")
            
        print(f"💥 COLLAPSING DIMENSION: Removing {slave} from search space!")
        save_relation(slave, master, a, b, c, eq_type) # Ensure save_relation accepts eq_type!
        return True
    
    print("No strong predictive laws (R² > 0.50) found this generation. Search continues.")
    return False

def run_optuna_search():
    sampler = optuna.samplers.TPESampler(multivariate=True, constant_liar=True)
    study = optuna.create_study(direction="maximize", study_name="TriadicCollapse", sampler=sampler)
    
    num_configs = 128
    num_seeds = 8
    
    generation = 0
    while True:
        generation += 1
        relations = load_relations()
        active_space = get_active_space(relations)
        
        if len(active_space) == 0:
            print("\n ALL HYPERPARAMETERS ELIMINATED. ARCHITECTURE IS FULLY SELF-DERIVED!")
            break
            
        print(f"\n" + "="*60)
        print(f"GENERATION {generation} | Active Dimensions: {len(active_space)}")
        print(f"Active: {list(active_space.keys())}")
        print("="*60)
        
        # Randomly select Structural Dimensions for this batch
        N = random.choice([8, 16, 24, 32])
        D = random.choice([8, 16, 24, 32])
        print(f"Scale selected for batch: Nodes={N}, Dim={D}")
        
        # Ask Optuna for trials
        trials =[study.ask() for _ in range(num_configs)]
        
        # Inject user attributes so Optuna can track structural scaling
        for t in trials:
            t.set_user_attr("N", N)
            t.set_user_attr("D", D)
            
        hp_tensors = {}
        for hp_name, space in active_space.items():
            vals = [t.suggest_float(hp_name, space['low'], space['high']) for t in trials]
            hp_tensors[hp_name] = torch.tensor(vals, device=DEVICE, dtype=torch.float32).repeat_interleave(num_seeds)
            
        # Run Batch
        penalized_fitness = run_tracking_task(hp_tensors, relations, N, D, num_configs, num_seeds)
        
        # Tell Optuna
        for i, trial in enumerate(trials):
            study.tell(trial, penalized_fitness[i])
            
        best_val = study.best_value
        print(f"🏆 Best Penalized Fitness So Far: {best_val:.5f}")
        
        # Attempt to discover laws and collapse dimensions every 5 generations
        if generation > 5 and generation % 5 == 0:
            collapsed = analyze_and_collapse_dimension(study, active_space)
            if collapsed:
                print("⚠️ Restarting TPE Sampler to account for collapsed space...")
                # Start a fresh study to let Optuna focus purely on the remaining variables
                study = optuna.create_study(direction="maximize", study_name=f"TriadicCollapse_Gen{generation}", sampler=sampler)

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    # Ensure relations file starts clean on fresh runs (comment out to resume previous state)
    if os.path.exists(RELATIONS_FILE):
        os.remove(RELATIONS_FILE)
        
    run_optuna_search()