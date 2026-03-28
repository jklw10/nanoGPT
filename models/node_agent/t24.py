import os
import json
import random
import math
import time
import warnings
import itertools
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.linear_model import RANSACRegressor, LinearRegression

import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
from optuna.importance import MeanDecreaseImpurityImportanceEvaluator
import concurrent.futures

import parts.utils as utils

# ==========================================
# 0. SETUP & DYNAMIC HP MANAGER
# ==========================================
torch.set_float32_matmul_precision('high')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware: {DEVICE}")

RELATIONS_FILE = "RL/hp_relations.json"
CHECKPOINT_FILE = "RL/best_blueprint.json"
SUSPEND_FILE = "RL/suspended_rules.json"

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
    """ Central nervous system for dynamic hyperparameter equations. """
    def __init__(self, active_tensors, relations, N, D, batch_size):
        self.N_val, self.D_val = float(N), float(D)
        self.batch_size = batch_size
        self.device = DEVICE
        self.defaults = {'speed': 0.5, 'width': 0.01, 'lateral_decay': 0.01}
        self.update(active_tensors, relations)

    def update(self, active_tensors, relations):
        self.tensors = active_tensors
        self.relations = relations
        self.cache = {
            'N': torch.full((self.batch_size,), self.N_val, device=self.device),
            'D': torch.full((self.batch_size,), self.D_val, device=self.device)
        }

    def resolve_all(self):
        res = {k: self.get(k, 1) for k in FULL_SPACE.keys()}
        res.update({k: self.get(k, 1) for k in self.defaults.keys()})
        return res

    def get(self, name, ndim):
        if name in self.cache:
            return self.cache[name].view(-1, *([1] * (ndim - 1)))
            
        if name in self.tensors:
            val = self.tensors[name]
            self.cache[name] = val
            return val.view(-1, *([1] * (ndim - 1)))

        if name not in self.relations:
            def_val = self.defaults.get(name, 1.0)
            val = torch.tensor(def_val, device=self.device)
            return val.expand(self.batch_size).view(-1, *([1] * (ndim - 1)))

        rel = self.relations[name]
        eq_type = rel.get('type', 'linear')
        master = rel['master']
        
        if isinstance(master, list):
            m1 = self.get(master[0], 1).flatten()
            m2 = self.get(master[1], 1).flatten()
            if eq_type == 'multi_power':
                p1 = torch.pow(m1 + 1e-8, rel['b'])
                p2 = torch.pow(m2 + 1e-8, rel['c'])
                val = rel['a'] * p1 * p2
            else: 
                val = rel['a'] * m1 + rel['b'] * m2 + rel['c']
        else:
            m_val = self.get(master, 1).flatten() 
            if eq_type == 'power':
                val = rel['a'] * torch.pow(m_val + 1e-8, rel['b'])
            elif eq_type == 'quadratic':
                val = rel['a'] * (m_val**2) + rel['b'] * m_val + rel['c']
            else: 
                val = rel['a'] * m_val + rel['b']
            
        if name in FULL_SPACE:
            val = torch.clamp(
                val, FULL_SPACE[name]['low'], FULL_SPACE[name]['high']
            )
            
        self.cache[name] = val
        return val.view(-1, *([1] * (ndim - 1)))

# ==========================================
# 1. BRAIN DEFINITION
# ==========================================
class TriadPiece(nn.Module):
    def __init__(self, dim, batch_size, num_nodes, hp_manager):
        super().__init__() 
        self.D = dim
        base_w = torch.randn(1, num_nodes, dim, dim, device=DEVICE)
        base_w = base_w / math.sqrt(self.D)
        
        self.register_buffer('weight', base_w.expand(batch_size,-1,-1,-1).clone())
        self.register_buffer('g_ema', torch.randn_like(self.weight) * 0.01) 
        
        base_e = 0.01 + 0.01 * torch.randn(batch_size, num_nodes, dim, device=DEVICE)
        self.register_buffer('E_baseline', base_e)
        
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

        base_w = torch.randn(1, self.weight.shape[1], self.D, self.D, device=DEVICE)
        self.weight.data.copy_(base_w.expand_as(self.weight) / math.sqrt(self.D))
        w_norm = utils.rms_norm(self.weight.data, dim=[-1,-2])
        self.weight.data.copy_(w_norm * self.hp_w_scale) 
        
        self.g_ema.normal_().mul_(0.01)
        self.E_baseline.normal_().mul_(0.01).add_(0.01)

    def forward(self, x):
        raw_pred = (x.unsqueeze(2) @ self.weight).squeeze(2)
        return utils.buntanh(raw_pred, self.hp_untanh, math.sqrt(2 + math.sqrt(2)))
    
    def get_gradient(self, layer_input, error_signal):
        E_curr = F.softmax(error_signal, dim=-1)
        self.E_baseline.mul_(1.0 - self.hp_ema_speed).add_(self.hp_ema_speed * E_curr)
        advantage = E_curr - self.E_baseline
        plasticity = 1.0 + utils.rms_norm(advantage.unsqueeze(3) * self.hp_rew_sens)  
        noisy_state = utils.noised_input(layer_input, error_signal, self.hp_noise_scale)
        lg1 = utils.rms_norm(-torch.einsum('bni,bnj->bnij', error_signal, noisy_state))
        return plasticity * lg1
    
    def step(self, grad, step_counter):
        grad = grad - (self.hp_wd * self.weight)
        center = (step_counter * self.hp_speed) % self.D
        grad = grad * utils.wrapping_gaussian_kernel2(grad, center, self.hp_width)
        
        self.g_ema.mul_(self.hp_momentum).add_((1.0 - self.hp_momentum) * grad)
        self.weight.add_(self.hp_lr * self.g_ema)
        w_norm = utils.rms_norm(self.weight.data, dim=[-1,-2])
        self.weight.data.copy_(w_norm * self.hp_w_scale)

class PureTriadicBrain(nn.Module):
    def __init__(self, dim, batch_size, num_nodes, hp_manager):
        super().__init__()
        self.N = num_nodes
        self.D = dim
        self.hpm = hp_manager
        
        self.w1 = TriadPiece(dim, batch_size, self.N, hp_manager)
        self.w2 = TriadPiece(dim, batch_size, self.N, hp_manager)
        self.w3 = TriadPiece(dim, batch_size, self.N, hp_manager)
        
        base_t = 1e-5 * torch.randn(batch_size, self.N, dim, device=DEVICE)
        self.register_buffer('state', base_t.clone())
        self.register_buffer('output', base_t.clone())
        self.register_buffer('target', base_t.clone())
        self.register_buffer('step_counter', torch.tensor(0.0, device=DEVICE))
        
        self.register_buffer('A', torch.randn(batch_size, self.N, self.N, device=DEVICE))
        self.register_buffer('A_ema', torch.zeros(batch_size, self.N, self.N, device=DEVICE))
        self.A_ema[:, 2, 0] += 3.0  
        self.A_ema[:, 2, 1] -= 2.0 
        self.A.data.copy_(F.softmax(self.A, dim=2))
        
        self.register_buffer('A_mask', torch.ones_like(self.A))
        self.A_mask[:, 0:2, :] = 0.0

    def reset(self, active_tensors, relations):
        self.hpm.update(active_tensors, relations)
        hps = self.hpm.resolve_all()
        
        self.hp_c_moment = hps['c_moment'].view(-1, 1, 1)
        self.hp_a_variance = hps['a_variance'].view(-1, 1, 1)

        self.w1.reset(hps)
        self.w2.reset(hps)
        self.w3.reset(hps)
        
        self.state.normal_().mul_(1e-5)
        self.output.normal_().mul_(1e-5)
        self.target.normal_().mul_(1e-5)
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
        q = self.w3(self.state)
        k = self.w2(self.state)
        pred = self.w1(self.state) 

        raw_A = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.D)
        raw_A = utils.rms_norm(raw_A, dim=[-1,-2])
        
        is_first = (self.step_counter == 1.0).float().view(1, 1, 1)
        c_mom = self.hp_c_moment
        
        new_A_ema = raw_A * is_first + \
                    (self.A_ema * c_mom + raw_A * (1.0 - c_mom)) * (1.0 - is_first)
        self.A_ema.copy_(new_A_ema * self.A_mask)
        
        raw_A = utils.sinkhorn_knopp(self.A_ema)
        sparse_A = torch.where(raw_A > (0.5 / self.N), raw_A, 0.0)
        self.A.copy_(utils.sinkhorn_knopp(sparse_A) * self.A_mask)
        
        vd = F.relu(self.hp_a_variance - self.A.var(dim=[-1, -2], keepdim=True))
        target = torch.bmm(self.A, self.output + (vd * torch.randn_like(self.output)))
        target[:, 0, :] = eye_env
        target[:, 1, :] = stomach_env

        err1_base = pred - target

        err2 = k - err1_base.detach()
        err3 = q - err2.detach()
        err1 = pred - (target + err3.detach())

        self.w1.step(self.w1.get_gradient(self.state, err1), self.step_counter)
        self.w2.step(self.w2.get_gradient(self.state, err2), self.step_counter)
        self.w3.step(self.w3.get_gradient(self.state, err3), self.step_counter)

        self.output.copy_(pred)
        self.state.copy_(target)

# ==========================================
# 2. VECTORIZED BATCH EVALUATOR 
# ==========================================
class TrajectoryRollout(nn.Module):
    """ Compiles entire 1500-step trace into a single GPU graph. """
    def __init__(self, brain, total_steps, eval_start, D):
        super().__init__()
        self.brain = brain
        self.total_steps = total_steps
        self.eval_start = eval_start
        self.D = D

    def forward(self, all_food_pos, agent_pos, agent_vel, stomach_noise):
        batch_size = agent_pos.shape[0]
        device = agent_pos.device
        
        dist_history = torch.zeros(self.total_steps, batch_size, device=device)
        eye_env = torch.zeros(batch_size, self.D, device=device)
        
        for step in range(self.total_steps):
            diffs = all_food_pos[step] - agent_pos
            dists = torch.norm(diffs, dim=1, keepdim=True)
            
            eye_env[:, 0:2] = diffs / (dists + 1e-5)
            stomach_env = stomach_noise[step] * (dists * 1e-5)
            
            self.brain(eye_env, stomach_env)
            
            motor_out = self.brain.output[:, 2, 0:2]
            agent_vel.mul_(0.85).add_(motor_out, alpha=0.2)
            agent_pos.add_(agent_vel).clamp_(-4.0, 4.0)
            
            dist_history[step] = dists.squeeze(1)

        return dist_history[self.eval_start:]

def run_tracking_task(task, hp_tensors, relations, configs, seeds, stream):
    with torch.cuda.stream(stream):
        b_size = configs * seeds
        fitness_total = 0
        total_steps, _ = 1500, 500
        
        phase_x = (torch.rand(b_size, device=DEVICE) * 2 * math.pi).unsqueeze(0)
        phase_y = (torch.rand(b_size, device=DEVICE) * 2 * math.pi).unsqueeze(0)
        T = torch.arange(total_steps, device=DEVICE).float().view(-1, 1) / 100.0
        
        f_smooth = torch.stack([
            torch.sin(T + phase_x) * 2.0, 
            torch.sin(T * 2.0 + phase_y) * 1.5
        ], dim=-1)
        
        f_chaos = torch.stack([
            torch.sin(T * 1.3 + phase_x) * 2.0 + torch.cos(T*2.7 + phase_x),
            torch.sin(T * 0.9 + phase_y) * 1.5 + torch.cos(T*1.5 + phase_y)*1.5
        ], dim=-1)

        st_smooth = torch.randn(total_steps, b_size, task.D, device=DEVICE)
        st_chaos = torch.randn(total_steps, b_size, task.D, device=DEVICE)

        for t_type in["smooth", "chaotic"]:
            task.brain.reset(hp_tensors, relations)
            agent_pos = torch.full((b_size, 2), -3.0, device=DEVICE)
            agent_vel = torch.zeros(b_size, 2, device=DEVICE)
            
            food_pos = f_smooth if t_type == "smooth" else f_chaos
            st_noise = st_smooth if t_type == "smooth" else st_chaos

            dist_hist = task(food_pos, agent_pos, agent_vel, st_noise)
            
            mean_dist, var_dist = dist_hist.mean(dim=0), dist_hist.var(dim=0)
            fit_mat = torch.exp(-(mean_dist + var_dist)).view(configs, seeds)
            fitness_total += fit_mat.mean(dim=1) - fit_mat.var(dim=1)

    stream.synchronize()
    return (fitness_total / 2.0).cpu().numpy()

# ==========================================
# 3. RULE ENGINE & PHASE STATE MACHINE
# ==========================================
def load_file(file):
    return json.load(open(file, 'r')) if os.path.exists(file) else {}

def save_file(data, path):
    with open(path, 'w') as f: 
        json.dump(data, f, indent=4)

def suspend_rule_to_file(slave, law, reason=""):
    data = load_file(SUSPEND_FILE)
    key = f"{slave}_{int(time.time())}"
    data[key] = {
        "parameter": slave, "law": law, "reason": reason, 
        "date": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    save_file(data, SUSPEND_FILE)

def get_active_space(relations):
    return {k: v for k, v in FULL_SPACE.items() if k not in relations}

class DiscoveryEngine:
    """ Phase Transitions, Yield Tracking, and Rule Acceptance """
    def __init__(self, gen_count):
        self.phase = "WARMUP"
        self.tested_rule, self.suspended_rule, self.suspended_law = None, None, None
        self.start_gen, self.start_best, self.strikes = 0, -float('inf'), 0
        
        self.test_yields, self.start_emas = [],[]
        self.ablation_yields, self.ablation_emas = [],[]
        self.blacklist = {}
        self.gen_count = gen_count

    def print_status(self, generation, num_active):
        print("\n" + "="*80)
        match self.phase:
            case "TESTING": 
                status = f"🧪 PHASE 1: GAUNTLET '{self.tested_rule}' " \
                         f"(Strikes: {self.strikes}/2)"
            case "SUSPENDING": 
                status = f"🔪 PHASE 2: ABLATING '{self.suspended_rule}'"
            case _:
                status = f"🌍 MAPPING LANDSCAPE ({self.phase})"
                                
        print(f"MACRO-GEN {generation//self.gen_count + 1} | "
              f"Active HPs: {num_active} | {status}")
        print("="*80)

    def record_metrics(self, current_yield, ema_yield):
        if self.phase == "TESTING":
            self.test_yields.append(current_yield)
            self.start_emas.append(ema_yield)
        elif self.phase == "SUSPENDING":
            self.ablation_yields.append(current_yield)
            self.ablation_emas.append(ema_yield)

    def check_gauntlet(self, top_fits, hist_fits, current_yield, ema_yield, 
                       relations, generation):
        if not (self.phase == "TESTING" and len(hist_fits) > 0):
            return
        _, p_val = mannwhitneyu(top_fits, hist_fits, alternative='less')
        if not (p_val < 0.01 and current_yield < (0.6 * ema_yield)):
            return
            
        self.strikes += 1
        if self.strikes < 2:
            print(f"\n⚠️ ANOMALY DETECTED! (Strike {self.strikes}/2)")
            return
            
        print(f"\n📉 FATAL FAILURE! '{self.tested_rule}' failed the Gauntlet.")
        suspend_rule_to_file(
            self.tested_rule, relations[self.tested_rule], "Failed Gauntlet"
        )
        del relations[self.tested_rule]
        self.blacklist[self.tested_rule] = generation + self.gen_count * 3 
        self.transition_to("IDLE")
        save_file(relations, RELATIONS_FILE)
        
    def transition_to(self, new_phase):
        self.phase = new_phase
        if not new_phase == "IDLE":
            return
        self.tested_rule = None
        self.suspended_rule = None
        self.suspended_law = None
        self.test_yields.clear()
        self.start_emas.clear()
        self.ablation_yields.clear()
        self.ablation_emas.clear()

    def evaluate_cycle(self, gen, best_fit, relations, study, active_space):
        if gen < (self.gen_count * 2) or gen % self.gen_count != 0:
            return 
            
        if self.phase == "TESTING":
            self.test_cycle(best_fit, relations)
            self.transition_to("IDLE")
        elif self.phase == "SUSPENDING":
            self.suspend_cycle(best_fit, relations)
            self.transition_to("IDLE")

        if self.phase in ["IDLE", "WARMUP"]:
            self.default_cycle(gen, best_fit, relations, study, active_space)

    def default_cycle(self, gen, best_fit, relations, study, active_space):
        search_space = {k: v for k, v in active_space.items() 
                        if k not in self.blacklist}
        proposed = analyze_and_collapse_dimension(study, search_space)
            
        if proposed:
            self.tested_rule = proposed.pop('slave')
            relations[self.tested_rule] = proposed
            self.strikes = 0
            print(f"\n💡 Initiating Phase 1 Testing for '{self.tested_rule}'")
            self.transition_to("TESTING")
        elif relations:
            self.suspended_rule = random.choice(list(relations.keys()))
            self.suspended_law = relations.pop(self.suspended_rule)
            print(f"\n🧪 No new rules detected. Ablating "
                  f"'{self.suspended_rule}' to test landscape.")
            self.transition_to("SUSPENDING")
            
        self.start_gen, self.start_best = gen, best_fit
        save_file(relations, RELATIONS_FILE)

    def suspend_cycle(self, global_best_fitness, relations):
        avg_yield = np.mean(self.ablation_yields) if self.ablation_yields else 0
        avg_ema = np.mean(self.ablation_emas) if self.ablation_emas else 0
            
        if global_best_fitness > self.start_best or avg_yield > (avg_ema*1.05):
            print(f"\n💥 ABLATION SUCCESS! Removing '{self.suspended_rule}' "
                  f"unleashed performance!")
            suspend_rule_to_file(
                self.suspended_rule, self.suspended_law, "Ablation improved"
            )
        else:
            print(f"\n🛡️ RULE VINDICATED! '{self.suspended_rule}' is "
                  f"structural. Reinstating.")
            relations[self.suspended_rule] = self.suspended_law
        save_file(relations, RELATIONS_FILE)

    def test_cycle(self, global_best_fitness, relations):
        avg_yield = np.mean(self.test_yields) if self.test_yields else 0
        avg_ema = np.mean(self.start_emas) if self.start_emas else 1e-5
            
        if global_best_fitness > self.start_best or avg_yield > (0.90*avg_ema):
            print(f"\n🎓 HYPOTHESIS ACCEPTED! Rule '{self.tested_rule}' "
                  f"is structural. (Yield: {avg_yield:.4f})")
            return
        
        print(f"\n📉 HYPOTHESIS REJECTED! Rule '{self.tested_rule}' "
              f"degraded yield. Pruning.")
        suspend_rule_to_file(
            self.tested_rule, relations[self.tested_rule], "Degraded yield"
        )
        del relations[self.tested_rule]
        save_file(relations, RELATIONS_FILE)

# ==========================================
# 4. STATISTICAL ANALYSIS MODULE
# ==========================================
def calculate_r2(y_true, y_pred):
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1.0 - (np.sum((y_true - y_pred)**2) / ss_tot) if ss_tot != 0 else 0.0

def get_adj_r2(r2, n, k):
    return 1 - ((1 - r2) * (n - 1) / (n - k - 1)) if n > k + 1 else 0.0

def fit_and_score(X, y_target, n_samples):
    """ Helper to keep RANSAC logic cleanly out of the main loop """
    min_s = max(int(len(y_target) * 0.25), 3) 
    try:
        model = LinearRegression(fit_intercept=True)
        ransac = RANSACRegressor(model, min_samples=min_s, random_state=42)
        ransac.fit(X, y_target)
        coef, inter = ransac.estimator_.coef_, ransac.estimator_.intercept_
        mask = ransac.inlier_mask_
    except Exception:
        return 0.0, 0.0, None, None
        
    n_in = np.sum(mask)
    if coef is None or n_in <= X.shape[1] + 1:
        return 0.0, 0.0, None, None
        
    y_in = y_target[mask]
    y_pred = (X[mask] @ coef) + inter
    r2 = calculate_r2(y_in, y_pred)
    adj_r2 = get_adj_r2(r2, n_in, X.shape[1])
    score = adj_r2 * (n_in / n_samples)
    
    return score, r2, coef, inter

def analyze_and_collapse_dimension(study, active_space):
    print(" INITIATING MULTI-VARIATE DIMENSIONAL COLLAPSE...")
    eval_f = MeanDecreaseImpurityImportanceEvaluator()
    try: 
        imp = optuna.importance.get_param_importances(study, evaluator=eval_f)
    except Exception: 
        return None

    df = study.trials_dataframe()
    df = df[df['state'] == 'COMPLETE']
    if len(df) < 20: 
        return None
    
    top_df = df[df['value'] >= df['value'].quantile(0.90)].copy()
    top_df = top_df.rename(columns={c: c.replace('params_','') for c in top_df})
    cols =[c for c in list(active_space.keys()) + ['N', 'D'] if c in top_df]

    best_collapse, highest_score = None, 0.0
    n_samples = len(top_df)
    v_masters =[c for c in cols if c in['N','D'] or imp.get(c, 0.0) > 0.05]
    
    # 1. TEST 1D LAWS
    for col1, col2 in itertools.combinations(cols, 2):
        imp1, imp2 = imp.get(col1, 1.0), imp.get(col2, 1.0)
        if imp1 >= imp2 and col1 in v_masters: 
            master, slave = col1, col2
        elif imp2 > imp1 and col2 in v_masters:
            master, slave = col2, col1
        else: 
            continue
        
        if not master or slave in ['N', 'D']: 
            continue

        x, y = top_df[master].values, top_df[slave].values
        
        # 1A. LINEAR
        score, r2, coef, inter = fit_and_score(x.reshape(-1, 1), y, n_samples)
        if score > highest_score: 
            highest_score = score
            best_collapse = (slave, master, coef[0], inter, 0.0, 'linear')
        
        # 1B. POWER LAW
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if (x > 0).all() and (y > 0).all():
                score, r2, coef, inter = fit_and_score(
                    np.log(x).reshape(-1, 1), np.log(y), n_samples
                )
                if score > highest_score: 
                    highest_score = score
                    best_collapse = (slave, master, np.exp(inter), 
                                     coef[0], 0.0, 'power')

    # 2. TEST 2D LAWS (Restored)
    for slave in cols:
        if slave in ['N', 'D']: 
            continue
        for m1, m2 in itertools.combinations(v_masters, 2):
            if slave in[m1, m2]: 
                continue
            
            x1, x2 = top_df[m1].values, top_df[m2].values
            y = top_df[slave].values
            
            # 2A. MULTI-LINEAR
            X_mlin = np.column_stack((x1, x2))
            score, r2, coef, inter = fit_and_score(X_mlin, y, n_samples)
            if score > highest_score:
                highest_score = score
                best_collapse = (slave, [m1, m2], coef[0], 
                                 coef[1], inter, 'multi_linear')

            # 2B. MULTI-POWER LAW
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if (x1 > 0).all() and (x2 > 0).all() and (y > 0).all():
                    X_mpow = np.column_stack((np.log(x1), np.log(x2)))
                    score, r2, coef, inter = fit_and_score(
                        X_mpow, np.log(y), n_samples
                    )
                    if score > highest_score:
                        highest_score = score
                        best_collapse = (slave, [m1, m2], np.exp(inter), 
                                         coef[0], coef[1], 'multi_power')

    if best_collapse and highest_score > 0.35: 
        slave, master, a, b, c, eq_type = best_collapse
        print(f"\n💡 COUPLING DETECTED: {slave} bound to {master} "
              f"(Confidence: {highest_score:.3f})")
        return {"slave": slave, "master": master, "a": a, "b": b, "c": c, 
                "type": eq_type, "strikes": 0}
    return None

# ==========================================
# 5. MAIN ORCHESTRATOR
# ==========================================
def setup_environment():
    os.makedirs("RL", exist_ok=True)
    for old_file in["hp_relations.json", "best_blueprint.json"]:
        if os.path.exists(old_file) and not os.path.exists(f"RL/{old_file}"): 
            os.rename(old_file, f"RL/{old_file}")

    ckpt = load_file(CHECKPOINT_FILE)
    return (load_file(RELATIONS_FILE), 
            ckpt.get("active_hyperparameters", None), 
            ckpt.get("global_best_fitness", -float('inf')))

def run_optuna_search():
    relations, best_params, global_best_fitness = setup_environment()
    num_configs, num_seeds = 128, 8
    b_size = num_configs * num_seeds
    nd_combinations = list(itertools.product([8, 16, 24],[8, 16, 24]))
    
    # Optuna Setup
    sampler = optuna.samplers.TPESampler(
        multivariate=True, constant_liar=True, 
        n_startup_trials=len(nd_combinations)*num_configs
    )
    study = optuna.create_study(
        direction="maximize", study_name="TriadicCollapse", sampler=sampler
    )
    if best_params: 
        active_bp = {k: v for k, v in best_params.items() 
                     if k in get_active_space(relations)}
        study.enqueue_trial(active_bp)

    # Hardware Setup
    print("\n[INIT] Compiling static computation graphs...")
    brain_cache, streams = {}, {}
    for N, D in nd_combinations:
        hpm_dummy = HPManager({}, relations, N, D, b_size)
        brain = PureTriadicBrain(
            dim=D, batch_size=b_size, num_nodes=N, hp_manager=hpm_dummy
        ).to(DEVICE)
        
        task = TrajectoryRollout(brain, total_steps=1500, eval_start=500, D=D)
        brain_cache[(N, D)] = torch.compile(task, mode="reduce-overhead")
        streams[(N, D)] = torch.cuda.Stream()
        
    engine = DiscoveryEngine(gen_count=len(nd_combinations))
    scale_base = {nd: {'yield_ema': 0.0, 'best': -float('inf'), 'top':[]} 
                  for nd in nd_combinations}
    
    print("[INIT] Compilations Complete! Launching Engine...\n")
    generation = 0
    
    while True:
        active_space = get_active_space(relations)
        if not active_space: 
            break
            
        engine.print_status(generation, len(active_space))
        
        # Parallel Task Generation
        macro_tasks = {}
        for N, D in nd_combinations:
            trials =[study.ask() for _ in range(num_configs)]
            for t in trials:
                t.suggest_categorical("N", [8, 16, 24])
                t.suggest_categorical("D", [8, 16, 24])
                
            hp_tensors = {}
            for hp_name, space in active_space.items():
                vals =[t.suggest_float(hp_name, space['low'], space['high']) 
                       for t in trials]
                t_tensor = torch.tensor(vals, device=DEVICE, dtype=torch.float32)
                hp_tensors[hp_name] = t_tensor.repeat_interleave(num_seeds)
                
            macro_tasks[(N, D)] = (trials, hp_tensors)
        
        # Parallel Execution
        results = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(nd_combinations)
        ) as executor:
            futures = {
                executor.submit(
                    run_tracking_task, brain_cache[nd], tasks[1], relations, 
                    num_configs, num_seeds, streams[nd]
                ): nd for nd, tasks in macro_tasks.items()
            }
            for f in concurrent.futures.as_completed(futures): 
                results[futures[f]] = f.result()
                
        # Synchronous Update
        for N, D in nd_combinations:
            generation += 1
            trials, penalized_fitness = macro_tasks[(N, D)][0], results[(N, D)]
            
            for i, trial in enumerate(trials): 
                study.tell(trial, penalized_fitness[i])
                
            sorted_fits = np.sort(penalized_fitness)[::-1]
            c_yield = np.mean(sorted_fits[:int(num_configs * 0.20)])
            ema_yield = scale_base[(N,D)]['yield_ema']
            print(f"[{N}x{D}] Best: {study.best_value:.4f} "
                  f"(Hist: {scale_base[(N,D)]['best']:.4f}) | "
                  f"Yield: {c_yield:.4f} (EMA: {ema_yield:.4f})")
            
            # Baseline tracking
            if engine.phase not in["TESTING", "SUSPENDING"]:
                scale_base[(N,D)]['yield_ema'] = c_yield if ema_yield == 0.0 \
                                                 else (0.8*ema_yield+0.2*c_yield)
                scale_base[(N,D)]['best'] = max(
                    scale_base[(N,D)]['best'], study.best_value
                )
                top_new = sorted_fits[:int(num_configs * 0.20)].tolist()
                scale_base[(N,D)]['top'] = (scale_base[(N,D)]['top']+top_new)[-500:]

            engine.record_metrics(c_yield, ema_yield)
            if study.best_value > global_best_fitness:
                global_best_fitness = study.best_value
                print(f"🌟 NEW GLOBAL BEST! Fitness: {global_best_fitness:.5f}")
                best_clean = {k: v for k, v in study.best_trial.params.items() 
                              if k not in ['N', 'D']}
                save_file({
                    "global_best_fitness": global_best_fitness, 
                    "generation": generation, 
                    "structural_scale": {"N": N, "D": D}, 
                    "active_laws": relations, 
                    "active_hyperparameters": best_clean
                }, CHECKPOINT_FILE)

            engine.check_gauntlet(
                sorted_fits[:int(num_configs * 0.20)], scale_base[(N,D)]['top'], 
                c_yield, ema_yield, relations, generation
            )
            
            engine.evaluate_cycle(
                generation, global_best_fitness, relations, study, active_space
            )

    print("\n🚀 ARCHITECTURE IS FULLY SELF-DERIVED!")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    run_optuna_search()