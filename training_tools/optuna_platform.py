import os
import json
import random
import torch
import numpy as np
import optuna
from scipy.stats import mannwhitneyu
import time
from sklearn.linear_model import RANSACRegressor, LinearRegression
import warnings
import itertools
from optuna.importance import MeanDecreaseImpurityImportanceEvaluator
import concurrent.futures
from enum import Enum

# ==========================================
# RELATION MAPPER (Fully Generic)
# ==========================================
class RelationMapper:
    def __init__(self, defaults: dict, bounds: dict = None):
        self.defaults = defaults
        self.bounds = bounds or {}

    def resolve(self, active_tensors, relations, constants, batch_size, device):
        cache = {}
        for k, v in constants.items():
            cache[k] = torch.full((batch_size,), float(v), device=device)
            
        for k, v in active_tensors.items():
            cache[k] = v

        def get_val(name):
            if name in cache:
                return cache[name]
                
            if name in relations:
                rel = relations[name]
                eq_type = rel.get('type', 'linear')
                
                if isinstance(rel['master'], list):
                    m1_val = get_val(rel['master'][0]).flatten()
                    m2_val = get_val(rel['master'][1]).flatten()
                    
                    if eq_type == 'multi_power':
                        val = rel['a'] * torch.pow(m1_val + 1e-8, rel['b']) * torch.pow(m2_val + 1e-8, rel['c'])
                    else: 
                        val = rel['a'] * m1_val + rel['b'] * m2_val + rel.get('c', 0.0)
                else:
                    master_val = get_val(rel['master']).flatten() 
                    if eq_type == 'power':
                        val = rel['a'] * torch.pow(master_val + 1e-8, rel['b'])
                    elif eq_type == 'quadratic':
                        val = rel['a'] * (master_val**2) + rel['b'] * master_val + rel.get('c', 0.0)
                    else: 
                        val = rel['a'] * master_val + rel.get('b', 0.0)
            else:
                default_val = self.defaults.get(name, 1.0)
                val = torch.tensor(default_val, device=device).expand(batch_size)
            
            if name in self.bounds:
                val = torch.clamp(val, self.bounds[name]['low'], self.bounds[name]['high'])
                
            cache[name] = val
            return val

        all_targets = set(self.defaults.keys()) | set(relations.keys()) | set(self.bounds.keys())
        resolved_dict = {}
        for key in all_targets:
            resolved_dict[key] = get_val(key)
            
        return resolved_dict

# ==========================================
# CONFIG & STATE MANAGEMENT
# ==========================================
RELATIONS_FILE = "RL/hp_relations.json"
CHECKPOINT_FILE = "RL/best_blueprint.json"
SUSPENDED_RULES_FILE = "RL/suspended_rules.json"

class EnginePhase(Enum):
    WARMUP = "WARMUP"
    IDLE = "IDLE"
    TESTING = "TESTING"
    SUSPENDING = "SUSPENDING"

class EquationType(str, Enum):
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    POWER = "power"
    MULTI_LINEAR = "multi_linear"
    MULTI_POWER = "multi_power"

def load_file(file):
    if not os.path.exists(file):
        return {}
    with open(file, 'r') as f:
        return json.load(f)

def save_relations(rels, path):
    with open(path, 'w') as f:
        json.dump(rels, f, indent=4)

def suspend_rule_to_file(slave, law, reason=""):
    data = load_file(SUSPENDED_RULES_FILE)
    key = f"{slave}_{int(time.time())}"
    data[key] = {
        "parameter": slave,
        "law": law,
        "reason": reason,
        "generation_pruned": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(SUSPENDED_RULES_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def get_active_space(relations, param_space):
    return {k: v for k, v in param_space.items() if k not in relations}

# ==========================================
# SEARCH ENGINE
# ==========================================
class SearchEngine:
    def __init__(self, gen_count, struct_keys):
        self._phase = EnginePhase.WARMUP
        self._tested_rule = None
        self._suspended_rule = None
        self._suspended_law = None
        self._start_gen = 0
        self._start_best = -float('inf')
        self._yields =[]
        self._emas =[]
        self._strikes = 0
        self._gen_count = gen_count
        self._struct_keys = struct_keys
        self._blacklist = {}

        self._epoch_handlers = {
            EnginePhase.TESTING: self._handle_testing_epoch,
            EnginePhase.SUSPENDING: self._handle_suspending_epoch,
            EnginePhase.IDLE: self._handle_idle_epoch,
            EnginePhase.WARMUP: self._handle_idle_epoch,
        }

    def get_phase(self):
        return self._phase

    def evaluate_epoch(self, generation, global_best_fitness, relations, study, active_space):
        if not self._is_epoch_ready(generation):
            return
        handler = self._epoch_handlers.get(self._phase)
        if handler:
            handler(generation, global_best_fitness, relations, study, active_space)

    def record_trial_results(self, current_top_fits, historical_fits, current_yield, ema_yield, relations, generation):
        if self._phase == EnginePhase.SUSPENDING:
            self._yields.append(current_yield)
            self._emas.append(ema_yield)

        if self._phase != EnginePhase.TESTING or len(historical_fits) == 0:
            return

        if self._is_anomaly(current_top_fits, historical_fits, current_yield, ema_yield):
            self._register_strike(relations, generation)

    def _transition_to_idle(self):
        self._phase = EnginePhase.IDLE
        self._tested_rule = None
        self._suspended_rule = None
        self._suspended_law = None

    def _transition_to_testing(self, rule, new_law, relations, generation, best_fitness):
        self._tested_rule = rule
        relations[rule] = new_law.copy()
        
        self._phase = EnginePhase.TESTING
        self._start_gen = generation
        self._start_best = best_fitness
        self._strikes = 0
        save_relations(relations, RELATIONS_FILE)

    def _transition_to_suspending(self, rule, relations, generation, best_fitness):
        self._suspended_rule = rule
        self._suspended_law = relations[rule].copy()
        del relations[rule]
        
        self._phase = EnginePhase.SUSPENDING
        self._start_gen = generation
        self._start_best = best_fitness
        self._yields.clear()
        self._emas.clear()
        save_relations(relations, RELATIONS_FILE)

    def _prune_rule(self, rule, law, reason, relations, generation, blacklist_delay=0):
        suspend_rule_to_file(rule, law, reason)
        if rule in relations:
            del relations[rule]
        if blacklist_delay > 0:
            self._blacklist[rule] = generation + blacklist_delay
        save_relations(relations, RELATIONS_FILE)

    def _handle_idle_epoch(self, generation, global_best_fitness, relations, study, active_space):
        search_space = self.get_search_space(active_space, generation)
        proposed = analyze_and_collapse_dimension(study, search_space, self._struct_keys, False)
        
        if proposed:
            rule = proposed['slave']
            del proposed['slave']
            print(f"\n Initiating Phase 1 Testing for '{rule}'")
            self._transition_to_testing(rule, proposed, relations, generation, global_best_fitness)
            return
            
        if len(relations) > 0:
            rule = random.choice(list(relations.keys()))
            print(f"\n No new rules detected. Ablating '{rule}' to test landscape.")
            self._transition_to_suspending(rule, relations, generation, global_best_fitness)

    def _handle_testing_epoch(self, generation, global_best_fitness, relations, study, active_space):
        if self._is_hypothesis_proven(global_best_fitness):
            print(f"\n HYPOTHESIS ACCEPTED! Phase 1 Rule '{self._tested_rule}' pushed the peak.")
            self._transition_to_idle()
            return
            
        candidate = self._get_ablation_candidate(relations)
        if candidate:
            print(f"\n Rule survived but didn't improve peak. Initiating Phase 2 Ablation on '{candidate}'")
            self._transition_to_suspending(candidate, relations, generation, global_best_fitness)
            return
            
        print("\n Rule didn't improve peak, no others to ablate. Pruning.")
        self._prune_rule(self._tested_rule, relations[self._tested_rule], "Failed to push peak", relations, generation)
        self._transition_to_idle()

    def _handle_suspending_epoch(self, generation, global_best_fitness, relations, study, active_space):
        if self._is_ablation_successful(global_best_fitness):
            print(f"\n ABLATION SUCCESS! Removing '{self._suspended_rule}' unleashed performance!")
            suspend_rule_to_file(self._suspended_rule, self._suspended_law, "Ablation increased yield")
            self._transition_to_idle()
            return
            
        print(f"\n RULE VINDICATED! '{self._suspended_rule}' is structural. Reinstating.")
        relations[self._suspended_rule] = self._suspended_law
        
        if self._tested_rule and self._tested_rule in relations:
            self._prune_rule(self._tested_rule, relations[self._tested_rule], "Failed Phase 2", relations, generation, blacklist_delay=self._gen_count * 4)
        else:
            save_relations(relations, RELATIONS_FILE)
            
        self._transition_to_idle()

    def _register_strike(self, relations, generation):
        self._strikes += 1
        if self._strikes < 2:
            print(f"\n ANOMALY DETECTED! (Strike {self._strikes}/2)")
            return
        print(f"\n FATAL FAILURE! '{self._tested_rule}' failed the Gauntlet.")
        self._prune_rule(self._tested_rule, relations[self._tested_rule], "Failed Gauntlet", relations, generation, blacklist_delay=self._gen_count * 3)
        self._transition_to_idle()

    def get_status_message(self, active_hps_count, generation):
        if self._phase == EnginePhase.TESTING:
            status = f" PHASE 1: GAUNTLET '{self._tested_rule}' (Strikes: {self._strikes}/2)"
        elif self._phase == EnginePhase.SUSPENDING:
            status = f" PHASE 2: ABLATING '{self._suspended_rule}'"
        else:
            status = f" MAPPING LANDSCAPE ({self._phase.name})"
        return f"MACRO-GENERATION {generation // self._gen_count + 1} | Active HPs: {active_hps_count} | {status}"

    def get_search_space(self, active_space, generation):
        active_blacklist = [k for k, exp in self._blacklist.items() if generation < exp]
        return {k: v for k, v in active_space.items() if k not in active_blacklist}

    def _is_epoch_ready(self, generation):
        return generation >= (self._gen_count * 2) and generation % self._gen_count == 0

    def _is_anomaly(self, current_top_fits, historical_fits, current_yield, ema_yield):
        if current_yield >= (0.6 * ema_yield):
            return False
        stat, p_val = mannwhitneyu(current_top_fits, historical_fits, alternative='less')
        return p_val < 0.01

    def _is_hypothesis_proven(self, global_best_fitness):
        return global_best_fitness > self._start_best

    def _is_ablation_successful(self, global_best_fitness):
        avg_yield = np.mean(self._yields) if self._yields else 0
        avg_ema = np.mean(self._emas) if self._emas else 0
        return global_best_fitness > self._start_best or avg_yield > (avg_ema * 1.05)

    def _get_ablation_candidate(self, relations):
        avail = [k for k in relations.keys() if k != self._tested_rule]
        return random.choice(avail) if avail else None

# ==========================================
# MATHEMATICS & STATISTICS
# ==========================================
def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    if ss_tot == 0: 
        return 0.0
    return 1.0 - (ss_res / ss_tot)

def get_adj_r2(r2, n, k):
    if n <= k + 1: 
        return 0.0
    return 1 - ((1 - r2) * (n - 1) / (n - k - 1))

def fit_ransac(X, y_target):
    try:
        min_s = max(int(len(y_target) * 0.25), 3) 
        ransac = RANSACRegressor(LinearRegression(fit_intercept=True), min_samples=min_s, random_state=42)
        ransac.fit(X, y_target)
        return ransac.estimator_.coef_, ransac.estimator_.intercept_, ransac.inlier_mask_
    except Exception:
        return None, None, None

def _evaluate_1d_strategies(x, y):
    strategies =[
        {
            'name': EquationType.LINEAR.value,
            'X': x.reshape(-1, 1),
            'y_target': y,
            'k': 1,
            'evaluate': lambda coef, inter, mask: (
                calculate_r2(y[mask], coef[0] * x[mask] + inter),
                (coef[0], inter, 0.0)
            )
        },
        {
            'name': EquationType.QUADRATIC.value,
            'X': np.column_stack((x**2, x)),
            'y_target': y,
            'k': 2,
            'evaluate': lambda coef, inter, mask: (
                calculate_r2(y[mask], coef[0] * (x[mask]**2) + coef[1] * x[mask] + inter),
                (coef[0], coef[1], inter)
            )
        }
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if (x > 0).all() and (y > 0).all():
            strategies.append({
                'name': EquationType.POWER.value,
                'X': np.log(x).reshape(-1, 1),
                'y_target': np.log(y),
                'k': 1,
                'evaluate': lambda coef, inter, mask: (
                    calculate_r2(y[mask], np.exp(inter) * (x[mask] ** coef[0])),
                    (np.exp(inter), coef[0], 0.0)
                )
            })
    return strategies

def _evaluate_2d_strategies(x1, x2, y):
    strategies =[
        {
            'name': EquationType.MULTI_LINEAR.value,
            'X': np.column_stack((x1, x2)),
            'y_target': y,
            'k': 2,
            'evaluate': lambda coef, inter, mask: (
                calculate_r2(y[mask], coef[0]*x1[mask] + coef[1]*x2[mask] + inter),
                (coef[0], coef[1], inter)
            )
        }
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if (x1 > 0).all() and (x2 > 0).all() and (y > 0).all():
            strategies.append({
                'name': EquationType.MULTI_POWER.value,
                'X': np.column_stack((np.log(x1), np.log(x2))),
                'y_target': np.log(y),
                'k': 2,
                'evaluate': lambda coef, inter, mask: (
                    calculate_r2(y[mask], np.exp(inter) * (x1[mask]**coef[0]) * (x2[mask]**coef[1])),
                    (np.exp(inter), coef[0], coef[1])
                )
            })
    return strategies

def analyze_and_collapse_dimension(study, active_space, struct_keys, profile=True):
    print(" INITIATING MULTI-VARIATE DIMENSIONAL COLLAPSE...")
    try:
        importances = optuna.importance.get_param_importances(study, evaluator=MeanDecreaseImpurityImportanceEvaluator())
    except Exception:
        return None

    df = study.trials_dataframe()
    df = df[df['state'] == 'COMPLETE']
    if len(df) < 20: 
        return None
    
    top_df = df[df['value'] >= df['value'].quantile(0.90)].copy()
    rename_map = {c: c.replace('params_', '') for c in top_df.columns}
    top_df = top_df.rename(columns=rename_map)
    cols =[c for c in list(active_space.keys()) + struct_keys if c in top_df.columns]

    best_collapse = None
    highest_score = 0.0  
    best_raw_r2 = 0.0    
    n_samples = len(top_df)
    valid_masters =[c for c in cols if c in struct_keys or importances.get(c, 0.0) > 0.05]
    
    for col1, col2 in itertools.combinations(cols, 2):
        imp1 = importances.get(col1, 1.0 if col1 in struct_keys else 0.0)
        imp2 = importances.get(col2, 1.0 if col2 in struct_keys else 0.0)
        
        if imp1 >= imp2 and col1 in valid_masters and col2 not in struct_keys:
            master, slave = col1, col2
        elif imp2 > imp1 and col2 in valid_masters and col1 not in struct_keys:
            master, slave = col2, col1
        else:
            continue

        x, y = top_df[master].values, top_df[slave].values
        
        for strategy in _evaluate_1d_strategies(x, y):
            coef, intercept, inlier_mask = fit_ransac(strategy['X'], strategy['y_target'])
            if coef is None or inlier_mask is None:
                continue
                
            n_inliers = np.sum(inlier_mask)
            if n_inliers <= strategy['k'] + 1:
                continue
                
            r2_val, (a, b, c) = strategy['evaluate'](coef, intercept, inlier_mask)
            adj_r2 = get_adj_r2(r2_val, n_inliers, strategy['k'])
            score = adj_r2 * (n_inliers / n_samples)
            
            if score > highest_score:
                highest_score, best_raw_r2 = score, r2_val
                best_collapse = (slave, master, a, b, c, strategy['name'])

    for slave in cols:
        if slave in struct_keys: 
            continue 
        imp_s = importances.get(slave, 0.0)
        
        for m1, m2 in itertools.combinations(valid_masters, 2):
            if slave in (m1, m2): 
                continue
            
            imp_m1 = importances.get(m1, 1.0 if m1 in struct_keys else 0.0)
            imp_m2 = importances.get(m2, 1.0 if m2 in struct_keys else 0.0)
            if imp_s >= imp_m1 or imp_s >= imp_m2: 
                continue
            
            x1, x2, y = top_df[m1].values, top_df[m2].values, top_df[slave].values
            
            for strategy in _evaluate_2d_strategies(x1, x2, y):
                coef, intercept, inlier_mask = fit_ransac(strategy['X'], strategy['y_target'])
                if coef is None or inlier_mask is None:
                    continue
                    
                n_inliers = np.sum(inlier_mask)
                if n_inliers <= strategy['k'] + 1:
                    continue
                    
                r2_val, (a, b, c) = strategy['evaluate'](coef, intercept, inlier_mask)
                adj_r2 = get_adj_r2(r2_val, n_inliers, strategy['k'])
                score = adj_r2 * (n_inliers / n_samples)
                
                if score > highest_score:
                    highest_score, best_raw_r2 = score, r2_val
                    best_collapse = (slave,[m1, m2], a, b, c, strategy['name'])

    if not best_collapse or highest_score <= 0.25:
        return None

    slave, master, a, b, c, eq_type = best_collapse
    print(f"\n COUPLING DETECTED: {slave} is bound to {master}")
    print(f"   (Inlier R²: {best_raw_r2:.3f} | Confidence Score: {highest_score:.3f})")
    
    return {"slave": slave, "master": master, "a": a, "b": b, "c": c, "type": eq_type, "strikes": 0}


# ==========================================
# OPTUNA & RUN ENGINE HELPERS
# ==========================================
def get_fresh_study(name, g_par, active_space, warmup):
    fresh_sampler = optuna.samplers.TPESampler(multivariate=True, constant_liar=True, n_startup_trials=warmup, warn_independent_sampling=False)
    study = optuna.create_study(direction="maximize", study_name=name, sampler=fresh_sampler)
    if g_par is not None:
        active_golden_params = {k: v for k, v in g_par.items() if k in active_space}
        study.enqueue_trial(active_golden_params)
    return study

def _update_ema_baselines(scale_baselines, struct_key, current_yield, scale_best, current_top_fits):
    ema_yield = scale_baselines[struct_key]['yield_ema']
    scale_baselines[struct_key]['yield_ema'] = current_yield if ema_yield == 0.0 else (0.8 * ema_yield + 0.2 * current_yield)
    
    if scale_best > scale_baselines[struct_key]['best']:
        scale_baselines[struct_key]['best'] = scale_best
        
    scale_baselines[struct_key]['top_fits'].extend(current_top_fits.tolist())
    scale_baselines[struct_key]['top_fits'] = scale_baselines[struct_key]['top_fits'][-500:]

def _save_global_checkpoint(study, struct_kwargs, struct_keys, global_best_fitness, generation, relations):
    best_params_clean = {k: v for k, v in study.best_trial.params.items() if k not in struct_keys}
    print(f" NEW GLOBAL BEST! Fitness: {global_best_fitness:.5f}. Saving to RL/...")
    
    ckpt_data = {
        "global_best_fitness": global_best_fitness,
        "generation": generation,
        "structural_scale": struct_kwargs,
        "active_hyperparameters": best_params_clean,
        "active_laws": relations 
    }
    
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(ckpt_data, f, indent=4)
    with open(f"RL/peak_gen{generation}_fit{global_best_fitness:.5f}.json", 'w') as f:
        json.dump(ckpt_data, f, indent=4)


# ==========================================
# MAIN EXECUTION
# ==========================================
def run_optuna_search(
    model_constructor, 
    task_runner, 
    param_space, 
    structural_space, 
    defaults, 
    device, 
    warmup_fn=None,
    **task_args
):
    os.makedirs("RL", exist_ok=True)
    for old_file in ["hp_relations.json", "best_blueprint.json"]:
        if os.path.exists(old_file) and not os.path.exists(f"RL/{old_file}"):
            os.rename(old_file, f"RL/{old_file}")

    global_best_fitness = -float('inf')
    num_configs = 128
    num_seeds = 8
    batch_size = num_configs * num_seeds
    
    relations = load_file(RELATIONS_FILE)
    active_space = get_active_space(relations, param_space)
    
    global_best_params = None
    if os.path.exists(CHECKPOINT_FILE):
        ckpt = load_file(CHECKPOINT_FILE)
        global_best_params = ckpt.get("active_hyperparameters", None)
        global_best_fitness = ckpt.get("global_best_fitness", -float('inf'))

    # Generic execution grid logic
    struct_keys = list(structural_space.keys())
    struct_combinations = list(itertools.product(*structural_space.values()))
    gen_count = max(len(struct_combinations), 1)
    
    study = get_fresh_study("GeneralSearch", global_best_params, active_space, gen_count * num_configs)
    scale_baselines = {struct_vals: {'yield_ema': 0.0, 'best': -float('inf'), 'top_fits':[]} for struct_vals in struct_combinations}
    
    print("\n[INIT] Compiling static computation graphs for all structural combinations...")
    brain_cache = {}
    streams = {}
    
    for struct_vals in struct_combinations:
        struct_kwargs = dict(zip(struct_keys, struct_vals))
        
        # Generic model initialization
        brain = model_constructor(batch_size=batch_size, device=device, **struct_kwargs).to(device)
        brain = torch.compile(brain, mode="reduce-overhead")
        
        # Optional user-provided warmup routine (to avoid graph recompilations mid-run)
        if warmup_fn:
            warmup_fn(brain, batch_size, device, **struct_kwargs)
            
        brain_cache[struct_vals] = brain
        streams[struct_vals] = torch.cuda.Stream()
        
    print("[INIT] Compilations Complete! Launching Multi-Scale Engine...\n")

    engine = SearchEngine(gen_count, struct_keys)
    generation = 0
    
    while True:
        active_space = get_active_space(relations, param_space)
        if not active_space:
            print("\n ALL HYPERPARAMETERS ELIMINATED. ARCHITECTURE IS FULLY SELF-DERIVED!")
            break
            
        print("\n" + "="*80)
        print(engine.get_status_message(len(active_space), generation))
        print("="*80)
        
        macro_tasks = {}
        for struct_vals in struct_combinations:
            struct_kwargs = dict(zip(struct_keys, struct_vals))
            
            trials =[study.ask() for _ in range(num_configs)]
            
            # Force Optuna trial to strictly attach to THIS iteration's structure
            for t in trials:
                for k, v in struct_kwargs.items():
                    t.suggest_categorical(k, [v]) 
                    
            hp_tensors = {}
            for hp_name, space in active_space.items():
                vals = [t.suggest_float(hp_name, space['low'], space['high']) for t in trials]
                hp_tensors[hp_name] = torch.tensor(vals, device=device, dtype=torch.float32).repeat_interleave(num_seeds)
                
            macro_tasks[struct_vals] = (trials, hp_tensors, struct_kwargs)
            
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=gen_count) as executor:
            futures = {}
            for struct_vals, (_, hp_tensors, struct_kwargs) in macro_tasks.items():
                # Task runner takes exactly the pieces it needs, ignoring N/D specific args
                f = executor.submit(
                    task_runner, 
                    brain_cache[struct_vals], 
                    hp_tensors, 
                    relations, 
                    struct_kwargs, 
                    num_configs, 
                    num_seeds, 
                    streams[struct_vals], 
                    **task_args
                )
                futures[f] = struct_vals
                
            for future in concurrent.futures.as_completed(futures):
                struct_vals = futures[future]
                results[struct_vals] = future.result()
                
        for struct_vals in struct_combinations:
            generation += 1
            trials, _, struct_kwargs = macro_tasks[struct_vals]
            penalized_fitness = results[struct_vals]
            
            for i, trial in enumerate(trials):
                study.tell(trial, penalized_fitness[i])
                
            sorted_fitness = np.sort(penalized_fitness)[::-1]
            current_top_fits = sorted_fitness[:int(num_configs * 0.20)]
            current_yield = np.mean(current_top_fits)
            
            scale_best = study.best_value
            ema_yield = scale_baselines[struct_vals]['yield_ema']
            
            # Formatting log purely dynamically based on the kwargs
            struct_str = "x".join([str(v) for v in struct_vals])
            print(f"[{struct_str}] Best: {scale_best:.4f} (Hist: {scale_baselines[struct_vals]['best']:.4f}) | Yield: {current_yield:.4f} (EMA: {ema_yield:.4f})")
            
            if engine.get_phase() not in (EnginePhase.TESTING, EnginePhase.SUSPENDING):
                _update_ema_baselines(scale_baselines, struct_vals, current_yield, scale_best, current_top_fits)

            if study.best_value > global_best_fitness:
                global_best_fitness = study.best_value
                _save_global_checkpoint(study, struct_kwargs, struct_keys, global_best_fitness, generation, relations)
            
            engine.record_trial_results(
                current_top_fits, 
                scale_baselines[struct_vals]['top_fits'], 
                current_yield, 
                scale_baselines[struct_vals]['yield_ema'], 
                relations, 
                generation
            )
            
            engine.evaluate_epoch(generation, global_best_fitness, relations, study, active_space)