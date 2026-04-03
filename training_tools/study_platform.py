import itertools
import numpy as np
import torch
import optuna
from dataclasses import dataclass, field
import concurrent.futures

class OpCode:
    ASSIGN, LINEAR, QUADRATIC, POWER, MULTI_LINEAR, MULTI_POWER = 0, 1, 2, 3, 4, 5

class RelKey:
    SLAVE, MASTER, A, B, C, TYPE, STRIKES = "slave", "master", "a", "b", "c", "type", "strikes"
    
OP_TO_EQ = {
    OpCode.LINEAR:      "linear", 
    OpCode.QUADRATIC:   "quadratic", 
    OpCode.POWER:       "power", 
    OpCode.MULTI_LINEAR:"multi_linear", 
    OpCode.MULTI_POWER: "multi_power"
}
EQ_TO_OP = {v: k for k, v in OP_TO_EQ.items()}

class EnginePhase:
    WARMUP, IDLE, TESTING, SUSPENDING = 0, 1, 2, 3

@dataclass
class EngineState:
    phase: int = EnginePhase.WARMUP
    tested_rule: str = ""
    suspended_rule: str = ""
    suspended_law: dict = field(default_factory=dict)
    start_gen: int = 0
    start_best: float = -float('inf')
    yields: list = field(default_factory=list)
    emas: list = field(default_factory=list)
    blacklist: dict = field(default_factory=dict)

@dataclass
class ScaleMetrics:
    ema_yields: np.ndarray
    best_fitness: np.ndarray
    top_fits_history: np.ndarray  # Shape: (num_scales, max_history)
    history_counts: np.ndarray

@dataclass
class GraphVM:
    registry: dict          # Maps string_name -> int_index
    idx_to_name: list       # Maps int_index -> string_name
    bounds: torch.Tensor    # Shape: (num_vars, 2) [low, high]
    bytecode: np.ndarray    # Shape: (num_rules, 7)[OP, TGT, M1, M2, A, B, C]
    defaults: torch.Tensor  # Shape: (num_vars,)

def compile_graph(defaults: dict, bounds: dict, relations: dict, struct_keys: list) -> GraphVM:
    all_keys = list(set(defaults.keys()) | set(relations.keys()) | set(bounds.keys()) | set(struct_keys))
    registry = {k: i for i, k in enumerate(all_keys)}
    idx_to_name = all_keys
    num_vars = len(all_keys)

    # 1. Allocate default and bounds arrays
    def_arr = np.ones(num_vars, dtype=np.float32)
    bnd_arr = np.zeros((num_vars, 2), dtype=np.float32)
    bnd_arr[:, 0] = -np.inf
    bnd_arr[:, 1] = np.inf

    for k, v in defaults.items():
        def_arr[registry[k]] = v
    for k, b in bounds.items():
        idx = registry[k]
        bnd_arr[idx, 0] = b['low']
        bnd_arr[idx, 1] = b['high']

    # 2. Topological Sort (Kahn's Algorithm)
    in_degree = {k: 0 for k in all_keys}
    adj = {k:[] for k in all_keys}
    for slave, rel in relations.items():
        masters = rel[RelKey.MASTER] if isinstance(rel[RelKey.MASTER], list) else [rel[RelKey.MASTER]]
        for m in masters:
            adj[m].append(slave)
            in_degree[slave] += 1
            
    queue = [k for k in all_keys if in_degree[k] == 0]
    sorted_keys =[]
    while queue:
        node = queue.pop(0)
        sorted_keys.append(node)
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0: queue.append(neighbor)

    bytecode =[]
    for k in sorted_keys:
        if k not in relations: 
            continue
        rel = relations[k]
        op = OpCode.LINEAR
        tgt = registry[k]
        a, b, c = rel.get(RelKey.A, 1.0), rel.get(RelKey.B, 0.0), rel.get(RelKey.C, 0.0)
        
        master = rel[RelKey.MASTER]
        if isinstance(master, list):
            m1, m2 = registry[master[0]], registry[master[1]]
            bytecode.append([op, tgt, m1, m2, a, b, c])
        else:
            m1 = registry[master]
            bytecode.append([op, tgt, m1, -1, a, b, c])

    return GraphVM(
        registry=registry,
        idx_to_name=idx_to_name,
        bounds=torch.tensor(bnd_arr, dtype=torch.float32),
        bytecode=np.array(bytecode, dtype=np.float32) if bytecode else np.empty((0, 7), dtype=np.float32),
        defaults=torch.tensor(def_arr, dtype=torch.float32)
    )

def execute_vm(vm: GraphVM, batch_size: int, device: torch.device, active_tensors: dict, constants: dict) -> dict:
    #num_vars = len(vm.registry)
    
    buffer = vm.defaults.to(device).unsqueeze(1).expand(-1, batch_size).clone()
    
    for k, v in active_tensors.items(): 
        buffer[vm.registry[k]] = v
    for k, v in constants.items(): 
        buffer[vm.registry[k]] = v

    for inst in vm.bytecode:
        op, tgt, m1, m2, a, b, c = int(inst[0]), int(inst[1]), int(inst[2]), int(inst[3]), inst[4], inst[5], inst[6]
        v1 = buffer[m1]
        
        if op == OpCode.LINEAR:
            buffer[tgt] = a * v1 + b
        elif op == OpCode.QUADRATIC:
            buffer[tgt] = a * (v1**2) + b * v1 + c
        elif op == OpCode.POWER:
            buffer[tgt] = a * torch.pow(v1 + 1e-8, b)
        else:
            v2 = buffer[m2]
            if op == OpCode.MULTI_LINEAR:
                buffer[tgt] = a * v1 + b * v2 + c
            elif op == OpCode.MULTI_POWER:
                buffer[tgt] = a * torch.pow(v1 + 1e-8, b) * torch.pow(v2 + 1e-8, c)

    buffer = torch.clamp(buffer, vm.bounds[:, 0:1].to(device), vm.bounds[:, 1:2].to(device))
    return {name: buffer[i] for i, name in enumerate(vm.idx_to_name)}

def calculate_adj_r2(y_true: np.ndarray, y_pred: np.ndarray, n: int, k: int) -> float:
    if n <= k + 1:
        return 0.0
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    if ss_tot == 0:
        return 0.0
    r2 = 1.0 - (ss_res / ss_tot)
    return 1 - ((1 - r2) * (n - 1) / (n - k - 1))

def batched_score_models(A_batch: torch.Tensor, y_batch: torch.Tensor, k: int):
    B, S, K_cols = A_batch.shape
    ridge = torch.eye(K_cols, device=A_batch.device).unsqueeze(0) * 1e-6
    
    At = A_batch.transpose(1, 2)
    coefs = torch.linalg.solve(At @ A_batch + ridge, At @ y_batch)
    
    preds = torch.bmm(A_batch, coefs)
    err = torch.abs(y_batch - preds).squeeze(-1)
    
    k_thresh = max(int(S * 0.75), 1)
    thresholds, _ = torch.kthvalue(err, k_thresh, dim=1, keepdim=True)
    
    mask = (err <= thresholds).float().unsqueeze(-1)
    A_w = A_batch * mask
    y_w = y_batch * mask
    At_w = A_w.transpose(1, 2)
    
    coefs_robust = torch.linalg.solve(At_w @ A_w + ridge, At_w @ y_w)
    preds_robust = torch.bmm(A_batch, coefs_robust)
    
    # Squeeze down to 1D [B] early to prevent insidious [B, 1] * [B] -> [B, B] broadcasting bugs
    ss_res = torch.sum(mask * (y_batch - preds_robust)**2, dim=1).squeeze(-1) 
    n_in = mask.squeeze(-1).sum(dim=1)
    
    y_mean = torch.sum(mask * y_batch, dim=1).squeeze(-1) / (n_in + 1e-8)
    ss_tot = torch.sum(mask * (y_batch - y_mean.unsqueeze(1).unsqueeze(-1))**2, dim=1).squeeze(-1)
    
    r2 = 1.0 - (ss_res / (ss_tot + 1e-8))
    adj_r2 = 1.0 - ((1.0 - r2) * (n_in - 1) / (n_in - k - 1 + 1e-8))
    
    valid = (n_in > k + 1) & (ss_tot > 0)
    adj_r2 = torch.where(valid, adj_r2, torch.zeros_like(adj_r2))
    
    scores = adj_r2 * (n_in / S)
    return scores, coefs_robust.squeeze(-1)


def analyze_and_collapse_dimension(X, imp, struct, device):
    num_features = struct.shape[0]

    val_m = struct | (imp > 0.05)

    S1 = torch.arange(num_features, device=device).view(-1, 1)
    M1 = torch.arange(num_features, device=device).view(1, -1)
    mask_1d = val_m[M1] & ~struct[S1] & (imp[M1] > imp[S1])
    s_1d, m_1d = torch.nonzero(mask_1d, as_tuple=True)
    
    S2 = torch.arange(num_features, device=device).view(-1, 1, 1)
    M2_1 = torch.arange(num_features, device=device).view(1, -1, 1)
    M2_2 = torch.arange(num_features, device=device).view(1, 1, -1)
    mask_2d = val_m[M2_1] & val_m[M2_2] & ~struct[S2] & (imp[M2_1] > imp[S2]) & (imp[M2_2] > imp[S2]) & (M2_1 < M2_2)
    s_2d, m1_2d, m2_2d = torch.nonzero(mask_2d, as_tuple=True)

    best_score = 0.25
    best_op = None

    for optype in[OpCode.LINEAR, OpCode.QUADRATIC, OpCode.MULTI_LINEAR]:
        is_multi = (optype == OpCode.MULTI_LINEAR)
        s_idx = s_2d if is_multi else s_1d
        
        if len(s_idx) == 0: 
            continue
            
        m1_idx = m1_2d if is_multi else m_1d
        m2_idx = m2_2d if is_multi else None

        y = X[:, s_idx].T.unsqueeze(-1)
        x1 = X[:, m1_idx].T.unsqueeze(-1)
        ones = torch.ones_like(y)
        
        if optype == OpCode.LINEAR:
            A, k = torch.cat([x1, ones], dim=2), 1
        elif optype == OpCode.QUADRATIC:
            A, k = torch.cat([x1**2, x1, ones], dim=2), 2
        else:
            A, k = torch.cat([x1, X[:, m2_idx].T.unsqueeze(-1), ones], dim=2), 2

        sc, cf = batched_score_models(A, y, k)
        idx = torch.argmax(sc).item()
        
        if sc[idx] > best_score:
            best_score = sc[idx].item()
            c = cf[idx].cpu().tolist()
            s_i, m_i = s_idx[idx].item(), m1_idx[idx].item()
            
            match optype:
                case OpCode.LINEAR:       
                    best_op = (s_i, m_i, c[0], 0.0, c[1], optype)
                case OpCode.QUADRATIC:    
                    best_op = (s_i, m_i, c[0], c[1], c[2], optype)
                case OpCode.MULTI_LINEAR: 
                    best_op = (s_i, [m_i, m2_idx[idx].item()], c[0], c[1], c[2], optype)

    return best_score, best_op

def optuna_study_to_tensor(study, active_space, struct_keys, device):
    
    df = study.trials_dataframe()
    df = df[df['state'] == 'COMPLETE']
    if len(df) < 20:
        return None
        
    top_df = df[df['value'] >= df['value'].quantile(0.90)]
    rename_map = {c: c.replace('params_', '') for c in top_df.columns}
    top_df = top_df.rename(columns=rename_map)
    
    all_cols = list(active_space.keys()) + struct_keys
    present_cols =[c for c in all_cols if c in top_df.columns]

    try:
        imp_dict = optuna.importance.get_param_importances(study, evaluator=optuna.importance.MeanDecreaseImpurityImportanceEvaluator())
    except Exception: 
        return None
        
    X = torch.tensor(top_df[present_cols].to_numpy(), dtype=torch.float32, device=device)
    imp = torch.tensor([imp_dict.get(c, 0.0) if c not in struct_keys else 1.0 for c in present_cols], device=device)
    
    struct = torch.tensor([c in struct_keys for c in present_cols], dtype=torch.bool, device=device)
    return X, imp, struct, present_cols


def map_operation(present_cols, best_score, best_op):
    s_idx, m_idx_data, a, b, c_val, op_code = best_op

    mapped_slave = present_cols[s_idx]
    if isinstance(m_idx_data, list):
        mapped_master = [present_cols[m_idx_data[0]], present_cols[m_idx_data[1]]] 
    else:
        mapped_master = present_cols[m_idx_data]

    print(f"\n COUPLING DETECTED: {mapped_slave} bound to {mapped_master} | Score: {best_score:.3f}")
    
    return {
        RelKey.SLAVE: mapped_slave,
        RelKey.MASTER: mapped_master,
        RelKey.A: float(a), RelKey.B: float(b), RelKey.C: float(c_val),
        RelKey.TYPE: op_code, 
        RelKey.STRIKES: 0
    }

def get_scores(num_samples, best_score, m_idx, s_idx, y_arr, A_quad, coef, mask, optype):
    if mask is not None:
        n_in = np.sum(mask)
        r2 = calculate_adj_r2(y_arr[mask], A_quad[mask] @ coef, n_in, 2)
        score = r2 * (n_in / num_samples)

        if score > best_score:
            best_score, best_op = score, (s_idx, m_idx, coef[0], coef[1], coef[2], optype)
    return best_score, best_op

def update_scale_metrics(metrics: ScaleMetrics, scale_id: int, current_yield: float, best_val: float, top_fits: np.ndarray):

    ema = metrics.ema_yields[scale_id]
    metrics.ema_yields[scale_id] = current_yield if ema == 0.0 else (0.8 * ema + 0.2 * current_yield)
    metrics.best_fitness[scale_id] = max(metrics.best_fitness[scale_id], best_val)
    
    hist_count = metrics.history_counts[scale_id]
    fits_len = len(top_fits)
    start_idx = max(0, hist_count - fits_len)
    metrics.top_fits_history[scale_id, start_idx:start_idx+fits_len] = top_fits[:500]
    metrics.history_counts[scale_id] = min(hist_count + fits_len, 500)

def evaluate_epoch(state: EngineState, gen: int, global_best: float, relations: dict, study, gen_count, device):
    if gen < (gen_count * 2) or gen % gen_count != 0: 
        return state

    if state.phase in (EnginePhase.IDLE, EnginePhase.WARMUP):
        tensors = study.get_tensors()

        if tensors:
            X, imp, struct, present_cols = tensors
            analysis = analyze_and_collapse_dimension(X, imp, struct, device)
        
        if analysis:
            best_score, best_op = analysis
            proposed = map_operation(present_cols, best_score, best_op)
            state.phase, state.tested_rule = EnginePhase.TESTING, proposed[RelKey.SLAVE]
            relations[proposed[RelKey.SLAVE]] = proposed
            state.start_gen, state.start_best = gen, global_best
        elif relations:
            rule = list(relations.keys())[0]
            state.phase, state.suspended_rule, state.suspended_law = EnginePhase.SUSPENDING, rule, relations.pop(rule)
            state.start_gen, state.start_best = gen, global_best
            state.yields.clear()
            state.emas.clear()

    elif state.phase == EnginePhase.TESTING:
        if global_best > state.start_best: state.phase = EnginePhase.IDLE
        else:
            del relations[state.tested_rule]
            state.blacklist[state.tested_rule] = gen + (gen_count * 3)
            state.phase = EnginePhase.IDLE

    elif state.phase == EnginePhase.SUSPENDING:
        if global_best > state.start_best or (state.yields and np.mean(state.yields) > np.mean(state.emas) * 1.05):
            state.phase = EnginePhase.IDLE 
        else:
            relations[state.suspended_rule] = state.suspended_law
            state.phase = EnginePhase.IDLE

    return state

def run_search(task_runner, param_space, structural_space, defaults, bounds, device, study_wrapper_cls, **task_args):
    struct_keys = list(structural_space.keys())
    struct_combinations = list(itertools.product(*structural_space.values()))
    num_scales = len(struct_combinations)
    gen_count = max(num_scales, 1)

    metrics = ScaleMetrics(
        ema_yields=np.zeros(num_scales, dtype=np.float32),
        best_fitness=np.full(num_scales, -np.inf, dtype=np.float32),
        top_fits_history=np.zeros((num_scales, 500), dtype=np.float32),
        history_counts=np.zeros(num_scales, dtype=np.int32)
    )

    state = EngineState()
    relations = {} 
    vm = compile_graph(defaults, bounds, relations, struct_keys)

    generation = 0
    global_best = -float('inf')

    active_space = {k: v for k, v in param_space.items() if k not in relations and k not in state.blacklist}
    study_wrapper = study_wrapper_cls(active_space, struct_keys, device)

    while True:
        if not active_space: 
            break
        
        results_by_scale = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=gen_count) as executor:
            futures = {}
            for scale_id, struct_vals in enumerate(struct_combinations):
                active_tensors, context = study_wrapper.ask(batch_size=128)
                struct_kwargs = dict(zip(struct_keys, struct_vals))

                mapped_vars = execute_vm(vm, 128, device, active_tensors, struct_kwargs)
                
                futures[executor.submit(task_runner, mapped_vars, struct_kwargs, **task_args)] = (scale_id, context, struct_vals)
            
            for f in concurrent.futures.as_completed(futures):
                scale_id, context, struct_vals = futures[f]
                results_by_scale[scale_id] = (f.result(), context, struct_vals)

        for scale_id, (fitnesses, context, struct_vals) in results_by_scale.items():
            
            study_wrapper.update(context, fitnesses, struct_vals)
            
            fits_arr = np.sort(fitnesses)
            top_fits = fits_arr[-25:] 
            batch_best = float(top_fits[-1])
            
            if batch_best > global_best: global_best = batch_best
            
            if state.phase not in (EnginePhase.TESTING, EnginePhase.SUSPENDING):
                update_scale_metrics(metrics, scale_id, np.mean(top_fits), batch_best, top_fits)

        old_phase = state.phase
        
        state = evaluate_epoch(state, generation, global_best, relations, study_wrapper, gen_count, device)

        if state.phase != old_phase:
            vm = compile_graph(defaults, bounds, relations, struct_keys)
            active_space = {k: v for k, v in param_space.items() if k not in relations and k not in state.blacklist}
            study_wrapper = study_wrapper_cls(active_space, struct_keys, device)

        generation += 1