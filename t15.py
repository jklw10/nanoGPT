
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
import optuna
import utils

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
        return {self.p_x_name: self.grid_x_flat, self.p_y_name: self.grid_y_flat}
        
    def get_params(self, flat_idx):
        w_idx = flat_idx // self.grid_x
        s_idx = flat_idx % self.grid_x
        return self.vals_x[s_idx].item(), self.vals_y[w_idx].item()
        
    def to_2d_grid(self, flat_tensor):
        return flat_tensor.view(self.grid_y, self.grid_x)
        
    def get_extent(self):
        return[self.vals_x.min().item(), self.vals_x.max().item(),
                self.vals_y.min().item(), self.vals_y.max().item()]

# ==========================================
# 1. BRAIN DEFINITION
# ==========================================
class PureTriadicBrain(nn.Module):
    def __init__(self, dim, batch_size, num_nodes=10, **hp_grids):
        super().__init__()
        self.N = num_nodes
        self.D = dim
        self.batch_size = batch_size
        
        hps = {
            'EMA_SPEED': 0.05,
            'FIXED_REWARD_SENSITIVITY': 1.0,
            'LR': 0.006,
            'MOMENTUM': 0.4,
            'UNTANH': 0.6, 
            'speed': 0.5,
            'width': 0.1,
            'weight_decay': 0.01,
            'lateral_decay': 0.01,
            'NOISE_SCALE': 0.8, # HungryLinear config
            'G_NORM': 1.0,       # 1.0 = True, 0.0 = False
            'conn_count': 1500,
            'c_moment': 0.99,
            'w_scale': 0.27
        }
        hps.update(hp_grids)
        
        for k, v in hps.items():
            if isinstance(v, torch.Tensor):
                self.register_buffer(k, v.flatten())
            else:
                setattr(self, k, v)
        
        w_scale = self.hp('w_scale', 4)
        self.W1 = nn.Parameter(w_scale*torch.randn(batch_size, self.N, dim, dim, device=DEVICE) / math.sqrt(self.D))
        self.W2 = nn.Parameter(w_scale*torch.randn(batch_size, self.N, dim, dim, device=DEVICE) / math.sqrt(self.D))
        self.W3 = nn.Parameter(w_scale*torch.randn(batch_size, self.N, dim, dim, device=DEVICE) / math.sqrt(self.D))
        
        self.W1.copy_(utils.rms_norm(self.W1)*w_scale)
        self.W2.copy_(utils.rms_norm(self.W2)*w_scale)
        self.W3.copy_(utils.rms_norm(self.W3)*w_scale)

        self.register_buffer('M1', torch.randn_like(self.W1)*0.01)
        self.register_buffer('M2', torch.randn_like(self.W2)*0.01)
        self.register_buffer('M3', torch.randn_like(self.W3)*0.01)
        
        self.register_buffer('E_baseline', 0.01+ 0.01*torch.randn(batch_size, self.N, dim, device=DEVICE))
        #init state to noise
        self.register_buffer('state', 0.01*torch.randn(batch_size, self.N, dim, device=DEVICE))
        self.register_buffer('output', 0.01*torch.randn(batch_size, self.N, dim, device=DEVICE))
        self.register_buffer('target', 0.01*torch.randn(batch_size, self.N, dim, device=DEVICE))
        self.register_buffer('step_counter', torch.tensor(0.0, device=DEVICE))
        
        self.register_buffer('A', torch.randn(batch_size, self.N, self.N, device=DEVICE))
        
        self.register_buffer('A_ema', torch.zeros(batch_size, self.N, self.N, device=DEVICE))
        
        conn_count = self.hp('conn_count', 1)
        #for b in range(batch_size):
        #    if isinstance(conn_count, torch.Tensor):
        #        cc = conn_count[b].int()
        #    else:
        #        cc = conn_count
        #    for _ in range(cc):
        #        src = torch.randint(0, self.N, (1,)).item()
        #        dst = torch.randint(2, self.N, (1,)).item() 
        #        self.A[b, dst, src] = 1.0
        self.A = F.softmax(self.A, dim=2)  
        #self.A = self.A / (self.A.sum(dim=2, keepdim=True) + 1e-5)

    def hp(self, name, ndim):
        val = getattr(self, name)
        if isinstance(val, torch.Tensor):
            return val.view(-1, *([1] * (ndim - 1)))
        return val

    def step(self, eye_env, stomach_env):
        self.step_counter += 1.0
        # 1. Project state through W3 and W2 to get Queries and Keys
        # state: [B, N, D], W: [B, N, D, D] -> Output:[B, N, D]
        queries = torch.einsum('bnj,bnjk->bnk', self.state, self.W3) 
        keys = torch.einsum('bnj,bnjk->bnk', self.state, self.W2)    

        # Compute raw affinities [B, N, N]
        raw_A = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(self.D)
        #raw_A = raw_A / math.sqrt(self.D)
        #exploration_temp = max(1.0, 3.0 - (self.step_counter.item() / 250.0))
        raw_A = raw_A + torch.randn_like(raw_A) * 1e-5
        #raw_A = utils.gumbell_noise(raw_A, 1.0)
        #P_dense = F.softmax(raw_A, dim=2)
        #uniform_baseline = 1.0 / (self.N+1.0)
        #above_avg_mask = P_dense > uniform_baseline
        #sparse_logits = torch.where(above_avg_mask, raw_A, torch.tensor(float('-inf'), device=DEVICE))
        
        routing_momentum = self.hp('c_moment', 3)
        if self.step_counter.item() == 1.0:
            self.A_ema.copy_(raw_A)  
        else:
            self.A_ema.copy_(self.A_ema * routing_momentum + raw_A * (1.0 - routing_momentum))

        #topk_vals, topk_indices = torch.topk(self.A_ema, k=5, dim=2)
        #sparse_A = torch.full_like(self.A_ema, float('-inf'))
        #sparse_A.scatter_(2, topk_indices, topk_vals)
        sparse_A = self.A_ema
        #sparse_A.clamp_(min=1e-8)
        #for _ in range(5): 
        #    dir = 0.5*(sparse_A.sum(dim=1, keepdim=True)+sparse_A.sum(dim=0, keepdim=True))
        #    sparse_A.div_(dir)
        P = F.softmax(sparse_A, dim=2)
        self.A.copy_(self.A * routing_momentum + P * (1.0 - routing_momentum))

        # 3. Protect your sensory nodes (Eyes and Stomach shouldn't receive signals)
        # Sinkhorn makes them receive 1.0, so we forcefully zero them out after.
        self.A[:, 0:2, :] = 0.0

        # 4. Route the signals
        total_in = torch.bmm(self.A, self.output)
        
        self.target = total_in 

        self.target[:, 0, :] = eye_env
        self.target[:, 1, :] = stomach_env

        raw_pred = torch.einsum('bnj,bnjk->bnk', self.state, self.W1)
        
        untanh = self.hp('UNTANH', 3)
        raw = raw_pred - torch.tanh(raw_pred) * untanh
        # Capped slightly lower using the dynamic bounds you discovered
        prediction = torch.tanh(raw) * math.sqrt(2 + math.sqrt(2))
        
        error_signal = prediction - self.target
        
        # Softmax feature-level fidelity
        E_curr = F.softmax(error_signal, dim=-1)
        
        baseline_mask = (self.E_baseline == 0).float()
        self.E_baseline.copy_(self.E_baseline * (1 - baseline_mask) + E_curr * baseline_mask)
        
        ema_speed = self.hp('EMA_SPEED', 3)
        self.E_baseline.mul_(1.0 - ema_speed).add_(ema_speed * E_curr)
        
        advantage = E_curr - self.E_baseline
        
        rew_sens = self.hp('FIXED_REWARD_SENSITIVITY', 4)
        R = -utils.rms_norm(advantage.unsqueeze(3) * rew_sens) 
        # 1. Base error from the environment
        state_mag = self.state.abs().sum(dim=[-2,-1], keepdim=True) 
        n_scale = self.hp('NOISE_SCALE', 3)
        
        def get_noisy_state():
            noise1 = torch.randn_like(self.state) / (1.0 + state_mag)
            noise2 = torch.randn_like(self.state) * error_signal.std() * n_scale
            return self.state + noise1 + noise2
        
        g_norm_flag = self.hp('G_NORM', 4)
        def apply_gnorm(lg):
            if isinstance(g_norm_flag, torch.Tensor):
                gn_mask = (g_norm_flag > 0.5).float()
                rms_scale = torch.rsqrt(lg.pow(2).mean(dim=(2,3), keepdim=True) + 1e-6)
                return lg * (rms_scale*gn_mask + 1-gn_mask)
            else:
                return utils.rms_norm(lg, dim=(2,3)) if g_norm_flag > 0.5 else lg
        
        wd = self.hp('weight_decay', 4)
        lat_decay = self.hp('lateral_decay', 4)
        plasticity = 1.0 - R  
        speed = self.hp('speed', 4)
        width = self.hp('width', 4)
        
        indices = torch.arange(self.D, device=DEVICE, dtype=torch.float32).view(1, 1, 1, self.D)
        center = (self.step_counter * speed) % self.D
        diff = torch.abs(indices - center)
        diff = torch.minimum(diff, self.D - diff)
        g_mask = torch.exp(-(diff ** 2) / (2 * width ** 2 + 1e-6))
        
        lg1 = -torch.einsum('bni,bnj->bnij', error_signal, get_noisy_state())
        #lg2 = -torch.einsum('bni,bnj->bnij', error_signal, get_noisy_state())
        #lg3 = -torch.einsum('bni,bnj->bnij', error_signal, get_noisy_state())
        
        lg1 = apply_gnorm(lg1)
        #lg2 = apply_gnorm(lg2)
        #lg3 = apply_gnorm(lg3)
        
        grad1 = (plasticity * lg1) + (lat_decay * self.W3) - (wd * self.W1)
        adv_expanded = advantage.unsqueeze(3)

        w3_drift = torch.randn_like(self.W3) * F.relu(adv_expanded) 
        w2_drift = torch.randn_like(self.W2) * F.relu(adv_expanded)

        grad2 = w2_drift + (lat_decay * self.W2) - (wd * self.W2)
        grad3 = w3_drift + (lat_decay * self.W3) - (wd * self.W3)

        grad1 = grad1 * g_mask
        grad2 = grad2 * g_mask
        grad3 = grad3 * g_mask

        momentum = self.hp('MOMENTUM', 4)
        lr = self.hp('LR', 4)

        self.M1.mul_(momentum).add_((1.0 - momentum) * grad1)
        self.M2.mul_(momentum).add_((1.0 - momentum) * grad2)
        self.M3.mul_(momentum).add_((1.0 - momentum) * grad3)
        
        self.W1.add_(lr * self.M1)
        self.W2.add_(lr * self.M2) 
        self.W3.add_(lr * self.M3)
        
        w_scale = self.hp('w_scale', 4)
        self.W1.copy_(utils.rms_norm(self.W1)*w_scale)
        self.W2.copy_(utils.rms_norm(self.W2)*w_scale)
        self.W3.copy_(utils.rms_norm(self.W3)*w_scale)
        
        self.output.copy_(prediction)
        self.state.copy_(self.target*0.95)

# ==========================================
# 2. RUNNER & RENDERER
# ==========================================


def run_tracking_task(sweep, dim=16):
    batch_size = sweep.batch_size
    print(f"Initializing {batch_size}  Agents on {DEVICE}...")
    
    brain = PureTriadicBrain(dim=dim, batch_size=batch_size, **sweep.get_agent_kwargs()).to(DEVICE)
    brain = torch.compile(brain)
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
        
        starvation = torch.clamp(dists / 4.0, 0.0, 1.0)
        stomach_env = torch.randn(batch_size, brain.D, device=DEVICE) * starvation * 2.0

        brain.step(eye_env, stomach_env)
        
        motor_out = brain.output[:, 2, 0:2].clone()
        agent_vel = (agent_vel * 0.85) + (motor_out * 0.2)
        agent_pos += agent_vel
        agent_pos = torch.clamp(agent_pos, -4.0, 4.0)
        
        agent_pos_history[step] = agent_pos
        food_pos_history[step] = food_pos[0]
        
        if step >= eval_start:
            dist_history[step - eval_start] = dists.squeeze()
    #print(analyze_brains(brain))
    mean_dist = dist_history.mean(dim=0).cpu()
    var_dist = dist_history.var(dim=0).cpu()
    fitness = torch.exp(-(mean_dist + var_dist))
    return fitness, mean_dist, var_dist, agent_pos_history, food_pos_history, eval_start, brain

def visualize_sweep(sweep, fitness, mean_dist, var_dist, agent_pos_history, food_pos_history, eval_start):
    perf_grid = sweep.to_2d_grid(fitness).numpy()
    best_idx = torch.argmax(fitness).item()
    
    threshold = torch.quantile(fitness, 0.99)
    top_indices = torch.nonzero(fitness >= threshold).squeeze()
    if top_indices.dim() == 0: top_indices = top_indices.unsqueeze(0)
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
def diagnose_batch_health(brain, fitness):
    print("\n" + "="*80)
    print("🧠 RUNNING BATCH AUTOPSY & FEATURE CORRELATION")
    print("="*80)
    fitness=fitness.to(brain.W1.device)
    B = brain.batch_size
    metrics = {}
    eps = 1e-7
    
    # ---------------------------------------------------------
    # 1. ROUTING TOPOLOGY (Did it actually build the right brain?)
    # ---------------------------------------------------------
    # Does the Motor (Node 2) listen to the Eye (Node 0)?
    metrics['Motor_Listens_To_Eye'] = brain.A[:, 2, 0]
    # Does the Motor listen to the Stomach (Node 1)? (Usually causes thrashing)
    metrics['Motor_Listens_To_Stomach'] = brain.A[:, 2, 1]
    # Self-obsession: Do nodes just listen to themselves?
    metrics['Avg_Self_Attention'] = torch.diagonal(brain.A, dim1=1, dim2=2).mean(dim=1)
    
    # Routing Entropy (Are attention maps sharp or smeared into a grey paste?)
    A_norm = brain.A / (brain.A.sum(dim=-1, keepdim=True) + eps)
    metrics['Routing_Entropy'] = -(A_norm * torch.log(A_norm + eps)).sum(dim=-1).mean(dim=-1)

    # ---------------------------------------------------------
    # 2. WEIGHT COLLAPSE / SYMMETRY (Are the matrices cloning each other?)
    # ---------------------------------------------------------
    w1_flat = brain.W1.view(B, -1)
    w2_flat = brain.W2.view(B, -1)
    w3_flat = brain.W3.view(B, -1)
    
    # If W1 and W2 share the same gradients, they might just become identical.
    metrics['CosSim_W1_W2'] = F.cosine_similarity(w1_flat, w2_flat, dim=1)
    metrics['CosSim_W2_W3'] = F.cosine_similarity(w2_flat, w3_flat, dim=1)

    # ---------------------------------------------------------
    # 3. SATURATION & DEAD NEURONS (Is the signal exploding?)
    # ---------------------------------------------------------
    limit = math.sqrt(2 + math.sqrt(2))
    # Percentage of outputs that are pinned completely against the tanh ceiling
    metrics['Output_Saturation_Pct'] = (brain.output.abs()).mean(dim=(1,2))
    #ignore inputs
    metrics['Mean_State_Magnitude'] = brain.state[:, 2:, :].abs().mean(dim=(1,2))
    
    # ---------------------------------------------------------
    # 4. LEARNING / ERROR SIGNALS
    # ---------------------------------------------------------
    metrics['E_Baseline_Mag'] = brain.E_baseline.abs().mean(dim=(1,2))
    metrics['W1_Norm'] = brain.W1.view(B, -1).norm(dim=1)
    metrics['W2_Norm'] = brain.W2.view(B, -1).norm(dim=1)

    # =========================================================
    # CORRELATION ENGINE
    # =========================================================
    threshold = torch.quantile(fitness, 0.95)
    top_mask = fitness >= threshold
    bot_mask = fitness < threshold
    
    results =[]
    for name, vals in metrics.items():
        # Pearson Correlation: cov(x,y) / (std(x)*std(y))
        vx = vals - vals.mean()
        vy = fitness - fitness.mean()
        corr = (vx * vy).sum() / (vx.norm() * vy.norm() + eps)
        
        top_mean = vals[top_mask].mean().item()
        bot_mean = vals[bot_mask].mean().item()
        
        results.append((corr.item(), name, top_mean, bot_mean))
        
    # Sort by absolute correlation strength
    results.sort(key=lambda x: abs(x[0]), reverse=True)
    
    print(f"{'Metric Name':<28} | {'Corr w/ Fitness':<17} | {'Top 5% Mean':<12} | {'Bottom 95% Mean':<12}")
    print("-" * 78)
    for corr, name, top_mean, bot_mean in results:
        # Add visual markers for strong correlations
        marker = "🔥" if abs(corr) > 0.4 else "  "
        print(f"{marker} {name:<25} | {corr:>15.4f}   | {top_mean:>12.4f} | {bot_mean:>12.4f}")
        
    return metrics
# ==========================================
# 3. EXECUTION SCRIPT
# ==========================================
if __name__ == "__main__":
    torch.set_grad_enabled(False) 
    #hps = {
    #        'EMA_SPEED': 0.05,
    #        'FIXED_REWARD_SENSITIVITY': 1.0,
    #        'LR': 2e-3,
    #        'MOMENTUM': 0.4,
    #        'UNTANH': utils.phi, # Phi anchored!
    #        'speed': 0.5,
    #        'width': 0.1,
    #        'weight_decay': 0.01,
    #        'lateral_decay': 0.01,
    #        'NOISE_SCALE': 0.3, # HungryLinear config
    #        'G_NORM': 0.0,       # 1.0 = True, 0.0 = False
    #        'conn_count': 1500,
    #        'c_moment': 0.99,
    #         w_scale'
    #    }
    sweep = Sweep2D(
        param_x_name='LR', param_x_vals=np.linspace(0.001, 0.01, 32),
        param_y_name='untanh', param_y_vals=np.linspace(0.3, 1.8, 32),
    )
    #fitness, mean_dist, var_dist, agent_pos_hist, food_pos_hist, eval_start = run_tracking_task(sweep, dim=16)
    #visualize_sweep(sweep, fitness, mean_dist, var_dist, agent_pos_hist, food_pos_hist, eval_start)
#
    #sweep = Sweep2D(
    #    param_x_name='LR', param_x_vals=np.linspace(0.0005, 1.0, 32),
    #    param_y_name='w_scale', param_y_vals=np.linspace(0.1, 2.0, 32),
    #)
    #fitness, mean_dist, var_dist, agent_pos_hist, food_pos_hist, eval_start = run_tracking_task(sweep, dim=16)
    #visualize_sweep(sweep, fitness, mean_dist, var_dist, agent_pos_hist, food_pos_hist, eval_start)

    #sweep = Sweep2D(
    #    param_x_name='LR', param_x_vals=np.linspace(0.00005, 1.0, 32),
    #    param_y_name='UNTANH', param_y_vals=np.linspace(0.1, 2.0, 32),
    #)
    fitness, mean_dist, var_dist, agent_pos_hist, food_pos_hist, eval_start, brain = run_tracking_task(sweep, dim=16)
    diagnose_batch_health(brain,fitness)
    visualize_sweep(sweep, fitness, mean_dist, var_dist, agent_pos_hist, food_pos_hist, eval_start)
