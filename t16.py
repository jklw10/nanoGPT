
import copy
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
torch.manual_seed(42)
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
class TriadPiece(nn.Module):
    def __init__(self, dim, batch_size, num_nodes=10, **hp_grids):
        super().__init__() # Don't forget this!
        self.D = dim
        hps = {
            'EMA_SPEED': 0.05, 'FIXED_REWARD_SENSITIVITY': 1.0, 'LR': 0.006,
            'MOMENTUM': 0.4, 'UNTANH': 0.6, 'speed': 0.5, 'width': 0.1,
            'weight_decay': 0.01, 'lateral_decay': 0.01, 'NOISE_SCALE': 0.8, 
            'G_NORM': 1.0, 'conn_count': 1500, 'c_moment': 0.99, 'w_scale': 0.27
        }
        hps.update(hp_grids)
        
        for k, v in hps.items():
            if isinstance(v, torch.Tensor):
                self.register_buffer(k, v.flatten())
            else:
                setattr(self, k, v)
                
        self.weight = nn.Parameter(torch.randn(batch_size, num_nodes, dim, dim, device=DEVICE) / math.sqrt(self.D))
        
        w_scale = self.hp('w_scale', 4)
        self.weight.copy_(utils.rms_norm(self.weight,dim=[-1,-2]) * w_scale) 

        self.register_buffer('g_ema', torch.randn_like(self.weight) * 0.01) 
        self.register_buffer('E_baseline', 0.01 + 0.01 * torch.randn(batch_size, num_nodes, dim, device=DEVICE))
        
        self.calibrate_w_scale()
    def hp(self, name, ndim):
        val = getattr(self, name)
        if isinstance(val, torch.Tensor):
            return val.view(-1, *([1] * (ndim - 1)))
        return val
    def calibrate_w_scale(self):
        untanh = self.hp('UNTANH', 3)  # shape (B, 1, 1, 1) after view
        if not isinstance(untanh,torch.Tensor):
            untanh = torch.tensor(untanh)
        bound = math.sqrt(2 + math.sqrt(2))
        gain =  ((1.0 - untanh).abs()) 
        scale = torch.where(
            untanh > 1.0,
            1.0 / (bound * gain*gain + 1e-6), 
            1.0 / (bound * gain + 1e-6)    
        ).squeeze()
        self.register_buffer('dynamic_w_scale', scale)
        
        w_scale = self.hp('w_scale', 4)
        self.weight.data.copy_(utils.rms_norm(self.weight,dim=[-1,-2])*w_scale * scale.view(-1, 1, 1, 1))
    def forward(self, x):
        untanh = self.hp('UNTANH', 3)
        raw_pred = torch.einsum('bnj,bnjk->bnk', x, self.weight)
        prediction = utils.buntanh(raw_pred, untanh, math.sqrt(2 + math.sqrt(2)))
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
    def __init__(self, dim, batch_size, num_nodes=80, **hp_grids):
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
            'w_scale': 1.0,
            'sharpness': 1.0,
            'K_connect': num_nodes/2
        }
        hps.update(hp_grids)
        
        for k, v in hps.items():
            if isinstance(v, torch.Tensor):
                self.register_buffer(k, v.flatten())
            else:
                setattr(self, k, v)
        
        self.w1 = TriadPiece(dim, batch_size, num_nodes, **hp_grids)
        self.w2 = TriadPiece(dim, batch_size, num_nodes, **hp_grids)
        self.w3 = TriadPiece(dim, batch_size, num_nodes, **hp_grids)
        
        #init state to noise
        self.register_buffer('state', 0.01*torch.randn(batch_size, self.N, dim, device=DEVICE))
        self.register_buffer('output', 0.01*torch.randn(batch_size, self.N, dim, device=DEVICE))
        self.register_buffer('target', 0.01*torch.randn(batch_size, self.N, dim, device=DEVICE))
        self.register_buffer('step_counter', torch.tensor(0.0, device=DEVICE))
        
        self.register_buffer('A', torch.randn(batch_size, self.N, self.N, device=DEVICE))
        
        self.register_buffer('A_ema', torch.zeros(batch_size, self.N, self.N, device=DEVICE))
        self.A_ema[:, 2, 0] += 3.0  
        self.A_ema[:, 2, 1] -= 2.0 
        self.A = F.softmax(self.A, dim=2)  
        self.register_buffer('node_pos',torch.randn(batch_size, self.N, 3)) 
        #self.node_pos = nn.Parameter(torch.randn(batch_size, self.N, 3)) 
    def get_A(self):
        pos = self.node_pos  # (B, N, 3)
        diff = pos.unsqueeze(2) - pos.unsqueeze(1)  # (B, N, N, 3)
        dist = diff.norm(dim=-1)  # (B, N, N, 3) -> (B, N, N)
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
        #raw_A = raw_A + torch.randn_like(raw_A)/(1.0+raw_A.var(dim=-1,keepdim=True))
        #attention_sharpness = 2.0 # (Or make this a learnable parameter!)
        #attention_sharpness = self.hp('sharpness', 3)
        raw_A = utils.rms_norm(raw_A,dim=[-1,-2]) #* attention_sharpness
        routing_momentum = self.hp('c_moment', 3)
        if self.step_counter.item() == 1.0:
            self.A_ema.copy_(raw_A)  
        else:
            self.A_ema.copy_(self.A_ema * routing_momentum + raw_A * (1.0 - routing_momentum))
        
        #raw_A = self.A_ema + 1e-5*torch.randn_like(raw_A) / (1.0+ self.A_ema.sum(dim=-1,keepdim=True))
        
        #raw_A = self.A_ema + 1e-5*torch.randn_like(self.A_ema)
       
            
        #raw_A = utils.gumbell_noise(self.A_ema,1.0)
        raw_A = self.A_ema
        self.A[:, 0:2, :] = 0.0
        #raw_A = F.softmax(raw_A, dim=[-1,-2])
        raw_A = utils.sinkhorn_knopp(raw_A)
        #K = self.hp('K_connect', 3)
        #sort_indices = torch.argsort(self.A_ema, dim=2, descending=True)
        #
        #ranks = torch.argsort(sort_indices, dim=2)
        #top_k_mask = ranks < K
        #sparse_A = torch.where(top_k_mask, self.A_ema, torch.full_like(self.A_ema, float('-inf')))
        uniform_baseline = 1.0 / (self.N*self.N+1.0)
        above_avg_mask = raw_A > uniform_baseline
        sparse_A = torch.where(above_avg_mask, raw_A, torch.tensor(float(0), device=DEVICE))
        
        K = 1 + above_avg_mask.sum(dim=[-1,-2],keepdim=True)
        #A_try = self.get_A()
        A_try = utils.sinkhorn_knopp(sparse_A)
        self.A.copy_(A_try)
        self.A[:, 0:2, :] = 0.0

        #self.A = self.A * torch.sqrt(K)
        target = torch.bmm(self.A, self.output)
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
    print(f"Initializing {batch_size}  Agents on {DEVICE}...")
    
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
        #stomach_env = torch.randn(batch_size, brain.D, device=DEVICE) * starvation * 1.0
        
        #no distance loss:
        stomach_env = torch.zeros(batch_size, brain.D, device=DEVICE) 

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
    return fitness, mean_dist, var_dist, agent_pos_history, food_pos_history, eval_start, brain, initial_snapshots

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
def diagnose_batch_health(brain,initial_brain, fitness):
    print("\n" + "="*80)
    print("🧠 RUNNING BATCH AUTOPSY & FEATURE CORRELATION")
    print("="*80)
    fitness=fitness.to(brain.w1.weight.device)
    B = brain.batch_size
    metrics = {}
    eps = 1e-7
    if initial_brain is not None:
        # Did the motor node move far from its init? (unstable = bad)
        metrics['init_motor_state_mag'] = initial_brain.state[:, 2, :].norm(dim=1)
        metrics['init_eye_state_mag'] = initial_brain.state[:, 0, :].norm(dim=1)

        # How far did weights travel from init? (big journey = chaotic)
        w1_init = initial_brain.w1.weight.view(B, -1)
        w1_final = brain.w1.weight.view(B, -1)
        metrics['w1_travel_dist'] = (w1_final - w1_init).norm(dim=1)
        metrics['w1_init_cosim_final'] = F.cosine_similarity(w1_init, w1_final, dim=1)

        # Node position geometry at init
        pos = initial_brain.node_pos
        diff = pos.unsqueeze(2) - pos.unsqueeze(1)
        dist = diff.norm(dim=-1)
        metrics['init_motor_eye_dist'] = dist[:, 2, 0]
        metrics['init_motor_stomach_dist'] = dist[:, 2, 1]
        metrics['init_min_nn_dist'] = dist.topk(2, dim=-1, largest=False).values[:,:,1].min(dim=1).values

        # Was motor node already roughly aligned with eye direction at init?
        motor_init = initial_brain.state[:, 2, :]
        eye_init = initial_brain.state[:, 0, :]
        metrics['init_motor_eye_alignment'] = F.cosine_similarity(motor_init, eye_init, dim=1)

        # Prediction quality at step 0 — did w1 start in a useful regime?
        init_pred = initial_brain.w1(initial_brain.state)
        metrics['init_prediction_mag'] = init_pred.abs().mean(dim=(1,2))
        metrics['init_prediction_variance'] = init_pred.var(dim=(1,2))
    # Does the Motor (Node 2) listen to the Eye (Node 0)?
    metrics['Motor_Listens_To_Eye'] = brain.A[:, 2, 0]
    # Does the Motor listen to the Stomach (Node 1)? (Usually causes thrashing)
    metrics['Motor_Listens_To_Stomach'] = brain.A[:, 2, 1]
    # Self-obsession: Do nodes just listen to themselves?
    metrics['Avg_Self_Attention'] = torch.diagonal(brain.A, dim1=1, dim2=2).mean(dim=1)
    
    # Routing Entropy (Are attention maps sharp or smeared into a grey paste?)
    A_norm = brain.A / (brain.A.sum(dim=-1, keepdim=True) + eps)
    metrics['Routing_Entropy'] = -(A_norm * torch.log(A_norm + eps)).sum(dim=-1).mean(dim=-1)
    metrics['Route_Variance'] = brain.A.view(B,-1).var(dim=-1)

    # ---------------------------------------------------------
    # 2. WEIGHT COLLAPSE / SYMMETRY (Are the matrices cloning each other?)
    # ---------------------------------------------------------
    w1_flat = brain.w1.weight.view(B, -1)
    w2_flat = brain.w2.weight.view(B, -1)
    w3_flat = brain.w3.weight.view(B, -1)
    
    # If W1 and W2 share the same gradients, they might just become identical.
    metrics['CosSim_W1_W2'] = F.cosine_similarity(w1_flat, w2_flat, dim=1)
    metrics['CosSim_W2_W3'] = F.cosine_similarity(w2_flat, w3_flat, dim=1)

    # ---------------------------------------------------------
    # 3. SATURATION & DEAD NEURONS (Is the signal exploding?)
    # ---------------------------------------------------------
    # Percentage of outputs that are pinned completely against the tanh ceiling
    metrics['Output_Saturation_Pct'] = (brain.output.abs()).mean(dim=(1,2))
    #ignore inputs
    metrics['Mean_State_Magnitude'] = brain.state[:, 2:, :].abs().mean(dim=(1,2))
    
    # ---------------------------------------------------------
    # 4. LEARNING / ERROR SIGNALS
    # ---------------------------------------------------------
    metrics['w1_E_Baseline_Mag'] = brain.w1.E_baseline.abs().mean(dim=(1,2))
    metrics['w2_E_Baseline_Mag'] = brain.w2.E_baseline.abs().mean(dim=(1,2))
    metrics['w3_E_Baseline_Mag'] = brain.w3.E_baseline.abs().mean(dim=(1,2))
    metrics['W1_Norm'] = w1_flat.norm(dim=1)
    metrics['W2_Norm'] = w2_flat.norm(dim=1)

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
        param_x_name='UNTANH', param_x_vals=np.linspace(0.1, 3.01, 32),
        param_y_name='w_scale', param_y_vals=np.linspace(0.1, 1.5, 32),
    )
    fitness, mean_dist, var_dist, agent_pos_hist, food_pos_hist, eval_start, brain,initial = run_tracking_task(sweep, dim=16)
    diagnose_batch_health(brain, initial, fitness)
    #diagnose_batch_health(initial,fitness)
    visualize_sweep(sweep, fitness, mean_dist, var_dist, agent_pos_hist, food_pos_hist, eval_start)
