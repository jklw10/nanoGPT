import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math

import parts.utils as utils
torch.autograd.set_detect_anomaly(True)
# ==========================================
# 0. SETUP & UTILS (Standalone)
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
# 1. QUANTIZER MODULES (From quantizer.py)
# ==========================================
class TopKHot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k):
        _, indices = torch.topk(x, k=k, dim=-1)
        k_hot = torch.zeros_like(x).scatter_(-1, indices, 1.0)
        ctx.save_for_backward(x)
        ctx.k = k
        return k_hot

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        k = ctx.k
        with torch.no_grad():
            _, topk_indices = torch.topk(-g, k=k, dim=-1)
            hard_target = torch.zeros_like(x).scatter_(-1, topk_indices, 1.0)
        grad_x = F.sigmoid(x) - hard_target
        return grad_x, None

class ThresHot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        probs = F.softmax(x, dim=-1)
        k_hot = torch.where(probs >= (1.0 / x.shape[-1]), 1.0, 0.0)
        ctx.save_for_backward(x)
        return k_hot

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        with torch.no_grad():
            g_mean = torch.mean(g, dim=-1, keepdim=True)
            hard_target = torch.where(g < g_mean, 1.0, 0.0)
        grad_x = F.sigmoid(x) - hard_target
        return grad_x    

class ThresHot2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0.0).float()

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        with torch.no_grad():
            g_mean = torch.mean(g, dim=-1, keepdim=True)
            soft_target = torch.where(g < g_mean, 1.0, 0.0)
        grad_x = F.sigmoid(x) - soft_target
        return grad_x

class ThresHot3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        probs = F.softmax(x, dim=-1)
        k_hot = torch.where(probs > (1.0 / x.shape[-1]), 1.0, 0.0)
        ctx.save_for_backward(x)
        return k_hot

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        with torch.no_grad():
            k_percentile = torch.quantile(g, 0.1, dim=-1, keepdim=True) 
            hard_target = torch.where(g < k_percentile, 1.0, 0.0)
        grad_x = F.sigmoid(x) - hard_target
        return grad_x


class sinkhot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        probs = utils.sinkhorn_knopp(x)
        k_hot = torch.where(probs > (1.0 / x.shape[-1]), probs, 0.0)
        ctx.save_for_backward(x)
        return k_hot

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        with torch.no_grad():
            k_percentile = torch.quantile(g, 0.1, dim=-1, keepdim=True) 
            hard_target = torch.where(g < k_percentile, 1.0, 0.0)
        grad_x = F.sigmoid(x) - hard_target
        return grad_x

class avgHot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        k_hot = torch.where(x > x.mean(), 1.0, 0.0)
        ctx.save_for_backward(x)
        return k_hot

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        with torch.no_grad():
            hard_target = torch.where(g < g.mean(), 1.0, 0.0)
        grad_x = F.sigmoid(x) - hard_target
        return grad_x


# ==========================================
# 2. CUSTOM LINEAR (Adapted from modules.py)
# ==========================================
class BatchedHungryLinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, config):
        ctx.save_for_backward(input, weight, bias)
        ctx.config = config
        
        # input: [B, N, D]
        # weight:[B, N, D, D]  (Batched independent weights)
        out = torch.einsum('bnj,bnjk->bnk', input, weight)
        if bias is not None:
            out = out + bias
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        conf = ctx.config
        
        if conf.get('noise', False):
            input_magnitude = input.abs().sum(dim=-1, keepdim=True)
            noise = torch.randn_like(input) / (1.0 + input_magnitude + 1e-6)
            #scale = conf.get('noise_scale', 0.1)
            #noise2 = torch.randn_like(input) * grad_output.std() * scale
            input = input + noise #+ noise2

        # grad_output: [B, N, D]
        # input: [B, N, D]
        grad_weight = torch.einsum('bnj,bnk->bnjk', input, grad_output)
        
        alpha = conf.get('lookahead_alpha', 0.001)
        gwin = grad_weight
        if conf.get('g_norm', True):
            gwin = utils.rms_norm(gwin, dim=[-1, -2])
            
        future_weight = weight - (gwin * alpha)
        grad_input = torch.einsum('bnk,bnjk->bnj', grad_output, future_weight)
        
        grad_bias = grad_output if bias is not None else None
        
        return grad_input, grad_weight, grad_bias, None


class BatchedHungryLinear(nn.Module):
    def __init__(self, dim, batch_size, num_nodes, bias=False, **hp_grids):
        super().__init__()
        self.D = dim
        self.N = num_nodes
        
        base_weight = torch.randn(1, num_nodes, dim, dim, device=DEVICE) / math.sqrt(dim)
        self.weight = nn.Parameter(base_weight.expand(batch_size, -1, -1, -1).clone())
        
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(batch_size, num_nodes, dim, device=DEVICE))
            
        self.config = {
            'lookahead_alpha': hp_grids.get('LR', 0.01),
            'noise': True,
            'noise_scale': hp_grids.get('NOISE_SCALE', 0.1),
            'g_norm': True,
        }
        
    def forward(self, x):
        return BatchedHungryLinearFunc.apply(x, self.weight, self.bias, self.config)


# ==========================================
# 3. BRAIN DEFINITION
# ==========================================
class PureTriadicBrain(nn.Module):
    def __init__(self, dim, batch_size, num_nodes=16, **hp_grids):
        super().__init__()
        self.N = num_nodes
        self.D = dim
        self.H = dim // 4
        self.head_dim = 4
        self.batch_size = batch_size
        
        hps = {
            'UNTANH': 0.6, 
            'a_variance': 0.0026,
            'K_connect': num_nodes // 2
        }
        hps.update(hp_grids)
        
        for k, v in hps.items():
            if isinstance(v, torch.Tensor):
                self.register_buffer(k, v.flatten())
            else:
                setattr(self, k, v)
        
        # Triad pieces are now just advanced batched linear layers!
        self.w1 = BatchedHungryLinear(dim, batch_size, num_nodes, **hp_grids)
        self.w2 = BatchedHungryLinear(dim, batch_size, num_nodes, **hp_grids)
        self.w3 = BatchedHungryLinear(dim, batch_size, num_nodes, **hp_grids)
        
        # Recurrent Buffers (Detached from Autograd graphs to prevent memory leaks)
        self.register_buffer('state', 1e-5 * torch.randn(batch_size, self.N, dim, device=DEVICE))
        self.register_buffer('output', 1e-5 * torch.randn(batch_size, self.N, dim, device=DEVICE))
        
        self.register_buffer('A', torch.zeros(batch_size, self.H, self.N, self.N, device=DEVICE))
         
    def hp(self, name, ndim):
        val = getattr(self, name)
        if isinstance(val, torch.Tensor):
            return val.view(-1, *([1] * (ndim - 1)))
        return val

    def forward(self, env_input):
        batch_size, input_dim = env_input.shape
        num_input_nodes = math.ceil(input_dim / self.D)
        untanh = self.hp('UNTANH', 3)
        state = self.state.clone()
        prediction = utils.buntanh(self.w1(state), untanh, math.sqrt(2 + math.sqrt(2)))
        keys       = utils.buntanh(self.w2(state), untanh, math.sqrt(2 + math.sqrt(2)))
        queries    = utils.buntanh(self.w3(state), untanh, math.sqrt(2 + math.sqrt(2)))

        Q = queries.view(batch_size, self.N, self.H, self.head_dim).transpose(1, 2)
        K = keys.view(batch_size, self.N, self.H, self.head_dim).transpose(1, 2)
        V = prediction.view(batch_size, self.N, self.H, self.head_dim).transpose(1, 2)

        raw_A = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        raw_A = utils.rms_norm(raw_A, dim=[-1, -2])
        
        A = sinkhot.apply(raw_A) 
        with torch.no_grad():
            self.A.copy_(A.detach().clone()) 

        routed_V = torch.matmul(A, V) 
        routed_V = routed_V.transpose(1, 2).reshape(batch_size, self.N, self.D)

        pad_len = num_input_nodes * self.D - input_dim
        env_input_padded = F.pad(env_input, (0, pad_len)).view(batch_size, num_input_nodes, self.D)
        
        
        loss_env = F.mse_loss(routed_V[:, :num_input_nodes, :], env_input_padded)
        
        loss_memory = F.mse_loss(routed_V[:, num_input_nodes:, :], state[:, num_input_nodes:, :].detach())
        
        total_loss = loss_env + (0.5 * loss_memory)

        with torch.no_grad():
            new_state = routed_V.clone().detach()
            new_state[:, :num_input_nodes, :] = env_input_padded 
            self.output.copy_(routed_V.detach().clone())
            self.state.copy_(new_state.detach().clone())

        return total_loss

# ==========================================
# 4. TRAINING LOOP
# ==========================================
def run_tracking_task(sweep, dim=16):
    batch_size = sweep.batch_size
    print(f"Initializing {batch_size} Agents on {DEVICE}...")
    
    brain = PureTriadicBrain(dim=dim, batch_size=batch_size, **sweep.get_agent_kwargs()).to(DEVICE)
    # Using standard AdamW since PyTorch now fully controls the graph
    optimizer = torch.optim.AdamW(brain.parameters(), lr=0.005, weight_decay=0.01)
    
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
        
        starvation = dists 
        environment = torch.randn(batch_size, 2, device=DEVICE) * starvation * 1e-5
        environment[:, 0:2] = direction
        
        # ---------------------------------------------
        # STANDARD PYTORCH BACKPROP CYCLE
        # ---------------------------------------------
        optimizer.zero_grad()
        loss = brain(environment) # Forward pass applies quantizers and routing
        loss.backward()           # Straight-Through Estimators route gradient correctly
        optimizer.step()          # Adam updates weights, HungryLinear applies lookahead!
        
        motor_out = brain.output[:, 2, 0:2].clone()
        agent_vel = (agent_vel * 0.85) + (motor_out * 0.2)
        agent_pos += agent_vel
        agent_pos = torch.clamp(agent_pos, -4.0, 4.0)
        
        agent_pos_history[step] = agent_pos
        food_pos_history[step] = food_pos[0]
        
        if step >= eval_start:
            dist_history[step - eval_start] = dists.squeeze()
            
    mean_dist = dist_history.mean(dim=0).cpu()
    var_dist = dist_history.var(dim=0).cpu()
    fitness = torch.exp(-(mean_dist + var_dist))
    return fitness, mean_dist, var_dist, agent_pos_history, food_pos_history, eval_start, brain, initial_snapshots

# ==========================================
# 5. VISUALIZATION
# ==========================================
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


if __name__ == "__main__":
    
    sweep = Sweep2D(
        param_x_name='UNTANH', param_x_vals=np.linspace(0.1, 1.7, 8),
        param_y_name='NOISE_SCALE', param_y_vals=np.linspace(0.01, 0.5, 8),
    )
    
    fitness, mean_dist, var_dist, agent_pos_hist, food_pos_hist, eval_start, brain, initial = run_tracking_task(sweep, dim=16)
    
    visualize_sweep(sweep, fitness, mean_dist, var_dist, agent_pos_hist, food_pos_hist, eval_start)