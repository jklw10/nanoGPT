import math
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from parts import quantizer
import parts.utils as utils

USE_DIMENSIONLESS_BOREDOM = True       # True: (1 - LP/MSE)^(gamma/2), False: exp(-50 * LP) [1]
USE_ENTROPIC_RESTORING_FORCE = True    # True: Attractive drift to 0 that scales with (1 - boredom) [1]
GAMMA_COUPLING = 2.0                   # Physical coupling parameter (gamma) for both equations [1]

# ==========================================
# 1. CONFIGURATIONS & HYPERPARAMETERS
# ==========================================
DEFAULT_HYPERPARAMS = {
    'UNTANH': 0.740565,
    'w_scale': 0.622592,
    'EMA_SPEED': 0.01,
    'REW_SENS': 1.343345,
    'speed': 1.343435,
    'width': 0.100000,
    'sprs': 0.511729,
    'c_moment': 0.744751,
    'a_variance': 0.05,
    
    'opt_gd_lr': 0.01,
    'opt_es_lr': 0.03,
    'opt_beta_cov': 0.99,
    'opt_damping': 1e-5,
    'opt_decay': 1e-5,
    'opt_R': 1.0,
    'opt_gamma': 1e-5
}
TUNABLE_KEYS = [
    'UNTANH', 'w_scale', 'EMA_SPEED', 'REW_SENS', 'speed', 'width', 'sprs', 'c_moment',
    #'opt_gd_lr', 'opt_es_lr'
]

class HyperParamsConfig:
    """
    Standardized parameter dictionary managing searchable model and 
    optimizer hyperparameters across the modular training pipelines.
    """
    def __init__(self, device, init_dict=None):
        self.device = device
        self.params = {}
        source = init_dict if init_dict is not None else DEFAULT_HYPERPARAMS
        for k, v in source.items():
            self.params[k] = nn.Parameter(torch.tensor(v, dtype=torch.float32, device=device))
            
    def get(self, key):
        return self.params[key]
        
    def clone(self):
        cloned = HyperParamsConfig(self.device, init_dict={})
        for k, v in self.params.items():
            cloned.params[k] = nn.Parameter(v.clone())
        return cloned


class ExecutionConfig:
    def __init__(self):
        self.batch_size = 64
        self.num_nodes = 32
        self.dim = 16
        self.maze_size = 10
        self.max_episode_steps = 40
        
        self.gd_total_steps = 500
        self.es_sigma = 0.08
        self.es_pop_size = 12
        self.es_rank = 4
        self.es_generations = 40


# ==========================================
# 2. VECTORIZED ENVIRONMENT & KERNELS
# ==========================================
def step_kernel(agent_pos, food_grid, hunger, steps, actions, grid, size):
    dx = torch.tensor([-1, 1, 0, 0], device=agent_pos.device)
    dy = torch.tensor([0, 0, -1, 1], device=agent_pos.device)
    
    new_x = (agent_pos[:, 0] + dx[actions]).clamp(0, size - 1)
    new_y = (agent_pos[:, 1] + dy[actions]).clamp(0, size - 1)
    
    batch_size = agent_pos.shape[0]
    batch_idx = torch.arange(batch_size, device=agent_pos.device)
    
    is_wall = (grid[batch_idx, new_x, new_y] == 1.0)
    
    next_x = torch.where(is_wall, agent_pos[:, 0], new_x)
    next_y = torch.where(is_wall, agent_pos[:, 1], new_y)
    
    next_hunger = (hunger + 0.05).clamp(0.0, 1.0)
    
    food_present = food_grid[batch_idx, next_x, next_y] == 1.0
    next_hunger = torch.where(food_present, (next_hunger - 0.4).clamp(0.0, 1.0), next_hunger)
    
    next_food_grid = food_grid.clone()
    next_food_grid[batch_idx, next_x, next_y] = 0.0
    
    is_goal = (grid[batch_idx, next_x, next_y] == 2.0)
    next_hunger = torch.where(is_goal, torch.zeros_like(next_hunger), next_hunger)
    
    next_steps = steps + 1
    next_pos = torch.stack([next_x, next_y], dim=1)
    
    return next_pos, next_food_grid, next_hunger, next_steps


def raymarch_grid(agent_pos, grid, food_grid, size, max_dist=10):
    batch_size = agent_pos.shape[0]
    num_dirs = 8
    dirs = torch.tensor([
        [-1, 0], [1, 0], [0, -1], [0, 1],
        [-1, -1], [-1, 1], [1, -1], [1, 1]
    ], dtype=torch.long, device=agent_pos.device)
    
    already_hit = torch.zeros(batch_size, num_dirs, dtype=torch.bool, device=agent_pos.device)
    hit_dist = torch.full((batch_size, num_dirs), float(max_dist), device=agent_pos.device)
    hit_type = torch.zeros(batch_size, num_dirs, device=agent_pos.device)
    batch_idx = torch.arange(batch_size, device=agent_pos.device).unsqueeze(1).expand(-1, num_dirs)
    
    for step in range(1, max_dist + 1):
        cur_x = agent_pos[:, 0].unsqueeze(1) + dirs[:, 0].unsqueeze(0) * step
        cur_y = agent_pos[:, 1].unsqueeze(1) + dirs[:, 1].unsqueeze(0) * step
        out_of_bounds = (cur_x < 0) | (cur_x >= size) | (cur_y < 0) | (cur_y >= size)
        
        safe_x = cur_x.clamp(0, size - 1)
        safe_y = cur_y.clamp(0, size - 1)
        
        grid_vals = grid[batch_idx, safe_x, safe_y]
        food_vals = food_grid[batch_idx, safe_x, safe_y]
        
        is_wall = (grid_vals == 1.0) | out_of_bounds
        is_goal = (grid_vals == 2.0) & ~out_of_bounds
        is_food = (food_vals == 1.0) & ~out_of_bounds
        
        any_hit = is_wall | is_goal | is_food
        update_mask = any_hit & ~already_hit
        
        if update_mask.any():
            hit_dist = torch.where(update_mask, float(step), hit_dist)
            types = torch.zeros_like(hit_type)
            types = torch.where(is_wall, 1.0, types)
            types = torch.where(is_goal, 2.0, types)
            types = torch.where(is_food, 3.0, types)
            
            hit_type = torch.where(update_mask, types, hit_type)
            already_hit = already_hit | update_mask
            
        if already_hit.all():
            break
            
    return hit_dist / float(max_dist), hit_type / 3.0


def get_obs_kernel(agent_pos, hunger, grid, food_grid, size):
    batch_size = agent_pos.shape[0]
    batch_idx = torch.arange(batch_size, device=agent_pos.device)
    x, y = agent_pos[:, 0], agent_pos[:, 1]
    
    norm_x = x.float() / (size - 1)
    norm_y = y.float() / (size - 1)
    
    up_wall = torch.where(x == 0, torch.tensor(1.0, device=agent_pos.device), grid[batch_idx, (x - 1).clamp(0, size-1), y])
    down_wall = torch.where(x == size - 1, torch.tensor(1.0, device=agent_pos.device), grid[batch_idx, (x + 1).clamp(0, size-1), y])
    left_wall = torch.where(y == 0, torch.tensor(1.0, device=agent_pos.device), grid[batch_idx, x, (y - 1).clamp(0, size-1)])
    right_wall = torch.where(y == size - 1, torch.tensor(1.0, device=agent_pos.device), grid[batch_idx, x, (y + 1).clamp(0, size-1)])
    
    up_flag, down_flag = (up_wall == 1.0).float(), (down_wall == 1.0).float()
    left_flag, right_flag = (left_wall == 1.0).float(), (right_wall == 1.0).float()
    
    ray_dist, ray_type = raymarch_grid(agent_pos, grid, food_grid, size, max_dist=size)
    
    base_obs = torch.cat([
        norm_x.unsqueeze(1), norm_y.unsqueeze(1), 
        up_flag.unsqueeze(1), down_flag.unsqueeze(1), 
        left_flag.unsqueeze(1), right_flag.unsqueeze(1),
        ray_dist, ray_type
    ], dim=1)
    
    spatial_noise = torch.randn_like(base_obs) * (hunger * 0.20).unsqueeze(1)
    base_obs_noisy = base_obs + spatial_noise
    noise = torch.randn(batch_size, 10, device=agent_pos.device) * (hunger * 0.5).unsqueeze(1)
    
    return torch.cat([base_obs_noisy, noise], dim=1)


class MazeEnvTensor:
    _maze_pools = {}

    def __init__(self, batch_size, size=6, grid=None, food_density=0.15, pool_size=1000, device="cuda"):
        self.batch_size = batch_size
        self.size = size
        self.device = device
        self.food_density = food_density
        self.pool_size = pool_size
        
        self.grid = torch.zeros(batch_size, size, size, device=device)
        self.food_grid = torch.zeros(batch_size, size, size, device=device)
        self.agent_pos = torch.full((batch_size, 2), size // 2, dtype=torch.long, device=device)
        self.hunger = torch.full((batch_size,), 0.5, device=device)
        self.steps = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        self.last_food_eaten = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self.last_at_goal = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self.needs_reset = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        pool_key = (self.size, str(self.device))
        if pool_key not in MazeEnvTensor._maze_pools:
            pool_tensors = [self._generate_solvable_maze_cpu() for _ in range(self.pool_size)]
            MazeEnvTensor._maze_pools[pool_key] = torch.stack(pool_tensors).to(self.device)

        self.static_grid = (grid is not None)
        if self.static_grid:
            self.grid.copy_(grid)
            
        self.reset()

    def _generate_solvable_maze_cpu(self):
        start_pos = (self.size // 2, self.size // 2)
        while True:
            goal_pos = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if goal_pos == start_pos:
                continue
                
            temp_grid = torch.zeros(self.size, self.size)
            for r in range(self.size):
                for c in range(self.size):
                    if (r, c) == start_pos or (r, c) == goal_pos:
                        continue
                    if random.random() < 0.20:
                        temp_grid[r, c] = 1.0
            
            queue, visited = [start_pos], {start_pos}
            solvable = False
            while queue:
                r, c = queue.pop(0)
                if (r, c) == goal_pos:
                    solvable = True
                    break
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.size and 0 <= nc < self.size:
                        if temp_grid[nr, nc] != 1.0 and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
            if solvable:
                temp_grid[goal_pos[0], goal_pos[1]] = 2.0
                return temp_grid

    def reset(self, indices=None):
        if indices is None:
            indices = torch.arange(self.batch_size, device=self.device)
            
        self.agent_pos[indices] = self.size // 2
        self.hunger[indices] = 0.5
        self.steps[indices] = 0
        self.last_food_eaten[indices] = False
        self.last_at_goal[indices] = False
        self.needs_reset[indices] = False
        
        if not self.static_grid:
            pool = MazeEnvTensor._maze_pools[(self.size, str(self.device))]
            pool_idx = torch.randint(0, self.pool_size, (len(indices),), device=self.device)
            self.grid[indices] = pool[pool_idx]
            
        for idx in indices:
            i = idx.item()
            empty_mask = (self.grid[i] == 0.0).float()
            random_mask = (torch.rand(self.size, self.size, device=self.device) < self.food_density).float()
            new_food = empty_mask * random_mask
            new_food[self.size // 2, self.size // 2] = 0.0
            goal_idx = torch.nonzero(self.grid[i] == 2.0)
            if len(goal_idx) > 0:
                new_food[goal_idx[0, 0], goal_idx[0, 1]] = 0.0
            self.food_grid[i].copy_(new_food)
            
        return self._get_obs()

    def _get_obs(self):
        return get_obs_kernel(self.agent_pos, self.hunger, self.grid, self.food_grid, self.size)

    def step(self, actions):
        if self.needs_reset.any():
            reset_idx = torch.nonzero(self.needs_reset).squeeze(-1)
            self.reset(indices=reset_idx)
            
        batch_idx = torch.arange(self.batch_size, device=self.device)
        old_food_grid = self.food_grid.clone()
        
        self.agent_pos, self.food_grid, self.hunger, self.steps = step_kernel(
            self.agent_pos, self.food_grid, self.hunger, self.steps, actions, self.grid, self.size
        )
        
        self.last_food_eaten = (old_food_grid[batch_idx, self.agent_pos[:, 0], self.agent_pos[:, 1]] == 1.0)
        self.last_at_goal = (self.grid[batch_idx, self.agent_pos[:, 0], self.agent_pos[:, 1]] == 2.0)
        self.needs_reset = (self.steps >= 40) | self.last_at_goal
            
        return self._get_obs()

    def get_state(self):
        return (self.agent_pos.clone(), self.food_grid.clone(), self.hunger.clone(), self.steps.clone(), self.grid.clone(), self.needs_reset.clone())

    def restore_state(self, state):
        self.agent_pos.copy_(state[0])
        self.food_grid.copy_(state[1])
        self.hunger.copy_(state[2])
        self.steps.copy_(state[3])
        self.grid.copy_(state[4])
        self.needs_reset.copy_(state[5])


# ==========================================
# 3. NEURAL NETWORK ARCHITECTURE
# ==========================================
class NodeParams3(nn.Module):
    def __init__(self, num_nodes, in_dim, out_dim, device):
        super().__init__() 
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        
        base_weight = torch.randn(num_nodes, in_dim, out_dim, device=device) / math.sqrt(self.in_dim)
        self.weight = nn.Parameter(base_weight)
        self.untanh_slope = torch.tensor(0.5, device=device)
        self.w_scale = torch.tensor(0.8, device=device)

    def forward(self, x, pop_weights=None):
        if pop_weights is not None:
            w = pop_weights.repeat_interleave(x.shape[0] // pop_weights.shape[0], dim=0)
        else:
            w = self.weight.unsqueeze(0).expand(x.shape[0], -1, -1, -1)
            
        norm_weight = F.normalize(w, p=2, dim=-2) * self.w_scale
        raw_pred = torch.einsum('bni,bnio->bno', x, norm_weight)
        return utils.buntanh(raw_pred, self.untanh_slope, math.sqrt(2.0 + math.sqrt(2.0)))
    
    def apply_hyperparams(self, config):
        self.untanh_slope = config.get('UNTANH').view(-1, 1, 1)
        self.w_scale = config.get('w_scale').view(-1, 1, 1)


class StatelessNodeNet3(nn.Module):
    def __init__(self, num_nodes, dim, device, input_dim=32):
        super().__init__()
        self.N = num_nodes
        self.D = dim
        self.device = device
        self.num_input_nodes = math.ceil(input_dim / dim)
        
        self.w1 = NodeParams3(num_nodes, dim, dim * 2, device)
        self.router = NodeParams3(num_nodes, dim, num_nodes, device)
        self.wnll = utils.wnormlearnloss(dim)
        
        self.c_moment = 0.8
        self.a_variance = 0.05
        self.lp_alpha = 0.1 
        self.sprs = 0.5

    def init_states(self, batch_size):
        state = 1e-5 * torch.randn(batch_size, self.N, self.D, device=self.device)
        output = 1e-5 * torch.randn(batch_size, self.N, self.D, device=self.device)
        error_ema = torch.ones(batch_size, self.N, device=self.device)
        lp_ema = torch.zeros(batch_size, self.N, device=self.device)
        return state, output, error_ema, lp_ema

    def apply_hyperparams(self, config):
        self.sprs = config.get('sprs').view(-1, 1, 1)
        self.c_moment = config.get('c_moment').view(-1, 1, 1, 1)
        self.a_variance = config.get('a_variance').view(-1, 1, 1)
        self.lp_alpha = float(config.get('EMA_SPEED').item())
        self.w1.apply_hyperparams(config)
        self.router.apply_hyperparams(config)

    def get_action_logits(self, state, count, voters=4):
        num_out_nodes = math.ceil(count / self.D) * voters
        return state[:, -num_out_nodes:, :].sum(dim=1)[:, :count]

    def forward(self, env_input, state, output, error_ema, lp_ema, pop_weights=None):
        PB = env_input.shape[0]
        input_dim = env_input.shape[1]
        
        w1_p = pop_weights.get(self.w1.weight) if pop_weights is not None else None
        router_p = pop_weights.get(self.router.weight) if pop_weights is not None else None
        
        prediction = self.w1(state, w1_p)
        router_logits = self.router(state, router_p)
        mean, error_pred = torch.chunk(prediction, 2, dim=-1)
        
        route_logits = router_logits + F.relu(lp_ema).unsqueeze(1) * 5.0
        route, aux, mask = sparse_route_batched(route_logits, self.N, self.sprs)
        
        current_var = route.var(dim=[-1], keepdim=True)
        vd = F.relu(self.a_variance - current_var)
        base_noise = torch.sqrt(vd) * torch.randn_like(output)
        
        if USE_DIMENSIONLESS_BOREDOM:
            relative_lp = torch.clamp(F.relu(lp_ema) / (error_ema + 1e-8), min=0.0, max=1.0 - 1e-8)
            boredom_scale = torch.pow(1.0 - relative_lp, GAMMA_COUPLING / 2.0).unsqueeze(-1)
        else:
            boredom_scale = torch.exp(-50.0 * F.relu(lp_ema)).unsqueeze(-1)
            
        target = torch.matmul(route, output) + base_noise * boredom_scale
        
        if USE_ENTROPIC_RESTORING_FORCE:
            # Fix: Avoid inplace addition (+=) on tensors tracked in backward graph
            target = target - (GAMMA_COUPLING * 0.01) * state * (1.0 - boredom_scale)

        target = utils.softsign(target)
        
        pad_len = self.num_input_nodes * self.D - input_dim
        env_input_padded = F.pad(env_input, (0, pad_len)) if pad_len > 0 else env_input
        env_input_reshaped = env_input_padded.view(PB, self.num_input_nodes, self.D)
        
        if self.num_input_nodes < self.N:
            target = torch.cat([env_input_reshaped, target[:, self.num_input_nodes:, :]], dim=1)
        else:
            target = env_input_reshaped
            
        loss_pred = F.mse_loss(mean, target.detach(), reduction='none')
        loss_err_pred = F.mse_loss(error_pred, torch.abs(target - mean).detach(), reduction='none')
        loss_w1 = loss_pred + 0.1 * loss_err_pred
        
        with torch.no_grad():
            current_error = loss_w1.mean(dim=-1).detach()
            next_error_ema = error_ema * (1.0 - self.lp_alpha) + current_error * self.lp_alpha
            next_lp_ema = lp_ema * (1.0 - self.lp_alpha) + (error_ema - current_error) * self.lp_alpha
            
        total_loss = (loss_w1.sum() + aux) / PB
        
        if USE_DIMENSIONLESS_BOREDOM:
            next_relative_lp = torch.clamp(F.relu(next_lp_ema) / (next_error_ema + 1e-8), min=0.0, max=1.0 - 1e-8)
            next_boredom_scale = torch.pow(1.0 - next_relative_lp, GAMMA_COUPLING / 2.0).unsqueeze(-1)
        else:
            next_boredom_scale = torch.exp(-50.0 * torch.relu(next_lp_ema)).unsqueeze(-1)
            
        # Fix: Avoid inplace addition (+=) on target tensor
        target = target + torch.randn_like(target) * torch.sqrt(next_boredom_scale) * 0.01
        
        return total_loss, target.detach(), mean.detach(), next_error_ema, next_lp_ema, loss_pred


def sparse_route_batched(router, node_count, sprs):
    probs = torch.sigmoid(router)
    mask = quantizer.topkhotSTE(router, int(math.sqrt(node_count)))
    sparse_probs = mask * probs
    sparse_probs = sparse_probs / (sparse_probs.sum(dim=-1, keepdim=True) + 1e-8)
    
    flat_probs = torch.ones_like(probs) / node_count
    route = ((1.0 - sprs) * sparse_probs) + (sprs * flat_probs)
    return route, 0.0, mask


# ==========================================
# 4. OPTIMIZER & SPECIALIZED TRAINERS
# ==========================================

def get_rank1_approximation(G):
    m, n = G.shape
    device = G.device
    dtype = G.dtype
    
    v = torch.sum(G, dim=0, keepdim=True).T
    norm_v = torch.norm(v)
    
    v = torch.where(
        norm_v > 1e-8,
        v / (norm_v + 1e-12),
        torch.randn((n, 1), dtype=dtype, device=device) / math.sqrt(n)
    )
        
    u = torch.matmul(G, v)
    norm_u = torch.norm(u)
    u = torch.where(
        norm_u > 1e-8,
        u / (norm_u + 1e-12),
        torch.randn((m, 1), dtype=dtype, device=device) / math.sqrt(m)
    )
        
    v = torch.matmul(G.T, u)
    norm_v = torch.norm(v)
    v = torch.where(
        norm_v > 1e-8,
        v / (norm_v + 1e-12),
        torch.randn((n, 1), dtype=dtype, device=device) / math.sqrt(n)
    )
        
    return u, v



def sherman_morrison_update(A_inv, x, beta, A_inv_init, leak):
    """
    Updates the inverse matrix A_inv under a rank-1 update with diagonal leakage
    to prevent estimator wind-up and preserve positive-definiteness.
    """
    alpha = 1.0 - beta
    Ax = torch.matmul(A_inv, x)  # (d, 1)
    xAx = torch.matmul(x.T, Ax)
    
    num = alpha * torch.matmul(Ax, Ax.T)
    denom = beta + alpha * xAx
    
    # Use torch.clamp to avoid CPU-GPU sync and compiler graph breaks
    denom = torch.clamp(denom, min=1e-12)
    A_inv_new = (A_inv - num / denom) / beta
    
    # Apply Leaky RLS step to guarantee numerical stability
    return (1.0 - leak) * A_inv_new + leak * A_inv_init


class KreinKroneckerSM(torch.optim.Optimizer):
    """
    Krein Kronecker Sherman-Morrison Optimizer.
    
    Maintains dual positive-definite channels (W = W_plus - W_minus) 
    operating in an indefinite Krein space. Employs left and right Kronecker-factored 
    preconditioning matrices updated via direct online Sherman-Morrison rank-1 inversions.
    
    Modes:
      - 'sphere': Projections onto the tangent space of a sphere of radius R, 
                  applying Jordan Symmetrized Products to resolve geometric drift.
      - 'hypercube': Coordinates are restricted independently to the interval [eps_geom, R].
    """
    def __init__(self, params, lr=1e-3, beta_cov=0.99, eps=1e-8,
                 damping=1e-5, gamma=1e-3, weight_decay=1e-2, 
                 R=1.0, mode='hypercube', kernel='softsign'):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= beta_cov < 1.0:
            raise ValueError(f"Invalid beta_cov: {beta_cov}")
        if damping < 0.0:
            raise ValueError(f"Invalid damping: {damping}")
        if gamma < 0.0:
            raise ValueError(f"Invalid gamma: {gamma}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if R <= 0.0:
            raise ValueError(f"Invalid boundary scale R: {R}")
        
        mode_lower = mode.lower()
        if mode_lower not in ('sphere', 'hypercube'):
            raise ValueError(f"Unknown mode: {mode}. Choose 'sphere' or 'hypercube'.")

        defaults = dict(
            lr=lr, 
            beta_cov=beta_cov, 
            eps=eps, 
            damping=damping, 
            gamma=gamma,
            weight_decay=weight_decay, 
            R=R,
            mode=mode_lower,
            kernel=kernel,
        )
        super(KreinKroneckerSM, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta_cov = group['beta_cov']
            damping = group['damping']
            gamma = group['gamma']
            weight_decay = group['weight_decay']
            R = group['R']
            mode = group['mode']
            kernel = group['kernel']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("KreinKroneckerSM does not support sparse gradients.")

                state = self.state[p]
                eps_geom = 1e-12

                # 1. State and Dual-Channel Initialization
                orig_shape = p.shape
                is_multidim = p.dim() > 1
                
                # Treat higher-dimensional tensors (e.g., Conv2D) by flattening to 2D
                if is_multidim:
                    m = orig_shape[0]
                    n = p.numel() // m
                else:
                    m, n = p.numel(), 1

                if len(state) == 0:
                    state['step'] = 0
                    # Initialize dual channels
                    state['p_plus'] = torch.clamp(p.data, min=0.0).add_(1e-4)
                    state['p_minus'] = torch.clamp(-p.data, min=0.0).add_(1e-4)
                    
                    scale_factor = 1.0 / math.sqrt(max(damping, eps_geom))
                    
                    # Store identity priors for leakage stabilization
                    state['L_inv_init'] = scale_factor * torch.eye(m, dtype=torch.float32, device=p.device)
                    state['R_inv_init'] = scale_factor * torch.eye(n, dtype=torch.float32, device=p.device)
                    
                    for ch in ['plus', 'minus']:
                        state[f'L_inv_{ch}'] = state['L_inv_init'].clone()
                        state[f'R_inv_{ch}'] = state['R_inv_init'].clone()

                state['step'] += 1

                p_plus = state['p_plus'].to(torch.float32)
                p_minus = state['p_minus'].to(torch.float32)
                grad_fp32 = grad.to(torch.float32)

                # 2. Decoupled Weight Decay on Channels
                if is_multidim and weight_decay != 0:
                    decay_factor = 1.0 - lr * weight_decay
                    p_plus.mul_(decay_factor)
                    p_minus.mul_(decay_factor)

                # 3. Gradient Splitting
                g_plus = torch.clamp(grad_fp32, min=0.0)
                g_minus = torch.clamp(-grad_fp32, min=0.0)

                # Flatten gradients to 2D matrices for Kronecker-based preconditioning
                g_plus_2d = g_plus.view(m, n)
                g_minus_2d = g_minus.view(m, n)

                # 4. Update the Preconditioning Inverse Matrices via Sherman-Morrison
                if is_multidim:
                    leak = 1.05 * (1.0 - beta_cov)
                    
                    # Plus Channel
                    u_plus, v_plus = get_rank1_approximation(g_plus_2d)
                    state['L_inv_plus'] = sherman_morrison_update(
                        state['L_inv_plus'], u_plus, beta_cov, state['L_inv_init'], leak
                    )
                    state['R_inv_plus'] = sherman_morrison_update(
                        state['R_inv_plus'], v_plus, beta_cov, state['R_inv_init'], leak
                    )
                    
                    # Minus Channel
                    u_minus, v_minus = get_rank1_approximation(g_minus_2d)
                    state['L_inv_minus'] = sherman_morrison_update(
                        state['L_inv_minus'], u_minus, beta_cov, state['L_inv_init'], leak
                    )
                    state['R_inv_minus'] = sherman_morrison_update(
                        state['R_inv_minus'], v_minus, beta_cov, state['R_inv_init'], leak
                    )
                # 5. Independent Geometric/Entropic Updates
                channels = {
                    'plus': (p_plus, g_plus, g_plus_2d, state['L_inv_plus'], state['R_inv_plus']),
                    'minus': (p_minus, g_minus, g_minus_2d, state['L_inv_minus'], state['R_inv_minus'])
                }

                for ch_name, (p_ch, g_ch, g_ch_2d, L_inv, R_inv) in channels.items():
                    step_size = lr

                    # 1D fallback (e.g., biases)
                    if not is_multidim:
                        # Fall back to scalar update if weight dimensions cannot be factored
                        p_ch.add_(g_ch, alpha=-step_size)
                        p_ch.clamp_(min=1e-6)
                        continue

                    # --- MODE 1: HYPERSPHERE GEOMETRY ---
                    if mode == 'sphere':
                        dims = list(range(1, p.dim()))
                        r = torch.norm(p_ch, p=2, dim=dims, keepdim=True)
                        r_clamped = torch.clamp(r, max=R - eps_geom)
                        r_safe = torch.clamp(r_clamped, min=eps_geom)
                        w = p_ch / r_safe

                        # Jordan Symmetrized Term 1: H(A(g))
                        # We precondition the gradient, then project onto the sphere's tangent space
                        step_precond_2d = torch.matmul(L_inv, torch.matmul(g_ch_2d, R_inv))
                        step_precond = step_precond_2d.view_as(p_ch)
                        dot_precond = torch.sum(w * step_precond, dim=dims, keepdim=True)
                        step_nat_forward = step_precond - dot_precond * w

                        # Jordan Symmetrized Term 2: A(H(g))
                        # We project the gradient first, then apply the Kronecker preconditioning
                        dot_g = torch.sum(w * g_ch, dim=dims, keepdim=True)
                        g_projected = g_ch - dot_g * w
                        g_proj_2d = g_projected.view(m, n)
                        step_nat_reverse_2d = torch.matmul(L_inv, torch.matmul(g_proj_2d, R_inv))
                        step_nat_reverse = step_nat_reverse_2d.view_as(p_ch)

                        # Compute final natural update direction via the Jordan Symmetric Product
                        step_natural = 0.5 * (step_nat_forward + step_nat_reverse)

                        # Metric Scales and Kernel parameters
                        x_norm = r_safe / R
                        if kernel == 'softsign':
                            K_norm = x_norm / (x_norm + 1.0)
                        elif kernel == 'tanh':
                            K_norm = torch.tanh(torch.clamp(x_norm, max=20.0))
                        else:
                            raise ValueError(f"Unknown kernel: {kernel}")

                        # Entropic Restoring Force (stabilizes boundaries)
                        gamma_stable = min(gamma, 1.9 / step_size) if step_size > 0 else 0.0
                        entropic_force = gamma_stable * w * (1.0 + x_norm.square())

                        proposed_update = step_natural + entropic_force

                        # Sliced Boundary Censorship (Radial-only limits)
                        radial_step = torch.sum(w * proposed_update, dim=dims, keepdim=True)
                        tangent_step = proposed_update - radial_step * w
                        
                        max_radial_step = 0.9 * (R - r_safe)
                        scale_radial = torch.where(
                            radial_step > 0,
                            torch.clamp(max_radial_step / torch.clamp(radial_step, min=eps_geom), max=1.0),
                            torch.ones_like(radial_step)
                        )
                        final_update = (radial_step * scale_radial) * w + tangent_step

                    # --- MODE 2: HYPERCUBE GEOMETRY ---
                    else:
                        r = torch.abs(p_ch)
                        r_safe = torch.clamp(r, min=eps_geom, max=R - eps_geom)

                        x_norm = r_safe / R
                        if kernel == 'softsign':
                            K_norm = x_norm / (x_norm + 1.0)
                            K_prime = 1.0 / (x_norm + 1.0).square()
                        elif kernel == 'tanh':
                            K_norm = torch.tanh(x_norm)
                            K_prime = 1.0 - K_norm.square()
                        else:
                            raise ValueError(f"Unknown kernel: {kernel}")

                        one_minus_K2 = torch.clamp(1.0 - K_norm.square(), min=eps_geom)
                        scale = one_minus_K2.square() / (K_prime + damping * one_minus_K2.square())

                        # Precondition raw gradient with matrix-covariance operators
                        step_precond_2d = torch.matmul(L_inv, torch.matmul(g_ch_2d, R_inv))
                        step_precond = step_precond_2d.view_as(p_ch)
                        
                        # Apply coordinate-wise metric scaling
                        step_natural = step_precond * scale

                        gamma_stable = min(gamma, 1.9 / step_size) if step_size > 0 else 0.0
                        coeff_num = K_norm * K_prime * one_minus_K2
                        coeff_denom = torch.clamp(K_prime.square() + damping * one_minus_K2.square(), min=eps_geom)
                        
                        entropic_force = (gamma_stable * (coeff_num / coeff_denom)) * (p_ch / r_safe)
                        proposed_update = step_natural + entropic_force

                        # Coordinate-wise Boundary Censorship
                        max_step = 0.9 * (R - r_safe)
                        final_update = torch.clamp(proposed_update, min=-max_step, max=max_step)

                    # Apply updates to each dual component
                    p_ch.add_(final_update, alpha=-step_size)
                    p_ch.clamp_(min=1e-6)  # Rigid physical barrier for positive-definiteness

                # 6. Reconstruct and write back physical parameter
                state['p_plus'].copy_(p_plus)
                state['p_minus'].copy_(p_minus)
                p.copy_((p_plus - p_minus).to(p.dtype))

        return loss



def build_optimizer(params, hyper_cfg, lr):
    return KreinKroneckerSM(
        params,
        lr=lr,
        beta_cov=float(hyper_cfg.get('opt_beta_cov').item()),
        damping=float(hyper_cfg.get('opt_damping').item()),
        weight_decay=float(hyper_cfg.get('opt_decay').item()),
        R=float(hyper_cfg.get('opt_R').item()),
        gamma=float(hyper_cfg.get('opt_gamma').item()),
        mode='hypercube'
    )

def generate_low_rank_noise(shape, rank=4, device=None):
    if device is None:
        device = torch.device('cpu')
    if len(shape) < 2:
        return torch.randn(shape, device=device)
    if len(shape) == 2:
        d_in, d_out = shape
        A = torch.randn(d_in, rank, device=device)
        B = torch.randn(d_out, rank, device=device)
        return (A @ B.T) / math.sqrt(rank)
    elif len(shape) == 3:
        n, d_in, d_out = shape
        A = torch.randn(n, d_in, rank, device=device)
        B = torch.randn(n, d_out, rank, device=device)
        return torch.bmm(A, B.transpose(-1, -2)) / math.sqrt(rank)
    return torch.randn(shape, device=device)

def model_and_env_step(model, env, obs, state, output, error_ema, lp_ema, pop_weights=None):
    loss, next_state, next_output, next_error_ema, next_lp_ema, loss_w1 = model(
        obs, state, output, error_ema, lp_ema, pop_weights=pop_weights
    )
    logits = model.get_action_logits(next_state, 4)
    actions = torch.argmax(logits, dim=-1)
    next_obs = env.step(actions)
    return loss, next_state, next_output, next_error_ema, next_lp_ema, loss_w1, actions, next_obs

def run_rollout(model, env, num_steps, state_tuple=None, pop_weights=None, step_callback=None, collect_trajectories=False):
    if state_tuple is None:
        state, output, error_ema, lp_ema = model.init_states(env.batch_size)
    else:
        state, output, error_ema, lp_ema = state_tuple
        
    obs = env._get_obs()
    
    total_losses = []
    total_loss_w1 = []
    actions_list = []
    
    current_paths = [[] for _ in range(env.batch_size)]
    sample_grids = [env.grid[i].clone() for i in range(env.batch_size)] if collect_trajectories else None
    sample_food_grids = [env.food_grid[i].clone() for i in range(env.batch_size)] if collect_trajectories else None
    sample_paths = [[] for _ in range(env.batch_size)] if collect_trajectories else None
    has_completed = [False] * env.batch_size if collect_trajectories else None
    
    for t in range(num_steps):
        if collect_trajectories:
            pos_cpu = env.agent_pos.cpu().numpy()
            for i in range(env.batch_size):
                current_paths[i].append((int(pos_cpu[i, 0]), int(pos_cpu[i, 1])))
                
        loss, state, output, error_ema, lp_ema, loss_w1, actions, obs = model_and_env_step(
            model, env, obs, state, output, error_ema, lp_ema, pop_weights=pop_weights
        )
        
        total_losses.append(loss)
        total_loss_w1.append(loss_w1)
        actions_list.append(actions)
        
        if step_callback is not None:
            step_callback(loss, state, output, error_ema, lp_ema)
            
        if collect_trajectories:
            at_goal = env.last_at_goal
            for i in range(env.batch_size):
                if at_goal[i].item():
                    sample_grids[i] = env.grid[i].clone()
                    sample_food_grids[i] = env.food_grid[i].clone()
                    sample_paths[i] = list(current_paths[i])
                    has_completed[i] = True
                    current_paths[i] = []
                elif env.needs_reset[i].item():
                    if not has_completed[i]:
                        sample_grids[i] = env.grid[i].clone()
                        sample_food_grids[i] = env.food_grid[i].clone()
                        sample_paths[i] = list(current_paths[i])
                    current_paths[i] = []
                    
    if collect_trajectories:
        for i in range(env.batch_size):
            if len(sample_paths[i]) == 0:
                sample_paths[i] = list(current_paths[i])
                
    next_state_tuple = (state.detach(), output.detach(), error_ema.detach(), lp_ema.detach())
    
    return {
        'losses': total_losses,
        'loss_w1': total_loss_w1,
        'actions': actions_list,
        'state_tuple': next_state_tuple,
        'grids': sample_grids,
        'food_grids': sample_food_grids,
        'paths': sample_paths
    }


class GradientOptimizer:
    def __init__(self, model, exec_config, hyper_cfg, use_es_lookahead=False):
        self.model = model
        self.exec_config = exec_config
        self.hyper_cfg = hyper_cfg
        self.use_es_lookahead = use_es_lookahead
        
        self.optimizer = build_optimizer(
            model.parameters(), 
            hyper_cfg, 
            lr=float(hyper_cfg.get('opt_gd_lr').item())
        )
        self.profiler = CUDAProfiler()
        self.state_tuple = model.init_states(exec_config.batch_size)
        self.es_sigma = 0.03
        self.es_pop_size = 4
        self.es_rank = 4
        self.es_check_interval = 10

    def step(self, env, step_idx):
        self.profiler.start("GD Step")
        losses = []
        def step_callback(loss, s, o, e, l):
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            
        # Ensure state_tuple matches incoming env batch size dynamically
        if self.state_tuple is None or self.state_tuple[0].shape[0] != env.batch_size:
            self.state_tuple = self.model.init_states(env.batch_size)

        rollout_results = run_rollout(self.model, env, num_steps=1, state_tuple=self.state_tuple, step_callback=step_callback)
        self.state_tuple = rollout_results['state_tuple']
        current_loss = losses[0]
        self.profiler.stop("GD Step")
        
        if self.use_es_lookahead and step_idx % self.es_check_interval == 0 and step_idx > 0:
            self.profiler.start("Parallel Lookahead Search")
            best_fitness, best_params = self._search_lookahead(env)
            if best_params is not None:
                for p, param_val in best_params.items():
                    p.data.copy_(param_val)
                self.optimizer.state.clear()
            self.profiler.stop("Parallel Lookahead Search")
        return current_loss

    def _evaluate_rollout(self, env, P, pop_weights=None):
        B = env.batch_size
        device = env.device
        if pop_weights is not None:
            grid_tiled = env.grid[:B].repeat(P, 1, 1)
            vec_env = MazeEnvTensor(batch_size=P * B, size=env.size, grid=grid_tiled, food_density=env.food_density, device=device)
            vec_env.restore_state((
                env.agent_pos.repeat(P, 1), env.food_grid.repeat(P, 1, 1), env.hunger.repeat(P),
                env.steps.repeat(P), env.grid.repeat(P, 1, 1), env.needs_reset.repeat(P)
            ))
            eval_batch_size = P * B
        else:
            vec_env = MazeEnvTensor(batch_size=B, size=env.size, grid=env.grid, food_density=env.food_density, device=device)
            vec_env.restore_state(env.get_state())
            eval_batch_size = B
            P = 1
            
        total_food_eaten = torch.zeros(P, device=device)
        reached_goal_count = torch.zeros(P, device=device)
        total_hunger = torch.zeros(P, device=device)
        steps_evaluated = 15
        
        def eval_callback(loss, s, o, e, l):
            nonlocal total_food_eaten, reached_goal_count, total_hunger
            total_food_eaten += vec_env.last_food_eaten.view(P, B).sum(dim=1).float()
            total_hunger += vec_env.hunger.view(P, B).sum(dim=1)
            batch_idx_vec = torch.arange(eval_batch_size, device=device)
            at_goal = (vec_env.grid[batch_idx_vec, vec_env.agent_pos[:, 0], vec_env.agent_pos[:, 1]] == 2.0)
            reached_goal_count += at_goal.view(P, B).sum(dim=1).float()
            
        with torch.no_grad():
            rollout_data = run_rollout(self.model, vec_env, num_steps=steps_evaluated, pop_weights=pop_weights, step_callback=eval_callback)
        stacked_losses = torch.stack(rollout_data['loss_w1'], dim=0)
        avg_pred_loss = stacked_losses.view(steps_evaluated, P, B, self.model.N, self.model.D).mean(dim=[0, 2, 3, 4])
        avg_phys_success = ((total_food_eaten * 2.5) + (reached_goal_count * 15.0) - (total_hunger * 0.05)) / B
        return -avg_pred_loss + 0.02 * avg_phys_success

    def _search_lookahead(self, env):
        P = self.es_pop_size
        half_P = P // 2
        self.profiler.start("Setup & Noise Generation")
        master_params = {p: p.clone() for p in self.model.parameters() if p.requires_grad}
        noise_dict, pop_weights = {}, {}
        for p, param_val in master_params.items():
            noise = torch.stack([generate_low_rank_noise(param_val.shape, self.es_rank, device=param_val.device) for _ in range(half_P)], dim=0)
            noise_dict[p] = noise
            w_pos = param_val.unsqueeze(0) + self.es_sigma * noise
            w_neg = param_val.unsqueeze(0) - self.es_sigma * noise
            pop_weights[p] = torch.cat([w_pos, w_neg], dim=0)
        self.profiler.stop("Setup & Noise Generation")
        
        self.profiler.start("Rollout Loop")
        candidate_fitnesses = self._evaluate_rollout(env, P, pop_weights=pop_weights)
        baseline_fitness = self._evaluate_rollout(env, 1, pop_weights=None).item()
        self.profiler.stop("Rollout Loop")
                
        self.profiler.start("Reduce & State Restorations")
        elite_indices = torch.nonzero(candidate_fitnesses > baseline_fitness).squeeze(-1)
        if len(elite_indices) > 0:
            if len(elite_indices) == 1:
                best_idx = elite_indices[0].item()
            else:
                centers = {p: pop_weights[p][elite_indices].mean(dim=0) for p in master_params.keys()}
                num_elites = len(elite_indices)
                dists = torch.zeros(num_elites, device=self.model.device)
                for p in master_params.keys():
                    dists += ((pop_weights[p][elite_indices] - centers[p].unsqueeze(0)).view(num_elites, -1) ** 2).sum(dim=-1)
                best_idx = elite_indices[torch.argmin(dists).item()].item()
                
            best_p = {p: pop_weights[p][best_idx] for p in master_params.keys()}
            elite_fitness = candidate_fitnesses[best_idx].item()
            self.profiler.stop("Reduce & State Restorations")
            return elite_fitness, best_p
            
        self.profiler.stop("Reduce & State Restorations")
        return baseline_fitness, None


class EGGROLLSurpriseOptimizer:
    def __init__(self, model, exec_config, hyper_cfg):
        self.model = model
        self.exec_config = exec_config
        self.hyper_cfg = hyper_cfg
        self.profiler = CUDAProfiler()
        
        device = next(model.parameters()).device
        self.master_param_map = {
            p: nn.Parameter(p.clone().to(device), requires_grad=True) for p in model.parameters() if p.requires_grad
        }
        self.optimizer = build_optimizer(self.master_param_map.values(), hyper_cfg, lr=float(hyper_cfg.get('opt_es_lr').item()))

    def _apply_master_to_model(self):
        for p, master_p in self.master_param_map.items():
            p.data.copy_(master_p.data)

    def step(self, env, step_idx=0):
        # Read batch size directly from env.batch_size for dynamic environments
        P, half_P, B = self.exec_config.es_pop_size, self.exec_config.es_pop_size // 2, env.batch_size
        self.profiler.start("1. Low-Rank Noise Generation")
        noise_dict, pop_weights = {}, {}
        for p, master_p in self.master_param_map.items():
            noise = torch.stack([generate_low_rank_noise(master_p.shape, self.exec_config.es_rank, device=master_p.device) for _ in range(half_P)], dim=0)
            noise_dict[p] = noise
            w_pos = master_p.unsqueeze(0) + self.exec_config.es_sigma * noise
            w_neg = master_p.unsqueeze(0) - self.exec_config.es_sigma * noise
            pop_weights[p] = torch.cat([w_pos, w_neg], dim=0)
        self.profiler.stop("1. Low-Rank Noise Generation")
            
        self.profiler.start("2. Vectorized Environment Init")
        grid_tiled = env.grid[:B].repeat(P, 1, 1)
        vec_env = MazeEnvTensor(batch_size=P * B, size=env.size, grid=grid_tiled, food_density=env.food_density, device=env.device)
        vec_env.reset()
        self.profiler.stop("2. Vectorized Environment Init")
        
        self.profiler.start("3. Complete Rollout Loop")
        with torch.no_grad():
            rollout_data = run_rollout(self.model, vec_env, num_steps=self.exec_config.max_episode_steps, pop_weights=pop_weights)
        self.profiler.stop("3. Complete Rollout Loop")

        self.profiler.start("4. Parameter Update & Math Reduction")
        stacked_losses = torch.stack(rollout_data['loss_w1'], dim=0)
        candidate_surprises = stacked_losses.view(self.exec_config.max_episode_steps, P, B, self.model.N, self.model.D).mean(dim=[0, 2, 3, 4])
        surprises_pos, surprises_neg = candidate_surprises[:half_P], candidate_surprises[half_P:]
        mean_s, std_s = candidate_surprises.mean(), candidate_surprises.std() + 1e-8
        
        self.optimizer.zero_grad()
        for p, master_p in self.master_param_map.items():
            grad = torch.zeros_like(p)
            noise = noise_dict[p]
            for i in range(half_P):
                grad += (((surprises_pos[i] - mean_s) / std_s) - ((surprises_neg[i] - mean_s) / std_s)) * noise[i]
            master_p.grad = grad / (P * self.exec_config.es_sigma)
            
        self.optimizer.step()
        self._apply_master_to_model()
        self.profiler.stop("4. Parameter Update & Math Reduction")
        return mean_s.item()


# ==========================================
# 5. HYPERPARAMETER TUNING ENGINE
# ==========================================

class HyperParamTuner:
    """
    Modular optimization pipeline that performs separate tuning steps
    based on the exact engine dynamics of the selected training mode.
    """
    def __init__(self, device, exec_config, generations=3, pop_size=16, sigma=0.08):
        self.device = device
        self.exec_config = exec_config
        self.generations = generations
        self.pop_size = pop_size
        self.sigma = sigma
        self.base_cfg = HyperParamsConfig(device)

    def _evaluate_config_offline(self, hyper_cfg, mode):
        # Re-instantiate model per evaluation to avoid weight leakage across trials
        eval_model = StatelessNodeNet3(num_nodes=8, dim=16, device=self.device).to(self.device)
        eval_model.apply_hyperparams(hyper_cfg)
        
        # Build evaluation environment
        env = MazeEnvTensor(batch_size=16, size=self.exec_config.maze_size, food_density=0.15, device=self.device)
        env.reset()
        
        max_steps = self.exec_config.max_episode_steps
        
        if mode == 'gd':
            trainer = GradientOptimizer(eval_model, self.exec_config, hyper_cfg, use_es_lookahead=False)
            losses = []
            for step_idx in range(max_steps):
                loss = trainer.step(env, step_idx)
                losses.append(loss)
            
            # Split the run in half to measure learning progress (Epoch 0 vs Epoch 1)
            half = max_steps // 2
            mean_loss_0 = sum(losses[:half]) / half
            mean_loss_1 = sum(losses[half:]) / (max_steps - half)
            
            # Ratio objective: encourages starting with high-entropy representations 
            # (high mean_loss_0) and successfully reducing error (low mean_loss_1).
            composite_score = - (mean_loss_1 ** 2) / (mean_loss_0 + 1e-8)
            return composite_score
            
        elif mode == 'es':
            mini_exec = type('MiniExec', (), {
                'batch_size': 16, 
                'es_pop_size': 4, 
                'es_rank': 2, 
                'es_sigma': self.exec_config.es_sigma, 
                'max_episode_steps': max_steps
            })()
            trainer = EGGROLLSurpriseOptimizer(eval_model, mini_exec, hyper_cfg)
            surprises = []
            total_generations = 10
            for step in range(total_generations):  
                surprise = trainer.step(env, step)
                surprises.append(surprise)
                
            # Split generations to measure surprise reduction rate
            half = total_generations // 2
            mean_loss_0 = sum(surprises[:half]) / half
            mean_loss_1 = sum(surprises[half:]) / (total_generations - half)
            
            composite_score = - (mean_loss_1 ** 2) / (mean_loss_0 + 1e-8)
            return composite_score
            
        elif mode == 'hybrid':
            trainer = GradientOptimizer(eval_model, self.exec_config, hyper_cfg, use_es_lookahead=True)
            trainer.es_check_interval = 5
            losses = []
            for step_idx in range(max_steps):
                loss = trainer.step(env, step_idx)
                losses.append(loss)
                
            half = max_steps // 2
            mean_loss_0 = sum(losses[:half]) / half
            mean_loss_1 = sum(losses[half:]) / (max_steps - half)
            
            composite_score = - (mean_loss_1 ** 2) / (mean_loss_0 + 1e-8)
            return composite_score
            
        return -999.0

    def _clamp_param(self, k, val):
        if k in ['UNTANH', 'w_scale', 'speed', 'REW_SENS', 'opt_R']:
            return max(0.1, min(val, 3.0))
        elif k in ['sprs', 'c_moment', 'opt_beta_cov']:
            return max(0.01, min(val, 0.999))
        elif k == 'EMA_SPEED':
            return max(0.001, min(val, 0.1))
        elif k == 'width':
            return max(0.001, min(val, 0.1))
        elif k in ['opt_gd_lr', 'opt_es_lr']:
            return max(1e-4, min(val, 0.02))  
        elif k in ['opt_damping', 'opt_decay', 'opt_gamma']:
            return max(1e-6, min(val, 1e-3))  
        return val

    def tune(self, mode):
        print(f"=== Tuning Architectural & Optimizer Hyperparameters for {mode.upper()} ===")
        keys = TUNABLE_KEYS
        all_keys = list(DEFAULT_HYPERPARAMS.keys())
        all_evals = []  
        start_time = time.perf_counter()
        
        for gen in range(1, self.generations + 1):
            noises, scores = [], []
            base_score = self._evaluate_config_offline(self.base_cfg, mode)
            print(f"Gen {gen:02d} | Base Config Score: {base_score:.4f}")
            all_evals.append((base_score, {k: self.base_cfg.get(k).item() for k in all_keys}))
            
            for _ in range(self.pop_size):
                noise = {k: random.normalvariate(0, 1) for k in keys}
                noises.append(noise)
                
                # Clone parent config to retain untuned keys (like a_variance)
                perturbed_cfg = self.base_cfg.clone()
                for k in keys:
                    base_val = self.base_cfg.get(k).item()
                    val = self._clamp_param(k, base_val + self.sigma * noise[k])
                    perturbed_cfg.params[k].data.fill_(val)
                    
                score = self._evaluate_config_offline(perturbed_cfg, mode)
                scores.append(score)
                all_evals.append((score, {k: perturbed_cfg.get(k).item() for k in all_keys}))
                
            mean_score = sum(scores) / len(scores)
            std_score = math.sqrt(sum((s - mean_score)**2 for s in scores) + 1e-8)
            
            for k in keys:
                grad = 0.0
                for i in range(self.pop_size):
                    grad += ((scores[i] - mean_score) / std_score) * noises[i][k]
                new_val = self._clamp_param(k, self.base_cfg.get(k).item() + 0.05 * (grad / (self.pop_size * self.sigma)))
                self.base_cfg.params[k].data.fill_(new_val)

        all_evals.sort(key=lambda x: x[0], reverse=True)
        top_k = all_evals[:4]
        vectors = [[cfg_dict[k] for k in keys] for _, cfg_dict in top_k]
        
        if len(vectors) >= 3:
            mean_vec = [sum(v[i] for v in vectors) / len(vectors) for i in range(len(keys))]
            dists = [math.sqrt(sum((v[i] - mean_vec[i])**2 for i in range(len(keys)))) for v in vectors]
            mean_dist = sum(dists) / len(dists)
            elites = [top_k[idx][1] for idx, v in enumerate(vectors) if dists[idx] <= 1.5 * mean_dist or idx < 2]
            
            center_vec = {k: sum(e[k] for e in elites) / len(elites) for k in keys}
            best_elite = elites[0]
            min_dist_to_center = float('inf')
            for e in elites:
                dist = math.sqrt(sum((e[k] - center_vec[k])**2 for k in keys))
                if dist < min_dist_to_center:
                    min_dist_to_center, best_elite = dist, e
            centroid_params = best_elite
        else:
            centroid_params = top_k[0][1]

        elapsed = time.perf_counter() - start_time
        print(f"Tuning for {mode.upper()} completed in {elapsed:.2f} seconds!")
        
        final_cfg = HyperParamsConfig(self.device, init_dict=centroid_params)
        print(f"--- Optimized Hyperparameters Found ({mode.upper()}) ---")
        for k, v in centroid_params.items():
            print(f"  {k:15s}: {v:.6f}")
        print(f"  Best evaluation score reached: {all_evals[0][0]:.4f}")
        print("========================================\n")
        return final_cfg


# ==========================================
# 6. TRAINING COORDINATOR & EVALUATION UTILITIES
# ==========================================
class CUDAProfiler:
    def __init__(self):
        self.totals, self.counts, self.events = {}, {}, {}

    def start(self, name):
        if name not in self.events:
            self.events[name] = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        self.events[name][0].record()

    def stop(self, name):
        if name in self.events:
            self.events[name][1].record()
            torch.cuda.synchronize()
            elapsed = self.events[name][0].elapsed_time(self.events[name][1])
            self.totals[name] = self.totals.get(name, 0.0) + elapsed
            self.counts[name] = self.counts.get(name, 0) + 1

    def print_report(self, title):
        print(f"\n=== PROFILING REPORT: {title} ===")
        for name, total_ms in sorted(self.totals.items()):
            avg = total_ms / self.counts[name]
            print(f"  {name:35s} | Total: {total_ms/1000:7.3f}s | Count: {self.counts[name]:4d} | Avg: {avg:6.2f}ms")
        print("==================================\n")


def run_training(optimizer, env, total_steps, label):
    print(f"--- Training via {label.upper()} ---")
    start_time = time.perf_counter()
    env.reset()
    is_es = isinstance(optimizer, EGGROLLSurpriseOptimizer)
    losses = []
    
    for step in range(total_steps):
        loss_val = optimizer.step(env, step)
        losses.append(loss_val)
        if is_es and ((step + 1) % 10 == 0 or step == 0):
            print(f"Generation {step+1:02d} | Cumulative Surprise: {loss_val:.2f}")
                
    elapsed = time.perf_counter() - start_time
    if not is_es:
        print(f"{label.upper()} Training completed in {elapsed:.2f} seconds ({elapsed/total_steps*1000:.2f} ms/step).")
        print(f"Initial 10-step average loss: {sum(losses[:10])/10:.4f}")
        print(f"Final 10-step average loss:   {sum(losses[-10:])/10:.4f}\n")
    else:
        print(f"ES Training completed in {elapsed:.2f} seconds.\n")
    optimizer.profiler.print_report(f"{label.upper()} OPTIMIZATION BREAKDOWN")


def run_evaluation(model, env, exec_config, label):
    print(f"=== Final Evaluation for {label} ===")
    env.reset()
    total_food_eaten = torch.zeros(env.batch_size, device=env.device)
    goals_reached = torch.zeros(env.batch_size, device=env.device)
    ever_reached_goal = torch.zeros(env.batch_size, dtype=torch.bool, device=env.device)
    
    def eval_metric_callback(loss, s, o, e, l):
        nonlocal total_food_eaten, ever_reached_goal, goals_reached
        total_food_eaten += env.last_food_eaten.float()
        ever_reached_goal = ever_reached_goal | env.last_at_goal
        goals_reached += env.last_at_goal.float()
        
    with torch.no_grad():
        rollout_data = run_rollout(model, env, num_steps=exec_config.max_episode_steps, step_callback=eval_metric_callback, collect_trajectories=True)
    
    avg_food = total_food_eaten.mean().item()
    success_rate = ever_reached_goal.float().mean().item() * 100.0
    avg_goals = goals_reached.mean().item()
    
    print(f"  Avg Food Eaten per Episode:        {avg_food:.2f}")
    print(f"  Goal Success Rate (Exit Reached):  {success_rate:.1f}%")
    print(f"  Avg Goal Reaches per Episode:      {avg_goals:.2f} (Max: {int(goals_reached.max().item())}, Min: {int(goals_reached.min().item())})\n")
    
    return {
        'grids': rollout_data['grids'], 'food_grids': rollout_data['food_grids'],
        'paths': rollout_data['paths'], 'goals_reached': goals_reached.cpu().numpy()
    }


def render_trajectory_tensor(grid, food_grid, path):
    size = grid.shape[0]
    output_grid = []
    for r in range(size):
        row = []
        for c in range(size):
            if path and (r, c) == path[-1]:
                row.append("A")
            elif (r, c) in path:
                row.append(".")
            elif grid[r, c] == 1.0:
                row.append("#")
            elif grid[r, c] == 2.0:
                row.append("G")
            elif food_grid[r, c] == 1.0:
                row.append("*")
            else:
                row.append("_")
        output_grid.append(" ".join(row))
    print("\n".join(output_grid))


def select_sample_to_render(eval_results):
    return int(eval_results['goals_reached'].argmax())


# ==========================================
# 7. EXECUTION ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    torch.set_float32_matmul_precision('high')
    exec_config = ExecutionConfig()
    
    ref_env = MazeEnvTensor(batch_size=1, size=exec_config.maze_size, food_density=0.15, device=device)
    grid_batched = ref_env.grid[0].unsqueeze(0).expand(exec_config.batch_size, -1, -1)
    
    # ------------------------------------------
    # TRIAL 1: GRADIENT DESCENT (GD)
    # ------------------------------------------
    tuner_gd = HyperParamTuner(device, exec_config, generations=3, pop_size=16)
    gd_hyperparams = tuner_gd.tune(mode='gd')
    
    envs_gd = MazeEnvTensor(batch_size=exec_config.batch_size, size=exec_config.maze_size, grid=grid_batched, food_density=0.15, device=device)
    model_gd = StatelessNodeNet3(num_nodes=exec_config.num_nodes, dim=exec_config.dim, device=device).to(device)
    model_gd.apply_hyperparams(gd_hyperparams)
    
    optimizer_gd = GradientOptimizer(model_gd, exec_config, gd_hyperparams, use_es_lookahead=False)
    run_training(optimizer_gd, envs_gd, total_steps=exec_config.gd_total_steps, label="GD")
    gd_eval_results = run_evaluation(model_gd, envs_gd, exec_config, "Gradient Descent (Trial 1)")
    
    best_gd_idx = select_sample_to_render(gd_eval_results)
    print(f"Rendering Last Completed Maze for GD (Batch Element {best_gd_idx + 1}):")
    render_trajectory_tensor(gd_eval_results['grids'][best_gd_idx].cpu(), gd_eval_results['food_grids'][best_gd_idx].cpu(), gd_eval_results['paths'][best_gd_idx])
    print("\n" + "="*50 + "\n")
    
    # ------------------------------------------
    # TRIAL 2: EVOLUTION STRATEGIES (ES)
    # ------------------------------------------
    tuner_es = HyperParamTuner(device, exec_config, generations=3, pop_size=16)
    es_hyperparams = tuner_es.tune(mode='es')
    
    envs_es = MazeEnvTensor(batch_size=exec_config.batch_size, size=exec_config.maze_size, grid=grid_batched, food_density=0.15, device=device)
    model_es = StatelessNodeNet3(num_nodes=exec_config.num_nodes, dim=exec_config.dim, device=device).to(device)
    model_es.apply_hyperparams(es_hyperparams)
    
    optimizer_es = EGGROLLSurpriseOptimizer(model_es, exec_config, es_hyperparams)
    run_training(optimizer_es, envs_es, total_steps=exec_config.es_generations, label="ES")
    es_eval_results = run_evaluation(model_es, envs_es, exec_config, "EGGROLL ES (Trial 2)")
    
    best_es_idx = select_sample_to_render(es_eval_results)
    print(f"Rendering Last Completed Maze for ES (Batch Element {best_es_idx + 1}):")
    render_trajectory_tensor(es_eval_results['grids'][best_es_idx].cpu(), es_eval_results['food_grids'][best_es_idx].cpu(), es_eval_results['paths'][best_es_idx])
    print("\n" + "="*50 + "\n")
    
    # ------------------------------------------
    # TRIAL 3: HYBRID SEARCH
    # ------------------------------------------
    tuner_hybrid = HyperParamTuner(device, exec_config, generations=3, pop_size=16)
    hybrid_hyperparams = tuner_hybrid.tune(mode='hybrid')
    
    envs_hybrid = MazeEnvTensor(batch_size=exec_config.batch_size, size=exec_config.maze_size, grid=grid_batched, food_density=0.15, device=device)
    model_hybrid = StatelessNodeNet3(num_nodes=exec_config.num_nodes, dim=exec_config.dim, device=device).to(device)
    model_hybrid.apply_hyperparams(hybrid_hyperparams)
    
    optimizer_hybrid = GradientOptimizer(model_hybrid, exec_config, hybrid_hyperparams, use_es_lookahead=True)
    run_training(optimizer_hybrid, envs_hybrid, total_steps=exec_config.gd_total_steps, label="Hybrid")
    hybrid_eval_results = run_evaluation(model_hybrid, envs_hybrid, exec_config, "Interspersed Hybrid (Trial 3)")
    
    best_hybrid_idx = select_sample_to_render(hybrid_eval_results)
    print(f"Rendering Last Completed Maze for Hybrid (Batch Element {best_hybrid_idx + 1}):")
    render_trajectory_tensor(hybrid_eval_results['grids'][best_hybrid_idx].cpu(), hybrid_eval_results['food_grids'][best_hybrid_idx].cpu(), hybrid_eval_results['paths'][best_hybrid_idx])
    print()