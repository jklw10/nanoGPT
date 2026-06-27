import math
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from parts import quantizer
import parts.utils as utils

# ==========================================
# MATHEMATICAL OPTIMIZATION TOGGLES
# ==========================================
USE_DIMENSIONLESS_BOREDOM = True       # True: (1 - LP/MSE)^(gamma/2), False: exp(-50 * LP) [1]
USE_ENTROPIC_RESTORING_FORCE = True    # True: Attractive drift to 0 that scales with (1 - boredom) [1]
GAMMA_COUPLING = 2.0                   # Physical coupling parameter (gamma) for both equations [1]


# ==========================================
# CUDA PROFILING UTILITY
# ==========================================
class CUDAProfiler:
    def __init__(self):
        self.totals = {}
        self.counts = {}
        self.events = {}

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


# ==========================================
# CONFIGURATIONS
# ==========================================
class HyperParamsConfig:
    def __init__(self, device):
        self.device = device
        self.param_count = 10
        self.cfg = type('Config', (), {})()
        self.cfg.device = self.device
        self.cfg.UNTANH         = nn.Parameter(torch.tensor(0.740565, device=self.device))
        self.cfg.w_scale        = nn.Parameter(torch.tensor(0.622592, device=self.device))
        self.cfg.EMA_SPEED      = nn.Parameter(torch.tensor(0.01    , device=self.device))
        self.cfg.REW_SENS       = nn.Parameter(torch.tensor(1.343345, device=self.device))
        self.cfg.speed          = nn.Parameter(torch.tensor(1.343435, device=self.device))
        self.cfg.width          = nn.Parameter(torch.tensor(0.100000, device=self.device))
        self.cfg.sprs           = nn.Parameter(torch.tensor(0.511729, device=self.device))
        self.cfg.c_moment       = nn.Parameter(torch.tensor(0.744751, device=self.device))
        self.cfg.a_variance     = nn.Parameter(torch.tensor(0.05    , device=self.device))

    def to_tensor_config(self):
        return self.cfg


class ExecutionConfig:
    def __init__(self):
        self.batch_size = 64
        self.num_nodes = 32
        self.dim = 16
        
        self.maze_size = 10
        self.max_episode_steps = 40
        
        self.gd_lr = 0.001
        self.gd_total_steps = 500
        
        self.es_lr = 0.03
        self.es_sigma = 0.08
        self.es_pop_size = 12
        self.es_rank = 4
        self.es_generations = 40


def sparse_route_batched(router, node_count, spr, step = 0):
    probs = torch.sigmoid(router)
    mask = quantizer.topkhotSTE(router, int(math.sqrt(node_count)))
    sparse_probs = mask * probs
    
    sparse_probs = sparse_probs / (sparse_probs.sum(dim=-1, keepdim=True) + 1e-8)
    
    flat_probs = torch.ones_like(probs) / node_count
    ratio = spr 

    route = ((1.0-ratio) * sparse_probs) + (ratio * flat_probs)
    
    return route, 0.0, mask


# ==========================================
# VECTORIZED ENVIRONMENT TRANSITIONS
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
            
    norm_hit_dist = hit_dist / float(max_dist)
    norm_hit_type = hit_type / 3.0
    
    return norm_hit_dist, norm_hit_type


def get_obs_kernel(agent_pos, hunger, grid, food_grid, size):
    batch_size = agent_pos.shape[0]
    batch_idx = torch.arange(batch_size, device=agent_pos.device)
    x = agent_pos[:, 0]
    y = agent_pos[:, 1]
    
    norm_x = x.float() / (size - 1)
    norm_y = y.float() / (size - 1)
    
    up_wall = torch.where(x == 0, torch.tensor(1.0, device=agent_pos.device), grid[batch_idx, (x - 1).clamp(0, size-1), y])
    down_wall = torch.where(x == size - 1, torch.tensor(1.0, device=agent_pos.device), grid[batch_idx, (x + 1).clamp(0, size-1), y])
    left_wall = torch.where(y == 0, torch.tensor(1.0, device=agent_pos.device), grid[batch_idx, x, (y - 1).clamp(0, size-1)])
    right_wall = torch.where(y == size - 1, torch.tensor(1.0, device=agent_pos.device), grid[batch_idx, x, (y + 1).clamp(0, size-1)])
    
    up_flag = (up_wall == 1.0).float()
    down_flag = (down_wall == 1.0).float()
    left_flag = (left_wall == 1.0).float()
    right_flag = (right_wall == 1.0).float()
    
    ray_dist, ray_type = raymarch_grid(agent_pos, grid, food_grid, size, max_dist=size)
    
    base_obs = torch.cat([
        norm_x.unsqueeze(1), 
        norm_y.unsqueeze(1), 
        up_flag.unsqueeze(1), 
        down_flag.unsqueeze(1), 
        left_flag.unsqueeze(1), 
        right_flag.unsqueeze(1),
        ray_dist,
        ray_type
    ], dim=1)
    
    spatial_noise = torch.randn_like(base_obs) * (hunger * 0.20).unsqueeze(1)
    base_obs_noisy = base_obs + spatial_noise
    
    noise_std = (hunger * 0.5).unsqueeze(1)
    noise = torch.randn(batch_size, 10, device=agent_pos.device) * noise_std
    
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
            print(f"[{self.device}] Generating a pool of {self.pool_size} solvable {self.size}x{self.size} mazes on CPU...")
            pool_tensors = []
            for _ in range(self.pool_size):
                pool_tensors.append(self._generate_solvable_maze_cpu())
            
            MazeEnvTensor._maze_pools[pool_key] = torch.stack(pool_tensors).to(self.device)
            print(f"[{self.device}] GPU Warmup complete. Resets will now be non-blocking.")

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
                        temp_grid[r, c] = 1.0  # Wall
            
            queue = [start_pos]
            visited = {start_pos}
            solvable = False
            while queue:
                r, c = queue.pop(0)
                if (r, c) == goal_pos:
                    solvable = True
                    break
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    # RESOLVED: Checks nc boundary rather than duplicate nr check
                    if 0 <= nr < self.size and 0 <= nc < self.size:
                        if temp_grid[nr, nc] != 1.0 and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
            if solvable:
                temp_grid[goal_pos[0], goal_pos[1]] = 2.0  # Goal
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
            pool_key = (self.size, str(self.device))
            pool = MazeEnvTensor._maze_pools[pool_key]
            
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
                gr, gc = goal_idx[0]
                new_food[gr, gc] = 0.0
                
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
        if len(state) > 5:
            self.needs_reset.copy_(state[5])
        else:
            self.needs_reset.fill_(False)


# ==========================================
# NEURAL NETWORK ARCHITECTURE
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
            P = pop_weights.shape[0]
            B = x.shape[0] // P
            w = pop_weights.repeat_interleave(B, dim=0)
        else:
            w = self.weight.unsqueeze(0).expand(x.shape[0], -1, -1, -1)
            
        norm_weight = F.normalize(w, p=2, dim=-2) * self.w_scale
        raw_pred = torch.einsum('bni,bnio->bno', x, norm_weight)
            
        bound = math.sqrt(2.0 + math.sqrt(2.0))
        return utils.buntanh(raw_pred, self.untanh_slope, bound)
    
    def apply_hyperparams(self, config):
        self.untanh_slope = config.UNTANH.view(-1, 1, 1)
        self.w_scale = config.w_scale.view(-1, 1, 1)


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
        if hasattr(config, 'sprs'):
            self.sprs = config.sprs.view(-1, 1, 1)
        if hasattr(config, 'c_moment'):
            self.c_moment = config.c_moment.view(-1, 1, 1, 1)
        if hasattr(config, 'a_variance'):
            self.a_variance = config.a_variance.view(-1, 1, 1)
        if hasattr(config, 'EMA_SPEED'):
            self.lp_alpha = float(config.EMA_SPEED.item())
        self.w1.apply_hyperparams(config)
        self.router.apply_hyperparams(config)

    def get_action_logits(self, state, count, voters=4):
        num_out_nodes = math.ceil(count / self.D) * voters
        latent_nodes = state[:, -num_out_nodes:, :]
        return latent_nodes.sum(dim=1)[:, :count]

    def forward(self, env_input, state, output, error_ema, lp_ema, pop_weights=None):
        PB = env_input.shape[0]
        input_dim = env_input.shape[1]
        
        w1_p = pop_weights.get(self.w1.weight) if pop_weights is not None else None
        router_p = pop_weights.get(self.router.weight) if pop_weights is not None else None
        
        prediction = self.w1(state, w1_p)
        router_logits = self.router(state, router_p)
        
        mean, error_pred = torch.chunk(prediction, 2, dim=-1)
        
        curiosity_bias = F.relu(lp_ema).unsqueeze(1) * 5.0
        route_logits = router_logits + curiosity_bias
        route, aux, mask = sparse_route_batched(route_logits, self.N, self.sprs)
        
        current_var = route.var(dim=[-1], keepdim=True)
        vd = F.relu(self.a_variance - current_var)
        
        base_noise = torch.sqrt(vd) * torch.randn_like(output)
        
        # ==========================================
        # BOREDOM SCALE MODULATION [1]
        # ==========================================
        if USE_DIMENSIONLESS_BOREDOM:
            eps = 1e-8
            relative_lp = F.relu(lp_ema) / (error_ema + eps)
            relative_lp_clipped = torch.clamp(relative_lp, min=0.0, max=1.0 - eps)
            boredom_scale = torch.pow(1.0 - relative_lp_clipped, GAMMA_COUPLING / 2.0).unsqueeze(-1)
        else:
            boredom_scale = torch.exp(-50.0 * F.relu(lp_ema)).unsqueeze(-1)
            
        boredom_noise = base_noise * boredom_scale
        
        target = torch.matmul(route, output) + boredom_noise
        
        # ==========================================
        # ENTROPIC COORDINATE RESTORING DRIFT [1]
        # ==========================================
        if USE_ENTROPIC_RESTORING_FORCE:
            restoring_drift = - (GAMMA_COUPLING * 0.01) * state * (1.0 - boredom_scale)
            target = target + restoring_drift

        target = utils.softsign(target)
        
        pad_len = self.num_input_nodes * self.D - input_dim
        if pad_len > 0:
            env_input_padded = F.pad(env_input, (0, pad_len))
        else:
            env_input_padded = env_input
            
        env_input_reshaped = env_input_padded.view(PB, self.num_input_nodes, self.D)
        
        if self.num_input_nodes < self.N:
            target = torch.cat([env_input_reshaped, target[:, self.num_input_nodes:, :]], dim=1)
        else:
            target = env_input_reshaped
            
        loss_pred = F.mse_loss(mean, target.detach(), reduction='none')
        
        actual_error = torch.abs(target - mean).detach()
        loss_err_pred = F.mse_loss(error_pred, actual_error, reduction='none')
        
        loss_w1 = loss_pred + 0.1*loss_err_pred
        
        with torch.no_grad():
            current_error = loss_w1.mean(dim=-1).detach()
            current_lp = error_ema - current_error
            next_error_ema = error_ema * (1.0 - self.lp_alpha) + current_error * self.lp_alpha
            next_lp_ema = lp_ema * (1.0 - self.lp_alpha) + current_lp * self.lp_alpha
            
        total_loss = ((loss_w1).sum() + aux) / PB
        
        # ==========================================
        # NEXT STEP RECURSIVE ERROR PASS [1]
        # ==========================================
        if USE_DIMENSIONLESS_BOREDOM:
            eps = 1e-8
            next_relative_lp = F.relu(next_lp_ema) / (next_error_ema + eps)
            next_relative_lp_clipped = torch.clamp(next_relative_lp, min=0.0, max=1.0 - eps)
            next_boredom_scale = torch.pow(1.0 - next_relative_lp_clipped, GAMMA_COUPLING / 2.0).unsqueeze(-1)
        else:
            next_boredom_scale = torch.exp(-50.0 * torch.relu(next_lp_ema)).unsqueeze(-1)
            
        errm = torch.sqrt(next_boredom_scale) * 0.01
        error_pass = torch.randn_like(target) * errm
        target = target + error_pass
        
        next_output = mean.detach()
        next_state = target.detach()
        
        return total_loss, next_state, next_output, next_error_ema, next_lp_ema, loss_pred


# ==========================================
# STATE UTILITIES & EXTRINSICS
# ==========================================
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


# ==========================================
# DECOUPLED ROLLOUT ENGINE
# ==========================================
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


# ==========================================
# TRAINING OPTIMIZERS
# ==========================================
class GradientOptimizer:
    def __init__(self, model, exec_config, use_es_lookahead=False, lr=0.01):
        self.model = model
        self.exec_config = exec_config
        self.use_es_lookahead = use_es_lookahead
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
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
            
        rollout_results = run_rollout(
            self.model, env, 
            num_steps=1, 
            state_tuple=self.state_tuple, 
            step_callback=step_callback
        )
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
                env.agent_pos.repeat(P, 1),
                env.food_grid.repeat(P, 1, 1),
                env.hunger.repeat(P),
                env.steps.repeat(P),
                env.grid.repeat(P, 1, 1),
                env.needs_reset.repeat(P)
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
            rollout_data = run_rollout(
                self.model, vec_env, 
                num_steps=steps_evaluated, 
                pop_weights=pop_weights,
                step_callback=eval_callback
            )
        
        stacked_losses = torch.stack(rollout_data['loss_w1'], dim=0)
        avg_pred_loss = stacked_losses.view(steps_evaluated, P, B, self.model.N, self.model.D).mean(dim=[0, 2, 3, 4])
        
        avg_phys_success = ((total_food_eaten * 2.5) + (reached_goal_count * 15.0) - (total_hunger * 0.05)) / B
        composite_fitnesses = -avg_pred_loss + 0.02 * avg_phys_success
        return composite_fitnesses

    def _search_lookahead(self, env):
        P = self.es_pop_size
        half_P = P // 2
        
        self.profiler.start("Setup & Noise Generation")
        master_params = {p: p.clone() for p in self.model.parameters() if p.requires_grad}
        
        noise_dict = {}
        pop_weights = {}
        for p, param_val in master_params.items():
            noise = torch.stack([
                generate_low_rank_noise(param_val.shape, self.es_rank, device=param_val.device)
                for _ in range(half_P)
            ], dim=0)
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
        
        # ==========================================
        # HYBRID LOOKAHEAD ELITE CENTROMOST SELECTION [1]
        # ==========================================
        # Identifies all mutant coordinates that beat the baseline.
        # Finds the geometric mean and picks the actual, evaluated candidate closest to that mean (Medoid) [1].
        elite_indices = torch.nonzero(candidate_fitnesses > baseline_fitness).squeeze(-1)
        
        if len(elite_indices) > 0:
            if len(elite_indices) == 1:
                best_idx = elite_indices[0].item()
            else:
                centers = {}
                for p in master_params.keys():
                    centers[p] = pop_weights[p][elite_indices].mean(dim=0)
                
                num_elites = len(elite_indices)
                dists = torch.zeros(num_elites, device=self.model.device)
                for p in master_params.keys():
                    diff = pop_weights[p][elite_indices] - centers[p].unsqueeze(0)
                    squared_diff = diff.view(num_elites, -1) ** 2
                    dists += squared_diff.sum(dim=-1)
                
                # Minimum distance to geometric mean ensures actual validated configuration [1]
                centermost_local_idx = torch.argmin(dists).item()
                best_idx = elite_indices[centermost_local_idx].item()
                
            best_p = {}
            for p in master_params.keys():
                best_p[p] = pop_weights[p][best_idx]
            
            elite_fitness = candidate_fitnesses[best_idx].item()
            self.profiler.stop("Reduce & State Restorations")
            return elite_fitness, best_p
            
        self.profiler.stop("Reduce & State Restorations")
        return baseline_fitness, None


class EGGROLLSurpriseOptimizer:
    def __init__(self, model, exec_config):
        self.model = model
        self.exec_config = exec_config
        self.profiler = CUDAProfiler()
        
        device = next(model.parameters()).device
        
        self.master_param_map = {
            p: nn.Parameter(p.clone().to(device), requires_grad=True) 
            for p in model.parameters() if p.requires_grad
        }
        
        self.optimizer = torch.optim.AdamW(
            self.master_param_map.values(), 
            lr=exec_config.es_lr, 
            weight_decay=1e-4
        )

    def _apply_master_to_model(self):
        for p, master_p in self.master_param_map.items():
            p.data.copy_(master_p.data)

    def step(self, env, step_idx=0):
        P = self.exec_config.es_pop_size
        half_P = P // 2
        B = self.exec_config.batch_size
        
        self.profiler.start("1. Low-Rank Noise Generation")
        noise_dict = {}
        pop_weights = {}
        for p, master_p in self.master_param_map.items():
            noise = torch.stack([
                generate_low_rank_noise(master_p.shape, self.exec_config.es_rank, device=master_p.device)
                for _ in range(half_P)
            ], dim=0)
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
            rollout_data = run_rollout(
                self.model, vec_env, 
                num_steps=self.exec_config.max_episode_steps, 
                pop_weights=pop_weights
            )
        self.profiler.stop("3. Complete Rollout Loop")

        self.profiler.start("4. Parameter Update & Math Reduction")
        stacked_losses = torch.stack(rollout_data['loss_w1'], dim=0)
        candidate_surprises = stacked_losses.view(self.exec_config.max_episode_steps, P, B, self.model.N, self.model.D).mean(dim=[0, 2, 3, 4])
        
        surprises_pos = candidate_surprises[:half_P]
        surprises_neg = candidate_surprises[half_P:]
        
        mean_s = candidate_surprises.mean()
        std_s = candidate_surprises.std() + 1e-8
        
        self.optimizer.zero_grad()
        
        for p, master_p in self.master_param_map.items():
            grad = torch.zeros_like(p)
            noise = noise_dict[p]
            
            for i in range(half_P):
                f_pos = (surprises_pos[i] - mean_s) / std_s
                f_neg = (surprises_neg[i] - mean_s) / std_s
                grad += (f_pos - f_neg) * noise[i]
                
            grad /= (P * self.exec_config.es_sigma)
            master_p.grad = grad
            
        self.optimizer.step()
        self._apply_master_to_model()
        self.profiler.stop("4. Parameter Update & Math Reduction")
        
        return mean_s.item()


# ==========================================
# TRAINING COORDINATOR
# ==========================================
def run_training(optimizer, env, total_steps, label):
    print(f"--- Training via {label.upper()} ---")
    start_time = time.perf_counter()
    env.reset()
    
    is_es = isinstance(optimizer, EGGROLLSurpriseOptimizer)
    losses = []
    
    for step in range(total_steps):
        loss_val = optimizer.step(env, step)
        losses.append(loss_val)
        
        if is_es:
            gen = step + 1
            if gen % 10 == 0 or gen == 1:
                print(f"Generation {gen:02d} | Cumulative Surprise: {loss_val:.2f}")
                
    elapsed = time.perf_counter() - start_time
    
    if not is_es:
        print(f"{label.upper()} Training completed in {elapsed:.2f} seconds ({elapsed/total_steps*1000:.2f} ms/step).")
        print(f"Initial 10-step average loss: {sum(losses[:10])/10:.4f}")
        print(f"Final 10-step average loss:   {sum(losses[-10:])/10:.4f}\n")
    else:
        print(f"ES Training completed in {elapsed:.2f} seconds.\n")
        
    optimizer.profiler.print_report(f"{label.upper()} OPTIMIZATION BREAKDOWN")


# ==========================================
# HYPERPARAMETER TUNING ENGINE
# ==========================================
class HyperParamTuner:
    def __init__(self, device, exec_config, generations=3, pop_size=16, sigma=0.08):
        self.device = device
        self.exec_config = exec_config
        self.generations = generations
        self.pop_size = pop_size
        self.sigma = sigma
        self.base_cfg = HyperParamsConfig(device)
        
        self.eval_model = StatelessNodeNet3(num_nodes=8, dim=16, device=self.device).to(self.device)

    def _evaluate_config_offline(self, hyper_cfg):
        self.eval_model.apply_hyperparams(hyper_cfg.to_tensor_config())
        optimizer = torch.optim.AdamW(self.eval_model.parameters(), lr=0.01)
        
        env = MazeEnvTensor(batch_size=16, size=self.exec_config.maze_size, food_density=0.15, device=self.device)
        env.reset()
        
        state_tuple = self.eval_model.init_states(16)
        steps_evaluated = 40
        
        epoch0_losses = []
        epoch1_losses = []
        
        def tuner_callback_0(loss, s, o, e, l):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch0_losses.append(loss.item())
            
        rollout_results = run_rollout(
            self.eval_model, env, 
            num_steps=steps_evaluated, 
            state_tuple=state_tuple, 
            step_callback=tuner_callback_0
        )
        state_tuple = rollout_results['state_tuple']
        
        def tuner_callback_1(loss, s, o, e, l):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch1_losses.append(loss.item())
            
        rollout_results = run_rollout(
            self.eval_model, env, 
            num_steps=steps_evaluated, 
            state_tuple=state_tuple, 
            step_callback=tuner_callback_1
        )
        
        mean_loss_0 = sum(epoch0_losses) / len(epoch0_losses)
        mean_loss_1 = sum(epoch1_losses) / len(epoch1_losses)
        
        # Scale-invariant quadratic ratio to target velocity [1]
        composite_score = - (mean_loss_1 ** 2) / (mean_loss_0 + 1e-8)
        return composite_score
    def _clamp_param(self, k, val):
        """Standardized clamping boundaries for parameters during tuning."""
        if k in ['UNTANH', 'w_scale', 'speed', 'REW_SENS']:
            return max(0.1, min(val, 3.0))
        elif k in ['sprs', 'c_moment']:
            return max(0.01, min(val, 0.99))
        elif k == 'EMA_SPEED':
            return max(0.001, min(val, 0.1))
        elif k == 'width':
            return max(0.001, min(val, 0.1))
        return val

    def tune(self):
        print("=== Tuning Architectural Hyperparameters via Offline Benchmark ===")
        
        keys = [
            'UNTANH', 'w_scale', 'EMA_SPEED', 'REW_SENS', 
            'speed', 'width', 'sprs', 'c_moment', 
        ]
        
        all_evals = []  # Records every evaluation: (score, dict_of_params)
        start_time = time.perf_counter()
        
        for gen in range(1, self.generations + 1):
            noises = []
            scores = []
            
            base_score = self._evaluate_config_offline(self.base_cfg)
            print(f"Gen {gen:02d} | Base Config Score (Learning Progress Rate): {base_score:.4f}")
            all_evals.append((base_score, {k: getattr(self.base_cfg.cfg, k).item() for k in keys}))
            
            for _ in range(self.pop_size):
                noise = {k: random.normalvariate(0, 1) for k in keys}
                noises.append(noise)
                
                perturbed_cfg = HyperParamsConfig(self.device)
                for k in keys:
                    base_val = getattr(self.base_cfg.cfg, k).item()
                    val = base_val + self.sigma * noise[k]
                    val = self._clamp_param(k, val)
                    getattr(perturbed_cfg.cfg, k).data.fill_(val)
                    
                score = self._evaluate_config_offline(perturbed_cfg)
                scores.append(score)
                all_evals.append((score, {k: getattr(perturbed_cfg.cfg, k).item() for k in keys}))
                
            mean_score = sum(scores) / len(scores)
            std_score = math.sqrt(sum((s - mean_score)**2 for s in scores) + 1e-8)
            
            for k in keys:
                grad = 0.0
                for i in range(self.pop_size):
                    norm_fitness = (scores[i] - mean_score) / std_score
                    grad += norm_fitness * noises[i][k]
                grad /= (self.pop_size * self.sigma)
                
                current_val = getattr(self.base_cfg.cfg, k).item()
                new_val = current_val + 0.05 * grad
                new_val = self._clamp_param(k, new_val)
                getattr(self.base_cfg.cfg, k).data.fill_(new_val)

        # =========================================================================
        # TUNER ELITE DENSITY CENTROMOST SELECTION [1]
        # =========================================================================
        all_evals.sort(key=lambda x: x[0], reverse=True)
        top_k = all_evals[:4]  # Take top 4 performers
        
        vectors = [[cfg_dict[k] for k in keys] for _, cfg_dict in top_k]
        
        if len(vectors) >= 3:
            mean_vec = [sum(v[i] for v in vectors) / len(vectors) for i in range(len(keys))]
            dists = [math.sqrt(sum((v[i] - mean_vec[i])**2 for i in range(len(keys)))) for v in vectors]
            
            mean_dist = sum(dists) / len(dists)
            elites = []
            for idx, v in enumerate(vectors):
                # Filter out outliers further than 1.5x the average distance
                if dists[idx] <= 1.5 * mean_dist or len(elites) < 2:
                    elites.append(top_k[idx][1])
            
            # Find the actual evaluated candidate closest to the cluster center (medoid) [1]
            center_vec = {}
            for k in keys:
                center_vec[k] = sum(e[k] for e in elites) / len(elites)
                
            best_elite = elites[0]
            min_dist_to_center = float('inf')
            for e in elites:
                dist = math.sqrt(sum((e[k] - center_vec[k])**2 for k in keys))
                if dist < min_dist_to_center:
                    min_dist_to_center = dist
                    best_elite = e
                    
            centroid_params = best_elite
        else:
            centroid_params = top_k[0][1]

        elapsed = time.perf_counter() - start_time
        print(f"\nTuning completed in {elapsed:.2f} seconds!")
        print("--- Optimized Hyperparameters Found ---")
        
        for k, v in centroid_params.items():
            getattr(self.base_cfg.cfg, k).data.fill_(v)
            print(f"  {k:15s}: {v:.6f}")
            
        print(f"  Best evaluation score reached: {all_evals[0][0]:.4f}")
        print("========================================\n")
        return self.base_cfg


# ==========================================
# VISUALIZATION & EVALUATION FUNCTIONS
# ==========================================
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
    goals = eval_results['goals_reached']
    return int(goals.argmax())


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
        rollout_data = run_rollout(
            model, env, 
            num_steps=exec_config.max_episode_steps, 
            step_callback=eval_metric_callback,
            collect_trajectories=True
        )
    
    avg_food = total_food_eaten.mean().item()
    success_rate = ever_reached_goal.float().mean().item() * 100.0
    avg_goals = goals_reached.mean().item()
    max_goals = goals_reached.max().item()
    min_goals = goals_reached.min().item()
    
    print(f"  Avg Food Eaten per Episode:        {avg_food:.2f}")
    print(f"  Goal Success Rate (Exit Reached):  {success_rate:.1f}%")
    print(f"  Avg Goal Reaches per Episode:      {avg_goals:.2f} (Max: {int(max_goals)}, Min: {int(min_goals)})\n")
    
    return {
        'grids': rollout_data['grids'],
        'food_grids': rollout_data['food_grids'],
        'paths': rollout_data['paths'],
        'goals_reached': goals_reached.cpu().numpy()
    }


# ==========================================
# EXECUTION ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    torch.set_float32_matmul_precision('high')
    exec_config = ExecutionConfig()
    
    tuner = HyperParamTuner(device, exec_config, generations=3, pop_size=16)
    optimized_hyper_config = tuner.tune()
    tensor_hyper_config = optimized_hyper_config.to_tensor_config()
    
    ref_env = MazeEnvTensor(batch_size=1, size=exec_config.maze_size, food_density=0.15, device=device)
    ref_grid = ref_env.grid[0]
    grid_batched = ref_grid.unsqueeze(0).expand(exec_config.batch_size, -1, -1)
    
    envs_gd = MazeEnvTensor(batch_size=exec_config.batch_size, size=exec_config.maze_size, grid=grid_batched, food_density=0.15, device=device)
    envs_es = MazeEnvTensor(batch_size=exec_config.batch_size, size=exec_config.maze_size, grid=grid_batched, food_density=0.15, device=device)
    envs_hybrid = MazeEnvTensor(batch_size=exec_config.batch_size, size=exec_config.maze_size, grid=grid_batched, food_density=0.15, device=device)
    
    # GD
    model_gd = StatelessNodeNet3(
        num_nodes=exec_config.num_nodes, 
        dim=exec_config.dim, 
        device=device
    ).to(device)
    model_gd.apply_hyperparams(tensor_hyper_config)
    
    optimizer_gd = GradientOptimizer(
        model_gd, 
        exec_config, 
        use_es_lookahead=False, 
        lr=exec_config.gd_lr
    )
    
    run_training(optimizer_gd, envs_gd, total_steps=exec_config.gd_total_steps, label="GD")
    gd_eval_results = run_evaluation(model_gd, envs_gd, exec_config, "Gradient Descent (Trial 1)")
    
    best_gd_idx = select_sample_to_render(gd_eval_results)
    print(f"Rendering Last Completed Maze for GD (Batch Element {best_gd_idx + 1}):")
    render_trajectory_tensor(
        gd_eval_results['grids'][best_gd_idx].cpu(),
        gd_eval_results['food_grids'][best_gd_idx].cpu(),
        gd_eval_results['paths'][best_gd_idx]
    )
    print("\n" + "="*50 + "\n")
    
    # ES
    model_es = StatelessNodeNet3(
        num_nodes=exec_config.num_nodes, 
        dim=exec_config.dim, 
        device=device
    ).to(device)
    model_es.apply_hyperparams(tensor_hyper_config)
    
    optimizer_es = EGGROLLSurpriseOptimizer(
        model_es, 
        exec_config
    )
    
    run_training(optimizer_es, envs_es, total_steps=exec_config.es_generations, label="ES")
    es_eval_results = run_evaluation(model_es, envs_es, exec_config, "EGGROLL ES (Trial 2)")
    
    best_es_idx = select_sample_to_render(es_eval_results)
    print(f"Rendering Last Completed Maze for ES (Batch Element {best_es_idx + 1}):")
    render_trajectory_tensor(
        es_eval_results['grids'][best_es_idx].cpu(),
        es_eval_results['food_grids'][best_es_idx].cpu(),
        es_eval_results['paths'][best_es_idx]
    )
    print("\n" + "="*50 + "\n")
        
    # Hybrid
    model_hybrid = StatelessNodeNet3(
        num_nodes=exec_config.num_nodes, 
        dim=exec_config.dim, 
        device=device
    ).to(device)
    model_hybrid.apply_hyperparams(tensor_hyper_config)
    
    optimizer_hybrid = GradientOptimizer(
        model_hybrid, 
        exec_config, 
        use_es_lookahead=True, 
        lr=exec_config.gd_lr
    )
    
    run_training(optimizer_hybrid, envs_hybrid, total_steps=exec_config.gd_total_steps, label="Hybrid")
    hybrid_eval_results = run_evaluation(model_hybrid, envs_hybrid, exec_config, "Interspersed Hybrid (Trial 3)")
    
    best_hybrid_idx = select_sample_to_render(hybrid_eval_results)
    print(f"Rendering Last Completed Maze for Hybrid (Batch Element {best_hybrid_idx + 1}):")
    render_trajectory_tensor(
        hybrid_eval_results['grids'][best_hybrid_idx].cpu(),
        hybrid_eval_results['food_grids'][best_hybrid_idx].cpu(),
        hybrid_eval_results['paths'][best_hybrid_idx]
    )
    print()