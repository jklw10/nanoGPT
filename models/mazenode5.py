import math
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from parts import quantizer
import parts.utils as utils
from unit import bench

USE_DIMENSIONLESS_BOREDOM = True       # True: (1 - LP/MSE)^(gamma/2), False: exp(-50 * LP)
USE_ENTROPIC_RESTORING_FORCE = True    # True: Attractive drift to 0 that scales with (1 - boredom)
GAMMA_COUPLING = 2.0                   # Physical coupling parameter (gamma) for both equations

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

successful_params = {
    'UNTANH': 0.725268,
    'w_scale': 0.548835,
    'EMA_SPEED': 0.071671,
    'REW_SENS': 1.330760,
    'speed': 1.355162,
    'width': 0.001000,
    'sprs': 0.580194,
    'c_moment': 0.905514,
    'a_variance': 0.05,
    
    'opt_gd_lr': 0.01,
    'opt_es_lr': 0.03,
    'opt_beta_cov': 0.99,
    'opt_damping': 1e-5,
    'opt_decay': 1e-5,
    'opt_R': 1.0,
    'opt_gamma': 1e-5
}

class HyperParamsConfig:
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
        self.es_pop_size = 16          # Increased population for broader spectral rank coverage
        self.es_rank = 8              # Increased from 4 to 8 to avoid degenerate 1D subspace locks
        self.es_generations = 50


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
    
    # Stay in place if moving into a wall
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
    
    # Track wall bumps to penalize static corner-pinning in active inference
    bumped_wall = is_wall
    
    return next_pos, next_food_grid, next_hunger, next_steps, bumped_wall


def raymarch_grid(agent_pos, grid, food_grid, size, max_dist=10):
    B = agent_pos.shape[0]
    dirs = torch.tensor([
        [-1, 0], [1, 0], [0, -1], [0, 1],
        [-1, -1], [-1, 1], [1, -1], [1, 1]
    ], dtype=torch.long, device=agent_pos.device)
    
    steps = torch.arange(1, max_dist + 1, device=agent_pos.device)
    cur_x = agent_pos[:, 0].view(B, 1, 1) + dirs[:, 0].view(1, 8, 1) * steps.view(1, 1, -1)
    cur_y = agent_pos[:, 1].view(B, 1, 1) + dirs[:, 1].view(1, 8, 1) * steps.view(1, 1, -1)
    
    out_of_bounds = (cur_x < 0) | (cur_x >= size) | (cur_y < 0) | (cur_y >= size)
    safe_x = cur_x.clamp(0, size - 1)
    safe_y = cur_y.clamp(0, size - 1)
    
    batch_idx = torch.arange(B, device=agent_pos.device).view(B, 1, 1)
    grid_vals = grid[batch_idx, safe_x, safe_y]
    food_vals = food_grid[batch_idx, safe_x, safe_y]
    
    is_wall = (grid_vals == 1.0) | out_of_bounds
    is_goal = (grid_vals == 2.0) & ~out_of_bounds
    is_food = (food_vals == 1.0) & ~out_of_bounds
    
    any_hit = is_wall | is_goal | is_food
    has_hit = any_hit.any(dim=-1)
    
    hit_idx = torch.argmax(any_hit.float(), dim=-1)
    hit_dist = torch.where(has_hit, hit_idx.float() + 1.0, float(max_dist))
    
    gather_idx = hit_idx.unsqueeze(-1)
    hit_wall = torch.gather(is_wall, dim=-1, index=gather_idx).squeeze(-1)
    hit_goal = torch.gather(is_goal, dim=-1, index=gather_idx).squeeze(-1)
    hit_food = torch.gather(is_food, dim=-1, index=gather_idx).squeeze(-1)
    
    hit_type = torch.zeros(B, 8, device=agent_pos.device)
    hit_type = torch.where(hit_wall, 1.0, hit_type)
    hit_type = torch.where(hit_goal, 2.0, hit_type)
    hit_type = torch.where(hit_food, 3.0, hit_type)
    hit_type = torch.where(has_hit, hit_type, 0.0)
    
    return hit_dist / float(max_dist), hit_type / 3.0
def get_obs_kernel(agent_pos, hunger, grid, food_grid, size, prev_noise=None):
    batch_size = agent_pos.shape[0]
    batch_idx = torch.arange(batch_size, device=agent_pos.device)
    x, y = agent_pos[:, 0], agent_pos[:, 1]
    
    norm_x = x.float() / (size - 1)
    norm_y = y.float() / (size - 1)
    
    up_wall = torch.where(x == 0, 1.0, grid[batch_idx, (x - 1).clamp(0, size-1), y])
    down_wall = torch.where(x == size - 1, 1.0, grid[batch_idx, (x + 1).clamp(0, size-1), y])
    left_wall = torch.where(y == 0, 1.0, grid[batch_idx, x, (y - 1).clamp(0, size-1)])
    right_wall = torch.where(y == size - 1, 1.0, grid[batch_idx, x, (y + 1).clamp(0, size-1)])
    
    up_flag, down_flag = (up_wall == 1.0).float(), (down_wall == 1.0).float()
    left_flag, right_flag = (left_wall == 1.0).float(), (right_wall == 1.0).float()
    
    ray_dist, ray_type = raymarch_grid(agent_pos, grid, food_grid, size, max_dist=size)
    
    base_obs = torch.cat([
        norm_x.unsqueeze(1), norm_y.unsqueeze(1), 
        up_flag.unsqueeze(1), down_flag.unsqueeze(1), 
        left_flag.unsqueeze(1), right_flag.unsqueeze(1),
        ray_dist, ray_type
    ], dim=1) # (B, 22)
    
    # -------------------------------------------------------------
    # SENSORY OVERLAY: Starvation degrades reality itself
    # -------------------------------------------------------------
    h = hunger.unsqueeze(1) # (B, 1)
    
    # 1. Direct overlay noise across all 22 informative dimensions
    # When hunger=0 (full), signal is 100% crisp.
    # When hunger=1 (starving), noise overlays heavily and distorts signal geometry.
    raw_noise = torch.randn_like(base_obs)
    spectral_noise = utils.apply_spectral_filter(raw_noise, center_offset=1.0, sigma_val=0.5)
    
    # Signal attenuation + noise overlay
    signal_clarity = 1.0 - 0.4 * h
    noise_amplitude = 0.6 * (h ** 2) # Quadratic surge as starvation sets in
    
    obs_distorted = base_obs * signal_clarity + spectral_noise * noise_amplitude
    
    # The remaining 10 channels are pure hallucinated somatic channels (hunger static)
    somatic_raw = torch.randn(batch_size, 10, device=agent_pos.device)
    somatic_noise = utils.apply_spectral_filter(somatic_raw, center_offset=1.0, sigma_val=0.5) * (h ** 1.5)
    
    return torch.cat([obs_distorted, somatic_noise], dim=1) # (B, 32)


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
        self.last_bump = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
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
            if goal_pos == start_pos or (abs(goal_pos[0] - start_pos[0]) + abs(goal_pos[1] - start_pos[1])) < 3:
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
        self.last_bump[indices] = False
        
        if not self.static_grid:
            pool = MazeEnvTensor._maze_pools[(self.size, str(self.device))]
            pool_idx = torch.randint(0, self.pool_size, (len(indices),), device=self.device)
            self.grid[indices] = pool[pool_idx]
            
        num_reset = len(indices)
        empty_mask = (self.grid[indices] == 0.0).float()
        random_mask = (torch.rand(num_reset, self.size, self.size, device=self.device) < self.food_density).float()
        new_food = empty_mask * random_mask
        new_food[:, self.size // 2, self.size // 2] = 0.0
        
        is_not_goal = (self.grid[indices] != 2.0).float()
        new_food = new_food * is_not_goal
        self.food_grid[indices] = new_food
            
        return self._get_obs()

    def _masked_reset(self):
        mask = self.needs_reset
        mask_2d = mask.unsqueeze(1)
        mask_3d = mask.unsqueeze(1).unsqueeze(2)
        
        self.agent_pos = torch.where(mask_2d, torch.full_like(self.agent_pos, self.size // 2), self.agent_pos)
        self.hunger = torch.where(mask, torch.full_like(self.hunger, 0.5), self.hunger)
        self.steps = torch.where(mask, torch.zeros_like(self.steps), self.steps)
        self.last_food_eaten = torch.where(mask, torch.zeros_like(self.last_food_eaten), self.last_food_eaten)
        self.last_at_goal = torch.where(mask, torch.zeros_like(self.last_at_goal), self.last_at_goal)
        self.last_bump = torch.where(mask, torch.zeros_like(self.last_bump), self.last_bump)
        
        if not self.static_grid:
            pool = MazeEnvTensor._maze_pools[(self.size, str(self.device))]
            pool_idx = torch.randint(0, self.pool_size, (self.batch_size,), device=self.device)
            new_grids = pool[pool_idx]
            self.grid = torch.where(mask_3d, new_grids, self.grid)
            
        empty_mask = (self.grid == 0.0).float()
        random_mask = (torch.rand(self.batch_size, self.size, self.size, device=self.device) < self.food_density).float()
        new_food = empty_mask * random_mask
        new_food[:, self.size // 2, self.size // 2] = 0.0
        is_not_goal = (self.grid != 2.0).float()
        new_food = new_food * is_not_goal
        
        self.food_grid = torch.where(mask_3d, new_food, self.food_grid)
        self.needs_reset = torch.where(mask, torch.zeros_like(self.needs_reset), self.needs_reset)

    def _get_obs(self):
        return get_obs_kernel(self.agent_pos, self.hunger, self.grid, self.food_grid, self.size, self.last_bump)

    def step(self, actions):
        self._masked_reset()
        batch_idx = torch.arange(self.batch_size, device=self.device)
        old_food_grid = self.food_grid.clone()
        
        self.agent_pos, self.food_grid, self.hunger, self.steps, self.last_bump = step_kernel(
            self.agent_pos, self.food_grid, self.hunger, self.steps, actions, self.grid, self.size
        )
        
        self.last_food_eaten = (old_food_grid[batch_idx, self.agent_pos[:, 0], self.agent_pos[:, 1]] == 1.0)
        self.last_at_goal = (self.grid[batch_idx, self.agent_pos[:, 0], self.agent_pos[:, 1]] == 2.0)
        self.needs_reset = (self.steps >= 40) | self.last_at_goal
            
        return self._get_obs()

    def get_state(self):
        return (self.agent_pos.clone(), self.food_grid.clone(), self.hunger.clone(), self.steps.clone(), self.grid.clone(), self.needs_reset.clone(), self.last_bump.clone())

    def restore_state(self, state):
        self.agent_pos.copy_(state[0])
        self.food_grid.copy_(state[1])
        self.hunger.copy_(state[2])
        self.steps.copy_(state[3])
        self.grid.copy_(state[4])
        self.needs_reset.copy_(state[5])
        self.last_bump.copy_(state[6])


# ==========================================
# 3. NEURAL NETWORK ARCHITECTURE (UNTOUCHED)
# ==========================================
def color_noise(size, center, width, device):
    noise = torch.randn(size, device=device)
    return utils.apply_spectral_filter(noise, center_offset=center, sigma_val=width)


class NodeParams3(nn.Module):
    def __init__(self, num_nodes, in_dim, out_dim, device):
        super().__init__() 
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        
        base_weight = color_noise((num_nodes, in_dim, out_dim), 0.5, 0.2, device=self.device) / math.sqrt(self.in_dim)
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


def sparse_route_batched(router, node_count):
    probs = torch.sigmoid(router)
    mask = quantizer.topkhotSTE(router, int(math.sqrt(node_count)))
    sparse_probs = mask * probs
    sparse_probs = sparse_probs / (sparse_probs.sum(dim=-1, keepdim=True) + 1e-8)
    return sparse_probs, 0.0, mask

class StatelessNodeNet3(nn.Module):
    def __init__(self, num_nodes, dim, device, input_dim=32, num_actions=4):
        super().__init__()
        self.N = num_nodes
        self.D = dim
        self.device = device
        self.num_actions = num_actions
        self.num_input_nodes = math.ceil(input_dim / dim) # 2 nodes
        
        # Sensory projection into latent space
        self.sensory_proj = nn.Linear(input_dim, self.num_input_nodes * dim, bias=False, device=device)
        nn.init.orthogonal_(self.sensory_proj.weight, gain=0.8)
        
        # Direct sensory-to-motor reflex bypass (fast-path reflex arc)
        self.reflex_proj = nn.Linear(input_dim, num_actions * dim, bias=False, device=device)
        nn.init.normal_(self.reflex_proj.weight, std=0.02)
        
        self.w1 = NodeParams3(num_nodes, dim, dim * 2, device)
        self.router = NodeParams3(num_nodes, dim, num_nodes, device)
        
        self.motor_proj = nn.Parameter(
            color_noise((self.num_actions, dim), 0.5, 0.2, device=device) / math.sqrt(dim)
        )
        
        self.c_moment = 0.8
        self.a_variance = 0.05
        self.lp_alpha = 0.25 
        self.sprs = 0.5

    def init_states(self, batch_size):
        state = 1e-4 * color_noise((batch_size, self.N, self.D), 1.0, 1.0, device=self.device)
        output = 1e-4 * color_noise((batch_size, self.N, self.D), 1.0, 1.0, device=self.device)
        error_ema = torch.ones(batch_size, self.N, device=self.device) * 0.05
        lp_ema = torch.zeros(batch_size, self.N, device=self.device)
        return state, output, error_ema, lp_ema

    def apply_hyperparams(self, config):
        self.sprs = config.get('sprs').view(-1, 1, 1)
        self.c_moment = config.get('c_moment').view(-1, 1, 1, 1)
        self.a_variance = config.get('a_variance').view(-1, 1, 1)
        self.lp_alpha = float(config.get('EMA_SPEED').item())
        self.w1.apply_hyperparams(config)
        self.router.apply_hyperparams(config)

    def get_action_logits(self, state, count=4):
        motor_states = state[:, -count:, :] # (B, 4, D)
        return torch.einsum('bad,ad->ba', motor_states, self.motor_proj)

    def forward(self, env_input, hunger, state, output, error_ema, lp_ema, pop_weights=None):
        PB = env_input.shape[0]
        h = hunger.view(PB, 1, 1) # (PB, 1, 1)
        
        w1_p = pop_weights.get(self.w1.weight) if pop_weights is not None else None
        router_p = pop_weights.get(self.router.weight) if pop_weights is not None else None
        
        # 1. Sensory Projection
        sensory_latent = utils.softsign(self.sensory_proj(env_input).view(PB, self.num_input_nodes, self.D))
        
        # Fast-path reflex: sensory signals directly prime motor nodes
        motor_reflex = utils.softsign(self.reflex_proj(env_input).view(PB, self.num_actions, self.D))
        
        # 2. Sparse Routing Gated by Internal Learning Progress
        router_logits = self.router(state, router_p)
        cognitive_lp = lp_ema.clone()
        cognitive_lp[:, :self.num_input_nodes] = 0.0 # Shield from sensory noise
        route_logits = router_logits + F.relu(cognitive_lp).unsqueeze(1) * 3.0
        sparse_probs, aux, mask = sparse_route_batched(route_logits, self.N)
        
        # 3. Factorized Graph Diffusion
        local_diffusion = torch.matmul(sparse_probs, output)
        global_mean = torch.mean(output, dim=1, keepdim=True)
        target = (1.0 - self.sprs) * local_diffusion + self.sprs * global_mean
        
        # 4. Dimensionless Boredom Metric
        # Relative LP = LP / (MSE + eps)
        relative_lp = torch.clamp(F.relu(lp_ema) / (error_ema + 1e-6), min=0.0, max=1.0)
        # Boredom: (1 - relative_lp)^(gamma/2)
        boredom_scale = torch.pow(1.0 - relative_lp, GAMMA_COUPLING / 2.0).unsqueeze(-1) # (PB, N, 1)
        
        if USE_ENTROPIC_RESTORING_FORCE:
            target = target - (GAMMA_COUPLING * 0.01) * state * (1.0 - boredom_scale)
            
        target = utils.softsign(target)
        
        # Inject sensory nodes at the front, prime motor nodes at the back
        target = torch.cat([
            sensory_latent, 
            target[:, self.num_input_nodes:-self.num_actions, :],
            utils.softsign(target[:, -self.num_actions:, :] + 0.3 * motor_reflex)
        ], dim=1)
        
        # 5. Prediction & MULTIPLICATIVE Loss
        prediction = self.w1(state, w1_p)
        mean, error_pred = torch.chunk(prediction, 2, dim=-1)
        
        loss_pred = F.mse_loss(mean, target.detach(), reduction='none')
        loss_err_pred = F.mse_loss(error_pred, torch.abs(target - mean).detach(), reduction='none')
        
        # MULTIPLICATIVE LOSS: Error is scaled by bodily demand (hunger)
        # When hunger is high, predictive error carries higher thermodynamic cost
        loss_w1 = (loss_pred + 0.1 * loss_err_pred) * (1.0 + 2.0 * h)
        
        with torch.no_grad():
            current_error = loss_w1.mean(dim=-1).detach()
            
            # Log-ratio Learning Progress
            ratio_lp = torch.log((error_ema + 1e-6) / (current_error + 1e-6))
            
            alpha = self.lp_alpha
            next_error_ema = error_ema * (1.0 - alpha) + current_error * alpha
            next_lp_ema = lp_ema * (1.0 - alpha) + ratio_lp * alpha
            
        total_loss = (loss_w1.sum() + aux) / PB

        next_relative_lp = torch.clamp(F.relu(next_lp_ema) / (next_error_ema + 1e-6), min=0.0, max=1.0)
        next_boredom_scale = torch.pow(1.0 - next_relative_lp, GAMMA_COUPLING / 2.0).unsqueeze(-1)

        # 6. MULTIPLICATIVE JOINT NOISE COUPLING
        # Total exploratory turbulence is the PRODUCT of Hunger and Boredom:
        # Omega = Hunger * Boredom
        omega = h * next_boredom_scale # (PB, N, 1)
        
        v_pragmatic = target.detach() - state
        
        target_noise = torch.randn_like(target)
        filtered_target_noise = utils.apply_spectral_filter(target_noise, center_offset=1.0, sigma_val=0.25)
        
        # Multiplicative epistemic kick:
        # Scales strictly with omega = (Hunger * Boredom)
        v_epistemic = filtered_target_noise * omega * 0.40
        
        settled_state = utils.softsign(state + 0.7 * v_pragmatic + v_epistemic)
        next_state = torch.cat([sensory_latent, settled_state[:, self.num_input_nodes:, :]], dim=1)
        next_output = mean.detach()
        
        return total_loss, next_state.detach(), next_output.detach(), next_error_ema, next_lp_ema, loss_pred

# ==========================================
# 4. OPTIMIZER & SPECIALIZED TRAINERS
# ==========================================
def build_optimizer(params, hyper_cfg, lr):
    return bench.UnifiedKreinEntropicOptimizer(
        params,
        lr=lr,
        gamma=float(hyper_cfg.get('opt_gamma').item()),
        damping=float(hyper_cfg.get('opt_damping').item()),
        beta_cov=float(hyper_cfg.get('opt_beta_cov').item()),
        use_adam=True,
        use_kron=True,
        use_krein_space=False,
        use_grms=True,
        kron_freq=1,
        weight_decay=0.01
    ) 


def generate_low_rank_noise(shape, rank=8, device=None):
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
        obs, env.hunger, state, output, error_ema, lp_ema, pop_weights=pop_weights
    )
    logits = model.get_action_logits(next_state, 4)
    actions = torch.argmax(logits, dim=-1)
    next_obs = env.step(actions)
    return loss, next_state, next_output, next_error_ema, next_lp_ema, loss_w1, actions, next_obs


def run_rollout(model, env, num_steps, state_tuple=None, pop_weights=None, step_callback=None, collect_trajectories=False, temperature=0.1):
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


class EGGROLLSurpriseOptimizer:
    def __init__(self, model, exec_config, hyper_cfg):
        self.model = model
        self.exec_config = exec_config
        self.hyper_cfg = hyper_cfg
        
        device = next(model.parameters()).device
        self.master_param_map = {
            p: nn.Parameter(p.clone().to(device), requires_grad=True) for p in model.parameters() if p.requires_grad
        }
        self.optimizer = build_optimizer(self.master_param_map.values(), hyper_cfg, lr=float(hyper_cfg.get('opt_es_lr').item()))
        self.vec_env = None

    def _apply_master_to_model(self):
        for p, master_p in self.master_param_map.items():
            p.data.copy_(master_p.data)

    def step(self, env, step_idx=0):
        P, half_P, B = self.exec_config.es_pop_size, self.exec_config.es_pop_size // 2, env.batch_size
        
        noise_dict, pop_weights = {}, {}
        for p, master_p in self.master_param_map.items():
            white_noise = torch.stack([
                generate_low_rank_noise(master_p.shape, self.exec_config.es_rank, device=master_p.device) 
                for _ in range(half_P)
            ], dim=0)
            
            colored_noise = utils.apply_spectral_filter(white_noise, center_offset=1.0, sigma_val=0.10)
            noise_dict[p] = colored_noise
            
            w_pos = master_p.unsqueeze(0) + self.exec_config.es_sigma * colored_noise
            w_neg = master_p.unsqueeze(0) - self.exec_config.es_sigma * colored_noise
            pop_weights[p] = torch.cat([w_pos, w_neg], dim=0)
            
        # Draw from the general pool for population evaluation to prevent golden-ticket overfitting
        if self.vec_env is None or self.vec_env.batch_size != P * B:
            self.vec_env = MazeEnvTensor(
                batch_size=P * B, 
                size=env.size, 
                food_density=env.food_density, 
                device=env.device
            )
        self.vec_env.reset()
        
        with torch.no_grad():
            rollout_data = run_rollout(
                self.model, self.vec_env, num_steps=self.exec_config.max_episode_steps, 
                pop_weights=pop_weights
            )

        stacked_losses = torch.stack(rollout_data['loss_w1'], dim=0) # (T, P*B, N, D)
        
        # Fitness combines predictive novelty with coverage
        candidate_surprises = stacked_losses.view(self.exec_config.max_episode_steps, P, B, self.model.N, self.model.D).mean(dim=[0, 2, 3, 4])
        surprises_pos, surprises_neg = candidate_surprises[:half_P], candidate_surprises[half_P:]
        
        mean_s, std_s = candidate_surprises.mean(), candidate_surprises.std() + 1e-8
        weights = (((surprises_pos - mean_s) / std_s) - ((surprises_neg - mean_s) / std_s))
        
        self.optimizer.zero_grad()
        for p, master_p in self.master_param_map.items():
            noise = noise_dict[p]
            grad = torch.einsum('i,i...->...', weights, noise)
            master_p.grad = grad / (P * self.exec_config.es_sigma)
            
        self.optimizer.step()
        self._apply_master_to_model()
        return mean_s.item()


class GradientOptimizer:
    def __init__(self, model, exec_config, hyper_cfg):
        self.model = model
        self.exec_config = exec_config
        self.hyper_cfg = hyper_cfg
        
        self.optimizer = build_optimizer(
            model.parameters(), 
            hyper_cfg, 
            lr=float(hyper_cfg.get('opt_gd_lr').item())
        )
        self.state_tuple = model.init_states(exec_config.batch_size)

    def step(self, env, step_idx):
        losses = []
        def step_callback(loss, s, o, e, l):
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            
        if self.state_tuple is None or self.state_tuple[0].shape[0] != env.batch_size:
            self.state_tuple = self.model.init_states(env.batch_size)

        rollout_results = run_rollout(
            self.model, env, num_steps=1, state_tuple=self.state_tuple, 
            step_callback=step_callback
        )
        self.state_tuple = rollout_results['state_tuple']
        return losses[0]


def run_evaluation(model, env, exec_config, label):
    print(f"=== Robust Multi-Maze Evaluation for {label} ===")
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
            model, env, num_steps=exec_config.max_episode_steps, 
            step_callback=eval_metric_callback, collect_trajectories=True
        )
    
    avg_food = total_food_eaten.mean().item()
    success_rate = ever_reached_goal.float().mean().item() * 100.0
    avg_goals = goals_reached.mean().item()
    
    print(f"  Avg Food Eaten per Episode:        {avg_food:.2f}")
    print(f"  Goal Success Rate (Exit Reached):  {success_rate:.1f}%")
    print(f"  Avg Goal Reaches per Episode:      {avg_goals:.2f} (Max: {int(goals_reached.max().item())}, Min: {int(goals_reached.min().item())})\n")


# ==========================================
# 5. EXECUTION ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    torch.set_float32_matmul_precision('high')
    exec_config = ExecutionConfig()
    
    # ------------------------------------------
    # TRIAL 1: GRADIENT DESCENT (GD)
    # ------------------------------------------
    gd_hyperparams = HyperParamsConfig(device, init_dict=successful_params)
    envs_gd = MazeEnvTensor(batch_size=exec_config.batch_size, size=exec_config.maze_size, food_density=0.15, device=device)
    model_gd = StatelessNodeNet3(num_nodes=exec_config.num_nodes, dim=exec_config.dim, device=device).to(device)
    model_gd.apply_hyperparams(gd_hyperparams)
    
    optimizer_gd = GradientOptimizer(model_gd, exec_config, gd_hyperparams)
    print("--- Training GD on Multi-Maze Pool ---")
    for step in range(exec_config.gd_total_steps):
        loss_val = optimizer_gd.step(envs_gd, step)
    run_evaluation(model_gd, envs_gd, exec_config, "Gradient Descent (Trial 1)")

    # ------------------------------------------
    # TRIAL 2: EVOLUTION STRATEGIES (ES)
    # ------------------------------------------
    es_hyperparams = HyperParamsConfig(device, init_dict=successful_params)
    envs_es = MazeEnvTensor(batch_size=exec_config.batch_size, size=exec_config.maze_size, food_density=0.15, device=device)
    model_es = StatelessNodeNet3(num_nodes=exec_config.num_nodes, dim=exec_config.dim, device=device).to(device)
    model_es.apply_hyperparams(es_hyperparams)
    
    optimizer_es = EGGROLLSurpriseOptimizer(model_es, exec_config, es_hyperparams)
    print("--- Training ES on Multi-Maze Pool ---")
    for step in range(exec_config.es_generations):
        mean_s = optimizer_es.step(envs_es, step)
        if (step + 1) % 10 == 0 or step == 0:
            print(f"Generation {step+1:02d} | Cumulative Surprise: {mean_s:.3f}")
    run_evaluation(model_es, envs_es, exec_config, "EGGROLL ES (Trial 2)")