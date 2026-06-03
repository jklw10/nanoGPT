import math
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from parts import quantizer
import parts.utils as utils

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
        self.params = {
            'untanh':       0.862396,
            'w_scale':      1.169223,
            'ema_speed':    0.120148,
            'rew_sens':     1.147007,
            'noise_scale':  0.021769,
            'speed':        1.177348,
            'width':        0.100000,
            'sprs':         0.408533,
            'c_moment':     0.801097,
            'a_variance':   0.050000
        }
        self.cfg = type('Config', (), {})()
        self.cfg.device = self.device
        self.cfg.UNTANH      = torch.tensor(self.params['untanh'], device=self.device).detach().clone()
        self.cfg.w_scale     = torch.tensor(self.params['w_scale'], device=self.device).detach().clone()
        self.cfg.EMA_SPEED   = torch.tensor(self.params['ema_speed'], device=self.device).detach().clone()
        self.cfg.rew_sens    = torch.tensor(self.params['rew_sens'], device=self.device).detach().clone()
        self.cfg.NOISE_SCALE = torch.tensor(self.params['noise_scale'], device=self.device).detach().clone()
        self.cfg.speed       = torch.tensor(self.params['speed'], device=self.device).detach().clone()
        self.cfg.width       = torch.tensor(self.params['width'], device=self.device).detach().clone()
        self.cfg.sprs        = torch.tensor(self.params['sprs'], device=self.device).detach().clone()
        self.cfg.c_moment    = torch.tensor(self.params['c_moment'], device=self.device).detach().clone()
        self.cfg.a_variance  = torch.tensor(self.params['a_variance'], device=self.device).detach().clone()
        
    def to_tensor_config(self):
        return self.cfg


class ExecutionConfig:
    def __init__(self):
        self.batch_size = 64
        self.num_nodes = 8
        self.dim = 16
        
        self.maze_size = 10
        self.max_episode_steps = 40
        
        self.gd_lr = 0.01
        self.gd_total_steps = 500
        
        self.es_lr = 0.03
        self.es_sigma = 0.08
        self.es_pop_size = 12
        self.es_rank = 4
        self.es_generations = 40


def sparse_route(router, node_count, spr):
    probs = torch.sigmoid(router)
    mask = quantizer.topkhotSTE(router, int(math.sqrt(node_count)))
    sparse_probs = mask * probs
    
    sparse_probs = sparse_probs / (sparse_probs.sum(dim=-1, keepdim=True) + 1e-8)
    
    flat_probs = torch.ones_like(probs) / node_count
    ratio = spr #* (0.999 ** step) 

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
    
    # Check if the step enters a goal tile dynamically (represented by 2.0 on the grid)
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
    
    # Apply hunger-dependent noise directly to the spatial channels (base_obs)
    spatial_noise = torch.randn_like(base_obs) * (hunger * 0.20).unsqueeze(1)
    base_obs_noisy = base_obs + spatial_noise
    
    # Separate channels for traditional noise channels
    noise_std = (hunger * 0.5).unsqueeze(1)
    noise = torch.randn(batch_size, 10, device=agent_pos.device) * noise_std
    
    return torch.cat([base_obs_noisy, noise], dim=1)


class MazeEnvTensor:
    def __init__(self, batch_size, size=6, grid=None, food_density=0.15, device="cuda"):
        self.batch_size = batch_size
        self.size = size
        self.device = device
        self.food_density = food_density
        
        if grid is not None:
            self.grid = grid.clone().to(device)
            if self.grid.dim() == 2:
                self.grid = self.grid.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            self.grid = torch.zeros(batch_size, size, size, device=device)
            for b in range(batch_size):
                single_maze = self._generate_solvable_maze_cpu()
                self.grid[b].copy_(single_maze)
                
        empty_mask = (self.grid == 0.0).float()
        random_mask = (torch.rand(self.grid.shape, device=device) < self.food_density).float()
        self.initial_food_grid = empty_mask * random_mask
        
        # Clear food on the start position (center) and goals dynamically
        self.initial_food_grid[:, size // 2, size // 2] = 0.0
        for b in range(batch_size):
            goal_idx = torch.nonzero(self.grid[b] == 2.0)
            if len(goal_idx) > 0:
                gr, gc = goal_idx[0]
                self.initial_food_grid[b, gr, gc] = 0.0
                        
        # Initialize agent directly in the center of the grid
        self.agent_pos = torch.full((batch_size, 2), size // 2, dtype=torch.long, device=device)
        self.food_grid = self.initial_food_grid.clone()
        self.hunger = torch.full((batch_size,), 0.5, device=device)
        self.steps = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        self.last_food_eaten = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self.last_at_goal = torch.zeros(batch_size, dtype=torch.bool, device=device)

    def _generate_solvable_maze_cpu(self):
        start_pos = (self.size // 2, self.size // 2)
        while True:
            # Randomize goal position across the board, excluding the center start position
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
                    if 0 <= nr < self.size and 0 <= nc < self.size:
                        if temp_grid[nr, nc] != 1.0 and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
            if solvable:
                temp_grid[goal_pos[0], goal_pos[1]] = 2.0
                return temp_grid

    def reset(self, indices=None):
        if indices is None:
            self.agent_pos.fill_(self.size // 2)
            self.food_grid.copy_(self.initial_food_grid)
            self.hunger.fill_(0.5)
            self.steps.zero_()
            self.last_food_eaten.zero_()
            self.last_at_goal.zero_()
        else:
            self.agent_pos[indices] = self.size // 2
            self.food_grid[indices] = self.initial_food_grid[indices]
            self.hunger[indices] = 0.5
            self.steps[indices] = 0
            self.last_food_eaten[indices] = False
            self.last_at_goal[indices] = False
        return self._get_obs()

    def _get_obs(self):
        return get_obs_kernel(self.agent_pos, self.hunger, self.grid, self.food_grid, self.size)

    def step(self, actions):
        batch_idx = torch.arange(self.batch_size, device=self.device)
        old_food_grid = self.food_grid.clone()
        
        self.agent_pos, self.food_grid, self.hunger, self.steps = step_kernel(
            self.agent_pos, self.food_grid, self.hunger, self.steps, actions, self.grid, self.size
        )
        
        self.last_food_eaten = (old_food_grid[batch_idx, self.agent_pos[:, 0], self.agent_pos[:, 1]] == 1.0)
        self.last_at_goal = (self.grid[batch_idx, self.agent_pos[:, 0], self.agent_pos[:, 1]] == 2.0)
        
        dones = (self.steps >= 40)
        if dones.any():
            done_idx = torch.nonzero(dones).squeeze(-1)
            self.reset(indices=done_idx)
            
        return self._get_obs()

    def get_state(self):
        return (self.agent_pos.clone(), self.food_grid.clone(), self.hunger.clone(), self.steps.clone())

    def restore_state(self, state):
        self.agent_pos.copy_(state[0])
        self.food_grid.copy_(state[1])
        self.hunger.copy_(state[2])
        self.steps.copy_(state[3])


# ==========================================
# NEURAL NETWORK ARCHITECTURE
# ==========================================
class NodeParams3(nn.Module):
    def __init__(self, batch_size, num_nodes, in_dim, out_dim, device):
        super().__init__() 
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        
        base_weight = torch.randn(num_nodes, in_dim, out_dim, device=device) / math.sqrt(self.in_dim)
        self.weight = nn.Parameter(base_weight)
        
        self.untanh_slope = torch.tensor(0.5, device=device)
        self.w_scale = torch.tensor(0.8, device=device)

    def forward(self, x, step_counter, pop_weights=None):
        if pop_weights is not None:
            norm_weight = F.normalize(pop_weights, p=2, dim=-2) * self.w_scale.view(-1, 1, 1)
            raw_pred = torch.einsum('pbni,pnio->pbno', x, norm_weight)
        else:
            norm_weight = F.normalize(self.weight, p=2, dim=-2) * self.w_scale
            raw_pred = torch.einsum('bnd,nde->bne', x, norm_weight)
            
        bound = math.sqrt(2.0 + math.sqrt(2.0))
        out = utils.buntanh(raw_pred, self.untanh_slope, bound)
        return out
    
    def apply_hyperparams(self, config):
        self.untanh_slope = config.UNTANH.view(-1, 1, 1)
        self.w_scale = config.w_scale.view(-1, 1, 1)


class StandardNodeNet3(nn.Module):
    def __init__(self, batch_size, num_nodes, dim, device):
        super().__init__()
        self.N = num_nodes
        self.D = dim
        self.device = device
        
        self.w1 = NodeParams3(batch_size, num_nodes, dim, dim, device)
        self.router = NodeParams3(batch_size, num_nodes, dim, num_nodes, device)
        self.wnll = utils.wnormlearnloss(dim)
        
        self.register_buffer('state', 1e-5 * torch.randn(batch_size, self.N, dim, device=device))
        self.register_buffer('output', 1e-5 * torch.randn(batch_size, self.N, dim, device=device))
        self.register_buffer('error_ema', torch.ones(batch_size, self.N, device=device))
        self.register_buffer('lp_ema', torch.zeros(batch_size, self.N, device=device))
        self.register_buffer('step_counter', torch.tensor(0.0, device=device))
        
        self.state_pop = None
        self.output_pop = None
        self.error_ema_pop = None
        self.lp_ema_pop = None
        
        self.c_moment = 0.8
        self.a_variance = 0.05
        self.lp_alpha = 0.1 
        self.sprs = 0.5
    
    def init_vectorized_states(self, P, B):
        self.state_pop = 1e-5 * torch.randn(P, B, self.N, self.D, device=self.device)
        self.output_pop = 1e-5 * torch.randn(P, B, self.N, self.D, device=self.device)
        self.error_ema_pop = torch.ones(P, B, self.N, device=self.device)
        self.lp_ema_pop = torch.zeros(P, B, self.N, device=self.device)

    def apply_hyperparams(self, config):
        if hasattr(config, 'sprs'):
            self.sprs = config.sprs.view(-1, 1, 1)
        if hasattr(config, 'c_moment'):
            self.c_moment = config.c_moment.view(-1, 1, 1, 1)
        if hasattr(config, 'a_variance'):
            self.a_variance = config.a_variance.view(-1, 1, 1)
        self.w1.apply_hyperparams(config)
        self.router.apply_hyperparams(config)

    def forward(self, env_input, pop_weights=None):
        self.step_counter.add_(1.0)
        
        if pop_weights is not None:
            # Cleanly retrieve population size P dynamically from object-identity keyed dict
            P = next(iter(pop_weights.values())).shape[0]
            B = env_input.shape[0] // P
            input_dim = env_input.shape[1]
            num_input_nodes = math.ceil(input_dim / self.D)
            
            # Map batched model parameter values cleanly using parameter identity keys
            prediction = self.w1(self.state_pop, self.step_counter, pop_weights.get(self.w1.weight))
            router_logits = self.router(self.state_pop, self.step_counter, pop_weights.get(self.router.weight))
            
            curiosity_bias = F.relu(self.lp_ema_pop).unsqueeze(2) * 5.0
            route_logits = router_logits + curiosity_bias
            route, aux, mask = sparse_route(route_logits, self.N, self.sprs)
            
            current_var = route.var(dim=[-1], keepdim=True)
            vd = F.relu(self.a_variance.unsqueeze(0) - current_var)
            
            V = self.output_pop
            base_noise = torch.sqrt(vd) * torch.randn_like(V)
            
            boredom_scale = torch.exp(-50.0 * F.relu(self.lp_ema_pop)).unsqueeze(-1)
            boredom_noise = base_noise * boredom_scale
            
            target = torch.matmul(route, V) + boredom_noise
            target = utils.softsign(target)
            
            pad_len = num_input_nodes * self.D - input_dim
            if pad_len > 0:
                env_input_padded = F.pad(env_input, (0, pad_len))
            else:
                env_input_padded = env_input
                
            env_input_reshaped = env_input_padded.view(P, B, num_input_nodes, self.D)
            
            if num_input_nodes < self.N:
                target = torch.cat([env_input_reshaped, target[:, :, num_input_nodes:, :]], dim=2)
            else:
                target = env_input_reshaped
                
            loss_w1 = F.mse_loss(prediction, target, reduction='none')
            
            with torch.no_grad():
                current_error = loss_w1.mean(dim=-1).detach()
                current_lp = self.error_ema_pop - current_error
                self.error_ema_pop = self.error_ema_pop * (1.0 - self.lp_alpha) + current_error * self.lp_alpha
                self.lp_ema_pop = self.lp_ema_pop * (1.0 - self.lp_alpha) + current_lp * self.lp_alpha
                
            total_loss = ((loss_w1).sum() + aux) / (P * B)
            
            boredom_scale = torch.exp(-50.0 * torch.relu(self.lp_ema_pop)).unsqueeze(-1)
            errm = torch.sqrt(boredom_scale) * 0.01
            error_pass = torch.randn_like(target) * errm
            target = target + error_pass
            
            self.output_pop = prediction.detach()
            self.state_pop = target.detach()
            
            candidate_losses = loss_w1.mean(dim=[1, 2, 3])
            return total_loss, candidate_losses
            
        else:
            batch_size, input_dim = env_input.shape
            num_input_nodes = math.ceil(input_dim / self.D)
            
            prediction = self.w1(self.state, self.step_counter) 
            route_logits = self.router(self.state, self.step_counter)
            
            curiosity_bias = F.relu(self.lp_ema).unsqueeze(1) * 5.0 
            route_logits = route_logits + curiosity_bias
            route, aux, mask = sparse_route(route_logits, self.N, self.sprs)
            
            current_var = route.var(dim=[-1], keepdim=True)
            vd = F.relu(self.a_variance - current_var)
            
            V = self.output
            base_noise = torch.sqrt(vd) * torch.randn_like(V)
            
            boredom_scale = torch.exp(-50.0 * F.relu(self.lp_ema)).unsqueeze(-1) 
            boredom_noise = base_noise * boredom_scale
            
            target = torch.matmul(route, V) + boredom_noise
            target = utils.softsign(target)

            pad_len = num_input_nodes * self.D - input_dim
            if pad_len > 0:
                env_input_padded = F.pad(env_input, (0, pad_len))
            else:
                env_input_padded = env_input
                
            env_input_reshaped = env_input_padded.view(batch_size, num_input_nodes, self.D)
            
            if num_input_nodes < self.N:
                target = torch.cat([env_input_reshaped, target[:, num_input_nodes:, :]], dim=1)
            else:
                target = env_input_reshaped

            loss_w1 = F.mse_loss(prediction, target, reduction='none')

            with torch.no_grad():
                current_error = loss_w1.mean(dim=-1).detach()
                current_lp = self.error_ema - current_error
                self.error_ema = self.error_ema * (1.0 - self.lp_alpha) + current_error * self.lp_alpha
                self.lp_ema = self.lp_ema * (1.0 - self.lp_alpha) + current_lp * self.lp_alpha

            total_loss = ((loss_w1).sum() + aux) / batch_size
            
            boredom_scale = torch.exp(-50.0 * torch.relu(self.lp_ema)).unsqueeze(-1) 
            errm = torch.sqrt(boredom_scale) * 0.01
            error_pass = torch.randn_like(target) * errm
            target = target + error_pass

            self.output = prediction.detach()
            self.state = target.detach()

            return total_loss


# ==========================================
# STATE UTILITIES & EXTRINSICS
# ==========================================
def reset_model_buffers(model):
    model.state.copy_(1e-5 * torch.randn_like(model.state))
    model.output.copy_(1e-5 * torch.randn_like(model.output))
    model.error_ema.copy_(torch.ones_like(model.error_ema))
    model.lp_ema.copy_(torch.zeros_like(model.lp_ema))
    model.step_counter.zero_()


def generate_low_rank_noise(shape, rank=4, device='cuda'):
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
# TRAINING PROCEDURES & OPTIMIZERS
# ==========================================
def run_gradient_descent_training(model, env, exec_config):
    print("--- Training via Step-level Adam Backpropagation ---")
    start_time = time.perf_counter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=exec_config.gd_lr, weight_decay=1e-4)
    reset_model_buffers(model)
    
    obs = env.reset()
    losses = []
    
    for step in range(exec_config.gd_total_steps):
        loss = model(obs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss)
        
        logits = model.state.sum(dim=1)[:, :4]
        actions = torch.argmax(logits, dim=-1)
        obs = env.step(actions)
            
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"GD Training completed in {elapsed:.2f} seconds ({elapsed/exec_config.gd_total_steps*1000:.2f} ms/step).")
    print(f"Initial 10-step average loss: {sum(losses[:10])/10:.4f}")
    print(f"Final 10-step average loss:   {sum(losses[-10:])/10:.4f}\n")


class EGGROLLSurpriseOptimizer:
    def __init__(self, model, exec_config):
        self.model = model
        self.exec_config = exec_config
        self.profiler = CUDAProfiler()
        
        device = next(model.parameters()).device
        
        # Keep master parameters indexed cleanly by Parameter object references
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

    def step_generation(self, env):
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
            
            # Map batched parameters cleanly using Parameter instance identity keys
            pop_weights[p] = torch.cat([w_pos, w_neg], dim=0)
        self.profiler.stop("1. Low-Rank Noise Generation")
            
        self.profiler.start("2. Vectorized Environment Init")
        grid_tiled = env.grid[:B].repeat(P, 1, 1)
        vec_env = MazeEnvTensor(batch_size=P * B, size=env.size, grid=grid_tiled, food_density=env.food_density, device=env.device)
        self.model.init_vectorized_states(P, B)
        obs = vec_env.reset()
        self.profiler.stop("2. Vectorized Environment Init")
        
        candidate_surprises = torch.zeros(P, device=env.device)
        
        self.profiler.start("3. Complete Rollout Loop")
        with torch.no_grad():
            for _ in range(self.exec_config.max_episode_steps):
                self.profiler.start("3a. Model Forward Pass (Batched)")
                _, step_surprises = self.model(obs, pop_weights=pop_weights)
                self.profiler.stop("3a. Model Forward Pass (Batched)")
                
                candidate_surprises += step_surprises
                
                self.profiler.start("3b. Action Extraction")
                logits = self.model.state_pop.sum(dim=2)[:, :, :4]
                actions = torch.argmax(logits, dim=-1).view(-1)
                self.profiler.stop("3b. Action Extraction")
                
                self.profiler.start("3c. Vectorized Environment Step")
                obs = vec_env.step(actions)
                self.profiler.stop("3c. Vectorized Environment Step")
        self.profiler.stop("3. Complete Rollout Loop")

        self.profiler.start("4. Parameter Update & Math Reduction")
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
        
        return mean_s.detach()


class InterspersedHybridOptimizer:
    def __init__(self, model, exec_config, lr=0.01, es_sigma=0.03, es_pop_size=4, es_rank=4):
        self.model = model
        self.exec_config = exec_config
        self.es_sigma = es_sigma
        self.es_pop_size = es_pop_size
        self.es_rank = es_rank
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.profiler = CUDAProfiler()

    def step(self, obs, env, step_idx, es_check_interval=10):
        self.profiler.start("Hybrid 1. Standard GD Backprop Step")
        loss = self.model(obs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        current_loss = loss
        self.profiler.stop("Hybrid 1. Standard GD Backprop Step")
        
        if step_idx % es_check_interval == 0 and step_idx > 0:
            self.profiler.start("Hybrid 2. Parallel Lookahead Search")
            best_fitness, best_params = self._evaluate_vectorized_lookahead(env)
            if best_params is not None:
                for p, param_val in best_params.items():
                    p.data.copy_(param_val)
                self.optimizer.state.clear()
            self.profiler.stop("Hybrid 2. Parallel Lookahead Search")
                
        return current_loss

    def _evaluate_vectorized_lookahead(self, env):
        P = self.es_pop_size
        half_P = P // 2
        B = env.batch_size
        
        self.profiler.start("Hybrid 2a. Setup & Noise Generation")
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
            
        grid_tiled = env.grid[:B].repeat(P, 1, 1)
        vec_env = MazeEnvTensor(batch_size=P * B, size=env.size, grid=grid_tiled, food_density=env.food_density, device=env.device)
        vec_env.restore_state((
            env.agent_pos.repeat(P, 1),
            env.food_grid.repeat(P, 1, 1),
            env.hunger.repeat(P),
            env.steps.repeat(P)
        ))
        
        state_backup = self.model.state.clone()
        output_backup = self.model.output.clone()
        error_ema_backup = self.model.error_ema.clone()
        lp_ema_backup = self.model.lp_ema.clone()
        step_counter_backup = self.model.step_counter.clone()
        
        self.model.init_vectorized_states(P, B)
        obs = vec_env._get_obs()
        self.profiler.stop("Hybrid 2a. Setup & Noise Generation")
        
        total_prediction_losses = torch.zeros(P, device=env.device)
        total_food_eaten = torch.zeros(P, device=env.device)
        reached_goal_count = torch.zeros(P, device=env.device)
        total_hunger = torch.zeros(P, device=env.device)
        steps_evaluated = 15
        
        self.profiler.start("Hybrid 2b. Rollout Loop (15 steps)")
        with torch.no_grad():
            for _ in range(steps_evaluated):
                self.profiler.start("Hybrid 2b1. Model Forward Pass")
                _, candidate_losses = self.model(obs, pop_weights=pop_weights)
                self.profiler.stop("Hybrid 2b1. Model Forward Pass")
                
                total_prediction_losses += candidate_losses
                
                self.profiler.start("Hybrid 2b2. Action Selection")
                logits = self.model.state_pop.sum(dim=2)[:, :, :4]
                actions = torch.argmax(logits, dim=-1).view(-1)
                self.profiler.stop("Hybrid 2b2. Action Selection")
                
                self.profiler.start("Hybrid 2b3. Vectorized Env Step")
                food_before = (vec_env.food_grid == 0.0).view(P, B, -1).sum(dim=[1, 2]).float()
                obs = vec_env.step(actions)
                food_after = (vec_env.food_grid == 0.0).view(P, B, -1).sum(dim=[1, 2]).float()
                self.profiler.stop("Hybrid 2b3. Vectorized Env Step")
                
                total_food_eaten += (food_after - food_before)
                total_hunger += vec_env.hunger.view(P, B).sum(dim=1)
                
                batch_idx_vec = torch.arange(P * B, device=vec_env.device)
                at_goal = (vec_env.grid[batch_idx_vec, vec_env.agent_pos[:, 0], vec_env.agent_pos[:, 1]] == 2.0)
                reached_goal_count += at_goal.view(P, B).sum(dim=1).float()
        self.profiler.stop("Hybrid 2b. Rollout Loop (15 steps)")
                
        self.profiler.start("Hybrid 2c. Reduce & State Restorations")
        self.model.state.copy_(state_backup)
        self.model.output.copy_(output_backup)
        self.model.error_ema.copy_(error_ema_backup)
        self.model.lp_ema.copy_(lp_ema_backup)
        self.model.step_counter.copy_(step_counter_backup)
        
        avg_pred_loss = total_prediction_losses / steps_evaluated
        avg_phys_success = ((total_food_eaten * 2.5) + (reached_goal_count * 15.0) - (total_hunger * 0.05)) / B
        composite_fitnesses = -avg_pred_loss + 0.20 * avg_phys_success
        
        baseline_fitness = self._evaluate_single_fitness(env)
        
        best_candidate_idx = torch.argmax(composite_fitnesses)
        best_fitness = composite_fitnesses[best_candidate_idx]
        self.profiler.stop("Hybrid 2c. Reduce & State Restorations")
        
        if best_fitness > baseline_fitness:
            best_p = {}
            for p in master_params.keys():
                best_p[p] = pop_weights[p][best_candidate_idx]
            return best_fitness, best_p
        return baseline_fitness, None

    def _evaluate_single_fitness(self, env):
        state_backup = self.model.state.clone()
        output_backup = self.model.output.clone()
        error_ema_backup = self.model.error_ema.clone()
        lp_ema_backup = self.model.lp_ema.clone()
        step_counter_backup = self.model.step_counter.clone()
        
        env_state = env.get_state()
        obs = env._get_obs()
        
        total_pred_loss = 0.0
        total_food_eaten = 0.0
        reached_goal_count = 0.0
        total_hunger = 0.0
        steps_evaluated = 15
        
        batch_idx = torch.arange(env.batch_size, device=env.device)
        
        with torch.no_grad():
            for _ in range(steps_evaluated):
                loss = self.model(obs)
                total_pred_loss += loss
                
                logits = self.model.state.sum(dim=1)[:, :4]
                actions = torch.argmax(logits, dim=-1)
                
                food_before = (env.food_grid == 0.0).sum()
                obs = env.step(actions)
                food_after = (env.food_grid == 0.0).sum()
                
                total_food_eaten += (food_after - food_before)
                total_hunger += env.hunger.sum()
                
                at_goal = (env.grid[batch_idx, env.agent_pos[:, 0], env.agent_pos[:, 1]] == 2.0)
                reached_goal_count += at_goal.sum()
                
        self.model.state.copy_(state_backup)
        self.model.output.copy_(output_backup)
        self.model.error_ema.copy_(error_ema_backup)
        self.model.lp_ema.copy_(lp_ema_backup)
        self.model.step_counter.copy_(step_counter_backup)
        env.restore_state(env_state)
        
        avg_pred_loss = total_pred_loss / steps_evaluated
        avg_phys_success = ((total_food_eaten * 2.5) + (reached_goal_count * 15.0) - (total_hunger * 0.05)) / env.batch_size
        return -avg_pred_loss + 0.20 * avg_phys_success


def run_interspersed_hybrid_training(model, env, exec_config):
    print("--- Training via Interspersed Hybrid (GD with Parallel ES Lookaheads) ---")
    start_time = time.perf_counter()
    
    hybrid_opt = InterspersedHybridOptimizer(
        model, 
        exec_config, 
        lr=exec_config.gd_lr, 
        es_sigma=0.03,
        es_pop_size=4,
        es_rank=4
    )
    reset_model_buffers(model)
    
    obs = env.reset()
    losses = []
    
    for step in range(exec_config.gd_total_steps):
        loss_val = hybrid_opt.step(obs, env, step, es_check_interval=10)
        losses.append(loss_val)
        
        logits = model.state.sum(dim=1)[:, :4]
        actions = torch.argmax(logits, dim=-1)
        obs = env.step(actions)
                
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"Hybrid Interspersed Training completed in {elapsed:.2f} seconds ({elapsed/exec_config.gd_total_steps*1000:.2f} ms/step).")
    print(f"Initial 10-step average loss: {sum(losses[:10])/10:.4f}")
    print(f"Final 10-step average loss:   {sum(losses[-10:])/10:.4f}\n")
    
    hybrid_opt.profiler.print_report("HYBRID OPTIMIZATION BREAKDOWN")


# ==========================================
# EXPANDED HYPERPARAMETER TUNER
# ==========================================
class HyperParamTuner:
    def __init__(self, device, exec_config, generations=3, pop_size=16, sigma=0.08):
        self.device = device
        self.exec_config = exec_config
        self.generations = generations
        self.pop_size = pop_size
        self.sigma = sigma
        self.base_cfg = HyperParamsConfig(device)
        
        # Instantiate model once
        self.eval_model = StandardNodeNet3(batch_size=16, num_nodes=8, dim=16, device=self.device).to(self.device)
        
        # 1. Warm up and collect a static offline transition dataset once
        print("Collecting static offline transition dataset for benchmarking...")
        self.offline_dataset = self._collect_static_dataset(steps=40)
        print(f"Dataset collected. Shape: {self.offline_dataset.shape} (Steps, Batch, Obs_Dim)\n")

    def _collect_static_dataset(self, steps=40):
        # We collect observations from a standard environment step sequence
        env = MazeEnvTensor(batch_size=16, size=self.exec_config.maze_size, food_density=0.15, device=self.device)
        obs = env.reset()
        dataset = [obs.clone()]
        
        for _ in range(steps):
            # Take random exploratory actions to generate a diverse transition map
            actions = torch.randint(0, 4, (16,), device=self.device)
            obs = env.step(actions)
            dataset.append(obs.clone())
            
        return torch.stack(dataset, dim=0) # Shape: [Steps+1, 16, Obs_Dim]

    def _evaluate_config_offline(self, hyper_cfg):
        """
        Blazing fast offline evaluation. Measures how quickly the world model 
        minimizes transition prediction error on the static GPU dataset.
        """
        # Apply updated hyperparameters
        self.eval_model.apply_hyperparams(hyper_cfg.to_tensor_config())
        reset_model_buffers(self.eval_model)
        
        # Optimize with a simple, high-speed optimizer over the static sequence
        optimizer = torch.optim.AdamW(self.eval_model.parameters(), lr=0.01)
        
        total_loss = 0.0
        steps_evaluated = len(self.offline_dataset) - 1
        
        # We run 3 fast optimization passes over the static sequence
        for epoch in range(2):
            for t in range(steps_evaluated):
                obs = self.offline_dataset[t]
                target = self.offline_dataset[t + 1]
                
                # Model forward pass
                loss = self.eval_model(obs)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if epoch == 1: # Record loss on the second epoch to measure learning speed
                    total_loss += loss.detach()
                    
        # Return a fitness score (lower predictive loss is better)
        return - (total_loss.item() / steps_evaluated)

    def tune(self):
        print("=== Tuning Architectural Hyperparameters via Offline Benchmark ===")
        keys = list(self.base_cfg.params.keys())
        
        best_overall_score = -float('inf')
        best_overall_params = dict(self.base_cfg.params)
        
        start_time = time.perf_counter()
        
        for gen in range(1, self.generations + 1):
            noises = []
            scores = []
            
            # Evaluate base configuration
            base_score = self._evaluate_config_offline(self.base_cfg)
            print(f"Gen {gen:02d} | Base Config Score (Neg Pred Loss): {base_score:.4f}")
            
            if base_score > best_overall_score:
                best_overall_score = base_score
                best_overall_params = dict(self.base_cfg.params)
                
            for _ in range(self.pop_size):
                noise = {k: random.normalvariate(0, 1) for k in keys}
                noises.append(noise)
                
                perturbed_cfg = HyperParamsConfig(self.device)
                for k in keys:
                    val = self.base_cfg.params[k] + self.sigma * noise[k]
                    # Clamp constraints
                    if k in ['untanh', 'w_scale', 'speed', 'rew_sens']:
                        val = max(0.1, min(val, 3.0))
                    elif k in ['ema_speed', 'sprs', 'c_moment']:
                        val = max(0.01, min(val, 0.99))
                    elif k == 'noise_scale':
                        val = max(0.0, min(val, 0.5))
                    elif k == 'width':
                        val = max(0.001, min(val, 0.1))
                    elif k == 'a_variance':
                        val = max(0.0001, min(val, 0.05))
                    perturbed_cfg.params[k] = val
                    
                score = self._evaluate_config_offline(perturbed_cfg)
                scores.append(score)
                
                if score > best_overall_score:
                    best_overall_score = score
                    best_overall_params = dict(perturbed_cfg.params)
                
            mean_score = sum(scores) / len(scores)
            std_score = math.sqrt(sum((s - mean_score)**2 for s in scores) + 1e-8)
            
            # Apply parameter updates
            for k in keys:
                grad = 0.0
                for i in range(self.pop_size):
                    norm_fitness = (scores[i] - mean_score) / std_score
                    grad += norm_fitness * noises[i][k]
                grad /= (self.pop_size * self.sigma)
                
                self.base_cfg.params[k] += 0.05 * grad
                
                # Apply boundary clamps
                if k in ['untanh', 'w_scale', 'speed', 'rew_sens']:
                    self.base_cfg.params[k] = max(0.1, min(self.base_cfg.params[k], 3.0))
                elif k in ['ema_speed', 'sprs', 'c_moment']:
                    self.base_cfg.params[k] = max(0.01, min(self.base_cfg.params[k], 0.99))
                elif k == 'noise_scale':
                    self.base_cfg.params[k] = max(0.0, min(self.base_cfg.params[k], 0.5))
                elif k == 'width':
                    self.base_cfg.params[k] = max(0.001, min(self.base_cfg.params[k], 0.1))
                elif k == 'a_variance':
                    self.base_cfg.params[k] = max(0.0001, min(self.base_cfg.params[k], 0.05))

        elapsed = time.perf_counter() - start_time
        print(f"\nTuning completed in {elapsed:.2f} seconds!")
        print("--- Optimized Hyperparameters Found ---")
        self.base_cfg.params = best_overall_params
        for k, v in self.base_cfg.params.items():
            print(f"  {k:15s}: {v:.6f}")
        print(f"  Best evaluation score reached: {best_overall_score:.4f}")
        print("========================================\n")
        return self.base_cfg

# ==========================================
# VISUALIZATION & EVALUATION FUNCTIONS
# ==========================================
def render_trajectory_tensor(env, path, batch_idx=0):
    grid = env.grid[batch_idx].cpu()
    food_grid = env.food_grid[batch_idx].cpu()
    size = env.size
    
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


def find_best_path(paths, env):
    best_idx = 0
    best_score = (float('inf'), float('inf'))
    
    for i, path in enumerate(paths):
        # Dynamically fetch the random goal location of batch element i
        goal_idx = torch.nonzero(env.grid[i] == 2.0)
        goal_pos = (int(goal_idx[0, 0]), int(goal_idx[0, 1]))
        
        reached = False
        min_dist = float('inf')
        steps_to_reach = float('inf')
        
        for step_idx, pos in enumerate(path):
            dist = abs(pos[0] - goal_pos[0]) + abs(pos[1] - goal_pos[1])
            if dist < min_dist:
                min_dist = dist
            if dist == 0 and not reached:
                reached = True
                steps_to_reach = step_idx + 1
        
        score = (min_dist, steps_to_reach)
        if score < best_score:
            best_score = score
            best_idx = i
            
    return best_idx


def run_evaluation(model, env, exec_config, label):
    """Runs a single clean 40-step evaluation episode and records exact positions."""
    print(f"=== Final Evaluation for {label} ===")
    reset_model_buffers(model)
    obs = env.reset()
    
    total_food_eaten = torch.zeros(env.batch_size, device=env.device)
    goals_reached = torch.zeros(env.batch_size, device=env.device)
    ever_reached_goal = torch.zeros(env.batch_size, dtype=torch.bool, device=env.device)
    
    paths = [[] for _ in range(env.batch_size)]
    
    with torch.no_grad():
        for _ in range(exec_config.max_episode_steps):
            pos_cpu = env.agent_pos.cpu().numpy()
            for i in range(env.batch_size):
                paths[i].append((int(pos_cpu[i, 0]), int(pos_cpu[i, 1])))
                
            _ = model(obs)
            logits = model.state.sum(dim=1)[:, :4]
            actions = torch.argmax(logits, dim=-1)
            
            obs = env.step(actions)
            
            total_food_eaten += env.last_food_eaten.float()
            ever_reached_goal = ever_reached_goal | env.last_at_goal
            goals_reached += env.last_at_goal.float()
            
    avg_food = total_food_eaten.mean()
    success_rate = ever_reached_goal.float().mean() * 100.0
    avg_goals = goals_reached.mean()
    
    print(f"  Avg Food Eaten per Episode:        {avg_food:.2f}")
    print(f"  Goal Success Rate (Exit Reached):  {success_rate:.1f}%")
    print(f"  Avg Goal Reaches per Episode:      {avg_goals:.2f}\n")
    return paths


# ==========================================
# EXECUTION ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    exec_config = ExecutionConfig()
    
    # --- STEP A: Tuning Architectural Hyperparameters via JIT-Compiled ES ---
    tuner = HyperParamTuner(device, exec_config, generations=3, pop_size=16)
    optimized_hyper_config = tuner.tune()#exec_config.maze_size)
    tensor_hyper_config = optimized_hyper_config.to_tensor_config()
    
    # --- STEP B: Initialize Batched GPU Environments ---
    # Use standard 15% sparse food layouts for trials
    ref_env = MazeEnvTensor(batch_size=1, size=exec_config.maze_size, food_density=0.15, device=device)
    ref_grid = ref_env.grid[0]
    grid_batched = ref_grid.unsqueeze(0).expand(exec_config.batch_size, -1, -1)
    
    envs_gd = MazeEnvTensor(batch_size=exec_config.batch_size, size=exec_config.maze_size, grid=grid_batched, food_density=0.15, device=device)
    envs_es = MazeEnvTensor(batch_size=exec_config.batch_size, size=exec_config.maze_size, grid=grid_batched, food_density=0.15, device=device)
    envs_hybrid = MazeEnvTensor(batch_size=exec_config.batch_size, size=exec_config.maze_size, grid=grid_batched, food_density=0.15, device=device)
    
    # --- TRIAL 1: Backpropagation (AdamW in High-speed Eager Mode) ---
    model_gd = StandardNodeNet3(
        batch_size=exec_config.batch_size, 
        num_nodes=exec_config.num_nodes, 
        dim=exec_config.dim, 
        device=device
    ).to(device)
    model_gd.apply_hyperparams(tensor_hyper_config)
    
    run_gradient_descent_training(model_gd, envs_gd, exec_config)
    gd_eval_paths = run_evaluation(model_gd, envs_gd, exec_config, "Gradient Descent (Trial 1)")
    
    best_gd_idx = find_best_path(gd_eval_paths, envs_gd)
    print(f"Rendering Best Path for GD (Batch Element {best_gd_idx + 1}):")
    render_trajectory_tensor(envs_gd, gd_eval_paths[best_gd_idx], batch_idx=best_gd_idx)
    print()
    
    print("\n" + "="*50 + "\n")
    
    # --- TRIAL 2: Parallel EGGROLL ES Evolution (with Profiler) ---
    model_es = StandardNodeNet3(
        batch_size=exec_config.batch_size, 
        num_nodes=exec_config.num_nodes, 
        dim=exec_config.dim, 
        device=device
    ).to(device)
    model_es.apply_hyperparams(tensor_hyper_config)
    
    es_opt = EGGROLLSurpriseOptimizer(model_es, exec_config)
    es_start = time.perf_counter()
    for gen in range(1, exec_config.es_generations + 1):
        mean_surprise = es_opt.step_generation(envs_es)
        if gen % 10 == 0 or gen == 1:
            print(f"Generation {gen:02d} | Cumulative Surprise: {mean_surprise:.2f}")
    es_elapsed = time.perf_counter() - es_start
    print(f"ES Training completed in {es_elapsed:.2f} seconds.\n")
    
    es_opt.profiler.print_report("EGGROLL ES OPTIMIZATION BREAKDOWN")
            
    es_eval_paths = run_evaluation(model_es, envs_es, exec_config, "EGGROLL ES (Trial 2)")
    best_es_idx = find_best_path(es_eval_paths, envs_es)
    print(f"Rendering Best Path for ES (Batch Element {best_es_idx + 1}):")
    render_trajectory_tensor(envs_es, es_eval_paths[best_es_idx], batch_idx=best_es_idx)
    print()
        
    print("\n" + "="*50 + "\n")
    
    # --- TRIAL 3: Interspersed Hybrid (GD with Parallel ES Lookaheads) ---
    model_hybrid = StandardNodeNet3(
        batch_size=exec_config.batch_size, 
        num_nodes=exec_config.num_nodes, 
        dim=exec_config.dim, 
        device=device
    ).to(device)
    model_hybrid.apply_hyperparams(tensor_hyper_config)
    
    run_interspersed_hybrid_training(model_hybrid, envs_hybrid, exec_config)
    hybrid_eval_paths = run_evaluation(model_hybrid, envs_hybrid, exec_config, "Interspersed Hybrid (Trial 3)")
    
    best_hybrid_idx = find_best_path(hybrid_eval_paths, envs_hybrid)
    print(f"Rendering Best Path for Hybrid (Batch Element {best_hybrid_idx + 1}):")
    render_trajectory_tensor(envs_hybrid, hybrid_eval_paths[best_hybrid_idx], batch_idx=best_hybrid_idx)
    print()