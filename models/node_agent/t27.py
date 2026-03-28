import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
import gymnasium as gym
from training_tools.sweep import Sweep3D
from training_tools.sweep import Sweep2D
import parts.utils as utils
import minigrid
from minigrid.wrappers import FlatObsWrapper

# ==========================================
# 0. SETUP
# ==========================================

torch.set_float32_matmul_precision('high')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware: {DEVICE}")

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
    def __init__(self, active_tensors, relations, N, D, batch_size, device=DEVICE):
        self.N_val = float(N)
        self.D_val = float(D)
        self.batch_size = batch_size
        self.device = device
        
        self.defaults = {
            'speed': 0.5,
            'width': 0.01,
            'lateral_decay': 0.01,
            "EMA_SPEED": 0.20310950837532799,
            "FIXED_REWARD_SENSITIVITY": 1.360218046607915,
            "LR": 0.005720721430830338,
            "MOMENTUM": 0.999,
            "weight_decay": 0.021698750753204772,
            "NOISE_SCALE": 1.5,
            "c_moment": 0.8021549684435622,
            "w_scale": 0.6340345603476942,
            "sharpness": 1.0390895413424572,
            "a_variance": 0.002633467302632888
        }
        self.update(active_tensors, relations)

    def update(self, active_tensors, relations):
        self.tensors = active_tensors
        self.relations = relations
        self.cache = {
            'N': torch.full((self.batch_size,), self.N_val, device=self.device),
            'D': torch.full((self.batch_size,), self.D_val, device=self.device)
        }
    
    def resolve_all(self):
        res = {}
        for k in FULL_SPACE.keys():
            res[k] = self.get(k, 1)
        for k in self.defaults.keys():
            res[k] = self.get(k, 1)
        return res
    
    def get(self, name, ndim):
        if name in self.cache:
            val = self.cache[name]
        elif name in self.tensors:
            val = self.tensors[name]
            self.cache[name] = val
        elif name in self.relations:
            rel = self.relations[name]
            eq_type = rel.get('type', 'linear')
            
            if isinstance(rel['master'], list):
                m1_val = self.get(rel['master'][0], 1).flatten()
                m2_val = self.get(rel['master'][1], 1).flatten()
                
                if eq_type == 'multi_power':
                    val = rel['a'] * torch.pow(m1_val + 1e-8, rel['b']) * torch.pow(m2_val + 1e-8, rel['c'])
                else: 
                    val = rel['a'] * m1_val + rel['b'] * m2_val + rel['c']
            else:
                master_val = self.get(rel['master'], 1).flatten() 
                if eq_type == 'power':
                    val = rel['a'] * torch.pow(master_val + 1e-8, rel['b'])
                elif eq_type == 'quadratic':
                    val = rel['a'] * (master_val**2) + rel['b'] * master_val + rel['c']
                else: 
                    val = rel['a'] * master_val + rel['b']
                
            if name in FULL_SPACE:
                val = torch.clamp(val, FULL_SPACE[name]['low'], FULL_SPACE[name]['high'])
            self.cache[name] = val
        else:
            val = torch.tensor(self.defaults.get(name, 1.0), device=self.device).expand(self.cache['N'].shape[0])

        return val.view(-1, *([1] * (ndim - 1)))

class TriadPiece(nn.Module):
    def __init__(self, batch_size, num_nodes, dim):
        super().__init__() 
        self.D = dim
        
        base_weight = torch.randn(1, num_nodes, dim, dim, device=DEVICE) / math.sqrt(self.D)
        self.register_buffer('weight', base_weight.expand(batch_size, -1, -1, -1).clone())
        
        self.register_buffer('g_ema', torch.randn_like(self.weight) * 0.01) 
        self.register_buffer('error_ema', 0.01 + 0.01 * torch.randn(batch_size, num_nodes, dim, device=DEVICE))
        
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

        base_weight = torch.zeros(1, self.weight.shape[1], self.D, self.D, device=DEVICE) / math.sqrt(self.D)
        self.weight.data.copy_(base_weight.expand_as(self.weight))
        self.weight[...,:1] = 1
        
        self.weight.data.copy_(utils.rms_norm(self.weight.data, dim=[-1,-2]) * self.hp_w_scale) 
        
        self.g_ema.normal_().mul_(0.01)
        self.error_ema.normal_().mul_(0.01).add_(0.01)

    def forward(self, x):
        raw_pred = (x.unsqueeze(2) @ self.weight).squeeze(2)
        return utils.buntanh(raw_pred, self.hp_untanh, math.sqrt(2 + math.sqrt(2)))
    
    def get_gradient(self, layer_input, error_signal):
        ema_speed = self.hp_ema_speed
        rew_sens = self.hp_rew_sens
        n_scale = self.hp_noise_scale
        
        self.error_ema.mul_(1.0 - ema_speed).add_(ema_speed * error_signal)
        E_curr = F.softmax(error_signal, dim=-1)
        advantage = E_curr - self.error_ema
        plasticity = 1.0 + utils.rms_norm(advantage.unsqueeze(3) * rew_sens)  
        
        noisy_state = utils.noised_input(layer_input, error_signal, n_scale)
        lg1 = utils.rms_norm(-torch.einsum('bni,bnj->bnij', error_signal, noisy_state))
        return (plasticity * lg1) 
    
    def step(self, grad, step_counter):
        wd = self.hp_wd
        grad = grad - (wd * self.weight)
        
        speed = self.hp_speed 
        width = self.hp_width 
        center = (step_counter * speed) % self.D
        g_mask = utils.wrapping_gaussian_kernel2(grad, center, width)
        grad = grad * g_mask
        
        momentum = self.hp_momentum 
        lr = self.hp_lr 

        self.g_ema.mul_(momentum).add_((1.0 - momentum) * grad)
        self.weight.add_(lr * self.g_ema)
        
        w_scale = self.hp_w_scale
        self.weight.data.copy_(utils.rms_norm(self.weight.data, dim=[-1,-2]) * w_scale )

class PureTriadicBrain(nn.Module):
    def __init__(self, batch_size, num_nodes, dim, hp_manager):
        super().__init__()
        self.N = num_nodes
        self.D = dim
        self.H = dim//4
        self.head_dim = 4
        self.batch_size = batch_size
        self.hpm = hp_manager
        
        self.w1 = TriadPiece(batch_size, num_nodes, dim)
        self.w2 = TriadPiece(batch_size, num_nodes, dim)
        self.w3 = TriadPiece(batch_size, num_nodes, dim)
        
        self.register_buffer('state', 1e-5 * torch.randn(batch_size, self.N, dim, device=DEVICE))
        self.register_buffer('output', 1e-5 * torch.randn(batch_size, self.N, dim, device=DEVICE))
        self.register_buffer('step_counter', torch.tensor(0.0, device=DEVICE))
        
        
        self.register_buffer('A', torch.randn(batch_size, self.H, self.N, self.N, device=DEVICE))
        self.register_buffer('A_ema', torch.zeros(batch_size, self.H, self.N, self.N, device=DEVICE))
        self.A.data.copy_(F.softmax(self.A, dim=-1))
        
        self.register_buffer('A_mask', torch.ones_like(self.A))
        self.current_input_nodes = 0
    def reset(self, active_tensors, relations):
        self.hpm.update(active_tensors, relations)
        hps = self.hpm.resolve_all()
        
        U = hps['UNTANH']
        if 'w_scale' not in active_tensors:
            w_scale = 0.95 / (U**2)
            w_scale_sub = 1.08 * U + 0.28
            w_scale = torch.where(U < 1.0, w_scale_sub, w_scale)
            hps['w_scale'] = torch.clamp(w_scale, 0.001, 10.0)

        # Add an extra dimension for Heads broadcasting: view(-1, 1, 1, 1)
        self.hp_c_moment = hps['c_moment'].view(-1, 1, 1, 1)
        self.hp_a_variance = hps['a_variance'].view(-1, 1, 1, 1)
        self.hp_untanh = hps['UNTANH'].view(-1, 1, 1)

        self.w1.reset(hps)
        self.w2.reset(hps)
        self.w3.reset(hps)
        
        self.state.normal_().mul_(1e-5)
        self.output.normal_().mul_(1e-5)
        self.step_counter.fill_(0.0)
        
        self.A.normal_()
        self.A_ema.zero_()
        self.A.data.copy_(F.softmax(self.A, dim=-1))
        
        self.A_mask.fill_(1.0)
        self.current_input_nodes = 0

    def forward(self, env_input):
        
        self.step_counter.add_(1.0)
        
        batch_size, input_dim = env_input.shape
        num_input_nodes = math.ceil(input_dim / self.D)
        
        if num_input_nodes >= self.N:
            raise ValueError(f"Input requires {num_input_nodes} nodes, but brain only has {self.N} nodes. Increase N or D.")
             
        if num_input_nodes != self.current_input_nodes:
            self.A_mask.fill_(1.0)
            self.A_mask[:, :, :num_input_nodes, :] = 0.0  # Note the extra ':' for Heads
            self.A_ema[:, :, num_input_nodes, :num_input_nodes] += 3.0 / num_input_nodes
            self.current_input_nodes = num_input_nodes

        queries    = self.w3(self.state)
        keys       = self.w2(self.state) 
        prediction = self.w1(self.state) 

        Q = queries.view(batch_size, self.N, self.H, self.head_dim).transpose(1, 2)
        K = keys.view(batch_size, self.N, self.H, self.head_dim).transpose(1, 2)

        raw_A = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        raw_A = utils.rms_norm(raw_A, dim=[-1, -2]) 

        is_first = (self.step_counter == 1.0).float().view(1, 1, 1, 1)
        new_A_ema = raw_A * is_first + (self.A_ema * self.hp_c_moment + raw_A * (1.0 - self.hp_c_moment)) * (1.0 - is_first)
        self.A_ema.copy_(new_A_ema * self.A_mask)
        
        raw_A = self.A_ema 
        
        B_H_shape = (batch_size * self.H, self.N, self.N)
        raw_A_3D = raw_A.reshape(B_H_shape)
        raw_A_3D = utils.sinkhorn_knopp(raw_A_3D)
        raw_A = raw_A_3D.reshape(batch_size, self.H, self.N, self.N)

        uniform_baseline = 1.0 / (self.N + 1.0)
        sparse_A = torch.where(raw_A > uniform_baseline, raw_A, 0.0)
        
        sparse_A_3D = sparse_A.reshape(B_H_shape)
        self.A.copy_((utils.sinkhorn_knopp(sparse_A_3D).reshape(batch_size, self.H, self.N, self.N)) * self.A_mask)
        
        current_var = self.A.var(dim=[-1, -2], keepdim=True)
        vd = F.relu(self.hp_a_variance - current_var)
        V = (self.output).view(batch_size, self.N, self.H, self.head_dim).transpose(1, 2)
        boredom_noise = vd * torch.randn_like(V)
        V=V+boredom_noise 
        
        target = torch.matmul(self.A, V) 
        target = target.transpose(1, 2).reshape(batch_size, self.N, self.D)
        target = utils.softsign(target)

        # Pad and inject the dynamic env_input into the reserved nodes
        pad_len = num_input_nodes * self.D - input_dim
        if pad_len > 0:
            env_input_padded = F.pad(env_input, (0, pad_len))
        else:
            env_input_padded = env_input
            
        env_input_reshaped = env_input_padded.view(batch_size, num_input_nodes, self.D)
        target[:, :num_input_nodes, :] = env_input_reshaped

        err1_base = prediction - target
        err2 = keys - err1_base.detach()
        err3 = queries - err2.detach()
        err1 = prediction - (target + err3.detach())

        self.w1.step(self.w1.get_gradient(self.state, err1), self.step_counter)
        self.w2.step(self.w2.get_gradient(self.state, err2), self.step_counter)
        self.w3.step(self.w3.get_gradient(self.state, err3), self.step_counter)

        self.output.copy_(prediction)
        self.state.copy_(target)
def get_random_baseline(env_name, total_steps=1000, eval_start=200, num_seeds=4):
    """Calculates what a completely random agent scores in this environment."""
    env = gym.make(env_name)
    total_rewards =[]
    
    for seed in range(num_seeds):
        env.reset(seed=seed + 42)
        score = 0
        for step in range(total_steps):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            
            if step >= eval_start:
                score += reward
                
            if terminated or truncated:
                env.reset()
        total_rewards.append(score)
        
    env.close()
    return sum(total_rewards) / len(total_rewards)
class MinigridImageOnly(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(147,), dtype=np.float32
        )
    def observation(self, obs):
        return obs['image'].flatten()
# Use SyncVectorEnv directly instead of make_vec
def run_gym_task(sweep, dim=24, N=16, num_seeds=8, total_steps = 1000, eval_start = 200, env_name='Pendulum-v1'):
    batch_size = sweep.batch_size
    print(f"Initializing {batch_size} Agents across {num_seeds} global seeds on {DEVICE}...")
    
    active_tensors = sweep.get_agent_kwargs()
    hpm = HPManager(active_tensors, relations={}, N=N, D=dim, batch_size=batch_size, device=DEVICE)
    brain = PureTriadicBrain(batch_size=batch_size, num_nodes=N, dim=dim, hp_manager=hpm).to(DEVICE)
     
    
    #vecenv = gym.make_vec(env_name, num_envs=batch_size, vectorization_mode="sync")
    def make_env():
        env = gym.make(env_name)
        env = MinigridImageOnly(env) 
        return env
    vecenv = gym.vector.SyncVectorEnv([make_env for _ in range(batch_size)])
    obs_dim = math.prod(vecenv.single_observation_space.shape)
    
    is_continuous = isinstance(vecenv.single_action_space, gym.spaces.Box)
    if is_continuous:
        act_dim = vecenv.single_action_space.shape[0]
        act_high = torch.tensor(vecenv.single_action_space.high, device=DEVICE, dtype=torch.float32)
        act_low = torch.tensor(vecenv.single_action_space.low, device=DEVICE, dtype=torch.float32)
    else:
        act_dim = vecenv.single_action_space.n
        
    all_seed_fitness =[]
    stomach_dim = 4 
    input_dim = obs_dim + stomach_dim
    num_input_nodes = math.ceil(input_dim / brain.D)
    
    assert num_input_nodes < brain.N, f"Brain lacks nodes ({brain.N}) for env input ({input_dim}). Increase N!"
    
    print(f"Running {num_seeds} distinct environment initializations for {env_name}...")
    for seed in range(num_seeds):
        torch.manual_seed(seed + 42) 
        brain.reset(active_tensors, relations={})
        
        obs, _ = vecenv.reset(seed=seed + 42)
        obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        
        seed_fitness = torch.zeros(batch_size, device=DEVICE)
        recent_rewards = torch.zeros(batch_size, device=DEVICE)
        
        for step in range(total_steps):
            starvation = torch.clamp(torch.exp(-recent_rewards) - 0.5, 0.0, 1.0).unsqueeze(1)
            
            base_noise = torch.randn(batch_size, stomach_dim, device=DEVICE)
            stomach_signal = base_noise * starvation * 1e-2
            env_input = torch.cat([obs.view(batch_size, -1), stomach_signal], dim=1)

            brain(env_input)
            
            available_output_dims = brain.output[:, num_input_nodes:, :].reshape(batch_size, -1)
            motor_out = available_output_dims[:, :act_dim].clone()
            
            if is_continuous:
                motor_out = torch.tanh(motor_out) 
                actions = act_low + (motor_out + 1.0) * 0.5 * (act_high - act_low)
                actions_np = actions.cpu().numpy()
            else:
                actions = torch.argmax(motor_out, dim=-1)
                actions_np = actions.cpu().numpy()
            
            obs_np, rewards_np, terminated_np, truncated_np, infos = vecenv.step(actions_np)
            
            obs = torch.tensor(obs_np, dtype=torch.float32, device=DEVICE)
            rewards = torch.tensor(rewards_np, dtype=torch.float32, device=DEVICE)
            
            recent_rewards = (recent_rewards * 0.9) + (rewards * 0.1)
            
            done_mask = torch.tensor(terminated_np | truncated_np, dtype=torch.float32, device=DEVICE).view(-1, 1, 1)
            if done_mask.any():
                brain.state = brain.state * (1.0 - done_mask)
                brain.output = brain.output * (1.0 - done_mask)
            
            if step >= eval_start:
                seed_fitness += rewards

        all_seed_fitness.append(seed_fitness)

    vecenv.close()

    stacked_fitness = torch.stack(all_seed_fitness)  
    mean_fitness = stacked_fitness.mean(dim=0).cpu() 
    var_fitness = stacked_fitness.var(dim=0).cpu()   

    return mean_fitness, var_fitness


def visualize_best_agent(sweep, best_idx, dim=24, N=16, env_name='Pendulum-v1', total_steps=1000):
    print(f"\n--- VISUALIZING BEST AGENT (Sweep Index: {best_idx}) ---")
    
    # 1. Extract hyperparameters for ONLY the best agent
    active_tensors = sweep.get_agent_kwargs()
    best_tensors = {k: v[best_idx:best_idx+1].clone() for k, v in active_tensors.items()}
    
    # 2. Setup a single brain with batch_size=1
    hpm = HPManager(best_tensors, relations={}, N=N, D=dim, batch_size=1, device=DEVICE)
    brain = PureTriadicBrain(batch_size=1, num_nodes=N, dim=dim, hp_manager=hpm).to(DEVICE)
    brain.reset(best_tensors, relations={})
    
    # 3. Setup standard non-vectorized Gym env with rendering
    env = gym.make(env_name, render_mode="human")
    if "MiniGrid" in env_name:
        env = MinigridImageOnly(env)
    obs, _ = env.reset(seed=42)
    obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    
    obs_dim = math.prod(env.observation_space.shape)
    is_continuous = isinstance(env.action_space, gym.spaces.Box)
    if is_continuous:
        act_dim = env.action_space.shape[0]
        act_high = torch.tensor(env.action_space.high, device=DEVICE, dtype=torch.float32)
        act_low = torch.tensor(env.action_space.low, device=DEVICE, dtype=torch.float32)
    else:
        act_dim = env.action_space.n
        
    stomach_dim = 4
    input_dim = obs_dim + stomach_dim
    num_input_nodes = math.ceil(input_dim / brain.D)
    
    recent_reward = torch.zeros(1, device=DEVICE)
    visual_score = 0
    
    for step in range(total_steps):
        starvation = torch.clamp(torch.exp(-recent_reward) - 0.5, 0.0, 1.0).unsqueeze(1)
        base_noise = torch.randn(1, stomach_dim, device=DEVICE)
        stomach_signal = base_noise * starvation * 1e-2
        env_input = torch.cat([obs.view(1, -1), stomach_signal], dim=1)
        
        brain(env_input)
        
        available_output_dims = brain.output[:, num_input_nodes:, :].reshape(1, -1)
        motor_out = available_output_dims[:, :act_dim].clone()
        
        if is_continuous:
            motor_out = torch.tanh(motor_out) 
            actions = act_low + (motor_out + 1.0) * 0.5 * (act_high - act_low)
            action_out = actions[0].cpu().numpy()
        else:
            actions = torch.argmax(motor_out, dim=-1)
            action_out = actions[0].cpu().item() # standard env requires native python int
            
        obs_np, reward, terminated, truncated, _ = env.step(action_out)
        obs = torch.tensor(obs_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        
        rew_t = torch.tensor([reward], dtype=torch.float32, device=DEVICE)
        recent_reward = (recent_reward * 0.9) + (rew_t * 0.1)
        #visual_score += reward
        if step >= 200: # Match eval_start
            visual_score += reward
        if terminated or truncated:
            obs_np, _ = env.reset()
            obs = torch.tensor(obs_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            brain.state = brain.state * 0.0
            brain.output = brain.output * 0.0
            
    env.close()
    print(f"Visualization complete. Displayed Run Score: {visual_score:.2f}\n")

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    torch.set_grad_enabled(False) 
    
    sweep = Sweep2D(
        param_x_name='w_scale', param_x_vals=np.linspace(0.01, 0.7, 8),
        param_y_name='UNTANH', param_y_vals=np.linspace(1.0, 3.0, 8),
        device=DEVICE
    )
    
    #ENVIRONMENT = 'Pendulum-v1' 
    ENVIRONMENT = 'MiniGrid-MemoryS7-v0'
    DIM = 32
    N = 16
    NUM_SEEDS = 8
    TOTAL_STEPS = 1000
    EVAL_START = 200
    # 1. Evaluate Random Baseline
    random_score = get_random_baseline(ENVIRONMENT, total_steps=TOTAL_STEPS, eval_start=EVAL_START, num_seeds=NUM_SEEDS*16)
    print(f"Random Agent Average Score: {random_score:.2f}\n")
    
    # 2. Run the Triadic Brain Sweep
    mean_fit, var_fit = run_gym_task(
        sweep, 
        dim=DIM, 
        N=N, 
        num_seeds=NUM_SEEDS, 
        total_steps=TOTAL_STEPS, 
        eval_start=EVAL_START, 
        env_name=ENVIRONMENT)
    
    best_idx = torch.argmax(mean_fit).item()
    best_score = mean_fit[best_idx].item()
    
    print(f"Random Baseline Score: {random_score:.2f}")
    print(f"Best Triadic Brain Score: {best_score:.2f}")
   
    
    fit_min, fit_max = mean_fit.min(), mean_fit.max()
    mean_fit_normalized = (mean_fit - fit_min) / (fit_max - fit_min + 1e-8)
    sweep.visualize(mean_fit_normalized, var_fit)
    
    visualize_best_agent(sweep, best_idx, dim=DIM, N=N, env_name=ENVIRONMENT, total_steps=TOTAL_STEPS)