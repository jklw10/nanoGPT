import torch
import numpy as np
import minigrid

from parts import utils # This registration step is mandatory i hate this

try:
    import gymnasium as gym
except ImportError:
    import gym  # Fallback for older versions

import pygame

def stack_obs(obs, batch_size):
    """Safely replicates a single unvectorized observation up to a full batch size."""
    if isinstance(obs, dict):
        return {k: np.stack([v] * batch_size) for k, v in obs.items()}
    elif isinstance(obs, tuple):
        return tuple(np.stack([v] * batch_size) for v in obs)
    else:
        return np.stack([obs] * batch_size)

def process_obs(obs_batch, device):
    """
    Strips task text/strings (like 'mission' or NLP prompts) and flattens 
    the observations to purely numeric vectors, keeping dimension count manageable.
    """
    if isinstance(obs_batch, dict):
        flat_parts =[]
        for key, val in obs_batch.items():
            if key == 'mission' or 'text' in key:
                continue
            if isinstance(val, (list, tuple)) and len(val) > 0 and isinstance(val[0], str):
                continue
            if isinstance(val, np.ndarray) and val.dtype.kind in {'U', 'S', 'O'}:
                continue
            
            t = torch.tensor(val, dtype=torch.float32, device=device)
            t_flat = t.view(t.shape[0], -1)
            #t_norm = (t_flat - t_flat.mean(dim=-1, keepdim=True)) / (t_flat.std(dim=-1, keepdim=True) + 1e-5)
            t_norm = utils.range_norm(t_flat,dim=None,scale=1.0).clone()
            flat_parts.append(t_norm)
        return torch.cat(flat_parts, dim=-1) if flat_parts else torch.zeros((obs_batch.shape[0], 1), device=device)
        
    elif isinstance(obs_batch, tuple):
        flat_parts =[]
        for val in obs_batch:
            if isinstance(val, (list, tuple)) and len(val) > 0 and isinstance(val[0], str):
                continue
            if isinstance(val, np.ndarray) and val.dtype.kind in {'U', 'S', 'O'}:
                continue
            
            t = torch.tensor(val, dtype=torch.float32, device=device)
            t_flat = t.view(t.shape[0], -1)
            t_norm = (t_flat - t_flat.mean(dim=-1, keepdim=True)) / (t_flat.std(dim=-1, keepdim=True) + 1e-5)
            flat_parts.append(t_norm)
        return torch.cat(flat_parts, dim=-1)
        
    else:
        t = torch.tensor(obs_batch, dtype=torch.float32, device=device)
        t_flat = t.view(t.shape[0], -1)
        t_norm = (t_flat - t_flat.mean(dim=-1, keepdim=True)) / (t_flat.std(dim=-1, keepdim=True) + 1e-5)
        return t_norm
def get_random_baseline(env_name, total_steps=2000, eval_start=1000, num_actors=100):
    print(f"Running random baseline with {num_actors} actors for {env_name}...")
    
    envs = gym.vector.SyncVectorEnv([lambda: gym.make(env_name) for _ in range(num_actors)])
    envs.reset()
    
    eval_steps = total_steps - eval_start
    rewards_history = np.zeros((eval_steps, num_actors))
    
    for step in range(total_steps):
        # Natively sample random actions for the entire batch
        actions = envs.action_space.sample()
        
        step_out = envs.step(actions)
        if len(step_out) == 5:
            obs, rewards, terminated, truncated, infos = step_out
        else:
            obs, rewards, dones, infos = step_out
            
        if step >= eval_start:
            rewards_history[step - eval_start] = rewards
            
    envs.close()
    
    # Calculate the average reward per actor, then the overall mean/std
    mean_reward_per_actor = rewards_history.mean(axis=0)
    overall_mean = mean_reward_per_actor.mean()
    overall_std = mean_reward_per_actor.std()
    
    return overall_mean, overall_std
def run_gym_task(brain, optimizer, device, batch_size, total_steps=2000, eval_start=1000, env_name="Pendulum-v1"):
    print(f"Initializing {batch_size} Agents on {device} for {env_name}...")
    
    envs = gym.vector.SyncVectorEnv([lambda: gym.make(env_name) for _ in range(batch_size)])
    
    reset_out = envs.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    
    processed_obs = process_obs(obs, device)
    eval_steps = total_steps - eval_start
    
    agent_pos_history = torch.zeros((total_steps, batch_size, 2), device=device)
    food_pos_history = torch.zeros((total_steps, 2), device=device)
    rewards_history = torch.zeros((eval_steps, batch_size), device=device)
    
    single_action_space = envs.single_action_space
    is_discrete = hasattr(single_action_space, 'n')
    
    if is_discrete:
        action_dim = single_action_space.n
    elif hasattr(single_action_space, 'shape') and len(single_action_space.shape) > 0:
        action_dim = np.prod(single_action_space.shape)
    else:
        action_dim = 1
    
    rewards_t = torch.zeros(batch_size, device=device)
    for step in range(total_steps):
        penalty = (1.0 - rewards_t).clamp(min=0.0).unsqueeze(1)
        
        # Create noise scaled by the penalty (matching your 1e-5 scale)
        pain_noise = torch.randn_like(processed_obs) * penalty * 1e-5
        
        # Overlay the pain noise over the standard observations
        env_input = processed_obs + pain_noise
        loss, output = brain(env_input)
        if optimizer:
            loss.backward()
            optimizer.step()

        # Node 2 controls Motor Outputs
        actual_action_dim = min(action_dim, output.shape[-1])
        motor_out = output[:, 2, :actual_action_dim]
        
        if is_discrete:
            # THE FIX: Softmax sampling instead of Argmax locking
            probs = torch.softmax(motor_out, dim=-1)
            actions = torch.multinomial(probs, 1).squeeze(-1).cpu().numpy()
        else:
            actions_raw = motor_out.cpu().detach().numpy()
            if hasattr(single_action_space, 'low') and hasattr(single_action_space, 'high'):
                if actual_action_dim < action_dim:
                    pad = np.zeros((batch_size, action_dim - actual_action_dim))
                    actions_raw = np.concatenate([actions_raw, pad], axis=-1)
                actions = np.clip(actions_raw, single_action_space.low.flatten(), single_action_space.high.flatten())
            else:
                actions = actions_raw
            actions = actions.reshape((batch_size,) + single_action_space.shape)
        
        step_out = envs.step(actions)
        if len(step_out) == 5:
            next_obs, rewards, terminated, truncated, infos = step_out
        else:
            next_obs, rewards, dones, infos = step_out
            
        processed_obs = process_obs(next_obs, device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        
        # Extracted purely for signature compatibility
        if processed_obs.shape[1] >= 2:
            agent_pos_history[step] = processed_obs[:, :2]
        else:
            agent_pos_history[step, :, 0] = processed_obs[:, 0]
            agent_pos_history[step, :, 1] = processed_obs[:, 0]
        
        food_pos_history[step] = torch.tensor([0.0, 0.0], device=device)
        if step >= eval_start:
            rewards_history[step - eval_start] = rewards_t

    envs.close()

    mean_reward = rewards_history.mean(dim=0).cpu()
    var_reward = rewards_history.var(dim=0).cpu()
    
    mean_dist = -mean_reward
    var_dist = var_reward
    fitness = torch.sigmoid(mean_reward * 0.1 - var_reward * 0.01)
    
    return fitness, mean_dist, var_dist, agent_pos_history, food_pos_history, eval_start, brain

def visualize_gym_pygame(brain, sweep, fitness, env_name="Pendulum-v1", device="cpu", max_steps=1000):
    """
    Dual-pane Pygame visualizer: 
    Left Pane: Hyperparameter landscape mapping.
    Right Pane: Best Agent real-time RGB Render.
    """
    pygame.init()
    
    # Isolate best agent
    best_idx = torch.argmax(fitness).item()
    best_x, best_y = sweep.get_params(best_idx)
    best_fitness = fitness[best_idx].item()
    
    env = gym.make(env_name, render_mode="rgb_array")
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    
    # Process Landscape details
    perf_grid = sweep.to_2d_grid(fitness).cpu().numpy()
    rows, cols = perf_grid.shape
    max_r, max_c = np.unravel_index(np.argmax(perf_grid), perf_grid.shape)
    min_val, max_val = np.min(perf_grid), np.max(perf_grid)
    
    # Probe one frame to configure window size natively
    frame = env.render()
    frame_h, frame_w, _ = frame.shape if frame is not None else (400, 400, 3)
    grid_pane_w = 400
    grid_pane_h = frame_h
    
    screen = pygame.display.set_mode((grid_pane_w + frame_w, frame_h))
    pygame.display.set_caption(f"PyGame Evaluator: {env_name}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    
    # Define action geometry for single env
    single_action_space = env.action_space
    is_discrete = hasattr(single_action_space, 'n')
    if is_discrete:
        action_dim = single_action_space.n
    elif hasattr(single_action_space, 'shape') and len(single_action_space.shape) > 0:
        action_dim = np.prod(single_action_space.shape)
    else:
        action_dim = 1
        
    def get_color(val):
        norm = (val - min_val) / (max_val - min_val + 1e-5)
        r = min(255, int(norm * 255 * 1.5))
        g = min(255, max(0, int((norm - 0.3) * 255 * 1.5)))
        b = min(255, max(0, int((norm - 0.6) * 255 * 1.5)))
        return (r, g, b)
        
    for step in range(max_steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                env.close()
                return
        
        # --- DRAW PANE 1: Landscape Grid ---
        screen.fill((30, 30, 30))
        cell_w = grid_pane_w / cols
        cell_h = grid_pane_h / rows
        
        for r in range(rows):
            for c in range(cols):
                color = get_color(perf_grid[r, c])
                rect = pygame.Rect(c * cell_w, grid_pane_h - (r+1) * cell_h, cell_w, cell_h)
                pygame.draw.rect(screen, color, rect)
        
        # Agent marker
        center_x = max_c * cell_w + cell_w / 2
        center_y = grid_pane_h - (max_r * cell_h + cell_h / 2)
        pygame.draw.circle(screen, (0, 255, 255), (int(center_x), int(center_y)), 10)
        pygame.draw.circle(screen, (0, 0, 0), (int(center_x), int(center_y)), 10, 2)
        
        grid_text =[
            f"Env: {env_name} | Best Agent: {best_idx}",
            f"{sweep.p_x_name}: {best_x:.4f} | {sweep.p_y_name}: {best_y:.4f}",
            f"Fitness: {best_fitness:.4f}"
        ]
        for i, text_str in enumerate(grid_text):
            ts = font.render(text_str, True, (255, 255, 255))
            screen.blit(ts, (10, 10 + i * 25))

        # --- FORWARD PASS PANE 2: Agent Render ---
        # Match expected batch layout by broadcasting unvectorized observation
        obs_batch = stack_obs(obs, sweep.batch_size)
        processed_obs = process_obs(obs_batch, device)
        
        with torch.no_grad():
            loss, output = brain(processed_obs)
            
        actual_action_dim = min(action_dim, output.shape[-1])
        motor_out = output[:, 2, :actual_action_dim]
        
        if is_discrete:
            actions = torch.argmax(motor_out, dim=-1).cpu().numpy()
        else:
            actions_raw = motor_out.cpu().detach().numpy()
            if hasattr(single_action_space, 'low') and hasattr(single_action_space, 'high'):
                if actual_action_dim < action_dim:
                    pad = np.zeros((sweep.batch_size, action_dim - actual_action_dim))
                    actions_raw = np.concatenate([actions_raw, pad], axis=-1)
                actions = np.clip(actions_raw, single_action_space.low.flatten(), single_action_space.high.flatten())
            else:
                actions = actions_raw
            actions = actions.reshape((sweep.batch_size,) + single_action_space.shape)
            
        best_action = actions[best_idx]
        
        step_out = env.step(best_action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            obs, reward, done, info = step_out
            
        frame = env.render()
        if frame is not None:
            # Note: Pygame expects [Width, Height, RGB], Gym yields [Height, Width, RGB]
            surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            screen.blit(surf, (grid_pane_w, 0))
            
        pygame.display.flip()
        clock.tick(60) # 60 FPS cap
        
        if done:
            reset_out = env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            
    pygame.quit()
    env.close()