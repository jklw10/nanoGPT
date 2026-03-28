import torch
import copy
import math

def run_tracking_task(model_constructor, optimizer, device, batch_size, total_steps = 2000, eval_start = 1000, **model_args):
    print(f"Initializing {batch_size}  Agents on {device}...")
    
    brain = model_constructor(batch_size=batch_size, **model_args).to(device)
    brain = torch.compile(brain)
    optimizer = optimizer(
        brain.parameters(), 
        lr=0.005, 
        weight_decay=0.02, 
        #betas=(0.9, 0.999)
    )
    initial_snapshots = copy.deepcopy(brain)
    agent_pos = torch.full((batch_size, 2), -3.0, device=device)
    agent_vel = torch.zeros(batch_size, 2, device=device)
    
    eval_steps = total_steps - eval_start
    
    agent_pos_history = torch.zeros((total_steps, batch_size, 2), device=device)
    food_pos_history = torch.zeros((total_steps, 2), device=device)
    dist_history = torch.zeros((eval_steps, batch_size), device=device)
    
    for step in range(total_steps):
        t = step / 100.0
        food_pos = torch.tensor([[math.sin(t) * 2.0, math.sin(t * 2.0) * 1.5]], device=device).expand(batch_size, 2)
        
        diffs = food_pos - agent_pos
        dists = torch.norm(diffs, dim=1, keepdim=True)
        direction = diffs / (dists + 1e-5)
        
        starvation = dists 
        environment = torch.randn(batch_size, 2, device=device) * starvation * 1e-5
        environment[:, 0:2] = direction
        
        output, loss = brain(environment)
        loss.backward()
        optimizer.step()

        motor_out = output[:, 2, 0:2].clone()
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
