
import torch
import torch.nn as nn
import numpy as np

import copy
from models import nodenet
from environments.figure8 import run_tracking_task, visualize_best
from training_tools.sweep import Sweep2D
from environments.gym_runner import run_gym_task, visualize_gym_pygame, get_random_baseline

torch.set_float32_matmul_precision('high')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware: {DEVICE}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BrainWrapper(nn.Module):
    def __init__(self, model_class, config, num_nodes, dim, batch_size, device):
        super().__init__()
        self.core = model_class(batch_size=batch_size, num_nodes=num_nodes, dim=dim, device=device)
        self.core.apply_hyperparams(config)
    def forward(self, x):
        loss = self.core(x)              
        return loss, self.core.output 
    
def get_model_constructor(model_class, config, num_nodes, dim, device):
    def constructor(batch_size, **kwargs):
        return BrainWrapper(model_class, config, num_nodes, dim, batch_size, device, **kwargs)
    return constructor


#if __name__ == "__main__":
#    print(f"Executing on: {DEVICE}")
#    
#    sweep = Sweep2D(
#        param_x_name='UNTANH', param_x_vals=np.linspace(0.5, 1.7, 8),
#        param_y_name='w_scale', param_y_vals=np.linspace(0.1, 1.0, 8),
#        device=DEVICE
#    )
#    
#    config = sweep.get_config(nodenet.defaults, nodenet.param_space)
#    
#    model_class = nodenet.StandardMultiHeadTriadNodeNet
#    constructor = get_model_constructor(model_class, config, num_nodes=16, dim=16,device=DEVICE)
#    #TODO, different task loading 
#    result = run_tracking_task(
#        model_constructor=constructor,
#        optimizer = torch.optim.SGD,
#        device = DEVICE,
#        batch_size=sweep.batch_size,
#        total_steps=2000,
#        eval_start=1000
#    )
#    
#    fitness, mean_dist, var_dist, agent_pos_hist, food_pos_hist, eval_start, brain, initial = result
#    visualize_best(sweep,fitness,mean_dist,var_dist,agent_pos_hist, food_pos_hist,eval_start)
def duplicate_best_initialization(model, best_idx, batch_size):
    new_model = copy.deepcopy(model)
    with torch.no_grad():
        for name, param in new_model.named_parameters():
            #could be safer but eh
            if param.dim() > 0 and param.shape[0] == batch_size:
                best_slice = param[best_idx].clone().unsqueeze(0).expand(batch_size, *param.shape[1:])
                param.copy_(best_slice)
                
        for name, buf in new_model.named_buffers():
            if buf.dim() > 0 and buf.shape[0] == batch_size:
                best_slice = buf[best_idx].clone().unsqueeze(0).expand(batch_size, *buf.shape[1:])
                buf.copy_(best_slice)
                
    return new_model

#eh fuck it, cleanup soon:tm:

def figure8():
    optimizer = "adam"
    if optimizer is None:
        torch.set_grad_enabled(False)
    print(f"Executing on: {DEVICE}")
    
    sweep = Sweep2D(
        param_x_name='UNTANH', param_x_vals=np.linspace(0.5, 1.7, 8),
        param_y_name='w_scale', param_y_vals=np.linspace(0.5, 1.2, 8),
        device=DEVICE
    )
    
    config = sweep.get_config(nodenet.defaults, nodenet.param_space)
    
    torch.manual_seed(42)
    model_class = nodenet.MemNodeNet # Swap with NodeNet, TriadNodeNet, etc
    constructor = get_model_constructor(model_class, config, num_nodes=16, dim=16,device =DEVICE)
    brain = constructor(batch_size=sweep.batch_size).to(DEVICE)
    brain = torch.compile(brain)

    if optimizer is not None:
        optimizer = torch.optim.AdamW(
            brain.parameters(), 
            lr=0.005, 
            weight_decay=0.02, 
            betas=(0.9, 0.999)
        )
    initial_snapshots = copy.deepcopy(brain)
    fitness, mean_dist, var_dist, agent_pos_hist, food_pos_hist, eval_start, brain = run_tracking_task(
        brain=brain,
        optimizer=optimizer,
        device=DEVICE,
        batch_size=sweep.batch_size,
        total_steps=2000,
        eval_start=1000
    )
    
    visualize_best(sweep,fitness,mean_dist,var_dist,agent_pos_hist, food_pos_hist,eval_start)
    best_idx = torch.argmax(fitness).item()
    print(f"Best instance was {best_idx}. Duplicating weights for Run 2...")
    
    brain2 = duplicate_best_initialization(initial_snapshots, best_idx, sweep.batch_size)
    brain2.core.apply_hyperparams(config)
    brain2 = torch.compile(brain2)
    
    optimizer2 = None
    if optimizer is not None:
        optimizer2 = torch.optim.AdamW(
            brain2.parameters(), 
            lr=0.005, 
            weight_decay=0.02, 
            betas=(0.9, 0.999)
        )
        
    torch.manual_seed(42)
    fitness2, mean_dist2, var_dist2, agent_pos_hist2, food_pos_hist2, eval_start2, brain2 = run_tracking_task(
        brain=brain2,
        optimizer=optimizer2,
        device=DEVICE,
        batch_size=sweep.batch_size,
        total_steps=2000,
        eval_start=1000
    )
    
    # If the resulting heatmap is drastically different, it means the 
    # initialization weights are overpowering your hyperparameters.
    visualize_best(sweep, fitness2, mean_dist2, var_dist2, agent_pos_hist2, food_pos_hist2, eval_start2)

def gym():
    
    ENVIRONMENT = "MiniGrid-DoorKey-5x5-v0"
    optimizer = "adam"
    if optimizer is None:
        torch.set_grad_enabled(False)
    print(f"Executing on: {DEVICE}")
    
    sweep = Sweep2D(
        param_x_name='UNTANH', param_x_vals=np.linspace(0.5, 1.7, 8),
        param_y_name='w_scale', param_y_vals=np.linspace(0.5, 1.2, 8),
        device=DEVICE
    )
    
    config = sweep.get_config(nodenet.defaults, nodenet.param_space)
    
    torch.manual_seed(42)
    model_class = nodenet.StandardNodeNet2 # Swap with NodeNet, TriadNodeNet, etc
    constructor = get_model_constructor(model_class, config, num_nodes=16, dim=16,device =DEVICE)
    brain = constructor(batch_size=sweep.batch_size).to(DEVICE)
    brain = torch.compile(brain)

    if optimizer is not None:
        optimizer = torch.optim.AdamW(
            brain.parameters(), 
            lr=0.005, 
            weight_decay=0.02, 
            betas=(0.9, 0.999)
        )
    initial_snapshots = copy.deepcopy(brain)
    fitness, mean_dist, var_dist, agent_pos_hist, food_pos_hist, eval_start, brain = run_gym_task(
        brain=brain,
        optimizer=optimizer,
        device=DEVICE,
        batch_size=sweep.batch_size,
        env_name=ENVIRONMENT,  
        total_steps=2000,
        eval_start=1000
    )
    visualize_gym_pygame(
        brain=brain,
        sweep=sweep,
        fitness=fitness,
        env_name=ENVIRONMENT,
        device=DEVICE,
        max_steps=1500
    )
    best_idx = torch.argmax(fitness).item()
    print(f"Best instance was {best_idx}. Duplicating weights for Run 2...")
    
    brain2 = duplicate_best_initialization(initial_snapshots, best_idx, sweep.batch_size)
    brain2.core.apply_hyperparams(config)
    brain2 = torch.compile(brain2)
    
    optimizer2 = None
    if optimizer is not None:
        optimizer2 = torch.optim.AdamW(
            brain2.parameters(), 
            lr=0.005, 
            weight_decay=0.02, 
            betas=(0.9, 0.999)
        )
        
    torch.manual_seed(42)
    fitness2, mean_dist2, var_dist2, agent_pos_hist2, food_pos_hist2, eval_start2, brain2 =run_gym_task(
        
        env_name=ENVIRONMENT, 
        brain=brain2,
        optimizer=optimizer2,
        device=DEVICE,
        batch_size=sweep.batch_size,
        total_steps=2000,
        eval_start=1000
    )
    visualize_gym_pygame(
        brain=brain2,
        sweep=sweep,
        fitness=fitness2,
        env_name=ENVIRONMENT,
        device=DEVICE,
        max_steps=3000
    )
    
    rand_mean, rand_std = get_random_baseline(
        env_name=ENVIRONMENT, 
        total_steps=2000, 
        eval_start=1000, 
        num_actors=100
    )
    # 3. Print the Comparison
    best_idx = torch.argmax(fitness).item()
    best_brain_reward = -mean_dist[best_idx].item() # Negate to flip back to actual reward
    print(f"Random Actors (100):  {rand_mean:>8.4f} ± {rand_std:.4f}")
    print(f"Best Brain Model:     {best_brain_reward:>8.4f}")
    best_idx = torch.argmax(fitness2).item()
    best_brain_reward = -mean_dist2[best_idx].item() # Negate to flip back to actual reward
    run2_mean = -mean_dist2.mean().item() 
    print(f"Best Brain2 Model:     {best_brain_reward:>8.4f}")
    print(f"Brain2 mean:     {run2_mean:>8.4f}")
    #visualize_best(sweep, fitness2, mean_dist2, var_dist2, agent_pos_hist2, food_pos_hist2, eval_start2)


if __name__ == "__main__":
    gym()