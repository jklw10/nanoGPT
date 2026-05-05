
import torch
import torch.nn as nn
import numpy as np

import copy
from models import nodenet
from environments.figure8 import run_vis as f8rv
from training_tools.sweep import Sweep2D
from environments.gym_runner import run_vis, get_random_baseline

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

def gym(model_class,run_vis, time = 3000, environment = None):
    optimizer = "adam"
    if optimizer is None:
        torch.set_grad_enabled(False)
    print(f"Executing on: {DEVICE}")
    xswp = np.linspace(0.5, 0.9, 8)
    yswp = np.linspace(0.6, 0.8, 8)
    sweep = Sweep2D(
        param_x_name='UNTANH', param_x_vals=xswp,
        param_y_name='w_scale', param_y_vals=yswp,
        device=DEVICE
    )
    
    config = sweep.get_config(nodenet.defaults, nodenet.param_space)
    
    torch.manual_seed(42) # Swap with NodeNet, TriadNodeNet, etc
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
    fitness, mean_dist = run_vis(time, environment, brain, optimizer, sweep, device=DEVICE)
    best_idx = torch.argmax(fitness).item()
    print(f"Best instance was {best_idx}. Duplicating weights for Run 2...")
    
    brain2 = duplicate_best_initialization(initial_snapshots, best_idx, sweep.batch_size)
    
    ideal_x  = xswp[best_idx%8]
    ideal_y  = xswp[best_idx//8]
    
    sweep2 = Sweep2D(
        param_x_name='UNTANH',  param_x_vals = [ideal_x for _ in range(0,8)],
        param_y_name='w_scale', param_y_vals = [ideal_y for _ in range(0,8)],
        device=DEVICE
    )
    config = sweep2.get_config(nodenet.defaults, nodenet.param_space)
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
    
    fitness2, mean_dist2 = run_vis(time, environment, brain2, optimizer2, sweep, device=DEVICE)
    if environment is not None:
        rand_mean, rand_std, rand_max = get_random_baseline(
            env_name=environment, 
            total_steps=time, 
            eval_start=1000, 
            num_actors=sweep.batch_size
        )
        # 3. Print the Comparison
        print(f"Random Actors ({sweep.batch_size}):  {rand_mean:>8.4f} ± {rand_std:.4f}")
        print(f"Random Actors ({sweep.batch_size}):  max {rand_max:>8.4f}")
    
    best_idx2 = torch.argmax(fitness2).item()
    best_brain_reward  = -mean_dist[best_idx].item() # Negate to flip back to actual reward
    best_brain_reward2 = -mean_dist2[best_idx2].item() # Negate to flip back to actual reward
    run_mean  = -mean_dist.mean().item() 
    run2_mean = -mean_dist2.mean().item() 
    print(f"Best Brain Model:     {best_brain_reward:>8.4f}")
    print(f"Best Brain2 Model:     {best_brain_reward2:>8.4f}")
    print(f"Brain mean:     {run_mean:>8.4f}")
    print(f"Brain2 mean:     {run2_mean:>8.4f}")

if __name__ == "__main__":
    ENVIRONMENT = "MiniGrid-DoorKey-8x8-v0"
    model_class = nodenet.StandardNodeNet2
    #gym(model_class, f8rv)
    gym(model_class,run_vis,environment=ENVIRONMENT)
    ##figure8()