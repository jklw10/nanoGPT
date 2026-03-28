
import torch
import torch.nn as nn
import numpy as np
#import nodenet
import matplotlib.pyplot as plt
from models.node_agent import nodenet
from training_tools.environments.figure8_env import run_tracking_task
from training_tools.sweep import Sweep2D

torch.set_float32_matmul_precision('high')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware: {DEVICE}")
torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BrainWrapper(nn.Module):
    """
    figure8.env expects output = brain(environment) to return a tensor it can slice. 
    nodenet.py models update self.output but return None. This wrapper bridges the gap.
    """
    def __init__(self, model_class, config, num_nodes, dim, batch_size, device):
        super().__init__()
        # Instantiate the selected core nodenet model
        self.core = model_class(batch_size=batch_size, num_nodes=num_nodes, dim=dim, device=device)
        # Apply the batched hyperparameter dataclass
        self.core.apply_hyperparams(config)
        
    def forward(self, x):
        self.core(x)              # Update internal states
        return self.core.output  # Return to figure8 so it can slice nodes

def get_model_constructor(model_class, config, num_nodes, dim):
    """
    Returns a constructor function matching figure8.env's expected signature:
    `constructor(batch_size, **model_args)`
    """
    def constructor(batch_size, **kwargs):
        # We ignore kwargs because all parameters are successfully loaded inside `config`
        return BrainWrapper(model_class, config, num_nodes, dim, batch_size, config.UNTANH.device)
    return constructor

def visualize_best(sweep, fitness, mean_dist, var_dist, agent_pos_history, food_pos_history, eval_start):
    perf_grid = sweep.to_2d_grid(fitness).numpy()
    best_idx = torch.argmax(fitness).item()
    
    threshold = torch.quantile(fitness, 0.99)
    top_indices = torch.nonzero(fitness >= threshold).squeeze()
    if top_indices.dim() == 0:
        top_indices = top_indices.unsqueeze(0)
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
    torch.set_grad_enabled(False)
    print(f"Executing on: {DEVICE}")
    
    # 1. Setup Sweep Space
    sweep = Sweep2D(
        param_x_name='UNTANH', param_x_vals=np.linspace(0.5, 1.7, 8),
        param_y_name='w_scale', param_y_vals=np.linspace(0.1, 1.0, 8),
        device=DEVICE
    )
    
    # 2. Automatically generate a full batched HyperParamConfig based on Nodenet's space
    config = sweep.get_config(nodenet.defaults, nodenet.param_space)
    
    # 3. Choose the model from nodenet, wrap it in our life-fitting adapter
    model_class = nodenet.NodeNet # Swap with NodeNet, TriadNodeNet, etc
    constructor = get_model_constructor(model_class, config, num_nodes=16, dim=16)
    
    fitness, mean_dist, var_dist, agent_pos_hist, food_pos_hist, eval_start, brain, initial = run_tracking_task(
        model_constructor=constructor,
        device=DEVICE,
        batch_size=sweep.batch_size,
        total_steps=2000,
        eval_start=1000
    )
    visualize_best(sweep,fitness,mean_dist,var_dist,agent_pos_hist, food_pos_hist,eval_start)
    #sweep.visualize(fitness, fitness)
    
