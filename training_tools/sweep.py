import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from types import SimpleNamespace

class Sweep2D:
    def __init__(self, param_x_name, param_x_vals, param_y_name, param_y_vals, device):
        self.p_x_name = param_x_name
        self.p_y_name = param_y_name
        
        self.vals_x = torch.tensor(param_x_vals, dtype=torch.float32, device=device)
        self.vals_y = torch.tensor(param_y_vals, dtype=torch.float32, device=device)
        
        self.grid_x = len(self.vals_x)
        self.grid_y = len(self.vals_y)
        self.batch_size = self.grid_x * self.grid_y
        
        grid_y_mat, grid_x_mat = torch.meshgrid(self.vals_y, self.vals_x, indexing='ij')
        
        self.grid_x_flat = grid_x_mat.flatten()
        self.grid_y_flat = grid_y_mat.flatten()
        
    def get_agent_kwargs(self):
        return {
            self.p_x_name: self.grid_x_flat,
            self.p_y_name: self.grid_y_flat
        }

    def get_config(self, defaults, param_space):
        """
        Generates a batched HyperParamConfig matching the sweep grid.
        Swept parameters use the grid, while others use the scalar from defaults.
        """
        config_dict = {}
        all_keys = set(defaults.keys()).union(set(param_space.keys()))
        
        for key in all_keys:
            if key == self.p_x_name:
                val_tensor = self.grid_x_flat.clone()
            elif key == self.p_y_name:
                val_tensor = self.grid_y_flat.clone()
            else:
                # Fallback sequentially: defaults -> middle of param_space -> 0.0
                if key in defaults:
                    val = defaults[key]
                elif key in param_space:
                    val = (param_space[key]['low'] + param_space[key]['high']) / 2.0
                else:
                    val = 0.0
                val_tensor = torch.full((self.batch_size,), float(val), dtype=torch.float32, device=self.vals_x.device)
            
            config_dict[key] = val_tensor
            
        try:
            # Try returning the actual Dataclass so it perfectly fits type hints
            from models.node_agent.nodenet import HyperParamConfig
            import dataclasses
            valid_keys = {f.name for f in dataclasses.fields(HyperParamConfig)}
            filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
            return HyperParamConfig(**filtered_dict)
        except ImportError:
            # Fallback to SimpleNamespace if the file is moved / unavailable
            return SimpleNamespace(**config_dict)
        
    def get_params(self, flat_idx):
        w_idx = flat_idx // self.grid_x
        s_idx = flat_idx % self.grid_x
        return self.vals_x[s_idx].item(), self.vals_y[w_idx].item()
        
    def to_2d_grid(self, flat_tensor):
        return flat_tensor.view(self.grid_y, self.grid_x)
        
    def get_extent(self):
        return[
            self.vals_x.min().item(), self.vals_x.max().item(),
            self.vals_y.min().item(), self.vals_y.max().item()
        ]
        
    def visualize(self, mean_fitness, var_fitness):
        perf_grid = self.to_2d_grid(mean_fitness).numpy()
        var_grid = self.to_2d_grid(var_fitness).numpy()

        best_idx = torch.argmax(mean_fitness).item()
        valid_mask = ~torch.isnan(mean_fitness)
        if not valid_mask.any():
            print("CRITICAL ERROR: All agents collapsed to NaN!")
            return

        valid_fitness = mean_fitness[valid_mask]
        threshold = torch.quantile(valid_fitness, 0.99)

        top_indices = torch.nonzero(torch.nan_to_num(mean_fitness, nan=-1.0) >= threshold).squeeze()
        if top_indices.dim() == 0: 
            top_indices = top_indices.unsqueeze(0)

        if len(top_indices) == 0:
            rep_idx = best_idx 
        else:
            top_indices = top_indices[torch.argsort(mean_fitness[top_indices])]
            rep_idx = top_indices[len(top_indices) // 2].item()

        best_x, best_y = self.get_params(best_idx)
        rep_x, rep_y = self.get_params(rep_idx)

        fig = plt.figure(figsize=(16, 12))

        # --- PLOT 1: Mean Fitness ---
        ax1 = plt.subplot(2, 2, 1)
        im1 = ax1.imshow(perf_grid, origin='lower', extent=self.get_extent(), aspect='auto', cmap='viridis', vmin=0, vmax=1.0)
        ax1.set_title("Mean Fitness Landscape\n(1.0 = Perfect Overlap, Averaged across Seeds)")
        ax1.set_xlabel(f"{self.p_x_name}")
        ax1.set_ylabel(f"{self.p_y_name}")
        ax1.scatter(best_x, best_y, color='cyan', marker='*', s=150, edgecolor='black', label='Best Outlier')
        ax1.scatter(rep_x, rep_y, color='magenta', marker='o', s=80, edgecolor='white', label='Top 1% Rep')
        ax1.legend(loc="upper right")
        fig.colorbar(im1, ax=ax1, label='Fitness')

        # --- PLOT 2: Variance ---
        ax2 = plt.subplot(2, 2, 2)
        im2 = ax2.imshow(var_grid, origin='lower', extent=self.get_extent(), aspect='auto', cmap='inferno')
        ax2.set_title("Seed Instability Landscape\n(Variance of Fitness Across 8 Seeds)")
        ax2.set_xlabel(f"{self.p_x_name}")
        ax2.set_ylabel(f"{self.p_y_name}")
        fig.colorbar(im2, ax=ax2, label='Cross-Seed Variance')

        plt.tight_layout()
        plt.show()

class Sweep3D:
    def __init__(self, param_x_name, param_x_vals, param_y_name, param_y_vals, param_z_name, param_z_vals, device):
        self.p_x_name = param_x_name
        self.p_y_name = param_y_name
        self.p_z_name = param_z_name
        
        self.vals_x = torch.tensor(param_x_vals, dtype=torch.float32, device=device)
        self.vals_y = torch.tensor(param_y_vals, dtype=torch.float32, device=device)
        self.vals_z = torch.tensor(param_z_vals, dtype=torch.float32, device=device)
        
        self.grid_x = len(self.vals_x)
        self.grid_y = len(self.vals_y)
        self.grid_z = len(self.vals_z)
        self.batch_size = self.grid_x * self.grid_y * self.grid_z
        
        # Create 3D grid
        grid_z_mat, grid_y_mat, grid_x_mat = torch.meshgrid(self.vals_z, self.vals_y, self.vals_x, indexing='ij')
        
        self.grid_x_flat = grid_x_mat.flatten()
        self.grid_y_flat = grid_y_mat.flatten()
        self.grid_z_flat = grid_z_mat.flatten()
        
    def get_agent_kwargs(self):
        return {
            self.p_x_name: self.grid_x_flat,
            self.p_y_name: self.grid_y_flat,
            self.p_z_name: self.grid_z_flat
        }

    def get_config(self, defaults, param_space):
        """
        Generates a batched HyperParamConfig matching the sweep grid for 3 parameters.
        """
        config_dict = {}
        all_keys = set(defaults.keys()).union(set(param_space.keys()))
        
        for key in all_keys:
            if key == self.p_x_name:
                val_tensor = self.grid_x_flat.clone()
            elif key == self.p_y_name:
                val_tensor = self.grid_y_flat.clone()
            elif key == self.p_z_name:
                val_tensor = self.grid_z_flat.clone()
            else:
                if key in defaults:
                    val = defaults[key]
                elif key in param_space:
                    val = (param_space[key]['low'] + param_space[key]['high']) / 2.0
                else:
                    val = 0.0
                val_tensor = torch.full((self.batch_size,), float(val), dtype=torch.float32, device=self.vals_x.device)
            
            config_dict[key] = val_tensor
            
        try:
            from models.node_agent.nodenet import HyperParamConfig
            import dataclasses
            valid_keys = {f.name for f in dataclasses.fields(HyperParamConfig)}
            filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
            return HyperParamConfig(**filtered_dict)
        except ImportError:
            return SimpleNamespace(**config_dict)
    
    def visualize(self, mean_fitness):
        # Convert tensors to numpy for Plotly
        x_vals = self.grid_x_flat.cpu().numpy()
        y_vals = self.grid_y_flat.cpu().numpy()
        z_vals = self.grid_z_flat.cpu().numpy()
        fit_vals = mean_fitness.cpu().numpy()
        
        # Handle NaNs
        valid_mask = ~np.isnan(fit_vals)
        x_vals = x_vals[valid_mask]
        y_vals = y_vals[valid_mask]
        z_vals = z_vals[valid_mask]
        fit_vals = fit_vals[valid_mask]
        
        if len(fit_vals) == 0:
            print("CRITICAL ERROR: All agents collapsed to NaN!")
            return
    
        # Define the slider steps (Percentiles to keep)
        percentiles_to_keep =[100, 50, 25, 10, 5, 2, 1, 0.5]
        
        fig = go.Figure()
        steps =[]
        
        # Create a separate 3D scatter trace for each percentile threshold
        for i, p in enumerate(percentiles_to_keep):
            # Calculate the fitness threshold for this top %
            threshold = np.percentile(fit_vals, 100.0 - p)
            mask = fit_vals >= threshold
            
            # Add the trace (set only the first one to visible by default)
            fig.add_trace(go.Scatter3d(
                x=x_vals[mask], 
                y=y_vals[mask], 
                z=z_vals[mask],
                mode='markers',
                marker=dict(
                    size=5,
                    color=fit_vals[mask],
                    colorscale='Plasma',
                    opacity=0.8,
                    colorbar=dict(title="Mean Fitness") if i == 0 else None
                ),
                name=f"Top {p}%",
                visible=(i == 3) # Default view: Top 10%
            ))
            
            # Build the slider behavior
            step = dict(
                method="update",
                args=[{"visible": [False] * len(percentiles_to_keep)}],
                label=f"{p}%"
            )
            step["args"][0]["visible"][i] = True
            steps.append(step)
    
        # Attach slider to layout
        sliders =[dict(
            active=3, # Matches the default visible trace
            currentvalue={"prefix": "Showing Hull of Top: "},
            pad={"t": 50},
            steps=steps
        )]
    
        fig.update_layout(
            title="3D Hyperparameter Collision Hunter",
            scene=dict(
                xaxis_title=self.p_x_name,
                yaxis_title=self.p_y_name,
                zaxis_title=self.p_z_name,
            ),
            sliders=sliders,
            width=1800,
            height=1000
        )
    
        # Opens in your default web browser interactively
        fig.show()