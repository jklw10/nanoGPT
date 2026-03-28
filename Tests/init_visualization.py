import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 1. Define Custom Activations
class Gaussian(nn.Module):
    def forward(self, x):
        return torch.exp(-x**2)

class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class untanh(nn.Module):
    def forward(self, x):
        return x - torch.tanh(x)* 0.99


# 2. Define the MLP Architecture
class RandomMLP(nn.Module):
    def __init__(self, activation_cls, hidden_dim=100, num_layers=4):
        super().__init__()
        layers =[]
        
        # Input layer: 2D coordinate (x, y) -> hidden_dim
        layers.append(nn.Linear(2, hidden_dim))
        layers.append(activation_cls())
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation_cls())
            
        # Output layer: hidden_dim -> 1D scalar
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

# 3. Create the 2D coordinate grid
def get_2d_grid(resolution=128, range_val=2.0):
    """Generates a 2D meshgrid to represent the input space."""
    x = torch.linspace(-range_val, range_val, resolution)
    y = torch.linspace(-range_val, range_val, resolution)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    
    # Flatten grid to pass through MLP: shape (resolution*resolution, 2)
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    return grid

# 4. Custom Weight Initialization (Uniform)
def init_weights_uniform(m, scale_multiplier=1.0):
    """Initializes weights to Uniform(-limit, limit) based on the multiplier."""
    if isinstance(m, nn.Linear):
        fan_in = m.weight.size(1)
        # Base scale 's' is conventionally 1/sqrt(fan_in) to keep variance stable
        s = 1.0 / np.sqrt(fan_in)
        limit = s * scale_multiplier
        
        # Apply U(-s, s) or U(-6s, 6s)
        nn.init.uniform_(m.weight, -limit, limit)
        if m.bias is not None:
            nn.init.uniform_(m.bias, -limit, limit)

# 5. Visualization Logic
def main():
    # Activations mapped to their PyTorch classes
    activations = {
        'ReLU': nn.ReLU,
        'GELU': nn.GELU,
        'Swish': nn.SiLU, # SiLU is PyTorch's Swish
        'SELU': nn.SELU,
        'untanh': untanh,
        'Gaussian': Gaussian,
        'Sin': Sin
    }
    
    grid = get_2d_grid(resolution=128, range_val=2.0)
    
    # --- PART 1: Replicate Image 1 (Activation Spaces) ---
    print("Generating Activation Spaces...")
    fig1, axes1 = plt.subplots(len(activations), 5, figsize=(10, 2 * len(activations)))
    fig1.suptitle("Function Spaces by Activation (Random Initializations)", fontsize=16)
    
    for row, (name, act_cls) in enumerate(activations.items()):
        for col in range(5): # 5 different random network instances
            model = RandomMLP(act_cls, hidden_dim=100, num_layers=4)
            model.apply(lambda m: init_weights_uniform(m, scale_multiplier=1.0))
            
            with torch.no_grad():
                # Pass grid through model, reshape back to 2D image
                output = model(grid).view(128, 128).numpy()
            
            ax = axes1[row, col]
            ax.imshow(output, cmap='gray')
            ax.axis('off')
            if col == 0:
                ax.set_title(name, loc='left', fontsize=14)
                
    plt.tight_layout()
    plt.show()

    # --- PART 2: Replicate Image 2 (Weight Scaling Effects) ---
    print("Generating Weight Scaling Effects...")
    fig2, axes2 = plt.subplots(2, 2, figsize=(6, 6))
    fig2.suptitle("Weight Scaling: U(-s, s) vs U(-6s, 6s)", fontsize=14)
    
    test_acts =[('ReLU', nn.ReLU), ('untanh', untanh)]
    scales =[(1.0, 'Weights ~ U(-s, s)'), (6.0, 'Weights ~ U(-6s, 6s)')]
    
    for col, (act_name, act_cls) in enumerate(test_acts):
        for row, (scale, scale_name) in enumerate(scales):
            # Using a fixed seed here so you can clearly see the transformation
            torch.manual_seed(42) 
            model = RandomMLP(act_cls, hidden_dim=100, num_layers=5)
            model.apply(lambda m: init_weights_uniform(m, scale_multiplier=scale))
            
            with torch.no_grad():
                output = model(grid).view(128, 128).numpy()
                
            ax = axes2[row, col]
            ax.imshow(output, cmap='gray')
            ax.axis('off')
            
            if row == 0:
                ax.set_title(act_name)
            if col == 0:
                ax.text(-0.1, 0.5, scale_name, va='center', ha='right', 
                        transform=ax.transAxes, fontsize=12)
                
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()