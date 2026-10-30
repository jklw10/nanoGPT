import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt

# =========================================================================
# STEP 1: Parse the Real Calibration CSV
# =========================================================================
def load_calibration_data(filepath="data/qbit/qbit.csv"):
    if not os.path.exists(filepath):
        if os.path.exists("qbits.csv"):
            filepath = "qbits.csv"
        else:
            raise FileNotFoundError(f"Could not find the calibration file at '{filepath}' or 'qbits.csv'")

    qubits_data = {}
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        
        for row in reader:
            if not row or len(row) < 3:
                continue
            try:
                qubit_id = int(row[0].replace('"', '').strip())
                t1 = float(row[1].replace('"', '').strip())
                t2 = float(row[2].replace('"', '').strip())
                
                gamma1 = 1.0 / t1 if t1 > 0 else 0.0
                gamma2 = 1.0 / t2 if t2 > 0 else 0.0
                gamma_phi = gamma2 - (0.5 * gamma1)
                
                qubits_data[qubit_id] = {
                    'gamma_phi': gamma_phi
                }
            except Exception:
                continue
                
    return qubits_data

# =========================================================================
# STEP 2: All-to-All Vectorized Loss (No Sparse Edges)
# =========================================================================
def compute_all_to_all_loss(coords, observed_dephasing, C0, Gamma0, d_min, epsilon, lambda_reg):
    num_nodes = coords.size(0)
    
    # 1. Pairwise distance matrix D (127, 127)
    D = torch.cdist(coords, coords)
    
    # Mask out the diagonal (self-interaction is zero)
    diagonal_mask = torch.eye(num_nodes, device=coords.device).bool()
    
    # 2. Predicted Dephasing from All-to-All Entropic Potential
    # Softening with epsilon prevents infinite gradients if nodes overlap during training
    inv_D = 1.0 / (D + epsilon)
    inv_D = torch.where(diagonal_mask, torch.zeros_like(inv_D), inv_D)
    
    pred_coupling = torch.sum(C0 * inv_D, dim=1)
    pred_total = Gamma0 + pred_coupling
    
    # Model Fit Loss
    model_loss = torch.sum((pred_total - observed_dephasing) ** 2)
    
    # 3. All-to-All Repulsion / Physical Exclusion (transmons cannot overlap)
    repulsion = torch.clamp(d_min - D, min=0.0)
    repulsion = torch.where(diagonal_mask, torch.zeros_like(repulsion), repulsion)
    repulsion_loss = torch.sum(repulsion ** 2) / 2.0
    
    # 4. Weak Central Gravity regularizer (prevents low-noise nodes from drifting to infinity)
    center_loss = torch.sum(torch.norm(coords, p=2, dim=1) ** 2)
    
    return model_loss + 20.0 * repulsion_loss + lambda_reg * center_loss

# =========================================================================
# STEP 3: Multi-Stage Optimization
# =========================================================================
def self_organize_layout(qubits_data):
    num_nodes = 127
    device = torch.device("cuda" if torch.cuda.is_available() 
                          else "mps" if torch.backends.mps.is_available() 
                          else "cpu")
    print(f"Running self-organization on device: {device}")
    
    # Process observed data
    observed_dephasing = np.zeros(num_nodes)
    for i in range(num_nodes):
        if i in qubits_data:
            observed_dephasing[i] = max(qubits_data[i]['gamma_phi'], 1e-4)
        else:
            observed_dephasing[i] = 1e-3
            
    C0 = float(np.mean(observed_dephasing))
    Gamma0 = float(np.min(observed_dephasing) * 0.5)
    
    # Hyperparameters
    d_min = 0.08      # Physical transmon footprint exclusion (80 um)
    epsilon = 0.01    # Softening parameter to avoid division-by-zero
    lambda_reg = 0.002 # Gravity coefficient to keep the cluster unified

    # 1. Initialize coordinates completely randomly in a tight cluster
    # No grids, no predefined structures.
    torch.manual_seed(42)
    initial_coords = torch.randn(num_nodes, 2) * 0.05
    coords_tensor = torch.tensor(initial_coords, dtype=torch.float32, device=device, requires_grad=True)
    observed_dephasing_t = torch.tensor(observed_dephasing, dtype=torch.float32, device=device)

    # Attempt compilation
    try:
        loss_fn = torch.compile(compute_all_to_all_loss)
        print("Compiled all-to-all loss function successfully.")
    except Exception as e:
        print(f"[INFO] Bypassed torch.compile, running eager: {e}")
        loss_fn = compute_all_to_all_loss

    # Stage 1: Coarse Untangling (High LR)
    print("\n[STAGE 1] Untangling randomly initialized nodes...")
    optimizer = torch.optim.Adam([coords_tensor], lr=0.08)
    for step in range(4000):
        optimizer.zero_grad()
        loss = loss_fn(coords_tensor, observed_dephasing_t, C0, Gamma0, d_min, epsilon, lambda_reg)
        loss.backward()
        optimizer.step()
        if (step + 1) % 100 == 0:
            print(f"  Step {step+1:3d} | Total Loss: {loss.item():.6f}")

    # Stage 2: Fine Crystallization (Low LR)
    print("\n[STAGE 2] Crystallizing positions based on local dephasing correlations...")
    optimizer = torch.optim.Adam([coords_tensor], lr=0.01)
    for step in range(4000):
        optimizer.zero_grad()
        loss = loss_fn(coords_tensor, observed_dephasing_t, C0, Gamma0, d_min, epsilon, lambda_reg)
        loss.backward()
        optimizer.step()
        if (step + 1) % 100 == 0:
            print(f"  Step {step+401:3d} | Total Loss: {loss.item():.6f}")

    final_coords = coords_tensor.detach().cpu().numpy()
    
    return initial_coords.numpy(), final_coords, observed_dephasing

# =========================================================================
# STEP 4: Run & Export Figure
# =========================================================================
def main():
    torch.set_float32_matmul_precision('high')
    try:
        qubits_data = load_calibration_data()
        print(f"Successfully loaded {len(qubits_data)} qubits from datasets/qbits.csv.")
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    init_coords, final_coords, observed_dephasing = self_organize_layout(qubits_data)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left Plot: The Initial Random Configuration
    axes[0].scatter(init_coords[:, 0], init_coords[:, 1], c='darkgray', s=35, edgecolors='k')
    axes[0].set_title("Initial Unstructured Random Configuration", fontsize=13)
    axes[0].set_xlabel("X (mm)")
    axes[0].set_ylabel("Y (mm)")
    axes[0].set_aspect('equal')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Right Plot: The Self-Organized Coordinate Map
    # Colors represent actual observed dephasing from datasets/qbits.csv
    sc = axes[1].scatter(final_coords[:, 0], final_coords[:, 1], 
                        c=observed_dephasing, cmap='plasma', 
                        edgecolors='k', s=60, zorder=3)
    axes[1].set_title("Self-Organized Layout (All-to-All Optimization)", fontsize=13)
    axes[1].set_xlabel("Reconstructed X (mm)")
    axes[1].set_ylabel("Reconstructed Y (mm)")
    axes[1].set_aspect('equal')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    cbar = fig.colorbar(sc, ax=axes[1])
    cbar.set_label("Observed Dephasing Rate $\\Gamma_\\phi$ ($\\mu s^{-1}$)", fontsize=11)

    plt.tight_layout()
    output_filename = "out/estimated_qubit_layout.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\nSaved self-organized map to: {output_filename}")
    plt.show()

if __name__ == "__main__":
    main()