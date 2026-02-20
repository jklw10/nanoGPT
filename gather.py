import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Assuming you have this module installed locally
import quantizer

# --- Main Visualization Loop ---
def run_viz():
    torch.manual_seed(42)
    np.random.seed(42)
    
    N_STEPS = 5000
    N_CANDIDATES = 40
    K = 4           
    LR = 0.01       
    
    # Logits: Initialize zeros
    gate_logits = nn.Parameter(torch.zeros(1, N_CANDIDATES))
    candidate_pool = nn.Parameter(torch.randn(1, N_CANDIDATES, 2) * 0.8) 
    
    optimizer = torch.optim.Adam([gate_logits, candidate_pool], lr=LR)

    steps_per_target = 200
    fixed_targets = [[0.0, 0.0], [0.8, 0.8], [-0.8, -0.8], [0.0, 0.8]]
    
    def get_target(step):
        phase = step // steps_per_target
        if phase < len(fixed_targets):
            t = fixed_targets[phase]
            mode = "Fixed"
        else:
            rng = np.random.RandomState(phase)
            t = [rng.uniform(-0.9, 0.9), rng.uniform(-0.9, 0.9)]
            mode = "Random"
        return torch.tensor([t], dtype=torch.float32), f"{mode} Target {phase}"

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    
    print("Green  = Actual Selection (Noisy / Training)")
    print("Orange = Validation Selection (Clean / Deterministic)")
    
    for step in range(N_STEPS):
        target_pos, phase_msg = get_target(step)
        optimizer.zero_grad()
        
        gathered_candidates, selected_indices = quantizer.RLMAXgbg3.apply(
            gate_logits, candidate_pool, K, 0.1, False # True enables Gumbel noise
        )

        model_output = gathered_candidates.mean(dim=1) 
        loss = ((model_output - target_pos)**2).sum()
        loss.backward()
        
        optimizer.step()
        
        # Decay logits to prevent explosion
        with torch.no_grad():
            gate_logits.data *= 0.99 

        # --- VISUALIZATION ---
        if step % 20 == 0:
            ax.clear()
            
            # Convert basic tensors to numpy
            pool_np = candidate_pool.detach().cpu().numpy()[0]
            target_np = target_pos.numpy()[0]
            all_probs = F.softmax(gate_logits.detach(), dim=-1).numpy()[0]
            
            # 1. ACTUAL (Noisy) Selection from Kernel
            noisy_indices_np = selected_indices.detach().cpu().numpy()[0]
            noisy_com_np = pool_np[noisy_indices_np].mean(axis=0)

            # 2. VALIDATION (Clean) Selection - Derived from Logits directly
            # We select the top K logits purely (no noise)
            _, valid_indices = torch.topk(gate_logits.detach(), K, dim=-1)
            valid_indices_np = valid_indices.cpu().numpy()[0]
            valid_com_np = pool_np[valid_indices_np].mean(axis=0)

            # Setup Plot
            ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
            ax.grid(True, alpha=0.1)
            ax.set_title(f"{phase_msg}\nStep {step} | Loss {loss.item():.4f}")
            
            # Target
            ax.scatter(target_np[0], target_np[1], c='red', marker='X', s=200, zorder=20, label='Target')
            
            # All Candidates (Background)
            sizes = 20 + all_probs * 1500
            ax.scatter(pool_np[:, 0], pool_np[:, 1], s=sizes, c=all_probs, 
                       cmap='Blues', alpha=0.3, edgecolors='grey', zorder=5)
            
            # --- Draw Validation (Clean) ---
            # Using dashed Orange lines
            for idx in valid_indices_np:
                p = pool_np[idx]
                ax.plot([p[0], valid_com_np[0]], [p[1], valid_com_np[1]], 
                        c='orange', linewidth=2, linestyle='--', alpha=0.7, zorder=10)
            ax.scatter(valid_com_np[0], valid_com_np[1], c='orange', s=100, marker='s', zorder=25, label='Valid (No Noise)')

            # --- Draw Actual (Noisy) ---
            # Using solid Green lines
            for idx in noisy_indices_np:
                p = pool_np[idx]
                ax.plot([p[0], noisy_com_np[0]], [p[1], noisy_com_np[1]], 
                        c='green', linewidth=2, alpha=0.6, zorder=15)
                # Highlight selected nodes
                ax.scatter(p[0], p[1], s=150, facecolors='none', edgecolors='green', linewidth=2, zorder=16)
            
            ax.scatter(noisy_com_np[0], noisy_com_np[1], c='green', s=100, zorder=30, label='Train (Noisy)')

            ax.legend(loc='upper right')
            plt.pause(0.01)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_viz()