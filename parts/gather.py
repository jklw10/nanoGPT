import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import parts.utils as utils

class GumbelGatherByGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate_logits, candidate_pool, k, tau=1.0, training=False, eps=1e-10):
        # --- Gumbel-Top-K Trick ---
        # Only apply noise during training
        if training:
            # Sample from Gumbel(0, 1) distribution
            gumbels = -torch.log(-torch.log(
                torch.rand_like(gate_logits) + eps
            ) + eps)
            
            # Add scaled noise to the logits
            gate_logits = (gate_logits + gumbels) / tau
        else:
            # Use original logits for deterministic evaluation
            gate_logits = gate_logits
        
        b, ct, c = candidate_pool.shape
        b, gt = gate_logits.shape
        assert gt == ct
        _, top_indices = torch.topk(gate_logits, k=k, dim=-1) #b, k
        expanded_indices = top_indices.unsqueeze(-1).expand(b, k, c)
        output = torch.gather(candidate_pool, 1, expanded_indices) #b, k, c
        ctx.save_for_backward(gate_logits, candidate_pool, output)
        ctx.k = k
        return output, top_indices

    @staticmethod
    def backward(ctx, g, _):
        gate_logits, candidate_pool, y = ctx.saved_tensors
        k = ctx.k
        grad_gate_logits = grad_candidate_pool = None

        with torch.no_grad():
            #ideal_targets = y - g#scale gradient somehow. normalize both, hyper param, rangenorm
            
            ideal_targets = y - g
            Q = F.normalize(ideal_targets)#B, K, C
            K = F.normalize(candidate_pool) #B, P, C
          
            # b,k,c dot b, c, p ->  b, k, p
            #A has shape (..., M, N)
            #B has shape (..., N, P)
            #The result will have shape (..., M, P)
            attn_scores = Q @ K.transpose(-2, -1) # b, K, pool
            attn_scores = F.softmax(attn_scores, dim=-1)
            scoresum = attn_scores.sum(dim=1) # b, p
            _, best_candidate_indices = torch.topk(scoresum, k=k, dim=-1) 
           
        
        if ctx.needs_input_grad[0]:
            # The goal is to make the model's logits produce these top_overall_indices.
            hard_target = torch.zeros_like(gate_logits) # Shape (b, p)
            #the indices should be in range for this
            hard_target.scatter_(dim=1, index=best_candidate_indices, value=1.0)
            
            # Use a stable, cross-entropy-like gradient.
            #probs = F.softmax(gate_logits, dim=-1)
            probs = F.sigmoid(gate_logits)
            grad_gate_logits = probs - hard_target
        if ctx.needs_input_grad[1]:
            b,p,c = candidate_pool.shape
            
        
            ideal_indices = best_candidate_indices.unsqueeze(-1).expand(b, k, c)
            
            y_ideal = torch.gather(candidate_pool, 1, ideal_indices)
            delta = y - y_ideal
            g_corrected = g - delta
            grad_candidate_pool = torch.zeros_like(candidate_pool)
            grad_candidate_pool.scatter_add_(1, ideal_indices, g_corrected)
        
        
        return grad_gate_logits, grad_candidate_pool, None, None, None


class RLMAXgbg(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate_logits, candidate_pool, k, tau=1.0, training=False, sigma=0.3):
        
        if training:
            gumbels = -torch.log(-torch.log(
                torch.rand_like(gate_logits) + 1e-10
            ) + 1e-10)
            
            gate_logits = (gate_logits + gumbels) / tau
        else:
            gate_logits = gate_logits
        
        b, ct, c = candidate_pool.shape
        b, gt = gate_logits.shape
        assert gt == ct
        _, top_indices = torch.topk(gate_logits, k=k, dim=-1) #b, k
        expanded_indices = top_indices.unsqueeze(-1).expand(b, k, c)
        output = torch.gather(candidate_pool, 1, expanded_indices) #b, k, c
        ctx.save_for_backward(gate_logits, candidate_pool, output,top_indices)
        ctx.k = k
        ctx.sigma = sigma
        return output, top_indices

    @staticmethod
    def backward(ctx, grad_output, _):
        gate_logits, candidate_pool, output, top_indices = ctx.saved_tensors
        k = ctx.k
        sigma = ctx.sigma
        grad_gate_logits = grad_candidate_pool = None

        with torch.no_grad():
            ideal_targets = output - grad_output 
            
            
            dist_sq = ((output - ideal_targets)**2).sum(dim=-1, keepdim=True)
            mask = torch.exp(-dist_sq / (2 * sigma**2 + 1e-5))

            
            dot_products = torch.bmm(ideal_targets, candidate_pool.transpose(1, 2))
            candidate_norm_sq = torch.sum(candidate_pool ** 2, dim=-1).unsqueeze(1)
            scores = 2 * dot_products - candidate_norm_sq
            
            _, best_candidate_indices = torch.max(scores, dim=-1) # (b, k)
           
        
        if ctx.needs_input_grad[0]:
            hard_target = torch.zeros_like(gate_logits) 
            hard_target.scatter_(dim=1, index=best_candidate_indices, value=1.0)
            
            probs = torch.sigmoid(gate_logits)
            grad_gate_logits = probs - hard_target
        if ctx.needs_input_grad[1]:
            b,p,c = candidate_pool.shape
            
            
            is_correct_selection = (top_indices == best_candidate_indices).float()

            # Expand mask for dimensions (b, k, dim)
            mask = is_correct_selection.unsqueeze(-1)

            # Zero out gradients for the "Wrong" selections
            masked_grad = grad_output * mask

            # Scatter the masked gradient
            expanded_indices = top_indices.unsqueeze(-1).expand(b, k, c)
            grad_candidate_pool = torch.zeros_like(candidate_pool)
            grad_candidate_pool.scatter_add_(1, expanded_indices, masked_grad)
        
        
        return grad_gate_logits, grad_candidate_pool, None, None, None
    
class RLMAXgbg2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate_logits, candidate_pool, k, tau=1.0, training=False, sigma=0.5):
        
        ctx.sigma = sigma
        if training:
            gumbels = -torch.log(-torch.log(torch.rand_like(gate_logits) + 1e-10) + 1e-10)
            noisy_logits = (gate_logits + gumbels) / tau
        else:
            noisy_logits = gate_logits
        b, ct, c = candidate_pool.shape
        _, top_indices = torch.topk(noisy_logits, k=k, dim=-1)
        expanded_indices = top_indices.unsqueeze(-1).expand(b, k, c)
        output = torch.gather(candidate_pool, 1, expanded_indices) 
        ctx.save_for_backward(gate_logits, candidate_pool, output, top_indices)
        return output, top_indices

    @staticmethod
    def backward(ctx, grad_output, _):
        gate_logits, candidate_pool, output, top_indices = ctx.saved_tensors
        sigma = ctx.sigma
        
        grad_gate_logits = grad_candidate_pool = None

        ideal_pos = output - grad_output 
        # --- GATE GRADIENT (The "Hindsight" Switch) ---
        if ctx.needs_input_grad[0]:
            target_norm = (ideal_pos**2).sum(-1).unsqueeze(2) # (b, k, 1)
            cand_norm   = (candidate_pool**2).sum(-1).unsqueeze(1) # (b, 1, n)
            dot         = torch.bmm(ideal_pos, candidate_pool.transpose(1, 2)) # (b, k, n)
            
            dists = target_norm + cand_norm - 2*dot # (b, k, n)
            
            _, best_indices = torch.min(dists, dim=-1)
            
            target_gate = torch.zeros_like(gate_logits)
            target_gate.scatter_(1, best_indices, 1.0)
            
            probs = torch.sigmoid(gate_logits)
            grad_gate_logits = probs - target_gate
        if ctx.needs_input_grad[1]:
            b, k, c = ideal_pos.shape
            
            dist_sq = ((output - ideal_pos)**2).sum(dim=-1, keepdim=True) # (b, k, 1)
            mask = torch.exp(-dist_sq / (2 * sigma**2 + 1e-5))
            
            masked_grad = grad_output * mask
            
            b, n, c = candidate_pool.shape
            expanded_indices = top_indices.unsqueeze(-1).expand(b, k, c)
            grad_candidate_pool = torch.zeros_like(candidate_pool)
            grad_candidate_pool.scatter_add_(1, expanded_indices, masked_grad)

        

        return grad_gate_logits, grad_candidate_pool, None, None, None, None

class RLMAXgbg3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate_logits, candidate_pool, k, tau=1.0, training=False, sigma=0.5):
        
        ctx.sigma = sigma
        ctx.k = k
        if training:
            gumbels = -torch.log(-torch.log(torch.rand_like(gate_logits) + 1e-10) + 1e-10)
            noisy_logits = (gate_logits + gumbels) / tau
        else:
            noisy_logits = gate_logits
        b, ct, c = candidate_pool.shape
        _, top_indices = torch.topk(noisy_logits, k=k, dim=-1)
        expanded_indices = top_indices.unsqueeze(-1).expand(b, k, c)
        output = torch.gather(candidate_pool, 1, expanded_indices) 
        ctx.save_for_backward(gate_logits, candidate_pool, output, top_indices)
        return output, top_indices

    @staticmethod
    def backward(ctx, grad_output, _):
        gate_logits, candidate_pool, output, top_indices = ctx.saved_tensors
        #candidate_pool = utils.rms(candidate_pool) 
        sigma = ctx.sigma
        k = ctx.k
        grad_gate_logits = grad_candidate_pool = None
        
        #grad_norm = grad_output.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        #clamp_factor = torch.tanh(grad_norm) / grad_norm 
        #grad_output = grad_output  * clamp_factor # Optional: Uncomment if unstable
        grad_output = utils.rms_norm(grad_output)
        ideal_pos = output - grad_output 
        if ctx.needs_input_grad[0]:
            target_norm = (ideal_pos**2).sum(-1).unsqueeze(2) # (b, k, 1)
            cand_norm   = (candidate_pool**2).sum(-1).unsqueeze(1) # (b, 1, n)
            #candidate_pool = utils.rms(candidate_pool) 
            dot         = torch.bmm(ideal_pos, candidate_pool.transpose(1, 2)) # (b, k, n)
            
            dists = target_norm + cand_norm - 2*dot # (b, k, n)
            
            b_dist, best_indices = torch.min(dists, dim=-1)
            #b_dist, best_indices = torch.max(dot, dim=-1)
            
            target_gate = torch.zeros_like(gate_logits)
            target_gate.scatter_(1, best_indices, 1.0)
            
            probs = torch.sigmoid(gate_logits)
            grad_gate_logits = probs - target_gate
        if ctx.needs_input_grad[1]:
            b,p,c = candidate_pool.shape
            
        
            ideal_indices = best_indices.unsqueeze(-1).expand(b, k, c)
            
            y_ideal = torch.gather(candidate_pool, 1, ideal_indices)
            delta = output - y_ideal
            g_corrected = grad_output - delta 
            g_corrected = g_corrected / (b_dist.unsqueeze(-1)+1.0)
            #g_corrected = torch.tanh(g_corrected)
            grad_candidate_pool = torch.zeros_like(candidate_pool)
            grad_candidate_pool.scatter_add_(1, ideal_indices, g_corrected)
        
        return grad_gate_logits, grad_candidate_pool, None, None, None, None

class HungryGatherFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        gate_logits, 
        candidate_pool, 
        k, 
        tau=1.0, 
        training=False, 
        lookahead_lr    = 0.2,
        magnet_strength = 0.0,):
        # Forward is standard Gumbel Top-K
        if training:
            gumbels = -torch.log(-torch.log(torch.rand_like(gate_logits) + 1e-10) + 1e-10)
            gate_logits = (gate_logits + gumbels) / tau
        else:
            gate_logits = gate_logits
        b, ct, c = candidate_pool.shape
        _, top_indices = torch.topk(gate_logits, k=k, dim=-1) # b, k
        
        # Save indices for standard backprop path
        ctx.save_for_backward(gate_logits, candidate_pool, top_indices)
        ctx.k = k
        ctx.magnet_strength=magnet_strength
        ctx.lookahead_lr=lookahead_lr
        # Gather the actual output
        expanded_indices = top_indices.unsqueeze(-1).expand(b, k, c)
        output = torch.gather(candidate_pool, 1, expanded_indices) 
        
        return output, top_indices

    @staticmethod
    def backward(ctx, g,_):
        gate_logits, candidate_pool, top_indices, = ctx.saved_tensors
        k = ctx.k
        
        magnet_strength = ctx.magnet_strength
        lookahead_lr    = ctx.lookahead_lr
        b, p, c = candidate_pool.shape
        
        grad_gate = torch.zeros_like(gate_logits)
        grad_pool = torch.zeros_like(candidate_pool)
        
        expanded_indices = top_indices.unsqueeze(-1).expand(b, k, c)
        grad_pool.scatter_add_(1, expanded_indices, g)

        with torch.no_grad():
            y = torch.gather(candidate_pool, 1, expanded_indices)
            
            ideal_target = y - F.normalize(g) * lookahead_lr #lr

            Q = F.normalize(ideal_target)   # b, k, c
            K = F.normalize(candidate_pool)   # b, p, c
            
            sim_scores = torch.bmm(Q, K.transpose(1, 2))
            
            _, best_match_indices = torch.max(sim_scores, dim=-1) # b, k
        
        target_gate = torch.zeros_like(gate_logits)
        target_gate.scatter_(1, best_match_indices, 1.0)
        
        probs = F.sigmoid(gate_logits)
        grad_gate = probs - target_gate
        
        expanded_best = best_match_indices.unsqueeze(-1).expand(b, k, c)
        best_vectors = torch.gather(candidate_pool, 1, expanded_best)
        
        magnetic_force = (best_vectors - ideal_target)
        
        grad_pool.scatter_add_(1, expanded_best, magnetic_force * magnet_strength)

        return grad_gate, grad_pool, None, None, None, None

class HungryGatherFunc2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate_logits, candidate_pool, k, tau=1.0, training=False, repulsion=0.1):
        # 1. Gumbel Noise (Exploration)
        if training:
            gumbels = -torch.log(-torch.log(torch.rand_like(gate_logits) + 1e-10) + 1e-10)
            noisy_logits = (gate_logits + gumbels)/ tau 
        else:
            noisy_logits = gate_logits
            
        b, num_candidates, dim = candidate_pool.shape
        
        # 2. Select Top-K
        _, top_indices = torch.topk(noisy_logits, k=k, dim=-1) # (b, k)
        
        # 3. Save for Backward
        ctx.k = k
        ctx.repulsion = repulsion
        
        # 4. Gather Output
        expanded_indices = top_indices.unsqueeze(-1).expand(b, k, dim)
        output = torch.gather(candidate_pool, 1, expanded_indices) 
        
        ctx.save_for_backward(gate_logits, candidate_pool, top_indices, output)
        return output, top_indices

    @staticmethod
    def backward(ctx, g,_):
        gate_logits, candidate_pool, top_indices, output = ctx.saved_tensors
        k = ctx.k
        repulsion = ctx.repulsion
        b, num_candidates, dim = candidate_pool.shape
        
        grad_gate = torch.zeros_like(gate_logits)
        grad_pool = torch.zeros_like(candidate_pool)
        
        expanded_indices = top_indices.unsqueeze(-1).expand(b, k, dim)
        grad_pool.scatter_add_(1, expanded_indices, g)

        with torch.no_grad():
            y_selected = torch.gather(candidate_pool, 1, expanded_indices)
            
            estimated_target = y_selected - g 

            dists = torch.norm(estimated_target.unsqueeze(2) - candidate_pool.unsqueeze(1), dim=-1)
            _, best_match_indices = torch.min(dists, dim=-1) 
            
        target_gate = torch.zeros_like(gate_logits)
        target_gate.scatter_(1, best_match_indices, 1.0)
        probs = F.sigmoid(gate_logits)
        grad_gate = probs - target_gate
        
        expanded_indices = top_indices.unsqueeze(-1).expand(b, k, dim)
        
        if repulsion > 0.0:
            repulsion_grad = torch.zeros_like(output)

            for i in range(k):
                for j in range(k):
                    if i == j: 
                        continue
                    diff = output[:, i] - output[:, j] # (b, dim)
                    dist_sq = (diff**2).sum(dim=-1, keepdim=True) + 1e-6
                    force = diff / dist_sq 
                    repulsion_grad[:, i] += force #could be just noise, change if expensive.

            grad_pool.scatter_add_(1, expanded_indices, -repulsion_grad * repulsion)

        return grad_gate, grad_pool, None, None, None, None



class GatherByGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate_logits, candidate_pool, k):
        
        b, ct, c = candidate_pool.shape
        b, gt = gate_logits.shape
        assert gt == ct
        _, top_indices = torch.topk(gate_logits, k=k, dim=-1) #b, k
        expanded_indices = top_indices.unsqueeze(-1).expand(b, k, c)
        output = torch.gather(candidate_pool, 1, expanded_indices) #b, k, c
        ctx.save_for_backward(gate_logits, candidate_pool, output)
        ctx.k = k
        return output,top_indices

    @staticmethod
    def backward(ctx, g,_):
        gate_logits, candidate_pool, y = ctx.saved_tensors
        k = ctx.k
        grad_gate_logits = grad_candidate_pool = None

        with torch.no_grad():
            ideal_targets = y - g#*0.5
            Q = ideal_targets #B, K, C
            K = candidate_pool #B, P, C
            # b,k,c dot b, c, p ->  b, k, p
            #A has shape (..., M, N)
            #B has shape (..., N, P)
            #The result will have shape (..., M, P)
            attn_scores = Q @ K.transpose(-2, -1) # b, K, pool
            attn_scores = F.softmax(attn_scores, dim=-1)
            scoresum = attn_scores.sum(dim=1) # b, p
            _, best_candidate_indices = torch.topk(scoresum, k=k, dim=-1) 
           
        
        if ctx.needs_input_grad[0]:
            T = 0.25 # A hyperparameter to tune.
            soft_target = F.softmax(scoresum / T, dim=-1)

            # The gradient is the standard cross-entropy gradient. (bce works too, not tested enough)
            current_probs = F.sigmoid(gate_logits)
            grad_gate_logits = current_probs - soft_target.detach()
        if ctx.needs_input_grad[1]:
            b,p,c = candidate_pool.shape
            
        
            ideal_indices = best_candidate_indices.unsqueeze(-1).expand(b, k, c)
            
            y_ideal = torch.gather(candidate_pool, 1, ideal_indices)
            delta = y - y_ideal
            g_corrected = g - delta * 0.5 # A hyperparameter to tune.
            grad_candidate_pool = torch.zeros_like(candidate_pool)
            grad_candidate_pool.scatter_add_(1, ideal_indices, g_corrected)
        
        
        return grad_gate_logits, grad_candidate_pool, None

class GatherByGate2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate_logits, candidate_pool, k):
        
        b, ct, c = candidate_pool.shape
        b, gt = gate_logits.shape
        assert gt == ct
        _, top_indices = torch.topk(gate_logits, k=k, dim=-1) #b, k
        expanded_indices = top_indices.unsqueeze(-1).expand(b, k, c)
        output = torch.gather(candidate_pool, 1, expanded_indices) #b, k, c
        ctx.save_for_backward(gate_logits, candidate_pool, output, top_indices)
        ctx.k = k
        return output

    @staticmethod
    def backward(ctx, g):
        gate_logits, candidate_pool, y, top_indices = ctx.saved_tensors
        k = ctx.k
        grad_gate_logits = grad_candidate_pool = None
        
        b, p, c = candidate_pool.shape

        with torch.no_grad():
            ideal_targets = y - g#*0.5
            Q = ideal_targets #B, K, C
            K = candidate_pool #B, P, C
            # b,k,c dot b, c, p ->  b, k, p
            #A has shape (..., M, N)
            #B has shape (..., N, P)
            #The result will have shape (..., M, P)
            attn_scores = Q @ K.transpose(-2, -1) # b, K, pool
            attn_scores = F.softmax(attn_scores, dim=-1)
            scoresum = attn_scores.sum(dim=1) # b, p
            _, best_indices = torch.topk(scoresum, k=k, dim=-1) 
           
        
        if ctx.needs_input_grad[0]:
            hard_target = torch.zeros_like(gate_logits)
            hard_target.scatter_(1, best_indices, 1.0)

            current_probs = F.sigmoid(gate_logits)
            grad_gate_logits = current_probs - hard_target.detach()
        if ctx.needs_input_grad[1]:
            grad_pool = torch.zeros_like(candidate_pool)
            
            expanded_indices = top_indices.unsqueeze(-1).expand(b, k, c)
            grad_pool.scatter_add_(1, expanded_indices, g)
            
            ideal_best_indices = best_indices.unsqueeze(-1).expand(b, k, c)
            y_ideal = torch.gather(candidate_pool, 1, ideal_best_indices)
            
            delta = y - y_ideal
            correction = -delta * 0.5 # Tuning param
            
            grad_pool.scatter_add_(1, ideal_best_indices, correction)
        
        
        return grad_gate_logits, grad_candidate_pool, None


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
        
        gathered_candidates, selected_indices = RLMAXgbg3.apply(
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


class ScatterByGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate_logits, patches, output_len):
        batch_size, num_patches, dim = patches.shape
        
        
        _, indices = torch.topk(gate_logits, k=num_patches, dim=-1)

        output = torch.zeros(batch_size, output_len, dim, device=patches.device, dtype=patches.dtype)
        
        expanded_indices = indices.unsqueeze(-1).expand_as(patches)
        output.scatter_add_(1, expanded_indices, patches)
        
        ctx.save_for_backward(gate_logits, patches, indices)
        return output

    
    @staticmethod
    def backward(ctx, grad_output):
        gate_logits, patches, indices = ctx.saved_tensors
        grad_gate_logits = grad_patches = None

        if ctx.needs_input_grad[1]:
            expanded_indices = indices.unsqueeze(-1).expand_as(patches)
            grad_patches = torch.gather(grad_output, 1, expanded_indices)

        if ctx.needs_input_grad[0]:
            grad_gate_logits = torch.zeros_like(gate_logits)
            
            grad_at_scatter_locs = torch.gather(grad_output, 1, indices.unsqueeze(-1).expand_as(patches))
            scores_for_logits = (grad_at_scatter_locs * patches).sum(-1) # Shape: (batch, k)
            
            grad_gate_logits.scatter_(dim=1, index=indices, src=scores_for_logits)
            
        return grad_gate_logits, grad_patches, None

def run_seed_viz():
    torch.manual_seed(42)
    
    # Config
    SEQ_LEN = 100
    N_SEEDS = 12       # Only 12 points to reconstruct 100!
    DIM = 1
    LR_GATE = 0.05
    LR_PATCH = 0.05
    STEPS = 300
    
    # 1. Create a Target Signal (A nice wavy curve)
    x = torch.linspace(0, 4*np.pi, SEQ_LEN).view(1, SEQ_LEN, 1)
    target = torch.sin(x) + 0.5 * torch.cos(3*x)
    
    # 2. Learnable Params
    # Gate: Initialize uniform-ish so it explores
    gate_logits = nn.Parameter(torch.randn(1, SEQ_LEN) * 0.1)
    
    # Patches: The "Ink" values. 
    # We initialize them random. Some might become positive "Peak" seeds, others negative "Trough" seeds.
    # We give them a bit of magnitude to start.
    patches = nn.Parameter(torch.randn(1, N_SEEDS, DIM))
    
    optimizer = torch.optim.Adam([
        {'params': gate_logits, 'lr': LR_GATE},
        {'params': patches, 'lr': LR_PATCH}
    ])
    
    # 3. Diffusion Layer (Fixed Gaussian Blur)
    # This simulates "diffusing from there"
    # A generic Gaussian kernel
    kernel_size = 15
    sigma = 3.0
    k_range = torch.arange(kernel_size).float() - (kernel_size-1)/2
    kernel = torch.exp(-0.5 * (k_range / sigma)**2)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size) # (Out, In, K)
    
    diffusion_layer = nn.Conv1d(1, 1, kernel_size, padding=kernel_size//2, bias=False)
    diffusion_layer.weight.data = kernel
    diffusion_layer.requires_grad_(False) # Fixed physics
    
    # --- Live Viz ---
    plt.ion()
    fig, (ax_sig, ax_gate) = plt.subplots(2, 1, figsize=(10, 8))
    
    print(f"Task: Reconstruct signal using only {N_SEEDS} seeds + Blur.")
    
    for step in range(STEPS):
        optimizer.zero_grad()
        
        # 1. Scatter (Place Seeds)
        sparse_canvas = ScatterByGate.apply(gate_logits, patches, SEQ_LEN)
        
        # 2. Diffuse (Blur)
        # Conv1d expects (Batch, Dim, Seq)
        sparse_reshaped = sparse_canvas.permute(0, 2, 1) 
        dense_prediction = diffusion_layer(sparse_reshaped)
        dense_prediction = dense_prediction.permute(0, 2, 1) # Back to (B, L, D)
        
        # 3. Loss
        loss = ((dense_prediction - target)**2).sum()
        
        loss.backward()
        optimizer.step()
        
        if step % 5 == 0:
            ax_sig.clear()
            ax_gate.clear()
            
            # Extract Data
            t_np = target.detach().numpy()[0, :, 0]
            pred_np = dense_prediction.detach().numpy()[0, :, 0]
            
            # Find where the seeds actually are
            # We reconstruct the sparse locations for plotting
            _, indices = torch.topk(gate_logits, k=N_SEEDS, dim=-1)
            active_indices = indices[0].detach().numpy()
            active_values = patches[0, :, 0].detach().numpy()
            
            # --- Plot 1: Signal Reconstruction ---
            ax_sig.set_title(f"Step {step} | Loss {loss.item():.4f}")
            ax_sig.set_ylim(-2, 2)
            
            # Target
            ax_sig.plot(t_np, 'k--', label='Target Signal', alpha=0.5)
            
            # Prediction (Diffused)
            ax_sig.plot(pred_np, 'g-', linewidth=2, label='Diffused Output', alpha=0.8)
            
            ax_sig.scatter(active_indices, active_values, c='red', s=100, zorder=10, label='Seeds', edgecolors='white')
            
            # Draw stems for seeds
            for x, y in zip(active_indices, active_values):
                ax_sig.plot([x, x], [0, y], 'r-', alpha=0.3)
                
            ax_sig.legend(loc='lower right')
            ax_sig.grid(True, alpha=0.3)
            
            # --- Plot 2: Gate Logic ---
            ax_gate.set_title("Gate Logits (Where to place seeds)")
            logits_np = gate_logits.detach().numpy()[0]
            ax_gate.bar(range(SEQ_LEN), logits_np, color='skyblue')
            
            # Highlight chosen spots
            ax_gate.scatter(active_indices, logits_np[active_indices], c='red', zorder=5)
            ax_gate.set_xlabel("Position Index")
            
            plt.draw()
            plt.pause(0.01)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    run_viz()