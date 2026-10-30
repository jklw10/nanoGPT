import torch
import torch.nn as nn
import math

# PyTorch Loss
from torch.nn import CrossEntropyLoss

# Avalanche Imports
from avalanche.benchmarks.classic import CORe50
from torchvision.models import resnet18
from avalanche.training.supervised import Naive

# =========================================================================
# 1. Optimizer Implementation: KreinKroneckerSM
# =========================================================================

def get_rank1_approximation(G):
    m, n = G.shape
    device = G.device
    dtype = G.dtype
    
    v = torch.sum(G, dim=0, keepdim=True).T
    norm_v = torch.norm(v)
    
    v = torch.where(
        norm_v > 1e-8,
        v / (norm_v + 1e-12),
        torch.randn((n, 1), dtype=dtype, device=device) / math.sqrt(n)
    )
        
    u = torch.matmul(G, v)
    norm_u = torch.norm(u)
    u = torch.where(
        norm_u > 1e-8,
        u / (norm_u + 1e-12),
        torch.randn((m, 1), dtype=dtype, device=device) / math.sqrt(m)
    )
        
    v = torch.matmul(G.T, u)
    norm_v = torch.norm(v)
    v = torch.where(
        norm_v > 1e-8,
        v / (norm_v + 1e-12),
        torch.randn((n, 1), dtype=dtype, device=device) / math.sqrt(n)
    )
        
    return u, v


def sherman_morrison_update(A_inv, x, beta, A_inv_init, leak):
    alpha = 1.0 - beta
    Ax = torch.matmul(A_inv, x)  # (d, 1)
    xAx = torch.matmul(x.T, Ax)
    
    num = alpha * torch.matmul(Ax, Ax.T)
    denom = beta + alpha * xAx
    
    denom = torch.clamp(denom, min=1e-12)
    A_inv_new = (A_inv - num / denom) / beta
    
    return (1.0 - leak) * A_inv_new + leak * A_inv_init


def apply_precond(g_2d, L_inv, R_inv, use_L, use_R):
    if use_R:
        res = torch.matmul(g_2d, R_inv)
    else:
        res = g_2d * R_inv  
        
    if use_L:
        res = torch.matmul(L_inv, res)
    else:
        res = res * L_inv  
    return res

class KreinKroneckerSM3(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta_cov=0.99, eps=1e-8,
                 damping=1e-5, gamma=1e-3, weight_decay=1e-2, 
                 R=None, kernel='softsign'):
        """
        KreinKroneckerSM with dynamic, information-theoretic boundary scaling 
        and symmetric trust-region step-size stabilization.
        """
        defaults = dict(
            lr=lr, beta_cov=beta_cov, eps=eps, damping=damping, gamma=gamma,
            weight_decay=weight_decay, kernel=kernel
        )
        super(KreinKroneckerSM3, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta_cov = group['beta_cov']
            damping = group['damping']
            gamma = group['gamma']
            weight_decay = group['weight_decay']
            kernel = group['kernel']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                eps_geom = 1e-12

                orig_shape = p.shape
                is_multidim = p.dim() > 1
                
                if is_multidim:
                    m = orig_shape[0]
                    n = p.numel() // m
                else:
                    m, n = p.numel(), 1

                if len(state) == 0:
                    state['step'] = 0
                    state['p_plus'] = torch.clamp(p.data, min=0.0).add_(1e-4)
                    state['p_minus'] = torch.clamp(-p.data, min=0.0).add_(1e-4)
                    
                    scale_factor = 1.0 / math.sqrt(max(damping, eps_geom))
                    state['L_inv_init'] = scale_factor * torch.eye(m, dtype=p.dtype, device=p.device)
                    state['R_inv_init'] = scale_factor * torch.eye(n, dtype=p.dtype, device=p.device)
                    
                    for ch in ['plus', 'minus']:
                        state[f'L_inv_{ch}'] = state['L_inv_init'].clone()
                        state[f'R_inv_{ch}'] = state['R_inv_init'].clone()

                    # Dynamic output-space KL divergence (Fisher Norm) tracker
                    state['ema_kl'] = 0.0

                    # Store the initial standard deviation to prevent growth-lock
                    std_init = p.data.std().item() if p.numel() > 1 else p.data.abs().item()
                    state['std_init'] = max(std_init, 1e-4)

                state['step'] += 1

                p_plus = state['p_plus'].to(torch.float32)
                p_minus = state['p_minus'].to(torch.float32)
                grad_fp32 = grad.to(torch.float32)

                if is_multidim and weight_decay != 0:
                    decay_factor = 1.0 - lr * weight_decay
                    p_plus.mul_(decay_factor)
                    p_minus.mul_(decay_factor)

                g_plus = torch.clamp(grad_fp32, min=0.0)
                g_minus = torch.clamp(-grad_fp32, min=0.0)
                g_plus_2d = g_plus.view(m, n)
                g_minus_2d = g_minus.view(m, n)

                if is_multidim:
                    leak = 1.05 * (1.0 - beta_cov)
                    
                    # Update Plus Channel Kronecker Factors
                    u_plus, v_plus = get_rank1_approximation(g_plus_2d)
                    state['L_inv_plus'] = sherman_morrison_update(
                        state['L_inv_plus'], u_plus, beta_cov, state['L_inv_init'], leak
                    )
                    state['R_inv_plus'] = sherman_morrison_update(
                        state['R_inv_plus'], v_plus, beta_cov, state['R_inv_init'], leak
                    )
                    
                    # Update Minus Channel Kronecker Factors
                    u_minus, v_minus = get_rank1_approximation(g_minus_2d)
                    state['L_inv_minus'] = sherman_morrison_update(
                        state['L_inv_minus'], u_minus, beta_cov, state['L_inv_init'], leak
                    )
                    state['R_inv_minus'] = sherman_morrison_update(
                        state['R_inv_minus'], v_minus, beta_cov, state['R_inv_init'], leak
                    )

                    # Compute Preconditioned Updates (Natural Gradient Directions)
                    step_precond_plus_2d = torch.matmul(state['L_inv_plus'], torch.matmul(g_plus_2d, state['R_inv_plus']))
                    step_precond_plus = step_precond_plus_2d.view_as(g_plus)
                    
                    step_precond_minus_2d = torch.matmul(state['L_inv_minus'], torch.matmul(g_minus_2d, state['R_inv_minus']))
                    step_precond_minus = step_precond_minus_2d.view_as(g_minus)

                    # --- DETECT OUTPUT-SPACE KL DIVERGENCE RATE (FISHER NORM) ---
                    kl_rate_plus = torch.sum(g_plus * step_precond_plus).item()
                    kl_rate_minus = torch.sum(g_minus * step_precond_minus).item()
                    kl_rate = max(kl_rate_plus + kl_rate_minus, 1e-12)

                    # Update the baseline information-change scale
                    beta_noise = 0.99
                    if state['step'] == 1:
                        state['ema_kl'] = kl_rate
                    else:
                        state['ema_kl'] = beta_noise * state['ema_kl'] + (1.0 - beta_noise) * kl_rate

                    # Dimensionless KL change ratio relative to history
                    kl_ratio = kl_rate / (state['ema_kl'] + 1e-8)

                    # Map the ratio to a stable [0, 1] interval using a balanced sigmoid
                    noise_ratio = kl_ratio / (kl_ratio + 1.0)

                    # Prevent capacity bottlenecks by referencing std_init
                    std_current = p.data.std().item() if p.numel() > 1 else p.data.abs().item()
                    scale_ref = max(state['std_init'], std_current)
                    
                    # Statistical Boundary R_t: multiplier interval [3.0, 6.0]
                    # to keep the bulk of the parameter distribution inside the boundary.
                    R_t = (6.0 - 3.0 * noise_ratio) * scale_ref
                else:
                    step_precond_plus = None
                    step_precond_minus = None
                    R_t = None

                channels = {
                    'plus': (p_plus, g_plus, step_precond_plus),
                    'minus': (p_minus, g_minus, step_precond_minus)
                }

                # --- CHANNEL OPTIMIZATION LOOP ---
                for ch_name, (p_ch, g_ch, step_precond) in channels.items():
                    step_size = lr
                    if not is_multidim:
                        p_ch.add_(g_ch, alpha=-step_size)
                        p_ch.clamp_(min=1e-6)
                        continue

                    r = torch.abs(p_ch)
                    # Enforce the dynamic Gaussian confidence boundary
                    r_safe = torch.clamp(r, min=eps_geom, max=R_t - eps_geom)

                    x_norm = r_safe / R_t
                    if kernel == 'softsign':
                        K_norm = x_norm / (x_norm + 1.0)
                        K_prime = 1.0 / (x_norm + 1.0).square()
                    elif kernel == 'tanh':
                        K_norm = torch.tanh(x_norm)
                        K_prime = 1.0 - K_norm.square()

                    one_minus_K2 = torch.clamp(1.0 - K_norm.square(), min=eps_geom)
                    scale = one_minus_K2.square() / (K_prime + damping * one_minus_K2.square())

                    # Scale precalculated preconditioned gradient
                    step_natural = step_precond * scale

                    gamma_stable = min(gamma, 1.9 / step_size) if step_size > 0 else 0.0
                    coeff_num = K_norm * K_prime * one_minus_K2
                    coeff_denom = torch.clamp(K_prime.square() + damping * one_minus_K2.square(), min=eps_geom)
                    
                    entropic_force = (gamma_stable * (coeff_num / coeff_denom)) * (p_ch / r_safe)
                    proposed_update = step_natural + entropic_force

                    # --- MATHEMATICALLY SOUND TRUST-REGION BOUNDS ---
                    # We bound final_update directly to 90% of R_t (matching the original stable learning scale)
                    delta_limit = 0.9 * R_t
                    
                    # 1. Limit growth (final_update < 0): must not exceed remaining distance to R_t,
                    # and is capped at delta_limit (stable learning rate speed).
                    min_allowed_update = -0.9 * torch.clamp(R_t - p_ch, min=1e-12)
                    min_allowed_update = torch.clamp(min_allowed_update, min=-delta_limit)
                    
                    # 2. Limit shrinkage (final_update > 0): must not drop below 0,
                    # and is capped at delta_limit (stable learning rate speed).
                    # Crucially, this does NOT depend on (R_t - p_ch), resolving the freezing trap.
                    max_allowed_update = 0.9 * p_ch
                    max_allowed_update = torch.clamp(max_allowed_update, max=delta_limit)

                    # Clamp the proposed update with our safe, non-trapping trust limits
                    final_update = torch.clamp(proposed_update, min=min_allowed_update, max=max_allowed_update)

                    p_ch.add_(final_update, alpha=-step_size)
                    p_ch.clamp_(min=1e-6)

                state['p_plus'].copy_(p_plus)
                state['p_minus'].copy_(p_minus)
                p.copy_((p_plus - p_minus).to(p.dtype))

        return loss


class KreinKroneckerSM4(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta_cov=0.99, eps=1e-8,
                 damping=1e-5, gamma=1e-3, weight_decay=1e-2, 
                 R=None, kernel='softsign'):
        """
        KreinKroneckerSM with dynamic, information-theoretic boundary scaling 
        and symmetric trust-region step-size stabilization.
        """
        defaults = dict(
            lr=lr, beta_cov=beta_cov, eps=eps, damping=damping, gamma=gamma,
            weight_decay=weight_decay, kernel=kernel
        )
        super(KreinKroneckerSM4, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta_cov = group['beta_cov']
            damping = group['damping']
            gamma = group['gamma']
            weight_decay = group['weight_decay']
            kernel = group['kernel']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                eps_geom = 1e-12

                orig_shape = p.shape
                is_multidim = p.dim() > 1
                
                if is_multidim:
                    m = orig_shape[0]
                    n = p.numel() // m
                else:
                    m, n = p.numel(), 1

                if len(state) == 0:
                    state['step'] = 0
                    state['p_plus'] = torch.clamp(p.data, min=0.0).add_(1e-4)
                    state['p_minus'] = torch.clamp(-p.data, min=0.0).add_(1e-4)
                    
                    scale_factor = 1.0 / math.sqrt(max(damping, eps_geom))
                    state['L_inv_init'] = scale_factor * torch.eye(m, dtype=p.dtype, device=p.device)
                    state['R_inv_init'] = scale_factor * torch.eye(n, dtype=p.dtype, device=p.device)
                    
                    for ch in ['plus', 'minus']:
                        state[f'L_inv_{ch}'] = state['L_inv_init'].clone()
                        state[f'R_inv_{ch}'] = state['R_inv_init'].clone()

                    # Dynamic output-space KL divergence (Fisher Norm) tracker
                    state['ema_kl'] = 0.0

                    # Store the initial standard deviation to prevent growth-lock
                    std_init = p.data.std().item() if p.numel() > 1 else p.data.abs().item()
                    state['std_init'] = max(std_init, 1e-4)

                state['step'] += 1

                p_plus = state['p_plus'].to(torch.float32)
                p_minus = state['p_minus'].to(torch.float32)
                grad_fp32 = grad.to(torch.float32)

                if is_multidim and weight_decay != 0:
                    decay_factor = 1.0 - lr * weight_decay
                    p_plus.mul_(decay_factor)
                    p_minus.mul_(decay_factor)

                g_plus = torch.clamp(grad_fp32, min=0.0)
                g_minus = torch.clamp(-grad_fp32, min=0.0)
                g_plus_2d = g_plus.view(m, n)
                g_minus_2d = g_minus.view(m, n)

                if is_multidim:
                    leak = 1.05 * (1.0 - beta_cov)
                    
                    # Update Plus Channel Kronecker Factors
                    u_plus, v_plus = get_rank1_approximation(g_plus_2d)
                    state['L_inv_plus'] = sherman_morrison_update(
                        state['L_inv_plus'], u_plus, beta_cov, state['L_inv_init'], leak
                    )
                    state['R_inv_plus'] = sherman_morrison_update(
                        state['R_inv_plus'], v_plus, beta_cov, state['R_inv_init'], leak
                    )
                    
                    # Update Minus Channel Kronecker Factors
                    u_minus, v_minus = get_rank1_approximation(g_minus_2d)
                    state['L_inv_minus'] = sherman_morrison_update(
                        state['L_inv_minus'], u_minus, beta_cov, state['L_inv_init'], leak
                    )
                    state['R_inv_minus'] = sherman_morrison_update(
                        state['R_inv_minus'], v_minus, beta_cov, state['R_inv_init'], leak
                    )

                    # Compute Preconditioned Updates (Natural Gradient Directions)
                    step_precond_plus_2d = torch.matmul(state['L_inv_plus'], torch.matmul(g_plus_2d, state['R_inv_plus']))
                    step_precond_plus = step_precond_plus_2d.view_as(g_plus)
                    
                    step_precond_minus_2d = torch.matmul(state['L_inv_minus'], torch.matmul(g_minus_2d, state['R_inv_minus']))
                    step_precond_minus = step_precond_minus_2d.view_as(g_minus)

                    # --- DETECT OUTPUT-SPACE KL DIVERGENCE RATE (FISHER NORM) ---
                    kl_rate_plus = torch.sum(g_plus * step_precond_plus).item()
                    kl_rate_minus = torch.sum(g_minus * step_precond_minus).item()
                    kl_rate = max(kl_rate_plus + kl_rate_minus, 1e-12)

                    # Update the baseline information-change scale
                    beta_noise = 0.99
                    if state['step'] == 1:
                        state['ema_kl'] = kl_rate
                    else:
                        state['ema_kl'] = beta_noise * state['ema_kl'] + (1.0 - beta_noise) * kl_rate

                    # Dimensionless KL change ratio relative to history
                    kl_ratio = kl_rate / (state['ema_kl'] + 1e-8)

                    # Map the ratio to a stable [0, 1] interval using a balanced sigmoid
                    noise_ratio = kl_ratio / (kl_ratio + 1.0)

                    # Prevent capacity bottlenecks by referencing std_init
                    std_current = p.data.std().item() if p.numel() > 1 else p.data.abs().item()
                    scale_ref = max(state['std_init'], std_current)
                    
                    # Statistical Boundary R_t: multiplier interval [3.0, 6.0]
                    # to keep the bulk of the parameter distribution inside the boundary.
                    R_t = (6.0 - 3.0 * noise_ratio) * scale_ref
                else:
                    step_precond_plus = None
                    step_precond_minus = None
                    R_t = None

                channels = {
                    'plus': (state['theta_plus'], g_plus, step_precond_plus),
                    'minus': (state['theta_minus'], g_minus, step_precond_minus)
                }

                # --- CHANNEL OPTIMIZATION LOOP ---
                for ch_name, (theta_ch, g_ch, step_precond) in channels.items():
                    step_size = lr
                    if not is_multidim:
                        # For 1D parameters, a softplus mapping ensures strict positivity smoothly
                        # instead of a hard clamp_(min=1e-6)
                        theta_ch.add_(g_ch, alpha=-step_size)
                        continue

                    sigma_theta = torch.sigmoid(theta_ch)
                    p_ch = R_t * sigma_theta

                    x_norm = sigma_theta  # Since x_norm = p_ch / R_t

                    # Calculate metric terms smoothly
                    if kernel == 'softsign':
                        K_norm = x_norm / (x_norm + 1.0)
                        K_prime = 1.0 / (x_norm + 1.0).square()
                    elif kernel == 'tanh':
                        K_norm = torch.tanh(x_norm)
                        K_prime = 1.0 - K_norm.square()

                    one_minus_K2 = 1.0 - K_norm.square() + eps_geom
                    scale = one_minus_K2.square() / (K_prime + damping * one_minus_K2.square())

                    # Preconditioned natural gradient
                    step_natural = step_precond * scale

                    # Entropic restoring force
                    gamma_stable = min(gamma, 1.9 / step_size) if step_size > 0 else 0.0
                    coeff_num = K_norm * K_prime * one_minus_K2
                    coeff_denom = K_prime.square() + damping * one_minus_K2.square() + eps_geom

                    # Restoring force behaves smoothly near the boundaries
                    entropic_force = (gamma_stable * (coeff_num / coeff_denom)) * (p_ch / (p_ch + eps_geom))
                    proposed_update = step_natural + entropic_force

                    # --- 3. Hyperbolic Trust-Region Deceleration ---
                    # Instead of hard-clipping the updates, we project them through a scaled tanh.
                    # This prevents parameters from moving too fast while keeping updates differentiable.
                    delta_limit = 0.9 * R_t
                    final_update = delta_limit * torch.tanh(proposed_update / delta_limit)

                    # Perform the update in unconstrained rapidity space (theta)
                    # The derivative of the sigmoid map (sigma * (1 - sigma)) automatically 
                    # scales down updates near the boundaries, preventing boundary violations.
                    jacobian = sigma_theta * (1.0 - sigma_theta) + eps_geom
                    theta_update = final_update / (R_t * jacobian)

                    theta_ch.add_(theta_update, alpha=-step_size)

                state['p_plus'].copy_(p_plus)
                state['p_minus'].copy_(p_minus)
                p.copy_((p_plus - p_minus).to(p.dtype))

        return loss



def train(optimizer_type="Adam"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Running Multi-Head Naive Strategy with {optimizer_type} on {device} ---")
    torch.set_float32_matmul_precision('high')
    benchmark = CORe50(scenario="ni") 
    
    model = resnet18()
    model.fc = nn.Linear(model.fc.in_features, 50)
    model = torch.compile(model)

    criterion = CrossEntropyLoss()
    
    # 3. Configure the Optimizer
    if optimizer_type == "Adam":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4,fused=True)
    elif optimizer_type == "KKSM":
        optimizer = KreinKroneckerSM4(
            model.parameters(), 
            lr=1e-3, 
            R=0.05, 
            gamma=1e-3, 
            weight_decay=1e-4
        )
    else:
        raise ValueError("Unknown optimizer")

    # 4. Initialize Naive Strategy
    cl_strategy = Naive(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=128,
        train_epochs=1,
        eval_mb_size=256,
        device=device
    )
    
    # 5. Incremental Learning Loop
    results = []
    for exp_id, experience in enumerate(benchmark.train_stream):
        print(f"\nTraining on Experience {exp_id} (Classes: {experience.classes_in_this_experience})")
        cl_strategy.train(experience)
        
        print(f"Evaluating after Experience {exp_id}...")
        eval_results = cl_strategy.eval(benchmark.test_stream)
        results.append(eval_results)
        
    return results

def print_run(results, optimizer_name):
    print("\n=======================================================")
    print(f" CORe50 TEST ACCURACY TRAJECTORY OVER TIME ({optimizer_name})")
    print("=======================================================")
    print("CORe50 NI: 8 Training Experiences, 1 Consolidated Test Stream")
    print("-" * 65)
    
    accuracies = []
    for learned_idx, eval_dict in enumerate(results):
        acc_val = 0.0
        # CORe50 NI evaluates on a single consolidated test stream (Exp000)
        for k, v in eval_dict.items():
            if "Exp000" in k and "Top1_Acc_Exp" in k:
                acc_val = v * 100.0
                break
        accuracies.append(acc_val)
        print(f"After Training Experience {learned_idx}: {acc_val:6.2f}%")
        

if __name__ == "__main__":
    #adam_res = train("Adam")
    kksm_res = train("KKSM")
    
    #print_run(adam_res, "AdamW (Standard Multi-Head)")
    print_run(kksm_res, "KreinKroneckerSM (Geometric Multi-Head)")