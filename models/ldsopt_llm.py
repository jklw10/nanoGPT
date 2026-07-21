import math
import os
import time

from attr import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from parts.modules import HungryLinear, HungryPattLayer, PattLayer
import parts.utils as utils
torch._inductor.config.disable_cpp_codegen = True

torch._dynamo.config.force_parameter_static_shapes = False
Linear          = HungryLinear
Patt            = HungryPattLayer
dataset         = "openwebtext"
data_dir        = os.path.join('data', dataset)

device          = 'cuda' if torch.cuda.is_available() else 'cpu'
bias            = False

batch_size      = 64
block_size      = 64

pattcount       = 512

n_embd          = 256
n_head          = 8
n_layer         = 4
dropout         = 0.0
weight_decay    = 1e-1

eval_interval   = 1000
eval_iters      = 200
warmup_iters    = 100
max_iters       = 50000
lr_decay_iters  = 30000
decay_iters     = 5000

learning_rate   = 1e-4
min_lr          = learning_rate / 10

spl             = False
soloblock       = False
sspl            = False
wnll            = False
patt            = False
gen             = True
lr_ramp         = False
always_byte     = False


config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
train_logging = False
exec(open('training_tools/configurator.py').read()) # overrides from command line or config file

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
    """
    Updates the inverse matrix A_inv under a rank-1 update with diagonal leakage
    to prevent estimator wind-up and preserve positive-definiteness.
    """
    alpha = 1.0 - beta
    Ax = torch.matmul(A_inv, x)  # (d, 1)
    xAx = torch.matmul(x.T, Ax)
    
    num = alpha * torch.matmul(Ax, Ax.T)
    denom = beta + alpha * xAx
    
    denom = torch.clamp(denom, min=1e-12)
    A_inv_new = (A_inv - num / denom) / beta
    
    return (1.0 - leak) * A_inv_new + leak * A_inv_init


class KreinKroneckerSM(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, beta_cov=0.999, eps=1e-8,
                 damping=1e-5, gamma=1e-5, weight_decay=1e-5, 
                 R=2.5, mode='hypercube', kernel='softsign', max_kron_dim=2048):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= beta_cov < 1.0:
            raise ValueError(f"Invalid beta_cov: {beta_cov}")
        if damping < 0.0:
            raise ValueError(f"Invalid damping: {damping}")
        if gamma < 0.0:
            raise ValueError(f"Invalid gamma: {gamma}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if R <= 0.0:
            raise ValueError(f"Invalid boundary scale R: {R}")
        
        mode_lower = mode.lower()
        if mode_lower not in ('sphere', 'hypercube'):
            raise ValueError(f"Unknown mode: {mode}. Choose 'sphere' or 'hypercube'.")

        defaults = dict(
            lr=lr, 
            beta_cov=beta_cov, 
            eps=eps, 
            damping=damping, 
            gamma=gamma,
            weight_decay=weight_decay, 
            R=R,
            mode=mode_lower,
            kernel=kernel,
            max_kron_dim=max_kron_dim
        )
        super(KreinKroneckerSM, self).__init__(params, defaults)
    
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
            R = group['R']
            mode = group['mode']
            kernel = group['kernel']
            max_kron_dim = group['max_kron_dim']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("KreinKroneckerSM does not support sparse gradients.")

                state = self.state[p]
                eps_geom = 1e-12

                # 1. State and Dual-Channel Initialization
                orig_shape = p.shape
                is_multidim = p.dim() > 1
                
                if is_multidim:
                    m = orig_shape[0]
                    n = p.numel() // m
                else:
                    m, n = p.numel(), 1

                if len(state) == 0:
                    state['step'] = 0
                    # Initialize dual channels
                    state['p_plus'] = torch.clamp(p.data, min=0.0).add_(1e-4)
                    state['p_minus'] = torch.clamp(-p.data, min=0.0).add_(1e-4)
                    
                    scale_factor = 1.0 / math.sqrt(max(damping, eps_geom))
                    
                    # Decouple preconditioning for dimensions exceeding the threshold
                    state['use_L'] = (m <= max_kron_dim)
                    state['use_R'] = (n <= max_kron_dim)
                    
                    # Allocate actual 2D matrices or store scalars
                    if state['use_L']:
                        state['L_inv_init'] = scale_factor * torch.eye(m, dtype=torch.float32, device=p.device)
                    else:
                        state['L_inv_init'] = scale_factor  # Scalar float
                        
                    if state['use_R']:
                        state['R_inv_init'] = scale_factor * torch.eye(n, dtype=torch.float32, device=p.device)
                    else:
                        state['R_inv_init'] = scale_factor  # Scalar float
                    
                    for ch in ['plus', 'minus']:
                        if state['use_L']:
                            state[f'L_inv_{ch}'] = state['L_inv_init'].clone()
                        else:
                            state[f'L_inv_{ch}'] = scale_factor
                            
                        if state['use_R']:
                            state[f'R_inv_{ch}'] = state['R_inv_init'].clone()
                        else:
                            state[f'R_inv_{ch}'] = scale_factor

                state['step'] += 1

                p_plus = state['p_plus'].to(torch.float32)
                p_minus = state['p_minus'].to(torch.float32)
                grad_fp32 = grad.to(torch.float32)

                # 2. Decoupled Weight Decay on Channels
                if is_multidim and weight_decay != 0:
                    decay_factor = 1.0 - lr * weight_decay
                    p_plus.mul_(decay_factor)
                    p_minus.mul_(decay_factor)

                # 3. Gradient Splitting
                g_plus = torch.clamp(grad_fp32, min=0.0)
                g_minus = torch.clamp(-grad_fp32, min=0.0)

                g_plus_2d = g_plus.view(m, n)
                g_minus_2d = g_minus.view(m, n)

                # 4. Update the Preconditioning Inverse Matrices via Sherman-Morrison
                if is_multidim:
                    leak = 1.05 * (1.0 - beta_cov)
                    
                    # Plus Channel
                    u_plus, v_plus = get_rank1_approximation(g_plus_2d)
                    if state['use_L']:
                        state['L_inv_plus'] = sherman_morrison_update(
                            state['L_inv_plus'], u_plus, beta_cov, state['L_inv_init'], leak
                        )
                    if state['use_R']:
                        state['R_inv_plus'] = sherman_morrison_update(
                            state['R_inv_plus'], v_plus, beta_cov, state['R_inv_init'], leak
                        )
                    
                    # Minus Channel
                    u_minus, v_minus = get_rank1_approximation(g_minus_2d)
                    if state['use_L']:
                        state['L_inv_minus'] = sherman_morrison_update(
                            state['L_inv_minus'], u_minus, beta_cov, state['L_inv_init'], leak
                        )
                    if state['use_R']:
                        state['R_inv_minus'] = sherman_morrison_update(
                            state['R_inv_minus'], v_minus, beta_cov, state['R_inv_init'], leak
                        )

                # 5. Independent Geometric/Entropic Updates
                channels = {
                    'plus': (p_plus, g_plus, g_plus_2d, state['L_inv_plus'], state['R_inv_plus']),
                    'minus': (p_minus, g_minus, g_minus_2d, state['L_inv_minus'], state['R_inv_minus'])
                }

                for ch_name, (p_ch, g_ch, g_ch_2d, L_inv, R_inv) in channels.items():
                    step_size = lr

                    # 1D fallback
                    if not is_multidim:
                        p_ch.add_(g_ch, alpha=-step_size)
                        p_ch.clamp_(min=1e-6)
                        continue

                    # --- MODE 1: HYPERSPHERE GEOMETRY ---
                    if mode == 'sphere':
                        dims = list(range(1, p.dim()))
                        r = torch.norm(p_ch, p=2, dim=dims, keepdim=True)
                        r_clamped = torch.clamp(r, max=R - eps_geom)
                        r_safe = torch.clamp(r_clamped, min=eps_geom)
                        w = p_ch / r_safe

                        # Jordan Symmetrized Term 1: H(A(g))
                        step_precond_2d = apply_precond(g_ch_2d, L_inv, R_inv, state['use_L'], state['use_R'])
                        step_precond = step_precond_2d.view_as(p_ch)
                        dot_precond = torch.sum(w * step_precond, dim=dims, keepdim=True)
                        step_nat_forward = step_precond - dot_precond * w

                        # Jordan Symmetrized Term 2: A(H(g))
                        dot_g = torch.sum(w * g_ch, dim=dims, keepdim=True)
                        g_projected = g_ch - dot_g * w
                        g_proj_2d = g_projected.view(m, n)
                        step_nat_reverse_2d = apply_precond(g_proj_2d, L_inv, R_inv, state['use_L'], state['use_R'])
                        step_nat_reverse = step_nat_reverse_2d.view_as(p_ch)

                        step_natural = 0.5 * (step_nat_forward + step_nat_reverse)

                        # Metric Scales and Kernel parameters
                        x_norm = r_safe / R
                        if kernel == 'softsign':
                            K_norm = x_norm / (x_norm + 1.0)
                        elif kernel == 'tanh':
                            K_norm = torch.tanh(torch.clamp(x_norm, max=20.0))
                        else:
                            raise ValueError(f"Unknown kernel: {kernel}")

                        gamma_stable = min(gamma, 1.9 / step_size) if step_size > 0 else 0.0
                        entropic_force = gamma_stable * w * (1.0 + x_norm.square())

                        proposed_update = step_natural + entropic_force

                        radial_step = torch.sum(w * proposed_update, dim=dims, keepdim=True)
                        tangent_step = proposed_update - radial_step * w
                        
                        max_radial_step = 0.9 * (R - r_safe)
                        scale_radial = torch.where(
                            radial_step > 0,
                            torch.clamp(max_radial_step / torch.clamp(radial_step, min=eps_geom), max=1.0),
                            torch.ones_like(radial_step)
                        )
                        final_update = (radial_step * scale_radial) * w + tangent_step

                    # --- MODE 2: HYPERCUBE GEOMETRY ---
                    else:
                        r = torch.abs(p_ch)
                        r_safe = torch.clamp(r, min=eps_geom, max=R - eps_geom)

                        x_norm = r_safe / R
                        if kernel == 'softsign':
                            K_norm = x_norm / (x_norm + 1.0)
                            K_prime = 1.0 / (x_norm + 1.0).square()
                        elif kernel == 'tanh':
                            K_norm = torch.tanh(x_norm)
                            K_prime = 1.0 - K_norm.square()
                        else:
                            raise ValueError(f"Unknown kernel: {kernel}")

                        one_minus_K2 = torch.clamp(1.0 - K_norm.square(), min=eps_geom)
                        scale = one_minus_K2.square() / (K_prime + damping * one_minus_K2.square())

                        # Precondition raw gradient
                        step_precond_2d = apply_precond(g_ch_2d, L_inv, R_inv, state['use_L'], state['use_R'])
                        step_precond = step_precond_2d.view_as(p_ch)
                        
                        step_natural = step_precond * scale

                        gamma_stable = min(gamma, 1.9 / step_size) if step_size > 0 else 0.0
                        coeff_num = K_norm * K_prime * one_minus_K2
                        coeff_denom = torch.clamp(K_prime.square() + damping * one_minus_K2.square(), min=eps_geom)
                        
                        entropic_force = (gamma_stable * (coeff_num / coeff_denom)) * (p_ch / r_safe)
                        proposed_update = step_natural + entropic_force

                        max_step = 0.9 * (R - r_safe)
                        final_update = torch.clamp(proposed_update, min=-max_step, max=max_step)

                    # Apply updates
                    p_ch.add_(final_update, alpha=-step_size)
                    p_ch.clamp_(min=1e-6)

                # 6. Reconstruct and write back physical parameter
                state['p_plus'].copy_(p_plus)
                state['p_minus'].copy_(p_minus)
                p.copy_((p_plus - p_minus).to(p.dtype))

        return loss

class KreinKroneckerSM3(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta_cov=0.99, eps=1e-8,
                 damping=1e-5, gamma=1e-3, weight_decay=1e-2, 
                 R=None, kernel='softsign', max_kron_dim=2048):
        """
        KreinKroneckerSM3 with dynamic, information-theoretic boundary scaling, 
        symmetric trust-region stabilization, and dimensional decoupling (max_kron_dim)
        to prevent Out-Of-Memory (OOM) errors on large vocabulary/linear projection heads.
        """
        defaults = dict(
            lr=lr, beta_cov=beta_cov, eps=eps, damping=damping, gamma=gamma,
            weight_decay=weight_decay, kernel=kernel, max_kron_dim=max_kron_dim
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
            max_kron_dim = group['max_kron_dim']

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
                    
                    # Decouple preconditioning for dimensions exceeding the threshold (e.g. lm_head)
                    state['use_L'] = (m <= max_kron_dim)
                    state['use_R'] = (n <= max_kron_dim)
                    
                    # Allocate actual 2D matrices or fall back to lightweight scalar representations
                    if state['use_L']:
                        state['L_inv_init'] = scale_factor * torch.eye(m, dtype=p.dtype, device=p.device)
                    else:
                        state['L_inv_init'] = scale_factor
                        
                    if state['use_R']:
                        state['R_inv_init'] = scale_factor * torch.eye(n, dtype=p.dtype, device=p.device)
                    else:
                        state['R_inv_init'] = scale_factor
                    
                    for ch in ['plus', 'minus']:
                        if state['use_L']:
                            state[f'L_inv_{ch}'] = state['L_inv_init'].clone()
                        else:
                            state[f'L_inv_{ch}'] = scale_factor
                            
                        if state['use_R']:
                            state[f'R_inv_{ch}'] = state['R_inv_init'].clone()
                        else:
                            state[f'R_inv_{ch}'] = scale_factor

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
                    
                    # Update Plus Channel Kronecker Factors conditionally
                    u_plus, v_plus = get_rank1_approximation(g_plus_2d)
                    if state['use_L']:
                        state['L_inv_plus'] = sherman_morrison_update(
                            state['L_inv_plus'], u_plus, beta_cov, state['L_inv_init'], leak
                        )
                    if state['use_R']:
                        state['R_inv_plus'] = sherman_morrison_update(
                            state['R_inv_plus'], v_plus, beta_cov, state['R_inv_init'], leak
                        )
                    
                    # Update Minus Channel Kronecker Factors conditionally
                    u_minus, v_minus = get_rank1_approximation(g_minus_2d)
                    if state['use_L']:
                        state['L_inv_minus'] = sherman_morrison_update(
                            state['L_inv_minus'], u_minus, beta_cov, state['L_inv_init'], leak
                        )
                    if state['use_R']:
                        state['R_inv_minus'] = sherman_morrison_update(
                            state['R_inv_minus'], v_minus, beta_cov, state['R_inv_init'], leak
                        )

                    # Compute Preconditioned Updates using the memory-saving apply_precond logic
                    step_precond_plus_2d = apply_precond(
                        g_plus_2d, state['L_inv_plus'], state['R_inv_plus'], state['use_L'], state['use_R']
                    )
                    step_precond_plus = step_precond_plus_2d.view_as(g_plus)
                    
                    step_precond_minus_2d = apply_precond(
                        g_minus_2d, state['L_inv_minus'], state['R_inv_minus'], state['use_L'], state['use_R']
                    )
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
                 R=None, kernel='softsign', max_kron_dim=2048, update_freq=10):
        """
        KreinKroneckerSM4: An optimized version of KreinKroneckerSM3 that uses 
        Lazy Preconditioning (update_freq) to drastically reduce computational overhead 
        on GPUs. All trust-region and memory protections of SM3 are preserved.
        """
        if update_freq < 1:
            raise ValueError(f"Invalid update_freq: {update_freq}. Must be >= 1.")
            
        defaults = dict(
            lr=lr, beta_cov=beta_cov, eps=eps, damping=damping, gamma=gamma,
            weight_decay=weight_decay, kernel=kernel, max_kron_dim=max_kron_dim,
            update_freq=update_freq
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
            max_kron_dim = group['max_kron_dim']
            update_freq = group['update_freq']

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
                    
                    # Decouple preconditioning for dimensions exceeding the threshold
                    state['use_L'] = (m <= max_kron_dim)
                    state['use_R'] = (n <= max_kron_dim)
                    
                    # Allocate actual 2D matrices or store scalar fallbacks
                    if state['use_L']:
                        state['L_inv_init'] = scale_factor * torch.eye(m, dtype=p.dtype, device=p.device)
                    else:
                        state['L_inv_init'] = scale_factor
                        
                    if state['use_R']:
                        state['R_inv_init'] = scale_factor * torch.eye(n, dtype=p.dtype, device=p.device)
                    else:
                        state['R_inv_init'] = scale_factor
                    
                    for ch in ['plus', 'minus']:
                        if state['use_L']:
                            state[f'L_inv_{ch}'] = state['L_inv_init'].clone()
                        else:
                            state[f'L_inv_{ch}'] = scale_factor
                            
                        if state['use_R']:
                            state[f'R_inv_{ch}'] = state['R_inv_init'].clone()
                        else:
                            state[f'R_inv_{ch}'] = scale_factor

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
                    # Decide whether to execute the expensive O(d^2) inverse updates this step
                    should_update = (state['step'] == 1) or (state['step'] % update_freq == 0)
                    
                    if should_update:
                        leak = 1.05 * (1.0 - beta_cov)
                        
                        # Update Plus Channel Kronecker Factors
                        u_plus, v_plus = get_rank1_approximation(g_plus_2d)
                        if state['use_L']:
                            state['L_inv_plus'] = sherman_morrison_update(
                                state['L_inv_plus'], u_plus, beta_cov, state['L_inv_init'], leak
                            )
                        if state['use_R']:
                            state['R_inv_plus'] = sherman_morrison_update(
                                state['R_inv_plus'], v_plus, beta_cov, state['R_inv_init'], leak
                            )
                        
                        # Update Minus Channel Kronecker Factors
                        u_minus, v_minus = get_rank1_approximation(g_minus_2d)
                        if state['use_L']:
                            state['L_inv_minus'] = sherman_morrison_update(
                                state['L_inv_minus'], u_minus, beta_cov, state['L_inv_init'], leak
                            )
                        if state['use_R']:
                            state['R_inv_minus'] = sherman_morrison_update(
                                state['R_inv_minus'], v_minus, beta_cov, state['R_inv_init'], leak
                            )

                    # Compute Preconditioned Updates (utilizing updated or cached matrices)
                    step_precond_plus_2d = apply_precond(
                        g_plus_2d, state['L_inv_plus'], state['R_inv_plus'], state['use_L'], state['use_R']
                    )
                    step_precond_plus = step_precond_plus_2d.view_as(g_plus)
                    
                    step_precond_minus_2d = apply_precond(
                        g_minus_2d, state['L_inv_minus'], state['R_inv_minus'], state['use_L'], state['use_R']
                    )
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
                    delta_limit = 0.9 * R_t
                    
                    # 1. Limit growth (final_update < 0): must not exceed remaining distance to R_t,
                    # and is capped at delta_limit (stable learning rate speed).
                    min_allowed_update = -0.9 * torch.clamp(R_t - p_ch, min=1e-12)
                    min_allowed_update = torch.clamp(min_allowed_update, min=-delta_limit)
                    
                    # 2. Limit shrinkage (final_update > 0): must not drop below 0,
                    # and is capped at delta_limit (stable learning rate speed).
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

def get_lr(it):

    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)

    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    if it > lr_decay_iters:
        return min_lr
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
    return min_lr + coeff * (learning_rate - min_lr)

decode, vocab_size = utils.get_meta(dataset,always_byte=always_byte)
get_batch = utils.get_get_batch(block_size,batch_size,dataset,device,always_byte=always_byte)


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int    = 12
    n_head: int     = 12
    n_embd: int     = 768
    dropout: float  = 0.0
    bias: bool      = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    
    device          = 'cuda'
    optimizer: str  = 'dca'
    batch_size      = 64
    step            = 0

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.n_embd))
        self.bias = nn.Parameter(torch.zeros(config.n_embd)) if config.bias else None
       
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        
        self.q_attn = (Linear(config.n_embd, config.n_embd, bias=config.bias))
        self.k_attn = (Linear(config.n_embd, config.n_embd, bias=config.bias))
        
        self.v_attn = Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head 
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim =  self.n_embd //  self.n_head 
        
        self.qm = nn.Parameter(torch.ones(self.head_dim).normal_(mean=1, std=0.1))
    
    def forward(self, x, causal=True):
        B, T, C = x.size() 
        
        q = self.q_attn(x)
        k = self.k_attn(x)
        v = self.v_attn(x)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)

        q_scale = math.log(T)*self.qm
        y = torch.nn.functional.scaled_dot_product_attention(q*q_scale,k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal= causal)
       
        y = y.transpose(1, 2).contiguous().view(B, T, C) 
        
        y = self.resid_dropout(self.c_proj(y))

        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        up = 4
        
        self.c_fc    = Linear(config.n_embd, up * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = Linear( up * config.n_embd, config.n_embd, bias=config.bias)
        
        self.dropout = nn.Dropout(config.dropout)
        if patt:
            self.patt = Patt(config.n_embd, config.n_embd, pattcount)

    def forward(self, x):
        if patt:
            return self.patt(x)
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = Attention(config)
        
        self.ln_1 = LayerNorm(config)
        self.ln_2 = LayerNorm(config)
        self.mlp = MLP(config)
        self.register_buffer("block_input",torch.zeros([config.block_size, config.n_embd]))
        
    def forward(self, x, causal = True):
        self.block_input = x
        
        
        x = x + self.attn(self.ln_1(x), causal)
        x = x + self.mlp(self.ln_2(x))
        return x

    def spl(self, x):
        predicted_output = self(-x, causal = False)
        
        sploss = F.mse_loss(
            utils.rms_norm(predicted_output.flatten()),
            utils.rms_norm(self.block_input.flatten())
        ) 
        return sploss
        
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.n_embd = config.n_embd
        self.config = config
        blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        
        wpe = nn.Embedding(config.block_size, config.n_embd)

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            
            wpe = wpe,
            drop = nn.Dropout(config.dropout),
            h = blocks,
            ln_f = LayerNorm(config),
        ))

        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight 
        
        self.wnl = utils.wnormlearnloss(config.n_embd)
        
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, module in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(module, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
       

    def forward(self, idx, targets=None, decaying = 0.0, get_emb = False):
        
        device = idx.device
        b, t = idx.size()
        
        
        tok_emb = self.transformer.wte(idx)

        pos_emb = None
        
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (b, t, n_embd)
        
        x = tok_emb + pos_emb
        
        x = self.transformer.drop(x)
        
        if self.training:
            if wnll:
                wnl = self.wnl(x)
        x1 = x
        for i, block in enumerate(self.transformer.h):
            if soloblock:
                x = self.transformer.h[0](x) 
            else:
                x = block(x) 


        if isinstance(x,tuple):
            x = x[0]
       
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        if targets is not None:
            reduce = 'mean'
            if wnll:
                reduce = 'none'
            
                
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),reduction=reduce)
            
            
            if self.training:
                if wnll:
                    mean_surprise = ((1.0-wnl.detach())/2.0).mean()*0.99
                    loss = loss / (1+ wnl.detach()*mean_surprise)
                    loss = (loss + wnl).mean()
                if spl:
                    if sspl:
                        self.transformer.h[1].block_input = x1
                        loss = loss + self.transformer.h[1].spl(x)
                    else:
                        loss = loss + self.transformer.h[0].spl(x)
            else:
                if wnll:
                    loss = loss.mean()
        else:
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
        
        
        return logits, loss
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        
        if isinstance(module, Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    

    def configure_optimizers(self, lr, wd):
        
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        decay_params = []
        nodecay_params = []
        for n, p in param_dict.items():
            if p.dim() >= 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)

        optim_groups = [
                {'params': nodecay_params,  'weight_decay': 0.0},
                {'params': decay_params,    'weight_decay': wd},
        ]
        
        # Configure model to run on our custom Lorentzian Dual-Signature Optimizer
        #return KreinDCAOptimizer(
        #    optim_groups, 
        #    lr=lr, 
        #    lambda_reg=wd
        #)
        #return KreinKroneckerSM(
        #    optim_groups,
        #    lr=lr, 
        #    mode="hypercube",
        #)
        return KreinKroneckerSM4(
            optim_groups,
            lr=lr, 
        )


if __name__ == "__main__":
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                      bias=bias, vocab_size=vocab_size, dropout=dropout) 
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.to(device)
    optimizer = model.configure_optimizers(lr=learning_rate, wd=weight_decay)

    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    torch.set_float32_matmul_precision('medium')

    model.compile()
    print(f"Starting training on {device} using {type(optimizer).__name__}")
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")


    t0 = time.time()
    t1 = time.time()
    t2 = time.time()

    def eval(max_iters, get_batch, model, optimizer, t0,t1,t2, iter, decaying):

        t2 = time.time()
        model.eval()
        losses = {}
        for split in ['train', 'val']:
            losses[split] = 0.0
            for k in range(eval_iters):
                X_val, Y_val = get_batch(split)
                with torch.no_grad():
                    _, loss = model(X_val, Y_val, decaying= decaying)
                losses[split] += loss.detach().item()
            losses[split] /= eval_iters
        optimizer.zero_grad(set_to_none=True)
        t1 = time.time()
        print(f"step {iter:{len(str(max_iters))}d}: ",end="")
        print(f"train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}, ",end="")
        print(f"time: {(t2-t0)*1000:.0f} ms, Tval: {(t1-t2)*1000:.0f} ms")
        model.train()
        t0 = time.time()
        return t0, t1, t2
    
    for iter in range(max_iters):
        if lr_ramp:
            lr = get_lr(iter)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        decaying = 1.0 - (iter / decay_iters)
        decaying = torch.tensor(decaying, device=device, dtype=torch.float32)
        decaying = torch.clamp(decaying, 0.0, 1.0)
        torch.compiler.cudagraph_mark_step_begin()
        X, Y = get_batch('train')
        logits, loss = model(X, Y, decaying= decaying)
        loss.backward()
        optimizer.step()
        if iter % eval_interval == 0 or iter == max_iters - 1:
            t0,t1,t2 =eval(max_iters, get_batch, model, optimizer, t0,t1,t2, iter, decaying)
        optimizer.zero_grad(set_to_none=True)

    # =========================================================================
    # SANITY GENERATION TEST
    # =========================================================================
    if gen:
        print("\n--- Running Sanity Generation Test ---")
        model.eval()
        # Initialize context with zeros (batch_size=1, sequence_length=1)
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated_tokens = []
        
        for _ in range(30):
            with torch.no_grad():
                logits, _ = model(context)
                # Extract the logits of the last token, apply softmax, and sample
                probs = F.softmax(logits[:, -1, :], dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                # Append to context and keep within our block size window
                context = torch.cat((context, next_token), dim=1)
                if context.size(1) > block_size:
                    context = context[:, -block_size:]
                generated_tokens.append(next_token.item())
                
        try:
            decoded_text = decode(generated_tokens)
            print(f"Generated Tokens: {generated_tokens}")
            print(f"Decoded Output:\n{decoded_text}")
        except Exception as e:
            print(f"Generated Tokens (could not decode): {generated_tokens}")
            print(f"Decoding Error: {e}")