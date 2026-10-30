import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

from parts import utils

torch.set_float32_matmul_precision('high')  


@torch.compile(backend='inductor', options={"triton.cudagraphs": False})
def compute_zca_whitening_matrix(Cov, epsilon=1e-5, steps=2, power_iters=2):
    """Computes the ZCA preconditioning matrix W = Cov^{-1/2} efficiently."""
    d = Cov.shape[0]
    device, dtype = Cov.device, Cov.dtype
    eye_d = torch.eye(d, device=device, dtype=dtype)

    Cov_reg = Cov + epsilon * eye_d

    # Power Iteration for spectral scale estimation
    v = torch.ones((d, 1), device=device, dtype=dtype)
    for _ in range(power_iters):
        u = torch.matmul(Cov_reg, v)
        u = u / (u.norm() + 1e-12)
        v = torch.matmul(Cov_reg.T, u)
        v = v / (v.norm() + 1e-12)

    s = torch.matmul(u.T, torch.matmul(Cov_reg, v)).squeeze().clamp(min=1e-8)

    B = Cov_reg / s
    Y = eye_d.clone()

    for _ in range(steps):
        Y = 0.5 * torch.matmul(Y, 3.0 * eye_d - torch.matmul(Y, torch.matmul(B, Y)))

    W = Y / torch.sqrt(s)
    return W


# =====================================================================
# 2. Channel-Aware Fused Krein Update
# =====================================================================


@torch.compile(backend='inductor', options={"triton.cudagraphs": False})
def update_krein_channels_fused(
    p_plus, p_minus, u_step,
    lr: float, gamma: float, scale_ref: float
):
    eps_geom = 1e-6
    R_t = 6.0 * scale_ref + eps_geom

    u_plus = torch.clamp(-u_step, min=0.0)
    u_minus = torch.clamp(u_step, min=0.0)

    # Channel Plus Entropic Force
    x_norm_plus = torch.clamp(p_plus / R_t, min=eps_geom, max=0.99)
    K_plus = x_norm_plus / (1.0 + x_norm_plus)
    K_prime_plus = 1.0 / (1.0 + x_norm_plus).square()
    force_plus = gamma * (K_plus * K_prime_plus) / (1.0 - K_plus.square() + eps_geom)

    # Channel Minus Entropic Force
    x_norm_minus = torch.clamp(p_minus / R_t, min=eps_geom, max=0.99)
    K_minus = x_norm_minus / (1.0 + x_norm_minus)
    K_prime_minus = 1.0 / (1.0 + x_norm_minus).square()
    force_minus = gamma * (K_minus * K_prime_minus) / (1.0 - K_minus.square() + eps_geom)

    p_plus_next = torch.clamp(p_plus + lr * (u_plus - force_plus), min=eps_geom)
    p_minus_next = torch.clamp(p_minus + lr * (u_minus - force_minus), min=eps_geom)
    p_data_next = p_plus_next - p_minus_next

    return p_plus_next, p_minus_next, p_data_next

@torch.compile(backend='inductor', options={"triton.cudagraphs": False})
def compute_soft_kronecker_matrix(Cov, epsilon=1e-5, steps=2, power_iters=2, alpha=0.5):
    d = Cov.shape[0]
    device, dtype = Cov.device, Cov.dtype
    eye_d = torch.eye(d, device=device, dtype=dtype)

    Cov_reg = Cov + epsilon * eye_d

    v = torch.ones((d, 1), device=device, dtype=dtype)
    for _ in range(power_iters):
        u = torch.matmul(Cov_reg, v)
        u = u / (u.norm() + 1e-12)
        v = torch.matmul(Cov_reg.T, u)
        v = v / (v.norm() + 1e-12)

    s = torch.matmul(u.T, torch.matmul(Cov_reg, v)).squeeze().clamp(min=1e-8)

    B = Cov_reg / s
    Y = eye_d.clone()

    for _ in range(steps):
        Y = 0.5 * torch.matmul(Y, 3.0 * eye_d - torch.matmul(Y, torch.matmul(B, Y)))

    W_hard = Y / torch.sqrt(s)
    
    W_soft = alpha * W_hard + (1.0 - alpha) * eye_d
    return W_soft



# =====================================================================
# 3. Corrected Entropic Krein Kronecker Optimizer
# =====================================================================

class EntropicKreinKronOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, beta_cov=0.125, alpha=0.5,
                 eps=1e-8, gamma=1e-4, damping=1e-5, weight_decay=1e-2, 
                 max_kron_dim=2048, steps_ns=2, power_iters=2, kron_freq=10,
                 use_kron=False):  # Set default False to verify exact baseline
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, beta_cov=beta_cov, alpha=alpha,
                        eps=eps, gamma=gamma, damping=damping, weight_decay=weight_decay,
                        max_kron_dim=max_kron_dim, steps_ns=steps_ns,
                        power_iters=power_iters, kron_freq=kron_freq,
                        use_kron=use_kron)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            beta_cov = group['beta_cov']
            alpha = group['alpha']
            eps = group['eps']
            gamma = group['gamma']
            damping = group['damping']
            weight_decay = group['weight_decay']
            max_kron_dim = group['max_kron_dim']
            steps_ns = group['steps_ns']
            power_iters = group['power_iters']
            kron_freq = group['kron_freq']
            use_kron = group['use_kron']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                orig_shape = p.shape
                is_multidim = p.dim() > 1

                if is_multidim:
                    m = orig_shape[0]
                    n = p.numel() // m
                else:
                    m, n = p.numel(), 1

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data, dtype=torch.float32)
                    state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=torch.float32)

                    state['p_plus'] = torch.clamp(p.data, min=0.0).add_(1e-4)
                    state['p_minus'] = torch.clamp(-p.data, min=0.0).add_(1e-4)
                    state['std_init'] = float(torch.nan_to_num(torch.std(p.data), nan=1e-4))
                    
                    state['L_ema'] = None
                    state['R_ema'] = None
                    state['W_L'] = None
                    state['W_R'] = None

                state['step'] += 1
                step_idx = state['step']

                p_plus = state['p_plus']
                p_minus = state['p_minus']

                if weight_decay != 0:
                    decay_factor = 1.0 - lr * weight_decay
                    p_plus.mul_(decay_factor)
                    p_minus.mul_(decay_factor)

                grad_fp32 = grad.to(torch.float32)
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                # Moment tracking
                exp_avg.mul_(beta1).add_(grad_fp32, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1.0 - beta2)

                bias_correction1 = 1.0 - (beta1 ** step_idx)
                bias_correction2 = 1.0 - (beta2 ** step_idx)

                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                # =====================================================
                # Direct Un-corrupted Adam Step for Baseline Accuracy
                # =====================================================
                adam_step = m_hat / (torch.sqrt(v_hat) + eps)

                if use_kron and is_multidim and m <= max_kron_dim and n <= max_kron_dim:
                    # Precondition raw adam_step directly
                    m_2d = adam_step.view(m, n)
                    m_2d_centered = m_2d - m_2d.mean(dim=0, keepdim=True)

                    L_batch = (1.0 / n) * torch.matmul(m_2d_centered, m_2d_centered.T)
                    R_batch = (1.0 / m) * torch.matmul(m_2d_centered.T, m_2d_centered)

                    if state['L_ema'] is None:
                        state['L_ema'] = torch.zeros_like(L_batch)
                        state['R_ema'] = torch.zeros_like(R_batch)
                    
                    state['L_ema'].mul_(beta_cov).add_(L_batch, alpha=1.0 - beta_cov)
                    state['R_ema'].mul_(beta_cov).add_(R_batch, alpha=1.0 - beta_cov)

                    bias_corr_cov = 1.0 - (beta_cov ** step_idx)
                    L_hat = state['L_ema'] / bias_corr_cov
                    R_hat = state['R_ema'] / bias_corr_cov

                    if state['W_L'] is None or step_idx % kron_freq == 0:
                        state['W_L'] = compute_soft_kronecker_matrix(
                            L_hat, epsilon=damping, steps=steps_ns, power_iters=power_iters, alpha=alpha
                        )
                        state['W_R'] = compute_soft_kronecker_matrix(
                            R_hat, epsilon=damping, steps=steps_ns, power_iters=power_iters, alpha=alpha
                        )

                    u_kron_2d = torch.matmul(state['W_L'], torch.matmul(m_2d_centered, state['W_R']))
                    scale = m_2d.norm() / (u_kron_2d.norm() + 1e-8)
                    u_step = (u_kron_2d * scale).view_as(p.data)
                else:
                    # Exact Original 1.6518 zcaless Baseline Path
                    u_step = adam_step

                # Fused Krein update step
                p_plus_next, p_minus_next, p_data_next = update_krein_channels_fused(
                    p_plus, p_minus, u_step,
                    lr, gamma, state['std_init'],
                )

                p_plus.copy_(p_plus_next)
                p_minus.copy_(p_minus_next)
                p.data.copy_(p_data_next.to(p.dtype))

        return loss

@torch.compile(backend='inductor', options={"triton.cudagraphs": False})
def update_krein_channels_unsmared(
    p_plus: torch.Tensor,
    p_minus: torch.Tensor,
    u_grad_precond: torch.Tensor,  # Preconditioned gradient drive ONLY
    lr: float,
    gamma: float,
    scale_ref: float
):
    eps_geom = 1e-6
    R_t = 6.0 * scale_ref + eps_geom

    # 1. Gradient drive direction
    u_plus = torch.clamp(-u_grad_precond, min=0.0)
    u_minus = torch.clamp(u_grad_precond, min=0.0)

    # 2. Purely coordinate-wise 1D Shannon entropic restoring force (UN-SMEARED)
    x_norm_plus = torch.clamp(p_plus / R_t, min=eps_geom, max=0.99)
    K_plus = x_norm_plus / (1.0 + x_norm_plus)
    K_prime_plus = 1.0 / (1.0 + x_norm_plus).square()
    force_plus = gamma * (K_plus * K_prime_plus) / (1.0 - K_plus.square() + eps_geom)

    x_norm_minus = torch.clamp(p_minus / R_t, min=eps_geom, max=0.99)
    K_minus = x_norm_minus / (1.0 + x_norm_minus)
    K_prime_minus = 1.0 / (1.0 + x_norm_minus).square()
    force_minus = gamma * (K_minus * K_prime_minus) / (1.0 - K_minus.square() + eps_geom)

    # 3. Add entropic barrier AFTER preconditioning (0% force smearing!)
    p_plus_next = torch.clamp(p_plus + lr * (u_plus - force_plus), min=eps_geom)
    p_minus_next = torch.clamp(p_minus + lr * (u_minus - force_minus), min=eps_geom)
    p_data_next = p_plus_next - p_minus_next

    return p_plus_next, p_minus_next, p_data_next

class UnsmearedKreinKronOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, beta_cov=0.01, alpha=1.0,
                 eps=1e-8, gamma=1e-4, damping=1e-5, weight_decay=1e-2, 
                 max_kron_dim=2048, steps_ns=2, power_iters=2, kron_freq=10,
                 use_kron=False):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, beta_cov=beta_cov, alpha=alpha,
                        eps=eps, gamma=gamma, damping=damping, weight_decay=weight_decay,
                        max_kron_dim=max_kron_dim, steps_ns=steps_ns,
                        power_iters=power_iters, kron_freq=kron_freq,
                        use_kron=use_kron)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            beta_cov = group['beta_cov']
            alpha = group['alpha']
            eps = group['eps']
            gamma = group['gamma']
            damping = group['damping']
            weight_decay = group['weight_decay']
            max_kron_dim = group['max_kron_dim']
            steps_ns = group['steps_ns']
            power_iters = group['power_iters']
            kron_freq = group['kron_freq']
            use_kron = group['use_kron']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                orig_shape = p.shape
                is_multidim = p.dim() > 1

                if is_multidim:
                    m = orig_shape[0]
                    n = p.numel() // m
                else:
                    m, n = p.numel(), 1

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data, dtype=torch.float32)
                    state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=torch.float32)

                    state['p_plus'] = torch.clamp(p.data, min=0.0).add_(1e-4)
                    state['p_minus'] = torch.clamp(-p.data, min=0.0).add_(1e-4)
                    state['std_init'] = float(torch.nan_to_num(torch.std(p.data), nan=1e-4))
                    
                    state['L_ema'] = None
                    state['R_ema'] = None
                    state['W_L'] = None
                    state['W_R'] = None

                state['step'] += 1
                step_idx = state['step']

                p_plus = state['p_plus']
                p_minus = state['p_minus']

                if weight_decay != 0:
                    decay_factor = 1.0 - lr * weight_decay
                    p_plus.mul_(decay_factor)
                    p_minus.mul_(decay_factor)

                grad_fp32 = grad.to(torch.float32)
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                # Moment tracking
                exp_avg.mul_(beta1).add_(grad_fp32, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1.0 - beta2)

                bias_correction1 = 1.0 - (beta1 ** step_idx)
                bias_correction2 = 1.0 - (beta2 ** step_idx)

                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                # Half-step Adam scale A^{1/2}
                denom = torch.sqrt(v_hat) + eps
                d_half = denom.rsqrt()

                # Step 1 of Symmetric Sandwich: A^{1/2}
                g_half = m_hat * d_half

                # =====================================================
                # Precondition Gradient Drive ONLY (Symmetric Sandwich)
                # =====================================================
                if use_kron and is_multidim and m <= max_kron_dim and n <= max_kron_dim:
                    g_half_2d = g_half.view(m, n)
                    g_half_centered = g_half_2d - g_half_2d.mean(dim=0, keepdim=True)

                    # 1. Compute Left and Right Covariances
                    L_batch = (1.0 / n) * torch.matmul(g_half_centered, g_half_centered.T)
                    R_batch = (1.0 / m) * torch.matmul(g_half_centered.T, g_half_centered)

                    # 2. Update EMA buffers
                    if state['L_ema'] is None:
                        state['L_ema'] = torch.zeros_like(L_batch)
                        state['R_ema'] = torch.zeros_like(R_batch)
                    
                    state['L_ema'].mul_(beta_cov).add_(L_batch, alpha=1.0 - beta_cov)
                    state['R_ema'].mul_(beta_cov).add_(R_batch, alpha=1.0 - beta_cov)

                    # 3. Bias correction
                    bias_corr_cov = 1.0 - (beta_cov ** step_idx)
                    L_hat = state['L_ema'] / bias_corr_cov
                    R_hat = state['R_ema'] / bias_corr_cov

                    # 4. Re-calculate W_L and W_R using soft exponent (alpha)
                    if state['W_L'] is None or step_idx % kron_freq == 0:
                        state['W_L'] = compute_soft_kronecker_matrix(
                            L_hat, epsilon=damping, steps=steps_ns, power_iters=power_iters, alpha=alpha
                        )
                        state['W_R'] = compute_soft_kronecker_matrix(
                            R_hat, epsilon=damping, steps=steps_ns, power_iters=power_iters, alpha=alpha
                        )

                    # 5. Apply Kronecker to GRADIENT DRIVE ONLY
                    u_kron_2d = torch.matmul(state['W_L'], torch.matmul(g_half_centered, state['W_R']))

                    # 6. Norm preservation
                    scale = g_half_2d.norm() / (u_kron_2d.norm() + 1e-8)
                    u_whitened = (u_kron_2d * scale).view_as(p.data)

                    # 7. Complete Symmetric Sandwich for Gradient Drive: A^{1/2} (W_L g W_R) A^{1/2}
                    u_grad_precond = u_whitened * d_half
                else:
                    # Pure Adam gradient drive
                    u_grad_precond = g_half * d_half

                # Un-smeared Fused Krein update step (Entropic force added AFTER)
                p_plus_next, p_minus_next, p_data_next = update_krein_channels_unsmared(
                    p_plus, p_minus, u_grad_precond,
                    lr, gamma, state['std_init'],
                )

                p_plus.copy_(p_plus_next)
                p_minus.copy_(p_minus_next)
                p.data.copy_(p_data_next.to(p.dtype))

        return loss
@torch.compile(backend='inductor', mode='max-autotune')
def zca_newton_schulz_trace(G, epsilon=1e-5, steps=2):
    if G.ndim < 2:
        return G
    G = G - G.mean(dim=0, keepdim=True)
    n, d = G.shape
    device, dtype = G.device, G.dtype
    
    # Compute covariance matrix with regularization
    Cov = (1/n) * torch.matmul(G.T, G) + epsilon * torch.eye(d, device=device, dtype=dtype)
    
    # Exact upper bound on spectral norm: s = Trace(Cov) >= max_eigenvalue
    # Guarantees ||B|| <= 1, making divergence/NaN mathematically impossible.
    s = torch.trace(Cov) + epsilon
    
    # Scale Cov to guarantee convergence
    B = Cov / s
    Y = torch.eye(d, device=device, dtype=dtype)
    eye_d = torch.eye(d, device=device, dtype=dtype)
    
    # Newton-Schulz iterations for inverse square root
    for _ in range(steps):
        Y = 0.5 * torch.matmul(Y, 3 * eye_d - torch.matmul(Y, torch.matmul(B, Y)))
    
    # Rescale to get Cov^{-1/2}
    W = Y / (torch.sqrt(s) + 1e-12)
    
    return torch.matmul(G, W)


@torch.compile(fullgraph=True)
def apply_precond_fused(G_2d, L_inv, R_inv, use_L: bool, use_R: bool):
    """Applies left and right Kronecker preconditioners."""
    step = G_2d
    if use_R:
        step = torch.matmul(step, R_inv)
    if use_L:
        step = torch.matmul(L_inv, step)
    return step


@torch.compile(fullgraph=True)
def update_cov_and_invert_root(C, G_outer, beta_cov: float, damping: float, power_inv: float = 0.25):
    """EMA covariance update and exact matrix power inverse root (C + damping*I)^(-power_inv)."""
    C.mul_(beta_cov).add_(G_outer, alpha=1.0 - beta_cov)
    
    # Eigendecomposition of symmetric positive-definite covariance C
    evals, evecs = torch.linalg.eigh(C)
    
    # Compute inverse fractional power: evals^(-0.25) for Shampoo -1/4 power
    evals_inv_power = torch.pow(torch.clamp(evals, min=damping), -power_inv)
    
    # Reconstruct C^(-power_inv) = evecs @ diag(evals^(-power_inv)) @ evecs^T
    return torch.matmul(evecs * evals_inv_power.unsqueeze(0), evecs.T)

@torch.compile(fullgraph=True)
def compute_fused_ngd_step(
    p_data, step_precond,
    lr: float, gamma: float, damping: float, std_init, kernel: str
):
    eps_geom = 1e-6

    # 1. Dynamic Manifold Boundary Scaling R_t
    std_current = p_data.std() if p_data.numel() > 1 else torch.abs(p_data)
    scale_ref = torch.maximum(std_init, std_current)
    R_t = 6.0 * scale_ref + eps_geom

    # Normalized distance to boundary
    x_norm = torch.clamp(p_data / R_t, min=-0.999, max=0.999)

    # 2. Kernel & Derivative Calculations
    if kernel == 'softsign':
        x_abs = torch.abs(x_norm)
        K_norm = x_norm / (1.0 + x_abs)
        K_prime = 1.0 / (1.0 + x_abs).square()
    else:  # tanh
        K_norm = torch.tanh(x_norm)
        K_prime = 1.0 - K_norm.square()

    one_minus_K2 = torch.clamp(1.0 - K_norm.square(), min=eps_geom)

    # 3. Manifold Scale Factor (Theorem: damped_natural_gradient_stable)
    denom = torch.clamp(K_prime.square() + damping * one_minus_K2.square(), min=eps_geom)
    scale = one_minus_K2.square() / denom

    step_natural = step_precond * scale

    # 4. Entropic Restoring Force (Theorem: physical_potential_deriv_from_shannon)
    gamma_stable = min(gamma, 1.9 / lr) if lr > 0 else 0.0
    coeff = (K_norm * K_prime * one_minus_K2) / denom
    entropic_force = gamma_stable * coeff

    final_update = step_natural + entropic_force

    # Direct In-Place Parameter Step
    p_data.sub_(lr * final_update)


class KreinKroneckerSM3(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
                 damping=1e-4, gamma=1e-3, weight_decay=1e-2, 
                 kernel='softsign', max_kron_dim=2048,
                 steps_ns=2, power_iters_ns=2):
        
        defaults = dict(
            lr=lr, beta1=beta1, beta2=beta2, eps=eps, damping=damping, gamma=gamma,
            weight_decay=weight_decay, kernel=kernel, max_kron_dim=max_kron_dim,
            steps_ns=steps_ns, power_iters_ns=power_iters_ns
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            eps = group['eps']
            damping = group['damping']
            gamma = group['gamma']
            weight_decay = group['weight_decay']
            kernel = group['kernel']
            max_kron_dim = group['max_kron_dim']
            steps_ns = group['steps_ns']
            power_iters_ns = group['power_iters_ns']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                orig_shape = p.shape
                is_multidim = p.dim() > 1
                
                if is_multidim:
                    m = orig_shape[0]
                    n = p.numel() // m
                else:
                    m, n = p.numel(), 1

                # State Initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data, dtype=torch.float32)
                    state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=torch.float32)

                    std_init = p.data.std() if p.numel() > 1 else p.data.abs()
                    state['std_init'] = torch.clamp(std_init, min=1e-4)

                state['step'] += 1
                step_idx = state['step']

                # Decoupled Weight Decay
                if weight_decay != 0:
                    p.data.mul_(1.0 - lr * weight_decay)

                grad_fp32 = grad.to(torch.float32)

                # 1. First & Second Moment Tracking (AdamW Reference)
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                exp_avg.mul_(beta1).add_(grad_fp32, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1.0 - beta2)

                bias_correction1 = 1.0 - (beta1 ** step_idx)
                bias_correction2 = 1.0 - (beta2 ** step_idx)

                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                adam_step = m_hat / (torch.sqrt(v_hat) + eps)

                if is_multidim:
                    m_2d = m_hat.view(m, n)

                    # 2. Newton-Schulz Preconditioning (Compiled Max-Autotune Triton)
                    if m <= max_kron_dim and n <= max_kron_dim:
                        step_precond_2d = zca_newton_schulz_trace(
                            m_2d, epsilon=damping, steps=steps_ns
                        )
                    else:
                        step_precond_2d = m_2d

                    step_precond = step_precond_2d.view_as(grad)

                    # 3. Adam Grafting: Rescale Natural Gradient Direction to Adam Magnitude
                    adam_norm = torch.linalg.vector_norm(adam_step)
                    ngd_norm = torch.linalg.vector_norm(step_precond)
                    scale_ratio = adam_norm / (ngd_norm + 1e-8)

                    grafted_step = step_precond * scale_ratio

                    # 4. Execute Fused Manifold Step (Lean 4 entropic scaling)
                    compute_fused_ngd_step(
                        p.data, grafted_step,
                        lr, gamma, damping, state['std_init'], kernel
                    )
                else:
                    p.data.sub_(lr * adam_step)

        return loss


@torch.compile(fullgraph=True)
def apply_shannon_entropic_decay(p_data, lr_gamma: float, std_init, kernel: str):
    """Calculates and applies the exact Lean 4 Shannon Entropic Restoring Force:
    V'(theta) = - gamma * K(theta) * K'(theta) / (1 - K(theta)^2)
    proved in Theorem `physical_potential_deriv_from_shannon`.
    """
    eps_geom = 1e-6
    std_current = p_data.std() if p_data.numel() > 1 else torch.abs(p_data)
    scale_ref = torch.maximum(std_init, std_current)
    R_t = 6.0 * scale_ref + eps_geom

    # Normalize parameter to (-1, 1) manifold
    x_norm = torch.clamp(p_data / R_t, min=-0.999, max=0.999)

    if kernel == 'softsign':
        x_abs = torch.abs(x_norm)
        K_norm = x_norm / (1.0 + x_abs)
        K_prime = 1.0 / (1.0 + x_abs).square()
    else:  # tanh
        K_norm = torch.tanh(x_norm)
        K_prime = 1.0 - K_norm.square()

    one_minus_K2 = torch.clamp(1.0 - K_norm.square(), min=eps_geom)

    # Shannon entropic force: - gamma * K * K' / (1 - K^2)
    entropic_force = lr_gamma * (K_norm * K_prime) / one_minus_K2

    # Apply information-theoretic entropic decay in-place
    p_data.sub_(entropic_force)


class AdamEntropicW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
                 gamma=1e-2, kernel='softsign'):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps,
                        gamma=gamma, kernel=kernel)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            eps = group['eps']
            gamma = group['gamma']
            kernel = group['kernel']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data, dtype=torch.float32)
                    state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=torch.float32)

                    std_init = p.data.std() if p.numel() > 1 else p.data.abs()
                    state['std_init'] = torch.clamp(std_init, min=1e-4)

                state['step'] += 1
                step_idx = state['step']

                grad_fp32 = grad.to(torch.float32)

                # 1. Standard AdamW Moment Tracking
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                exp_avg.mul_(beta1).add_(grad_fp32, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1.0 - beta2)

                bias_correction1 = 1.0 - (beta1 ** step_idx)
                bias_correction2 = 1.0 - (beta2 ** step_idx)

                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                # 2. Element-Wise Variance-Normalized Adam Step
                adam_step = m_hat / (torch.sqrt(v_hat) + eps)
                p.data.sub_(lr * adam_step)

                # 3. Apply Lean 4 Shannon Differential Entropy Restoring Force
                if gamma > 0:
                    apply_shannon_entropic_decay(p.data, lr * gamma, state['std_init'], kernel)

        return loss



@torch.compile(fullgraph=True)
def apply_shannon_entropic_decay(p_data, lr_gamma: float, std_init, kernel: str):
    eps_geom = 1e-6
    std_current = p_data.std() if p_data.numel() > 1 else torch.abs(p_data)
    scale_ref = torch.maximum(std_init, std_current)
    R_t = 6.0 * scale_ref + eps_geom

    x_norm = torch.clamp(p_data / R_t, min=-0.999, max=0.999)

    if kernel == 'softsign':
        x_abs = torch.abs(x_norm)
        K_norm = x_norm / (1.0 + x_abs)
        K_prime = 1.0 / (1.0 + x_abs).square()
    else:  # tanh
        K_norm = torch.tanh(x_norm)
        K_prime = 1.0 - K_norm.square()

    one_minus_K2 = torch.clamp(1.0 - K_norm.square(), min=eps_geom)
    entropic_force = lr_gamma * (K_norm * K_prime) / one_minus_K2

    p_data.sub_(entropic_force)


class EntropicAdamZCA(torch.optim.Optimizer):
    """AdamW with post-variance ZCA Whitening and Lean 4 Shannon Entropic Decay."""
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
                 gamma=1e-3, damping=1e-4, kernel='softsign', max_kron_dim=2048, steps_ns=2):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps, gamma=gamma,
                        damping=damping, kernel=kernel, max_kron_dim=max_kron_dim, steps_ns=steps_ns)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            eps = group['eps']
            gamma = group['gamma']
            damping = group['damping']
            kernel = group['kernel']
            max_kron_dim = group['max_kron_dim']
            steps_ns = group['steps_ns']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                orig_shape = p.shape
                is_multidim = p.dim() > 1
                
                if is_multidim:
                    m = orig_shape[0]
                    n = p.numel() // m
                else:
                    m, n = p.numel(), 1

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data, dtype=torch.float32)
                    state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=torch.float32)

                    std_init = p.data.std() if p.numel() > 1 else p.data.abs()
                    state['std_init'] = torch.clamp(std_init, min=1e-4)

                state['step'] += 1
                step_idx = state['step']

                grad_fp32 = grad.to(torch.float32)

                # 1. Adam Variance Normalization
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                exp_avg.mul_(beta1).add_(grad_fp32, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1.0 - beta2)

                bias_correction1 = 1.0 - (beta1 ** step_idx)
                bias_correction2 = 1.0 - (beta2 ** step_idx)

                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                adam_step = m_hat / (torch.sqrt(v_hat) + eps)

                # 2. ZCA Spectral Whitening on the Calibrated Adam Step
                if is_multidim and m <= max_kron_dim and n <= max_kron_dim:
                    m_2d = adam_step.view(m, n)
                    step_2d = zca_newton_schulz_trace(m_2d, epsilon=damping, steps=steps_ns)
                    update_step = step_2d.view_as(p.data)
                else:
                    update_step = adam_step

                p.data.sub_(lr * update_step)

                # 3. Apply Shannon Entropic Restoring Force
                if gamma > 0:
                    apply_shannon_entropic_decay(p.data, lr * gamma, state['std_init'], kernel)

        return loss

@torch.compile(backend='inductor', options={"triton.cudagraphs": False})
def update_krein_channels_unsmeared(
    p_plus: torch.Tensor,
    p_minus: torch.Tensor,
    u_grad_precond: torch.Tensor,  # Preconditioned gradient drive ONLY
    lr: float,
    gamma: float,
    scale_ref: float
):
    eps_geom = 1e-6
    R_t = 6.0 * scale_ref + eps_geom

    # 1. Gradient drive direction
    u_plus = torch.clamp(-u_grad_precond, min=0.0)
    u_minus = torch.clamp(u_grad_precond, min=0.0)

    # 2. Purely coordinate-wise 1D Shannon entropic restoring force (UN-SMEARED)
    x_norm_plus = torch.clamp(p_plus / R_t, min=eps_geom, max=0.99)
    K_plus = x_norm_plus / (1.0 + x_norm_plus)
    K_prime_plus = 1.0 / (1.0 + x_norm_plus).square()
    force_plus = gamma * (K_plus * K_prime_plus) / (1.0 - K_plus.square() + eps_geom)

    x_norm_minus = torch.clamp(p_minus / R_t, min=eps_geom, max=0.99)
    K_minus = x_norm_minus / (1.0 + x_norm_minus)
    K_prime_minus = 1.0 / (1.0 + x_norm_minus).square()
    force_minus = gamma * (K_minus * K_prime_minus) / (1.0 - K_minus.square() + eps_geom)

    # 3. Add entropic barrier AFTER preconditioning (0% force smearing!)
    p_plus_next = torch.clamp(p_plus + lr * (u_plus - force_plus), min=eps_geom)
    p_minus_next = torch.clamp(p_minus + lr * (u_minus - force_minus), min=eps_geom)
    p_data_next = p_plus_next - p_minus_next

    return p_plus_next, p_minus_next, p_data_next


class ComparisonKreinOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, beta_cov=0.1, alpha=1.0,
                 eps=1e-8, gamma=1e-4, damping=1e-5, weight_decay=1e-2, 
                 max_kron_dim=2048, kron_freq=10,
                 use_adam=True, use_kron=False):
        """
        Modular Krein Optimizer supporting 4 Whitening Modes:
          - use_adam=False, use_kron=False : Pure SGD + Krein (No Whitening)
          - use_adam=False, use_kron=True  : SGD + Full ZCA + Krein (Kronecker Whitening on SGD)
          - use_adam=True,  use_kron=False : AdamW + Krein (Diagonal ZCA Whitening - Current Champion)
          - use_adam=True,  use_kron=True  : AdamW + Full ZCA + Krein (Symmetric Sandwich Double Whitening)
        """
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, beta_cov=beta_cov, alpha=alpha,
                        eps=eps, gamma=gamma, damping=damping, weight_decay=weight_decay,
                        max_kron_dim=max_kron_dim, kron_freq=kron_freq,
                        use_adam=use_adam, use_kron=use_kron)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            beta_cov = group['beta_cov']
            alpha = group['alpha']
            eps = group['eps']
            gamma = group['gamma']
            damping = group['damping']
            weight_decay = group['weight_decay']
            max_kron_dim = group['max_kron_dim']
            kron_freq = group['kron_freq']
            use_adam = group['use_adam']
            use_kron = group['use_kron']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                orig_shape = p.shape
                is_multidim = p.dim() > 1

                if is_multidim:
                    m = orig_shape[0]
                    n = p.numel() // m
                else:
                    m, n = p.numel(), 1

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data, dtype=torch.float32)
                    state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=torch.float32)

                    state['p_plus'] = torch.clamp(p.data, min=0.0).add_(1e-4)
                    state['p_minus'] = torch.clamp(-p.data, min=0.0).add_(1e-4)
                    state['std_init'] = float(torch.nan_to_num(torch.std(p.data), nan=1e-4))
                    
                    state['L_ema'] = None
                    state['R_ema'] = None
                    state['W_L'] = None
                    state['W_R'] = None

                state['step'] += 1
                step_idx = state['step']

                p_plus = state['p_plus']
                p_minus = state['p_minus']

                if weight_decay != 0:
                    decay_factor = 1.0 - lr * weight_decay
                    p_plus.mul_(decay_factor)
                    p_minus.mul_(decay_factor)

                grad_fp32 = grad.to(torch.float32)
                exp_avg = state['exp_avg']
                
                # -------------------------------------------------------------
                # 1. Momentum Drive (First Moment)
                # -------------------------------------------------------------
                if beta1 > 0:
                    exp_avg.mul_(beta1).add_(grad_fp32, alpha=1.0 - beta1)
                    bias_correction1 = 1.0 - (beta1 ** step_idx)
                    m_hat = exp_avg / bias_correction1
                else:
                    m_hat = grad_fp32

                # -------------------------------------------------------------
                # 2. Diagonal Whitening (AdamW Second Moment)
                # -------------------------------------------------------------
                if use_adam:
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg_sq.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1.0 - beta2)
                    bias_correction2 = 1.0 - (beta2 ** step_idx)
                    v_hat = exp_avg_sq / bias_correction2

                    denom = torch.sqrt(v_hat) + eps
                    d_half = denom.rsqrt()          # A^(1/2) = v^(-1/4)
                    g_drive = m_hat * d_half        # Scale momentum by A^(1/2)
                else:
                    d_half = torch.ones_like(m_hat)
                    g_drive = m_hat                 # Raw momentum drive

                # -------------------------------------------------------------
                # 3. Matrix Whitening (Full ZCA / Kronecker)
                # -------------------------------------------------------------
                if use_kron and is_multidim and m <= max_kron_dim and n <= max_kron_dim:
                    g_2d = g_drive.view(m, n)
                    g_centered = g_2d - g_2d.mean(dim=0, keepdim=True)

                    L_batch = (1.0 / n) * torch.matmul(g_centered, g_centered.T)
                    R_batch = (1.0 / m) * torch.matmul(g_centered.T, g_centered)

                    if state['L_ema'] is None:
                        state['L_ema'] = torch.zeros_like(L_batch)
                        state['R_ema'] = torch.zeros_like(R_batch)
                    
                    state['L_ema'].mul_(beta_cov).add_(L_batch, alpha=1.0 - beta_cov)
                    state['R_ema'].mul_(beta_cov).add_(R_batch, alpha=1.0 - beta_cov)

                    bias_corr_cov = 1.0 - (beta_cov ** step_idx)
                    L_hat = state['L_ema'] / bias_corr_cov
                    R_hat = state['R_ema'] / bias_corr_cov

                    if state['W_L'] is None or step_idx % kron_freq == 0:
                        state['W_L'] = compute_soft_kronecker_matrix(
                            L_hat, epsilon=damping, alpha=alpha
                        )
                        state['W_R'] = compute_soft_kronecker_matrix(
                            R_hat, epsilon=damping, alpha=alpha
                        )

                    # Apply Kronecker Whitening
                    u_kron_2d = torch.matmul(state['W_L'], torch.matmul(g_centered, state['W_R']))

                    # Norm preservation
                    scale = g_2d.norm() / (u_kron_2d.norm() + 1e-8)
                    u_whitened = (u_kron_2d * scale).view_as(p.data)

                    # Complete Symmetric Sandwich if Adam is active
                    u_grad_precond = u_whitened * d_half
                else:
                    # Pass through without Kronecker matrix whitening
                    u_grad_precond = g_drive * d_half

                p_plus_next, p_minus_next, p_data_next = update_krein_channels_unsmeared(
                    p_plus, p_minus, u_grad_precond,
                    lr, gamma, state['std_init'],
                )

                p_plus.copy_(p_plus_next)
                p_minus.copy_(p_minus_next)
                p.data.copy_(p_data_next.to(p.dtype))

        return loss


@torch.compile(backend='inductor', options={"triton.cudagraphs": False})
def update_entropic_krein_step(
    p: torch.Tensor,
    p_plus: torch.Tensor,
    p_minus: torch.Tensor,
    u_grad_precond: torch.Tensor,
    lr: float,
    gamma: float,
    scale_ref: float,
    use_krein_space: bool
):
    eps_geom = 1e-6
    R_t = 6.0 * scale_ref + eps_geom

    if use_krein_space:
        # 1. Non-negative gradient drives
        u_plus = torch.clamp(-u_grad_precond, min=0.0)
        u_minus = torch.clamp(u_grad_precond, min=0.0)

        # 2. 1D Shannon entropic restoring forces
        x_norm_plus = torch.clamp(p_plus / R_t, min=eps_geom, max=0.99)
        K_plus = x_norm_plus / (1.0 + x_norm_plus)
        K_prime_plus = 1.0 / (1.0 + x_norm_plus).square()
        force_plus = gamma * (K_plus * K_prime_plus) / (1.0 - K_plus.square() + eps_geom)

        x_norm_minus = torch.clamp(p_minus / R_t, min=eps_geom, max=0.99)
        K_minus = x_norm_minus / (1.0 + x_norm_minus)
        K_prime_minus = 1.0 / (1.0 + x_norm_minus).square()
        force_minus = gamma * (K_minus * K_prime_minus) / (1.0 - K_minus.square() + eps_geom)

        # 3. Channel updates
        p_plus_raw = torch.clamp(p_plus + lr * (u_plus - force_plus), min=eps_geom)
        p_minus_raw = torch.clamp(p_minus + lr * (u_minus - force_minus), min=eps_geom)
        
        # 4. Net weight value
        p_data_next = p_plus_raw - p_minus_raw

        # 5. KREIN GAUGE PROJECTION (Fixes the random walk / explosion!)
        p_plus_next = torch.clamp(p_data_next, min=0.0) + eps_geom
        p_minus_next = torch.clamp(-p_data_next, min=0.0) + eps_geom

        return p_plus_next, p_minus_next, p_data_next

    else:
        # Single-channel direct entropic decay on p
        p_abs = torch.abs(p)
        x_norm = torch.clamp(p_abs / R_t, min=eps_geom, max=0.99)
        K_val = x_norm / (1.0 + x_norm)
        K_prime = 1.0 / (1.0 + x_norm).square()

        # Direct 1D restoring force pointing toward origin
        force_direct = torch.sign(p) * gamma * (K_val * K_prime) / (1.0 - K_val.square() + eps_geom)

        p_data_next = p - lr * (u_grad_precond + force_direct)

        return p_plus, p_minus, p_data_next


class UnifiedKreinEntropicOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        beta_cov: float = 0.1,
        alpha: float = 1.0,
        eps: float = 1e-8,
        gamma: float = 1e-4,
        damping: float = 1e-5,
        weight_decay: float = 1e-2,
        max_kron_dim: int = 2048,
        kron_freq: int = 10,
        # Toggles
        use_adam: bool = True,
        use_kron: bool = False,
        use_krein_space: bool = True,
        use_grms: bool = True,
    ):
        defaults = dict(
            lr=lr, beta1=beta1, beta2=beta2, beta_cov=beta_cov, alpha=alpha,
            eps=eps, gamma=gamma, damping=damping, weight_decay=weight_decay,
            max_kron_dim=max_kron_dim, kron_freq=kron_freq,
            use_adam=use_adam, use_kron=use_kron,
            use_krein_space=use_krein_space, use_grms=use_grms
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            beta_cov = group['beta_cov']
            alpha = group['alpha']
            eps = group['eps']
            gamma = group['gamma']
            damping = group['damping']
            weight_decay = group['weight_decay']
            max_kron_dim = group['max_kron_dim']
            kron_freq = group['kron_freq']
            
            use_adam = group['use_adam']
            use_kron = group['use_kron']
            use_krein_space = group['use_krein_space']
            use_grms = group['use_grms']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                orig_shape = p.shape
                is_multidim = p.dim() > 1

                if is_multidim:
                    m = orig_shape[0]
                    n = p.numel() // m
                else:
                    m, n = p.numel(), 1

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data, dtype=torch.float32)
                    state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=torch.float32)

                    state['p_plus'] = torch.clamp(p.data, min=0.0).add_(1e-4)
                    state['p_minus'] = torch.clamp(-p.data, min=0.0).add_(1e-4)
                    state['std_init'] = torch.nan_to_num(torch.std(p.data), nan=1e-4)

                    state['L_ema'] = None
                    state['R_ema'] = None
                    state['W_L'] = None
                    state['W_R'] = None

                state['step'] += 1
                step_idx = state['step']

                p_plus = state['p_plus']
                p_minus = state['p_minus']

                # Decoupled Weight Decay
                if weight_decay != 0:
                    decay_factor = 1.0 - lr * weight_decay
                    p.data.mul_(decay_factor)
                    if use_krein_space:
                        p_plus.mul_(decay_factor)
                        p_minus.mul_(decay_factor)

                grad_fp32 = grad.to(torch.float32)

                # =============================================================
                # 1. GRMS AT THE RAW GRADIENT LEVEL (Matches train.py hook!)
                # =============================================================
                if use_grms:
                    d2 = not (grad_fp32.ndim < 2)
                    nd = [-1, -2] if d2 else [-1]
                    rms = torch.rsqrt(grad_fp32.pow(2).mean(dim=nd, keepdim=True) + 1e-8)
                    grad_fp32 = grad_fp32 * rms

                exp_avg = state['exp_avg']

                # =============================================================
                # 2. Momentum Drive (First Moment)
                # =============================================================
                if beta1 > 0:
                    exp_avg.mul_(beta1).add_(grad_fp32, alpha=1.0 - beta1)
                    bias_correction1 = 1.0 - (beta1 ** step_idx)
                    m_hat = exp_avg / bias_correction1
                else:
                    m_hat = grad_fp32

                # =============================================================
                # 3. Diagonal Whitening (Adam Second Moment)
                # =============================================================
                if use_adam:
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg_sq.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1.0 - beta2)
                    bias_correction2 = 1.0 - (beta2 ** step_idx)
                    v_hat = exp_avg_sq / bias_correction2

                    denom = torch.sqrt(v_hat) + eps
                    d_half = denom.rsqrt()          # A^(1/2) = v^(-1/4)
                    g_drive = m_hat * d_half
                else:
                    d_half = torch.ones_like(m_hat)
                    g_drive = m_hat

                # =============================================================
                # 4. Matrix Whitening (Kronecker ZCA)
                # =============================================================
                if use_kron and is_multidim and m <= max_kron_dim and n <= max_kron_dim:
                    g_2d = g_drive.view(m, n)
                    g_centered = g_2d - g_2d.mean(dim=0, keepdim=True)

                    L_batch = (1.0 / n) * torch.matmul(g_centered, g_centered.T)
                    R_batch = (1.0 / m) * torch.matmul(g_centered.T, g_centered)

                    if state['L_ema'] is None:
                        state['L_ema'] = torch.zeros_like(L_batch)
                        state['R_ema'] = torch.zeros_like(R_batch)

                    state['L_ema'].mul_(beta_cov).add_(L_batch, alpha=1.0 - beta_cov)
                    state['R_ema'].mul_(beta_cov).add_(R_batch, alpha=1.0 - beta_cov)

                    bias_corr_cov = 1.0 - (beta_cov ** step_idx)
                    L_hat = state['L_ema'] / bias_corr_cov
                    R_hat = state['R_ema'] / bias_corr_cov

                    if state['W_L'] is None or step_idx % kron_freq == 0:
                        state['W_L'] = compute_soft_kronecker_matrix(
                            L_hat, epsilon=damping, alpha=alpha
                        )
                        state['W_R'] = compute_soft_kronecker_matrix(
                            R_hat, epsilon=damping, alpha=alpha
                        )

                    u_kron_2d = torch.matmul(state['W_L'], torch.matmul(g_centered, state['W_R']))
                    scale = g_2d.norm() / (u_kron_2d.norm() + 1e-8)
                    u_whitened = (u_kron_2d * scale).view_as(p.data)

                    u_grad_precond = u_whitened * d_half
                else:
                    u_grad_precond = g_drive * d_half

                # =============================================================
                # 5. Entropic Step (Dual Channel vs Direct Entropic Decay)
                # =============================================================
                p_plus_next, p_minus_next, p_data_next = update_entropic_krein_step(
                    p.data, p_plus, p_minus, u_grad_precond,
                    lr, gamma, state['std_init'], use_krein_space
                )

                if use_krein_space:
                    p_plus.copy_(p_plus_next)
                    p_minus.copy_(p_minus_next)

                p.data.copy_(p_data_next.to(p.dtype))

        return loss

# =====================================================================
# 2. BENCHMARK & PROFILER SUITE
# =====================================================================
class BenchmarkTransformerLayerStack(nn.Module):
    def __init__(self, hidden_dim=2048, num_layers=2):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            layers.append(nn.Linear(hidden_dim, 4 * hidden_dim, bias=False))
            layers.append(nn.Linear(4 * hidden_dim, hidden_dim, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def run_benchmark(
    batch_size=8,
    hidden_dim=2048,
    num_layers=2,
    warmup_steps=15,
    bench_steps=100,
    max_kron_dim=512,
    device="cuda",
):
    print(f"\n=================================================================")
    print(f" FINAL BENCHMARK & PROFILER: GPU={torch.cuda.get_device_name(0)}")
    print(f" Layers Config: Dim={hidden_dim}, Layers={num_layers}")
    print(f" Sliced Tile Size: {max_kron_dim} x {max_kron_dim}")
    print(f"=================================================================\n")

    model_adam = BenchmarkTransformerLayerStack(hidden_dim, num_layers).to(device)
    model_kk = BenchmarkTransformerLayerStack(hidden_dim, num_layers).to(device)
    model_kk.load_state_dict(model_adam.state_dict())

    opt_adam = torch.optim.AdamW(model_adam.parameters(), lr=1e-3, fused=True)
    opt_kk = KreinKroneckerSM3(
        model_kk.parameters(), lr=1e-3, max_kron_dim=max_kron_dim
    )

    x = torch.randn(batch_size, hidden_dim, device=device)

    print("Compiling Triton & Torch Batched Kernels for KreinKroneckerSM5...")
    out = model_kk(x).sum()
    out.backward()
    opt_kk.step()
    print("Compilation complete.\n")

    # 1. Timing Loop
    results = {}
    for name, opt, model in [
        ("AdamW (Fused)", opt_adam, model_adam),
        ("KreinKroneckerSM (Amortized)", opt_kk, model_kk),
    ]:
        for _ in range(warmup_steps):
            opt.zero_grad()
            out = model(x).sum()
            out.backward()
            opt.step()

        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(bench_steps):
            opt.zero_grad()
            out = model(x).sum()
            out.backward()
            opt.step()
        end_event.record()

        torch.cuda.synchronize()
        total_time_ms = start_event.elapsed_time(end_event)
        per_step_ms = total_time_ms / bench_steps
        projected_1k_ms = per_step_ms * 1000

        results[name] = projected_1k_ms
        print(
            f"[{name}] Total Time ({bench_steps} steps): {total_time_ms:.2f} ms | "
            f"Per Step: {per_step_ms:.3f} ms | Projected 1k Steps: {projected_1k_ms:.2f} ms"
        )

    overhead = results["KreinKroneckerSM (Amortized)"] / results["AdamW (Fused)"]
    print(f"\n--> KreinKroneckerSM (Amortized) runtime relative to AdamW: {overhead:.2f}x\n")

    # 2. Detailed Profiler Table
    print("--------------------------------------------------------------------------------------------------")
    print(" DETAILED CUDA PROFILER BREAKDOWN (KreinKroneckerSM - Amortized)")
    print("--------------------------------------------------------------------------------------------------")

    opt_kk.zero_grad()
    out = model_kk(x).sum()
    out.backward()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=False,
    ) as prof:
        opt_kk.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA device required for benchmark.")
    else:
        run_benchmark()