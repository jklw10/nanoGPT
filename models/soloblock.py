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
# Eager in-place execution bypasses dynamic-shape compile & autotuning loops

class EntropicAdamW2(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 damping=1e-5, gamma=1e-3, weight_decay=1e-2, kernel='softsign'):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if damping < 0.0:
            raise ValueError(f"Invalid damping: {damping}")
        if gamma < 0.0:
            raise ValueError(f"Invalid gamma (entropic force coefficient): {gamma}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(
            lr=lr, 
            betas=betas, 
            eps=eps, 
            damping=damping, 
            gamma=gamma,
            weight_decay=weight_decay, 
            kernel=kernel,
        )
        super(EntropicAdamW2, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            damping = group['damping']
            gamma = group['gamma']
            weight_decay = group['weight_decay']
            kernel = group['kernel']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("EntropicAdamW does not support sparse gradients.")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                step = state['step']

                # 1. Decoupled weight decay
                if weight_decay != 0:
                    p.mul_(1.0 - lr * weight_decay)

                # 2. Update biased first and second moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # 3. Bias corrections and step scaling
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                step_size = lr / bias_correction1

                # 4. Adam denominator (RMS scale)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                
                gamma = min(gamma, 1.9 / step_size)
                if p.dim() > 1:
                    dims = list(range(1, p.dim()))
                    r = torch.norm(p, p=2, dim=dims, keepdim=True)
                else:
                    r = torch.norm(p, p=2, dim=0, keepdim=True)
                r_safe = torch.clamp(r, min=1e-12)
                if kernel == 'softsign':
                    K_norm = r / (r + 1.0)
                    K_prime = 1.0 / (r + 1.0).square()
                elif kernel == 'tanh':
                    K_norm = torch.tanh(r)
                    K_prime = 1.0 - K_norm.square()
                else:
                    raise ValueError(f"Unknown kernel option: {kernel}")
                one_minus_K2 = 1.0 - K_norm.square()
                # Corrected scale: accounts for Adam already dividing out one factor of K'
                scale = one_minus_K2.square() / (K_prime + damping * one_minus_K2.square())
                
                # Restoring force coefficient (always pulls vector back toward the origin)
                coeff_num = K_norm * K_prime * one_minus_K2
                coeff_denom = K_prime.square() + damping * one_minus_K2.square()
                coeff = gamma * (coeff_num / coeff_denom)
                # Total update step: preconditioned loss gradient + radial restoring force
                update = (exp_avg / denom) * scale + coeff * (p / r_safe)

                # 6. Apply step: p = p - step_size * update
                p.add_(update, alpha=-step_size)

        return loss


class EntropicPathAdamW(torch.optim.Optimizer):
    """
    Entropic Path AdamW (EPA) - Sliced Spherical Natural Gradient Optimizer.
    Dynamically scales the projection using the exact pullback metric tensor.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 damping=1.0, alpha=0.05, gamma=1e-3, weight_decay=1e-2, 
                 R=2.5, kernel='softsign'):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if damping < 0.0:
            raise ValueError(f"Invalid damping: {damping}")
        if alpha < 0.0:
            raise ValueError(f"Invalid alpha: {alpha}")
        if gamma < 0.0:
            raise ValueError(f"Invalid gamma: {gamma}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if R <= 0.0:
            raise ValueError(f"Invalid boundary scale R: {R}")

        defaults = dict(
            lr=lr, 
            betas=betas, 
            eps=eps, 
            damping=damping, 
            alpha=alpha,
            gamma=gamma,
            weight_decay=weight_decay, 
            R=R,
            kernel=kernel,
        )
        super(EntropicPathAdamW, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            damping = group['damping']
            alpha = group['alpha']
            gamma = group['gamma']
            weight_decay = group['weight_decay']
            R = group['R']
            kernel = group['kernel']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("EPA does not support sparse gradients.")

                state = self.state[p]
                eps_geom = 1e-12

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, dtype=torch.float32, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, dtype=torch.float32, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                step = state['step']

                p_fp32 = p.to(torch.float32)
                grad_fp32 = grad.to(torch.float32)

                # 1. Decoupled Weight Decay
                # Skip standard weight decay for multidimensional parameters because the entropic
                # restoring force acts as the sole, non-linear weight decay (Theorems: entropic_force_linear_near_zero)
                if p.dim() <= 1 and weight_decay != 0:
                    p_fp32.mul_(1.0 - lr * weight_decay)

                # 2. Update biased moment estimates
                exp_avg.mul_(beta1).add_(grad_fp32, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1.0 - beta2)

                # 3. Bias corrections
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                step_size = lr / bias_correction1

                # 4. Adam preconditioning denominator
                eps_stable = max(eps, eps_geom)
                exp_avg_sq_safe = torch.clamp(exp_avg_sq, min=0.0)
                denom = (exp_avg_sq_safe.sqrt() / math.sqrt(bias_correction2)).add_(eps_stable)
                step_direction_fp32 = exp_avg / denom

                # 5. Fallback for 1-dimensional parameters
                if p.dim() <= 1:
                    p_fp32.add_(step_direction_fp32, alpha=-step_size)
                    p.copy_(p_fp32)
                    continue

                # 6. Sliced norm computation and unit vector normalization
                dims = list(range(1, p.dim()))
                r = torch.norm(p_fp32, p=2, dim=dims, keepdim=True)
                r_clamped = torch.clamp(r, max=R - eps_geom)
                r_safe = torch.clamp(r_clamped, min=eps_geom)
                w = p_fp32 / r_safe

                # 7. Compute Kernel Parameters and Pullback Metric Scale
                x_norm = r_safe / R
                if kernel == 'softsign':
                    K_norm = x_norm / (x_norm + 1.0)
                    K_prime = 1.0 / (x_norm + 1.0).square()
                elif kernel == 'tanh':
                    K_norm = torch.tanh(torch.clamp(x_norm, max=20.0))
                    K_prime = torch.clamp(1.0 - K_norm.square(), min=0.0)
                else:
                    raise ValueError(f"Unknown kernel: {kernel}")

                # Dynamic projection strength based on pullback metric G(θ) = (1/R^2) * (K'^2 / (1 - K^2)^2)
                one_minus_K2 = torch.clamp(1.0 - K_norm.square(), min=eps_geom)
                metric_scale = K_prime / (R * one_minus_K2)
                alpha_dynamic = alpha * metric_scale.square()

                # 8. Unscaled Identity Sherman-Morrison Natural Projection
                dot_product = torch.sum(w * step_direction_fp32, dim=dims, keepdim=True)
                w_norm_sq = torch.sum(w * w, dim=dims, keepdim=True)
                
                damping_stable = max(damping, eps_geom)
                proj_denom = damping_stable + alpha_dynamic * w_norm_sq
                proj_denom_stable = torch.clamp(proj_denom, min=eps_geom)
                
                step_natural_fp32 = step_direction_fp32 - (alpha_dynamic * dot_product / proj_denom_stable) * w

                # 9. Adaptive Non-linear Entropic Restoring Force
                if step_size > 0:
                    gamma_stable = min(gamma, 1.9 / step_size)
                else:
                    gamma_stable = 0.0
                
                coeff_num = K_norm * K_prime * one_minus_K2
                coeff_denom = K_prime.square() + damping_stable * one_minus_K2.square()
                coeff_denom_stable = torch.clamp(coeff_denom, min=eps_geom)
                coeff = gamma_stable * (coeff_num / coeff_denom_stable)
                
                entropic_force_fp32 = coeff * w

                # 10. Compute proposed update vector
                proposed_update = step_natural_fp32 + entropic_force_fp32

                # 11. Stochastic Boundary Censorship
                update_norm = torch.norm(proposed_update, p=2, dim=dims, keepdim=True)
                max_step = 0.9 * (R - r_safe)
                scale_factor = torch.clamp(max_step / torch.clamp(update_norm, min=eps_geom), max=1.0)
                final_update = proposed_update * scale_factor

                # 12. Parameter Update
                p_fp32.add_(final_update, alpha=-step_size)
                p.copy_(p_fp32)

        return loss

class EntropicPathAdamW2(torch.optim.Optimizer):
    """
    Entropic Path AdamW (EPA) with Configurable Structural Tests.
    Provides toggleable booleans to isolate and test momentum projection,
    consensus preconditioning, and Jordan symmetrization.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 damping=0.05, alpha=0.05, gamma=1e-3, weight_decay=1e-2, 
                 R=2.5, kernel='softsign',
                 project_momentum=True,
                 use_consensus_denom=True,
                 use_jordan_symmetrized=True):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if damping < 0.0:
            raise ValueError(f"Invalid damping: {damping}")
        if alpha < 0.0:
            raise ValueError(f"Invalid alpha: {alpha}")
        if gamma < 0.0:
            raise ValueError(f"Invalid gamma: {gamma}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if R <= 0.0:
            raise ValueError(f"Invalid boundary scale R: {R}")

        defaults = dict(
            lr=lr, 
            betas=betas, 
            eps=eps, 
            damping=damping, 
            alpha=alpha,
            gamma=gamma,
            weight_decay=weight_decay, 
            R=R,
            kernel=kernel,
            project_momentum=project_momentum,
            use_consensus_denom=use_consensus_denom,
            use_jordan_symmetrized=use_jordan_symmetrized
        )
        super(EntropicPathAdamW2, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            damping = group['damping']
            alpha = group['alpha']
            gamma = group['gamma']
            weight_decay = group['weight_decay']
            R = group['R']
            kernel = group['kernel']
            project_momentum = group['project_momentum']
            use_consensus_denom = group['use_consensus_denom']
            use_jordan_symmetrized = group['use_jordan_symmetrized']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("EPA does not support sparse gradients.")

                state = self.state[p]
                eps_geom = 1e-12

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, dtype=torch.float32, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, dtype=torch.float32, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                step = state['step']

                p_fp32 = p.to(torch.float32)
                grad_fp32 = grad.to(torch.float32)

                # 1. Decoupled Weight Decay
                if p.dim() <= 1 and weight_decay != 0:
                    p_fp32.mul_(1.0 - lr * weight_decay)

                # 2. Update biased moment estimates
                exp_avg.mul_(beta1).add_(grad_fp32, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1.0 - beta2)

                # 3. Bias corrections
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                step_size = lr / bias_correction1

                # 4. Adam preconditioning denominator
                eps_stable = max(eps, eps_geom)
                exp_avg_sq_safe = torch.clamp(exp_avg_sq, min=0.0)
                denom = (exp_avg_sq_safe.sqrt() / math.sqrt(bias_correction2)).add_(eps_stable)
                step_direction_fp32 = exp_avg / denom

                # 5. Fallback for 1-dimensional parameters
                if p.dim() <= 1:
                    p_fp32.add_(step_direction_fp32, alpha=-step_size)
                    p.copy_(p_fp32)
                    continue

                # 6. Sliced norm computation and unit vector normalization
                dims = list(range(1, p.dim()))
                r = torch.norm(p_fp32, p=2, dim=dims, keepdim=True)
                r_clamped = torch.clamp(r, max=R - eps_geom)
                r_safe = torch.clamp(r_clamped, min=eps_geom)
                w = p_fp32 / r_safe

                # Optional: Momentum Projection (Aligns exp_avg to tangent space to resolve phase lag)
                if project_momentum:
                    dot_product_momentum = torch.sum(w * exp_avg, dim=dims, keepdim=True)
                    exp_avg.sub_(dot_product_momentum * w)
                    # Recalculate preconditioned direction with aligned momentum
                    step_direction_fp32 = exp_avg / denom

                # 7. Compute Kernel Parameters and Pullback Metric Scale
                x_norm = r_safe / R
                if kernel == 'softsign':
                    K_norm = x_norm / (x_norm + 1.0)
                    K_prime = 1.0 / (x_norm + 1.0).square()
                elif kernel == 'tanh':
                    K_norm = torch.tanh(torch.clamp(x_norm, max=20.0))
                    K_prime = torch.clamp(1.0 - K_norm.square(), min=0.0)
                else:
                    raise ValueError(f"Unknown kernel: {kernel}")

                one_minus_K2 = torch.clamp(1.0 - K_norm.square(), min=eps_geom)
                metric_scale = K_prime / (R * one_minus_K2)
                alpha_dynamic = alpha * metric_scale.square()

                # 8. Sliced Natural Projection Tests
                w_norm_sq = torch.sum(w * w, dim=dims, keepdim=True)
                damping_stable = max(damping, eps_geom)
                proj_denom = damping_stable + alpha_dynamic * w_norm_sq
                proj_denom_stable = torch.clamp(proj_denom, min=eps_geom)

                # Select denominator scaling for projection term
                if use_consensus_denom:
                    d_scale = torch.clamp(torch.mean(denom, dim=dims, keepdim=True), min=eps_geom)
                else:
                    d_scale = denom

                if use_jordan_symmetrized:
                    # S = 0.5 * (H(A(v)) + A(H(v)))
                    # Forward Term: H(A(v))
                    dot_product_forward = torch.sum(w * step_direction_fp32, dim=dims, keepdim=True)
                    step_natural_forward = step_direction_fp32 - (alpha_dynamic * dot_product_forward / proj_denom_stable) * w

                    # Reverse Term: A(H(v)) 
                    dot_product_raw = torch.sum(w * exp_avg, dim=dims, keepdim=True)
                    raw_projected = exp_avg - (alpha_dynamic * dot_product_raw / proj_denom_stable) * w
                    step_natural_reverse = raw_projected / d_scale

                    step_natural_fp32 = 0.5 * (step_natural_forward + step_natural_reverse)
                else:
                    # Standard Natural Step: H(A(v)) with optional consensus projection scale
                    dot_product_forward = torch.sum(w * step_direction_fp32, dim=dims, keepdim=True)
                    if use_consensus_denom:
                        # Strictly symmetric single-pass S_stab(v) = D^-1 v - c w w^T v / d_consensus
                        dot_product_raw = torch.sum(w * exp_avg, dim=dims, keepdim=True)
                        step_natural_fp32 = step_direction_fp32 - (alpha_dynamic * dot_product_raw / (proj_denom_stable * d_scale)) * w
                    else:
                        # Standard asymmetric forward-pass H(A(v)) using coordinate-wise denom
                        step_natural_fp32 = step_direction_fp32 - (alpha_dynamic * dot_product_forward / proj_denom_stable) * w

                # 9. Adaptive Non-linear Entropic Restoring Force
                if step_size > 0:
                    gamma_stable = min(gamma, 1.9 / step_size)
                else:
                    gamma_stable = 0.0
                
                #coeff_num = K_norm * K_prime * one_minus_K2
                #coeff_denom = K_prime.square() + damping_stable * one_minus_K2.square()
                #coeff_denom_stable = torch.clamp(coeff_denom, min=eps_geom)
                #coeff = gamma_stable * (coeff_num / coeff_denom_stable)
                #
                #entropic_force_fp32 = coeff * w
                entropic_force_fp32 = gamma_stable * w * (1.0 + x_norm.square())
                # 10. Compute proposed update vector
                proposed_update = step_natural_fp32 + entropic_force_fp32

                # 11. Stochastic Boundary Censorship
                update_norm = torch.norm(proposed_update, p=2, dim=dims, keepdim=True)
                max_step = 0.9 * (R - r_safe)
                scale_factor = torch.clamp(max_step / torch.clamp(update_norm, min=eps_geom), max=1.0)
                final_update = proposed_update * scale_factor

                # 12. Parameter Update
                p_fp32.add_(final_update, alpha=-step_size)
                p.copy_(p_fp32)

        return loss


class EntropicHypercubeAdamW(torch.optim.Optimizer):
    """
    Entropic Hypercube AdamW (EHA).
    
    An optimizer that restricts parameters to a d-dimensional hypercube [-R, R]^d 
    using coordinate-wise (decoupled) entropic potential dynamics and element-wise 
    boundary censorship. 
    
    By eliminating spatial reduction metrics, this optimizer runs with O(d) complexity, 
    requires zero GPU thread synchronization, and allows complete kernel fusion.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 damping=1e-5, gamma=1e-3, weight_decay=1e-2, R=1.5, kernel='softsign'):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if damping < 0.0:
            raise ValueError(f"Invalid damping: {damping}")
        if gamma < 0.0:
            raise ValueError(f"Invalid gamma (entropic force coefficient): {gamma}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if R <= 0.0:
            raise ValueError(f"Invalid boundary scale R: {R}")

        defaults = dict(
            lr=lr, 
            betas=betas, 
            eps=eps, 
            damping=damping, 
            gamma=gamma,
            weight_decay=weight_decay, 
            R=R,
            kernel=kernel,
        )
        super(EntropicHypercubeAdamW, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            damping = group['damping']
            gamma = group['gamma']
            weight_decay = group['weight_decay']
            R = group['R']
            kernel = group['kernel']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("EHA does not support sparse gradients.")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                step = state['step']

                # 1. Decoupled Weight Decay
                if weight_decay != 0:
                    p.mul_(1.0 - lr * weight_decay)

                # 2. Update biased first and second moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # 3. Bias corrections and step scaling
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                step_size = lr / bias_correction1

                # 4. Adam denominator (RMS scale)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_direction = exp_avg / denom

                # 5. Coordinate-wise geometry calculations (No sliced reductions!)
                r = torch.abs(p)
                r_safe = torch.clamp(r, min=1e-12, max=R - 1e-12)

                # 6. Kernel evaluation (Element-wise softsign or tanh)
                if kernel == 'softsign':
                    # Normalized mapping x / R -> K_norm
                    x_norm = r_safe / R
                    K_norm = x_norm / (x_norm + 1.0)
                    K_prime = 1.0 / (x_norm + 1.0).square()
                elif kernel == 'tanh':
                    x_norm = r_safe / R
                    K_norm = torch.tanh(x_norm)
                    K_prime = 1.0 - K_norm.square()
                else:
                    raise ValueError(f"Unknown kernel option: {kernel}")

                one_minus_K2 = torch.clamp(1.0 - K_norm.square(), min=1e-12)

                # Preconditioning scale adjustment
                scale = one_minus_K2.square() / (K_prime + damping * one_minus_K2.square())

                # 7. Adaptive non-linear entropic restoring force
                gamma_stable = min(gamma, 1.9 / step_size) if step_size > 0 else 0.0
                
                coeff_num = K_norm * K_prime * one_minus_K2
                coeff_denom = K_prime.square() + damping * one_minus_K2.square()
                coeff_denom_stable = torch.clamp(coeff_denom, min=1e-12)
                coeff = gamma_stable * (coeff_num / coeff_denom_stable)
                
                # Pull back along each parameter's coordinate axis (continuous scaling using p / r_safe)
                entropic_force = coeff * (p / r_safe)

                # 8. Combined proposed update
                proposed_update = step_direction * scale + entropic_force

                # 9. Coordinate-wise Boundary Censorship
                # Using the decoupled L_infinity boundary safety: max step size per coordinate is 0.9 * (R - |p_i|)
                max_step = 0.9 * (R - r_safe)
                final_update = torch.clamp(proposed_update, min=-max_step, max=max_step)

                # 10. Apply step: p = p - step_size * final_update
                p.add_(final_update, alpha=-step_size)

        return loss

class EntropicAdamW3(torch.optim.Optimizer):
    """
    Implements a mathematically robust Entropic AdamW optimizer.
    
    This version resolves:
      1. Dimension-Invariant Norm Scaling: Scales the radial parameter r by 
         the square root of the layer's reduction dimension to prevent immediate 
         boundary saturation in high-dimensional layers.
      2. Manifold-Space Clamping: Computes and clamps K_norm within the safe open 
         interval [0, 1 - epsilon] (and K within [-1 + epsilon, 1 - epsilon]) to 
         retain maximum representational capacity while protecting against underflow.
      3. Tanh Wall/Underflow Protection: Reconstructs K_prime safely from the clamped 
         K_norm/K, ensuring the logarithmic barrier restoring force never 
         underflows to 0.0.
      4. Tangential Learning Rate Preservation: Scales the Sherman-Morrison 
         base metric by (1.0 + damping) rather than damping, preserving tangential learning.
      5. Refined Boundary Stability: Caps the entropic coefficient gamma at 
         0.95 / step_size to satisfy the discrete stability bound (dt * c < 2.0).
      6. Fix for 0D Parameters: Scalar tensors of dimension 0 bypass the complex 
         boundary mapping to avoid out-of-bounds index errors on norms.
      7. Fix for Projection Lag: Recomputes parameter norms after applying updates 
         to guarantee exact manifold projection.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 damping=1e-5, gamma=1e-3, weight_decay=1e-2, kernel='softsign',
                 boundary_type='radial', safety_epsilon=1e-4):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if damping < 0.0:
            raise ValueError(f"Invalid damping: {damping}")
        if gamma < 0.0:
            raise ValueError(f"Invalid gamma (entropic force coefficient): {gamma}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if safety_epsilon <= 0.0 or safety_epsilon >= 1.0:
            raise ValueError(f"Invalid safety_epsilon: {safety_epsilon}")
        if boundary_type not in ('coordinate', 'radial'):
            raise ValueError(f"Unknown boundary_type: {boundary_type}")

        defaults = dict(
            lr=lr, 
            betas=betas, 
            eps=eps, 
            damping=damping, 
            gamma=gamma,
            weight_decay=weight_decay, 
            kernel=kernel,
            boundary_type=boundary_type,
            safety_epsilon=safety_epsilon
        )
        super(EntropicAdamW3, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            damping = group['damping']
            gamma = group['gamma']
            weight_decay = group['weight_decay']
            kernel = group['kernel']
            boundary_type = group['boundary_type']
            safety_epsilon = group['safety_epsilon']

            # Compute maximum parameter bounds dynamically (Hole 1 & 2 Fixes)
            if kernel == 'softsign':
                max_param_val = (1.0 - safety_epsilon) / safety_epsilon
            else: # tanh
                max_param_val = 0.5 * math.log((2.0 - safety_epsilon) / safety_epsilon)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("EntropicAdamW3 does not support sparse gradients.")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                step = state['step']

                # 1. Decoupled weight decay
                if weight_decay != 0:
                    p.mul_(1.0 - lr * weight_decay)

                # 2. Dynamic Stability Cap (Hole 2 Fix)
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                step_size = lr / bias_correction1
                gamma_capped = min(gamma, 0.95 / step_size)

                # 3. Update biased first and second moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # 4. Adam preconditioned update direction v
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                v = exp_avg / denom

                # 5. Geometrically Corrected Updates
                if p.dim() == 0:
                    # Treat scalar (0D) parameters normally, skipping manifold projections
                    p.add_(v, alpha=-step_size)

                elif boundary_type == 'radial':
                    # Determine reduction dimensions and scaling factor (Hole 3 Fix)
                    if p.dim() > 1:
                        dims = list(range(1, p.dim()))
                        r = torch.norm(p, p=2, dim=dims, keepdim=True)
                        num_elements = p.numel() // p.size(0)
                    else:
                        dims = [0]
                        r = torch.norm(p, p=2, dim=0, keepdim=True)
                        num_elements = p.numel()
                    
                    dim_scale = math.sqrt(num_elements)
                    r_scaled = r / dim_scale
                    r_safe = torch.clamp(r, min=1e-12)
                    u = p / r_safe  # Unit radial vector

                    # Map and clamp in Manifold Space (Hole 1 & 2 Fixes)
                    if kernel == 'softsign':
                        K_norm = r_scaled / (r_scaled + 1.0)
                    elif kernel == 'tanh':
                        K_norm = torch.tanh(r_scaled)

                    K_norm = torch.clamp(K_norm, max=1.0 - safety_epsilon)
                    one_minus_K2 = 1.0 - K_norm.square()

                    # Reconstruct K_prime safely from clamped manifold coordinates
                    if kernel == 'tanh':
                        K_prime = 1.0 - K_norm.square()
                    else: # softsign
                        K_prime = (1.0 - K_norm).square()

                    # Chain rule factor for dimension scaling
                    K_prime_eff = K_prime / dim_scale

                    # Sherman-Morrison inversion of G_damped = λ*I + α*(u x u^T)
                    identity_scale = 1.0 + damping
                    alpha = K_prime_eff.square() / (one_minus_K2.square() + eps)
                    alpha_tilde = alpha / (identity_scale + alpha + eps)
                    
                    # Compute inner product u^T v along the reduction dimensions
                    if p.dim() > 1:
                        u_dot_v = torch.sum(u * v, dim=dims, keepdim=True)
                    else:
                        u_dot_v = torch.sum(u * v, dim=0, keepdim=True)

                    # Natural gradient step direction
                    metric_v = (1.0 / identity_scale) * (v - alpha_tilde * u_dot_v * u)

                    # Radial Restoring Force
                    coeff_num = K_norm * K_prime_eff * one_minus_K2
                    coeff_denom = K_prime_eff.square() + damping * one_minus_K2.square()
                    coeff = gamma_capped * (coeff_num / (coeff_denom + eps))

                    update = metric_v + coeff * u
                    p.add_(update, alpha=-step_size)

                    # Strictly enforce parameter bounds on the newly updated values
                    # (Recomputes norm to resolve projection lag)
                    r_new = torch.norm(p, p=2, dim=dims, keepdim=True)
                    r_new_safe = torch.clamp(r_new, min=1e-12)
                    max_r = max_param_val * dim_scale
                    p.mul_(torch.clamp(max_r / r_new_safe, max=1.0))

                else:  # coordinate-wise
                    # Map and clamp in Manifold Space (Hole 1 & 2 Fixes)
                    if kernel == 'softsign':
                        K = p / (p.abs() + 1.0)
                    elif kernel == 'tanh':
                        K = torch.tanh(p)

                    K = torch.clamp(K, min=-1.0 + safety_epsilon, max=1.0 - safety_epsilon)
                    one_minus_K2 = 1.0 - K.square()

                    # Reconstruct K_prime safely from clamped manifold coordinates
                    if kernel == 'tanh':
                        K_prime = 1.0 - K.square()
                    else: # softsign
                        K_prime = (1.0 - K.abs()).square()

                    # Pointwise inversion of the diagonal metric
                    metric_scale = one_minus_K2.square() / (K_prime.square() + damping * one_minus_K2.square() + eps)
                    metric_v = v * metric_scale

                    # Coordinate restoring force
                    coeff_num = K * K_prime * one_minus_K2
                    coeff_denom = K_prime.square() + damping * one_minus_K2.square()
                    restoring_step = gamma_capped * (coeff_num / (coeff_denom + eps))

                    update = metric_v + restoring_step
                    p.add_(update, alpha=-step_size)

                    # Strictly enforce parameter bounds corresponding to maximum coordinate-wise K
                    p.clamp_(min=-max_param_val, max=max_param_val)

        return loss


class EntropicAdamW4(torch.optim.Optimizer):
    
    """
    Implements a mathematically optimized diagonal Entropic AdamW.
    
    This version combines the best features of previous versions:
      1. Diagonal Decoupling: Avoids the coordinate-coupling of Sherman-Morrison,
         preserving 100% of Adam's local per-parameter adaptivity.
      2. Single-Power Preconditioning: Uses the first power of K_prime in the 
         denominator to prevent double-division with Adam's running RMS scale.
      3. Smooth Softsign Derivative: Uses the C^∞ pseudo-absolute softsign mapping 
         to resolve the metric crease at the origin.
      4. Post-Update Projection: Enforces strict manifold bounds post-step to 
         prevent discretization boundary violations.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 damping=1e-5, gamma=1e-3, weight_decay=1e-2, kernel='softsign',
                 boundary_type='radial', safety_epsilon=1e-4, manifold_scale=1.0,
                 smooth_delta=1e-4):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if damping < 0.0:
            raise ValueError(f"Invalid damping: {damping}")
        if gamma < 0.0:
            raise ValueError(f"Invalid gamma: {gamma}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if safety_epsilon <= 0.0 or safety_epsilon >= 1.0:
            raise ValueError(f"Invalid safety_epsilon: {safety_epsilon}")
        if manifold_scale <= 0.0:
            raise ValueError(f"Invalid manifold_scale: {manifold_scale}")
        if smooth_delta <= 0.0:
            raise ValueError(f"Invalid smooth_delta: {smooth_delta}")
        if boundary_type not in ('coordinate', 'radial'):
            raise ValueError(f"Unknown boundary_type: {boundary_type}")

        defaults = dict(
            lr=lr, 
            betas=betas, 
            eps=eps, 
            damping=damping, 
            gamma=gamma,
            weight_decay=weight_decay, 
            kernel=kernel,
            boundary_type=boundary_type,
            safety_epsilon=safety_epsilon,
            manifold_scale=manifold_scale,
            smooth_delta=smooth_delta
        )
        super(EntropicAdamW4, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            damping = group['damping']
            gamma = group['gamma']
            weight_decay = group['weight_decay']
            kernel = group['kernel']
            boundary_type = group['boundary_type']
            safety_epsilon = group['safety_epsilon']
            manifold_scale = group['manifold_scale']
            delta = group['smooth_delta']

            # Compute maximum parameter bounds dynamically
            if kernel == 'softsign':
                max_param_val = (1.0 - safety_epsilon) / safety_epsilon
            else: # tanh
                max_param_val = 0.5 * math.log((2.0 - safety_epsilon) / safety_epsilon)

            # Apply scale factor to boundaries
            max_param_val_scaled = max_param_val * manifold_scale

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("EntropicAdamW does not support sparse gradients.")

                state = self.state[p]

                # State initialization (Strictly coordinate-wise)
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                step = state['step']

                # 1. Decoupled weight decay
                if weight_decay != 0:
                    p.mul_(1.0 - lr * weight_decay)

                # 2. Dynamic Stability Cap (dt * c < 2.0 stability threshold)
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                step_size = lr / bias_correction1
                gamma_capped = min(gamma, 0.95 / step_size)

                # 3. Standard Coordinate-wise Moment Updates
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # 4. Adam Denominator
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                v = exp_avg / denom

                # 5. Geometrically Corrected Updates
                if p.dim() == 0:
                    # Treat scalar (0D) parameters normally, skipping manifold projections
                    p.add_(v, alpha=-step_size)

                elif boundary_type == 'radial':
                    # Determine reduction dimensions and scaling factor
                    if p.dim() > 1:
                        dims = list(range(1, p.dim()))
                        r = torch.norm(p, p=2, dim=dims, keepdim=True)
                        num_elements = p.numel() // p.size(0)
                    else:
                        dims = [0]
                        r = torch.norm(p, p=2, dim=0, keepdim=True)
                        num_elements = p.numel()
                    
                    dim_scale = math.sqrt(num_elements)
                    r_scaled = r / (dim_scale * manifold_scale) 
                    r_safe = torch.clamp(r, min=1e-12)
                    u = p / r_safe

                    # Map and clamp in Manifold Space (C^∞ Softsign Smooth Fix)
                    if kernel == 'softsign':
                        s = torch.sqrt(r_scaled.square() + delta)
                        K_norm = r_scaled / (s + 1.0)
                    elif kernel == 'tanh':
                        K_norm = torch.tanh(r_scaled)

                    K_norm = torch.clamp(K_norm, max=1.0 - safety_epsilon)
                    one_minus_K2 = 1.0 - K_norm.square()

                    # Reconstruct K_prime safely from clamped manifold coordinates
                    if kernel == 'tanh':
                        K_prime = 1.0 - K_norm.square()
                    else: # softsign (Smooth Derivative)
                        s = torch.sqrt(r_scaled.square() + delta)
                        K_prime = (delta / s + 1.0) / (s + 1.0).square()

                    # Chain rule factor incorporating dimension and manifold scaling
                    K_prime_eff = K_prime / (dim_scale * manifold_scale)

                    # --- Element-wise Diagonal Scale (Corrected) ---
                    # To prevent global coordinate coupling, we apply the metric scale 
                    # element-wise. We use the first power of K_prime_eff to compensate 
                    # for Adam's existing division.
                    scale = one_minus_K2.square() / (K_prime_eff + damping * one_minus_K2.square() + eps)
                    metric_v = v * scale

                    # Radial Restoring Force
                    coeff_num = K_norm * K_prime_eff * one_minus_K2
                    coeff_denom = K_prime_eff.square() + damping * one_minus_K2.square()
                    coeff = gamma_capped * (coeff_num / (coeff_denom + eps))

                    update = metric_v + coeff * u
                    p.add_(update, alpha=-step_size)

                    # Strictly enforce parameter bounds corresponding to maximum K_norm
                    r_new = torch.norm(p, p=2, dim=dims, keepdim=True)
                    r_new_safe = torch.clamp(r_new, min=1e-12)
                    max_r = max_param_val_scaled * dim_scale
                    p.mul_(torch.clamp(max_r / r_new_safe, max=1.0))

                else:  # coordinate-wise
                    p_scaled = p / manifold_scale 

                    # Map and clamp in Manifold Space (C^∞ Softsign Smooth Fix)
                    if kernel == 'softsign':
                        s = torch.sqrt(p_scaled.square() + delta)
                        K = p_scaled / (s + 1.0)
                    elif kernel == 'tanh':
                        K = torch.tanh(p_scaled)

                    K = torch.clamp(K, min=-1.0 + safety_epsilon, max=1.0 - safety_epsilon)
                    one_minus_K2 = 1.0 - K.square()

                    # Reconstruct K_prime safely from clamped manifold coordinates
                    if kernel == 'tanh':
                        K_prime = 1.0 - K.square()
                    else: # softsign (Smooth Derivative)
                        s = torch.sqrt(p_scaled.square() + delta)
                        K_prime = (delta / s + 1.0) / (s + 1.0).square()

                    # Chain rule factor incorporating manifold scale
                    K_prime_eff = K_prime / manifold_scale

                    # Pointwise inversion of the diagonal metric (using first power of K_prime_eff)
                    metric_scale = one_minus_K2.square() / (K_prime_eff + damping * one_minus_K2.square() + eps)
                    metric_v = v * metric_scale

                    # Coordinate restoring force
                    coeff_num = K * K_prime_eff * one_minus_K2
                    coeff_denom = K_prime_eff.square() + damping * one_minus_K2.square()
                    restoring_step = gamma_capped * (coeff_num / (coeff_denom + eps))

                    update = metric_v + restoring_step
                    p.add_(update, alpha=-step_size)

                    # Strictly enforce parameter bounds corresponding to maximum coordinate-wise K
                    p.clamp_(min=-max_param_val_scaled, max=max_param_val_scaled)

                    # Prevent coordinate-wise momentum wind-up at the boundaries
                    at_pos_boundary = (p >= max_param_val_scaled)
                    at_neg_boundary = (p <= -max_param_val_scaled)
                    exp_avg.masked_fill_(at_pos_boundary & (exp_avg > 0), 0.0)
                    exp_avg.masked_fill_(at_neg_boundary & (exp_avg < 0), 0.0)

            return loss
    
class EntropicAdamW(torch.optim.Optimizer):
    """
    Implements Entropic AdamW, combining the fast, layer-wise adaptive 
    second-moment scaling of AdamW with the Lean-verified coordinate 
    boundary protection of Entropic Natural Gradient Descent.
    
    Optimized for torch.compile() compatibility.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 damping=1e-5, weight_decay=1e-2, kernel='softsign'):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if damping < 0.0:
            raise ValueError(f"Invalid damping: {damping}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(
            lr=lr, 
            betas=betas, 
            eps=eps, 
            damping=damping, 
            weight_decay=weight_decay, 
            kernel=kernel
        )
        super(EntropicAdamW, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            damping = group['damping']
            weight_decay = group['weight_decay']
            kernel = group['kernel']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("EntropicAdamW does not support sparse gradients.")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                step = state['step']

                # 1. Decoupled weight decay
                if weight_decay != 0:
                    p.mul_(1.0 - lr * weight_decay)

                # 2. Update biased first and second moment estimates
                # (Scalar values are used, which is fully compatible with addcmul_)
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # 3. Bias corrections and step scaling
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                step_size = lr / bias_correction1

                # 4. Adam denominator (RMS scale)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                # 5. Compute the Lean-verified Inverse Damped Metric scaling factor
                # (Using standard operators for easy torch.compile tracing)
                if kernel == 'softsign':
                    denom_k = p.abs() + 1.0
                    K = p / denom_k
                    K_prime = 1.0 / denom_k.square()
                elif kernel == 'tanh':
                    K = torch.tanh(p)
                    K_prime = 1.0 - K.square()
                else:
                    raise ValueError(f"Unknown kernel option: {kernel}")

                # G_d⁻¹ = (1 - K²)² / (K'² + λ * (1 - K²)²)
                one_minus_K2 = 1.0 - K.square()
                num = one_minus_K2.square()
                denom_metric = K_prime.square() + damping * num
                
                # Prevent division by zero underflow
                denom_metric = torch.clamp(denom_metric, min=1e-15)
                scale = num / denom_metric

                # 6. Apply step: p = p - step_size * scale * (exp_avg / denom)
                # We calculate the update tensor, and apply it in-place to p
                update = (exp_avg / denom) * scale
                p.add_(update, alpha=-step_size)

        return loss


class UnifiedEntropicAdamW(torch.optim.Optimizer):
    """
    Unified Entropic AdamW Optimizer with Hypersphere and Hypercube Modes.
    
    Provides mathematically rigorous boundary constraints for stable long-context
    and geometric parameter training.
    
    Modes:
      - 'sphere': Constrains parameter slices (e.g., rows of a weight matrix)
                  to reside within a hypersphere of radius R. Uses tangent momentum
                  projection and a radial-only boundary censorship mechanism.
      - 'hypercube': Constrains each coordinate of the parameters independently 
                     to the interval [-R, R]. Runs with O(d) complexity and zero
                     dimensional reductions.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 damping=0.05, alpha=0.05, gamma=1e-3, weight_decay=1e-2, 
                 R=2.5, mode='sphere', kernel='softsign'):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if damping < 0.0:
            raise ValueError(f"Invalid damping: {damping}")
        if alpha < 0.0:
            raise ValueError(f"Invalid alpha: {alpha}")
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
            betas=betas, 
            eps=eps, 
            damping=damping, 
            alpha=alpha,
            gamma=gamma,
            weight_decay=weight_decay, 
            R=R,
            mode=mode_lower,
            kernel=kernel,
        )
        super(UnifiedEntropicAdamW, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            damping = group['damping']
            alpha = group['alpha']
            gamma = group['gamma']
            weight_decay = group['weight_decay']
            R = group['R']
            mode = group['mode']
            kernel = group['kernel']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("UnifiedEntropicAdamW does not support sparse gradients.")

                state = self.state[p]
                eps_geom = 1e-12

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, dtype=torch.float32, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, dtype=torch.float32, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                step = state['step']

                p_fp32 = p.to(torch.float32)
                grad_fp32 = grad.to(torch.float32)

                # 1. Decoupled Weight Decay (Fixed: Applied to multi-dimensional parameters)
                if p_fp32.dim() > 1 and weight_decay != 0:
                    p_fp32.mul_(1.0 - lr * weight_decay)

                # 2. Update biased first and second moment estimates
                exp_avg.mul_(beta1).add_(grad_fp32, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1.0 - beta2)

                # 3. Bias corrections
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                step_size = lr / bias_correction1

                # 4. Adam preconditioning denominator
                eps_stable = max(eps, eps_geom)
                exp_avg_sq_safe = torch.clamp(exp_avg_sq, min=0.0)
                denom = (exp_avg_sq_safe.sqrt() / math.sqrt(bias_correction2)).add_(eps_stable)
                step_direction_fp32 = exp_avg / denom

                # 5. Fallback for 1D parameters (e.g., biases or normalization vectors)
                if p.dim() <= 1:
                    p_fp32.add_(step_direction_fp32, alpha=-step_size)
                    p.copy_(p_fp32.to(p.dtype))
                    continue

                # --- MODE 1: HYPERSPHERE GEOMETRY ---
                if mode == 'sphere':
                    # 6. Sliced norm computation and unit vector normalization
                    dims = list(range(1, p.dim()))
                    r = torch.norm(p_fp32, p=2, dim=dims, keepdim=True)
                    r_clamped = torch.clamp(r, max=R - eps_geom)
                    r_safe = torch.clamp(r_clamped, min=eps_geom)
                    w = p_fp32 / r_safe

                    # Momentum Projection to tangent space (reduces phase lag)
                    
                    dot_product_momentum = torch.sum(w * exp_avg, dim=dims, keepdim=True)
                    exp_avg.sub_(dot_product_momentum * w)
                    step_direction_fp32 = exp_avg / denom

                    # 7. Compute Kernel Parameters and Pullback Metric Scale
                    x_norm = r_safe / R
                    if kernel == 'softsign':
                        K_norm = x_norm / (x_norm + 1.0)
                        K_prime = 1.0 / (x_norm + 1.0).square()
                    elif kernel == 'tanh':
                        K_norm = torch.tanh(torch.clamp(x_norm, max=20.0))
                        K_prime = torch.clamp(1.0 - K_norm.square(), min=0.0)
                    else:
                        raise ValueError(f"Unknown kernel: {kernel}")

                    one_minus_K2 = torch.clamp(1.0 - K_norm.square(), min=eps_geom)
                    metric_scale = K_prime / (R * one_minus_K2)
                    alpha_dynamic = alpha * metric_scale.square()

                    # 8. Sliced Natural Projection
                    w_norm_sq = torch.sum(w * w, dim=dims, keepdim=True)
                    damping_stable = max(damping, eps_geom)
                    proj_denom = damping_stable + alpha_dynamic * w_norm_sq
                    proj_denom_stable = torch.clamp(proj_denom, min=eps_geom)

                    # Forward Term: H(A(v))
                    dot_product_forward = torch.sum(w * step_direction_fp32, dim=dims, keepdim=True)
                    step_natural_forward = step_direction_fp32 - (alpha_dynamic * dot_product_forward / proj_denom_stable) * w
                    # Reverse Term: A(H(v)) (Fixed: Uses element-wise denom to preserve adaptivity)
                    dot_product_raw = torch.sum(w * exp_avg, dim=dims, keepdim=True)
                    raw_projected = exp_avg - (alpha_dynamic * dot_product_raw / proj_denom_stable) * w
                    step_natural_reverse = raw_projected / denom

                    step_natural_fp32 = 0.5 * (step_natural_forward + step_natural_reverse)
                    
                    # 9. Adaptive Non-linear Entropic Restoring Force
                    gamma_stable = min(gamma, 1.9 / step_size) if step_size > 0 else 0.0
                    entropic_force_fp32 = gamma_stable * w * (1.0 + x_norm.square())
                    
                    # 10. Combine forces
                    proposed_update = step_natural_fp32 + entropic_force_fp32

                    # 11. Stochastic Boundary Censorship (Fixed: Radial-only projection)
                    radial_step = torch.sum(w * proposed_update, dim=dims, keepdim=True)
                    tangent_step = proposed_update - radial_step * w
                    
                    max_radial_step = 0.9 * (R - r_safe)
                    # Only scale down the radial step if it pushes the parameter outward (radial_step > 0)
                    scale_radial = torch.where(
                        radial_step > 0,
                        torch.clamp(max_radial_step / torch.clamp(radial_step, min=eps_geom), max=1.0),
                        torch.ones_like(radial_step)
                    )
                    final_update = (radial_step * scale_radial) * w + tangent_step

                # --- MODE 2: HYPERCUBE GEOMETRY ---
                else:
                    # 6. Coordinate-wise geometry calculations
                    r = torch.abs(p_fp32)
                    r_safe = torch.clamp(r, min=eps_geom, max=R - eps_geom)

                    # 7. Kernel evaluation (Element-wise softsign or tanh)
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

                    # Coordinate-wise preconditioning scale adjustment
                    scale = one_minus_K2.square() / (K_prime + damping * one_minus_K2.square())

                    # 8. Adaptive non-linear entropic restoring force
                    gamma_stable = min(gamma, 1.9 / step_size) if step_size > 0 else 0.0
                    
                    coeff_num = K_norm * K_prime * one_minus_K2
                    coeff_denom = K_prime.square() + damping * one_minus_K2.square()
                    coeff_denom_stable = torch.clamp(coeff_denom, min=eps_geom)
                    coeff = gamma_stable * (coeff_num / coeff_denom_stable)
                    
                    # Pull back along each parameter's coordinate axis (continuous scaling using p / r_safe)
                    entropic_force_fp32 = coeff * (p_fp32 / r_safe)

                    # 9. Combined proposed update
                    proposed_update = step_direction_fp32 * scale + entropic_force_fp32

                    # 10. Coordinate-wise Boundary Censorship (Protects coordinate boundaries independently)
                    max_step = 0.9 * (R - r_safe)
                    final_update = torch.clamp(proposed_update, min=-max_step, max=max_step)

                # 12. Apply final update and write back
                p_fp32.add_(final_update, alpha=-step_size)
                p.copy_(p_fp32.to(p.dtype))

        return loss



Linear          = nn.Linear
Patt            = HungryPattLayer
#dataset        = "shakespeare_char"
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

learning_rate   = 5e-4
min_lr          = learning_rate / 10

spl             = False
soloblock       = False
sspl            = False
wnll            = False
patt            = False

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
train_logging = False
exec(open('training_tools/configurator.py').read()) # overrides from command line or config file

def get_lr(it):

    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)

    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    if it > lr_decay_iters:
        return min_lr
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
    return min_lr + coeff * (learning_rate - min_lr)

decode, vocab_size = utils.get_meta(dataset,always_byte=True)
get_batch = utils.get_get_batch(block_size,batch_size,dataset,device,always_byte=True)


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
    optimizer: str  = 'adam'
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
            #utils.range_norm(predicted_output.flatten()),
            #utils.range_norm(self.block_input.flatten())
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
                    #cumsum_wnl = torch.cumsum(wnl, dim=1)
                    #indices = torch.arange(1, wnl.size(1) + 1, device=wnl.device).view(1, -1, 1)
                    #cumulative_surprise_avg = cumsum_wnl / indices
                    #wnl = cumulative_surprise_avg
                    mean_surprise = ((1.0-wnl.detach())/2.0).mean()*0.99
                    loss = loss / (1+ wnl.detach()*mean_surprise)
                    loss = (loss + wnl).mean()
                    #loss = (loss * torch.exp(-wnl) + wnl).mean()
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

        
        #optim_groups = [
        #        {'params': nodecay_params,  'weight_decay': 0.0},
        #        {'params': decay_params,    'weight_decay': wd},
        #]
        #return torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9,0.999))

        optim_groups = [
                {'params': nodecay_params,  'weight_decay': 0.0},
                #{'params': decay_params,    'weight_decay': wd},#TODO beware
                {'params': decay_params,    'weight_decay': 0.0},
        ]
        #return torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9,0.999))
        #return  EntropicAdamW2(
        #    optim_groups, 
        #    lr=lr, 
        #    damping=1e-5, 
        #    kernel='softsign'
        #)
        return  UnifiedEntropicAdamW(
            optim_groups, 
            lr=lr, 
        )
        #return LREntropicKFACOptimizer(
        #    self, 
        #    params=optim_groups,
        #    lr=lr, 
        #)


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
    #dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler

    model.compile()
    print(f"Starting training on {device}")
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

        #if iter % 100 == 0:
        #    for block in model.transformer.h:
        #        block.mlp.patt.add_tokens(1, optimizer)
        lr = get_lr(iter)
        decaying = 1.0 - (iter / decay_iters)
        decaying = torch.tensor(decaying, device=device, dtype=torch.float32)
        decaying = torch.clamp(decaying, 0.0, 1.0)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        torch.compiler.cudagraph_mark_step_begin()
        X, Y = get_batch('train')
        logits, loss = model(X, Y, decaying= decaying)
        loss.backward()
        optimizer.step()
        if iter % eval_interval == 0 or iter == max_iters - 1:
            t0,t1,t2 =eval(max_iters, get_batch, model, optimizer, t0,t1,t2, iter, decaying)
        optimizer.zero_grad(set_to_none=True)
    