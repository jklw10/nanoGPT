import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

import torch.optim.adamw
import torch.optim.sgd

import parts.utils as utils

from torch.func import functional_call, vmap

ema_thing = True
alpha = 0.9
class OptimizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, batch_size: int = None, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size # Max batch size the layer is designed for
        self.top_k_lr_selection = 5
        self.lr_update_momentum = 0.9
        self.start_perc = 0.8
        self.active_percentage = self.start_perc
        self.step = 0.0
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        
        self.register_buffer("wema", torch.zeros_like(self.weight))
        self.previous_bias_grad = None
        if bias:
            self.register_buffer("bema", torch.zeros_like(self.bias))
        
        # Stores the gradient from the *previous* backward pass practically a per weight bias, which gets added to the weight after a run with lr.
      
        self.register_buffer("previous_weight_grad", torch.zeros_like(self.weight))
        self.previous_bias_grad = None
        if bias:
            self.register_buffer("previous_bias_grad", torch.zeros_like(self.bias))
        
        # Stores the current batch of LR samples used for forward pass simulation
        if batch_size is not None:
            self.register_buffer("batch_lr_samples", torch.empty(batch_size))
        else:
            self.register_buffer("batch_lr_samples", torch.empty(0))

        self.current_lr_mean = nn.Parameter(torch.tensor(1e-5), requires_grad=False)
        self.current_lr_var = nn.Parameter(torch.tensor(1e-4), requires_grad=False) # Small initial variance

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Standard weight initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        # Initialize previous gradients and LR samples to zero/defaults
        with torch.no_grad():
            self.previous_weight_grad.zero_()
            if self.previous_bias_grad is not None:
                self.previous_bias_grad.zero_()
            self.current_lr_mean.fill_(0.5)
            self.current_lr_var.fill_(0.5)
            if(self.batch_size is not None):
                self._sample_batch_lrs() 
            if(ema_thing):
                num_active_out = math.ceil(self.out_features * self.active_percentage)
                self.wema =  self.weight.data.detach() 
                if self.bias is not None:
                    self.bema =  self.bias.data.detach() 
                    torch.zero_(self.bias.data[num_active_out:])
                torch.zero_(self.weight.data[num_active_out:,:])
            return
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input_data: (actual_batch_size, in_features)
        #if(not self.training):
        #    return nn.functional.linear(x,self.weight,self.bias)


        if(ema_thing):
            num_active_out = math.ceil(self.out_features * self.active_percentage)
            if(not self.training):
                aw = self.weight[:num_active_out, :]
                ab = None
                if self.bias is not None:
                    ab = self.bias[:num_active_out]
                active_output = nn.functional.linear(x,aw,ab)
                full_output = torch.zeros(x.shape[0],x.shape[1], self.out_features, device=x.device, dtype=x.dtype)
                full_output[:,:, :num_active_out] = active_output
                return full_output 
            # Create a mask to zero-out inactive weights
            # This is done on the fly and is very efficient on GPU
            #weight_mask = torch.zeros_like(self.weight.data, memory_format=torch.contiguous_format)
            #weight_mask[:num_active_out, :] = 1.0
            #skip ema on backward pass?
            aw = self.weight[:num_active_out, :]
            forward_weight = utils.funnyMulti(self.wema.detach()[:num_active_out, :], aw)
            simulated_weight = aw + (forward_weight - aw).detach()
            #simulated_weight = simulated_weight * weight_mask
            #simulated_weight = simulated_weight[:num_active_out, :]
            #simulated_weight = utils.funnyMulti(self.weight, self.wema)
            simulated_bias = None
            if self.bias is not None:
                ab = self.bias[:num_active_out]
                fwd_bias = utils.funnyMulti(self.bema.detach()[:num_active_out], ab)
                simulated_bias = ab + (fwd_bias - ab).detach()

            active_output = nn.functional.linear(x,simulated_weight,simulated_bias)
            full_output = torch.zeros(x.shape[0],x.shape[1], self.out_features, device=x.device, dtype=x.dtype)
            full_output[:,:, :num_active_out] = active_output
            return full_output 
        
        if(not self.training):
            return nn.functional.linear(x,self.weight,self.bias)
        if(self.batch_size is None):
            self.batch_size = x.shape[0]
            self._sample_batch_lrs()

        actual_batch_size = x.shape[0]
        if actual_batch_size != self.batch_size:
            if actual_batch_size > self.batch_size:
                raise ValueError(f"Actual batch size {actual_batch_size} exceeds layer's configured max batch size {self.batch_size}")
            
        simulated_weight = self.weight.unsqueeze(0) - \
                           self.previous_weight_grad.unsqueeze(0) * self.batch_lr_samples[:actual_batch_size].view(-1, 1, 1)
        
        output_per_sample = torch.bmm(x, simulated_weight.transpose(1, 2))
        #output_per_sample = torch.einsum('bsi,boi->bso', x, simulated_weight)
        simulated_bias = None
        if self.bias is not None:
            simulated_bias = self.bias.unsqueeze(0) - \
                             self.previous_bias_grad.unsqueeze(0) * self.batch_lr_samples[:actual_batch_size].view(-1, 1)
            output_per_sample += simulated_bias
        return output_per_sample

    def _weight_update(self) -> None:
        
        with torch.no_grad():
            
            if self.weight.grad is not None:
                if self.previous_weight_grad is not None:    
                    self.weight.data.sub_(self.previous_weight_grad * self.current_lr_mean) 

                adjusted_grad = self.weight.grad
                #adjusted_grad = utils.mmnorm(adjusted_grad)
                #adjusted_grad = utils.zca_newton_schulz(adjusted_grad, 2, 2)
                #
                self.previous_weight_grad.copy_(adjusted_grad)
                self.weight.grad.zero_()
            if self.bias is not None and self.bias.grad:
                if self.previous_bias_grad is not None:
                    self.bias.data.sub_(self.previous_bias_grad * self.current_lr_mean) 
                self.previous_bias_grad.copy_(self.bias.grad)
                self.bias.grad.zero_()
    
    def poststep(self):
        #pass
        if(ema_thing):
            num_active_out = math.ceil(self.out_features * self.active_percentage)
            self.step += 1.0
            self.active_percentage = min(self.start_perc+ (1.0-self.start_perc)*(self.step/30000.0),1.0 )
            #self.active_percentage = torch.sqrt(self.active_percentage) 
            with torch.no_grad():
                self.wema[:num_active_out,:] += alpha * (self.weight.data[:num_active_out,:] - self.wema[:num_active_out,:])
                if self.bias is not None:
                    self.bema[:num_active_out] += alpha * (self.bias.data[:num_active_out]  - self.bema[:num_active_out])

    def prestep(self, per_sample_losses: torch.Tensor):
        """
        Call this method *before* the optimizer.step().
        """
        if(ema_thing):
            #with torch.no_grad():
            #    self.wema += espeed * (self.weight.data - self.wema)
            #    if self.bias is not None:
            #        self.bema += espeed * (self.bias.data - self.bema)
            return
        
        #if(ema_thing):
        #    with torch.no_grad():
        #        self.wema += 0.1 * (self.weight.data - self.wema)
        #        if self.bias is not None:
        #            self.bema += 0.1 * (self.bias.data - self.bema)
        #    return
        self._update_lr_stats(per_sample_losses)
        self._weight_update()
        self._sample_batch_lrs()

    @torch.no_grad()
    def _sample_batch_lrs(self):
        mean_lr = self.current_lr_mean
        std_dev_lr = self.current_lr_var.sqrt()

        # Sample LRs from a normal distribution. Clamp to reasonable bounds (e.g., > 0)
        #lr_samples = torch.randn(self.batch_size, device=self.weight.device) * std_dev_lr + mean_lr
        lr_samples = torch.distributions.Beta(mean_lr, 3.0).sample([self.batch_size])
        lr_samples = torch.clamp(lr_samples, min=1e-8, max=1.0) # Ensure LRs are positive and bounded
        self.batch_lr_samples[:self.batch_size].copy_(lr_samples)

    @torch.no_grad()
    def _update_lr_stats(self, per_sample_losses: torch.Tensor):
        if per_sample_losses.ndim > 1:
            raise ValueError("per_sample_losses must be a 1D tensor (per-sample loss).")
        
        actual_batch_size = per_sample_losses.shape[0]

        lrs_used = self.batch_lr_samples
        # Find the top_k LRs that resulted in the lowest losses
        # If actual_batch_size is smaller than top_k_lr_selection, use actual_batch_size
        k_val = min(self.top_k_lr_selection, actual_batch_size)
        
        # Use topk to find the indices of the 'k_val' lowest losses
        # per_sample_losses.topk(k, largest=False) returns (values, indices)
        _, best_loss_indices = per_sample_losses.topk(k_val, largest=False)
        
        # Get the LRs corresponding to these best losses
        best_lrs = lrs_used[best_loss_indices]
        # Calculate new mean and variance from these best LRs
        new_mean = best_lrs.mean()
        new_var = best_lrs.var() if k_val > 1 else torch.tensor(0.0) # If only one sample, variance is 0
        # Smoothly update the layer's current_lr_mean and current_lr_var
        # These will be used to generate LR samples for the *next* forward pass.
        # check if sane, i'm suspicious of .mul on the data itself.
        self.current_lr_mean.data.mul_(1 - self.lr_update_momentum).add_(new_mean * self.lr_update_momentum)
        self.current_lr_var.data.mul_(1 - self.lr_update_momentum).add_(new_var * self.lr_update_momentum)


class ChaosRollbackOptimizer(torch.optim.Optimizer):
    def __init__(self, 
                 model, 
                 base_optimizer, 
                 checkpoint_freq=50, 
                 sensitivity=1.1, 
                 sigma=1e-4,
                 **kwargs):
        super().__init__(base_optimizer.param_groups, base_optimizer.defaults)
        
        self.model = model
        self.base_optim = base_optimizer
        self.state = base_optimizer.state
        
        self.checkpoint_freq = checkpoint_freq
        self.sensitivity = sensitivity
        self.sigma = sigma
        self.lgn = 0
        self.grad_ma = None
        self.steps = 0
        self.safe_harbor_state = None 
        self.safe_harbor_optim = None
        self.warmup_steps= 1000
        self._checkpoint()

    def _checkpoint(self):
        self.safe_harbor_state = {
            k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
        }
        self.safe_harbor_optim = copy.deepcopy(self.base_optim.state_dict())

    def _rollback_and_perturb(self):
        # 1. Load weights
        self.model.load_state_dict(self.safe_harbor_state)
        
        # 2. Load optimizer state (clears momentum buffers that were heading towards the cliff)
        self.base_optim.load_state_dict(self.safe_harbor_optim)
        
        with torch.no_grad():
            for group in self.base_optim.param_groups:
                for p in group['params']:
                    if p.grad is None: 
                        continue
                    scale = p.abs().mean() + 1e-6
                    noise = torch.randn_like(p) * scale * self.sigma
                    p.add_(noise)
   
    def _rollback(self):
        self.model.load_state_dict(self.safe_harbor_state)
        self.base_optim.load_state_dict(self.safe_harbor_optim)
        

    @torch.no_grad()
    def step(self, closure=None):
        grad_norm = self.get_gradnorm()
        
        if self.grad_ma is None:
            self.grad_ma = grad_norm

        is_singularity = False
        if self.steps > self.warmup_steps:
            is_singularity = grad_norm > (self.grad_ma * self.sensitivity)
            self.lgn = grad_norm

        if is_singularity:
            self._rollback_and_perturb()
            
            self.grad_ma = self.grad_ma * self.sensitivity
            self.zero_grad()
            with torch.enable_grad():
                loss = closure()
            self.base_optim.step()
            self.zero_grad()
            return loss

        # 3. Standard Step
        self.base_optim.step()
        
        # Update MA
        self.grad_ma = 0.9 * self.grad_ma + 0.1 * grad_norm
        
        # 4. Manage Checkpoints
        self.steps += 1
        if self.steps % self.checkpoint_freq == 0:
            self._checkpoint()

    def get_gradnorm(self):
        stack = [p.grad.norm(p=2) for group in self.param_groups for p in group['params'] if p.grad is not None]
        grad_norm = torch.norm(torch.stack(stack), p=2) if len(stack) > 0 else torch.tensor(0.0)
        return grad_norm
    

def orthogonalize_only(grad, param):
    # Flatten logic ensures this works for Conv2d (4D) or Linear (2D)
    g = grad.view(-1)
    w = param.view(-1)
    # Calculate projection of g onto w
    w_norm_sq = torch.dot(w, w).add(1e-30)
    proj = torch.dot(w, g).div(w_norm_sq)
    # Remove the parallel component (g_orth = g - proj * w)
    g_orth = g.sub(proj * w)
    # Return reshaped to original dims
    return g_orth.view_as(grad)

class Grms(torch.optim.AdamW):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                #grad = orthogonalize_only(grad, p) 
                d2 = not (grad.ndim < 2)
                nd = [-1,-2] if d2 else [-1]
                rms = torch.rsqrt(grad.pow(2).mean(dim=nd, keepdim=True) + 1e-8)
                grad = grad * rms
                grad = utils.orthogonalize_gradients(grad, p).clone()
                p.grad = grad
                # 3. Standard Step
        return super().step(closure)

class EAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999) , eps=1e-8,
                 gamma=1e-3, mode='sphere', kernel='softsign'):
        defaults = dict(
            lr=lr, 
            betas=betas, 
            eps=eps, 
            gamma=gamma,
            mode=mode.lower(),
            kernel=kernel,
        )
        super(EAdamW, self).__init__(params, defaults)

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
            gamma = group['gamma']
            mode = group['mode']
            kernel = group['kernel']
            R = 1.0 
            eps_geom = 1e-12

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, dtype=torch.float32)
                    state['exp_avg_sq'] = torch.zeros_like(p, dtype=torch.float32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                step = state['step']

                p_fp32 = p.to(torch.float32)
                grad_fp32 = grad.to(torch.float32)

                # 1. Update biased moments
                exp_avg.mul_(beta1).add_(grad_fp32, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1.0 - beta2)

                # 2. Bias corrections
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                step_size = lr / bias_correction1

                # 3. Preconditioning denominator
                exp_avg_sq_safe = torch.clamp(exp_avg_sq, min=0.0)
                denom = (exp_avg_sq_safe.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_direction_fp32 = exp_avg / denom

                if p.dim() <= 1:
                    p_fp32.add_(step_direction_fp32, alpha=-step_size)
                    p.copy_(p_fp32.to(p.dtype))
                    continue

                # --- MODE 1: HYPERSPHERE GEOMETRY ---
                if mode == 'sphere':
                    dims = list(range(1, p.dim()))
                    r = torch.norm(p_fp32, p=2, dim=dims, keepdim=True)
                    r_clamped = torch.clamp(r, max=R - eps_geom)
                    r_safe = torch.clamp(r_clamped, min=eps_geom)
                    w = p_fp32 / r_safe

                    # Tangent projection of momentum to prevent out-of-plane drift
                    dot_product_momentum = torch.sum(w * exp_avg, dim=dims, keepdim=True)
                    exp_avg.sub_(dot_product_momentum * w)
                    step_direction_fp32 = exp_avg / denom

                    # Kernel scale derivations
                    x_norm = r_safe / R
                    if kernel == 'softsign':
                        K_norm = x_norm / (x_norm + 1.0)
                        K_prime = 1.0 / (x_norm + 1.0).square()
                    elif kernel == 'tanh':
                        K_norm = torch.tanh(torch.clamp(x_norm, max=20.0))
                        K_prime = torch.clamp(1.0 - K_norm.square(), min=0.0)

                    one_minus_K2 = torch.clamp(1.0 - K_norm.square(), min=eps_geom)
                    metric_scale = K_prime / (R * one_minus_K2)
                    
                    # Dynamically compute scale factor
                    alpha_dynamic = eps * metric_scale.square()

                    # Symmetrized projection utilizing tied damping (damping = eps)
                    w_norm_sq = torch.sum(w * w, dim=dims, keepdim=True)
                    proj_denom_stable = torch.clamp(eps + alpha_dynamic * w_norm_sq, min=eps_geom)

                    # Jordan Product composition: 0.5 * (H(A(v)) + A(H(v)))
                    dot_product_forward = torch.sum(w * step_direction_fp32, dim=dims, keepdim=True)
                    step_natural_forward = step_direction_fp32 - (alpha_dynamic * dot_product_forward / proj_denom_stable) * w
                    
                    dot_product_raw = torch.sum(w * exp_avg, dim=dims, keepdim=True)
                    raw_projected = exp_avg - (alpha_dynamic * dot_product_raw / proj_denom_stable) * w
                    step_natural_reverse = raw_projected / denom

                    step_natural_fp32 = 0.5 * (step_natural_forward + step_natural_reverse)
                    
                    # Entropic restoring force handles norm regularization exclusively
                    gamma_stable = min(gamma, 1.9 / step_size) if step_size > 0 else 0.0
                    entropic_force_fp32 = gamma_stable * w * (1.0 + x_norm.square())
                    
                    proposed_update = step_natural_fp32 + entropic_force_fp32

                    # Radial-only boundary censorship
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
                    r = torch.abs(p_fp32)
                    r_safe = torch.clamp(r, min=eps_geom, max=R - eps_geom)

                    x_norm = r_safe / R
                    if kernel == 'softsign':
                        K_norm = x_norm / (x_norm + 1.0)
                        K_prime = 1.0 / (x_norm + 1.0).square()
                    elif kernel == 'tanh':
                        K_norm = torch.tanh(x_norm)
                        K_prime = 1.0 - K_norm.square()

                    one_minus_K2 = torch.clamp(1.0 - K_norm.square(), min=eps_geom)

                    # Tied preconditioning using eps
                    scale = one_minus_K2.square() / (K_prime + eps * one_minus_K2.square())

                    gamma_stable = min(gamma, 1.9 / step_size) if step_size > 0 else 0.0
                    coeff_num = K_norm * K_prime * one_minus_K2
                    coeff_denom_stable = torch.clamp(K_prime.square() + eps * one_minus_K2.square(), min=eps_geom)
                    coeff = gamma_stable * (coeff_num / coeff_denom_stable)
                    
                    entropic_force_fp32 = coeff * (p_fp32 / r_safe)
                    proposed_update = step_direction_fp32 * scale + entropic_force_fp32

                    max_step = 0.9 * (R - r_safe)
                    final_update = torch.clamp(proposed_update, min=-max_step, max=max_step)

                # 4. Apply final update and copy back
                p_fp32.add_(final_update, alpha=-step_size)
                p.copy_(p_fp32.to(p.dtype))

        return loss