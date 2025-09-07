
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import copy
from torch.utils.data import DataLoader, random_split

import utils

class MultiHotVQVAEQuantizer2(nn.Module):
    def __init__(self, quant_dim, embed_dim, k=15, commitment_cost=0.25):
        super().__init__()
        self.embed_dim = embed_dim
        self.quant_dim = quant_dim
        self.k = k

        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.quant_dim, self.embed_dim)
        self.embedding.weight.data.uniform_(-1/self.quant_dim, 1/self.quant_dim)

    def forward(self, z_e):
        dist = torch.sum(z_e**2, dim=1, keepdim=True) - \
               2 * torch.matmul(z_e, self.embedding.weight.t()) + \
               torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()
        
        _, top_k_indices = torch.topk(-dist, k=self.k, dim=1)
        
        z_q_k = self.embedding(top_k_indices)
        
        z_q = torch.sum(z_q_k, dim=1)

        vq_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        total_vq_loss = vq_loss + self.commitment_cost * commitment_loss

        z_q_ste = z_e + (z_q - z_e).detach()
        return z_q_ste, total_vq_loss

class MultiHotVQVAEQuantizer(nn.Module):
    def __init__(self, quant_dim, embed_dim, k=15, commitment_cost=0.25):
        super().__init__()
        self.embed_dim = embed_dim
        self.quant_dim = quant_dim
        self.k = k

        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.quant_dim, self.embed_dim)
        self.embedding.weight.data.uniform_(-1/self.quant_dim, 1/self.quant_dim)

    def forward(self, z_e):
        dist = torch.sum(z_e**2, dim=1, keepdim=True) - \
               2 * torch.matmul(z_e, self.embedding.weight.t()) + \
               torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()
        
        _, top_k_indices = torch.topk(-dist, k=self.k, dim=1)
        
        z_q_k = self.embedding(top_k_indices)
        
        z_q = torch.sum(z_q_k, dim=1)

        vq_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        total_vq_loss = vq_loss + self.commitment_cost * commitment_loss

        z_q_ste = z_e + (z_q - z_e).detach()
        return z_q_ste, total_vq_loss
    
class GumbelQuantizer(nn.Module):
    """
    Implementation of the Gumbel-Softmax quantizer.
    This is a strong baseline that uses a differentiable relaxation of the sampling process.
    """
    def __init__(self, quant_dim, embed_dim, temperature=1.0):
        """
        Args:
            quant_dim (int): The dimension of the input logits.
            embed_dim (int): The dimension of the output vectors.
            temperature (float): The tau parameter for the Gumbel-Softmax.
        """
        super().__init__()
        self.temperature = temperature
        # The codebook is a standard linear layer
        self.embedding = nn.Linear(quant_dim, embed_dim, bias=False)

    def forward(self, logits):
        """
        Args:
            logits (Tensor): The output of the encoder. Shape: (batch, quant_dim)
        """
        # --- Gumbel-Softmax Sampling ---
        # F.gumbel_softmax produces a soft, one-hot-like vector that is differentiable.
        # `hard=True` implements the straight-through variant automatically.
        # Forward pass: returns a hard one-hot vector.
        # Backward pass: uses the gradient from the soft, differentiable approximation.
        y_soft = F.gumbel_softmax(logits, tau=self.temperature, hard=True)
        
        # Use the one-hot vector for the embedding lookup
        z_q = self.embedding(y_soft)
        
        return z_q
# --- The NEW Multi-Hot STE Quantizer ---
class MultiHotSTEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, linear_weight, k):
        # Forward pass is identical to the other multi-hot models
        # This is complex because k can be a tensor. We use the same masking trick.
        batch_size, quant_dim = x.shape
        k_values = torch.full((batch_size, 1), k, device=x.device)
        
        sorted_indices = torch.argsort(x, dim=-1, descending=True)
        k_range = torch.arange(quant_dim, device=x.device)[None, :].expand(batch_size, -1)
        k_hot_mask = (k_range < k_values).float()
        k_hot = k_hot_mask.gather(-1, torch.argsort(sorted_indices, dim=-1))
        
        ctx.save_for_backward(linear_weight, k_hot)

        output = F.linear(k_hot, linear_weight)
        return output

    @staticmethod
    def backward(ctx, g):
        linear_weight, k_hot = ctx.saved_tensors

        # --- Part 1: Gradient for the linear layer's weight (Standard) ---
        grad_linear_weight = g.T @ k_hot

        # --- Part 2: Gradient for the input x (STE Logic) ---
        # The gradient is the signal that would have gone to the k_hot vector,
        # passed straight through to the logits x.
        grad_x = g @ linear_weight
        
        # Must return a grad for each input (x, linear_weight, k). Grad for k is None.
        return grad_x, grad_linear_weight, None

class MultiHotSTEQuantizer(nn.Module):
    def __init__(self, quant_dim, embed_dim, k=15):
        super().__init__()
        self.k = k
        self.linear = nn.Linear(quant_dim, embed_dim, bias=False)

    def forward(self, x):
        return MultiHotSTEFunction.apply(x, self.linear.weight, self.k)

class STEQuantizerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, linear_weight):
        _, indices = torch.topk(x, k=1, dim=-1)
        one_hot = F.one_hot(indices.squeeze(-1), num_classes=x.shape[-1]).float()
        ctx.save_for_backward(x, linear_weight, one_hot)
        output = F.linear(one_hot, linear_weight)
        return output

    @staticmethod
    def backward(ctx, g):
        x, linear_weight, one_hot = ctx.saved_tensors

        # --- Part 1: Gradient for the linear layer's weight (Standard) ---
        grad_linear_weight = g.T @ one_hot

        # --- Part 2: Gradient for the input x (STE Logic) ---
        # The STE simply passes the gradient that would have gone to the one_hot
        # vector straight through to x. This provides a direct, albeit approximate,
        # signal for the reconstruction loss.
        grad_x = g @ linear_weight

        return grad_x, grad_linear_weight

class STEQuantizer(nn.Module):
    def __init__(self, quant_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(quant_dim, output_dim, bias=False)

    def forward(self, x):
        return STEQuantizerFunction.apply(x, self.linear.weight)

class CustomGradientQuantizerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, linear_weight):
        _, indices = torch.topk(x, k=1, dim=-1)

        one_hot = F.one_hot(indices.squeeze(-1), num_classes=x.shape[-1]).float()

        ctx.save_for_backward(x, linear_weight, one_hot)

        output = F.linear(one_hot, linear_weight)
        return output

    @staticmethod
    def backward(ctx, g):
        x, linear_weight, one_hot = ctx.saved_tensors

        grad_linear_weight = g.T @ one_hot 

        gl = g @ linear_weight

        _, target_indices = torch.topk(-gl, k=1, dim=-1)

        target_one_hot = F.one_hot(target_indices.squeeze(-1), num_classes=x.shape[-1]).float()
        
        # 4. Calculate the gradient for x. This is the mathematical gradient of
        #    CrossEntropy(x, target), which is (softmax(x) - target_one_hot).
        #    This is the core of your idea, substituted in place of the real gradient.
        grad_x = F.softmax(x, dim=-1) - target_one_hot

        return grad_x, grad_linear_weight


class CustomQuantizer(nn.Module):
    def __init__(self, quant_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(quant_dim, output_dim, bias=False)

    def forward(self, x):
        return CustomGradientQuantizerFunction.apply(x, self.linear.weight)



# --- The NEW Hybrid Gradient Quantizer ---
class HybridQuantizerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, linear_weight, beta):
        _, indices = torch.topk(x, k=1, dim=-1)
        one_hot = F.one_hot(indices.squeeze(-1), num_classes=x.shape[-1]).float()
        ctx.save_for_backward(x, linear_weight, one_hot)
        ctx.beta = beta # Save the scalar beta
        output = F.linear(one_hot, linear_weight)
        return output

    @staticmethod
    def backward(ctx, g):
        x, linear_weight, one_hot = ctx.saved_tensors
        beta = ctx.beta

        # Part 1: Gradient for the linear layer's weight (Standard)
        grad_linear_weight = g.T @ one_hot

        # Part 2: Gradient for the input x (The Hybrid Logic)
        # 2a: The STE gradient (task-specific signal)
        ste_grad = g @ linear_weight
        
        # 2b: Your custom gradient (regularization signal)
        # Note: We use x.detach() because this part should only regularize,
        # not be influenced by the STE gradient path itself.
        with torch.no_grad():
            target_indices = torch.topk(-ste_grad, k=1, dim=-1)[1]
            target_one_hot = F.one_hot(target_indices.squeeze(-1), num_classes=x.shape[-1]).float()
        
        custom_grad_regularizer = F.softmax(x.detach(), dim=-1) - target_one_hot

        # 2c: Combine them!
        grad_x = ste_grad + beta * custom_grad_regularizer
        
        # Must return a grad for each input (x, linear_weight, beta). Grad for beta is None.
        return grad_x, grad_linear_weight, None

class DynamicKQuantizerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, main_linear_weight, k_decider_weight):
        # ... (full implementation from the previous answer) ...
        kd = F.linear(logits, k_decider_weight)
        _, k_indices = torch.topk(kd, k=1, dim=-1)
        k_values = k_indices + 1
        batch_size, quant_dim = logits.shape
        sorted_indices = torch.argsort(logits, dim=-1, descending=True)
        k_range = torch.arange(quant_dim, device=logits.device)[None, :].expand(batch_size, -1)
        k_hot_mask = (k_range < k_values).float()
        k_hot = k_hot_mask.gather(-1, torch.argsort(sorted_indices, dim=-1))
        ctx.save_for_backward(logits, main_linear_weight, k_decider_weight, k_hot, kd, k_values)
        output = F.linear(k_hot, main_linear_weight)
        return output

    @staticmethod
    def backward(ctx, g):
        logits, main_linear_weight, k_decider_weight, k_hot_from_fwd, kd, k_values = ctx.saved_tensors
        grad_main_linear_weight = g.T @ k_hot_from_fwd
        gl = g @ main_linear_weight
        with torch.no_grad():
            batch_size = logits.shape[0]
            max_k = kd.shape[-1]
            kfrom = (k_values - 1).clamp(min=1)
            k_probes = torch.cat([kfrom, kfrom + 1, (kfrom + 2).clamp(max=max_k)], dim=1)
            variances = torch.zeros((batch_size, 3), device=logits.device)
            top_vals_gl, _ = torch.topk(-gl, k=max_k, dim=-1)
            for i in range(3):
                current_k_probe = k_probes[:, i]
                mask = torch.arange(max_k, device=logits.device)[None, :] < current_k_probe[:, None]
                masked_vals = torch.where(mask, top_vals_gl, torch.tensor(-float('inf'), device=logits.device))
                soft_weights = F.softmax(masked_vals, dim=-1)
                variances[:, i] = soft_weights.var(dim=-1)
            best_probe_indices = torch.argmin(variances, dim=-1)
            target_k = k_probes.gather(-1, best_probe_indices.unsqueeze(-1)).squeeze(-1)
            target_k_indices = target_k - 1
            target_k_hot = F.one_hot(target_k_indices, num_classes=max_k).float()
        grad_kd = F.softmax(kd, dim=-1) - target_k_hot
        grad_k_decider_weight = grad_kd.T @ logits
        _, target_indices = torch.topk(-gl, k=1, dim=-1)
        target_one_hot = F.one_hot(target_indices.squeeze(-1), num_classes=logits.shape[-1]).float()
        grad_logits = F.softmax(logits, dim=-1) - target_one_hot
        return grad_logits, grad_main_linear_weight, grad_k_decider_weight

class AdaKQuantizer(nn.Module):
    def __init__(self, quant_dim, embed_dim, max_k):
        super().__init__()
        self.codebook = nn.Linear(quant_dim, embed_dim, bias=False)
        self.k_decider = nn.Linear(quant_dim, max_k, bias=False)

    def forward(self, x):
        return DynamicKQuantizerFunction.apply(x, self.codebook.weight, self.k_decider.weight) 

class SoftTargetQuantizerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, linear_weight, k):
        _, indices = torch.topk(x, k=k, dim=-1)
        # In this model, the forward pass is a hard k-hot lookup
        k_hot = torch.zeros_like(x).scatter_(-1, indices, 1.0)
        
        ctx.save_for_backward(x, linear_weight, k_hot)
        ctx.k = k

        # The output is still the sum of the k chosen vectors
        output = F.linear(k_hot, linear_weight)
        return output

    @staticmethod
    def backward(ctx, g):
        x, linear_weight, k_hot_from_fwd = ctx.saved_tensors
        k = ctx.k

        # --- Part 1: Standard gradient for the linear layer's weight ---
        grad_linear_weight = g.T @ k_hot_from_fwd

        # --- Part 2: Your custom "Soft-Target" gradient for the input x ---
        # 2a: Calculate the correction signal
        gl = g @ linear_weight

        with torch.no_grad():
            # 2b: Find the top k values and indices of the desired correction
            topk_vals, topk_indices = torch.topk(-gl, k=k, dim=-1)

            # 2c: THE CORE INNOVATION: Softmax within the k-hot window
            soft_weights = F.softmax(topk_vals, dim=-1)

            # 2d: Scatter these soft weights back to create the full target vector
            soft_target = torch.zeros_like(x).scatter_(-1, topk_indices, soft_weights)

        # 2e: The new gradient is a valid comparison of two probability distributions
        grad_x = F.softmax(x, dim=-1) - soft_target
        
        # Must return a grad for each input (x, linear_weight, k)
        return grad_x, grad_linear_weight, None

class NKQuantizer(nn.Module):
    def __init__(self, quant_dim, embed_dim, k):
        super().__init__()
        self.codebook = nn.Linear(quant_dim, embed_dim, bias=False)
        self.k=k

    def forward(self, x):
        return SoftTargetQuantizerFunction.apply(x, self.codebook.weight, self.k) 


class GatherByGateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate_logits, candidate_pool, k):
        _, top_indices = torch.topk(gate_logits, k=k, dim=-1)

        output = torch.gather(candidate_pool, -1, top_indices)

        ctx.save_for_backward(gate_logits, candidate_pool, top_indices)
        return output

    @staticmethod
    def backward(ctx, g):
        gate_logits, candidate_pool, top_indices = ctx.saved_tensors

        grad_candidate_pool = torch.zeros_like(candidate_pool)
        grad_candidate_pool.scatter_add_(-1, top_indices, g)

        with torch.no_grad():
            y = torch.gather(candidate_pool, -1, top_indices)
            
            ideal_targets = y - g

            distances = torch.cdist(ideal_targets.unsqueeze(-2), candidate_pool.unsqueeze(-2))

            _, ideal_indices = torch.min(distances, dim=-1)
            
            soft_target = torch.zeros_like(gate_logits)
            #replace by softmax of -g scattered. instead of clamp
            soft_target.scatter_add_(-1, ideal_indices, torch.ones_like(ideal_indices, dtype=gate_logits.dtype))

            soft_target.clamp_(0, 1)

        grad_gate_logits = F.softmax(gate_logits, dim=-1) - soft_target
        
        return grad_gate_logits, grad_candidate_pool, None
    

class TopKHot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x,  k):
        _, indices = torch.topk(x, k=k, dim=-1)
        k_hot = torch.zeros_like(x).scatter_(-1, indices, 1.0)
        ctx.save_for_backward(x, k)
        return k_hot

    @staticmethod
    def backward(ctx, g):
        x, k = ctx.saved_tensors

        with torch.no_grad():
            topk_vals, topk_indices = torch.topk(-g, k=k, dim=-1)

            soft_weights = F.softmax(topk_vals, dim=-1)

            soft_target = torch.zeros_like(x).scatter_(-1, topk_indices, soft_weights)

        grad_x = F.softmax(x, dim=-1) - soft_target
        
        return grad_x,  None

class TopKHot2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k):
        #_, indices = torch.topk(x, k=k, dim=-1)
        #k_hot = torch.zeros_like(x).scatter_(-1, indices, 1.0)
        
        batch_size, quant_dim = x.shape
        
        sorted_indices = torch.argsort(x, dim=-1, descending=True)
        
        k_range = torch.arange(quant_dim, device=x.device)[None, :].expand(batch_size, -1)
        
        k_hot_mask = (k_range < k).float()
        
        k_hot = k_hot_mask.gather(-1, torch.argsort(sorted_indices, dim=-1))

        ctx.save_for_backward(x)
        ctx.k = k
        ctx.max_k = x.shape[-1]
        return k_hot

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        k = ctx.k
        max_k = ctx.max_k
        
        sorted_g, sorted_g_indices = torch.sort(-g, dim=-1, descending=True)
        # --- PATH 1: Gradient for the input logits 'x' (Soft-Target Logic) ---
        with torch.no_grad():
            k_range = torch.arange(x.shape[-1], device=x.device)[None, :]
            k_mask = (k_range < k).float()
            
            # Apply the dynamic mask to the sorted correction signal
            masked_topk_gl_vals = torch.where(k_mask > 0, sorted_g, torch.tensor(-float('inf'), device=x.device))
            soft_weights = F.softmax(masked_topk_gl_vals, dim=-1)
            soft_target = soft_weights.gather(-1, torch.argsort(sorted_g_indices, dim=-1))
        grad_x = F.softmax(x, dim=-1) - soft_target
        
        # --- PATH 2: Surrogate Gradient for 'k_continuous' (Your Vectorized Logic) ---
        with torch.no_grad():
            batch_size = g.shape[0]
            
            # Define the probe window in a vectorized way
            kfrom = (k - 1).clamp(min=1, max=max_k - 2)
            # Shape: (1, 3) -> will broadcast to (batch_size, 3)
            k_probes = torch.cat([kfrom, kfrom + 1, kfrom + 2], dim=0).unsqueeze(0)

            variances = torch.zeros((batch_size, 3), device=g.device)
            means = torch.zeros((batch_size, 3), device=g.device)
            # This loop is now over a fixed size of 3, not a Python list
            for i in range(3):
                current_k_probe = k_probes[:, i] # Shape: (batch_size)
                mask = torch.arange(max_k, device=g.device)[None, :] < current_k_probe[:, None]
                masked_vals = torch.where(mask, sorted_g, torch.tensor(-float('inf'), device=g.device))
                soft_weights = F.softmax(masked_vals, dim=-1)
                variances[:, i] = soft_weights.var(dim=-1)
                means[:, i] = soft_weights.mean(dim=-1)

            # Find which probe had the minimum variance for each item in the batch
            best_probe_indices = torch.argmin(((0.5-means)**2)/(2*variances), dim=-1) # Shape: (batch_size)
            
            # Create the gradient signal based on the vote of the batch
            # -1 if k-1 was best, 0 if k was best, +1 if k+1 was best
            grad_signals = best_probe_indices.float() - 1.0
            
            grad_k_continuous = grad_signals#.mean(dim=-1)

        # Return gradients for each input: (x, k_continuous)
        return grad_x, grad_k_continuous

class dynKHot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k_scores):

        k_indices = torch.argmax(k_scores, dim=-1, keepdim=True) + 1
        k_values = k_indices  

        sorted_indices = torch.argsort(x, dim=-1, descending=True)
        quant_dim = x.shape[-1]
        k_range = torch.arange(quant_dim, device=x.device).expand(x.shape[0], -1)
        k_hot_mask = (k_range < k_values).float()
        k_hot = k_hot_mask.gather(-1, torch.argsort(sorted_indices, dim=-1))

        ctx.save_for_backward(x, k_scores)
        ctx.k_values = k_values 
        return k_hot

    @staticmethod
    def backward(ctx, g):
        x, k_scores = ctx.saved_tensors
        k_values = ctx.k_values
        batch_size, qdim = x.shape
        
        with torch.no_grad():
            sorted_g, sorted_g_indices = torch.sort(-g, dim=-1, descending=True)
            k_range = torch.arange(qdim, device=x.device).expand(batch_size, -1)
            k_mask = (k_range < k_values).float()
            
            masked_vals = torch.where(k_mask > 0, sorted_g, torch.tensor(-float('inf'), device=x.device))
            soft_weights = F.softmax(masked_vals, dim=-1)
            soft_target_x = soft_weights.gather(-1, torch.argsort(sorted_g_indices, dim=-1))
            
        grad_x = F.softmax(x, dim=-1) - soft_target_x

        with torch.no_grad():
            g_avg = g.mean(dim=-1, keepdim=True)
            g_std = g.std(dim=-1, keepdim=True)
            epsilon = 0.1*g_std 
            wanted_K = torch.sum(g < g_avg , dim=-1, keepdim=True)
            #num_yes_votes = torch.sum(g < (g_avg - epsilon), dim=-1, keepdim=True)
            #num_no_votes = torch.sum(g > (g_avg + epsilon), dim=-1, keepdim=True)

            #nudge = (num_yes_votes-num_no_votes)#/(num_yes_votes+num_no_votes)

            # 3. Calculate the new target k as a float
            target_k_float = torch.round(wanted_K).long().clamp(1, qdim) - 1
            
            # 4. --- YOUR GAUSSIAN KERNEL TARGET ---
            # Create the Gaussian "hump" centered at the float target.
            #k_indices = torch.arange(qdim, device=x.device, dtype=torch.float32)
            
            # The standard deviation of the target kernel is a new hyperparameter.
            # A small value (e.g., 0.5) creates a sharp, confident target.
            #target_std = 0.5
            
            # Unsqueeze for broadcasting
            #gaussian_target = torch.exp(-0.5 * ((k_indices - target_k_float) / target_std) ** 2)
            
            # Normalize to make it a valid probability distribution
            #soft_target_k = gaussian_target / gaussian_target.sum(dim=-1, keepdim=True)
            k_target = F.one_hot(target_k_float, qdim)

        grad_k_scores = F.softmax(k_scores, dim=-1) - k_target
        
        return grad_x, grad_k_scores

class dynFKHot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k_values):

        quant_dim = x.shape[-1]
        k_values = k_values.clamp(1, quant_dim)
        sorted_indices = torch.argsort(x, dim=-1, descending=True)
        k_range = torch.arange(quant_dim, device=x.device).expand(x.shape[0], -1)
        k_hot_mask = (k_range < k_values).float()
        k_hot = k_hot_mask.gather(-1, torch.argsort(sorted_indices, dim=-1))

        ctx.save_for_backward(x, k_values)
        return k_hot

    @staticmethod
    def backward(ctx, g):
        x, k_values = ctx.saved_tensors
        #k_values = ctx.k_values
        batch_size, qdim = x.shape
        
        with torch.no_grad():
            sorted_g, sorted_g_indices = torch.sort(-g, dim=-1, descending=True)
            k_range = torch.arange(qdim, device=x.device).expand(batch_size, -1)
            k_mask = (k_range < k_values).float()
            
            masked_vals = torch.where(k_mask > 0, sorted_g, torch.tensor(-float('inf'), device=x.device))
            soft_weights = F.softmax(masked_vals, dim=-1)
            soft_target_x = soft_weights.gather(-1, torch.argsort(sorted_g_indices, dim=-1))
            
        grad_x = F.softmax(x, dim=-1) - soft_target_x

        with torch.no_grad():
            g_avg = g.mean(dim=-1, keepdim=True)
            wanted_K = torch.sum(g < g_avg , dim=-1, keepdim=True)
            target_k_float = torch.round(wanted_K).long().clamp(1, qdim) - 1
            
        grad_k_values = k_values - target_k_float
        
        return grad_x, grad_k_values

class NKQuantizer2(nn.Module):
    def __init__(self, quant_dim, embed_dim, k):
        super().__init__()
        self.codebook = nn.Linear(quant_dim, embed_dim, bias=False)
        self.k=k

    def forward(self, x):
        k_hot = TopKHot.apply(x, self.k)
        return self.codebook(k_hot)
        
class DynKQuantizer2(nn.Module):
    def __init__(self, quant_dim, embed_dim, max_k):
        super().__init__()
        self.codebook = nn.Linear(quant_dim, embed_dim, bias=False)
        self.dynkselector = nn.Sequential(
            nn.Linear(quant_dim, quant_dim*2, bias=False),
            nn.ReLU(),
            nn.Linear(quant_dim*2, quant_dim, bias=False),
            nn.ReLU(),
            nn.Linear(quant_dim, quant_dim, bias=False),
        )
        self.mk = max_k
        
    def forward(self, x):
        k = self.dynkselector(x)
        k_hot = dynKHot.apply(x, k) 
        return self.codebook(k_hot)

def xor(x,y):
    return x * (1 - y) + (1 - x) * y


# --- The Autoencoder and Experiment Setup ---
class Autoencoder(nn.Module):
    def __init__(self,encoder, quantizer, decoder ):
        super().__init__()
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder

    def forward(self, x):
        logits = self.encoder(x)
        vq_loss = 0
        
        if isinstance(self.quantizer, MultiHotVQVAEQuantizer):
            quantized, vq_loss = self.quantizer(logits)
            logits = None
        else:
            quantized = self.quantizer(logits)
            
        reconstruction = self.decoder(quantized)
        return reconstruction, logits, vq_loss 
    
    
class NKQAE(nn.Module):
    def __init__(self, input_dim, n_embd, n_hdim, qdim, K = None):
        super().__init__()
        self.qdim = qdim
        self.k = qdim//2 if K is None else K
        self.encoder = nn.Sequential(nn.Linear(input_dim, n_hdim), nn.ReLU(), nn.Linear(n_hdim, self.qdim ))
        self.codebook = nn.Linear(self.qdim, n_embd, bias=False)
        self.decoder = nn.Sequential(nn.Linear(n_embd, n_hdim), nn.ReLU(), nn.Linear(n_hdim, input_dim))

    def forward(self, x):
        logits = self.encoder(x)
        khot = TopKHot.apply(logits, self.qdim//2)
        quantized = self.codebook(khot)
        reconstruction = self.decoder(quantized)
        return reconstruction, logits, 0.0

    def quant(self, x):
        logits = self.encoder(x)
        return TopKHot.apply(logits, self.qdim//2)
    
    def dequant(self, k_hot):
        q = self.codebook(k_hot)
        return self.decoder(q)


class DynFKhot(nn.Module):
    def __init__(self, input_dim, n_hdim, qdim):
        super().__init__()
        self.qdim = qdim
        self.k_predictor = nn.Sequential(
            nn.Linear(input_dim+qdim, n_hdim), 
            nn.ReLU(), 
            nn.Linear(n_hdim, n_hdim), 
            nn.ReLU(), 
            nn.Linear(n_hdim, 1),
            nn.Sigmoid(),
        )
        self.k_scale = nn.Parameter(torch.ones(1))
        self.encoder = nn.Sequential(nn.Linear(input_dim, n_hdim), nn.ReLU(), nn.Linear(n_hdim, self.qdim ))
        
    def forward(self, x):
        logits = self.encoder(x)
        k = self.k_predictor(torch.cat((x,logits.detach()),dim=-1)) * self.qdim 
        k = (k * F.sigmoid(self.k_scale) * 2 ).clamp(1, self.qdim) 
        khot = dynFKHot.apply(logits, k)
        return khot, k
    

class DynKQAE2(nn.Module):
    def __init__(self, input_dim, n_embd, n_hdim, qdim):
        super().__init__()
        self.qdim = qdim
        self.encoder = nn.Sequential(nn.Linear(input_dim, n_hdim), nn.ReLU(), nn.Linear(n_hdim, self.qdim ))
        self.codebook = nn.Linear(self.qdim, n_embd, bias=False)
        self.decoder = nn.Sequential(nn.Linear(n_embd, n_hdim), nn.ReLU(), nn.Linear(n_hdim, input_dim))
        self.k_predictor = nn.Sequential(
            nn.Linear(input_dim+qdim, n_hdim), 
            nn.ReLU(), 
            nn.Linear(n_hdim, n_hdim), 
            nn.ReLU(), 
            nn.Linear(n_hdim, 1),
            #nn.Tanh(),
            #nn.LayerNorm(qdim),
            nn.Sigmoid(),
        )
        
        self.k_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        khot, k = self.quant(x)

        reconstruction = self.dequant(khot, k)
        
        return reconstruction, khot, 0.0, k

    def quant(self, x):
        logits = self.encoder(x)
        k = self.k_predictor(torch.cat((x,logits.detach()),dim=-1)) * self.qdim #+self.k_bias#-self.k_penalty
        k = (k * F.sigmoid(self.k_scale)* 2 ).clamp(1, self.qdim) 
        khot = dynFKHot.apply(logits, k)
        return khot, k
    
    def dequant(self, k_hot, k):
        k_hot = k_hot / (k.detach())
        q = self.codebook(k_hot)
        return self.decoder(q)
    
    def optimizer(self, LR):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        ae_params = []
        k_selector_params = []

        for name, param in param_dict.items():
            if name.startswith('k_predictor') or name.startswith('k_bias'):
                k_selector_params.append(param)
            else:
                ae_params.append(param)
        num_ae = sum(p.numel() for p in ae_params)
        num_k_sel = sum(p.numel() for p in k_selector_params)
        print(f"ae parameter tensors: {len(ae_params)}, with {num_ae:,} parameters")
        print(f"k selector parameter tensors: {len(k_selector_params)}, with {num_k_sel:,} parameters")

        optim_groups = [
            {'params': ae_params, 'lr': LR},
            {
                'params': k_selector_params,
                'lr': LR ,#* 0.01,
                #'weight_decay': 0.0,  # CRITICAL: No decay for the bias
                #'betas': (0.0, 0.999),
            }
        ]
        return torch.optim.AdamW(optim_groups, lr=LEARNING_RATE) 
    
class DynKQAE(nn.Module):
    def __init__(self, input_dim, n_embd, n_hdim, qdim, voters, k_per):
        super().__init__()
        self.qdim = qdim
        self.voters = voters
        self.k_per = k_per
        self.encoders = nn.ModuleList([ 
                nn.Sequential(
                    nn.Linear(input_dim, n_hdim), 
                    nn.ReLU(), 
                    nn.Linear(n_hdim, self.qdim ),
                ) for _ in range(voters)
            ])
        self.codebook = nn.Linear(self.qdim, n_embd, bias=False)
        self.decoder = nn.Sequential(nn.Linear(n_embd, n_hdim), nn.ReLU(), nn.Linear(n_hdim, input_dim))

    def forward(self, x):
        khot, kr= self.quant(x)
        reconstruction = self.dequant(khot)
        return reconstruction, khot, 0.0#F.mse_loss(khot,kr)

    def quant(self, x):
        all_logits = [encoder(x) for encoder in self.encoders]
        
        stacked_logits = torch.stack(all_logits, dim=1)
        
        all_votes = TopKHot.apply(stacked_logits, self.k_per)
        
        khot_raw = torch.sum(all_votes, dim=1)
        #khot_xor = 0.5 * (1 - torch.cos(torch.pi * khot_raw))
        khot = khot_raw.clamp(0,1)
        return khot, khot_raw
    
    def dequant(self, k_hot):
        q = self.codebook(k_hot)
        return self.decoder(q)
    
def calculate_perplexity(logits):
    # Calculates the perplexity of the codebook usage for a batch
    probs = F.softmax(logits, dim=-1).mean(dim=0)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10))
    perplexity = torch.exp(entropy)
    return perplexity.mean().detach().item()

@torch.no_grad()
def run_validation(model, val_loader, device, num_batches=10):
    model.eval()
    total_loss = 0
    total_perplexity = 0
    total_k_perplexity = 0
    total_k = 0
    batches_processed = 1
    for i, (images, _) in enumerate(val_loader):
        if i >= num_batches:
            break
        images = images.view(-1, INPUT_DIM).to(device)
        recon, logits, _, k = model(images)
        total_loss += F.mse_loss(recon, images).mean().detach().item()
        if(logits is not None):
            total_perplexity += calculate_perplexity(logits)
        total_k_perplexity += k.var().detach().item()
        #k_indices = torch.argmax(k, dim=-1) + 1
        total_k += k.mean().detach().item()
        batches_processed += 1
    bp = batches_processed 
    avg_k = total_k / bp
    avg_loss = total_loss / bp
    avg_perplexity = total_perplexity / bp
    avg_k_perplexity = total_k_perplexity / bp
    model.train()
    return avg_loss, avg_perplexity, avg_k, avg_k_perplexity


wn = torch.nn.utils.parametrizations.weight_norm
class wnormlearnloss(nn.Module):
    def __init__(self, dims):
        super().__init__()
        # Create one learnable log_sigma_sq parameter for each loss
        self.head = nn.Sequential(
                wn(nn.Linear(dims, dims*2),dim=None),
                nn.GELU(),
                wn(nn.Linear(dims*2, dims*2),dim=None),
                nn.GELU(),
                wn(nn.Linear(dims*2, 1))
            )

    def forward(self, x):
        loss = self.head(x)
        return utils.minmaxnorm(loss).mean()

if __name__ == '__main__':
    # Hyperparameters
    INPUT_DIM, HIDDEN_DIM, QUANT_DIM, EMBED_DIM = 28*28, 256, 64, 32
    BATCH_SIZE, LEARNING_RATE, STEPS = 256, 1e-3, 10001
    
    VAL_LOG_INTERVAL = 1000
    TRAIN_LOG_INTERVAL = 100
    VALIDATION_STEPS = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- NEW: Data Splitting ---
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    full_train_dataset = torchvision.datasets.MNIST("./", train=True, download=True, transform=transform)
    train_size = 50000
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=4, pin_memory=True)
    data_iterator = iter(train_loader)

    # Base models
    encoder_for_vqvae = nn.Sequential(nn.Linear(INPUT_DIM, HIDDEN_DIM), nn.ReLU(),nn.Linear(HIDDEN_DIM, EMBED_DIM))
    encoder_base = nn.Sequential(nn.Linear(INPUT_DIM, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, QUANT_DIM))
    decoder_base = nn.Sequential(nn.Linear(EMBED_DIM, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, INPUT_DIM))
    # Instantiate the three models
    models = {
        #"STE": Autoencoder(copy.deepcopy(encoder_base), STEQuantizer(QUANT_DIM, EMBED_DIM), copy.deepcopy(decoder_base)),
        #"Hybrid": Autoencoder(copy.deepcopy(encoder_base), HybridQuantizer(QUANT_DIM, EMBED_DIM, beta=BETA), copy.deepcopy(decoder_base)),
        #"ADAK": Autoencoder(copy.deepcopy(encoder_base), AdaKQuantizer(QUANT_DIM, EMBED_DIM, QUANT_DIM), copy.deepcopy(decoder_base)),
        #"ADAK 15": Autoencoder(copy.deepcopy(encoder_base), AdaKQuantizer(QUANT_DIM, EMBED_DIM, 15), copy.deepcopy(decoder_base)),
        #"VQ-VAE 32": Autoencoder(copy.deepcopy(encoder_for_vqvae), MultiHotVQVAEQuantizer(QUANT_DIM, EMBED_DIM,  QUANT_DIM//2, 0.25), copy.deepcopy(decoder_base)),
        #"VQ-VAE":    Autoencoder(copy.deepcopy(encoder_for_vqvae), MultiHotVQVAEQuantizer(QUANT_DIM, EMBED_DIM, 1, 0.25), copy.deepcopy(decoder_base)),
        #"Gumbell":   Autoencoder(copy.deepcopy(encoder_base),      GumbelQuantizer(QUANT_DIM, EMBED_DIM, 1), copy.deepcopy(decoder_base)),
        #"CER":       Autoencoder(copy.deepcopy(encoder_base),      CustomQuantizer(QUANT_DIM, EMBED_DIM), copy.deepcopy(decoder_base)),
        #"STE":       Autoencoder(copy.deepcopy(encoder_base),      STEQuantizer(QUANT_DIM, EMBED_DIM), copy.deepcopy(decoder_base)),
        #"CERk 32":   Autoencoder(copy.deepcopy(encoder_base),      NKQuantizer(QUANT_DIM, EMBED_DIM, QUANT_DIM//2), copy.deepcopy(decoder_base)),
        #"CER":       Autoencoder(copy.deepcopy(encoder_base),      CustomQuantizer(QUANT_DIM, EMBED_DIM), copy.deepcopy(decoder_base)),
        #"adak":   Autoencoder(copy.deepcopy(encoder_base),      AdaKQuantizer(QUANT_DIM, EMBED_DIM,QUANT_DIM), copy.deepcopy(decoder_base)),
        #"dynk3":   Autoencoder(copy.deepcopy(encoder_base),      DynKQuantizer2(QUANT_DIM, EMBED_DIM, QUANT_DIM), copy.deepcopy(decoder_base)),
        #"cdynk":   Autoencoder(copy.deepcopy(encoder_base),      CER_DynamicK_Quantizer(QUANT_DIM, EMBED_DIM, QUANT_DIM), copy.deepcopy(decoder_base)),
        "dynk3": DynKQAE2(INPUT_DIM, EMBED_DIM, HIDDEN_DIM, QUANT_DIM),
        #"CERk 32":      NKQAE(INPUT_DIM, EMBED_DIM, HIDDEN_DIM, QUANT_DIM,  32),
        
        #"CER 16": Autoencoder(copy.deepcopy(encoder_base), NKQuantizer(QUANT_DIM, EMBED_DIM, 16), copy.deepcopy(decoder_base)),
        #"STE 16": Autoencoder(copy.deepcopy(encoder_base), MultiHotSTEQuantizer(QUANT_DIM, EMBED_DIM, 16), copy.deepcopy(decoder_base)),
        #"STE qd0.5": Autoencoder(copy.deepcopy(encoder_base), MultiHotSTEQuantizer(QUANT_DIM, EMBED_DIM, QUANT_DIM//2), copy.deepcopy(decoder_base)),
    }
    
    print("Compiling models...")
    torch.set_float32_matmul_precision('medium')
    models = {name: torch.compile(model.to(device)) for name, model in models.items()}
    auxl = torch.compile(wnormlearnloss(INPUT_DIM).to(device))
    print("Compilation complete.")
    
    optimizers = {name: model.optimizer(LEARNING_RATE) for name, model in models.items()}
    #optimizers = {name: torch.optim.Adam(model.k_predictor.parameters(), lr=LEARNING_RATE*0.001) for name, model in models.items()}
    # --- NEW: Expanded Metrics Tracking ---
    metrics = {
        name: {
            "train_loss": [], 
            "train_perp": [], 
            "train_k_var": [], 
            "train_k_mean": [], 
            "val_loss": [], 
            "val_perp": [], 
            "val_k_var": [], 
            "val_k_mean": [], 
            } for name in models.keys()}

    print("Starting training with validation...")
    for step in range(STEPS):
        try:
            images, _ = next(data_iterator)
        except StopIteration:
            data_iterator = iter(train_loader)
            images, _ = next(data_iterator)
        images = images.view(-1, INPUT_DIM).to(device)

        # Training Step
        for name, model in models.items():
            optimizers[name].zero_grad(set_to_none=True)
            if isinstance(model, DynKQAE2):
                recon, logits, aux, k = model(images)
            else:
                recon, logits, aux = model(images)
                k = model.qdim//2

            loss_raw = F.mse_loss(recon, images) 
            loss = loss_raw + aux
            loss = loss + auxl(recon)#*0.1
            #r2, rq, _, k2 = model(recon.detach())
            #loss = loss + F.mse_loss(F.softmax(recon.detach(),dim=-1), F.softmax(r2,dim=-1)) # The target distribution (from the first pass)

            #target_dist = F.softmax(logits.detach() / k, dim=-1)
            #input_dist_log = F.log_softmax(rq / k2, dim=-1)
            #loss = loss + F.kl_div(input_dist_log, target_dist, reduction="batchmean")
            #loss = loss + F.kl_div(F.softmax(logits.detach()/k.detach(),dim=-1), F.softmax(rq/k2,dim=-1), reduction="batchmean") 
            loss.backward()
            #lw = F.softmax(recon)-F.softmax(images)
            #loss.backward()
            optimizers[name].step()
            
            if step % 100 == 0:
                metrics[name]["train_loss"].append(loss_raw.detach().cpu().item())
                metrics[name]["train_k_var"].append(k.var().detach().cpu().item())
                metrics[name]["train_k_mean"].append(k.mean().detach().cpu().item())
                #metrics[name]["k_value"].append(k.detach().item())
                perp = calculate_perplexity(logits.detach()) if logits is not None else 0.0
                metrics[name]["train_perp"].append(perp)

        if step > 0 and step % VAL_LOG_INTERVAL == 0:
            print(f"--- Step {step:5d} ---")
            for name, model in models.items():
                val_loss, val_perp, kavg, kpavg = run_validation(model, val_loader, device, VALIDATION_STEPS)
                metrics[name]["val_loss"].append(val_loss)
                metrics[name]["val_perp"].append(val_perp)
                metrics[name]["val_k_var"].append(kpavg)
                metrics[name]["val_k_mean"].append(kavg)
                print(f"{name:>10} | Train Loss: {metrics[name]['train_loss'][-1]:.6f} | Val Loss: {val_loss:.6f} | Val Perp: {val_perp:.2f}| Val k: {kavg:.2f}| Val k var: {kpavg:.2f}")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    for name in models.keys():
        # --- Generate Correct X-Axes Based on Logged Data Length ---
        # This is more robust than assuming the training runs to full STEPS
        num_train_logs = len(metrics[name]["train_loss"])
        train_x_axis = np.arange(num_train_logs) * TRAIN_LOG_INTERVAL

        num_val_logs = len(metrics[name]["val_loss"])
        # The first validation happens at VAL_LOG_INTERVAL, not 0
        val_x_axis = (np.arange(num_val_logs) + 1) * VAL_LOG_INTERVAL

        #train_steps_x_axis = range(0, STEPS, TRAIN_LOG_INTERVAL) 
        #val_steps_x_axis = range(0, STEPS, VAL_LOG_INTERVAL) 

        # Plot training loss
        if num_train_logs > 0:
            ax1.plot(train_x_axis, metrics[name]["train_loss"], label=f'{name} Train Loss',       alpha=0.5)
            ax2.plot(train_x_axis, metrics[name]["train_perp"], label=f'{name} Train Perplexity', alpha=0.5)

            train_k_mean = np.array(metrics[name]["train_k_mean"])
            train_k_var = np.array(metrics[name]["train_k_var"])
            train_k_std = np.sqrt(train_k_var) 

            ax3.plot(train_x_axis, train_k_mean, label=f'{name} Train K', alpha=0.5)
            ax3.fill_between(train_x_axis, train_k_mean - train_k_std, train_k_mean + train_k_std, alpha=0.2)
    
        
        if num_val_logs > 0:
            ax1.plot(val_x_axis, metrics[name]["val_loss"], label=f'{name} Val Loss', linestyle='--', marker='o', markersize=4)
            ax2.plot(val_x_axis, metrics[name]["val_perp"], label=f'{name} Val Perplexity', linestyle='--', marker='o', markersize=4)

            val_k_mean = np.array(metrics[name]["val_k_mean"])
            val_k_var = np.array(metrics[name]["val_k_var"])
            val_k_std = np.sqrt(val_k_var)
            ax3.plot(val_x_axis, val_k_mean, label=f'{name} Train K', alpha=0.5)
            ax3.fill_between(val_x_axis, val_k_mean - val_k_std, val_k_mean + val_k_std, alpha=0.2)
    
    ax1.set_title('Training vs. Validation Loss')
    ax1.set_ylabel('Mean Squared Error (MSE) Loss')
    ax1.grid(True, which="both", ls="--")
    ax1.legend()
    ax1.set_yscale('log')
    ax1.set_xlim(0, STEPS)
    
    ax2.set_ylim(0, QUANT_DIM)
    ax2.set_title('Validation Codebook Perplexity')
    ax2.set_ylabel(f'Perplexity (Max = {QUANT_DIM})')
    ax2.set_xlabel('Training Step')
    ax2.grid(True, which="both", ls="--")
    ax2.legend()
    ax2.set_xlim(0, STEPS)

    # --- Formatting for the K Variance Plot (ax3) ---
    ax3.set_title('Variance of K (Codebook Usage)')
    ax3.set_ylabel('Variance')
    ax3.set_xlabel('Training Step')
    ax3.grid(True, which="both", ls="--")
    ax3.legend()
    ax3.set_ylim(bottom=0)

    # Apply shared settings
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(0, STEPS)

    plt.tight_layout()
    plt.show()
