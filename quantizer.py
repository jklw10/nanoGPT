
import copy
import inspect
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

import modules
import utils

class AsymmetricCausalGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, key_scores, k_full, v_full, q_full, k_pool_size):
        _, top_candidate_indices = torch.topk(key_scores, k=min(k_pool_size, k_full.size(2)), dim=-1)

        expanded_indices = top_candidate_indices.unsqueeze(-1).expand(-1, -1, -1, k_full.size(-1))
        
        candidate_k = torch.gather(k_full, dim=2, index=expanded_indices)
        candidate_v = torch.gather(v_full, dim=2, index=expanded_indices)

        ctx.save_for_backward(key_scores, k_full, v_full, q_full, top_candidate_indices)
        
        return candidate_k, candidate_v, top_candidate_indices

    @staticmethod
    def backward(ctx, g_candidate_k, g_candidate_v, g_top_indices): # Grad for indices is None
        key_scores, k_full, v_full, q_full, top_candidate_indices = ctx.saved_tensors
        B, nh, T, hs = k_full.shape
        k_pool_size = top_candidate_indices.shape[-1]

        grad_k_full = torch.zeros_like(k_full)
        grad_v_full = torch.zeros_like(v_full)
        
        expanded_indices = top_candidate_indices.unsqueeze(-1).expand(-1, -1, -1, hs)
        grad_k_full.scatter_add_(2, expanded_indices, g_candidate_k)
        grad_v_full.scatter_add_(2, expanded_indices, g_candidate_v)

        with torch.no_grad():
            ideal_logits = (q_full @ k_full.transpose(-2, -1))

            causal_mask = torch.tril(torch.ones(T, T, device=q_full.device)).view(1, 1, T, T)
            ideal_logits = ideal_logits.masked_fill(causal_mask == 0, float('-inf'))
            
            soft_target = ideal_logits.sum(dim=2) # Summing contributions from all queries

        grad_key_scores = F.softmax(key_scores, dim=-1) - F.softmax(soft_target, dim=-1)
        
        # Return grads for: key_scores, k_full, v_full, q_full, k_pool_size
        return grad_key_scores, grad_k_full, grad_v_full, None, None

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
        return output

    @staticmethod
    def backward(ctx, g):
        gate_logits, candidate_pool, y = ctx.saved_tensors
        k = ctx.k
        grad_gate_logits = grad_candidate_pool = None

        with torch.no_grad():
            #ideal_targets = y - g#scale gradient somehow. normalize both, hyper param, rangenorm
            
            ideal_targets = y + g
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
        
        
        return grad_gate_logits, grad_candidate_pool, None


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
        return output

    @staticmethod
    def backward(ctx, g):
        gate_logits, candidate_pool, y = ctx.saved_tensors
        k = ctx.k
        grad_gate_logits = grad_candidate_pool = None

        with torch.no_grad():
            ideal_targets = y - g#scale gradient somehow. normalize both, hyper param, rangenorm
            
            #ideal_targets = y + g
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
        
        
        return grad_gate_logits, grad_candidate_pool, None


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

class TopKHot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x,  k):
        _, indices = torch.topk(x, k=k, dim=-1)
        k_hot = torch.zeros_like(x).scatter_(-1, indices, 1.0)
        
        ctx.save_for_backward(x)
        ctx.k = k
        return k_hot

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        k = ctx.k

        with torch.no_grad():
            topk_vals, topk_indices = torch.topk(-g, k=k, dim=-1)

            soft_weights = F.softmax(topk_vals, dim=-1)

            soft_target = torch.zeros_like(x).scatter_(-1, topk_indices, soft_weights)

        grad_x = F.softmax(x, dim=-1) - soft_target
        
        return grad_x,  None

class TopKHotBCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x,  k):
        _, indices = torch.topk(x, k=k, dim=-1)
        k_hot = torch.zeros_like(x).scatter_(-1, indices, 1.0)
        ctx.save_for_backward(x)
        ctx.k = k
        return k_hot

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        k = ctx.k
        with torch.no_grad():
            
            topk_vals, topk_indices = torch.topk(-g, k=k, dim=-1)

            soft_weights = F.softmax(topk_vals, dim=-1)

            soft_target = torch.zeros_like(x).scatter_(-1, topk_indices, soft_weights)

        grad_x = F.sigmoid(x) - soft_target
        
        return grad_x,  None
    
class HardTopKHotBCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x,  k):
        _, indices = torch.topk(x, k=k, dim=-1)
        k_hot = torch.zeros_like(x).scatter_(-1, indices, 1.0)
        ctx.save_for_backward(x)
        ctx.k = k
        return k_hot

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        k = ctx.k
        with torch.no_grad():
            _, topk_indices = torch.topk(-g, k=k, dim=-1)
            hard_target = torch.zeros_like(x).scatter_(-1, topk_indices, 1.0)

        grad_x = F.sigmoid(x) - hard_target
        
        return grad_x,  None
    
class ThresHot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        probs = F.softmax(x, dim=-1)
        k_hot = torch.where(probs > (1.0 / x.shape[-1]), 1.0, 0.0)
        ctx.save_for_backward(x)
        return k_hot

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors

        with torch.no_grad():
            g_mean = torch.mean(g, dim=-1, keepdim=True)
            
            soft_target = torch.where(g < g_mean, 1.0, 0.0)

        # Calculate the BCE-style surrogate gradient
        grad_x = F.sigmoid(x) - soft_target
        
        # Optional: Scale by the magnitude of the centered gradient tiny bit worse
        #grad_x = grad_x * torch.abs(g - g_mean)
        
        return grad_x
    #todo test as kolmogorov function selector



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


def b_spline_basis(x, knots, degree):
   
    x = x.unsqueeze(-1)
    n_knots = len(knots)
    n_basis = n_knots - degree - 1

    # Zeroth degree basis functions (piecewise constant)
    b = (x >= knots[:-1]) & (x < knots[1:])
    b = b.to(x.dtype)

    # Cox-de Boor recursion
    for d in range(1, degree + 1):
        
        term1_denom = knots[d:-1] - knots[:-d-1]
        term1_num = x - knots[:-d-1]
        # Avoid division by zero
        term1_denom[term1_denom == 0] = 1e-6
        term1 = (term1_num / term1_denom) * b[:, :, :-1]

        term2_denom = knots[d+1:] - knots[1:-d]
        term2_num = knots[d+1:] - x
        # Avoid division by zero
        term2_denom[term2_denom == 0] = 1e-6
        term2 = (term2_num / term2_denom) * b[:, :, 1:]
        
        b = term1 + term2

    return b


class BSplineActivation(nn.Module):
    def __init__(self, in_features, n_basis=16, degree=8, grid_min=-2.0, grid_max=2.0):
        super().__init__()
        self.in_features = in_features
        self.n_basis = n_basis
        self.degree = degree

        h = (grid_max - grid_min) / (n_basis - 1)
        knots = torch.linspace(grid_min - degree * h, 
                               grid_max + degree * h, 
                               n_basis + degree + 1,
                               dtype=torch.float32)
        self.register_buffer('knots', knots)

        self.coeffs = nn.Parameter(torch.randn(in_features, n_basis))

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)
            
        basis_values = b_spline_basis(x, self.knots, self.degree)

        activation = torch.sum(basis_values * self.coeffs, dim=-1)

        return activation


class TaylorThresHotActivation(nn.Module):
  
    def __init__(self, in_features, n_experts=8, degree=8):
        super().__init__()
        self.in_features = in_features
        self.n_experts = n_experts
        self.degree = degree

        self.expert_weights = nn.Parameter(torch.randn(in_features, n_experts, degree + 1) * 0.1)

        self.expert_centers = nn.Parameter(
            torch.linspace(-2.0, 2.0, n_experts).repeat(in_features, 1)
        )

        self.gating_logits = nn.Parameter(torch.randn(in_features, n_experts))
        
        self.register_buffer('powers', torch.arange(degree + 1, dtype=torch.float32))

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)
        
        gating_mask = ThresHot.apply(self.gating_logits)

        x_reshaped = x.unsqueeze(-1)
        centers = self.expert_centers.unsqueeze(0)
        powers = self.powers.view(1, 1, 1, -1)
        shifted_x = x_reshaped - centers
        
        poly_features = shifted_x.unsqueeze(-1) ** powers
        
        expert_outputs = torch.sum(poly_features * self.expert_weights.unsqueeze(0), dim=-1)

        active_outputs = torch.sum(expert_outputs * gating_mask, dim=-1)
        
        return active_outputs


class SelAct(nn.Module):
  
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features
        
        self.activations = [
            F.relu,
            F.silu, 
            utils.sin_leakyrelu
        ]
        self.n_acts = len(self.activations)
        self.gate = nn.Parameter(torch.randn(in_features,self.n_acts) * 0.1)


    def forward(self, x):

        ghot = TopKHot.apply(self.gate,1)
        
        out = self.activations[0](x)*ghot[...,0]
        out += self.activations[1](x)*ghot[...,1]
        out += self.activations[2](x)*ghot[...,2]

        return out

class Autoencoder(nn.Module):
    def __init__(self, input, hidden, embed, qdim, quantizer , k = None):
        super().__init__()
        self.qdim = qdim
        self.encoder = nn.Sequential(nn.Linear(input, hidden), nn.ReLU(), nn.Linear(hidden, qdim))
        self.quantizer = Hotmod(quantizer, k)
        self.codebook = nn.Linear(self.qdim, embed, bias=False)
        self.decoder = nn.Sequential(nn.Linear(embed, hidden), nn.ReLU(), nn.Linear(hidden, input))
        
    def forward(self, x):
        khot = self.quant(x)
        reconstruction = self.dequant(khot)
        return reconstruction, khot

    def quant(self, x):
        logits = self.encoder(x)
        
        k_hot = self.quantizer(logits)
        
        return k_hot
    
    def dequant(self, k_hot, norm = True):
        if norm:
            k = self.k_from_hot(k_hot)
            k_hot = k_hot / k

        q = self.codebook(k_hot)
        return self.decoder(q)
    
    def k_from_hot(self, k_hot):
        return k_hot.sum(dim=-1, keepdim=True).clamp(1,self.qdim).detach()


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
        k_hot = torch.zeros(z_e.shape[0], self.quant_dim, device=z_e.device)
        k_hot.scatter_(1, top_k_indices, 1.0)
        z_q = torch.sum(z_q_k, dim=1)#why is this so much better than mean()?
        #z_q = F.normalize(z_q, p=2, dim=1) * self.embed_dim**0.5

        vq_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        total_vq_loss = vq_loss + self.commitment_cost * commitment_loss

        z_q_ste = z_e + (z_q - z_e).detach()
        return z_q_ste, total_vq_loss, k_hot

class mhvqvae(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_codes_to_select, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.k = num_codes_to_select
        self.commitment_cost = commitment_cost

        # The codebook is an nn.Embedding layer
        self.codebook = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, z_e):
        assert z_e.shape[-1] == self.embedding_dim

        distances = torch.sum(z_e.unsqueeze(1).pow(2), dim=2, keepdim=False) \
                  - 2 * torch.matmul(z_e, self.codebook.weight.t()) \
                  + torch.sum(self.codebook.weight.pow(2), dim=1)

        _, indices = torch.topk(-distances, self.k, dim=1)  # (B, k)

        k_hot = torch.zeros(z_e.shape[0], self.num_embeddings, device=z_e.device)
        k_hot.scatter_(1, indices, 1)

        quantized_k_vectors = self.codebook(indices)  # (B, k, D)
        z_q = quantized_k_vectors.mean(dim=1)  # (B, D)

        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = self.commitment_cost * F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + commitment_loss
        z_q = z_e + (z_q - z_e).detach()

        return z_q, vq_loss, k_hot

class debugAutoencoderWithVQ(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, quantizer, Act):
        super().__init__()
        # Note: We don't need qdim anymore, the quantizer knows it.
        self.embed_dim = embed_dim
        
        # The encoder now outputs a vector in the embedding space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            Act(),
            nn.Linear(hidden_dim, embed_dim) # Output size is embed_dim
        )
        
        self.quantizer = quantizer # This will be an instance of MultiHotVQVAEQuantizer
        
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            Act(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        # 1. Get continuous vector from encoder
        z_e = self.encoder(x)
        
        # The VQ-VAE quantizer handles everything in the middle
        z_q, vq_loss, k_hot = self.quantizer(z_e)
        
        # 3. Decode the quantized vector
        reconstruction = self.decoder(z_q)
        
        # For your logging, get the 'k' value from the k_hot vector
        k = None #k_hot.sum(dim=-1).mean().detach()

        return reconstruction, k_hot, vq_loss, k
    
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


class GatherByGate2(torch.autograd.Function):
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
        return output

    @staticmethod
    def backward(ctx, g):
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
            current_probs = F.softmax(gate_logits, dim=-1)
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

class GatherByGateAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, quant_dim, k):
        super().__init__()
        self.quant_dim = quant_dim
        self.embed_dim = embed_dim
        self.k = k
        self.shape_n = 16
        self.shape_dim = 16
        self.shape_K = quant_dim // self.shape_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.shape_n * self.shape_dim)
        )
        self.gate = nn.Sequential(
            nn.Linear(self.shape_n * self.shape_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.shape_n)
        )
        self.candidate_pool = nn.Parameter(torch.randn(1, self.shape_n, self.shape_dim))

        self.codebook = nn.Linear(self.quant_dim, embed_dim, bias=False)
       
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    
    def forward(self, x):
        khot, vq_loss = self.quant(x)
        k = self.k_from_hot(khot)
        reconstruction = self.dequant(khot)
        return reconstruction, khot, vq_loss, k

    def quant(self, x):
        batch_size = x.shape[0]
        encoding = self.encoder(x)

        gate = self.gate(encoding)
        #pool = self.candidate_pool.expand(batch_size, -1, -1)
        pool = encoding.view(batch_size, self.shape_n, self.shape_dim)
        gg = GatherByGate2.apply(gate, pool, self.shape_K)
        #gg = GumbelGatherByGate.apply(gate, pool, self.shape_K, 1.0, self.training, 1e-10)
        #pool = encoding.view(batch_size, self.shape_n, self.shape_dim)
        #agg = GatherByGate2.apply(gate, pool, self.shape_K, )
        #aux = F.mse_loss(gg.view(batch_size, -1), agg.view(batch_size, -1))
        aux = 0.0
        hot = ThresHot.apply(gg)
        hot = hot.view(batch_size, -1)
        return hot, aux
    
    def k_from_hot(self, hot):
        return hot.sum(dim=-1, keepdim=True).clamp(1,self.quant_dim).detach()

    def dequant(self, hot, norm = True):
        if norm:
            k = self.k_from_hot(hot)
            hot = hot / k

        q = self.codebook(hot)
        return self.decoder(q)
    def optimizer(self, LR):
        num_ae = sum(p.numel() for p in self.parameters())
        print(f"ae parameters: {num_ae}")
        # This optimizer setup is compatible with your harness
        return torch.optim.AdamW(self.parameters(), lr=LR)
    
# --- The Autoencoder and Experiment Setup ---
class debugAutoencoder(nn.Module):
    def __init__(self, input, hidden, embed, qdim, quantizer, Act , k = None):
        super().__init__()
        self.qdim = qdim
        
        params = inspect.signature(Act.__init__).parameters
        if('in_features' in params):
            ac = Act(hidden)
            ac2 = Act(hidden)
        else:
            ac=Act()
            ac2=Act()
        self.encoder = nn.Sequential(nn.Linear(input, hidden), ac, nn.Linear(hidden, qdim))
        self.quantizer = quantizer#Hotmod(quantizer, k)
        self.codebook = nn.Linear(self.qdim, embed, bias=False)
        self.decoder = nn.Sequential(nn.Linear(embed, hidden), ac2, nn.Linear(hidden, input))
        
    
    def forward(self, x):
        khot, vq_loss = self.quant(x)
        k = self.k_from_hot(khot)
        reconstruction = self.dequant(khot)
        return reconstruction, khot, vq_loss, k

    def quant(self, x):
        logits = self.encoder(x)
        vq_loss = 0.0

        k_hot = self.quantizer(logits)
        
        return k_hot, vq_loss
    
    def k_from_hot(self, k_hot):
        return k_hot.sum(dim=-1, keepdim=True).clamp(1,self.qdim).detach()

    def dequant(self, k_hot, norm = True):
        if norm:
            k = self.k_from_hot(k_hot)
            k_hot = k_hot / k

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
    
class Gumbell(nn.Module):
    def __init__(self,k):
        self.tau=1.0
        self.eps=1e-10
        self.k=k
        super().__init__()
    def forward(self, logits):
        gumbels = -torch.log(-torch.log(torch.rand_like(logits) + self.eps) + self.eps)
        perturbed_logits = logits + gumbels

        y_soft = torch.sigmoid(perturbed_logits / self.tau)

        _, indices = torch.topk(perturbed_logits, self.k, dim=-1)
        y_hard = torch.zeros_like(logits).scatter_(-1, indices, 1.0)

        output = y_hard - y_soft.detach() + y_soft

        return output
    
class Hotmod(nn.Module):
    def __init__(self, hotfunc, k):
        super().__init__()
        self.hotfunc = hotfunc
        params = inspect.signature(self.hotfunc.forward).parameters
        self._needs_k = 'k' in params
        self.k = k

    def forward(self, x):
        
        if self._needs_k:
            return self.hotfunc.apply(x, self.k)
        return self.hotfunc.apply(x)


class StochasticHotMod(nn.Module):
    def __init__(self, hotfunc, k):
        super().__init__()
        self.hotfunc = hotfunc
        params = inspect.signature(self.hotfunc.forward).parameters
        self._needs_k = 'k' in params
        self.k = k
        self.temp = 1.0

    def forward(self, x:torch.tensor):
        if self.training:
            uniform_samples = torch.rand_like(x)
            gumbels = -torch.log(-torch.log(uniform_samples + 1e-9) + 1e-9)
            gumbels = gumbels * torch.sqrt(x.norm(dim=-1,keepdim=True))
            noisy_x = (x + gumbels) 
            x_mod = noisy_x
        else:
            x_mod = x
        if self._needs_k:
            return self.hotfunc.apply(x_mod, self.k)
        return self.hotfunc.apply(x_mod)   
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
        if(k is not None):
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

def whiteloss( x, y):
    w = utils.zca_newton_schulz(torch.cat((x,y),dim=0))
    #x,y = torch.chunk(w,2,dim=0)
    x_w, y_w = w.split(x.size(0), dim=0)
    return utils.minmaxnorm(F.mse_loss(x_w , y_w,reduction="none")).sum()


if __name__ == '__main__':
    # Hyperparameters
    INPUT_DIM, HIDDEN_DIM, QUANT_DIM, EMBED_DIM = 28*28, 256, 64, 32
    BATCH_SIZE, LEARNING_RATE, STEPS = 256, 1e-3, 10001
    
    VAL_LOG_INTERVAL = 1000
    TRAIN_LOG_INTERVAL = 100
    VALIDATION_STEPS = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    full_train_dataset = torchvision.datasets.MNIST("./", train=True, download=True, transform=transform)
    train_size = 50000
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0, pin_memory=True)
    data_iterator = iter(train_loader)

    # Base models
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
        #"dynk3": DynKQAE2(INPUT_DIM, EMBED_DIM, HIDDEN_DIM, QUANT_DIM),
        #"threshk": ThreshotAE(INPUT_DIM, EMBED_DIM, HIDDEN_DIM, QUANT_DIM),
        #"BCEGR k 32":      NKQAE(INPUT_DIM, EMBED_DIM, HIDDEN_DIM, QUANT_DIM, 32),
        #"BCEGR k 1":      NKQAE(INPUT_DIM, EMBED_DIM, HIDDEN_DIM, QUANT_DIM, 1),
        #"CER k 1":    Autoencoder(INPUT_DIM, HIDDEN_DIM, EMBED_DIM, QUANT_DIM, TopOneHot,  1),
        #"CER2 k 1":   debugAutoencoder(INPUT_DIM, HIDDEN_DIM, EMBED_DIM, QUANT_DIM, TopKHot,    1),
        #"BCER k 1":   Autoencoder(INPUT_DIM, HIDDEN_DIM, EMBED_DIM, QUANT_DIM, TopKHotBCE, 1),
        #"thot":  debugAutoencoder(INPUT_DIM, HIDDEN_DIM, EMBED_DIM, QUANT_DIM, TopKHotBCE, TaylorThresHotActivation, 32),
        #"bspline":  debugAutoencoder(INPUT_DIM, HIDDEN_DIM, EMBED_DIM, QUANT_DIM, TopKHotBCE, BSplineActivation, 32),
        "GatherByGate": GatherByGateAutoencoder(
            input_dim=INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            embed_dim=EMBED_DIM,
            quant_dim=QUANT_DIM, # quant_dim is the size of our candidate pool
            k=32
        ),
        #"topkCE":  debugAutoencoder(INPUT_DIM, HIDDEN_DIM, EMBED_DIM, QUANT_DIM,  Hotmod(TopKHot,32), nn.SiLU ),
        #"topkBCE":  debugAutoencoder(INPUT_DIM, HIDDEN_DIM, EMBED_DIM, QUANT_DIM,  Hotmod(TopKHotBCE,32), nn.SiLU),
        #"topkHBCE": debugAutoencoder(INPUT_DIM, HIDDEN_DIM, EMBED_DIM, QUANT_DIM, Hotmod(HardTopKHotBCE, 32), nn.SiLU),
        #"treshot": debugAutoencoder(INPUT_DIM, HIDDEN_DIM, EMBED_DIM, QUANT_DIM, Hotmod(ThresHot, 32), nn.SiLU),
        #"stochhot": debugAutoencoder(INPUT_DIM, HIDDEN_DIM, EMBED_DIM, QUANT_DIM, StochasticHotMod(ThresHot, 32), nn.SiLU),
        #"gumbell": debugAutoencoder(INPUT_DIM, HIDDEN_DIM, EMBED_DIM, QUANT_DIM, Gumbell(32), nn.SiLU),
        #"mh-vqvae": debugAutoencoderWithVQ(
        #    input_dim=INPUT_DIM,
        #    hidden_dim=HIDDEN_DIM,
        #    embed_dim=EMBED_DIM,
        #    quantizer=mhvqvae(
        #        num_embeddings=QUANT_DIM,
        #        embedding_dim=EMBED_DIM,
        #        num_codes_to_select=32,
        #        commitment_cost=0.25
        #    ),
        #    Act=nn.SiLU
        #),
        #"ogimpl": debugAutoencoderWithVQ(
        #    input_dim=INPUT_DIM,
        #    hidden_dim=HIDDEN_DIM,
        #    embed_dim=EMBED_DIM,
        #    quantizer=MultiHotVQVAEQuantizer(
        #        quant_dim=QUANT_DIM,
        #        embed_dim=EMBED_DIM,
        #        k=32,
        #        commitment_cost=0.25
        #    ),
        #    Act=nn.SiLU
        #)
        #"CER k 32":   debugAutoencoder(INPUT_DIM, HIDDEN_DIM, EMBED_DIM, QUANT_DIM, TopKHot,    32),
        #"STE k1":    Autoencoder(INPUT_DIM, HIDDEN_DIM, EMBED_DIM, QUANT_DIM, tkSTE, 1),
        #"STE k32":   Autoencoder(INPUT_DIM, HIDDEN_DIM, EMBED_DIM, QUANT_DIM, tkSTE, 32),
        
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
            recon, logits, aux, k = model(images)
            #else:
            #    recon, logits, aux = model(images)
            #    k = model.qdim//2

            loss_raw = F.mse_loss(recon, images)
            
            loss = loss_raw+aux
            #loss = loss_raw + whiteloss(recon, images)
            #loss = loss + auxl(recon)#*0.1
            
            #r2, rq, _, k2 = model(recon.detach())
            #loss = loss + F.mse_loss(recon.detach(), r2) # The target distribution (from the first pass)

            #target_dist = F.softmax(logits.detach() / k, dim=-1)
            #input_dist_log = F.log_softmax(rq / k2, dim=-1)
            #loss = loss + F.kl_div(input_dist_log, target_dist, reduction="batchmean")
            #loss = loss + F.kl_div(F.softmax(logits.detach()/k.detach(),dim=-1), F.softmax(rq/k2,dim=-1), reduction="batchmean") 
            #loss.backward()
            #lw = F.softmax(recon)-F.softmax(images)
            loss.backward()
            optimizers[name].step()
            
            if step % 100 == 0:
                metrics[name]["train_loss"].append(loss_raw.detach().cpu().item())
                if(k is not None):
                    metrics[name]["train_k_var"].append(k.var().detach().cpu().item())
                    metrics[name]["train_k_mean"].append(k.mean().detach().cpu().item())
                else:
                    metrics[name]["train_k_var"].append(0)
                    metrics[name]["train_k_mean"].append(QUANT_DIM//2)
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
