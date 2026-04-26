
import torch
import torch.nn as nn
import torch.nn.functional as F

import parts.modules as modules
import parts.utils as utils

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

class TopKEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, embedding_layer, k):
        if k != 1:
            raise NotImplementedError("EfficientTopKEmbedding is currently implemented for k=1")
            
        predicted_indices = torch.topk(logits, k=1, dim=-1).indices.squeeze(-1)
        
        embedded_output = embedding_layer(predicted_indices)

        ctx.save_for_backward(logits, embedding_layer.weight)
        ctx.k = k
        
        return embedded_output

    @staticmethod
    def backward(ctx, grad_output_from_transformer):
        logits, embedding_weights = ctx.saved_tensors
        k = ctx.k

        grad_output_from_transformer = grad_output_from_transformer.unsqueeze(-2)
        embedding_weights = embedding_weights.unsqueeze(0).unsqueeze(0)
        grad_to_one_hot = torch.matmul(grad_output_from_transformer.squeeze(-2), embedding_weights.squeeze(0).squeeze(0).T)
        

        g = grad_to_one_hot

        with torch.no_grad():
            topk_vals, topk_indices = torch.topk(-g, k=k, dim=-1)
            soft_weights = F.softmax(topk_vals, dim=-1)
            soft_target = torch.zeros_like(logits).scatter_(-1, topk_indices, soft_weights)

        grad_logits = F.softmax(logits, dim=-1) - soft_target
        
        return grad_logits, None, None


class ExplorativeTKE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, embedding_layer, k):
        if k != 1:
            raise NotImplementedError("EfficientTopKEmbedding is currently implemented for k=1")
            
        predicted_indices = torch.topk(logits, k=1, dim=-1).indices.squeeze(-1)
        
        embedded_output = embedding_layer(predicted_indices)

        ctx.save_for_backward(logits, embedding_layer.weight)
        ctx.k = k
        
        return embedded_output

    @staticmethod
    def backward(ctx, grad_output_from_transformer):
        logits, embedding_weights = ctx.saved_tensors
        k = ctx.k

        grad_output_from_transformer = grad_output_from_transformer.unsqueeze(-2)
        embedding_weights = embedding_weights.unsqueeze(0).unsqueeze(0)
        g = torch.matmul(grad_output_from_transformer.squeeze(-2), embedding_weights.squeeze(0).squeeze(0).T)
        


        with torch.no_grad():
            topk_vals, topk_indices = torch.topk(-g, k=k, dim=-1)
            soft_weights = F.softmax(topk_vals, dim=-1)
            soft_target = torch.zeros_like(logits).scatter_(-1, topk_indices, soft_weights)

        grad_logits = F.softmax(logits, dim=-1) - soft_target
        # Calculate a per-token learning modulator
        with torch.no_grad():
            # High entropy -> high uncertainty -> learn more aggressively
            # Low entropy -> high certainty -> learn more cautiously
            entropy = -torch.sum(F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1), dim=-1, keepdim=True)
            # Normalize to a reasonable range, e.g., [0.5, 1.5]
            learning_modulator = 0.5 + (entropy / torch.log(torch.tensor(logits.size(-1))))

        grad_logits = grad_logits * learning_modulator
        return grad_logits, None, None
class DiscreteLookupFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # x: (B, T, H)
        # weight: (V, H) - Shared with wte usually
        logits = F.linear(x, weight) # (B, T, V)
        #logits = utils.gumbell_noise(logits)
        _, indices = torch.topk(logits, k=1, dim=-1) # (B, T, k)
        
        indices_flat = indices.squeeze(-1) # (B, T)
        y = F.embedding(indices_flat, weight) # (B, T, H)
        
        ctx.save_for_backward(x, weight, logits, indices)
        ctx.k = 1
        return y#, indices
  
    @staticmethod
    def backward(ctx, grad_y):
        x, weight, logits, indices_flat = ctx.saved_tensors
        B, T, H = grad_y.shape
        V = weight.shape[0]

        grad_proj = grad_y @ weight.t()
        
        _, target_idx = torch.topk(grad_proj, k=1, dim=-1, largest=False)

        probs = torch.sigmoid(logits)
        
        target_mask = torch.zeros_like(probs)
        target_mask.scatter_(-1, target_idx, 1.0)
        
        grad_logits = probs - target_mask 

        grad_W = torch.zeros_like(weight)
        
        grad_W.index_add_(0, indices_flat.reshape(-1), grad_y.reshape(-1, H))
        
        grad_logits_flat = grad_logits.view(-1, V)
        #x_flat = x.view(-1, H)
        
        #grad_head = grad_logits_flat.t() @ x_flat 
        grad_W = grad_W #+ grad_head# avg?
        
        grad_x = grad_logits_flat @ weight
        grad_x = grad_x.view(B, T, H)

        return grad_x, grad_W
    
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
            _, topk_indices = torch.topk(-g, k=k, dim=-1)
            hard_target = torch.zeros_like(x).scatter_(-1, topk_indices, 1.0)
        grad_x = F.sigmoid(x) - hard_target
        return grad_x,  None

class ThresHot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        probs = F.softmax(x, dim=-1)
        k_hot = torch.where(probs >= (1.0 / x.shape[-1]), 1.0, 0.0)
        ctx.save_for_backward(x)
        return k_hot

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors

        with torch.no_grad():
            g_mean = torch.mean(g, dim=-1, keepdim=True)
            
            hard_target = torch.where(g < g_mean, 1.0, 0.0)

        grad_x = F.sigmoid(x) - hard_target
        return grad_x    


class AdjThresHot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold_mult):
        probs = F.softmax(x, dim=-1)
        k_hot = torch.where(probs >= (threshold_mult / x.shape[-1]), 1.0, 0.0)
        ctx.save_for_backward(x)
        return k_hot

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        with torch.no_grad():
            g_mean = torch.mean(g, dim=-1, keepdim=True)
            hard_target = torch.where(g < g_mean, 1.0, 0.0)
        grad_x = torch.sigmoid(x) - hard_target
        return grad_x, None


class ThresHot2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        
        ctx.save_for_backward(x)
        return (x > 0.0).float()

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors

        with torch.no_grad():
            g_mean = torch.mean(g, dim=-1, keepdim=True)
            
            soft_target = torch.where(g < g_mean, 1.0, 0.0)

        grad_x = F.sigmoid(x) - soft_target
        return grad_x

class ThresHot3(torch.autograd.Function):
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
            k_percentile = torch.quantile(g, 0.1, dim=-1, keepdim=True) 
            hard_target = torch.where(g < k_percentile, 1.0, 0.0)

        grad_x = F.sigmoid(x) - hard_target
        return grad_x


class avgHot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        k_hot = torch.where(x > x.mean(), 1.0, 0.0)
        ctx.save_for_backward(x)
        return k_hot

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        with torch.no_grad():
            hard_target = torch.where(g < g.mean(), 1.0, 0.0)
        grad_x = F.sigmoid(x) - hard_target
        return grad_x

class BoltzmannTopKHot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k):
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
        
            k_tensor = torch.tensor(k, device=x.device).float().expand(x.shape[0], 1)
            
            mu = utils.get_boltzmann_mu(x, k_tensor)
        grad_x = torch.sigmoid(x - mu) - hard_target
        
        return grad_x, None
class BoltzmannThresHot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Forward remains similar: standard Softmax thresholding
        probs = F.softmax(x, dim=-1)
        k_hot = torch.where(probs >= (1.0 / x.shape[-1]), 1.0, 0.0)
        ctx.save_for_backward(x)
        return k_hot

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors

        with torch.no_grad():
            g_mean = torch.mean(g, dim=-1, keepdim=True)
            hard_target = torch.where(g < g_mean, 1.0, 0.0)
            
            current_k = hard_target.sum(dim=-1, keepdim=True)
            
            mu = utils.get_boltzmann_mu(x, current_k)

        grad_x = torch.sigmoid(x - mu) - hard_target
        
        return grad_x

# --- The Autoencoder and Experiment Setup ---
wn = torch.nn.utils.parametrizations.weight_norm
class RegenerativeLinear(nn.Module):
    def __init__(self, in_features, out_features, sparsity=0.5, 
                 bias=False, lookahead=1e-2, noise=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity
        self.k = int((1 - sparsity) * in_features)

        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        #nn.init.kaiming_normal_(self.weight)
        self.scores = nn.Parameter(torch.randn(out_features, in_features))
        
        wn(self, name='scores',dim=None) 
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))

        self.hungry_config = {
            'lookahead_alpha': lookahead,
            'noise': noise,
            'noise_scale': 0.01,
            'g_norm': True,
        }

    def forward(self, x, conf=None):
        if conf is not None:
            self.hungry_config['lookahead_alpha'] = conf.get('lr', 1e-2)

        mask = avgHot.apply(self.scores)

        effective_weight = self.weight * mask

        return modules.HungryLinearFunc.apply(
            x, effective_weight, self.bias, self.hungry_config
            )

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
    