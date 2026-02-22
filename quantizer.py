
import copy
import inspect
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.adamw
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

import modules
import optim
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


class Autoencoder(nn.Module):
    def __init__(self, input, hidden, embed, qdim, quantizer , k = None):
        super().__init__()
        self.qdim = qdim
        self.encoder = nn.Sequential(
            nn.Linear(input, hidden), 
            acmod(utils.dead_rat_relu), 
            nn.Linear(hidden, qdim))
        self.quantizer = Hotmod(quantizer, k)
        self.codebook = nn.Linear(self.qdim, embed, bias=False)
        self.decoder = nn.Sequential(
            nn.Linear(embed, hidden), 
            acmod(utils.dead_rat_relu), 
            nn.Linear(hidden, input))
        
    def forward(self, x):
        khot = self.quant(x)
        reconstruction = self.dequant(khot)
        return reconstruction, khot

    def quant(self, x):
        logits = self.encoder(x)
        
        k_hot = self.quantizer(logits)
        
        return k_hot
    
    def dequant(self, k_hot, norm = False):

        q = self.codebook(k_hot)
        if norm:
            q = utils.rms(q)
        return self.decoder(q)
    


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
        self.embed_dim = embed_dim
    
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            Act(),
            nn.Linear(hidden_dim, embed_dim) 
        )
        
        self.quantizer = quantizer 
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            Act(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        z_e = self.encoder(x)
        
        z_q, vq_loss, k_hot = self.quantizer(z_e)
        
        reconstruction = self.decoder(z_q)
        
        k = None #k_hot.sum(dim=-1).mean().detach()

        return reconstruction, k_hot, vq_loss, k
    
    def optimizer(self, LR):
        num_ae = sum(p.numel() for p in self.parameters())
        print(f"ae parameters: {num_ae}")
        return torch.optim.AdamW(self.parameters(), lr=LR)

class SelfAttentionPaletteAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, quant_dim, k):
        super().__init__()
        self.quant_dim = quant_dim
        self.embed_dim = embed_dim
        self.k = k # Not directly used in this version, but kept for consistency
        
        self.shape_n = 16 # How many shapes in our dynamic palette
        self.shape_dim = 16 # The dimension of each shape
        self.shape_K = quant_dim // self.shape_dim # How many shapes to select

        self.encoder_palette_generator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.shape_n * self.shape_dim)
        )
        
        self.query_generator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.shape_dim) # Query must have same dim as shapes
        )
        self.codebook = nn.Linear(self.quant_dim, embed_dim, bias=False)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.attn = modules.Attention(self.shape_dim, 1, 0.0, False)
    
    def forward(self, x):
        k_hot, vq_loss, logits = self.quant(x)
        k = self.k_from_hot(k_hot).detach() 
        reconstruction = self.dequant(k_hot)
        return reconstruction, logits, vq_loss, k

    def quant(self, x):
        batch_size = x.shape[0]
        
        flat_palette = self.encoder_palette_generator(x)
        dynamic_palette = flat_palette.view(batch_size, self.shape_n, self.shape_dim)

        selected_shapes = self.attn(dynamic_palette,causal=False)[:,:self.shape_K,:]

        hot = ThresHot.apply(selected_shapes)
        hot = hot.view(batch_size, -1)
        
        return hot, 0.0, selected_shapes
    
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
        return torch.optim.AdamW(self.parameters(), lr=LR)
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
Linear = RegenerativeLinear
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
        self.encoder = nn.Sequential(
            Linear(input, hidden), 
            ac, 
            nn.Linear(hidden, qdim))
        self.quantizer = Hotmod(quantizer, k)
        self.codebook = nn.Linear(self.qdim, embed, bias=False)
        self.decoder = nn.Sequential(
            Linear(embed, hidden), 
            ac2, 
            nn.Linear(hidden, input))
    
    def forward(self, x):
        k_hot, vq_loss = self.quant(x)
        reconstruction = self.dequant(k_hot)

        k = utils.k_from_hot(k_hot, self.qdim)
        return reconstruction, k_hot, vq_loss, k

    def quant(self, x):
        logits = self.encoder(x)
        vq_loss = 0.0

        k_hot = self.quantizer(logits)
        
        return k_hot, vq_loss
    

    def dequant(self, k_hot, norm = False):
        k = utils.k_from_hot(k_hot, self.qdim)
        if norm:
            k_hot = k_hot / k.detach()

        q = self.codebook(k_hot) #/ k.detach()
        #q = utils.rms(q)
        return self.decoder(q)

    
    def optimizer(self, LR):
        num_ae = sum(p.numel() for p in self.parameters())
        print(f"ae parameters: {num_ae}")
        return optim.Grms(params=self.parameters(), 
            lr=LR, 
            #nesterov    = True, 
            #momentum    = 0.9,
            fused       = True,
            #weight_decay= 0.01
            )
        #return torch.optim.adamw(params=self.parameters(), lr=LR, fused=True)
    
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
    def __init__(self, hotfunc, k=None):
        super().__init__()
        self.hotfunc = hotfunc
        params = inspect.signature(self.hotfunc.forward).parameters
        self._needs_k = 'k' in params
        self.k = k

    def forward(self, x):
        
        if self._needs_k:
            return self.hotfunc.apply(x, self.k)
        return self.hotfunc.apply(x)

def acmod(ac):
    class Mod(nn.Module):
        def __init__(self, x=None):
            super().__init__()

        def forward(self, x):
            return ac(x)
    return Mod

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
def run_validation(model, val_loader, device, num_batches=10, input_dim=28*28):
    model.eval()
    total_loss = 0
    total_perplexity = 0
    total_k_perplexity = 0
    total_k = 0
    batches_processed = 1
    for i, (images, _) in enumerate(val_loader):
        if i >= num_batches:
            break
        images = images.view(-1, input_dim).to(device)
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



def whiteloss( x, y):
    w = utils.zca_newton_schulz(torch.cat((x,y),dim=0))
    #x,y = torch.chunk(w,2,dim=0)
    x_w, y_w = w.split(x.size(0), dim=0)
    return utils.minmaxnorm(F.mse_loss(x_w , y_w,reduction="none")).sum()



def quantizer_run():
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
        #"bthot":  debugAutoencoder(INPUT_DIM, HIDDEN_DIM, EMBED_DIM, QUANT_DIM, BoltzmannThresHot, nn.LeakyReLU, 32),
        #"btkhot":  debugAutoencoder(INPUT_DIM, HIDDEN_DIM, EMBED_DIM, QUANT_DIM, BoltzmannTopKHot, nn.LeakyReLU, 32),
        #"tkhot":  debugAutoencoder(INPUT_DIM, HIDDEN_DIM, EMBED_DIM, QUANT_DIM, TopKHot, nn.LeakyReLU, 32),
        "thot":  debugAutoencoder(INPUT_DIM, HIDDEN_DIM, EMBED_DIM, QUANT_DIM, ThresHot, acmod(utils.dead_rat_relu), 32),
    }
    
    print("Compiling models...")
    torch.set_float32_matmul_precision('medium')
    models = {name: torch.compile(model.to(device)) for name, model in models.items()}
    auxl = torch.compile(utils.wnormlearnloss(INPUT_DIM).to(device))
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

if __name__ == '__main__':
    #run_seed_viz()
    #run_live_viz()
    quantizer_run()