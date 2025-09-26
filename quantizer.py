
import copy
import inspect
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

import utils

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
    def __init__(self, quant_dim, embed_dim, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.embedding = nn.Linear(quant_dim, embed_dim, bias=False)

    def forward(self, logits):
        y_soft = F.gumbel_softmax(logits, tau=self.temperature, hard=True)
        
        z_q = self.embedding(y_soft)
        
        return z_q
    
class tkSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x,  k):
        batch_size, quant_dim = x.shape
        k_values = torch.full((batch_size, 1), k, device=x.device)
        
        sorted_indices = torch.argsort(x, dim=-1, descending=True)
        k_range = torch.arange(quant_dim, device=x.device)[None, :].expand(batch_size, -1)
        k_hot_mask = (k_range < k_values).float()
        k_hot = k_hot_mask.gather(-1, torch.argsort(sorted_indices, dim=-1))
        
        return k_hot

    @staticmethod
    def backward(ctx, g):
        grad_x = g 
        return grad_x, None
    
class MultiHotSTEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, linear_weight, k):
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
        grad_linear_weight = g.T @ k_hot
        grad_x = g @ linear_weight
        return grad_x, grad_linear_weight, None

class TopOneHot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        _, indices = torch.topk(x, k=1, dim=-1)

        one_hot = F.one_hot(indices.squeeze(-1), num_classes=x.shape[-1]).float()

        ctx.save_for_backward(x)

        return one_hot

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors

        _, target_indices = torch.topk(-g, k=1, dim=-1)

        target_one_hot = F.one_hot(target_indices.squeeze(-1), num_classes=x.shape[-1]).float()
        
        grad_x = F.softmax(x, dim=-1) - target_one_hot

        return grad_x


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

class GatherByGate2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate_logits, candidate_pool, k):
        b,ct,c = candidate_pool.shape
        b,gt = gate_logits.shape
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
            ideal_targets = y - g

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
            probs = F.softmax(gate_logits, dim=-1)
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


# --- The Autoencoder and Experiment Setup ---
class debugAutoencoder(nn.Module):
    def __init__(self, input, hidden, embed, qdim, quantizer, Act , k = None):
        super().__init__()
        self.qdim = qdim
        

        self.encoder = nn.Sequential(nn.Linear(input, hidden), Act(hidden), nn.Linear(hidden, qdim))
        self.quantizer = Hotmod(quantizer, k)
        self.codebook = nn.Linear(self.qdim, embed, bias=False)
        self.decoder = nn.Sequential(nn.Linear(embed, hidden), Act(hidden), nn.Linear(hidden, input))
        
    
    def forward(self, x):
        khot, vq_loss = self.quant(x)
        k = self.k_from_hot(khot)
        reconstruction = self.dequant(khot)
        return reconstruction, khot, vq_loss, k

    def quant(self, x):
        logits = self.encoder(x)
        vq_loss = 0.0

        if isinstance(self.quantizer, MultiHotVQVAEQuantizer):
            k_hot, vq_loss = self.quantizer(logits)
        else:
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
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=4, pin_memory=True)
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
        "thot":  debugAutoencoder(INPUT_DIM, HIDDEN_DIM, EMBED_DIM, QUANT_DIM, TopKHotBCE, TaylorThresHotActivation, 32),
        "bspline":  debugAutoencoder(INPUT_DIM, HIDDEN_DIM, EMBED_DIM, QUANT_DIM, TopKHotBCE, BSplineActivation, 32),
        
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
            loss = loss_raw + aux
            loss = loss + auxl(recon)#*0.1
            
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
