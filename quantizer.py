
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import copy
from torch.utils.data import DataLoader, random_split

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

#class SoftTargetQuantizerFunction(torch.autograd.Function):
#    @staticmethod
#    def forward(ctx, x, linear_weight, k_decider):
#        kd = F.linear(x, k_decider)
#        _, k = torch.topk(kd, k=1, dim=-1)
#
#        _, indices = torch.topk(x, k=k, dim=-1)
#        # In this model, the forward pass is a hard k-hot lookup
#        k_hot = torch.zeros_like(x).scatter_(-1, indices, 1.0)
#        
#        ctx.save_for_backward(x, linear_weight, k_decider, k_hot, kd)
#        ctx.k = k
#
#        # The output is still the sum of the k chosen vectors
#        output = F.linear(k_hot, linear_weight)
#        return output
#
#    @staticmethod
#    def backward(ctx, g):
#        x, linear_weight, k_decider, k_hot_from_fwd, kd = ctx.saved_tensors
#        k_values = ctx.k
#
#        grad_linear_weight = g.T @ k_hot_from_fwd
#
#        gl = g @ linear_weight
#        
#        with torch.no_grad():
#            batch_size = x.shape[0]
#            max_k = kd.shape[-1]
#            kfrom = (k_values - 1).clamp(min=1,max=max_k-2)
#            k_probes = torch.cat([
#                kfrom,
#                kfrom+1,
#                kfrom + 2
#            ], dim=1) # Shape: (batch, 3)
#            variances = torch.zeros_like(k_probes, dtype=torch.float)
#
#            for i in range(3):
#                current_k_probe = k_probes[:, i]
#                top_vals, _ = torch.topk(-gl, k=max_k, dim=-1) # Get all sorted vals once
#
#                # Create a mask for the top-k values for each item
#                mask = torch.arange(max_k, device=x.device)[None, :] < current_k_probe[:, None]
#
#                # Apply softmax only to the valid top-k values
#                masked_vals = torch.where(mask, top_vals, torch.tensor(-float('inf'), device=x.device))
#                soft_weights = F.softmax(masked_vals, dim=-1)
#
#                variances[:, i] = soft_weights.var(dim=-1)
#    
#            
#            # For each item, find which probe (0, 1, or 2) had the minimum variance
#            best_probe_indices = torch.argmin(variances, dim=-1)
#            
#            # The target k is the k value from the probe that won
#            target_k = k_probes.gather(-1, best_probe_indices.unsqueeze(-1)).squeeze(-1)
#            
#            # Create a one-hot target for the k-decider logits
#            target_k_indices = target_k - 1 # Map k back to index
#            target_k_hot = F.one_hot(target_k_indices, num_classes=max_k).float()
#
#        # The gradient for the k-decider's output logits
#        grad_kd = F.softmax(kd, dim=-1) - target_k_hot
#        
#        # Propagate this gradient back to the k_decider's weights via chain rule
#        grad_k_decider_weight = grad_kd.T @ x
#
#        
#       #grad_linear_weight = g.T @ k_hot_from_fwd 
#
#       #_, indices = torch.topk(x, k=k_values, dim=-1)
#       ## In this model, the forward pass is a hard k-hot lookup
#       #target_k_hot = torch.zeros_like(x).scatter_(-1, indices, 1.0)
#       ## Calculate the gradient for x. This is the mathematical gradient of
#       ##    CrossEntropy(x, target), which is (softmax(x) - target_one_hot).
#       ##    This is the core of your idea, substituted in place of the real gradient.
#       #xsm = F.softmax(x, dim=-1)
#
#       #current_k_probe = k_probes[:, i]
#       #top_vals, _ = torch.topk(-grad_kd, k=max_k, dim=-1) # Get all sorted vals once
#       ## Create a mask for the top-k values for each item
#       #mask = torch.arange(max_k, device=x.device)[None, :] < current_k_probe[:, None]
#       ## Apply softmax only to the valid top-k values
#       #masked_vals = torch.where(mask, top_vals, torch.tensor(-float('inf'), device=x.device))
#       #soft_weights = F.softmax(masked_vals, dim=-1)
#       #
#       #
#       #masked_weight = torch.where(mask, top_vals, xsm, device=x.device)
#
#       #grad_x = masked_weight - target_k_hot
#       ##grad_x = F.softmax(x, dim=-1) - soft_target
#        # --- Part 3: REFINED Gradient for the input x ---
#        # This now correctly implements the dynamic Soft-Target logic.
#        with torch.no_grad():
#            # 3a: Get top values and indices from the CORRECT signal (-gl) using the DYNAMIC k_values
#            # We use the same masking trick as in the forward pass.
#            sorted_gl_indices = torch.argsort(-gl, dim=-1, descending=True)
#            topk_gl_vals = (-gl).gather(-1, sorted_gl_indices)
#            
#            # Create mask based on k_values chosen in the forward pass
#            k_range = torch.arange(x.shape[-1], device=x.device)[None, :]
#            k_mask = (k_range < k_values).float()
#            
#            # Apply mask to the sorted values
#            masked_topk_gl_vals = torch.where(k_mask > 0, topk_gl_vals, torch.tensor(-float('inf'), device=x.device))
#
#            # 3b: Softmax within the dynamic k-window to get soft_weights
#            soft_weights = F.softmax(masked_topk_gl_vals, dim=-1)
#
#            # 3c: Scatter the weights back to the original order to create the soft_target
#            soft_target = soft_weights.gather(-1, torch.argsort(sorted_gl_indices, dim=-1))
#
#        # 3d: The final, clean gradient calculation
#        grad_x = F.softmax(x, dim=-1) - soft_target
#
#        return grad_x, grad_linear_weight, grad_k_decider_weight
# (The DynamicKQuantizerFunction from the previous answer is correct and goes here)
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

class HybridQuantizer(nn.Module):
    def __init__(self, quant_dim, output_dim, beta=0.25):
        super().__init__()
        self.linear = nn.Linear(quant_dim, output_dim, bias=False)
        self.beta = beta
    def forward(self, x):
        return HybridQuantizerFunction.apply(x, self.linear.weight, self.beta)

# --- The Autoencoder and Experiment Setup ---
class Autoencoder(nn.Module):
    def __init__(self, encoder, quantizer, decoder):
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

def calculate_perplexity(logits):
    # Calculates the perplexity of the codebook usage for a batch
    probs = F.softmax(logits, dim=-1).mean(dim=0)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10))
    perplexity = torch.exp(entropy)
    return perplexity.item()

@torch.no_grad()
def run_validation(model, val_loader, device, num_batches=10):
    model.eval()
    total_loss = 0
    total_perplexity = 0
    batches_processed = 0
    for i, (images, _) in enumerate(val_loader):
        if i >= num_batches:
            break
        images = images.view(-1, INPUT_DIM).to(device)
        recon, logits, _ = model(images)
        total_loss += F.mse_loss(recon, images).item()
        if(logits is not None):
            total_perplexity += calculate_perplexity(logits)
        batches_processed += 1
    
    avg_loss = total_loss / batches_processed if batches_processed > 0 else 0
    avg_perplexity = total_perplexity / batches_processed if batches_processed > 0 else 0
    model.train()
    return avg_loss, avg_perplexity

if __name__ == '__main__':
    # Hyperparameters
    INPUT_DIM, HIDDEN_DIM, QUANT_DIM, EMBED_DIM = 28*28, 256, 64, 32
    BATCH_SIZE, LEARNING_RATE, STEPS = 256, 1e-3, 1000
    
    VALIDATION_INTERVAL = 1000
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
        "VQ-VAE 32": Autoencoder(copy.deepcopy(encoder_for_vqvae), MultiHotVQVAEQuantizer(QUANT_DIM, EMBED_DIM,  QUANT_DIM//2, 0.25), copy.deepcopy(decoder_base)),
        "VQ-VAE":    Autoencoder(copy.deepcopy(encoder_for_vqvae), MultiHotVQVAEQuantizer(QUANT_DIM, EMBED_DIM, 1, 0.25), copy.deepcopy(decoder_base)),
        "Gumbell":   Autoencoder(copy.deepcopy(encoder_base),      GumbelQuantizer(QUANT_DIM, EMBED_DIM, 1), copy.deepcopy(decoder_base)),
        "CER":       Autoencoder(copy.deepcopy(encoder_base),      CustomQuantizer(QUANT_DIM, EMBED_DIM), copy.deepcopy(decoder_base)),
        "STE":       Autoencoder(copy.deepcopy(encoder_base),      STEQuantizer(QUANT_DIM, EMBED_DIM), copy.deepcopy(decoder_base)),
        "CER 32":    Autoencoder(copy.deepcopy(encoder_base),      NKQuantizer(QUANT_DIM, EMBED_DIM, QUANT_DIM//2), copy.deepcopy(decoder_base)),
        #"CER 16": Autoencoder(copy.deepcopy(encoder_base), NKQuantizer(QUANT_DIM, EMBED_DIM, 16), copy.deepcopy(decoder_base)),
        #"STE 16": Autoencoder(copy.deepcopy(encoder_base), MultiHotSTEQuantizer(QUANT_DIM, EMBED_DIM, 16), copy.deepcopy(decoder_base)),
        #"STE qd0.5": Autoencoder(copy.deepcopy(encoder_base), MultiHotSTEQuantizer(QUANT_DIM, EMBED_DIM, QUANT_DIM//2), copy.deepcopy(decoder_base)),
    }
    
    print("Compiling models...")
    torch.set_float32_matmul_precision('medium')
    models = {name: torch.compile(model.to(device)) for name, model in models.items()}
    print("Compilation complete.")

    optimizers = {name: torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) for name, model in models.items()}
    # --- NEW: Expanded Metrics Tracking ---
    metrics = {name: {"train_loss": [], "val_loss": [], "train_perp": [], "val_perp": [], "steps": []} for name in models.keys()}

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
            recon, logits, aux = model(images)
            loss_raw = F.mse_loss(recon, images) 
            loss = loss_raw + aux if aux is not None else 0.0
            loss.backward()
            optimizers[name].step()
            
            if step % 100 == 0:
                metrics[name]["train_loss"].append(loss_raw.detach().item())
                perp = calculate_perplexity(logits.detach()) if logits is not None else 0.0
                metrics[name]["train_perp"].append(perp)

        if step > 0 and step % VALIDATION_INTERVAL == 0:
            print(f"--- Step {step:5d} ---")
            for name, model in models.items():
                val_loss, val_perp = run_validation(model, val_loader, device, VALIDATION_STEPS)
                metrics[name]["val_loss"].append(val_loss)
                metrics[name]["val_perp"].append(val_perp)
                metrics[name]["steps"].append(step)
                print(f"{name:>10} | Train Loss: {metrics[name]['train_loss'][-1]:.6f} | Val Loss: {val_loss:.6f} | Val Perp: {val_perp:.2f}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    
    for name in models.keys():
        # --- Fix 1: Generate the correct x-axis for the training data ---
        # The train_loss is recorded more frequently than the validation loss.
        # We must create a corresponding x-axis for it.
        # Assuming train_loss is recorded every 10 steps as in the previous setup.
        num_train_logs = len(metrics[name]["train_loss"])
        train_steps_x_axis = range(0, num_train_logs * 10, 10) # Creates an x-axis of the correct length and scale
        
        # Plot training loss
        if num_train_logs > 0:
            ax1.plot(train_steps_x_axis, metrics[name]["train_loss"], label=f'{name} Train Loss', alpha=0.5)
        
        # --- Fix 2: Guard validation plots to prevent errors on short runs ---
        # Only attempt to plot validation data if it actually exists.
        if len(metrics[name]["steps"]) > 0:
            ax1.plot(metrics[name]["steps"], metrics[name]["val_loss"], label=f'{name} Val Loss', linestyle='--', marker='o', markersize=4)
            ax2.plot(metrics[name]["steps"], metrics[name]["val_perp"], label=f'{name} Val Perplexity', linestyle='--', marker='o', markersize=4)
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

    plt.tight_layout()
    plt.show()
