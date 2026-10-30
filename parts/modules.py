
import math

from attr import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import parts.utils as utils
import parts.quantizer as quantizer

from torch.nn.attention.flex_attention import flex_attention, create_block_mask


class HungryLinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, config):
        ctx.save_for_backward(input, weight, bias)
        ctx.config = config
        return F.linear(input, weight, bias)
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        conf = ctx.config
        #print("real")
        if conf.get('noise', False):
            #larger batch = better sampling = less noise needed, i don't think i need to scale.
            input_magnitude = input.abs().sum() #/ input.shape[0]
            noise = torch.randn_like(input) / ( 1.0 + input_magnitude)
            scale = conf.get('noise_scale', 0.1)
            noise2 = torch.randn_like(input) * grad_output.std() * scale
            input = input + noise + noise2
            #grad_output = grad_output + noise2
        #grad_output = utils.rms(grad_output)
        input_flat = input.reshape(-1, input.shape[-1])
        grad_flat = grad_output.reshape(-1, grad_output.shape[-1])
        
        grad_weight = grad_flat.t().matmul(input_flat)
        
        alpha = conf.get('lookahead_alpha', 0.001)
        gwin = grad_weight
        if conf.get('g_norm', True):
            gwin = utils.rms_norm(gwin)
        future_weight = weight - (gwin * alpha)
        
        grad_input = grad_output.matmul(future_weight)
        
        grad_bias = None
        if bias is not None:
            grad_bias = grad_flat.sum(0) if grad_output is not None else None
        #grad_input = utils.rms(grad_input)
        return grad_input, grad_weight, grad_bias, None

class HungryLinear(nn.Module):
    def __init__(self, in_features, out_features, bias = False, lookahead=1e-5, noise=True, zero_init=True, g_norm =True):
        super().__init__()
        self.in_features = in_features
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_normal_(self.weight)
        if zero_init:
            self.weight.data.zero_()
            noise=True
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        
        self.config = {
            'lookahead_alpha': lookahead,
            'noise': noise,
            'noise_scale': 0.01, 
            'g_norm': g_norm,
        }
        
    def forward(self, x, conf = None):
        if conf is not None:
            self.config['lookahead_alpha'] = conf['lr']
        return HungryLinearFunc.apply(
            x, self.weight, self.bias, self.config, 
        )    


class SurfaceTensionLinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, config):
        ctx.save_for_backward(input, weight, bias)
        ctx.config = config
        return F.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        conf = ctx.config
        
        # --- 1. Original "Hungry" Noise Logic ---
        if conf.get('noise', False):
            input_magnitude = input.abs().sum()
            # Added slight epsilon to avoid div by zero
            noise = torch.randn_like(input) / (1.0 + input_magnitude + 1e-8)
            noise2 = torch.randn_like(input) * (grad_output.std() + 1e-8) * 0.1
            input = input + noise + noise2

        input_flat = input.reshape(-1, input.shape[-1])
        grad_flat = grad_output.reshape(-1, grad_output.shape[-1])
        
        # --- 2. Standard Task Gradient ---
        # dL / dw
        grad_weight = grad_flat.t().matmul(input_flat)
        
        # --- 3. INJECT SURFACE TENSION HERE ---
        tension_strength = conf.get('tension', 0.0)
        
        if tension_strength > 0.0:
            out_dim, in_dim = weight.shape
            
            # A. Normalize to project onto the Hypersphere Worldsheet
            # We treat normalization as a constant for the gradient direction calculation
            # to keep the vector math stable.
            w_norm = F.normalize(weight, p=2, dim=[0,1], eps=1e-8)
            
            # B. Gram Matrix (The induced metric on the worldsheet)
            # G_ij = w_i . w_j
            G = torch.mm(w_norm, w_norm.t())
            
            # C. Calculate the Expansion Force
            # We want to minimize Loss = -log(det(G))
            # The gradient direction is -inv(G) * W
            
            nambu_grad = None
            
            # CASE 1: Worldsheet has volume (fewer neurons than dimensions)
            if out_dim <= in_dim:
                # Add tiny jitter to diagonal for numerical stability of inverse
                jitter = torch.eye(out_dim, device=weight.device) * 1e-4
                try:
                    # The force that maximizes determinant is proportional to Inverse(G)
                    # We compute G^-1 @ W
                    G_inv = torch.linalg.inv(G + jitter)
                    
                    # The Gradient of -log(det(G)) is -G_inv.
                    # However, since we want to MAXIMIZE volume, we move in direction of G_inv.
                    # But this is a Gradient (direction of ascent of Loss), so we subtract the ascent.
                    # Effective Force: Push weights to align with the "holes" in the volume.
                    nambu_grad = -torch.mm(G_inv, w_norm)
                    
                except RuntimeError:
                    # Fallback if matrix is degenerate
                    nambu_grad = torch.mm(G - torch.eye(out_dim, device=weight.device), w_norm)

            # CASE 2: Worldsheet is crumpled (more neurons than dimensions)
            else:
                # Determinant is 0, so Inverse Nambu-Goto is undefined.
                # Fallback to Soft Orthogonality (Repulsion)
                # Force = (G - I) @ W
                eye = torch.eye(out_dim, device=weight.device)
                nambu_grad = torch.mm(G - eye, w_norm)

            # D. Add to main gradient
            # We assume batch-averaged magnitude for consistency
            batch_scale = grad_output.shape[0] if grad_output.shape[0] > 0 else 1.0
            grad_weight += nambu_grad * (tension_strength * batch_scale)

        alpha = conf.get('lookahead_alpha', 0.0)
        if alpha > 0.0:
            future_weight = weight - (grad_weight * alpha)
            grad_input = grad_output.matmul(future_weight)
        else:
            grad_input = grad_output.matmul(weight)
        
        grad_bias = None
        if bias is not None:
            grad_bias = grad_flat.sum(0)
            
        return grad_input, grad_weight, grad_bias, None

class TensionLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, 
                 tension=1e-7, lookahead=1e-4, noise=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        #apparently ortho is worse.
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.02)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.config = {
            'lookahead_alpha': lookahead,
            'noise': noise,
            'tension': tension  # Controls the "Surface Tension" strength
        }
        
    def forward(self, x, override_conf=None):
        # Merge config if provided
        c = self.config.copy()
        if override_conf:
            c.update(override_conf)
            
        return SurfaceTensionLinearFunc.apply(
            x, self.weight, self.bias, c
        )


class ModularProbeFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, activation_fn, config):
        ctx.save_for_backward(input, weight, bias)
        ctx.config = config
        ctx.activation_fn = activation_fn # Store the callable (e.g., F.relu)
        return F.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        activation_fn = ctx.activation_fn
        conf = ctx.config
        
        # 1. Standard Task Gradient (The boring part)
        
        # Flatten for dL/dW calculation: (Total_Tokens, Dim)
        input_flat = input.reshape(-1, input.shape[-1])
        grad_flat = grad_output.reshape(-1, grad_output.shape[-1])
        
        grad_weight = grad_flat.t().matmul(input_flat)
        
        grad_input = grad_output.matmul(weight)
        
        grad_bias = None
        if bias is not None:
            # Sum gradients across all batches and sequence positions
            grad_bias = grad_flat.sum(0)
        
        # 2. THE MODULAR PROBE
        tension = conf.get('tension', 0.0)
        
        if tension > 0.0:
            # We must turn on gradients temporarily to build the "Shadow Graph"
            # This allows PyTorch to traverse whatever activation_fn is used.
            with torch.enable_grad():
                # A. Create a detached copy of weights to trace gradients relative to THIS step
                # We need leaf tensors to call autograd.grad on them
                w_shadow = weight.detach().requires_grad_(True)
                b_shadow = bias.detach().requires_grad_(True) if bias is not None else None
                
                # B. Generate Noise
                noise = torch.randn_like(input_flat).detach()
                
                # C. Run Shadow Forward Pass (Linear + YOUR Activation)
                # This builds a tiny computational graph just for this noise batch
                shadow_linear = F.linear(noise, w_shadow, b_shadow)
                
                # --- HERE IS THE MAGIC ---
                # We call the function passed in. We don't care what it is.
                shadow_act = activation_fn(shadow_linear)
                
                # D. Calculate Orthogonality (Decorrelation)
                # Center and Normalize features
                centered = shadow_act - shadow_act.mean(0, keepdim=True)
                normed = F.normalize(centered, p=2, dim=0, eps=1e-8)
                
                # Gram Matrix (Correlations)
                gram = torch.mm(normed.t(), normed)
                
                # Loss = Distance from Identity
                eye = torch.eye(gram.shape[0], device=gram.device)
                ortho_loss = torch.norm(gram - eye, p='fro')
                
                # E. Autograd Magic
                # We ask PyTorch: "How do we change w_shadow to minimize ortho_loss?"
                # It automatically differentiates through the activation_fn.
                if grad_bias is not None:
                    grads = torch.autograd.grad(ortho_loss, [w_shadow, b_shadow], allow_unused=True)
                    d_w_probe = grads[0]
                    d_b_probe = grads[1]
                else:
                    grads = torch.autograd.grad(ortho_loss, w_shadow, allow_unused=True)
                    d_w_probe = grads[0]
                    d_b_probe = None

            # F. Inject the result into the real gradients
            if d_w_probe is not None:
                grad_weight += d_w_probe * tension
            
            if d_b_probe is not None and grad_bias is not None:
                grad_bias += d_b_probe * tension

        return grad_input, grad_weight, grad_bias, None, None

class HCLinear(nn.Module):
    def __init__(self, in_features, out_features, gain=1.0, learn_gain=False, force=False):
        super().__init__()
        if learn_gain:
            self.gain = nn.Parameter(torch.tensor(gain))
        else:
            self.register_buffer('gain', torch.tensor(gain))
        self.force = force
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_normal_(self.weight)
    def forward(self, x):
        if self.force:
            with torch.no_grad():
                w_norm = utils.range_norm(self.weight,dim=None,scale=self.gain)
                self.weight.data = w_norm
            return F.linear(x,self.weight)
        w_norm = utils.range_norm(self.weight,dim=None,scale=self.gain)
        return F.linear(x,w_norm)



class ProbingLinear(nn.Module):
    def __init__(self, in_features, out_features, activation=None, bias=True, tension=1e-6):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        if activation is None:
            def noop(x):
                return x
            activation = noop
        self.activation = activation # Pass in F.relu, F.gelu, torch.sigmoid, etc.
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_normal_(self.weight)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.config = {'tension': tension}

    def forward(self, x):
        return ModularProbeFunc.apply(x, self.weight, self.bias, self.activation, self.config)

Linear = nn.Linear
#Linear = HungryLinear

class Kattention(nn.Module):
    def __init__(self, config, h_dim=None):
        super().__init__()
        if h_dim:
            assert config.n_embd % h_dim == 0
        else:
            assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        
        self.q_attn = (Linear(config.n_embd,  config.n_embd, bias=config.bias))
        self.k_attn = (Linear(config.n_embd,  config.n_embd, bias=config.bias))
        self.v_attn = Linear(config.n_embd,  config.n_embd, bias=config.bias)
        
        # output projection
        self.c_proj = Linear(config.n_embd,  config.n_embd, bias=config.bias)
        
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_embd = config.n_embd
        if h_dim:
            self.n_head = self.n_embd // h_dim
        else:
            self.n_embd = config.n_head
        self.dropout = config.dropout
        self.head_dim =  self.n_embd // self.n_head 

        self.k_sparse=4

        self.qm = nn.Parameter(torch.ones(self.head_dim).normal_(mean=1, std=0.1))
        
    def forward(self, x, causal=True):
        B, T, C = x.size() 
        q = self.q_attn(x)
        k = self.k_attn(x)
        v = self.v_attn(x)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)

        k_val = min(self.k_sparse, T)

        att = (q*(math.log(T)*self.qm) @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(causal_mask == 0, float('-inf'))
        k_hot_mask = quantizer.TopKHot.apply(att,k_val)
        att = att.masked_fill(k_hot_mask == 0, float('-inf'))
        #att = att*k_hot_mask
        #att = att.masked_fill(causal_mask == 0, 0.0)
        #att = att.masked_fill(k_hot_mask == 0, 0.0)
        #att = F.tanh(att) #/ k_val #utils.k_from_hot(k_hot_mask).detach()
        #att = F.relu(att)
        #att = att / att.sum(dim =-1, keepdim=True)
        att = torch.softmax(att,dim=-1) 
        #att = F.relu(att)
        #sum_att = att.sum(dim=-1, keepdim=True)
        #att = att / (sum_att + 1e-8)
        
        y = att@v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        
        return y


class HydraKattention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.bias = config.bias

        self.head_dim = 16
        self.n_head = self.n_embd // self.head_dim 
        
        self.num_kv_heads = getattr(config, "num_kv_heads", self.n_head // 4)
        self.num_queries_per_kv = self.n_head // self.num_kv_heads
        
        self.q_dim = self.n_head * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim

        self.q_attn = nn.Linear(self.n_embd, self.q_dim, bias=self.bias)
        self.k_attn = nn.Linear(self.n_embd, self.kv_dim, bias=self.bias)
        self.v_attn = nn.Linear(self.n_embd, self.kv_dim, bias=self.bias)
        
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=self.bias)
        
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        self.k_sparse = 4
        self.qm = nn.Parameter(torch.ones(self.head_dim).normal_(mean=1, std=0.1))

        self.fattn = SparseFlexAttention(32)
        
    def forward(self, x, causal=True):
        B, T, C = x.size() 
        
        q = self.q_attn(x)
        k = self.k_attn(x)
        v = self.v_attn(x)
        
        q = q.view(B, T, self.num_kv_heads, self.num_queries_per_kv, self.head_dim).permute(0, 2, 3, 1, 4)
        
        k = k.view(B, T, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3).unsqueeze(2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3).unsqueeze(2)

        k_val = min(self.k_sparse, T)

        att = (q * (math.log(T) * self.qm) @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
    
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, 1, T, T)
        att = att.masked_fill(causal_mask == 0, float('-inf'))
        
        k_hot_mask = quantizer.TopKHot.apply(att, k_val)
        att = att.masked_fill(k_hot_mask == 0, float('-inf'))

        att = torch.softmax(att, dim=-1) 
        att = self.attn_dropout(att)
        
        y = att @ v

        y = y.permute(0, 3, 1, 2, 4).contiguous().view(B, T, C)
        
        y = self.resid_dropout(self.c_proj(y))
        
        return y


class SparseFlexAttention(nn.Module):
    def __init__(self, block_size=64):
        super().__init__()
        self.block_size = block_size

    def forward(self, q, k, v, k_val):
        B, H, T, D = q.shape
        num_blocks = T // self.block_size

        # 1. Compute a downsampled, coarse logit map for block selection
        # Instead of T x T, we compute (T/64) x (T/64)
        q_coarse = q.view(B, H, num_blocks, self.block_size, D).mean(dim=-2)
        k_coarse = k.view(B, H, num_blocks, self.block_size, D).mean(dim=-2)
        coarse_logits = torch.matmul(q_coarse, k_coarse.transpose(-1, -2))

        # 2. Use your custom autograd TopKHot on the coarse logits
        # This keeps the Straight-Through gradients intact!
        block_sparsity_mask, _ = quantizer.vmapTopKHot.apply(coarse_logits, k_val)

        # 3. Create a mask_mod using the coarse predictions
        def custom_mask_mod(b, h, q_idx, kv_idx):
            q_block = q_idx // self.block_size
            kv_block = kv_idx // self.block_size
            # Direct pointwise read from our learned coarse mask
            return (q_idx >= kv_idx) & (block_sparsity_mask[b, h, q_block, kv_block] == 1.0)

        # 4. Generate the optimized block mask for PyTorch's Triton backend
        block_mask = create_block_mask(
            custom_mask_mod, B=B, H=H, Q_LEN=T, KV_LEN=T, BLOCK_SIZE=self.block_size, _compile=True
        )

        # 5. Execute heavy attention computation only on valid blocks
        out = flex_attention(q, k, v, block_mask=block_mask)
        return out


class FKattention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.bias = config.bias

        # 16 is a clean dimension for Tensor Cores
        self.head_dim = 16
        self.n_head = self.n_embd // self.head_dim 
        
        # In GQA, the number of Q heads is a multiple of KV heads
        self.num_kv_heads = getattr(config, "num_kv_heads", self.n_head // 4)
        self.num_queries_per_kv = self.n_head // self.num_kv_heads
        
        self.q_dim = self.n_head * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim

        self.q_attn = nn.Linear(self.n_embd, self.q_dim, bias=self.bias)
        self.k_attn = nn.Linear(self.n_embd, self.kv_dim, bias=self.bias)
        self.v_attn = nn.Linear(self.n_embd, self.kv_dim, bias=self.bias)
        
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=self.bias)
        
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        self.k_sparse = 4
        self.qm = nn.Parameter(torch.ones(self.head_dim).normal_(mean=1, std=0.1))

        self.fattn = SparseFlexAttention(block_size=32)
        
    def forward(self, x, causal=True):
        B, T, C = x.size() 
        
        k_val = min(self.k_sparse, T // self.fattn.block_size)
        k_val = max(1, k_val) # Clamp to at least 1 block

        q = self.q_attn(x)
        k = self.k_attn(x)
        v = self.v_attn(x)
        
        # Reshape to standard 4D layout for Multi-Head / Grouped-Query Attention
        # Shape: (B, H, T, D)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply learnable scaling parameter `qm` directly on Q
        q = q * (math.log(T) * self.qm.view(1, 1, 1, self.head_dim))

        # Repeat K and V across the query groups to match Q's head dimension 
        # (This expands the KV heads to match the Q head count for flex_attention)
        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        # Execute block sparse flex attention
        y = self.fattn(q, k, v, k_val)
        
        # Reshape back to original 3D sequence format
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        
        return y

def tokenformer_theta(scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
    n = scores.shape[dim]
    l2 = torch.linalg.norm(scores, ord=2, dim=dim, keepdim=True).clamp(min=1e-8)
    scaled = scores * math.sqrt(n) / l2
    return F.gelu(scaled)

class PattLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_tokens: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_tokens = num_tokens

        self.key_tokens = nn.Parameter(torch.empty(num_tokens, input_dim))
        self.val_tokens = nn.Parameter(torch.empty(num_tokens, output_dim))

        nn.init.kaiming_uniform_(self.key_tokens, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.val_tokens, a=math.sqrt(5))

    def forward(self, x: torch.Tensor, get_scores=False) -> torch.Tensor:
        scores = x @ self.key_tokens.T #self.key_tokens.T
        attn = tokenformer_theta(scores, dim=-1)
        x_out= attn @ self.val_tokens
        if get_scores:#TODO gatherbygate
            return x_out, scores
        return x_out

    def add_tokens(self, n: int):
        device = self.key_tokens.device
        dtype = self.key_tokens.dtype

        new_keys = torch.zeros(n, self.input_dim, device=device, dtype=dtype)
        new_vals = torch.empty(n, self.output_dim, device=device, dtype=dtype)
        nn.init.kaiming_uniform_(new_vals, a=math.sqrt(5))

        self.key_tokens = nn.Parameter(torch.cat([self.key_tokens.data, new_keys], dim=0))
        self.val_tokens = nn.Parameter(torch.cat([self.val_tokens.data, new_vals], dim=0))
        self.num_tokens += n

    def remove_tokens(self, n: int):
        """Remove the last n parameter tokens."""
        if n >= self.num_tokens:
            raise ValueError(f"Cannot remove {n} tokens, only have {self.num_tokens}")

        self.key_tokens = nn.Parameter(self.key_tokens.data[:-n])
        self.val_tokens = nn.Parameter(self.val_tokens.data[:-n])
        self.num_tokens -= n


class HungryApply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, fn, config):
        ctx.fn = fn
        ctx.config = config
        
        # We execute the forward pass in a no_grad block because we will 
        # manually compute the exact gradients in the backward pass.
        with torch.no_grad():
            out = fn(x, weight)
            
        ctx.save_for_backward(x, weight)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        fn = ctx.fn
        conf = ctx.config
        alpha = conf.get('lookahead_alpha', 0.001)

        # 1. Noise injection (regularizing the weight gradient)
        if conf.get('noise', False):
            x_magnitude = x.abs().sum()
            noise_x = torch.randn_like(x) / (1.0 + x_magnitude)
            scale = conf.get('noise_scale', 0.01)
            noise_x2 = torch.randn_like(x) * grad_output.std() * scale
            x_noisy = x + noise_x + noise_x2
        else:
            x_noisy = x

        # 2. Compute grad_weight on the noisy input
        with torch.enable_grad():
            x_g = x_noisy.detach()
            weight_g = weight.detach().requires_grad_(True)
            out_g = fn(x_g, weight_g)
            grad_weight, = torch.autograd.grad(
                out_g, weight_g, grad_output, 
                retain_graph=False, allow_unused=True
            )

        if grad_weight is None:
            grad_weight = torch.zeros_like(weight)

        # 3. Lookahead update for the weights
        g_w = grad_weight
        if conf.get('g_norm', True) or conf.get('gnorm', True):
            g_w = utils.rms_norm(g_w) # Assuming utils.rms_norm is globally defined
        future_weight = weight - (g_w * alpha)

        # 4. Compute grad_x using the lookahead weight
        with torch.enable_grad():
            x_g2 = x.detach().requires_grad_(True)
            weight_g2 = future_weight.detach()
            out_g2 = fn(x_g2, weight_g2)
            grad_x, = torch.autograd.grad(
                out_g2, x_g2, grad_output, 
                retain_graph=False, allow_unused=True
            )

        if grad_x is None:
            grad_x = torch.zeros_like(x)

        return grad_x, grad_weight, None, None


class HungryPattLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_tokens: int, lookahead=1e-5, noise=True, zero_init=True, g_norm=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_tokens = num_tokens

        # Key projection: weight shape is [num_tokens, input_dim]
        self.linear1 = HungryLinear(
            in_features =input_dim,
            out_features=num_tokens,
            bias        =False,
            lookahead   =lookahead,
            noise       =noise,
            zero_init   =zero_init,
            g_norm      =g_norm
        )

        # Value projection: weight shape is [output_dim, num_tokens]
        self.linear2 = HungryLinear(
            in_features =num_tokens,
            out_features=output_dim,
            bias        =False,
            lookahead   =lookahead,
            noise       =noise,
            zero_init   =zero_init,
            g_norm      =g_norm
        )

    def forward(self, x: torch.Tensor, get_scores=False, conf=None) -> torch.Tensor:
        scores = self.linear1(x, conf=conf)
        #attn = tokenformer_theta(scores, dim=-1)
        attn = scores * quantizer.topkhotSTE(scores, 4)
        attn = attn / attn.sum(keepdim=True, dim=-1)
        attn = utils.dead_rat_relu(attn)
        #attn = tokenformer_theta(attn, dim=-1)
        x_out = self.linear2(attn, conf=conf)
        
        if get_scores:
            return x_out, scores
        return x_out

    def add_tokens(self, n: int, optimizer=None):
        # Cap additions so that self.num_tokens + n <= 512
        allowed_add = 512 - self.num_tokens
        if allowed_add <= 0:
            return  # Already at max capacity
            
        n = min(n, allowed_add)
        if n <= 0:
            return

        device = self.linear1.weight.device
        dtype = self.linear1.weight.dtype

        old_w1 = self.linear1.weight
        old_w2 = self.linear2.weight

        # 1. Grow key projection weights (dim 0)
        new_keys = torch.zeros(n, self.input_dim, device=device, dtype=dtype)
        self.linear1.weight = nn.Parameter(torch.cat([old_w1.data, new_keys], dim=0))

        # 2. Grow value projection weights (dim 1)
        new_vals = torch.empty(self.output_dim, n, device=device, dtype=dtype)
        nn.init.kaiming_uniform_(new_vals, a=math.sqrt(5))
        self.linear2.weight = nn.Parameter(torch.cat([old_w2.data, new_vals], dim=1))

        self.num_tokens += n

        # 3. Perform surgery on the optimizer's state if provided
        if optimizer is not None:
            # For linear1, append rows of zeros to match dim 0 growth
            self._update_optimizer_state(
                optimizer, 
                old_w1, self.linear1.weight, 
                lambda t: torch.cat([t, torch.zeros(n, t.shape[1], device=device, dtype=dtype)], dim=0)
            )
            # For linear2, append columns of zeros to match dim 1 growth
            self._update_optimizer_state(
                optimizer, 
                old_w2, self.linear2.weight, 
                lambda t: torch.cat([t, torch.zeros(t.shape[0], n, device=device, dtype=dtype)], dim=1)
            )

    def remove_tokens(self, indices, optimizer=None):
        """
        Removes specified token indices from the active parameter set.
        
        Args:
            indices: An integer, a list/tuple of integers, or a PyTorch tensor 
                     specifying which token index/indices to remove.
            optimizer: Optional PyTorch optimizer instance to update corresponding states.
        """
        # Convert indices input into a unified python set
        if isinstance(indices, int):
            indices_to_remove = {indices}
        elif isinstance(indices, (list, tuple)):
            indices_to_remove = set(indices)
        elif torch.is_tensor(indices):
            indices_to_remove = set(indices.tolist())
        else:
            raise TypeError("indices must be an int, a list/tuple of ints, or a torch.Tensor")

        # Validate that indices to remove fall within actual parameter bounds
        for idx in indices_to_remove:
            if idx < 0 or idx >= self.num_tokens:
                raise IndexError(f"Token index {idx} is out of bounds for {self.num_tokens} tokens")

        # Determine indices to retain
        keep_indices = [i for i in range(self.num_tokens) if i not in indices_to_remove]
        
        # Prevent shrinking below 1 token, as zero-sized weights break linear layer operations
        if len(keep_indices) == 0:
            raise ValueError("Cannot remove all tokens; at least 1 token must remain in the layer.")

        old_w1 = self.linear1.weight
        old_w2 = self.linear2.weight

        # 1. Index-based slicing of the weights
        self.linear1.weight = nn.Parameter(old_w1.data[keep_indices])
        self.linear2.weight = nn.Parameter(old_w2.data[:, keep_indices])

        self.num_tokens = len(keep_indices)

        # 2. Perform surgery on the optimizer's state if provided
        if optimizer is not None:
            # Slice key projection states (dim 0)
            self._update_optimizer_state(
                optimizer, old_w1, self.linear1.weight, 
                lambda t: t[keep_indices]
            )
            # Slice value projection states (dim 1)
            self._update_optimizer_state(
                optimizer, old_w2, self.linear2.weight, 
                lambda t: t[:, keep_indices]
            )

    def _update_optimizer_state(self, optimizer, old_param, new_param, resize_fn):
        """Swaps parameter references in optimizer and resizes internal AdamW states."""
        for group in optimizer.param_groups:
            found_idx = None
            for idx, p in enumerate(group['params']):
                if p is old_param:
                    found_idx = idx
                    break

            if found_idx is not None:
                group['params'][found_idx] = new_param
                state_key = None
                for key in list(optimizer.state.keys()):
                    if key is old_param:
                        state_key = key
                        break

                if state_key is not None:
                    state = optimizer.state.pop(state_key)
                    if 'exp_avg' in state:
                        state['exp_avg'] = resize_fn(state['exp_avg'])
                    if 'exp_avg_sq' in state:
                        state['exp_avg_sq'] = resize_fn(state['exp_avg_sq'])
                    optimizer.state[new_param] = state
                break

class MHKPattLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_tokens: int, heads = 6):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_tokens = num_tokens
        
        self.heads = heads
        self.key_head_dim = input_dim // heads
        self.val_head_dim = output_dim // heads
        self.key_tokens = nn.Parameter(torch.empty(heads, num_tokens, self.key_head_dim))
        self.val_tokens = nn.Parameter(torch.empty(heads, num_tokens, self.val_head_dim))
        #self.key_tokens = nn.Parameter(torch.empty(num_tokens, input_dim))
        #self.val_tokens = nn.Parameter(torch.empty(num_tokens, output_dim))

        nn.init.kaiming_uniform_(self.key_tokens, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.val_tokens, a=math.sqrt(5))

    def forward(self, x):
        B,T,C=x.shape
        # self.key_tokens: [Heads, Num_Tokens, Head_Dim]
        
        # 1. Project Input to Queries (Multi-Head)
        q = x.view(B, T, self.heads, self.key_head_dim).transpose(1, 2) 

        scores = q @ self.key_tokens.transpose(-2, -1) 

       # scores = scores * (1.0 / math.sqrt(self.key_head_dim))
        #scores = F.tanh(scores)
        scores = tokenformer_theta(scores, dim=-1)
        k_hot_mask = quantizer.ThresHot.apply(scores)
        scores = scores.masked_fill(k_hot_mask == 0, 0)
        #scores = scores / utils.k_from_hot(k_hot_mask,scores.shape[-1]).detach()
        #scores = tokenformer_theta(scores, dim=-1)
        #scores = torch.softmax(scores, dim=-1)

        out = scores @ self.val_tokens 
        return out.transpose(1, 2).reshape(B, T, -1)


class Pattention(nn.Module):
    def __init__(self, config, pt_count):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        
        #self.q_attn = PattLinear(config.n_embd,  config.n_embd, pt_count)
        #self.k_attn = PattLinear(config.n_embd,  config.n_embd, pt_count)
        
        self.q_attn = Linear(config.n_embd,  config.n_embd, bias=False)
        self.k_attn = Linear(config.n_embd,  config.n_embd, bias=False)
        self.v_attn = PattLayer(config.n_embd,  config.n_embd, pt_count)
        
        # output projection
        #self.c_proj = PattLinear(config.n_embd,  config.n_embd, pt_count)
        self.c_proj = Linear(config.n_embd,  config.n_embd, bias=False)
        
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head 
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim =  self.n_embd //  self.n_head 

        self.k_sparse=4

        self.qm = nn.Parameter(torch.ones(self.head_dim).normal_(mean=1, std=0.1))
        
    def forward(self, x, causal=True):
        B, T, C = x.size() 
        q = self.q_attn(x)
        k = self.k_attn(x)
        v = self.v_attn(x)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)

        #k_val = min(self.k_sparse, T)
#
        #att = (q*(math.log(T)*self.qm) @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    #
        #causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        #att = att.masked_fill(causal_mask == 0, float('-inf'))
        #k_hot_mask = quantizer.avgHot.apply(att)#, k_val)
        #att = att.masked_fill(k_hot_mask == 0, float('-inf'))
        #
        #
        ##att = F.tanh(att) #/ utils.k_from_hot(k_hot_mask).detach()
        #att = torch.softmax(att,dim=-1) 
        #y = att@v
        y = torch.nn.functional.scaled_dot_product_attention(
            q * (math.log(T)*self.qm),
            k, v, 
            attn_mask=None, dropout_p=self.dropout if self.training else 0,
            is_causal= causal)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        
        return y

class PerWeightActivationLayer(nn.Module):
    
    def __init__(self, in_features: int, out_features: int, activation=F.gelu, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
      
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.linear.weight.data.uniform_(-1, 1)
        self.linear.weight.data *= self.linear.weight.data 
        self.linear.weight.data *= self.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        weight = self.linear.weight 
        x_expanded = x.unsqueeze(-2) 
        
        weighted_inputs = x_expanded * weight
        
        activated_weighted_inputs = self.activation(weighted_inputs)
        
        # Sum over the last dimension (the 'In' dimension)
        summed_output = torch.sum(activated_weighted_inputs, dim=-1) #/ (self.in_features ** 0.5)
        
        if self.linear.bias is not None:
            summed_output += self.linear.bias
            
        return summed_output * (self.in_features ** -0.5)


class WonkyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation=torch.sin, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
      
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.bias = nn.Parameter(torch.randn_like(self.linear.weight))
        
        self.bias.data.uniform_(-0.01, 0.01) 
        self.linear.weight.data.uniform_(-0.1, 0.1) 

    def forward(self, x):
        linear_term = ( self.linear(x))
        
        ac_in = self.activation(x)
        ac_weight = (self.activation(self.linear.weight) + self.bias)
        
        ac_term = utils.gaussian_bump(F.linear(ac_in, ac_weight))
        return utils.gaussian_bump(linear_term - ac_term) 
    
class FakeSinLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear.weight.data.uniform_(-0.1, 0.1) 

    def forward(self, x):
        linear_term = self.linear(x)
        
        cubic_term = F.linear(x.pow(3), self.linear.weight.pow(3)) / 6.0
        
        return (linear_term - cubic_term) 
    

class SSM(nn.Module):
    def __init__(self, n_embd, dropout, bias):
        super().__init__()

        self.q_attn =  Linear(n_embd, n_embd, bias=bias)
        self.v_attn =  Linear(n_embd, n_embd, bias=bias)
        
        self.c_proj = Linear(n_embd,  n_embd, bias=bias)

        self.identity = nn.Parameter(torch.rand(n_embd))
        
        self.attn_dropout = nn.Dropout(dropout) #todo
        self.resid_dropout = nn.Dropout(dropout) #todo
        self.n_embd = n_embd
        self.dropout = dropout
        self.scanner = ProcrustesButterflyLinear(n_embd)
        #self.shl_lh = SinkhornLinear(n_embd)
        #self.shl_rh = SinkhornLinear(n_embd)
        self.shl_lh = ProcrustesLinear(n_embd,n_embd)
        self.shl_rh = ProcrustesLinear(n_embd,n_embd)
        #self.patt   = PattLayer(n_embd, n_embd, 256)#TODO
        with torch.no_grad():
            self.shl_lh.weight.data = (torch.eye(n_embd))
            self.shl_rh.weight.data = (torch.eye(n_embd))
    
    def get_weight(self):
        return self.scanner.compute_orthogonal_weights()

    def forward(self, x: torch.tensor, causal=True, get_q=False, weight = None,): #same signature to allow easy swapping.
        #B, T, C = x.size() 
        q = self.q_attn(x) 
        v = self.v_attn(x)
        if weight is None:
            weight = self.scanner.compute_orthogonal_weights()
        #with torch.no_grad():
        #    self.shl_lh.project()
        #    self.shl_rh.project()
        def scan(left, right):
            #z = self.shl_lh(left) 
            #z = z + self.shl_rh(right)
            #z = utils.rms(z,dim=-1)

            z = self.scanner(left,right,weight).to(x.dtype)
            z = utils.rms_norm(z)
            return z
        q = utils.pscan2(q, scan, self.identity)
        
        Y = q*v 
        Y = self.c_proj(Y)
        Y = self.resid_dropout(Y)
        if get_q:
            return Y, q
        return Y
    
    def nexts(self, prev: torch.tensor, x: torch.tensor,  causal=False, weight = None): #same signature to allow easy swapping.
       # B, T, C = x.size() 
        q = self.q_attn(x) 
        v = self.v_attn(x)
        if weight is None:
            weight = self.scanner.compute_orthogonal_weights()
        def scan(left, right):
            z = self.scanner(left,right,weight).to(x.dtype)
            return utils.rms_norm(z)
        q = scan(prev,q)
        
        Y = q*v 
        Y = self.c_proj(Y)
        Y = self.resid_dropout(Y)
        return Y, q

class AssociativeSSM(nn.Module):
    def __init__(self, n_embd, dropout, bias):
        super().__init__()

        self.q_attn = nn.Linear(n_embd, n_embd, bias=bias)
        self.v_attn = nn.Linear(n_embd, n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)

        self.identity = nn.Parameter(torch.rand(n_embd))
        self.resid_dropout = nn.Dropout(dropout)
        self.n_embd = n_embd
        
        # Orthogonal projections act on the sequence-wide inputs/outputs
        self.input_proj = ProcrustesLinear(n_embd, n_embd)
        self.output_proj = ProcrustesLinear(n_embd, n_embd)
        
        # Learnable diagonal decay
        self.log_decay = nn.Parameter(torch.empty(n_embd))
        nn.init.uniform_(self.log_decay, 2.2, 4.6)
        
    def get_weight(self):
        return self.input_proj.compute_orthogonal_weight(), self.output_proj.compute_orthogonal_weight()

    def forward(self, x: torch.Tensor, causal=True, get_q=False, weight=None):
        q = self.q_attn(x) 
        v = self.v_attn(x)
        
        if weight is None:
            w_in, w_out = self.get_weight()
        else:
            w_in, w_out = weight
            
        # 1. Project inputs using the first Procrustes matrix
        q_projected = self.input_proj(q, w_in)
        
        # 2. Extract stable diagonal decay and input scaling vectors
        D = torch.sigmoid(self.log_decay).view(1, 1, -1)
        input_scale = torch.sqrt(1.0 - D**2 + 1e-7)
        
        # 3. Define a strictly associative diagonal prefix scan
        # Tuple composition: (A_L, B_L) \otimes (A_R, B_R) = (A_R * A_L, A_R * B_L + B_R)
        def associative_scan_func(left_tuple, right_tuple):
            a_left, b_left = left_tuple
            a_right, b_right = right_tuple
            return (a_right * a_left, a_right * b_left + b_right)
        
        # Prepare scan inputs
        # Each step t is represented by the tuple (D_t, Input_t)
        # We expand D to match the batch and sequence dimensions
        B, T, C = x.shape
        decay_expanded = D.expand(B, T, C)
        scaled_input = input_scale * q_projected
        
        # Pack into a combined tensor to scan over
        # Shape: (B, T, 2 * C) to keep pscan2's single tensor signature
        scan_input = torch.cat([decay_expanded, scaled_input], dim=-1)
        
        def scan_adapter(left, right):
            # Split back into decay (A) and state (B) components
            a_l, b_l = left.chunk(2, dim=-1)
            a_r, b_r = right.chunk(2, dim=-1)
            
            a_out, b_out = associative_scan_func((a_l, b_l), (a_r, b_r))
            return torch.cat([a_out, b_out], dim=-1)
            
        # We run the parallel scan over the packed associative tuples
        init_packed = torch.cat([torch.ones_like(self.identity), self.identity], dim=-1)
        scanned_packed = utils.pscan2(scan_input, scan_adapter, init_packed)
        
        # Extract the accumulated state component (last C elements)
        _, q_scanned = scanned_packed.chunk(2, dim=-1)
        
        # 4. Project outputs using the second Procrustes matrix and apply RMSNorm
        q_out = self.output_proj(q_scanned, w_out)
        q_out = utils.rms_norm(q_out)
        
        Y = q_out * v 
        Y = self.c_proj(Y)
        Y = self.resid_dropout(Y)
        
        if get_q:
            return Y, q_out
        return Y
    
    def nexts(self, prev: torch.Tensor, x: torch.Tensor, causal=False, weight=None):
        q = self.q_attn(x) 
        v = self.v_attn(x)
        
        if weight is None:
            w_in, w_out = self.get_weight()
        else:
            w_in, w_out = weight
            
        D = torch.sigmoid(self.log_decay)
        input_scale = torch.sqrt(1.0 - D**2 + 1e-7)
        
        # Sequential step matches the parallel scan exactly
        q_projected = self.input_proj(q, w_in)
        q_scanned = D * prev + input_scale * q_projected
        
        q_out = self.output_proj(q_scanned, w_out)
        q_out = utils.rms_norm(q_out)
        
        Y = q_out * v 
        Y = self.c_proj(Y)
        Y = self.resid_dropout(Y)
        return Y, q_scanned



class KreinSSM(nn.Module):
    def __init__(self, n_embd, dropout, bias, delta=1e-4):
        super().__init__()

        self.q_attn = nn.Linear(n_embd, n_embd, bias=bias)
        self.v_attn = nn.Linear(n_embd, n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)

        # De-ghosted learnable initial state for the reconstructed difference channel
        self.identity = nn.Parameter(torch.empty(n_embd))
        nn.init.uniform_(self.identity, -0.01, 0.01)
        
        self.resid_dropout = nn.Dropout(dropout)
        self.n_embd = n_embd
        self.delta = delta
        
        # Orthogonal projections
        self.input_proj = ProcrustesLinear(n_embd, n_embd)
        self.output_proj = ProcrustesLinear(n_embd, n_embd)
        
        # Enforced Temporal Symmetry
        self.damping = nn.Parameter(torch.empty(n_embd))
        nn.init.uniform_(self.damping, 2.2, 4.6)
        
        # Independent rapidities for asymmetric write-scaling
        self.log_decay_plus = nn.Parameter(torch.empty(n_embd))
        self.log_decay_minus = nn.Parameter(torch.empty(n_embd))
        nn.init.uniform_(self.log_decay_plus, 0.25, 0.35)
        nn.init.uniform_(self.log_decay_minus, 0.25, 0.35)

    def get_weight(self):
        return self.input_proj.compute_orthogonal_weight(), self.output_proj.compute_orthogonal_weight()

    def forward(self, x: torch.Tensor, causal=True, get_q=False, weight=None):
        q = self.q_attn(x) 
        v = self.v_attn(x)
        
        if weight is None:
            w_in, w_out = self.get_weight()
        else:
            w_in, w_out = weight
            
        # Project first 
        q_projected = self.input_proj(q, w_in)
        
        # Split second
        q_reg = torch.sqrt(q_projected**2 + self.delta)
        q_plus = 0.5 * (q_reg + q_projected)
        q_minus = 0.5 * (q_reg - q_projected)
        
        # Share the decay rate D for both tracks to guarantee temporal coherence
        D = torch.sigmoid(self.damping).view(1, 1, -1)
        D_plus = D
        D_minus = D
        
        # Variance-Preserving Envelope-Coupling
        base_scale = torch.sqrt(1.0 - D**2 + 1e-7)
        input_scale_plus = base_scale * torch.tanh(self.log_decay_plus).view(1, 1, -1)
        input_scale_minus = base_scale * torch.tanh(self.log_decay_minus).view(1, 1, -1)
        
        # Dual-channel associative scan
        B, T, C = x.shape
        decay_all = torch.cat([D_plus.expand(B, T, C), D_minus.expand(B, T, C)], dim=0)
        scaled_input = torch.cat([
            input_scale_plus * q_plus, 
            input_scale_minus * q_minus
        ], dim=0)
        
        scan_input_all = torch.cat([decay_all, scaled_input], dim=-1)
        
        def associative_scan_func(left_tuple, right_tuple):
            a_left, b_left = left_tuple
            a_right, b_right = right_tuple
            return (a_right * a_left, a_right * b_left + b_right)

        def scan_adapter(left, right):
            a_l, b_l = left.chunk(2, dim=-1)
            a_r, b_r = right.chunk(2, dim=-1)
            a_out, b_out = associative_scan_func((a_l, b_l), (a_r, b_r))
            return torch.cat([a_out, b_out], dim=-1)
            
        # Zero-initialized scan state to ensure 1D init_packed shape (prevents view/expand crash)
        init_packed = torch.cat([torch.ones_like(self.damping), torch.zeros_like(self.damping)], dim=-1)
        scanned_packed_all = utils.pscan2(scan_input_all, scan_adapter, init_packed)
        
        _, q_scanned_all = scanned_packed_all.chunk(2, dim=-1)
        q_scanned_plus, q_scanned_minus = q_scanned_all.chunk(2, dim=0)
        
        # Indefinite Metric Reconstruction
        q_scanned = q_scanned_plus - q_scanned_minus
        
        # Analytically apply the decayed initial state (identical to non-zero scan initialization)
        decay_expanded = D.expand(B, T, C)
        decay_cumulative = torch.cumprod(decay_expanded, dim=1)
        initial_decayed = decay_cumulative * self.identity.view(1, 1, -1)
        q_scanned = q_scanned + initial_decayed
        
        q_out = self.output_proj(q_scanned, w_out)
        q_out = utils.rms_norm(q_out)
        
        Y = q_out * v 
        Y = self.c_proj(Y)
        Y = self.resid_dropout(Y)
        
        if get_q:
            return Y, q_out
        return Y

class KreinSSM2(nn.Module):
    def __init__(self, n_embd, dropout, bias, delta=1e-4):
        super().__init__()

        self.q_attn = nn.Linear(n_embd, n_embd, bias=bias)
        self.v_attn = nn.Linear(n_embd, n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)

        # Zero-initialized to prevent transient drift in split channels
        self.identity = nn.Parameter(torch.zeros(n_embd))
        self.resid_dropout = nn.Dropout(dropout)
        self.n_embd = n_embd
        self.delta = delta
        
        # Orthogonal projections
        self.input_proj = ProcrustesLinear(n_embd, n_embd)
        self.output_proj = ProcrustesLinear(n_embd, n_embd)
        
        # Unified Hyperbolic Coupling (Energy Preserving)
        # Initialized to match the memory horizon of ASSM [0.90, 0.99]
        self.rapidity_plus = nn.Parameter(torch.empty(n_embd))
        self.rapidity_minus = nn.Parameter(torch.empty(n_embd))
        nn.init.uniform_(self.rapidity_plus, 0.15, 0.45)
        nn.init.uniform_(self.rapidity_minus, 0.15, 0.45)

    def get_weight(self):
        return self.input_proj.compute_orthogonal_weight(), self.output_proj.compute_orthogonal_weight()

    def forward(self, x: torch.Tensor, causal=True, get_q=False, weight=None):
        q = self.q_attn(x) 
        v = self.v_attn(x)
        
        if weight is None:
            w_in, w_out = self.get_weight()
        else:
            w_in, w_out = weight
            
        # 1. Project input first (matches ASSM exactly)
        q_projected = self.input_proj(q, w_in)
        
        # 2. Smooth absolute split *after* projection (guarantees sign purity & coordinate symmetry)
        q_reg = torch.sqrt(q_projected**2 + self.delta)
        q_proj_plus = 0.5 * (q_reg + q_projected)
        q_proj_minus = 0.5 * (q_reg - q_projected)
        
        # 3. Energy-preserving hyperbolic coupling using rapidities
        theta_plus = torch.clamp(self.rapidity_plus.abs(), min=1e-5).view(1, 1, -1)
        theta_minus = torch.clamp(self.rapidity_minus.abs(), min=1e-5).view(1, 1, -1)
        
        D_plus = 1.0 / torch.cosh(theta_plus)
        D_minus = 1.0 / torch.cosh(theta_minus)
        
        input_scale_plus = torch.tanh(theta_plus)
        input_scale_minus = torch.tanh(theta_minus)
        
        # 4. Dual-channel associative scan
        B, T, C = x.shape
        decay_all = torch.cat([D_plus.expand(B, T, C), D_minus.expand(B, T, C)], dim=0)
        scaled_input = torch.cat([
            input_scale_plus * q_proj_plus, 
            input_scale_minus * q_proj_minus
        ], dim=0)
        
        scan_input_all = torch.cat([decay_all, scaled_input], dim=-1)
        
        def associative_scan_func(left_tuple, right_tuple):
            a_left, b_left = left_tuple
            a_right, b_right = right_tuple
            return (a_right * a_left, a_right * b_left + b_right)

        def scan_adapter(left, right):
            a_l, b_l = left.chunk(2, dim=-1)
            a_r, b_r = right.chunk(2, dim=-1)
            a_out, b_out = associative_scan_func((a_l, b_l), (a_r, b_r))
            return torch.cat([a_out, b_out], dim=-1)
            
        # Initialize with ones for decay multipliers, and zero state to prevent drift
        init_packed = torch.cat([torch.ones_like(self.identity), torch.zeros_like(self.identity)], dim=-1)
        scanned_packed_all = utils.pscan2(scan_input_all, scan_adapter, init_packed)
        
        _, q_scanned_all = scanned_packed_all.chunk(2, dim=-1)
        q_scanned_plus, q_scanned_minus = q_scanned_all.chunk(2, dim=0)
        
        # 5. Indefinite Metric Reconstruction
        q_scanned = q_scanned_plus - q_scanned_minus
        
        # 6. Output mapping & normalization
        q_out = self.output_proj(q_scanned, w_out)
        q_out = utils.rms_norm(q_out)
        
        Y = q_out * v 
        Y = self.c_proj(Y)
        Y = self.resid_dropout(Y)
        
        if get_q:
            return Y, q_out
        return Y

class CrossAttention(nn.Module):
    def __init__(self, config):
        n_embd = config.n_embd
        n_head = config.n_head
        dropout = config.dropout
        bias = config.bias
        super().__init__()
        assert n_embd % n_head == 0
        self.q_proj = Linear(n_embd, n_embd, bias=bias)
        self.k_proj = Linear(n_embd, n_embd, bias=bias)
        self.v_proj = Linear(n_embd, n_embd, bias=bias)
      
        self.c_proj = Linear(n_embd,  n_embd, bias=bias)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head 
        self.n_embd = n_embd
        self.dropout = dropout
        self.head_dim =  self.n_embd //  self.n_head 
        
        self.qm = nn.Parameter(torch.ones(self.head_dim).normal_(mean=1, std=0.1))
        
    def _create_mask(self, T_q, T_kv, device):
        scaling_factor = T_kv / T_q
        query_indices = torch.arange(T_q, device=device)
        key_indices = torch.arange(T_kv, device=device)
        max_key_indices = ((query_indices + 1) * scaling_factor).floor() - 1
        mask = key_indices[None, :] > max_key_indices[:, None]
        return mask
    
    def forward(self, x_kv, x_q, causal = False,conf=None):
        B, T, C = x_kv.size() 
        sB, sT, sC = x_q.size() 
        if isinstance(Linear, HungryLinear):
            q = self.q_proj(x_q,conf)
            k = self.k_proj(x_kv,conf)
            v = self.v_proj(x_kv,conf)
        else:
            q = self.q_proj(x_q)
            k = self.k_proj(x_kv)#TODO: beware
            v = self.v_proj(x_kv)
        q = q.view(B, sT, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, sT, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        attn_mask=None
        if causal:
            if sT != T:
                attn_mask = self._create_mask(sT, T, x_kv.device)
                causal = False

        y = torch.nn.functional.scaled_dot_product_attention(
            q * (math.log(T)*self.qm),
            k, v, 
            attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0,
            is_causal= causal)
        
        y = y.transpose(1, 2).contiguous().view(B, sT, C) 
        
        if isinstance(Linear, HungryLinear):
            y = self.c_proj(y,conf)
        else:
            y = self.c_proj(y)
        y = self.resid_dropout(y)

        return y
@dataclass
class bc():
    n_embd:int = 0
    n_head:int  = 0
    dropout:float  = 0
    bias:bool = False
class CombineBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout, bias):
        super().__init__()

        self.attn = CrossAttention(bc(n_embd, n_head, dropout, bias))

        self.ln_1 = LayerNorm(n_embd, bias)
        self.ln_2 = LayerNorm(n_embd, bias)
        self.ln_3 = LayerNorm(n_embd, bias)
        self.mlp = MLP(n_embd, dropout, bias)

    def forward(self, x, sx, causal = False,conf=None):
        a = self.attn(self.ln_1(x), self.ln_2(sx), causal,conf)
        sx = sx + a
        sx = sx + self.mlp(self.ln_3(sx),conf)
        return sx

class LayerNorm(nn.Module):
    def __init__(self, n_embd, bias, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd)) if bias else None
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class Attention(nn.Module):
    def __init__(self, n_embd, n_head, dropout, bias, **kwargs):
        super().__init__()
        assert n_embd % n_head == 0
        self.q_proj = Linear(n_embd, n_embd, bias=bias)
        self.k_proj = Linear(n_embd, n_embd, bias=bias)
        self.v_proj = Linear(n_embd, n_embd, bias=bias)
      
      
        self.c_proj = Linear(n_embd,  n_embd, bias=bias)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head 
        self.n_embd = n_embd
        self.dropout = dropout
        self.head_dim =  self.n_embd //  self.n_head 
        
        self.qm = nn.Parameter(torch.ones(self.head_dim).normal_(mean=1, std=0.1))
        
    def forward(self, x, causal=True, conf =None):
        B, T, C = x.size() 
        if isinstance(Linear, HungryLinear):
            q = self.q_proj(x,conf)
            k = self.k_proj(x,conf)
            v = self.v_proj(x,conf)
        else:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        
        #test unit linear, unitifying :D
        y = torch.nn.functional.scaled_dot_product_attention(
            q * (math.log(T)*self.qm),
            k, v, 
            attn_mask=None, dropout_p=self.dropout if self.training else 0,
            is_causal= causal)

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim) 
        
        if isinstance(Linear, HungryLinear):
            y = self.c_proj(y, conf)
        else:
            y = self.c_proj(y)

        y = self.resid_dropout(y)

        return y
class MLP(nn.Module):
    def __init__(self, n_embd, dropout, bias, **kwargs):
        super().__init__()
        up = 4
        self.c_fc    = Linear(n_embd, up * n_embd, bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = Linear( up * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.n_embd = n_embd

    def forward(self, x, conf=None):
        if isinstance(Linear, HungryLinear):
            x = self.c_fc(x,conf)
            x = self.gelu(x)
            x = self.c_proj(x,conf)
            x = self.dropout(x)
        else:
            x = self.c_fc(x)
            x = self.gelu(x)
            x = self.c_proj(x)
            x = self.dropout(x)
        return x
    
class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout, bias, **kwargs):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias)
        self.attn = Attention(n_embd, n_head, dropout, bias)
        self.ln_2 = LayerNorm(n_embd, bias)
        self.mlp = MLP(n_embd, dropout, bias)

    def forward(self, x, causal = True, conf = None):
        x = x + self.attn(self.ln_1(x),causal,conf)
        x = x + self.mlp(self.ln_2(x),conf)
        return x

class SSMBlock(nn.Module):
    def __init__(self, n_embd, dropout, bias, **kwargs):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias)
        self.attn = SSM(n_embd, dropout, bias)
        self.ln_2 = LayerNorm(n_embd, bias)
        self.mlp = MLP(n_embd, dropout, bias)

    def forward(self, x, w=None):
        x = x + self.attn(self.ln_1(x), w)[0]
        x = x + self.mlp(self.ln_2(x))
        return x

csbr = True

class FFTResampler(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_filters: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_filters = num_filters
        self.max_freq = input_dim // 2
        self.min_freq = 1.0

        initial_gaps = (self.max_freq - self.min_freq) / (num_filters)
        initial_logits = torch.log(torch.exp(torch.tensor(initial_gaps)) - 1).expand(num_filters)
        self.gap_logits = nn.Parameter(initial_logits)

        t_in = torch.linspace(0, 1, input_dim)
        t_out = torch.linspace(0, 1, output_dim)
        self.register_buffer("t_in", t_in)
        self.register_buffer("t_out", t_out)
    
    def forward(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert x.shape[dim] == self.input_dim, \
            f"Input tensor dim {dim} has size {x.shape[dim]}, " \
            f"but module was initialized with input_dim {self.input_dim}."

        x_permuted = x.transpose(dim, -1)
        original_shape = x_permuted.shape
        x_flattened = x_permuted.reshape(-1, self.input_dim)
        
        gaps = F.softplus(self.gap_logits)
        cumulative_gaps = torch.cumsum(gaps, dim=0)
        freqs = self.min_freq + cumulative_gaps
        freqs.unsqueeze_(1) 

        arg_in = 2 * math.pi * freqs * self.t_in.unsqueeze(0)
        sin_basis_in = torch.sin(arg_in)
        cos_basis_in = torch.cos(arg_in)
        
        arg_out = 2 * math.pi * freqs * self.t_out.unsqueeze(0)
        sin_basis_out = torch.sin(arg_out)
        cos_basis_out = torch.cos(arg_out)
        
        c_sin = torch.einsum('ns,fs->nf', x_flattened, sin_basis_in)
        c_cos = torch.einsum('ns,fs->nf', x_flattened, cos_basis_in)
        
        sin_recon = torch.einsum('nf,fs->ns', c_sin, sin_basis_out)
        cos_recon = torch.einsum('nf,fs->ns', c_cos, cos_basis_out)
        
        x_reconstructed = sin_recon + cos_recon 
        
        output_shape = original_shape[:-1] + (self.output_dim,)
        output = x_reconstructed.view(output_shape)
        
        output = output.transpose(dim, -1).contiguous()
        
        return output

class Dreamer(nn.Module):
    def __init__(self, mem_block_size, **config):
        super().__init__()
        mbs = mem_block_size
        self.block = Block(**config)
        self.comp = FFTResampler(mbs*2, mbs, mbs+1)
        self.ln = LayerNorm(**config)
        
    def forward(self, x):
        while x.shape[0] > 1:
            b, t, c = x.size()
            x = x.view(b // 2, 2, t, c)
            
            x = x.transpose(1, 2)
            
            x = x.reshape(b // 2, t * 2, c)
            
            x = self.comp(x, dim=1) 
            x = self.block(x, causal = False)
            x = self.ln(x)
        return x 

class OrthogonalLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = F.linear(input, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
       
        input, weight, bias = ctx.saved_tensors
        
        grad_input = grad_weight = grad_bias = None

        input_2d = input.reshape(-1, input.shape[-1])
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])

        if ctx.needs_input_grad[0]:
            grad_input = grad_output_2d @ weight
            grad_input = grad_input.reshape(grad_output.shape[:-1] + (input.shape[-1],))

        if ctx.needs_input_grad[1]:
            grad_weight = grad_output_2d.T @ input_2d

            w_flat = weight.flatten()
            grad_w_flat = grad_weight.flatten()
            
            projection_scalar = torch.dot(w_flat, grad_w_flat) / (torch.dot(w_flat, w_flat) + 1e-8)
            
            grad_parallel = projection_scalar * weight
            
            grad_weight_orthogonal = grad_weight - grad_parallel
            
            grad_weight = grad_weight_orthogonal

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output_2d.sum(dim=0)

        return grad_input, grad_weight, grad_bias

class OrthogonalLinear(nn.Linear):
    def forward(self, input):
        return OrthogonalLinearFunction.apply(input, self.weight, self.bias)

class ProcrustesLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias = None):
        
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.orthogonal_(self.weight)
    
    def compute_orthogonal_weight(self):
        return utils.fast_polar_decomposition(self.weight.to(torch.float32))
        
    def forward(self, x: torch.Tensor, weight_ortho: torch.Tensor =None) -> torch.Tensor:
        if weight_ortho is None:
            weight_ortho = self.compute_orthogonal_weight().to(self.weight.dtype)
        else:
            weight_ortho = weight_ortho
        
        return F.linear(x, weight_ortho, None)

class ProcrustesButterflyLinear(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        
        self.n = in_features

        self.proj1 = ProcrustesLinear(self.n, self.n)
        self.proj2 = ProcrustesLinear(self.n, self.n)
    
    def compute_orthogonal_weights(self):
        return self.proj1.compute_orthogonal_weight(), self.proj2.compute_orthogonal_weight()
    

    def forward(self, x: torch.Tensor, y: torch.Tensor, pw) -> torch.Tensor:
        y1_proj = self.proj1(x,pw[0])
        y2_proj = self.proj2(y,pw[1])
        
        y_out = y1_proj + y2_proj

        return y_out

class LearnableSpectralSinkhorn(nn.Module):
    def __init__(self, dim, n_iter=10):
        super().__init__()
        self.dim = dim
        
        self.up_proj = nn.Linear(dim, dim * 2, bias=False)
        self.shl_down_proj = nn.Linear(dim * 2, dim, bias=False)
        
        self.mag_sinkhorn = SinkhornLinear(dim, n_iter=n_iter)
        
        self._init_fourier_weights()
        with torch.no_grad():
            self.up_proj.weight   += torch.randn_like(self.up_proj.weight  )*1e-5
            self.shl_down_proj.weight += torch.randn_like(self.shl_down_proj.weight)*1e-5
            #self.down_proj.weight.requires_grad = False

    def _init_fourier_weights(self):
        n = torch.arange(self.dim)
        k = torch.arange(self.dim).view(-1, 1)
        
        M = (2 * math.pi * n * k) / self.dim
        
        cos_basis = torch.cos(M)
        sin_basis = torch.sin(M) 
        with torch.no_grad():
            self.up_proj.weight[:self.dim, :] = cos_basis
            self.up_proj.weight[self.dim:, :] = -sin_basis

        inv_scale = 1.0 / self.dim
        with torch.no_grad():
            self.shl_down_proj.weight[:, :self.dim] = cos_basis.t() * inv_scale
            self.shl_down_proj.weight[:, self.dim:] = -sin_basis.t() * inv_scale

    def forward(self, x):
        freq_rect = self.up_proj(x)
        
        real, imag = torch.chunk(freq_rect, 2, dim=-1)
        
        mag_sq = real.pow(2) + imag.pow(2)
        mag = torch.sqrt(mag_sq + 1e-8)
        
        processed_mag = self.mag_sinkhorn(mag)
        
        scale = processed_mag / (mag + 1e-6)
        
        new_real = real * scale
        new_imag = imag * scale
        
        freq_rect_new = torch.cat([new_real, new_imag], dim=-1)
        
        out = self.shl_down_proj(freq_rect_new)
        
        return out


class DoublyStochasticGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, lookahead_factor=1.0):
        ctx.save_for_backward(x, weight)
        ctx.lookahead = lookahead_factor
        return F.linear(x, weight)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        
        grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1])
        x_flat = x.reshape(-1, x.shape[-1])
        
        grad_weight = grad_output_flat.t().matmul(x_flat)

        row_mean = grad_weight.mean(dim=1, keepdim=True)
        col_mean = grad_weight.mean(dim=0, keepdim=True)
        global_mean = grad_weight.mean()
        
        grad_weight_projected = grad_weight - row_mean - col_mean + global_mean

        if ctx.lookahead > 0:
            effective_weight = weight + (ctx.lookahead * grad_weight_projected)
            grad_input = grad_output.matmul(effective_weight)
        else:
            grad_input = grad_output.matmul(weight)

        return grad_input, grad_weight_projected, None

class SinkhornLinear2(nn.Module):
    def __init__(self, size, lookahead=1.0):
        super().__init__()
        self.size = size
        self.lookahead = lookahead
        self.weight = nn.Parameter(torch.eye(size) + torch.randn(size, size) * 1e-4)
        self.project_weights() 
    def project_weights(self):
        
        with torch.no_grad():
            self.weight.clamp_(min=1e-8)
            for _ in range(2): 
                dir = 0.5*(self.weight.sum(dim=1, keepdim=True)+self.weight.sum(dim=0, keepdim=True))
                self.weight.div_(dir)
                
    def forward(self, x):
        if self.training:
            self.project_weights()
            
        return DoublyStochasticGradient.apply(x, self.weight, self.lookahead)
    
class RatRelu(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return utils.rat_relu(x)
    
class DeadRatRelu(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return utils.dead_rat_relu(x)
class WarmStartSinkhornLinear(nn.Module):
    def __init__(self, size, n_iter=10, temp=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(size, size))
        self.temp = temp
        self.n_iter = n_iter
        # Cache for warm start
        self.register_buffer('r_prev', torch.zeros(size, 1))
        self.register_buffer('c_prev', torch.zeros(1, size))

    def forward(self, x):
        # 1. Log-space weights
        Q = self.weight / self.temp
        
        # 2. Initialize with previous step's scalings (Warm Start)
        # This makes 1 iteration effectively as good as 10 iterations from scratch
        r = self.r_prev
        c = self.c_prev
        
        # 3. Sinkhorn Iterations
        # Since we warm start, we can reduce n_iter drastically (e.g. to 2 or 3)
        for _ in range(self.n_iter):
            # Log-space Sinkhorn updates: u <- log(1/N) - log(sum(exp(Q + v)))
            # r (rows) update
            # Q is (N, N), c is (1, N) -> broadcasting
            r = -torch.logsumexp(Q + c, dim=1, keepdim=True)
            # c (cols) update
            c = -torch.logsumexp(Q + r, dim=0, keepdim=True)
            
        # Save state for next forward pass
        if self.training:
            self.r_prev.data.copy_(r.detach())
            self.c_prev.data.copy_(c.detach())

        # 4. Final Matrix
        P = (Q + r + c).exp()
        
        return F.linear(x, P) 

class ZeroMeanLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, strict=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.strict = strict
        
        # Standard Linear initialization
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            bound = 1 / math.sqrt(in_features) if in_features > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        w_mean = self.weight.mean(dim=0, keepdim=True)
        
        if self.strict:
            w_centered = self.weight - w_mean
        else:
            w_centered = self.weight - w_mean + (1.0 / self.out_features)
            
        out = F.linear(x, w_centered)
        
        if self.bias is not None:
            b_centered = self.bias - self.bias.mean() if self.strict else self.bias
            out = out + b_centered
            
        return out

class SymmetricLowRankRouting(nn.Module):
    def __init__(self, n_embd, rank=3, temperature=0.1, n_iter=3):
        super().__init__()
        # Ensure n_embd is divisible by (rank + 1)
        assert n_embd % (rank + 1) == 0 
        
        self.rank = rank
        self.temperature = temperature
        self.n_iter = n_iter

    def forward(self, x):
        b, t, n_embd = x.shape
        chunks = x.chunk(self.rank + 1, dim=-1)
        
        payload = chunks[0] # Shape: [b, t, M]
        
        V = torch.stack(chunks[1:], dim=-1) # Shape: [b, t, M, rank]
        
        log_P = torch.matmul(V, V.transpose(-1, -2)) / self.temperature # [b, t, M, M]
        
        for _ in range(self.n_iter):
            r_sum = torch.logsumexp(log_P, dim=-1, keepdim=True) # [b, t, M, 1]
            log_P = log_P - 0.5 * r_sum - 0.5 * r_sum.transpose(-1, -2)
            
        P = log_P.exp()
        
        routed_payload = torch.einsum('btij,btj->bti', P, payload)
        
        return routed_payload 

class SinkhornLinear(nn.Module):
    def __init__(self, size, n_iter=10, eps=1e-6, grad_project=True, log_space=True,temp=0.125):
        super().__init__()
        self.size = size
        self.n_iter = n_iter
        self.eps = eps
        self.grad_project = grad_project
        self.log_space = log_space
        self.temperature = temp
        self.weight = nn.Parameter(torch.eye(size))
        
        with torch.no_grad():
            self.weight.add_(torch.randn(size, size) * 1e-5)
            
    def forward(self, x):
        if not self.grad_project:
            with torch.no_grad():
                self.weight.copy_(self.project())
            W = self.weight
        else:
            W = self.project()
        return F.linear(x, W)#, W
    
    def project(self):
        if self.log_space:
            log_P = self.weight / self.temperature
            #jacobi
            for _ in range(self.n_iter):
                r_sum = torch.logsumexp(log_P, dim=-1, keepdim=True)
                c_sum = torch.logsumexp(log_P, dim=-2, keepdim=True)
                log_P = log_P - 0.5 * (r_sum + c_sum) 
            return log_P.exp()
        #W = self.weight.clamp(min=1e-8)
        #W = F.gelu(self.weight)
        #W = utils.trap_relu(self.weight)+ 1e-8
       
        #gelu_bottom = 0.169971085

        #W = F.gelu(self.weight)  + 1e-8
        #W = F.leaky_relu(self.weight) + 1e-8
        #W = utils.sin_relu(self.weight) + 1e-8
        #W= torch.abs(self.weight)
        #W= self.weight*F.relu(self.weight)
        W = self.weight*self.weight
        #W = utils.gaussian_bump(self.weight)
        #W = F.leaky_relu(self.weight) + 1e-8
        #W = utils.synthrelu(self.weight)+ 1e-8
        for _ in range(self.n_iter):
            W_dir = 0.5*(W.sum(dim=1, keepdim=True)+W.sum(dim=0, keepdim=True))
            W = W / W_dir#todo sidmoid ese
            #W = W / W_dir
        return W
    
class GdiffLinearfunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, weight_ema, bias, config):
        w_eff = utils.VotedGeometricMean(weight, weight_ema)
        ctx.save_for_backward(input, weight, weight_ema, w_eff)
        ctx.config = config
        return F.linear(input,w_eff,bias) 

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, weight_ema, w_eff = ctx.saved_tensors
        conf = ctx.config
        
        grad_input = None
        grad_weight = None
        grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)

        # --- 2. Gradient w.r.t Weight ---
        if ctx.needs_input_grad[1]:
            input_flat = input.reshape(-1, input.shape[-1])
            grad_flat = grad_output.reshape(-1, grad_output.shape[-1])
            
            raw_grad = grad_flat.t().mm(input_flat)
            
            g_norm = utils.rms_norm(raw_grad)

            # C. Regularization
            wreg = weight
            reg = conf['lambda'] * wreg.sign() * (wreg.abs().pow(conf['reg_dim'] - 1))
            g_final = g_norm + reg
            
            # D. The "gdiff" Scheduler
            aw_diff = weight_ema - w_eff
            denom = weight_ema.abs()*0.5 + (g_final - aw_diff).abs()
            g_diff = 1.0 / (denom + 1e-9)
            
            # E. Final Gradient
            grad_weight = g_final * g_diff
            #grad_weight += torch.randn_like(grad_weight) * 1e-5 * (g_final - aw_diff).abs()
            smoothing = conf['smoothing']

            if 'grow' in conf and conf['grow'] < 1.0:
                rows = grad_weight.shape[0]
                neuron_idx = torch.arange(rows, device=grad_weight.device).unsqueeze(1)
                limit = rows * conf['grow']
                gkernel = torch.where(neuron_idx < limit, 1.0, 0.0)

                grad_weight *= gkernel
                # --- SIDE EFFECT: Update EMA ---
                smoothing *= gkernel
            
            weight_ema.add_(smoothing * (weight - weight_ema))

        # --- 3. Gradient w.r.t Bias ---
        if ctx.needs_input_grad[3] and grad_output is not None:
            # Collapse Batch and Time dimensions
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)


        return grad_input, grad_weight, None, grad_bias, None
phi = (1+math.sqrt(5))/2
def flattening_kernel(x, y, power=2.0, sensitivity=1.0):

    y = torch.as_tensor(y).clamp(min=1e-6, max=1.0)
    
    inv_width = sensitivity * (1.0 - y) / y
    
    kernel = torch.exp( - (x.abs() * inv_width).pow(power) )
    
    return kernel
class GdiffLinear(nn.Module):
    def __init__(self, in_features, out_features, 
                 smoothing=0.01, lambda_reg=0.0, reg_dim=2.0):
        super().__init__()
        self.in_features = in_features
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_normal_(self.weight)
        
        self.register_buffer('weight_ema', self.weight.clone().detach())
        
        self.weight.data.zero_()
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        self.config = {
            'smoothing': smoothing,
            'lambda': lambda_reg,
            'reg_dim': reg_dim,
            'grow': 0,
        }
        self.num = 10000.0
        
    def forward(self, x):
        with torch.no_grad():
            self.num += 1.0
            self.config['grow'] = self.num / 10000
        return GdiffLinearfunc.apply(
            x, self.weight, self.weight_ema, self.bias, self.config
        )
class GdiffEmbedding(nn.Module):
    def __init__(self, num, dim, smoothing=0.01):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num, dim))
        nn.init.normal_(self.weight)
        self.register_buffer('weight_ema', self.weight.clone().detach())
        self.smoothing = smoothing

    def forward(self, input):
        # Sparse EMA Update
        if self.training:
            with torch.no_grad():
                idx = input.unique()
                self.weight_ema[idx].lerp_(self.weight[idx], self.smoothing)

        w = F.embedding(input, self.weight)
        w_ema = F.embedding(input, self.weight_ema)

        mag = (w * w_ema).abs().sqrt()
        sign = (w + w_ema).sign()
        
        return sign * mag

class MLayer(nn.Module):
    def __init__(self, 
                 input_dim, 
                 dim_m, 
                 bias=False, 
                 matrix_squarings_exp=None):
        super(MLayer, self).__init__()
        self.input_dim = input_dim
        self.dim_m = dim_m
        self.with_bias = bias
        self.matrix_squarings_exp = matrix_squarings_exp

        self.rep_to_exp_tensor = nn.Parameter(
            torch.empty(input_dim, dim_m, dim_m)
        )

        if self.with_bias:
            self.matrix_bias = nn.Parameter(torch.empty(1, dim_m, dim_m))
        else:
            self.register_parameter('matrix_bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.rep_to_exp_tensor)
        if self.with_bias:
            nn.init.uniform_(self.matrix_bias, -0.05, 0.05)

    def forward(self, x):
        weight_flat = self.rep_to_exp_tensor.view(self.input_dim, -1)
        
        flat_output = torch.matmul(x, weight_flat)
        
        mat = flat_output.view(-1, self.dim_m, self.dim_m)

        if self.with_bias:
            mat = mat + self.matrix_bias

        if self.matrix_squarings_exp is None:
            return torch.matrix_exp(mat)

        scale_factor = 0.5 ** self.matrix_squarings_exp
        identity = torch.eye(self.dim_m, device=mat.device, dtype=mat.dtype)
        
        mat = mat * scale_factor + identity

        for _ in range(self.matrix_squarings_exp):
            mat = torch.matmul(mat, mat)
            
        return mat


class HungryGdiffLinearfunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, weight_ema, bias, config):
        w_eff = utils.VotedGeometricMean(weight, weight_ema)
        ctx.save_for_backward(input, weight, weight_ema, w_eff)
        ctx.config = config
        
        return F.linear(input,w_eff,bias) 


    @staticmethod
    def backward(ctx, grad_output):
        input, weight, weight_ema, w_eff = ctx.saved_tensors
        conf = ctx.config
        
        grad_input = None
        grad_weight = None
        grad_bias = None

        
        # --- 2. Gradient w.r.t Weight ---
        
        noise = torch.randn_like(input) / ( 1e-5 + input.sum())
        input = input + noise
        input_flat = input.reshape(-1, input.shape[-1])
        grad_flat = grad_output.reshape(-1, grad_output.shape[-1])
        
        raw_grad = grad_flat.t().mm(input_flat)
        
        adjusted_grad = raw_grad
        
        smoothing = conf['smoothing']
        # D. The "gdiff" Scheduler
        aw_diff = weight_ema - w_eff
        denom = weight_ema.abs() + (adjusted_grad - aw_diff).abs()  #joy a hyper param.
        g_diff = 1.0 / torch.sqrt(denom + 1e-9)

        adjusted_grad = adjusted_grad * g_diff
        gw2 = adjusted_grad
        adjusted_grad = utils.rms_norm(adjusted_grad)
        # E. Final Gradient
        grad_weight = adjusted_grad 
        weight_ema.add_(smoothing * (weight - weight_ema))


        if ctx.needs_input_grad[0]:
            
            #gw2 = gfi
            #gw2 = raw_grad
            #gw2 = grad_weight
            alpha = conf.get('lookahead_alpha', 0.1)

            future_weight = weight - (gw2 * alpha)
            future_weight = utils.VotedGeometricMean(future_weight, weight_ema)
            
            #future_weight = weight_ema - (gw2 * alpha)
            grad_input = grad_output.matmul(future_weight)
        
            #grad_input = grad_output.matmul(weight)

        if ctx.needs_input_grad[3] and grad_output is not None:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)


        return grad_input, grad_weight, None, grad_bias, None
    
class HungryGdiffLinear(nn.Module):
    def __init__(self, in_features, out_features, bias = False,
                 smoothing=0.01, lambda_reg=0.0, reg_dim=2.0, lookahead= 1e-4):
        super().__init__()
        self.in_features = in_features
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_normal_(self.weight)
        
        self.register_buffer('weight_ema', self.weight.clone().detach())
        
        self.weight.data.zero_()
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        
        
        self.config = {
            'smoothing': smoothing,
            'lambda': lambda_reg,
            'reg_dim': reg_dim,
            'grow': 0,
            'lookahead_alpha': lookahead,
        }
        
    def forward(self, x):
        return HungryGdiffLinearfunc.apply(
            x, self.weight, self.weight_ema, self.bias, self.config
        ) 


class HungryVGMLinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, weight_ema, bias, config):
        w_eff = utils.VotedGeometricMean(weight, weight_ema)
        ctx.save_for_backward(input, weight, weight_ema, w_eff)
        ctx.config = config
        
        return F.linear(input,w_eff,bias) 


    @staticmethod
    def backward(ctx, grad_output):
        input, weight, weight_ema, w_eff = ctx.saved_tensors
        conf = ctx.config
        
        grad_input = None
        grad_weight = None
        grad_bias = None

        noise = torch.randn_like(input) / ( 1e-5 + input.sum())
        input = input + noise
        input_flat = input.reshape(-1, input.shape[-1])
        grad_flat = grad_output.reshape(-1, grad_output.shape[-1])
        
        raw_grad = grad_flat.t().mm(input_flat)
        
        grad_weight = raw_grad
        #grad_weight = utils.rms(grad_weight)
        
        smoothing = conf['smoothing']
        with torch.no_grad():
            weight_ema.add_(smoothing * (weight - weight_ema))

        if ctx.needs_input_grad[0]:
            
            #gw2 = raw_grad
            gw2 = grad_weight
            alpha = conf.get('lookahead_alpha', 0.1)

            future_weight = weight - (gw2 * alpha)
            future_weight = utils.VotedGeometricMean(future_weight, weight_ema)
            
            #future_weight = weight_ema - (gw2 * alpha)
            grad_input = grad_output.matmul(future_weight)
        
            #grad_input = grad_output.matmul(weight)

        if ctx.needs_input_grad[3] and grad_output is not None:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)


        return grad_input, grad_weight, None, grad_bias, None
    
class HungryVGMLinear(nn.Module):
    def __init__(self, in_features, out_features, bias = False,
                smoothing = 0.01, lookahead= 1e-4, zero_init=False):
        super().__init__()
        self.in_features = in_features
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_normal_(self.weight)
        
        self.register_buffer('weight_ema', self.weight.clone().detach())
        if zero_init:
            self.weight.data.zero_()
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        
        self.config = {
            'lookahead_alpha': lookahead,
            'smoothing': smoothing,
        }
        
    def forward(self, x):
        return HungryVGMLinearFunc.apply(
            x, self.weight, self.weight_ema, self.bias, self.config
        ) 
    
class DRRelu(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(x):
        return utils.dead_rat_relu(x)

class Autoencoder(nn.Module):
    def __init__(self, input, hidden, embed, qdim, quantizer):
        super().__init__()
        self.qdim = qdim
        self.encoder = nn.Sequential(
            nn.Linear(input, hidden), 
            nn.LeakyReLU(), 
            nn.Linear(hidden, qdim))
        self.quantizer = quantizer
        self.codebook = nn.Linear(self.qdim, embed, bias=False)
        self.decoder = nn.Sequential(
            nn.Linear(embed, hidden), 
            nn.LeakyReLU(), 
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
            q = utils.rms_norm(q)
        return self.decoder(q)


class MSEOrderHeadChain(nn.Module):
    def __init__(self, dim: int, num_orders: int = 3, homeostatic_coef: float = 1e-4):
        super().__init__()
        self.num_orders = num_orders
        self.homeostatic_coef = homeostatic_coef
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.GELU(),
                nn.Linear(dim // 4, 1)
            ) for _ in range(num_orders)
        ])
        
    def forward(self, x: torch.Tensor, loss_0: torch.Tensor):
        """
        Args:
            x (Tensor): Input representation of shape (B, T, dim)
            loss_0 (Tensor): Target loss tensor. Should be compatible with shape (B, T).
        """
        B, T, _ = x.shape
        
        # Defensive shape checking to prevent silent (B, T) - (B, T, 1) broadcasting bugs
        if loss_0.dim() == 3 and loss_0.shape[-1] == 1:
            loss_0 = loss_0.squeeze(-1)
        elif loss_0.shape != (B, T):
            raise ValueError(
                f"Expected loss_0 to have shape ({B}, {T}) or ({B}, {T}, 1), "
                f"but got {loss_0.shape}."
            )
        
        predictions = []
        level_losses = []
        
        current_target = loss_0
        
        for i in range(self.num_orders):
            pred_i = self.heads[i](x).squeeze(-1)  # Shape: (B, T)
            predictions.append(pred_i)
            
            loss_i = (pred_i - current_target) ** 2
            level_losses.append(loss_i)
            
            # Detach here to isolate head parameter updates.
            # This forces each head to adapt to the static, realized error of the prior head,
            # while still allowing gradients from all level losses to flow back into x.
            current_target = loss_i.detach()
            
        mean_losses = [l.mean() for l in level_losses]
        aux_loss = sum(mean_losses)
        
        last_pred = predictions[-1]
        homeostatic_loss = torch.mean(last_pred ** 2) * self.homeostatic_coef
        
        total_chain_loss = aux_loss + homeostatic_loss
        
        metrics = {
            f"meta_loss_order_{i+1}": mean_losses[i].item() for i in range(self.num_orders)
        }
        metrics["meta_homeostatic_loss"] = homeostatic_loss.item()
        
        return total_chain_loss, predictions, metrics