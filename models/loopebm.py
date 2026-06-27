# --- START OF FILE loopebm.py ---

import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from datasets import load_dataset

from parts import quantizer  
from parts import utils

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Adjust block size to accommodate natural language math questions
batch_size      = 64
block_size      = 64      # Elevated to fit longer GSM8K text sequences
n_embd          = 64
n_head          = 8
n_layer         = 4
num_nodes       = 32       # Fixed size cognitive slots
thinking_steps  = 4        # Iterative EBM belief updates
max_cycles      = 4        # Loop limit
gate_penalty    = 0.01

@dataclass
class SolverConfig:
    block_size: int     = block_size
    vocab_size: int     = 256  # Byte-level modeling
    n_embd: int         = n_embd
    n_head: int         = n_head
    n_layer: int        = n_layer
    bias: bool          = False
    dropout: float      = 0.0
    num_nodes: int      = num_nodes
    thinking_steps: int = thinking_steps
    max_cycles: int     = max_cycles
    gate_penalty: float = gate_penalty
    
    # EBM hyper-parameters
    untanh_slope: float = 0.8
    w_scale: float      = 0.8
    sprs: float         = 0.5
    
    # StandardNodeNet3 hyper-parameters
    UNTANH: torch.Tensor = torch.tensor(1.03)
    c_moment: torch.Tensor = torch.tensor(0.8)
    a_variance: torch.Tensor = torch.tensor(0.0026)


# =====================================================================
# 1. DECODING DIAGNOSTIC & UTF-8 SANITY CHECK
# =====================================================================
def run_decoding_diagnostic(custom_decode=None):
    print("=== DECODING DIAGNOSTIC TEST ===")
    test_str = "Question: Weng earns $12/hr. Smart quotes: ’“−”. How much? \nAnswer: $60."
    
    # Method A: Naive modulo ord (Old code)
    ord_encoded = [ord(c) % 256 for c in test_str]
    ord_decoded = "".join([chr(i) if 32 <= i < 127 or i == 10 else f"\\x{i:02x}" for i in ord_encoded])
    
    # Method B: True UTF-8 Bytes (New proposed code)
    utf8_encoded = list(test_str.encode('utf-8', errors='ignore'))
    utf8_decoded = bytes(utf8_encoded).decode('utf-8', errors='replace')
    
    print(f"Original Text:  {test_str}")
    print(f"Method A (ord % 256) Decoded: {ord_decoded}")
    print(f"Method B (UTF-8 Bytes) Decoded: {utf8_decoded}")
    
    if custom_decode is not None:
        try:
            custom_decoded = custom_decode(utf8_encoded)
            print(f"Method C (utils.get_meta decode) Decoded: {custom_decoded}")
        except Exception as e:
            print(f"Method C (utils.get_meta decode) failed: {e}")
            
    is_ord_scuffed = ord_decoded != test_str
    print(f"Is old method scuffed on non-ASCII characters? {'YES' if is_ord_scuffed else 'NO'}")
    print("===============================\n")


# Safe UTF-8 / Custom byte decoder
def safe_decode(indices):
    # Lexically check if custom decode from get_meta is defined in __main__
    if 'decode' in globals() and callable(decode):
        try:
            return decode(indices)
        except Exception:
            pass
            
    clean_indices = [max(0, min(255, int(i))) for i in indices]
    return bytes(clean_indices).decode('utf-8', errors='replace')


def sparse_route(router, node_count, spr):
    probs = torch.sigmoid(router)
    mask = quantizer.topkhotSTE(router, int(math.sqrt(node_count)))
    sparse_probs = mask * probs
    
    sparse_probs = sparse_probs / (sparse_probs.sum(dim=-1, keepdim=True) + 1e-8)
    
    flat_probs = torch.ones_like(probs) / node_count
    ratio = spr #* (0.999 ** step) 

    route = ((1.0-ratio) * sparse_probs) + (ratio * flat_probs)
    
    return route, 0.0, mask

def get_batch_prefix_causal_mask(T, prefix_lens, device):
    """
    Constructs an attention mask for a batch where each item has its own prefix_len.
    Up to prefix_len (Prompt + Workspace), tokens can attend bidirectionally.
    Tokens past prefix_len (Answer) can attend bidirectionally to the prefix,
    but only causally to preceding suffix tokens.
    """
    B = prefix_lens.size(0)
    mask = torch.zeros(B, T, T, device=device)
    
    # Clamp prefix boundary to [1, T] to prevent out-of-bounds masking logic
    clamped_prefix_lens = torch.clamp(prefix_lens, min=1, max=T)
    
    # Grid of coordinates
    r = torch.arange(T, device=device)
    i_grid, j_grid = torch.meshgrid(r, r, indexing='ij')
    
    # Causal restriction (j > i)
    causal_cond = (j_grid > i_grid)
    
    prefix_lens_expanded = clamped_prefix_lens.view(B, 1, 1)
    past_prefix_i = (i_grid.unsqueeze(0) >= prefix_lens_expanded)
    past_prefix_j = (j_grid.unsqueeze(0) >= prefix_lens_expanded)
    
    # Apply causal masking only when both query and key are in the suffix area
    final_cond = causal_cond.unsqueeze(0) & past_prefix_i & past_prefix_j
    
    mask = mask.masked_fill(final_cond, -float('inf'))
    return mask

# =====================================================================
# 3. STATE-SPACE COGNITIVE NODE NETWORKS (WITH DYNAMIC RESET FOR BATCHES)
# =====================================================================
class NodeParams3(nn.Module):
    def __init__(self, batch_size, num_nodes, in_dim, out_dim, device):
        super().__init__() 
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        
        # Shared weights
        base_weight = torch.randn(num_nodes, in_dim, out_dim, device=device) / math.sqrt(self.in_dim)
        self.weight = nn.Parameter(base_weight)
        
        # State registered as a buffer so it migrates automatically
        self.register_buffer('error_ema', 0.01 + 0.01 * torch.randn(batch_size, num_nodes, out_dim, device=device))
        
        # Shared hyperparameters REGISTERED AS BUFFERS so they move with the model device
        self.register_buffer('untanh_slope', torch.tensor(1.0, device=device))
        self.register_buffer('w_scale', torch.tensor(0.8, device=device))
        self.register_buffer('ema_speed', torch.tensor(0.2, device=device))
        self.register_buffer('rew_sens', torch.tensor(1.36, device=device))
        self.register_buffer('n_scale', torch.tensor(0.0, device=device))
        self.register_buffer('speed', torch.tensor(0.5, device=device))
        self.register_buffer('width', torch.tensor(0.01, device=device))

    def reset_state(self, batch_size, device):
        self.error_ema = 0.01 + 0.01 * torch.randn(batch_size, self.weight.shape[0], self.out_dim, device=device)

    def forward(self, x, step_counter):
        # Dynamically cast buffer to input's device to ensure no device mismatches
        w_scale_dev = self.w_scale.to(x.device)
        untanh_slope_dev = self.untanh_slope.to(x.device)

        norm_weight = F.normalize(self.weight, p=2, dim=-2) * w_scale_dev
        raw_pred = torch.einsum('bnd,nde->bne', x, norm_weight)
        
        bound = math.sqrt(2.0 + math.sqrt(2.0))
        out = utils.buntanh(raw_pred, untanh_slope_dev, bound)
        
        return out
    
    def apply_hyperparams(self, config):
        # We re-register the updated values as buffers
        if hasattr(config, 'UNTANH'):
            val = config.UNTANH if isinstance(config.UNTANH, torch.Tensor) else torch.tensor(config.UNTANH, device=self.device)
            self.register_buffer('untanh_slope', val.view(-1, 1, 1))
        elif hasattr(config, 'untanh_slope'):
            val = config.untanh_slope if isinstance(config.untanh_slope, torch.Tensor) else torch.tensor(config.untanh_slope, device=self.device)
            self.register_buffer('untanh_slope', val.view(-1, 1, 1))
            
        if hasattr(config, 'w_scale'):
            val = config.w_scale if isinstance(config.w_scale, torch.Tensor) else torch.tensor(config.w_scale, device=self.device)
            self.register_buffer('w_scale', val.view(-1, 1, 1))
        
        if hasattr(config, 'EMA_SPEED'):
            self.register_buffer('ema_speed', (config.EMA_SPEED if isinstance(config.EMA_SPEED, torch.Tensor) else torch.tensor(config.EMA_SPEED, device=self.device)).view(-1, 1, 1))
            self.register_buffer('rew_sens', (config.FIXED_REWARD_SENSITIVITY if isinstance(config.FIXED_REWARD_SENSITIVITY, torch.Tensor) else torch.tensor(config.FIXED_REWARD_SENSITIVITY, device=self.device)).view(-1, 1, 1, 1))
            self.register_buffer('n_scale', (config.NOISE_SCALE if isinstance(config.NOISE_SCALE, torch.Tensor) else torch.tensor(config.NOISE_SCALE, device=self.device)).view(-1, 1, 1))
            self.register_buffer('speed', (config.speed if isinstance(config.speed, torch.Tensor) else torch.tensor(config.speed, device=self.device)).view(-1, 1, 1, 1))
            self.register_buffer('width', (config.width if isinstance(config.width, torch.Tensor) else torch.tensor(config.width, device=self.device)).view(-1, 1, 1, 1))

class StandardNodeNet3(nn.Module):
    def __init__(self, batch_size, num_nodes, dim, device):
        super().__init__()
        self.N = num_nodes
        self.D = dim
        self.device = device
        
        self.w1 = NodeParams3(batch_size, num_nodes, dim, dim, device)
        self.router = NodeParams3(batch_size, num_nodes, dim, num_nodes, device)
        self.wnll = utils.wnormlearnloss(dim)
        
        # State buffers
        self.register_buffer('state', 1e-5 * torch.randn(batch_size, self.N, dim, device=device))
        self.register_buffer('output', 1e-5 * torch.randn(batch_size, self.N, dim, device=device))
        self.register_buffer('error_ema', torch.ones(batch_size, self.N, device=device))
        self.register_buffer('lp_ema', torch.zeros(batch_size, self.N, device=device))
        self.register_buffer('step_counter', torch.tensor(0.0, device=device))
        
        self.current_input_nodes = 0
        self.lp_alpha = 0.1 
        
        # Register hyperparameters as buffers so they automatically stay on the correct device
        self.register_buffer('c_moment', torch.tensor(0.8, device=device))
        self.register_buffer('a_variance', torch.tensor(0.0026, device=device))
        self.register_buffer('sprs', torch.tensor(0.5, device=device))

    def reset_state(self, batch_size, device):
        self.state = 1e-5 * torch.randn(batch_size, self.N, self.D, device=device)
        self.output = 1e-5 * torch.randn(batch_size, self.N, self.D, device=device)
        self.error_ema = torch.ones(batch_size, self.N, device=device)
        self.lp_ema = torch.zeros(batch_size, self.N, device=device)
        self.step_counter = torch.tensor(0.0, device=device)
        self.w1.reset_state(batch_size, device)
        self.router.reset_state(batch_size, device)
    
    def apply_hyperparams(self, config):
        if hasattr(config, 'sprs'):
            val = config.sprs if isinstance(config.sprs, torch.Tensor) else torch.tensor(config.sprs, device=self.device)
            self.register_buffer('sprs', val.view(-1, 1, 1))
        if hasattr(config, 'c_moment'):
            val = config.c_moment if isinstance(config.c_moment, torch.Tensor) else torch.tensor(config.c_moment, device=self.device)
            self.register_buffer('c_moment', val.view(-1, 1, 1, 1))
        if hasattr(config, 'a_variance'):
            val = config.a_variance if isinstance(config.a_variance, torch.Tensor) else torch.tensor(config.a_variance, device=self.device)
            self.register_buffer('a_variance', val.view(-1, 1, 1))
            
        self.w1.apply_hyperparams(config)

    def forward(self, env_input):
        self.step_counter.add_(1.0)
        batch_size, input_dim = env_input.shape
        
        num_input_nodes = math.ceil(input_dim / self.D)
        
        prediction = self.w1(self.state, self.step_counter) 
        router = self.router(self.state, self.step_counter)
        
        curiosity_bias = F.relu(self.lp_ema).unsqueeze(1) * 5.0 
        route_logits = router + curiosity_bias
        
        # Safely convert sprs to env_input's device
        sprs_dev = self.sprs.to(env_input.device)
        route, aux, mask = sparse_route(route_logits, self.N, sprs_dev)
        
        current_var = route.var(dim=[-1], keepdim=True)
        
        # Safely convert a_variance to env_input's device
        a_variance_dev = self.a_variance.to(env_input.device)
        vd = F.relu(a_variance_dev - current_var)
        
        V = self.output
        base_noise = torch.sqrt(vd) * torch.randn_like(V)
        
        boredom_scale = torch.exp(-50.0 * F.relu(self.lp_ema)).unsqueeze(-1) 
        boredom_noise = base_noise * boredom_scale
        
        target = torch.matmul(route, V) + boredom_noise
        target = utils.softsign(target)

        pad_len = num_input_nodes * self.D - input_dim
        if pad_len > 0:
            env_input_padded = F.pad(env_input, (0, pad_len))
        else:
            env_input_padded = env_input
            
        env_input_reshaped = env_input_padded.view(batch_size, num_input_nodes, self.D)
        
        if num_input_nodes < self.N:
            target = torch.cat([env_input_reshaped, target[:, num_input_nodes:, :]], dim=1)
        else:
            target = env_input_reshaped

        loss_w1 = F.mse_loss(prediction, target, reduction='none')

        with torch.no_grad():
            current_error = loss_w1.mean(dim=-1).detach()
            current_lp = self.error_ema - current_error
            
            self.error_ema = self.error_ema * (1.0 - self.lp_alpha) + current_error * self.lp_alpha
            self.lp_ema = self.lp_ema * (1.0 - self.lp_alpha) + current_lp * self.lp_alpha

        total_loss = (loss_w1).sum() + aux
        
        boredom_scale = torch.exp(-50.0 * torch.relu(self.lp_ema)).unsqueeze(-1) 
        errm = torch.sqrt(boredom_scale) * 0.01
        error_pass = torch.randn_like(target) * errm
        target = target + error_pass

        self.output = prediction.detach()
        self.state = target.detach()

        return total_loss

# =====================================================================
# 4. MODEL ARCHITECTURE COMPONENTS
# =====================================================================

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias_flag):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias_flag else None
       
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings, device='cpu')

    def _set_cos_sin_cache(self, seq_len, device):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, device=x.device)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype, device=x.device),
            self.sin_cached[:seq_len].to(dtype=x.dtype, device=x.device),
        )

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    cos, sin = cos.unsqueeze(0).unsqueeze(1), sin.unsqueeze(0).unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class PrefixAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.q_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = config.dropout
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=config.block_size)

    def forward(self, x, prefix_lens):
        B, T, C = x.size()
        q = self.q_attn(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_attn(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_attn(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(q, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Dynamic variable-length prefix-causal mask
        mask = get_batch_prefix_causal_mask(T, prefix_lens, x.device)
        attn_mask = mask.unsqueeze(1)

        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, 
            dropout_p=self.dropout if self.training else 0, is_causal=False
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = PrefixAttention(config)
        self.ln_1 = LayerNorm(config.n_embd, config.bias)
        self.ln_2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)
        
    def forward(self, x, prefix_lens):
        x = x + self.attn(self.ln_1(x), prefix_lens)
        x = x + self.mlp(self.ln_2(x))
        return x

# =====================================================================
# 5. SOLVER ARCHITECTURE WITH ROLLING-WINDOW MEMORIZATION COGNITION
# =====================================================================

class GSM8KSolverEBM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Instantiate your custom stateful rolling-window NodeNet
        self.node_net = StandardNodeNet3(
            batch_size=batch_size,
            num_nodes=config.num_nodes,
            dim=config.n_embd,
            device=device
        )
        self.node_net.apply_hyperparams(config)
        
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight
        
        self.gate_head = nn.Linear(config.n_embd, 2, bias=config.bias)
        
    def forward(self, idx, prefix_lens, targets=None):
        B, T = idx.size()
        x = self.wte(idx)
        
        active_mask = torch.ones(B, 1, 1, device=idx.device, dtype=x.dtype)
        final_output = torch.zeros_like(x)
        
        gate_logits_list = []
        cycle_logits_list = []
        active_mask_list = []
        total_node_loss = 0.0
        
        # Clamp prefix lengths dynamically to [1, T] to ensure safety across all indexing ops
        clamped_prefix_lens = torch.clamp(prefix_lens, min=1, max=T)
        
        for cycle in range(self.config.max_cycles):
            # Reset recurrent state tracker at the start of each computational cycle
            self.node_net.reset_state(B, device=idx.device)
            
            # Step-by-step sequential processing through the rolling-window state-space
            pred_embs = []
            node_losses = []
            for t in range(T):
                # Detaching the input prevents the MSE loss inside NodeNet from shaping the embedding layers.
                # If you want embeddings to be shaped by both CE and NodeNet's MSE, remove the `.detach()`.
                env_input = x[:, t, :].detach()
                node_loss = self.node_net(env_input)
                node_losses.append(node_loss)
                pred_embs.append(self.node_net.output[:, 0, :])
                
            pred_embs = torch.stack(pred_embs, dim=1)
            
            # INTEGRATION BYPASS: We detach the accumulated pred_embs.
            # This completely stops downstream Cross-Entropy, ACT gating, and loop losses 
            # from propagating gradients back into the NodeNet parameters.
            x = x + pred_embs.detach()
            
            # Process representation using prefix attention blocks
            for block in self.blocks:
                x = block(x, prefix_lens)
                
            # Dynamic gather of boundary token per batch item: x[batch, prefix_lens[batch] - 1]
            batch_indices = torch.arange(B, device=idx.device)
            gate_input = x[batch_indices, clamped_prefix_lens - 1]
            gate_logits = self.gate_head(gate_input)
            
            gate_logits_list.append(gate_logits)
            active_mask_list.append(active_mask)
            
            cycle_logits = self.lm_head(self.ln_f(x))
            cycle_logits_list.append(cycle_logits)
            
            # Weighted transition loss representation over active computational graphs
            # (NodeNet's parameters are updated strictly and internally by this MSE transition loss)
            cycle_node_loss = sum(node_losses) / T
            total_node_loss = total_node_loss + cycle_node_loss * active_mask.mean()
            
            if cycle == self.config.max_cycles - 1:
                halt = torch.ones(B, 1, 1, device=idx.device, dtype=x.dtype)
                loop = torch.zeros(B, 1, 1, device=idx.device, dtype=x.dtype)
            else:
                if self.training:
                    probs = F.gumbel_softmax(gate_logits, tau=1.0, hard=True, dim=-1)
                    halt = probs[:, 0:1].unsqueeze(-1)
                    loop = probs[:, 1:2].unsqueeze(-1)
                else:
                    halt_decision = (gate_logits[:, 0:1] > gate_logits[:, 1:2]).to(x.dtype)
                    halt = halt_decision.unsqueeze(-1)
                    loop = 1.0 - halt
                    
            newly_halted = active_mask * halt
            final_output = final_output + newly_halted * x
            
            active_mask = active_mask * loop
            if not self.training and (active_mask == 0.0).all():
                break
                
        x_final = self.ln_f(final_output)
        logits = self.lm_head(x_final)
        
        loss = None
        if targets is not None:
            ce_losses = []
            for cycle in range(self.config.max_cycles):
                cycle_logits = cycle_logits_list[cycle]
                active_m = active_mask_list[cycle].unsqueeze(-1)
                
                # Suffix target mask generated dynamically based on each item's boundary
                r = torch.arange(T, device=idx.device).unsqueeze(0)
                prefix_lens_expanded = clamped_prefix_lens.unsqueeze(1)
                target_mask = (r >= prefix_lens_expanded).float()
                
                cycle_ce = F.cross_entropy(
                    cycle_logits.view(-1, cycle_logits.size(-1)), 
                    targets.view(-1), 
                    reduction='none'
                ).view(B, T)
                
                masked_ce = (cycle_ce * target_mask * active_m.squeeze(-1)).sum() / (target_mask.sum() + 1e-8)
                ce_losses.append(masked_ce)
                
            loss = sum(ce_losses) / len(ce_losses)
            
            # Combine the cross entropy target loss and EBM transition parameter alignment loss
            # Since pred_embs was detached, the CE part of this combined gradient only shapes self.wte and Blocks,
            # while the 0.01 * total_node_loss shapes NodeParams3 parameters.
            loss = loss + 0.01 * (total_node_loss / len(cycle_logits_list))
            
            # ACT resource allocation loss
            q_losses = []
            for cycle in range(self.config.max_cycles):
                gate_logits = gate_logits_list[cycle]
                cycle_logits = cycle_logits_list[cycle]
                active_m = active_mask_list[cycle]
                
                pred_tokens = cycle_logits.argmax(dim=-1)
                correct_matrix = (pred_tokens == targets).float()
                
                # Check accuracy over correct suffix predictions
                correct_preds = (correct_matrix * target_mask).sum(dim=-1, keepdim=True) / (target_mask.sum(dim=-1, keepdim=True) + 1e-8)
                
                halt_loss = F.binary_cross_entropy_with_logits(
                    gate_logits[:, 0:1], 
                    correct_preds, 
                    reduction='none'
                )
                masked_halt_loss = (halt_loss * active_m.squeeze(-1)).sum() / (active_m.sum() + 1e-8)
                q_losses.append(0.5 * masked_halt_loss)
                
                if cycle < self.config.max_cycles - 1:
                    next_gate_logits = gate_logits_list[cycle + 1]
                    if cycle + 1 == self.config.max_cycles - 1:
                        target_continue = torch.sigmoid(next_gate_logits[:, 0:1])
                    else:
                        next_q = torch.sigmoid(next_gate_logits)
                        target_continue = torch.max(next_q[:, 0:1], next_q[:, 1:2])
                        
                    continue_loss = F.binary_cross_entropy_with_logits(
                        gate_logits[:, 1:2], 
                        target_continue.detach(),
                        reduction='none'
                    )
                    masked_continue_loss = (continue_loss * active_m.squeeze(-1)).sum() / (active_m.sum() + 1e-8)
                    q_losses.append(0.5 * masked_continue_loss)
                    
            loss = loss + sum(q_losses)
            if len(active_mask_list) > 1:
                loop_penalty = torch.stack(active_mask_list[:-1]).mean() * self.config.gate_penalty
                loss = loss + loop_penalty
                
        return logits, loss

    def configure_optimizers(self, lr, wd):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': nodecay_params, 'weight_decay': 0.0},
            {'params': decay_params,   'weight_decay': wd},
        ]
        return torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.999))

# =====================================================================
# 6. TRAINING DATASETS (DYNAMIC LOADER UTILIZING CUSTOM META AND BATCH GETTERS)
# =====================================================================

class HuggingFaceGSM8KDataset:
    def __init__(self, block_size=384, workspace_size=32, device='cpu', split="train"):
        self.block_size = block_size
        self.workspace_size = workspace_size
        self.device = device
        self.vocab_size = 256
        
        # Load the official Grade School Math dataset from HF Hub
        hf_split = "train" if split == "train" else "test"
        print(f"Retrieving GSM8K '{hf_split}' data from Hugging Face Hub...")
        self.dataset = load_dataset("openai/gsm8k", "main", split=hf_split)
        print(f"Dataset successfully loaded. Total items: {len(self.dataset)}")
        
    def get_batch(self, batch_size=32):
        x_batch = []
        y_batch = []
        prefix_lens = []
        
        # Random index sampling
        indices = np.random.randint(0, len(self.dataset), size=batch_size)
        
        for idx in indices:
            row = self.dataset[int(idx)]
            question = row['question']
            answer = row['answer']
            
            # Format structure
            prompt = f"Question: {question}\nAnswer: "
            
            # Convert segments to UTF-8 bytes to prevent unicode size/mod scuffing
            prompt_bytes = prompt.encode('utf-8', errors='ignore')
            workspace_bytes = b"#" * self.workspace_size
            answer_bytes = answer.encode('utf-8', errors='ignore')
            
            # Maximum bytes allowed for the prompt
            max_prompt_bytes = self.block_size - self.workspace_size - 64
            if len(prompt_bytes) > max_prompt_bytes:
                prompt_bytes = b"Question: ... " + prompt_bytes[-max_prompt_bytes:]
            
            full_seq_bytes = prompt_bytes + workspace_bytes + answer_bytes
            encoded = list(full_seq_bytes)
            
            # Pad or truncate in byte-space
            if len(encoded) < self.block_size:
                encoded = encoded + [32] * (self.block_size - len(encoded)) # 0x20 space byte
            else:
                encoded = encoded[:self.block_size]
                
            prefix_len = len(prompt_bytes) + self.workspace_size
            target = encoded[1:] + [32]
            
            x_batch.append(torch.tensor(encoded, dtype=torch.long))
            y_batch.append(torch.tensor(target, dtype=torch.long))
            prefix_lens.append(prefix_len)
            
        x = torch.stack(x_batch).to(self.device)
        y = torch.stack(y_batch).to(self.device)
        prefix_lens_tensor = torch.tensor(prefix_lens, dtype=torch.long, device=self.device)
        
        return x, y, prefix_lens_tensor

# =====================================================================
# 7. INFERENCE ENGINE (AUTOREGRESSIVE GENERATIVE PIPELINE)
# =====================================================================

@torch.no_grad()
def generate_math_solution(model, prompt, workspace_size=32, max_new_tokens=128, temperature=0.7):
    model.eval()
    
    # Boundary constraints using UTF-8 bytes
    prompt_bytes = prompt.encode('utf-8', errors='ignore')
    workspace_bytes = b"#" * workspace_size
    
    max_prompt_len = model.config.block_size - workspace_size - max_new_tokens
    if len(prompt_bytes) > max_prompt_len:
        prompt_bytes = b"Question: ... " + prompt_bytes[-max_prompt_len:]
        
    full_prompt_bytes = prompt_bytes + workspace_bytes
    indices = list(full_prompt_bytes)
    x = torch.tensor([indices], dtype=torch.long, device=device)
    
    # Prefix length represents the fixed boundary
    prefix_len_val = len(indices)
    prefix_lens_tensor = torch.tensor([prefix_len_val], dtype=torch.long, device=device)
    
    for _ in range(max_new_tokens):
        logits, _ = model(x, prefix_lens=prefix_lens_tensor)
        
        next_token_logits = logits[:, -1, :] / temperature
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        x = torch.cat((x, next_token), dim=1)
        if next_token.item() == 10:  # Stop on newline (\n)
            break
            
    generated_indices = x[0, prefix_len_val:].tolist()
    return safe_decode(generated_indices)

# =====================================================================
# 8. EXECUTION / MODEL TRAINING SIMULATION
# =====================================================================

if __name__ == "__main__":
    # Load raw OpenWebText dataset from HF
    dataset = "openwebtext"
    decode, vocab_size = utils.get_meta(dataset, always_byte=True)
    owt_get_batch = utils.get_get_batch(block_size, batch_size, dataset, device, always_byte=True)
    
    # Run diagnostic with the newly loaded custom decoder
    run_decoding_diagnostic(custom_decode=decode)
    
    # Override configuration parameters with extracted meta vocab_size
    gptconf = SolverConfig()
    gptconf.vocab_size = vocab_size
    
    model = GSM8KSolverEBM(gptconf).to(device)
    optimizer = model.configure_optimizers(lr=3e-4, wd=1e-1)
    
    # Initialize the downstream GSM8K dataset
    gsm8k_dataset = HuggingFaceGSM8KDataset(block_size=block_size, workspace_size=32, device=device, split="train")
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    print("Beginning training loop...")
    t0 = time.time()
    
    # Run the pretrain-to-finetune stream sequence
    total_steps = 35000
    
    for step in range(total_steps):
        # Swap streams at step 30,000
        if step < 30000:
            # Defensive validation check on signature input variations
            
            X, Y = owt_get_batch('train')
            # Causal pretraining uses prefix length of 1
            prefix_lens = torch.ones(X.size(0), dtype=torch.long, device=device)
            phase_name = "OWT Pretrain"
        else:
            X, Y, prefix_lens = gsm8k_dataset.get_batch(batch_size=batch_size)
            phase_name = "GSM8K Finetune"
            
        logits, loss = model(X, prefix_lens, Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        if step % 100 == 0 or step == 29999 or step == 30000:
            print(f"Step {step:5d} | Phase: {phase_name:<14s} | Target Loss: {loss.item():.4f} | Batch Avg Prefix: {prefix_lens.float().mean().item():.1f} | Step Time: {(time.time() - t0)*1000:.0f}ms")
            t0 = time.time()

    # Post-training evaluation
    print("\n[Final Post-Training Evaluation]")
    sample_question = "Question: Weng earns $12 an hour for babysitting. Yesterday, she babysat for 5 hours. How much did she earn?\nAnswer: "
    solution = generate_math_solution(model, sample_question, workspace_size=32, max_new_tokens=64)
    print(f"Prompt:\n{sample_question}")
    print(f"Decoded Answer Suffix:\n{solution.strip()}")