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

import os
device          = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size      = 64
block_size      = 64      
n_embd          = 64
n_head          = 8
n_layer         = 4
num_nodes       = 4     
thinking_steps  = 4        
max_cycles      = 4        
gate_penalty    = 0.01

# Dynamic EBM Continuous Hyperparameters
untanh_slope    = 0.8
w_scale         = 0.8
c_moment        = 0.8
a_variance      = 0.0026

# Only run configurator if this file is executed as __main__ 
if __name__ == "__main__":
    configurator_file = 'configurator.py'
    try:
        exec(open('training_tools/configurator.py').read())
    except FileNotFoundError:
        pass

@dataclass
class SolverConfig:
    block_size: int     = globals().get('block_size', 64)
    vocab_size: int     = 256  
    n_embd: int         = globals().get('n_embd', 64)
    n_head: int         = globals().get('n_head', 8)
    n_layer: int        = globals().get('n_layer', 4)
    bias: bool          = False
    dropout: float      = 0.0
    num_nodes: int      = globals().get('num_nodes', 64)
    thinking_steps: int = globals().get('thinking_steps', 4)
    max_cycles: int     = globals().get('max_cycles', 4)
    gate_penalty: float = globals().get('gate_penalty', 0.01)
    
    use_regular_rnn: bool = False
    
    untanh_slope: float = globals().get('untanh_slope', 0.8)
    w_scale: float      = globals().get('w_scale', 0.8)
    
    UNTANH: torch.Tensor     = torch.tensor(globals().get('untanh_slope', 0.8))
    c_moment: torch.Tensor   = torch.tensor(globals().get('c_moment', 0.8))
    a_variance: torch.Tensor = torch.tensor(globals().get('a_variance', 0.0026))

# Safe UTF-8 / Custom byte decoder
def safe_decode(indices):
    if 'decode' in globals() and callable(decode):
        try:
            return decode(indices)
        except Exception:
            pass
            
    clean_indices = [max(0, min(255, int(i))) for i in indices]
    return bytes(clean_indices).decode('utf-8', errors='ignore')

def sparse_route(router, node_count, spr):
    probs = torch.sigmoid(router)
    mask = quantizer.topkhotSTE(router, int(math.sqrt(node_count)))
    sparse_probs = mask * probs
    sparse_probs = sparse_probs / (sparse_probs.sum(dim=-1, keepdim=True) + 1e-8)
    flat_probs = torch.ones_like(probs) / node_count
    ratio = spr 
    route = ((1.0-ratio) * sparse_probs) + (ratio * flat_probs)
    return route, 0.0, mask

def get_batch_prefix_causal_mask(T, prefix_lens, device):
    B = prefix_lens.size(0)
    mask = torch.zeros(B, T, T, device=device)
    clamped_prefix_lens = torch.clamp(prefix_lens, min=1, max=T)
    r = torch.arange(T, device=device)
    i_grid, j_grid = torch.meshgrid(r, r, indexing='ij')
    causal_cond = (j_grid > i_grid)
    prefix_lens_expanded = clamped_prefix_lens.view(B, 1, 1)
    past_prefix_i = (i_grid.unsqueeze(0) >= prefix_lens_expanded)
    past_prefix_j = (j_grid.unsqueeze(0) >= prefix_lens_expanded)
    final_cond = causal_cond.unsqueeze(0) & past_prefix_i & past_prefix_j
    mask = mask.masked_fill(final_cond, -float('inf'))
    return mask

# =====================================================================
# CNN-BASED STATE INITIALIZER MODULE
# =====================================================================
class CNNInit(nn.Module):
    def __init__(self, n_embd, num_nodes, node_dim):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=n_embd, out_channels=n_embd, kernel_size=3, padding=1)
        self.proj = nn.Linear(n_embd, num_nodes * node_dim)
        
    def forward(self, x, prefix_lens):
        B, T, C = x.size()
        x_t = x.transpose(1, 2)  # (B, C, T)
        x_conv = F.gelu(self.conv(x_t))  # (B, C, T)
        
        pooled = []
        for b in range(B):
            plen = int(prefix_lens[b].item())
            plen = max(1, min(plen, T))
            item_pooled = x_conv[b, :, :plen].mean(dim=-1)
            pooled.append(item_pooled)
            
        pooled = torch.stack(pooled, dim=0)  # (B, C)
        out = self.proj(pooled)  # (B, num_nodes * node_dim)
        return out.view(B, -1, C)

# =====================================================================
# 3. STATE-SPACE COGNITIVE NODE NETWORKS
# =====================================================================
class NodeParams3(nn.Module):
    def __init__(self, batch_size, num_nodes, in_dim, out_dim, device):
        super().__init__() 
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        
        base_weight = torch.randn(num_nodes, in_dim, out_dim, device=device) / math.sqrt(self.in_dim)
        self.weight = nn.Parameter(base_weight)
        self.register_buffer('error_ema', 0.01 + 0.01 * torch.randn(batch_size, num_nodes, out_dim, device=device))
        
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
        w_scale_dev = self.w_scale.to(x.device)
        untanh_slope_dev = self.untanh_slope.to(x.device)
        norm_weight = F.normalize(self.weight, p=2, dim=-2) * w_scale_dev
        raw_pred = torch.einsum('bnd,nde->bne', x, norm_weight)
        bound = math.sqrt(2.0 + math.sqrt(2.0))
        out = utils.buntanh(raw_pred, untanh_slope_dev, bound)
        return out
    
    def apply_hyperparams(self, config):
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
            self.register_buffer('width', (config.width if isinstance(config.width, torch.tensor) else torch.tensor(config.width, device=self.device)).view(-1, 1, 1, 1))

class StandardNodeNet3(nn.Module):
    def __init__(self, batch_size, num_nodes, dim, device):
        super().__init__()
        self.N = num_nodes
        self.D = dim
        self.device = device
        
        self.w1 = NodeParams3(batch_size, num_nodes, dim, dim, device)
        self.router = NodeParams3(batch_size, num_nodes, dim, num_nodes, device)
        self.wnll = utils.wnormlearnloss(dim)
        self.raw_sprs = nn.Parameter(torch.tensor(0.0, device=device))  
        
        self.register_buffer('state', 1e-5 * torch.randn(batch_size, self.N, dim, device=device))
        self.register_buffer('output', 1e-5 * torch.randn(batch_size, self.N, dim, device=device))
        self.register_buffer('error_ema', torch.ones(batch_size, self.N, device=device))
        self.register_buffer('lp_ema', torch.zeros(batch_size, self.N, device=device))
        self.register_buffer('step_counter', torch.tensor(0.0, device=device))
        self.current_input_nodes = 0
        self.lp_alpha = 0.1 
        
        self.register_buffer('c_moment', torch.tensor(0.8, device=device))
        self.register_buffer('a_variance', torch.tensor(0.0026, device=device))

    def reset_state(self, batch_size, device):
        self.state = 1e-5 * torch.randn(batch_size, self.N, self.D, device=device)
        self.output = 1e-5 * torch.randn(batch_size, self.N, self.D, device=device)
        self.error_ema = torch.ones(batch_size, self.N, device=device)
        self.lp_ema = torch.zeros(batch_size, self.N, device=device)
        self.step_counter = torch.tensor(0.0, device=device)
        self.w1.reset_state(batch_size, device)
        self.router.reset_state(batch_size, device)
    
    def apply_hyperparams(self, config):
        if hasattr(config, 'c_moment'):
            val = config.c_moment if isinstance(config.c_moment, torch.Tensor) else torch.tensor(config.c_moment, device=self.device)
            self.register_buffer('c_moment', val.view(-1, 1, 1, 1))
        if hasattr(config, 'a_variance'):
            val = config.a_variance if isinstance(config.a_variance, torch.Tensor) else torch.tensor(config.a_variance, device=self.device)
            self.register_buffer('a_variance', val.view(-1, 1, 1))
            
        self.w1.apply_hyperparams(config)

    def forward(self, env_input, state_in=None, output_in=None, error_ema_in=None, lp_ema_in=None):
        self.step_counter.add_(1.0)
        batch_size, input_dim = env_input.shape
        num_input_nodes = math.ceil(input_dim / self.D)
        
        current_state = state_in if state_in is not None else self.state
        current_output = output_in if output_in is not None else self.output
        current_error_ema = error_ema_in if error_ema_in is not None else self.error_ema
        current_lp_ema = lp_ema_in if lp_ema_in is not None else self.lp_ema
        
        prediction = self.w1(current_state, self.step_counter) 
        router = self.router(current_state, self.step_counter)
        
        curiosity_bias = F.relu(current_lp_ema).unsqueeze(1) * 5.0 
        route_logits = router + curiosity_bias
        
        sprs_dev = torch.sigmoid(self.raw_sprs)
        route, aux, mask = sparse_route(route_logits, self.N, sprs_dev)
        
        current_var = route.var(dim=[-1], keepdim=True)
        a_variance_dev = self.a_variance.to(env_input.device)
        vd = F.relu(a_variance_dev - current_var)
        
        V = current_output
        base_noise = torch.sqrt(vd) * torch.randn_like(V)
        
        boredom_scale = torch.exp(-50.0 * F.relu(current_lp_ema)).unsqueeze(-1) 
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
            current_lp = current_error_ema - current_error
            next_error_ema = current_error_ema * (1.0 - self.lp_alpha) + current_error * self.lp_alpha
            next_lp_ema = current_lp_ema * (1.0 - self.lp_alpha) + current_lp * self.lp_alpha

        total_loss = (loss_w1).sum() + aux
        
        boredom_scale = torch.exp(-50.0 * torch.relu(next_lp_ema)).unsqueeze(-1) 
        errm = torch.sqrt(boredom_scale) * 0.01
        error_pass = torch.randn_like(target) * errm
        target = target + error_pass

        # Maintain raw gradient graph copies for return outputs, but store detached representations in buffers
        returned_state = target
        returned_output = prediction

        next_output = prediction.detach()
        next_state = target.detach()

        if state_in is None:
            self.error_ema = next_error_ema
            self.lp_ema = next_lp_ema
            self.output = next_output
            self.state = next_state

        return total_loss, returned_state, returned_output, next_error_ema, next_lp_ema

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
# 5. SOLVER ARCHITECTURE WITH SUPERVISED MEMORY TRAINING (SMT)
# =====================================================================
class GSM8KSolverEBM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        
        print("==> Initializing solver with Cognitive Node Network (StandardNodeNet3)")
        self.node_net = StandardNodeNet3(
            batch_size=batch_size,
            num_nodes=config.num_nodes,
            dim=config.n_embd,
            device=device
        )
        self.node_net.apply_hyperparams(config)
            
        # Target memory state projections for SMT
        self.proj_state = nn.Linear(config.n_embd, config.num_nodes * config.n_embd, bias=config.bias)
        self.proj_output = nn.Linear(config.n_embd, config.num_nodes * config.n_embd, bias=config.bias)
        self.proj_error = nn.Linear(config.n_embd, config.num_nodes, bias=config.bias)
        self.proj_lp = nn.Linear(config.n_embd, config.num_nodes, bias=config.bias)
        
        # SMT CNN State Initializers
        self.cnn_init_state = CNNInit(config.n_embd, config.num_nodes, config.n_embd)
        self.cnn_init_output = CNNInit(config.n_embd, config.num_nodes, config.n_embd)
        
        nn.init.normal_(self.proj_state.weight, std=0.02)
        nn.init.normal_(self.proj_output.weight, std=0.02)
        nn.init.normal_(self.proj_error.weight, std=0.02)
        nn.init.normal_(self.proj_lp.weight, std=0.02)
        
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight
        
    def forward(self, idx, prefix_lens, targets=None, mode='smt'):
        """
        Supports multi-phase execution paths for decoupled SMT training.
          - 'teacher_only': Only train Transformer Teacher on task CE.
          - 'smt': Parallel SMT (Teacher -> Student transition imitation).
          - 'dmt': Recurrent DAgger Imitation Alignment (Student -> Teacher unrolled state correction).
          - 'student_task': Train Recurrent Student RNN directly on task CE outputs.
          - 'teach_teacher': Train Transformer Teacher using predictions from a frozen Student.
        """
        B, T = idx.size()
        x = self.wte(idx)
        clamped_prefix_lens = torch.clamp(prefix_lens, min=1, max=T)
        
        if self.training:
            if mode == 'teacher_only':
                # =========================================================
                # Warmup Mode: Train only Teacher representations
                # =========================================================
                h_teacher = x
                for block in self.blocks:
                    h_teacher = block(h_teacher, prefix_lens)
                logits = self.lm_head(self.ln_f(h_teacher))
                
                loss = None
                if targets is not None:
                    r = torch.arange(T, device=idx.device).unsqueeze(0)
                    target_mask = (r >= clamped_prefix_lens.unsqueeze(1)).float()
                    ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none').view(B, T)
                    loss = (ce_loss * target_mask).sum() / (target_mask.sum() + 1e-8)
                return logits, loss
                
            elif mode == 'smt':
                # =========================================================
                # Parallel SMT mode: Distill Teacher states into Student
                # =========================================================
                with torch.no_grad():
                    h_teacher = x
                    for block in self.blocks:
                        h_teacher = block(h_teacher, prefix_lens)
                    state_targets = self.proj_state(h_teacher).view(B, T, self.config.num_nodes, self.config.n_embd).detach()
                    output_targets = self.proj_output(h_teacher).view(B, T, self.config.num_nodes, self.config.n_embd).detach()
                    error_targets = self.proj_error(h_teacher).view(B, T, self.config.num_nodes).detach()
                    lp_targets = self.proj_lp(h_teacher).view(B, T, self.config.num_nodes).detach()
                
                # Dynamic initialization of the starting state using prompt/prefix CNN pooling
                init_state = self.cnn_init_state(x, clamped_prefix_lens)
                init_output = self.cnn_init_output(x, clamped_prefix_lens)
                init_error = torch.ones(B, self.config.num_nodes, device=idx.device)
                init_lp = torch.zeros(B, self.config.num_nodes, device=idx.device)
                
                state_in = torch.cat([init_state.unsqueeze(1), state_targets[:, :-1, :, :]], dim=1) 
                output_in = torch.cat([init_output.unsqueeze(1), output_targets[:, :-1, :, :]], dim=1) 
                error_in = torch.cat([init_error.unsqueeze(1), error_targets[:, :-1, :]], dim=1) 
                lp_in = torch.cat([init_lp.unsqueeze(1), lp_targets[:, :-1, :]], dim=1) 
                
                env_input_flat = x.reshape(B * T, self.config.n_embd)
                state_in_flat = state_in.reshape(B * T, self.config.num_nodes, self.config.n_embd)
                output_in_flat = output_in.reshape(B * T, self.config.num_nodes, self.config.n_embd)
                error_in_flat = error_in.reshape(B * T, self.config.num_nodes)
                lp_in_flat = lp_in.reshape(B * T, self.config.num_nodes)
                
                node_loss, pred_state, pred_output, pred_error, pred_lp = self.node_net(
                    env_input_flat, state_in_flat, output_in_flat, error_in_flat, lp_in_flat
                )
                
                state_targets_flat = state_targets.reshape(B * T, self.config.num_nodes, self.config.n_embd)
                output_targets_flat = output_targets.reshape(B * T, self.config.num_nodes, self.config.n_embd)
                error_targets_flat = error_targets.reshape(B * T, self.config.num_nodes)
                lp_targets_flat = lp_targets.reshape(B * T, self.config.num_nodes)
                
                loss_smt_state = F.mse_loss(pred_state, state_targets_flat)
                loss_smt_output = F.mse_loss(pred_output, output_targets_flat)
                loss_smt_err = F.mse_loss(pred_error, error_targets_flat)
                loss_smt_lp = F.mse_loss(pred_lp, lp_targets_flat)
                
                smt_alignment_loss = loss_smt_state + loss_smt_output + loss_smt_err + loss_smt_lp
                total_node_loss = (node_loss / T) + smt_alignment_loss
                
                # We project teacher's predictions to logits to establish task loss
                logits = self.lm_head(self.ln_f(h_teacher))
                loss = None
                if targets is not None:
                    r = torch.arange(T, device=idx.device).unsqueeze(0)
                    target_mask = (r >= clamped_prefix_lens.unsqueeze(1)).float()
                    ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none').view(B, T)
                    masked_ce = (ce_loss * target_mask).sum() / (target_mask.sum() + 1e-8)
                    loss = masked_ce + 0.1 * total_node_loss
                return logits, loss
                
            elif mode == 'dmt':
                # =========================================================
                # DMT Mode: Recurrent Imitation Alignment (Exposure Bias Correction)
                # =========================================================
                with torch.no_grad():
                    h_teacher = x
                    for block in self.blocks:
                        h_teacher = block(h_teacher, prefix_lens)
                    state_targets = self.proj_state(h_teacher).view(B, T, self.config.num_nodes, self.config.n_embd).detach()
                    output_targets = self.proj_output(h_teacher).view(B, T, self.config.num_nodes, self.config.n_embd).detach()

                self.node_net.reset_state(B, device=idx.device)
                
                # Apply dynamic CNN state initialization as starting anchor
                self.node_net.state = self.cnn_init_state(x, clamped_prefix_lens)
                self.node_net.output = self.cnn_init_output(x, clamped_prefix_lens)
                
                loss = 0.0
                H = 4  # Unrolling chunk length to prevent gradient explosion
                for t in range(0, T, H):
                    steps = min(H, T - t)
                    if t > 0:
                        # Teacher-guided correction at step boundary
                        self.node_net.state = state_targets[:, t-1].clone()
                        self.node_net.output = output_targets[:, t-1].clone()
                        
                    for k in range(steps):
                        curr_t = t + k
                        env_input = x[:, curr_t, :]
                        node_loss, next_state, next_output, _, _ = self.node_net(env_input)
                        
                        # Train RNN to align unrolled trajectories with target space coordinates
                        loss += F.mse_loss(next_state, state_targets[:, curr_t])
                        loss += F.mse_loss(next_output, output_targets[:, curr_t])
                
                loss = loss / T
                # Return dummy teacher logits during training to prevent graph execution blocks
                logits = self.lm_head(self.ln_f(h_teacher))
                return logits, loss
                
            elif mode == 'student_task':
                # =========================================================
                # Task mode: Direct unrolled Student training
                # =========================================================
                self.node_net.reset_state(B, device=idx.device)
                self.node_net.state = self.cnn_init_state(x, clamped_prefix_lens)
                self.node_net.output = self.cnn_init_output(x, clamped_prefix_lens)
                
                pred_embs = []
                for t in range(T):
                    env_input = x[:, t, :]
                    _, _, next_output, _, _ = self.node_net(env_input)
                    pred_embs.append(next_output[:, 0, :])
                    
                pred_embs = torch.stack(pred_embs, dim=1)
                logits = self.lm_head(self.ln_f(pred_embs))
                
                loss = None
                if targets is not None:
                    r = torch.arange(T, device=idx.device).unsqueeze(0)
                    target_mask = (r >= clamped_prefix_lens.unsqueeze(1)).float()
                    ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none').view(B, T)
                    loss = (ce_loss * target_mask).sum() / (target_mask.sum() + 1e-8)
                return logits, loss

            elif mode == 'teach_teacher':
                # =========================================================
                # Teach Teacher mode: Distill Student predictions into Teacher
                # =========================================================
                with torch.no_grad():
                    self.node_net.reset_state(B, device=idx.device)
                    self.node_net.state = self.cnn_init_state(x, clamped_prefix_lens)
                    self.node_net.output = self.cnn_init_output(x, clamped_prefix_lens)
                    
                    pred_embs = []
                    for t in range(T):
                        env_input = x[:, t, :]
                        _, _, next_output, _, _ = self.node_net(env_input)
                        pred_embs.append(next_output[:, 0, :])
                        
                    pred_embs = torch.stack(pred_embs, dim=1)
                    student_logits = self.lm_head(self.ln_f(pred_embs)).detach()
                
                # Forward Transformer Teacher
                h_teacher = x
                for block in self.blocks:
                    h_teacher = block(h_teacher, prefix_lens)
                teacher_logits = self.lm_head(self.ln_f(h_teacher))
                
                # Calculate alignment loss of teacher's projections against frozen student's logits
                distill_loss = F.mse_loss(teacher_logits, student_logits)
                loss = distill_loss
                
                if targets is not None:
                    r = torch.arange(T, device=idx.device).unsqueeze(0)
                    target_mask = (r >= clamped_prefix_lens.unsqueeze(1)).float()
                    ce_loss = F.cross_entropy(teacher_logits.view(-1, teacher_logits.size(-1)), targets.view(-1), reduction='none').view(B, T)
                    task_loss = (ce_loss * target_mask).sum() / (target_mask.sum() + 1e-8)
                    loss = task_loss + 0.1 * distill_loss
                return teacher_logits, loss
        else:
            # =========================================================
            # EVALUATION / AUTOREGRESSIVE GENERATION
            # =========================================================
            if mode == 'teacher_eval' or mode == 'teacher_only' or mode == 'teach_teacher':
                # Fallback generator for Teacher-only evaluations during Warmup
                h_teacher = x
                for block in self.blocks:
                    h_teacher = block(h_teacher, prefix_lens)
                logits = self.lm_head(self.ln_f(h_teacher))
                return logits, None
            else:
                # Standard deployment mode: Bypasses Transformer completely
                self.node_net.reset_state(B, device=idx.device)
                self.node_net.state = self.cnn_init_state(x, clamped_prefix_lens)
                self.node_net.output = self.cnn_init_output(x, clamped_prefix_lens)
                
                pred_embs = []
                for t in range(T):
                    env_input = x[:, t, :]
                    _, _, next_output, _, _ = self.node_net(env_input)
                    pred_embs.append(next_output[:, 0, :])
                    
                pred_embs = torch.stack(pred_embs, dim=1) 
                logits = self.lm_head(self.ln_f(pred_embs))
                return logits, None

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
# 6. TRAINING DATASETS
# =====================================================================
class HuggingFaceGSM8KDataset:
    def __init__(self, block_size=384, workspace_size=32, device='cpu', split="train"):
        self.block_size = block_size
        self.workspace_size = workspace_size
        self.device = device
        self.vocab_size = 256
        
        hf_split = "train" if split == "train" else "test"
        print(f"Retrieving GSM8K '{hf_split}' data from Hugging Face Hub...")
        self.dataset = load_dataset("openai/gsm8k", "main", split=hf_split)
        print(f"Dataset successfully loaded. Total items: {len(self.dataset)}")
        
    def get_batch(self, batch_size=32):
        x_batch = []
        y_batch = []
        prefix_lens = []
        
        indices = np.random.randint(0, len(self.dataset), size=batch_size)
        
        for idx in indices:
            row = self.dataset[int(idx)]
            question = row['question']
            answer = row['answer']
            prompt = f"Question: {question}\nAnswer: "
            
            prompt_bytes = prompt.encode('utf-8', errors='ignore')
            workspace_bytes = b"#" * self.workspace_size
            answer_bytes = answer.encode('utf-8', errors='ignore')
            
            max_prompt_bytes = self.block_size - self.workspace_size - 64
            if len(prompt_bytes) > max_prompt_bytes:
                prompt_bytes = b"Question: ... " + prompt_bytes[-max_prompt_bytes:]
            
            full_seq_bytes = prompt_bytes + workspace_bytes + answer_bytes
            encoded = list(full_seq_bytes)
            
            if len(encoded) < self.block_size:
                encoded = encoded + [32] * (self.block_size - len(encoded))
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
def generate_math_solution(model, prompt, workspace_size=32, max_new_tokens=128, temperature=0.7, mode='student_eval'):
    model.eval()
    
    prompt_bytes = prompt.encode('utf-8', errors='ignore')
    workspace_bytes = b"#" * workspace_size
    
    max_prompt_len = model.config.block_size - workspace_size - max_new_tokens
    if len(prompt_bytes) > max_prompt_len:
        prompt_bytes = b"Question: ... " + prompt_bytes[-max_prompt_len:]
        
    full_prompt_bytes = prompt_bytes + workspace_bytes
    indices = list(full_prompt_bytes)
    x = torch.tensor([indices], dtype=torch.long, device=device)
    
    prefix_len_val = len(indices)
    prefix_lens_tensor = torch.tensor([prefix_len_val], dtype=torch.long, device=device)
    
    for _ in range(max_new_tokens):
        logits, _ = model(x, prefix_lens=prefix_lens_tensor, mode=mode)
        
        next_token_logits = logits[:, -1, :] / temperature
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        x = torch.cat((x, next_token), dim=1)
        if next_token.item() == 10:  # Stop on newline (\n)
            break
            
    generated_indices = x[0, prefix_len_val:].tolist()
    return safe_decode(generated_indices)


# =====================================================================
# 8. VALIDATION LOSS EVALUATION UTILITIES
# =====================================================================
@torch.no_grad()
def estimate_val_loss(model, get_batch_fn, eval_iters=10, mode='teacher_only'):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        X, Y = get_batch_fn('val')
        prefix_lens = torch.ones(X.size(0), dtype=torch.long, device=device)
        _, loss = model(X, prefix_lens, Y, mode=mode)
        if loss is not None:
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else 0.0

@torch.no_grad()
def estimate_gsm8k_val_loss(model, gsm8k_dataset, eval_iters=10):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        X, Y, prefix_lens = gsm8k_dataset.get_batch(batch_size=batch_size)
        _, loss = model(X, prefix_lens, Y, mode='student_task')
        if loss is not None:
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else 0.0


# =====================================================================
# 9. EXECUTION / MODEL TRAINING WITH MULTI-PHASE SMT
# =====================================================================
if __name__ == "__main__":
    os.makedirs('ckpt', exist_ok=True)
    
    dataset = "openwebtext"
    decode, vocab_size = utils.get_meta(dataset, always_byte=True)
    owt_get_batch = utils.get_get_batch(block_size, batch_size, dataset, device, always_byte=True)
    
    gptconf = SolverConfig()
    gptconf.vocab_size = vocab_size
    
    model = GSM8KSolverEBM(gptconf).to(device)
    optimizer = model.configure_optimizers(lr=1e-3, wd=1e-1)
    
    gsm8k_dataset = HuggingFaceGSM8KDataset(block_size=block_size, workspace_size=32, device=device, split="train")
    gsm8k_val_dataset = HuggingFaceGSM8KDataset(block_size=block_size, workspace_size=32, device=device, split="test")
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # -----------------------------------------------------------------
    # CHECKPOINT LOADER / JUMP LOGIC
    # -----------------------------------------------------------------
    start_step = 0
    checkpoint_dir = 'ckpt'
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')] if os.path.exists(checkpoint_dir) else []
    
    if checkpoint_files:
        latest_ckpt = None
        max_step = -1
        for f in checkpoint_files:
            try:
                ckpt_path = os.path.join(checkpoint_dir, f)
                checkpoint = torch.load(ckpt_path, map_location=device)
                if 'step' in checkpoint and checkpoint['step'] > max_step:
                    max_step = checkpoint['step']
                    latest_ckpt = ckpt_path
            except Exception:
                pass
        
        if latest_ckpt:
            print(f"Found existing checkpoints. Resuming from {latest_ckpt} at step {max_step + 1}...")
            checkpoint = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_step = max_step + 1

    # -----------------------------------------------------------------
    # TIMING PROFILE STEP (dry-run probe for timing checks)
    # -----------------------------------------------------------------
    print("\n" + "="*50)
    print("PROFILING MICROPHASE TIMINGS (1 step each)...")
    print("="*50)
    
    X_owt, Y_owt = owt_get_batch('train')
    prefix_lens_owt = torch.ones(X_owt.size(0), dtype=torch.long, device=device)
    
    # Profile Teacher Only
    t_probe = time.time()
    _, _ = model(X_owt, prefix_lens_owt, Y_owt, mode='teacher_only')
    print(f"Microphase [teacher_only]: {(time.time() - t_probe)*1000:6.1f} ms")
    
    # Profile SMT
    t_probe = time.time()
    _, _ = model(X_owt, prefix_lens_owt, Y_owt, mode='smt')
    print(f"Microphase [smt]         : {(time.time() - t_probe)*1000:6.1f} ms")
    
    # Profile DMT
    t_probe = time.time()
    _, _ = model(X_owt, prefix_lens_owt, Y_owt, mode='dmt')
    print(f"Microphase [dmt]         : {(time.time() - t_probe)*1000:6.1f} ms")
    
    # Profile Student Task
    t_probe = time.time()
    _, _ = model(X_owt, prefix_lens_owt, Y_owt, mode='student_task')
    print(f"Microphase [student_task]: {(time.time() - t_probe)*1000:6.1f} ms")

    # Profile Teach Teacher
    t_probe = time.time()
    _, _ = model(X_owt, prefix_lens_owt, Y_owt, mode='teach_teacher')
    print(f"Microphase [teach_teacher]: {(time.time() - t_probe)*1000:6.1f} ms")
    print("="*50 + "\n")
    
    best_losses = {
        'warmup': float('inf'),
        'smt_pretrain': float('inf'),
        'dmt_pretrain': float('inf'),
        'std_pretrain': float('inf'),
        'teach_teacher_pretrain': float('inf'),
        'task_finetune': float('inf')
    }

    def save_phase_checkpoint(phase_name, current_loss, step):
        if phase_name in best_losses and current_loss < best_losses[phase_name]:
            best_losses[phase_name] = current_loss
            checkpoint_path = f"ckpt/best_checkpoint_{phase_name}.pt"
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': current_loss,
            }, checkpoint_path)
            print(f"  [Checkpoint] Updated best checkpoint for Phase: {phase_name} (Loss: {current_loss:.4f})")

    print(f"Beginning multi-phase pretraining pipeline starting at step {start_step}...")
    t0 = time.time()
    total_steps = 35000
    
    for step in range(start_step, total_steps):
        model.train()
        
        # A. WARMUP PHASE (Step 0 to 5,000)
        if step < 5000:
            phase_name = "warmup"
            mode = 'teacher_only'
            X, Y = owt_get_batch('train')
            prefix_lens = torch.ones(X.size(0), dtype=torch.long, device=device)
            logits, loss = model(X, prefix_lens, Y, mode=mode)
            
        # B. PRETRAINING SMT/DMT LOOP PHASE WITH TEACH TEACHER CYCLE (Step 5,000 to 30,000)
        elif step < 30000:
            X, Y = owt_get_batch('train')
            prefix_lens = torch.ones(X.size(0), dtype=torch.long, device=device)
            
            # Cyclic execution: smt -> dmt -> student_task -> teach teacher from student
            cycle_idx = step % 20
            if cycle_idx < 5:
                phase_name = "smt_pretrain"
                mode = 'smt'
            elif cycle_idx < 10:
                phase_name = "dmt_pretrain"
                mode = 'dmt'
            elif cycle_idx < 15:
                phase_name = "std_pretrain"
                mode = 'student_task'
            else:
                phase_name = "teach_teacher_pretrain"
                mode = 'teach_teacher'
                
            _, loss = model(X, prefix_lens, Y, mode=mode)
                
        # C. TASK FINETUNING PHASE (Step 30,000 to 35,000)
        else:
            X, Y, prefix_lens = gsm8k_dataset.get_batch(batch_size=batch_size)
            phase_name = "task_finetune"
            mode = 'student_task'
            logits, loss = model(X, prefix_lens, Y, mode=mode)
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        # Log training and validation statistics every 1,000 steps
        if step % 1000 == 0 or step == total_steps - 1:
            if phase_name == "task_finetune":
                val_loss = estimate_gsm8k_val_loss(model, gsm8k_val_dataset, eval_iters=10)
            else:
                val_loss = estimate_val_loss(model, owt_get_batch, eval_iters=10, mode=mode)
                
            print(f"Step {step:5d} | Phase: {phase_name:<22s} | Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Batch Avg Prefix: {prefix_lens.float().mean().item():.1f} | Step Time: {(time.time() - t0)*1000:.0f}ms")
            save_phase_checkpoint(phase_name, loss.item(), step)
            t0 = time.time()
            
        # FOUL PLAY GUARD: Periodic evaluations (uses Teacher during Warmup, and Student RNN afterwards)
        if step > 0 and step % 1000 == 0:
            sample_question = "Question: Weng earns $12 an hour for babysitting. Yesterday, she babysat for 5 hours. How much did she earn?\nAnswer: "
            eval_mode = 'teacher_eval' if step < 5000 else 'student_eval'
            solution = generate_math_solution(model, sample_question, workspace_size=32, max_new_tokens=48, mode=eval_mode)
            print(f"\n[Foul Play Check @ Step {step}]")
            print(f"Prompt: {sample_question.strip()}")
            print(f"{'Teacher' if step < 5000 else 'Student RNN'} Decoded Output: {solution.strip()}")
            print("-" * 50 + "\n")
            t0 = time.time()

# param searchdef
model = GSM8KSolverEBM
config = SolverConfig

config_space = {
    "untanh_slope": {
        "type": "dynamic",
        "scale": "linear",
        "low": 0.5,
        "high": 0.99,
        "default": 0.8
    },
    "w_scale": {
        "type": "dynamic",
        "scale": "linear",
        "low": 0.1,
        "high": 1.5,
        "default": 0.8
    },
    "c_moment": {
        "type": "dynamic",
        "scale": "linear",
        "low": 0.1,
        "high": 0.99,
        "default": 0.8
    },
    "a_variance": {
        "type": "dynamic",
        "scale": "log",       
        "low": 0.0001,
        "high": 0.01,
        "default": 0.0026
    },
    "dummy_scale": {
        "type": "static",
        "values": [0],
        "default": 0
    }
}