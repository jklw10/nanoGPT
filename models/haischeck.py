import math
import os
import time
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.model import LinearMetricSimplexAttention
from parts import quantizer
from parts.modules import HungryLinear as Linear
import parts.utils as utils

# Optimize compiler paths and disable redundant codegen
torch._inductor.config.disable_cpp_codegen = True
os.makedirs('out', exist_ok=True)


# =========================================================================
# CONFIGURATIONS & DATASET ORACLE SETUP (RESTORED FROM UTILS)
# =========================================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
always_byte = True
block_size = 256
batch_size = 64
iters_pretrain = 30000
iters_post = 10000

eval_interval = 1000
eval_iters = 100
warmup_iters = 100
learning_rate = 5e-4
min_lr = learning_rate / 10


def get_lr(iter):
    if iter < warmup_iters:
        return learning_rate * (iter + 1) / (warmup_iters + 1)
    decay_ratio = (iter - warmup_iters) / (iters_pretrain - warmup_iters)
    if iter > iters_pretrain:
        return min_lr
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# Restored utils metadata and get_batch
dataset = 'openwebtext'
decode, vocab_size = utils.get_meta(dataset, always_byte=always_byte)
get_batch = utils.get_get_batch(block_size, batch_size, dataset, device, always_byte=always_byte)


@dataclass
class GPTConfig:
    block_size: int = block_size
    vocab_size: int = vocab_size
    n_layer: int = 4
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = False
    
    # Active Inference & DiffusionBlocks Settings
    num_orders: int = 8         # EMA orders of learning progress
    lp_alpha: float = 0.01       # EMA decay smoothing factor
    gamma_coupling: float = utils.phi
    diffusion_ticks: int = 3    # Refinement iterations over the EBM loss surface
    langevin_lr: float = 0.1    # Step size down energy gradient
    
    # DiffusionBlocks Noise Schedule across depth layers
    sigma_max: float = 0.10
    sigma_min: float = 0.01


# =========================================================================
# ENTROPIC KREIN L4 NLL LOSS MODULE
# =========================================================================
class EntropicKreinL4NLLLoss(nn.Module):
    def __init__(self, S0_fallback: float = 10.0, eps: float = 1e-6, l4_delta: float = 2.0):
        super().__init__()
        self.S0_fallback = S0_fallback
        self.eps = eps
        self.l4_delta = l4_delta

    def forward(self, y_pred, y_true, log_var_raw, W):
        error = y_pred - y_true
        
        residual_var = torch.var(error.detach()) + self.eps
        dynamic_S0 = 0.5 * torch.log(residual_var / self.eps)
        S0 = torch.clamp(dynamic_S0, min=self.S0_fallback)

        log_var = log_var_raw / torch.sqrt(1.0 + (log_var_raw / S0) ** 2)
        var = torch.exp(log_var) + self.eps

        u = error / self.l4_delta
        l4_geometry_loss = (self.l4_delta ** 4 * torch.log1p(u ** 4)).mean()

        nll_uncertainty_loss = (0.5 * (error.detach() ** 2) / var + 0.5 * log_var).mean()

        W_norms = torch.norm(W, dim=0, keepdim=True) + self.eps
        W_normed = W / W_norms
        gram_matrix = torch.matmul(W_normed.T, W_normed)
        
        N = W.shape[1]
        off_diag_mask = ~torch.eye(N, dtype=torch.bool, device=W.device)
        r = gram_matrix[off_diag_mask]
        
        r_sq = torch.clamp(r ** 2, max=1.0 - self.eps)
        gram_penalty = -torch.mean(torch.log(1.0 - r_sq + self.eps))

        return l4_geometry_loss + nll_uncertainty_loss + gram_penalty


# =========================================================================
# PATTERN / SLOT MEMORY MODULE
# =========================================================================
class Patt(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_tokens: int):
        super().__init__()
        self.linear1 = Linear(input_dim, num_tokens, bias=False)
        self.linear2 = Linear(num_tokens, output_dim, bias=False)

    def forward(self, x: torch.Tensor, keys=None, vals=None, mask=None) -> torch.Tensor:
        if keys is not None:
            scores = torch.matmul(x, keys.transpose(-1, -2))
        else:
            scores = self.linear1(x)
            
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) < 0.5, -1e9)

        attn = scores * quantizer.topkhotSTE(scores, 4)
        attn = attn / (attn.sum(keepdim=True, dim=-1) + 1e-8)
        attn = utils.dead_rat_relu(attn)
        
        if vals is not None:
            return torch.matmul(attn, vals)
        return self.linear2(attn)


# =========================================================================
# DIFFUSIONBLOCKS LAYER MODULE (WITH IDENTITY RESIDUAL FIX)
# =========================================================================
class DiffusionBlock(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        
        self.attn = LinearMetricSimplexAttention(config)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp_proj = Linear(config.n_embd, config.n_embd * 4, bias=config.bias)
        self.mlp_out = Linear(config.n_embd * 4, config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.memory = Patt(config.n_embd, config.n_embd, 16)
        
        self.score_head = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # CRITICAL FIX: Initialize score_head to Identity so residual stream flows unbroken
        nn.init.eye_(self.score_head.weight)
        
        # DiffusionBlocks noise schedule
        if config.n_layer > 1:
            decay_ratio = layer_idx / (config.n_layer - 1)
            self.sigma_l = config.sigma_max * ((config.sigma_min / config.sigma_max) ** decay_ratio)
        else:
            self.sigma_l = config.sigma_max

    def forward(self, x, keys=None, vals=None, mask=None, local_dsm=True):
        if local_dsm and self.training:
            x_clean = x.detach()
            noise = torch.randn_like(x_clean)
            x_noisy = x_clean + self.sigma_l * noise
            
            h = x_noisy + self.attn(self.ln_1(x_noisy), causal=True)
            if keys is not None:
                h = h + self.memory(h, keys=keys, vals=vals, mask=mask)
            h = h + self.mlp_out(self.gelu(self.mlp_proj(self.ln_2(h))))
            
            x_denoised = self.score_head(h)
            block_dsm_loss = F.mse_loss(x_denoised, x_clean, reduction='mean')
            return x_denoised, block_dsm_loss
        else:
            h = x + self.attn(self.ln_1(x), causal=True)
            if keys is not None:
                h = h + self.memory(h, keys=keys, vals=vals, mask=mask)
            h = h + self.mlp_out(self.gelu(self.mlp_proj(self.ln_2(h))))
            x_out = self.score_head(h)
            return x_out, torch.tensor(0.0, device=x.device)


# =========================================================================
# HRM AGENT WITH DECOUPLED FORWARD ENGINE
# =========================================================================
class HRMAgent(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_embd = config.n_embd
        
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight
        
        self.token_proj = nn.Linear(config.vocab_size, config.n_embd, bias=False)
        self.blocks = nn.ModuleList([DiffusionBlock(config, l) for l in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        self.predictor = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.log_var_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Action Projections
        self.register_buffer("W_share", torch.randn(config.n_embd, 2) / math.sqrt(config.n_embd))
        self.register_buffer("W_talk", torch.randn(config.n_embd, config.vocab_size) / math.sqrt(config.n_embd))
        self.register_buffer("W_mem_write", torch.randn(config.n_embd, 16) / math.sqrt(config.n_embd))
        self.register_buffer("W_mem_delete", torch.randn(config.n_embd, 16) / math.sqrt(config.n_embd))
        
        self.register_buffer("ema_states", torch.zeros(config.num_orders))
        self.register_buffer("lp_ema", torch.zeros(1))

    def forward(self, idx_or_tokens, targets=None, is_text=True):
        if is_text:
            x = self.wte(idx_or_tokens)
            for block in self.blocks:
                x, _ = block(x, local_dsm=False)
            x = self.ln_f(x)
            logits = self.lm_head(x)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

    def forward_decoupled_active(self, x_one_hot, keys=None, vals=None, mask=None):
        B, T, _ = x_one_hot.shape
        x_curr = self.token_proj(x_one_hot)
        aux_loss_total = torch.tensor(0.0, device=x_curr.device)
        boredom_scale = torch.tensor(0.0, device=x_curr.device)

        # Isolated Block-Wise Training Loop (O(1) Activation Memory)
        for l, block in enumerate(self.blocks):
            x_in = x_curr.detach().requires_grad_(True)
            k_l = keys[l] if keys is not None else None
            v_l = vals[l] if vals is not None else None
            m_l = mask[l] if mask is not None else None
            
            x_out, block_dsm_loss = block(x_in, keys=k_l, vals=v_l, mask=m_l, local_dsm=True)
            aux_loss_total = aux_loss_total + block_dsm_loss.detach()

            if self.training:
                (0.1 * block_dsm_loss).backward()  # Immediate local backward step

            if self.training and T > 1:
                with torch.no_grad():
                    current_val = block_dsm_loss.detach()
                    next_ema_states = []
                    for k in range(self.config.num_orders):
                        prev_ema = self.ema_states[k]
                        deriv = prev_ema - current_val
                        new_ema = prev_ema * (1.0 - self.config.lp_alpha) + current_val * self.config.lp_alpha
                        next_ema_states.append(new_ema)
                        current_val = deriv
                    self.ema_states.copy_(torch.stack(next_ema_states))

                    eps = 1e-8
                    smoothed_error = self.ema_states[0]
                    total_lp = torch.stack([F.relu(self.ema_states[i]) for i in range(1, self.config.num_orders)]).sum()
                    self.lp_ema.copy_(total_lp)

                    relative_lp = total_lp / (smoothed_error + eps)
                    relative_lp_clipped = torch.clamp(relative_lp, min=0.0, max=1.0 - eps)
                    boredom_scale = torch.pow(1.0 - relative_lp_clipped, self.config.gamma_coupling / 2.0)

                restoring_drift = - (self.config.gamma_coupling * 0.01) * x_out.detach() * (1.0 - boredom_scale)
                errm = torch.sqrt(boredom_scale) * 0.01
                x_curr = x_out.detach() + restoring_drift + torch.randn_like(x_out) * errm
            else:
                x_curr = x_out.detach()

        x_head = self.ln_f(x_curr.requires_grad_(True))
        pred_next_state = self.predictor(x_head)
        latent_act = x_head[:, -1, :]

        return {
            'x_head': x_head,
            'proposal_logits': self.lm_head(x_head),
            'log_var_raw': self.log_var_head(x_head),
            'pred_state': pred_next_state,
            'share': latent_act @ self.W_share,
            'talk': latent_act @ self.W_talk,
            'mem_write': latent_act @ self.W_mem_write,
            'mem_delete': latent_act @ self.W_mem_delete,
            'latent': latent_act,
            'aux_loss': aux_loss_total,
            'boredom_scale': boredom_scale
        }


# =========================================================================
# ACCURATE VALIDATION EVALUATOR (PURE EVALUATION WITHOUT NOISE)
# =========================================================================
@torch.no_grad()
def estimate_loss(model, eval_iters=100):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, block_size)
            _, loss = model(X, targets=Y, is_text=True)
            losses[k] = loss.item()
        mean_loss = losses.mean().item()
        bpb = mean_loss / math.log(2.0)
        out[split] = (mean_loss, bpb)
    model.train()
    return out


# =========================================================================
# THERMODYNAMIC ENVIRONMENT & POPULATION SELECTION
# =========================================================================
class DatasetThermodynamicEnvironment:
    def __init__(self, get_batch_fn, batch_size, num_agents=4, device="cuda"):
        self.get_batch = get_batch_fn
        self.batch_size = batch_size
        self.num_agents = num_agents
        self.device = device
        self.energy = torch.full((batch_size, num_agents), 0.8, device=device)

    def reset(self):
        self.energy.fill_(0.8)
        return self.get_batch('train', block_size)

    def step(self, shares, target_energy_loss, active_agent_idx):
        B = self.batch_size
        self.energy = (self.energy - 0.005).clamp(0.0, 1.0)
        payout = torch.exp(-target_energy_loss) * 0.20
        
        batch_idx = torch.arange(B, device=self.device)
        self.energy[batch_idx, active_agent_idx] = (self.energy[batch_idx, active_agent_idx] + payout).clamp(0.0, 1.0)
        
        for i in range(self.num_agents):
            is_sharing = (shares[:, i] == 1)
            share_amount = 0.05
            can_share = is_sharing & (self.energy[:, i] > share_amount)
            self.energy[can_share, i] -= share_amount
            
            share_slice = share_amount / (self.num_agents - 1)
            for j in range(self.num_agents):
                if i == j: continue
                self.energy[can_share, j] = (self.energy[can_share, j] + share_slice).clamp(0.0, 1.0)
                
        return self.get_batch('train', block_size)


# =========================================================================
# EXECUTION PIPELINE
# =========================================================================
if __name__ == "__main__":
    gptconf = GPTConfig()
    
    # ---------------------------------------------------------------------
    # PHASE 1: Autoregressive Baseline Pretraining
    # ---------------------------------------------------------------------
    print("=== Phase 1: Pretraining Autoregressive Next-Byte Baseline ===")
    baseline = HRMAgent(gptconf).to(device).compile()
    baseline_path = 'out/baseline_ckpt.pt'
    
    if os.path.exists(baseline_path):
        print(f"[*] Found existing Phase 1 baseline checkpoint at '{baseline_path}'. Skipping.")
        baseline.load_state_dict(torch.load(baseline_path, map_location=device))
    else:
        opt = torch.optim.AdamW(baseline.parameters(), lr=learning_rate)
        t_start = time.time()
        t_eval = time.time()
        
        for step in range(1, iters_pretrain + 1):
            lr = get_lr(step)
            for param_group in opt.param_groups:
                param_group['lr'] = lr
                
            X, Y = get_batch('train', block_size)
            
            opt.zero_grad(set_to_none=True)
            _, loss = baseline(X, targets=Y, is_text=True)
            loss.backward()
            opt.step()
            
            if step % eval_interval == 0 or step == 1:
                eval_metrics = estimate_loss(baseline, eval_iters=eval_iters)
                train_nats, train_bpb = eval_metrics['train']
                val_nats, val_bpb = eval_metrics['val']
                
                print(
                    f"Pretrain Step {step:5d} | "
                    f"Train: {train_nats:.4f} nats ({train_bpb:.3f} BPB) | "
                    f"Val: {val_nats:.4f} nats ({val_bpb:.3f} BPB)"
                    f"in {(time.time() - t_start):.2f}s.\n"
                )
                t_eval = time.time()
        torch.save(baseline.state_dict(), baseline_path)
        print(f"Phase 1 completed in {(time.time() - t_start):.2f}s.\n")
