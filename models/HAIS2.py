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
from parts.modules import HungryLinear as Linear, AssociativeSSM
import parts.utils as utils

# Optimize compiler paths and disable redundant codegen
torch._inductor.config.disable_cpp_codegen = True
os.makedirs('out', exist_ok=True)


# =========================================================================
# PATTERN / SLOT MEMORY MODULE WITH ONLINE SURGERY
# =========================================================================
class Patt(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_tokens: int, lookahead=1e-5, noise=True, zero_init=True, g_norm=True, max_cap=512):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_tokens = num_tokens
        self.max_cap = max_cap

        self.linear1 = Linear(
            in_features=input_dim,
            out_features=num_tokens,
            bias=False,
            lookahead=lookahead,
            noise=noise,
            zero_init=zero_init,
            g_norm=g_norm
        )

        self.linear2 = Linear(
            in_features=num_tokens,
            out_features=output_dim,
            bias=False,
            lookahead=lookahead,
            noise=noise,
            zero_init=zero_init,
            g_norm=g_norm
        )

    def forward(self, x: torch.Tensor, keys=None, vals=None, mask=None, get_scores=False, conf=None) -> torch.Tensor:
        if keys is not None:
            scores = torch.matmul(x, keys.transpose(-1, -2))
        else:
            scores = self.linear1(x, conf=conf)
            
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) < 0.5, -1e9)

        attn = scores * quantizer.topkhotSTE(scores, 4)
        attn = attn / (attn.sum(keepdim=True, dim=-1) + 1e-8)
        attn = utils.dead_rat_relu(attn)
        
        if vals is not None:
            x_out = torch.matmul(attn, vals)
        else:
            x_out = self.linear2(attn, conf=conf)
        
        if get_scores:
            return x_out, scores
        return x_out


# =========================================================================
# CONFIGURATIONS & DATASET ORACLE setup
# =========================================================================
dataset_config = {'pretrain': 'openwebtext', 'sft': 'sft', 'cot': 'cot'}
task_probs = {'pretrain': 0.40, 'sft': 0.30, 'cot': 0.30}
tasks = list(task_probs.keys())
probs = list(task_probs.values())

device = 'cuda' if torch.cuda.is_available() else 'cpu'
always_byte = True
block_size = 256
batch_size = 64
iters_pretrain = 30000
iters_post = 10000

eval_interval = 1000
eval_iters = 200
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


decode, vocab_size = utils.get_meta('openwebtext', always_byte=always_byte)
get_batch = utils.get_get_batch(block_size, batch_size, dataset_config, device, always_byte=always_byte)


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
# DIFFUSIONBLOCKS LAYER MODULE
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
        
        # DiffusionBlocks noise schedule across depth
        if config.n_layer > 1:
            decay_ratio = layer_idx / (config.n_layer - 1)
            self.sigma_l = config.sigma_max * ((config.sigma_min / config.sigma_max) ** decay_ratio)
        else:
            self.sigma_l = config.sigma_max

    def forward(self, x, keys=None, vals=None, mask=None, local_dsm=True):
        if local_dsm and self.training:
            # Local Denoising Score-Matching
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
# HRM AGENT WITH DECOUPLED DIFFUSIONBLOCKS ARCHITECTURE
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
        
        # Action projections
        W_share = torch.randn(config.n_embd, 2) / math.sqrt(config.n_embd)
        W_talk = torch.randn(config.n_embd, config.vocab_size) / math.sqrt(config.n_embd)
        W_mem_write = torch.randn(config.n_embd, 16) / math.sqrt(config.n_embd)
        W_mem_delete = torch.randn(config.n_embd, 16) / math.sqrt(config.n_embd)
        self.register_buffer("W_share", W_share)
        self.register_buffer("W_talk", W_talk)
        self.register_buffer("W_mem_write", W_mem_write)
        self.register_buffer("W_mem_delete", W_mem_delete)
        
        self.register_buffer("ema_states", torch.zeros(config.num_orders))
        self.register_buffer("lp_ema", torch.zeros(1))

    def forward(self, idx_or_tokens, targets=None, is_text=True):
        """Standard Autoregressive Forward Pass for Phase 1 Pretraining."""
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

    def forward_decoupled_active(self, x_one_hot, keys=None, vals=None, mask=None, is_active=False):
        """
        FULLY DECOUPLED BLOCK-WISE TRAINING PASS (arXiv:2506.14202)
        Executes isolated block-by-block forward & local backward passes, 
        reducing activation memory overhead from O(L) to O(1) depth.
        """
        B, T, _ = x_one_hot.shape
        x_curr = self.token_proj(x_one_hot)
        aux_loss_total = torch.tensor(0.0, device=x_curr.device)
        boredom_scale = torch.tensor(0.0, device=x_curr.device)

        # -----------------------------------------------------------------
        # LAYER-BY-LAYER DECOUPLED FORWARD/BACKWARD LOOP
        # -----------------------------------------------------------------
        for l, block in enumerate(self.blocks):
            # Detach input tensor to break end-to-end graph dependencies
            x_in = x_curr.detach().requires_grad_(True)
            
            k_l = keys[l] if keys is not None else None
            v_l = vals[l] if vals is not None else None
            m_l = mask[l] if mask is not None else None
            
            x_out, block_dsm_loss = block(x_in, keys=k_l, vals=v_l, mask=m_l, local_dsm=True)
            aux_loss_total = aux_loss_total + block_dsm_loss.detach()

            # IMMEDIATE LOCAL BACKWARD STEP: Immediately releases layer 'l' activation graph!
            if self.training:
                local_block_loss = 0.1 * block_dsm_loss
                local_block_loss.backward()

            # Active inference higher-order derivative update
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

        # -----------------------------------------------------------------
        # FINAL HEAD COMPUTATION (ON DECOUPLED LATENT REPRESENTATION)
        # -----------------------------------------------------------------
        x_head = self.ln_f(x_curr.requires_grad_(True))
        pred_next_state = self.predictor(x_head)
        latent_act = x_head[:, -1, :]

        share_logits = latent_act @ self.W_share
        talk_logits = latent_act @ self.W_talk
        mem_write_logits = latent_act @ self.W_mem_write
        mem_delete_logits = latent_act @ self.W_mem_delete
        proposal_logits = self.lm_head(x_head)

        return {
            'x_head': x_head,
            'proposal_logits': proposal_logits,
            'pred_state': pred_next_state,
            'share': share_logits,
            'talk': talk_logits,
            'mem_write': mem_write_logits,
            'mem_delete': mem_delete_logits,
            'latent': latent_act,
            'aux_loss': aux_loss_total,
            'boredom_scale': boredom_scale
        }


# =========================================================================
# THERMODYNAMIC ENVIRONMENT & POPULATION RESET SELECTION
# =========================================================================
class DatasetThermodynamicEnvironment:
    def __init__(self, get_batch, batch_size, num_agents=4, device="cuda"):
        self.get_batch = get_batch
        self.batch_size = batch_size
        self.num_agents = num_agents
        self.device = device
        self.energy = torch.full((batch_size, num_agents), 0.8, device=device)

    def reset(self):
        self.energy.fill_(0.8)
        selected_task = np.random.choice(tasks, p=probs)
        return self.get_batch('train', task=selected_task)

    def step(self, shares, target_energy_loss, active_agent_idx):
        B = self.batch_size
        device = self.device
        
        self.energy = (self.energy - 0.005).clamp(0.0, 1.0)
        payout = torch.exp(-target_energy_loss) * 0.20
        
        batch_idx = torch.arange(B, device=device)
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
                
        selected_task = np.random.choice(tasks, p=probs)
        return self.get_batch('train', task=selected_task)


@torch.no_grad()
def select_medoid_agent(agents_list):
    target_device = agents_list[0].ema_states.device
    emas = torch.stack([agent.ema_states.to(target_device) for agent in agents_list], dim=0)
    diff = emas.unsqueeze(1) - emas.unsqueeze(0)
    dist_matrix = torch.sum(diff ** 2, dim=-1)
    
    avg_dists = dist_matrix.mean(dim=-1, keepdim=True)
    mask_valid = dist_matrix <= (1.5 * avg_dists)
    masked_dists = dist_matrix.masked_fill(~mask_valid, 1e9)
    total_valid_dists = masked_dists.sum(dim=-1)
    
    return torch.argmin(total_valid_dists)


# =========================================================================
# MAIN PIPELINE WITH BENCHMARKING & SPEED TRACING
# =========================================================================
if __name__ == "__main__":
    gptconf = GPTConfig()
    
    # PHASE 1: Autoregressive Baseline Pretraining
    print("=== Phase 1: Pretraining Autoregressive Next-Byte Baseline ===")
    baseline = HRMAgent(gptconf).to(device)
    baseline_path = 'out/baseline_ckpt.pt'
    
    if os.path.exists(baseline_path):
        print(f"[*] Found existing Phase 1 baseline checkpoint at '{baseline_path}'. Skipping.")
        baseline.load_state_dict(torch.load(baseline_path, map_location=device))
    else:
        opt = torch.optim.AdamW(baseline.parameters(), lr=1e-3)
        t_start = time.time()
        
        for step in range(iters_pretrain):
            lr = get_lr(step)
            for param_group in opt.param_groups:
                param_group['lr'] = lr
                
            selected_task = np.random.choice(tasks, p=probs)
            X, Y = get_batch('train', task=selected_task)
            
            opt.zero_grad(set_to_none=True)
            _, loss = baseline(X, targets=Y, is_text=True)
            loss.backward()
            opt.step()
            
            if step % 1000 == 0 and step > 0:
                print(f"Pretrain Step {step:5d} | Loss: {loss.item():.4f}")
                
        torch.save(baseline.state_dict(), baseline_path)
        print(f"Phase 1 finished in {time.time() - t_start:.2f}s.\n")

    # PHASE 2: Cooperative DiffusionBlocks Active Learning
    print("=== Phase 2: Cooperative Decoupled DiffusionBlocks Active Learning ===")
    agents = nn.ModuleList([HRMAgent(gptconf).to(device) for _ in range(4)])
    optimizers = [torch.optim.AdamW(a.parameters(), lr=1e-3) for a in agents]
    env = DatasetThermodynamicEnvironment(get_batch=get_batch, batch_size=batch_size, num_agents=4, device=device)
    
    mem_keys = [[torch.zeros(batch_size, 16, gptconf.n_embd, device=device) for _ in range(gptconf.n_layer)] for _ in range(4)]
    mem_vals = [[torch.zeros(batch_size, 16, gptconf.n_embd, device=device) for _ in range(gptconf.n_layer)] for _ in range(4)]
    mem_mask = [[torch.zeros(batch_size, 16, device=device) for _ in range(gptconf.n_layer)] for _ in range(4)]
    
    for a in agents:
        with torch.no_grad():
            a.token_proj.weight.copy_(a.wte.weight.t())
            
    x_bytes, y_bytes = env.reset()
    
    # Reset CUDA memory stats for benchmarking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    t_phase2_start = time.time()
    t_last_100 = time.time()

    for step in range(1, iters_post):
        shares, talks = [], []
        active_agent_idx = step % 4
        active_agent_loss = torch.tensor(0.0, device=device)
        
        for i in range(4):
            agent = agents[i]
            opt_a = optimizers[i]
            opt_a.zero_grad(set_to_none=True)

            x_one_hot = F.one_hot(x_bytes, num_classes=gptconf.vocab_size).float()
            
            # Execute Decoupled Block-wise Forward + Immediate Local Backward Passes
            out = agent.forward_decoupled_active(
                x_one_hot, 
                keys=mem_keys[i], 
                vals=mem_vals[i], 
                mask=mem_mask[i], 
                is_active=(i == active_agent_idx)
            )
            
            shares.append(torch.argmax(out['share'], dim=-1))
            talks.append(torch.argmax(out['talk'], dim=-1))
            
            # Head Loss & Langevin Refinement for Active Agent
            if i == active_agent_idx:
                with torch.no_grad():
                    latent_target = agent.token_proj(F.one_hot(y_bytes, num_classes=gptconf.vocab_size).float())
                    
                pred_loss = F.mse_loss(out['pred_state'], latent_target)
                target_tokens = y_bytes
                logits = out['proposal_logits']
                refined_logits = logits.detach().clone().requires_grad_(True)
                
                # Langevin Diffusion Loop with Gradient Clipping
                for tick in range(gptconf.diffusion_ticks):
                    energy = F.cross_entropy(refined_logits.view(-1, gptconf.vocab_size), target_tokens.view(-1))
                    energy_grad = torch.autograd.grad(energy, refined_logits)[0]
                    energy_grad = torch.clamp(energy_grad, -1.0, 1.0) # Prevent gradient explosion
                    
                    langevin_noise_scale = torch.sqrt(out['boredom_scale'].detach()) * 0.01
                    refined_logits = refined_logits - gptconf.langevin_lr * energy_grad + langevin_noise_scale * torch.randn_like(refined_logits)
                    
                final_energy_loss = F.cross_entropy(refined_logits.view(-1, gptconf.vocab_size), target_tokens.view(-1))
                proposal_loss = F.cross_entropy(logits.view(-1, gptconf.vocab_size), target_tokens.view(-1))
                
                # Softmax KL Divergence Amortization Loss
                amortization_loss = F.kl_div(
                    F.log_softmax(logits, dim=-1),
                    F.softmax(refined_logits.detach(), dim=-1),
                    reduction='batchmean'
                )
                
                head_loss = pred_loss + proposal_loss + 0.1 * amortization_loss
                head_loss.backward()
                
                active_agent_loss = final_energy_loss.detach()

            # Single optimizer step updates parameters accumulated across local block & head backwards
            opt_a.step()
            
            # Memory Slot Updates
            with torch.no_grad():
                batch_idx = torch.arange(batch_size, device=device)
                write_slot = torch.argmax(out['mem_write'], dim=-1)
                delete_slot = torch.argmax(out['mem_delete'], dim=-1)
                
                for l in range(gptconf.n_layer):
                    mem_mask[i][l][batch_idx, delete_slot] = 0.0
                    mem_keys[i][l][batch_idx, write_slot] = out['latent']
                    mem_vals[i][l][batch_idx, write_slot] = out['latent']
                    mem_mask[i][l][batch_idx, write_slot] = 1.0

        shares_tensor = torch.stack(shares, dim=1)
        x_bytes, y_bytes = env.step(shares_tensor, active_agent_loss, active_agent_idx)
        
        # Benchmark Metrics every 100 steps
        if step % 100 == 0:
            dt_100 = time.time() - t_last_100
            tok_per_sec = (100 * batch_size * block_size * 4) / dt_100
            ms_per_step = (dt_100 / 100.0) * 1000.0
            
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
            avg_energy = env.energy.mean().item()
            
            print(
                f"Decoupled Step {step:4d} | "
                f"Time/100st: {dt_100:5.2f}s ({ms_per_step:5.1f}ms/st) | "
                f"Throughput: {tok_per_sec:6.0f} tok/s | "
                f"Peak VRAM: {peak_vram_mb:6.1f} MB | "
                f"Avg Energy: {avg_energy:.3f}"
            )
            t_last_100 = time.time()