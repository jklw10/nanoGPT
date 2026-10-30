import math
import os
import time
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from parts import quantizer
from parts.modules import HungryLinear as Linear, AssociativeSSM
import parts.utils as utils

# Optimize compiler paths and disable redundant codegen
torch._inductor.config.disable_cpp_codegen = True
os.makedirs('out', exist_ok=True)


# =========================================================================
# 1. SLOT-ADDRESSABLE PATTERN MEMORY MODULE
# =========================================================================
class Patt(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_tokens: int, lookahead=1e-5, noise=True, zero_init=True, g_norm=True, max_cap=512):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_tokens = num_tokens
        self.max_cap = max_cap

        # Key projection: weight shape [num_tokens, input_dim]
        self.linear1 = Linear(
            in_features=input_dim, out_features=num_tokens, bias=False,
            lookahead=lookahead, noise=noise, zero_init=zero_init, g_norm=g_norm
        )
        # Value projection: weight shape [output_dim, num_tokens]
        self.linear2 = Linear(
            in_features=num_tokens, out_features=output_dim, bias=False,
            lookahead=lookahead, noise=noise, zero_init=zero_init, g_norm=g_norm
        )

    def forward(self, x: torch.Tensor, keys=None, vals=None, mask=None, get_scores=False, conf=None) -> torch.Tensor:
        if keys is not None:
            # Dynamic slot-addressable query-key similarity
            scores = torch.matmul(x, keys.transpose(-1, -2))
        else:
            # Static template-based retrieval
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

    def add_tokens(self, n: int, optimizer=None):
        allowed_add = self.max_cap - self.num_tokens
        if allowed_add <= 0 or n <= 0:
            return

        n = min(n, allowed_add)
        device, dtype = self.linear1.weight.device, self.linear1.weight.dtype
        old_w1, old_w2 = self.linear1.weight, self.linear2.weight

        new_keys = torch.zeros(n, self.input_dim, device=device, dtype=dtype)
        self.linear1.weight = nn.Parameter(torch.cat([old_w1.data, new_keys], dim=0))

        new_vals = torch.empty(self.output_dim, n, device=device, dtype=dtype)
        nn.init.kaiming_uniform_(new_vals, a=math.sqrt(5))
        self.linear2.weight = nn.Parameter(torch.cat([old_w2.data, new_vals], dim=1))

        self.num_tokens += n

        if optimizer is not None:
            self._update_optimizer_state(optimizer, old_w1, self.linear1.weight,
                                         lambda t: torch.cat([t, torch.zeros(n, t.shape[1], device=device, dtype=dtype)], dim=0))
            self._update_optimizer_state(optimizer, old_w2, self.linear2.weight,
                                         lambda t: torch.cat([t, torch.zeros(t.shape[0], n, device=device, dtype=dtype)], dim=1))

    def remove_tokens(self, indices, optimizer=None):
        if isinstance(indices, int):
            indices_to_remove = {indices}
        elif isinstance(indices, (list, tuple)):
            indices_to_remove = set(indices)
        elif torch.is_tensor(indices):
            indices_to_remove = set(indices.tolist())
        else:
            raise TypeError("indices must be an int, list/tuple, or torch.Tensor")

        for idx in indices_to_remove:
            if idx < 0 or idx >= self.num_tokens:
                raise IndexError(f"Token index {idx} out of bounds")

        keep_indices = [i for i in range(self.num_tokens) if i not in indices_to_remove]
        if len(keep_indices) == 0:
            raise ValueError("Cannot remove all tokens.")

        old_w1, old_w2 = self.linear1.weight, self.linear2.weight
        self.linear1.weight = nn.Parameter(old_w1.data[keep_indices])
        self.linear2.weight = nn.Parameter(old_w2.data[:, keep_indices])
        self.num_tokens = len(keep_indices)

        if optimizer is not None:
            self._update_optimizer_state(optimizer, old_w1, self.linear1.weight, lambda t: t[keep_indices])
            self._update_optimizer_state(optimizer, old_w2, self.linear2.weight, lambda t: t[:, keep_indices])

    def _update_optimizer_state(self, optimizer, old_param, new_param, resize_fn):
        for group in optimizer.param_groups:
            found_idx = next((i for i, p in enumerate(group['params']) if p is old_param), None)
            if found_idx is not None:
                group['params'][found_idx] = new_param
                state_key = next((k for k in optimizer.state if k is old_param), None)
                if state_key is not None:
                    state = optimizer.state.pop(state_key)
                    if 'exp_avg' in state: state['exp_avg'] = resize_fn(state['exp_avg'])
                    if 'exp_avg_sq' in state: state['exp_avg_sq'] = resize_fn(state['exp_avg_sq'])
                    optimizer.state[new_param] = state
                break


# =========================================================================
# 2. CONFIGURATIONS & ARCHITECTURE MODULES
# =========================================================================
dataset_config = {'pretrain': 'openwebtext', 'sft': 'sft', 'cot': 'cot'}
task_probs = {'pretrain': 0.40, 'sft': 0.30, 'cot': 0.30}
tasks, probs = list(task_probs.keys()), list(task_probs.values())

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


def get_lr(iter_step):
    if iter_step < warmup_iters:
        return learning_rate * (iter_step + 1) / (warmup_iters + 1)
    if iter_step > iters_pretrain:
        return min_lr
    decay_ratio = (iter_step - warmup_iters) / (iters_pretrain - warmup_iters)
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

    # Active Inference Settings
    num_orders: int = 8
    lp_alpha: float = 0.01
    gamma_coupling: float = utils.phi
    diffusion_ticks: int = 3
    langevin_lr: float = 0.1


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ssm = AssociativeSSM(config.n_embd, config.dropout, config.bias)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp_proj = Linear(config.n_embd, config.n_embd * 4, bias=config.bias)
        self.mlp_out = Linear(config.n_embd * 4, config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.memory = Patt(config.n_embd, config.n_embd, 16)

    def forward(self, x, keys=None, vals=None, mask=None):
        x = x + self.ssm(self.ln_1(x))
        if keys is not None:
            x = x + self.memory(x, keys=keys, vals=vals, mask=mask)
        x = x + self.mlp_out(self.gelu(self.mlp_proj(self.ln_2(x))))
        return x


class HRMAgent(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_embd = config.n_embd

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight

        self.token_proj = nn.Linear(config.vocab_size, config.n_embd, bias=False)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)

        self.predictor = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Active action heads
        self.register_buffer("W_share", torch.randn(config.n_embd, 2) / math.sqrt(config.n_embd))
        self.register_buffer("W_talk", torch.randn(config.n_embd, config.vocab_size) / math.sqrt(config.n_embd))
        self.register_buffer("W_mem_write", torch.randn(config.n_embd, 16) / math.sqrt(config.n_embd))
        self.register_buffer("W_mem_delete", torch.randn(config.n_embd, 16) / math.sqrt(config.n_embd))

        self.register_buffer("ema_states", torch.zeros(config.num_orders))
        self.register_buffer("lp_ema", torch.zeros(1))

    def forward(self, idx_or_tokens, targets=None, keys=None, vals=None, mask=None, is_text=True):
        if is_text:
            x = self.wte(idx_or_tokens)
            for block in self.blocks:
                x = block(x)
            x = self.ln_f(x)
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
            return logits, loss

        # Active Inference Pass
        B, T, _ = idx_or_tokens.shape
        x = self.token_proj(idx_or_tokens)
        aux_loss = torch.tensor(0.0, device=x.device)
        boredom_scale = torch.tensor(0.0, device=x.device)

        for l, block in enumerate(self.blocks):
            x_prev = x
            x = block(x, keys=keys, vals=vals, mask=mask)

            if self.training and T > 1:
                norm_current = F.normalize(x[:, :-1], p=2, dim=-1)
                norm_target = F.normalize(x_prev[:, 1:].detach(), p=2, dim=-1)
                predicted_next = self.predictor(norm_current)
                layer_loss = F.mse_loss(predicted_next, norm_target, reduction='mean')
                aux_loss += layer_loss

                with torch.no_grad():
                    current_val = layer_loss.detach()
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

                restoring_drift = - (self.config.gamma_coupling * 0.01) * x * (1.0 - boredom_scale)
                errm = torch.sqrt(boredom_scale) * 0.01
                x = x + restoring_drift + torch.randn_like(x) * errm

        x = self.ln_f(x)
        pred_next_state = self.predictor(x)
        latent_act = x[:, -1, :]

        return {
            'proposal_logits': self.lm_head(x),
            'pred_state': pred_next_state,
            'share': latent_act @ self.W_share,
            'talk': latent_act @ self.W_talk,
            'mem_write': latent_act @ self.W_mem_write,
            'mem_delete': latent_act @ self.W_mem_delete,
            'latent': latent_act,
            'aux_loss': aux_loss,
            'boredom_scale': boredom_scale if self.training and T > 1 else torch.tensor(0.0, device=x.device)
        }


# =========================================================================
# 3. THERMODYNAMIC ENVIRONMENT & POPULATION RESET SELECTION
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
    emas = torch.stack([agent.ema_states for agent in agents_list], dim=0)
    diff = emas.unsqueeze(1) - emas.unsqueeze(0)
    dist_matrix = torch.sum(diff ** 2, dim=-1)

    avg_dists = dist_matrix.mean(dim=-1, keepdim=True)
    mask_valid = dist_matrix <= (1.5 * avg_dists)
    masked_dists = dist_matrix.masked_fill(~mask_valid, 1e9)
    total_valid_dists = masked_dists.sum(dim=-1)

    return torch.argmin(total_valid_dists)


# =========================================================================
# 4. PHASE 1: BASELINE PRETRAINER
# =========================================================================
class BaselinePretrainer:
    def __init__(self, config, device="cuda"):
        self.config = config
        self.device = device
        self.model = HRMAgent(config).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.checkpoint_path = 'out/baseline_ckpt.pt'

    def run(self):
        print("=== Phase 1: Pretraining Autoregressive Next-Byte Baseline ===")
        if os.path.exists(self.checkpoint_path):
            print(f"[*] Found existing baseline checkpoint at '{self.checkpoint_path}'. Skipping.")
            self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
            return self.model

        self.model.train()
        t_start = time.time()
        for step in range(iters_pretrain):
            lr = get_lr(step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            selected_task = np.random.choice(tasks, p=probs)
            X, Y = get_batch('train', task=selected_task)

            self.optimizer.zero_grad(set_to_none=True)
            _, loss = self.model(X, targets=Y, is_text=True)
            loss.backward()
            self.optimizer.step()

            if step % 1000 == 0:
                print(f"Pretrain Step {step:5d} | Task: {selected_task:<10} | Loss: {loss.item():.4f}")

        torch.save(self.model.state_dict(), self.checkpoint_path)
        print(f"Phase 1 completed in {time.time() - t_start:.2f}s. Saved weights to '{self.checkpoint_path}'.\n")
        return self.model


# =========================================================================
# 5. PHASE 2: COOPERATIVE ACTIVE INFERENCE TRAINER
# =========================================================================
class CooperativeActiveTrainer:
    def __init__(self, config, num_agents=4, device="cuda"):
        self.config = config
        self.num_agents = num_agents
        self.device = device

        self.agents = nn.ModuleList([HRMAgent(config).to(device).compile() for _ in range(num_agents)])
        self.optimizers = [torch.optim.AdamW(a.parameters(), lr=1e-3) for a in self.agents]
        self.env = DatasetThermodynamicEnvironment(get_batch, batch_size, num_agents, device)

        self.checkpoint_path = 'out/latest_coop_checkpoint.pt'
        self.start_step = 1
        self.consecutive_steps = 0

        # Memory allocations across agents
        self.mem_keys = [torch.zeros(batch_size, 16, config.n_embd, device=device) for _ in range(num_agents)]
        self.mem_vals = [torch.zeros(batch_size, 16, config.n_embd, device=device) for _ in range(num_agents)]
        self.mem_mask = [torch.zeros(batch_size, 16, device=device) for _ in range(num_agents)]

    def initialize_agents(self, baseline_model=None):
        if os.path.exists(self.checkpoint_path):
            print(f"[*] Found recovery checkpoint: '{self.checkpoint_path}'. Resuming...")
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)
            self.start_step = ckpt['step'] + 1
            self.consecutive_steps = ckpt['consecutive_steps']
            self.agents.load_state_dict(ckpt['agents_state_dict'])
            for i, opt in enumerate(self.optimizers):
                opt.load_state_dict(ckpt['optimizers_state_dicts'][i])
            self.mem_keys = [mk.to(self.device) for mk in ckpt['mem_keys']]
            self.mem_vals = [mv.to(self.device) for mv in ckpt['mem_vals']]
            self.mem_mask = [mm.to(self.device) for mm in ckpt['mem_mask']]
        else:
            print("[*] Initializing cooperative active agents from baseline weights.")
            for agent in self.agents:
                if baseline_model is not None:
                    agent.load_state_dict(baseline_model.state_dict(), strict=False)
                with torch.no_grad():
                    agent.token_proj.weight.copy_(agent.wte.weight.t())

    def save_checkpoint(self, step):
        recovery_state = {
            'step': step,
            'consecutive_steps': self.consecutive_steps,
            'agents_state_dict': self.agents.state_dict(),
            'optimizers_state_dicts': [opt.state_dict() for opt in self.optimizers],
            'mem_keys': [mk.clone() for mk in self.mem_keys],
            'mem_vals': [mv.clone() for mv in self.mem_vals],
            'mem_mask': [mm.clone() for mm in self.mem_mask],
        }
        torch.save(recovery_state, self.checkpoint_path)

    def perform_medoid_reset(self, step):
        print(f"\n[x] Civilization collapsed at step {step}. Performing Medoid EGGROLL Reset.")
        med_idx = select_medoid_agent(self.agents)
        medoid_state = self.agents[med_idx].state_dict()

        with torch.no_grad():
            for agent in self.agents:
                agent.load_state_dict(medoid_state)
                for p in agent.parameters():
                    if p.requires_grad:
                        p.add_(torch.randn_like(p) * 0.03)

        for i in range(self.num_agents):
            self.mem_keys[i].fill_(0.0)
            self.mem_vals[i].fill_(0.0)
            self.mem_mask[i].fill_(0.0)

        self.consecutive_steps = 0

    def run(self):
        print("=== Phase 2: Cooperative Diffusion-EBM Active Learning ===")
        x_bytes, y_bytes = self.env.reset()

        try:
            for step in range(self.start_step, iters_post):
                shares, talks = [], []
                active_agent_idx = step % self.num_agents
                active_agent_loss = torch.tensor(0.0, device=self.device)

                for i in range(self.num_agents):
                    agent = self.agents[i]
                    opt_a = self.optimizers[i]
                    opt_a.zero_grad(set_to_none=True)

                    x_one_hot = F.one_hot(x_bytes, num_classes=self.config.vocab_size).float()
                    out = agent(x_one_hot, keys=self.mem_keys[i], vals=self.mem_vals[i], mask=self.mem_mask[i], is_text=False)

                    shares.append(torch.argmax(out['share'], dim=-1))
                    talks.append(torch.argmax(out['talk'], dim=-1))

                    with torch.no_grad():
                        latent_target = agent.token_proj(F.one_hot(y_bytes, num_classes=self.config.vocab_size).float())

                    pred_loss = F.mse_loss(out['pred_state'], latent_target)
                    total_loss = pred_loss + 0.1 * out['aux_loss']

                    # Langevin EBM Refinement Loop
                    logits = out['proposal_logits']
                    if i == active_agent_idx:
                        target_tokens = y_bytes
                        refined_logits = logits.detach().clone().requires_grad_(True)

                        for tick in range(self.config.diffusion_ticks):
                            energy = F.cross_entropy(refined_logits.view(-1, self.config.vocab_size), target_tokens.view(-1))
                            energy_grad = torch.autograd.grad(energy, refined_logits)[0]
                            langevin_noise_scale = torch.sqrt(out['boredom_scale'].detach()) * 0.01
                            refined_logits = refined_logits - self.config.langevin_lr * energy_grad + langevin_noise_scale * torch.randn_like(refined_logits)

                        final_energy_loss = F.cross_entropy(refined_logits.view(-1, self.config.vocab_size), target_tokens.view(-1))
                        proposal_loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), target_tokens.view(-1))
                        amortization_loss = F.mse_loss(logits, refined_logits.detach())

                        joint_loss = total_loss + proposal_loss + 0.1 * amortization_loss
                        joint_loss.backward()
                        opt_a.step()

                        active_agent_loss = final_energy_loss.detach()

                    # Memory writes
                    with torch.no_grad():
                        batch_idx = torch.arange(batch_size, device=self.device)
                        write_slot = torch.argmax(out['mem_write'], dim=-1)
                        delete_slot = torch.argmax(out['mem_delete'], dim=-1)

                        self.mem_mask[i][batch_idx, delete_slot] = 0.0
                        self.mem_keys[i][batch_idx, write_slot] = out['latent']
                        self.mem_vals[i][batch_idx, write_slot] = out['latent']
                        self.mem_mask[i][batch_idx, write_slot] = 1.0

                shares_tensor = torch.stack(shares, dim=1)
                x_bytes, y_bytes = self.env.step(shares_tensor, active_agent_loss, active_agent_idx)
                self.consecutive_steps += 1

                # Evaluation & Checkpointing
                if step % 100 == 0:
                    avg_energy = self.env.energy.mean().item()
                    lp_vals = torch.stack([a.lp_ema for a in self.agents])
                    mean_loss_0, mean_loss_1 = lp_vals.mean(), lp_vals.std()
                    composite_score = - (mean_loss_1 ** 2) / (mean_loss_0 + 1e-8)

                    if avg_energy < 0.15:
                        self.perform_medoid_reset(step)
                        x_bytes, y_bytes = self.env.reset()
                    else:
                        self.save_checkpoint(step)
                        print(f"Step {step:4d} | Lifetime: {self.consecutive_steps:3d} | Avg Energy: {avg_energy:.3f} | Velocity: {composite_score.item():.4f} | Saved.")

        except KeyboardInterrupt:
            print("\n[!] Training paused by user. Saving recovery checkpoint...")
            self.save_checkpoint(step)
            print("[*] Checkpoint saved successfully.")


# =========================================================================
# MAIN PIPELINE EXECUTION
# =========================================================================
if __name__ == "__main__":
    gptconf = GPTConfig()

    # Phase 1: Pretrain Baseline
    pretrainer = BaselinePretrainer(gptconf, device=device)
    baseline_model = pretrainer.run()

    # Phase 2: Cooperative Active Learning
    coop_trainer = CooperativeActiveTrainer(gptconf, num_agents=4, device=device)
    coop_trainer.initialize_agents(baseline_model)
    coop_trainer.run()