import math
import os
import time
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from parts import quantizer
from parts.modules import HungryLinear as Linear,  AssociativeSSM
import parts.utils as utils

# Optimize compiler paths and disable redundant codegen
torch._inductor.config.disable_cpp_codegen = True
os.makedirs('out', exist_ok=True)


class Patt(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_tokens: int, lookahead=1e-5, noise=True, zero_init=True, g_norm=True, max_cap = 512):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_tokens = num_tokens
        self.max_cap = max_cap
        # Key projection: weight shape is [num_tokens, input_dim]
        self.linear1 = Linear(
            in_features =input_dim,
            out_features=num_tokens,
            bias        =False,
            lookahead   =lookahead,
            noise       =noise,
            zero_init   =zero_init,
            g_norm      =g_norm
        )

        # Value projection: weight shape is [output_dim, num_tokens]
        self.linear2 = Linear(
            in_features =num_tokens,
            out_features=output_dim,
            bias        =False,
            lookahead   =lookahead,
            noise       =noise,
            zero_init   =zero_init,
            g_norm      =g_norm
        )

    def forward(self, x: torch.Tensor, keys=None, vals=None, mask=None, get_scores=False, conf=None) -> torch.Tensor:
        if keys is not None:
            # Dynamic slot-addressable query-key similarity [B, T, D] x [B, D, M] -> [B, T, M]
            scores = torch.matmul(x, keys.transpose(-1, -2))
        else:
            # Static template-based retrieval
            scores = self.linear1(x, conf=conf)
            
        if mask is not None:
            # Apply mask to dynamic memory slots (mask shape is [B, M])
            scores = scores.masked_fill(mask.unsqueeze(1) < 0.5, -1e9)

        attn = scores * quantizer.topkhotSTE(scores, 4)
        attn = attn / (attn.sum(keepdim=True, dim=-1) + 1e-8)
        attn = utils.dead_rat_relu(attn)
        
        if vals is not None:
            # Retrieve values dynamically [B, T, M] x [B, M, D] -> [B, T, D]
            x_out = torch.matmul(attn, vals)
        else:
            # Fall back to static projection
            x_out = self.linear2(attn, conf=conf)
        
        if get_scores:
            return x_out, scores
        return x_out

    def add_tokens(self, n: int, optimizer=None):
        # Cap additions so that self.num_tokens + n <= 512
        allowed_add = self.max_cap - self.num_tokens
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
# =========================================================================
# CONFIGURATIONS & TASK MAPPING
# =========================================================================
dataset_config = {
    'pretrain': 'openwebtext',
    'sft': 'sft',
    'cot': 'cot'
}

task_probs = {
    'pretrain': 0.40,
    'sft': 0.30,
    'cot': 0.30
}
tasks = list(task_probs.keys())
probs = list(task_probs.values())

device = 'cuda' if torch.cuda.is_available() else 'cpu'
always_byte = True
block_size = 256
batch_size = 64  # Reduced batch size for parallel multi-agent memory constraints
iters_pretrain = 30000
iters_post = 10000


eval_interval   = 1000
eval_iters      = 200
warmup_iters    = 100

learning_rate   = 5e-4
min_lr          = learning_rate / 10

def get_lr(iter):

    if (iter < warmup_iters):
        return learning_rate * (iter + 1) / (warmup_iters + 1)

    decay_ratio = (iter - warmup_iters) / (iters_pretrain - warmup_iters)
    if (iter > iters_pretrain):
        return min_lr
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
    return min_lr + coeff * (learning_rate - min_lr)

# Retrieve actual dataset metadata and build the batch loader
decode, vocab_size = utils.get_meta('openwebtext', always_byte=always_byte)
get_batch = utils.get_get_batch(
    block_size, 
    batch_size, 
    dataset_config, 
    device, 
    always_byte=always_byte
)

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
    num_orders: int = 8         # EMA orders of learning progress
    lp_alpha: float = 0.01       # EMA decay smoothing factor
    gamma_coupling: float = utils.phi
    diffusion_ticks: int = 3    # Refinement iterations over the EBM loss surface
    langevin_lr: float = 0.1    # Step size down the energy gradient

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ssm = AssociativeSSM(config.n_embd, config.dropout, config.bias)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp_proj = Linear(config.n_embd, config.n_embd * 4, bias=config.bias)
        self.mlp_out = Linear(config.n_embd * 4, config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.memory = Patt(config.n_embd, config.n_embd, 16)  # 16 slot-addressable keys/vals

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
        
        # Phase 1: Autoregressive embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight
        
        # Phase 2: Input projections for active sensory representations
        self.token_proj = nn.Linear(config.vocab_size, config.n_embd, bias=False)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        # Predictor mapping hidden state transition targets
        self.predictor = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Active action heads
        
        W_share = torch.randn(config.n_embd, 2) / math.sqrt(config.n_embd)
        W_talk = torch.randn(config.n_embd, config.vocab_size) / math.sqrt(config.n_embd)
        W_mem_write = torch.randn(config.n_embd, 16) / math.sqrt(config.n_embd) # Set to 16 slots to prevent bottlenecking
        W_mem_delete = torch.randn(config.n_embd, 16) / math.sqrt(config.n_embd)
        self.register_buffer("W_share", W_share)
        self.register_buffer("W_talk", W_talk)
        self.register_buffer("W_mem_write", W_mem_write)
        self.register_buffer("W_mem_delete", W_mem_delete)
        
        # Buffer states tracking learning progress and derivatives
        self.register_buffer("ema_states", torch.zeros(config.num_orders))
        self.register_buffer("lp_ema", torch.zeros(1))
        
    def forward(self, idx_or_tokens, targets=None, keys=None, vals=None, mask=None, is_text=True):
        if is_text:
            # Phase 1: Direct autoregressive next-token prediction
            x = self.wte(idx_or_tokens)
            for block in self.blocks:
                x = block(x)
            x = self.ln_f(x)
            logits = self.lm_head(x)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        else:
            # Phase 2: Multi-Layer HRM Gated Active Inference
            B, T, _ = idx_or_tokens.shape
            x = self.token_proj(idx_or_tokens)
            aux_loss = torch.tensor(0.0, device=x.device)
            boredom_scale = torch.tensor(0.0, device=x.device) # Prevent UnboundLocalError
            
            for l, block in enumerate(self.blocks):
                x_prev = x
                x = block(x, keys=keys, vals=vals, mask=mask)
                
                if self.training and T > 1:
                    norm_current = F.normalize(x[:, :-1], p=2, dim=-1)
                    norm_target = F.normalize(x_prev[:, 1:].detach(), p=2, dim=-1)
                    
                    predicted_next = self.predictor(norm_current)
                    layer_loss = F.mse_loss(predicted_next, norm_target, reduction='mean')
                    aux_loss = aux_loss + layer_loss
                    
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
                    
                    # Coordinate-Restoring Drift
                    restoring_drift = - (self.config.gamma_coupling * 0.01) * x * (1.0 - boredom_scale)
                    x = x + restoring_drift
                    
                    # Chaos exploration noise injection
                    errm = torch.sqrt(boredom_scale) * 0.01
                    x = x + torch.randn_like(x) * errm
            
            x = self.ln_f(x)
            pred_next_state = self.predictor(x)
            latent_act = x[:, -1, :]
            
            # Emit sequence-wide proposal logits to be refined over the energy surface
            share_logits      = latent_act @ self.W_share
            talk_logits       = latent_act @ self.W_talk
            mem_write_logits  = latent_act @ self.W_mem_write
            mem_delete_logits = latent_act @ self.W_mem_delete
            
            proposal_logits = self.lm_head(x)
            
            return {
                'proposal_logits': proposal_logits,
                'pred_state': pred_next_state,
                'share': share_logits,
                'talk': talk_logits,
                'mem_write': mem_write_logits,
                'mem_delete': mem_delete_logits,
                'latent': latent_act,
                'aux_loss': aux_loss,
                'boredom_scale': boredom_scale if self.training and T > 1 else torch.tensor(0.0)
            }

# =========================================================================
# REAL MULTI-TASK DATASET ORACLE ENVIRONMENT
# =========================================================================
class DatasetThermodynamicEnvironment:
    """
    Hosts the real task distributions (pretrain, sft, cot) [1].
    Payouts are scaled inversely to the minimized Langevin energy loss.
    """
    def __init__(self, get_batch, batch_size, num_agents=4, device="cuda"):
        self.get_batch = get_batch
        self.batch_size = batch_size
        self.num_agents = num_agents
        self.device = device
        
        # Energetic state tracking
        self.energy = torch.full((batch_size, num_agents), 0.8, device=device)

    def reset(self):
        self.energy.fill_(0.8)
        # Fetch initial dataset prompt stream
        selected_task = np.random.choice(tasks, p=probs)
        return self.get_batch('train', task=selected_task)

    def step(self, shares, target_energy_loss, active_agent_idx):
        B = self.batch_size
        device = self.device
        
        # Constant baseline metabolic decay
        self.energy = (self.energy - 0.005).clamp(0.0, 1.0)
        
        # Reward payout inversely scaled to EBM energy minimization performance
        payout = torch.exp(-target_energy_loss) * 0.20
        
        batch_idx = torch.arange(B, device=device)
        self.energy[batch_idx, active_agent_idx] = (self.energy[batch_idx, active_agent_idx] + payout).clamp(0.0, 1.0)
        
        # Vectorized energy sharing between active and peer agents (No .item() stalls)
        for i in range(self.num_agents):
            is_sharing = (shares[:, i] == 1)
            share_amount = 0.05
            
            can_share = is_sharing & (self.energy[:, i] > share_amount)
            self.energy[can_share, i] -= share_amount
            
            share_slice = share_amount / (self.num_agents - 1)
            for j in range(self.num_agents):
                if i == j: continue
                self.energy[can_share, j] = (self.energy[can_share, j] + share_slice).clamp(0.0, 1.0)
                
        # Draw next prompt batch from the real dataset oracle stream
        selected_task = np.random.choice(tasks, p=probs)
        return self.get_batch('train', task=selected_task)

# =========================================================================
# GPU MEDOID POPULATION RESET SELECTION
# =========================================================================
@torch.no_grad()
def select_medoid_agent(agents_list):
    emas = torch.stack([agent.ema_states for agent in agents_list], dim=0)
    diff = emas.unsqueeze(1) - emas.unsqueeze(0)
    dist_matrix = torch.sum(diff ** 2, dim=-1)
    
    avg_dists = dist_matrix.mean(dim=-1, keepdim=True)
    mask_valid = dist_matrix <= (1.5 * avg_dists)
    
    masked_dists = dist_matrix.masked_fill(~mask_valid, 1e9)
    total_valid_dists = masked_dists.sum(dim=-1)
    
    medoid_idx = torch.argmin(total_valid_dists)
    return medoid_idx

# =========================================================================
# RESILIENT CRASH-RECOVERY EXECUTION PIPELINE
# =========================================================================
if __name__ == "__main__":
    gptconf = GPTConfig()
    
    # -------------------------------------------------------------
    # PHASE 1: Autoregressive Baseline Multi-Task Pretraining
    # -------------------------------------------------------------
    print("=== Phase 1: Pretraining Autoregressive Next-Byte Baseline ===")
    baseline = HRMAgent(gptconf).to(device)
    baseline_path = 'out/baseline_ckpt.pt'
    
    if os.path.exists(baseline_path):
        print(f"[*] Found existing Phase 1 baseline checkpoint at '{baseline_path}'. Skipping.")
        baseline.load_state_dict(torch.load(baseline_path, map_location=device))
    else:
        opt = torch.optim.AdamW(baseline.parameters(), lr=1e-3)
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
            
            if step % 1000 == 0:
                print(f"Pretrain Step {step:3d} | Task: {selected_task:<10} | Cross Entropy: {loss.detach().cpu().item():.4f}")
                
        torch.save(baseline.state_dict(), baseline_path)
        print(f"Phase 1 completed. Saved Baseline weights to '{baseline_path}'.\n")

    # -------------------------------------------------------------
    # PHASE 2: Cooperative Diffusion-EBM Active Learning
    # -------------------------------------------------------------
    print("=== Phase 2: Cooperative Diffusion-EBM Active Learning ===")
    agents = nn.ModuleList([HRMAgent(gptconf).to(device) for _ in range(4)])
    optimizers = [torch.optim.AdamW(a.parameters(), lr=1e-3) for a in agents]
    env = DatasetThermodynamicEnvironment(get_batch=get_batch, batch_size=batch_size, num_agents=4, device=device)
    
    latest_coop_path = 'out/latest_coop_checkpoint.pt'
    start_step = 1
    consecutive_steps = 0
    
    mem_keys = [torch.zeros(batch_size, 16, gptconf.n_embd, device=device) for _ in range(4)]
    mem_vals = [torch.zeros(batch_size, 16, gptconf.n_embd, device=device) for _ in range(4)]
    mem_mask = [torch.zeros(batch_size, 16, device=device) for _ in range(4)]
    
    if os.path.exists(latest_coop_path):
        print(f"[*] Found active recovery checkpoint: '{latest_coop_path}'. Resuming...")
        checkpoint = torch.load(latest_coop_path, map_location=device)
        start_step = checkpoint['step'] + 1
        consecutive_steps = checkpoint['consecutive_steps']
        agents.load_state_dict(checkpoint['agents_state_dict'])
        for i, opt_a in enumerate(optimizers):
            opt_a.load_state_dict(checkpoint['optimizers_state_dicts'][i])
        mem_keys = [mk.to(device) for mk in checkpoint['mem_keys']]
        mem_vals = [mv.to(device) for mv in checkpoint['mem_vals']]
        mem_mask = [mm.to(device) for mm in checkpoint['mem_mask']]
    else:
        print("[*] No existing cooperative checkpoint found. Initializing from baseline.")
        for a in agents:
            with torch.no_grad():
                a.token_proj.weight.copy_(a.wte.weight.t())
            
    x_bytes, y_bytes = env.reset()
    t0 = time.time()
    
    try:
        for step in range(start_step, iters_post):
            shares, talks = [], []
            active_agent_idx = step % 4
            active_agent_loss = torch.tensor(0.0, device=device)
            
            for i in range(4):
                agent = agents[i]
                opt_a = optimizers[i]
                opt_a.zero_grad(set_to_none=True)

                x_one_hot = F.one_hot(x_bytes, num_classes=gptconf.vocab_size).float()
                out = agent(x_one_hot, keys=mem_keys[i], vals=mem_vals[i], mask=mem_mask[i], is_text=False)
                
                share_act = torch.argmax(out['share'], dim=-1)
                talk_act = torch.argmax(out['talk'], dim=-1)
                shares.append(share_act)
                talks.append(talk_act)
                
                # Baseline sequence predictive loss
                with torch.no_grad():
                    latent_target = agent.token_proj(F.one_hot(y_bytes, num_classes=gptconf.vocab_size).float())
                    
                pred_loss = F.mse_loss(out['pred_state'], latent_target)
                total_loss = pred_loss + 0.1 * out['aux_loss']
                
                # -------------------------------------------------------------
                # ITERACTIVE LANGEVIN DIFFUSION REFINEMENT LOOP (EBM Task Phase)
                # -------------------------------------------------------------
                logits = out['proposal_logits']
                
                if i == active_agent_idx:
                    # Target is the complete true sequence offset we are trying to predict
                    target_tokens = y_bytes
                    
                    # Detach to build a leaf node on GPU for the energy gradient calculation
                    refined_logits = logits.detach().clone().requires_grad_(True)
                    
                    # Run K diffusion steps down the energy surface
                    for tick in range(gptconf.diffusion_ticks):
                        # Calculate Energy E (Cross entropy against dataset targets)
                        energy = F.cross_entropy(refined_logits.view(-1, gptconf.vocab_size), target_tokens.view(-1))
                        
                        # Compute energy surface gradient
                        energy_grad = torch.autograd.grad(energy, refined_logits)[0]
                        
                        # Apply Langevin update: gradient step + boredom-scaled stochastic noise
                        langevin_noise_scale = torch.sqrt(out['boredom_scale'].detach()) * 0.01
                        refined_logits = refined_logits - gptconf.langevin_lr * energy_grad + langevin_noise_scale * torch.randn_like(refined_logits)
                        
                    # Commit the refined state to compute final optimized EBM loss (used for rewards)
                    final_energy_loss = F.cross_entropy(refined_logits.view(-1, gptconf.vocab_size), target_tokens.view(-1))
                    
                    # To allow parameter updates from Langevin refinement, calculate:
                    # 1. Direct CE proposal loss to update parameters towards targets.
                    # 2. Amortization loss to match refined targets.
                    proposal_loss = F.cross_entropy(logits.view(-1, gptconf.vocab_size), target_tokens.view(-1))
                    amortization_loss = F.mse_loss(logits, refined_logits.detach())
                    
                    # Backpropagate combined parameters loss
                    joint_loss = total_loss + proposal_loss + 0.1 * amortization_loss
                    joint_loss.backward()
                    opt_a.step()
                    
                    active_agent_loss = final_energy_loss.detach()
                
                # GPU-bound memory writes (Zero .item() calls)
                with torch.no_grad():
                    batch_idx = torch.arange(batch_size, device=device)
                    write_slot = torch.argmax(out['mem_write'], dim=-1)
                    delete_slot = torch.argmax(out['mem_delete'], dim=-1)
                    
                    mem_mask[i][batch_idx, delete_slot] = 0.0
                    mem_keys[i][batch_idx, write_slot] = out['latent']
                    mem_vals[i][batch_idx, write_slot] = out['latent']
                    mem_mask[i][batch_idx, write_slot] = 1.0

            # Step environment using sharing vectors and prediction losses
            shares_tensor = torch.stack(shares, dim=1)
            x_bytes, y_bytes = env.step(shares_tensor, active_agent_loss, active_agent_idx)
            consecutive_steps += 1
            
            # -------------------------------------------------------------
            # PERIODIC EVALUATION & RECOVERY CHECKPOINTING
            # -------------------------------------------------------------
            if step % 100 == 0:
                avg_energy = env.energy.mean().item()
                
                # Scale-Invariant Learning Velocity Metric
                lp_vals = torch.stack([a.lp_ema for a in agents])
                mean_loss_0 = lp_vals.mean()
                mean_loss_1 = lp_vals.std()
                composite_score = - (mean_loss_1 ** 2) / (mean_loss_0 + 1e-8)
                
                if avg_energy < 0.15:
                    print(f"\n[x] Civilization collapsed at step {step}. Performing Medoid EGGROLL Reset.")
                    
                    med_idx = select_medoid_agent(agents)
                    medoid_state = agents[med_idx].state_dict()
                    
                    # Modify variables requiring gradients under an explicit no_grad block
                    with torch.no_grad():
                        for i, agent in enumerate(agents):
                            agent.load_state_dict(medoid_state)
                            for p in agent.parameters():
                                if p.requires_grad:
                                    p.add_(torch.randn_like(p) * 0.03)
                    
                    for i in range(4):
                        mem_keys[i].fill_(0.0)
                        mem_vals[i].fill_(0.0)
                        mem_mask[i].fill_(0.0)
                    
                    x_bytes, y_bytes = env.reset()
                    consecutive_steps = 0
                else:
                    # Write recovery state to avoid losing training progress on sudden interrupt
                    recovery_state = {
                        'step': step,
                        'consecutive_steps': consecutive_steps,
                        'agents_state_dict': agents.state_dict(),
                        'optimizers_state_dicts': [opt.state_dict() for opt in optimizers],
                        'mem_keys': [mk.clone() for mk in mem_keys],
                        'mem_vals': [mv.clone() for mv in mem_vals],
                        'mem_mask': [mm.clone() for mm in mem_mask],
                    }
                    torch.save(recovery_state, latest_coop_path)
                    print(f"Step {step:4d} | Lifetime: {consecutive_steps:3d} | Average Energy: {avg_energy:.3f} | Velocity: {composite_score.item():.4f} | State Saved.")
                    
    except KeyboardInterrupt:
        print("\n[!] Paused by user. Saving recovery state...")
        recovery_state = {
            'step': step,
            'consecutive_steps': consecutive_steps,
            'agents_state_dict': agents.state_dict(),
            'optimizers_state_dicts': [opt.state_dict() for opt in optimizers],
            'mem_keys': [mk.clone() for mk in mem_keys],
            'mem_vals': [mv.clone() for mv in mem_vals],
            'mem_mask': [mm.clone() for mm in mem_mask],
        }
        torch.save(recovery_state, latest_coop_path)
        print("[*] Resumption state saved successfully.")