import math
import os
import time
import pickle
import numpy as np
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from parts import quantizer, utils, modules
from parts.modules import tokenformer_theta

# PyTorch compile configuration
torch._inductor.config.disable_cpp_codegen = True

# =====================================================================
# 1. HYPERPARAMETER CONFIGURATION (GPT style)
# =====================================================================

dataset         = "openwebtext"
data_dir        = os.path.join('data', dataset)
device          = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training controls matching soloblock.py
batch_size      = 64
block_size      = 64  # sequence context length
tbptt_horizon   = 10

n_head          = 8
n_embd          = 32  # core hidden dimension
num_nodes       = 32   # number of active inference nodes
num_queries     = 4    # size of wishlist query group
thinking_steps  = 5    # iteration depth per sequence token step

dropout         = 0.0
weight_decay    = 1e-1
eval_interval   = 1000
eval_iters      = 200
warmup_iters    = 100
max_iters       = 50000
lr_decay_iters  = 30000

learning_rate   = 5e-4
min_lr          = 5e-5

@dataclass
class CognitiveGPTConfig:
    block_size: int     = block_size
    vocab_size: int     = 50304  
    n_embd: int         = n_embd
    num_nodes: int      = num_nodes
    num_queries: int    = num_queries
    thinking_steps: int = thinking_steps
    n_head:int          = n_head
    dropout:int         = dropout
    bias:bool = False

    untanh_slope: float = 1.03
    w_scale: float      = 0.63
    ema_speed: float    = 0.20
    rew_sens: float     = 1.36
    n_scale: float      = 0.01
    c_moment: float     = 0.80
    a_variance: float   = 0.00
    sprs: float         = 0.5


decode, vocab_size = utils.get_meta(dataset, always_byte=True)
get_batch = utils.get_get_batch(block_size, batch_size, dataset, device, always_byte=True)

# =====================================================================
# 3. CORE ACTIVE INFERENCE MATHEMATICAL LAYERS
# =====================================================================

def sparse_route(router, node_count, spr):
    probs = torch.sigmoid(router)
    mask = quantizer.topkhotSTE(router, int(math.sqrt(node_count)))
    sparse_probs = mask * probs
    sparse_probs = sparse_probs / (sparse_probs.sum(dim=-1, keepdim=True) + 1e-8)
    
    flat_probs = torch.ones_like(probs) / node_count
    route = ((1.0 - spr) * sparse_probs) + (spr * flat_probs)
    return route, 0.0, mask

class NodeParams3(nn.Module):
    def __init__(self, batch_size, num_nodes, in_dim, out_dim, config, device):
        super().__init__() 
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        
        base_weight = torch.randn(num_nodes, in_dim, out_dim, device=device) / math.sqrt(self.in_dim)
        self.weight = nn.Parameter(base_weight)
        
        self.register_buffer('error_ema', 0.01 + 0.01 * torch.randn(batch_size, num_nodes, out_dim, device=device))
        
        self.untanh_slope = torch.tensor(config.untanh_slope, device=device)
        self.w_scale = torch.tensor(config.w_scale, device=device)
        
        # Norm weight cache to avoid redundant evaluations inside temporal loops
        self.norm_weight_cache = None

    def precompute_weight(self):
        self.norm_weight_cache = F.normalize(self.weight, p=2, dim=-2) * self.w_scale

    def clear_weight_cache(self):
        self.norm_weight_cache = None

    def forward(self, x):
        if self.norm_weight_cache is not None:
            norm_weight = self.norm_weight_cache
        else:
            norm_weight = F.normalize(self.weight, p=2, dim=-2) * self.w_scale
            
        raw_pred = torch.einsum('bnd,nde->bne', x, norm_weight)
        
        bound = math.sqrt(2.0 + math.sqrt(2.0))
        out = utils.buntanh(raw_pred, self.untanh_slope, bound)
        return out

class StandardNodeNet3(nn.Module):
    def __init__(self, batch_size, num_nodes, dim, config, device):
        super().__init__()
        self.N = num_nodes
        self.D = dim
        self.device = device
        self.config = config
        
        self.w1 = NodeParams3(batch_size, num_nodes, dim, dim, config, device)
        self.router = NodeParams3(batch_size, num_nodes, dim, num_nodes, config, device)
        
        self.register_buffer('state', 1e-5 * torch.randn(batch_size, self.N, dim, device=device))
        self.register_buffer('output', 1e-5 * torch.randn(batch_size, self.N, dim, device=device))
        self.register_buffer('error_ema', torch.ones(batch_size, self.N, device=device))
        self.register_buffer('lp_ema', torch.zeros(batch_size, self.N, device=device))
        self.register_buffer('step_counter', torch.tensor(0.0, device=device))
        
        self.c_moment = config.c_moment
        self.a_variance = config.a_variance
        self.lp_alpha = 0.1 
        self.sprs = config.sprs
        
        # Precompute static input layout dimensions
        input_dim = dim * (1 + config.num_queries)
        self.num_input_nodes = math.ceil(input_dim / self.D)
        self.pad_len = self.num_input_nodes * self.D - input_dim

    def precompute_weights(self):
        self.w1.precompute_weight()
        self.router.precompute_weight()

    def clear_weight_caches(self):
        self.w1.clear_weight_cache()
        self.router.clear_weight_cache()

    def reset_states(self, b_size):
        """Dynamically re-allocates state buffers to match incoming batch size."""
        self.state = 1e-5 * torch.randn(b_size, self.N, self.D, device=self.device)
        self.output = 1e-5 * torch.randn(b_size, self.N, self.D, device=self.device)
        self.error_ema = torch.ones(b_size, self.N, device=self.device)
        self.lp_ema = torch.zeros(b_size, self.N, device=self.device)
        self.step_counter = torch.tensor(0.0, device=self.device)

    def forward(self, env_input):
        self.step_counter = self.step_counter + 1.0
        batch_size, input_dim = env_input.shape
        
        prediction = self.w1(self.state) 
        router = self.router(self.state)

        curiosity_bias = F.relu(self.lp_ema).unsqueeze(1) * 5.0 
        route_logits = router + curiosity_bias
        route, aux, mask = sparse_route(route_logits, self.N, self.sprs)
        
        current_var = route.var(dim=[-1], keepdim=True)
        vd = F.relu(self.a_variance - current_var)
        
        V = self.output
        base_noise = torch.sqrt(vd) * torch.randn_like(V)
        
        # Compute boredom scale once and reuse
        boredom_scale = torch.exp(-50.0 * F.relu(self.lp_ema)).unsqueeze(-1) 
        boredom_noise = base_noise * boredom_scale
        
        target = torch.matmul(route, V) + boredom_noise
        target = utils.softsign(target)

        if self.pad_len > 0:
            env_input_padded = F.pad(env_input, (0, self.pad_len))
        else:
            env_input_padded = env_input
            
        env_input_reshaped = env_input_padded.view(batch_size, self.num_input_nodes, self.D)
        
        if self.num_input_nodes < self.N:
            target = torch.cat([env_input_reshaped, target[:, self.num_input_nodes:, :]], dim=1)
        else:
            target = env_input_reshaped

        loss_w1 = F.mse_loss(prediction, target, reduction='none')

        with torch.no_grad():
            current_error = loss_w1.mean(dim=-1).detach()
            current_lp = self.error_ema - current_error
            self.error_ema = self.error_ema * (1.0 - self.lp_alpha) + current_error * self.lp_alpha
            self.lp_ema = self.lp_ema * (1.0 - self.lp_alpha) + current_lp * self.lp_alpha

        total_loss = (loss_w1).sum() + aux
        
        errm = torch.sqrt(boredom_scale) * 0.01
        error_pass = torch.randn_like(target) * errm
        target = target + error_pass

        # Use out-of-place assignment to avoid modifying variables needed for gradient computation
        self.output = prediction.detach()
        self.state = target.detach()

        return total_loss

class CognitiveGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim = config.n_embd
        self.num_queries = config.num_queries
        self.thinking_steps = config.thinking_steps
        
        # Core active inference node net
        self.core = StandardNodeNet3(
            batch_size=batch_size,
            num_nodes=config.num_nodes,
            dim=config.n_embd,
            config=config,
            device=device
        )
        
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embedding = nn.Embedding(config.block_size, config.n_embd)
        
        # Final linear prediction head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        
        # Surprise tracking loss
        self.wnl = utils.wnormlearnloss(config.n_embd)
        self.retriever = modules.CrossAttention(config)

    def retrieve_token(self, query, keys_sliced, context_embeds_sliced, k=3):
        
        #scores = torch.bmm(query, keys_sliced.transpose(1, 2)) / (self.dim ** 0.5)
        #probs1 = tokenformer_theta(scores, dim=-1)
        #mask = quantizer.topkhotSTE(probs1)
        #masked_scores = scores.masked_fill(mask == 0, -1e9)
        #probs2 = tokenformer_theta(masked_scores, dim=-1)
        #
        ## Straight-Through Estimator: Sparse forward path, dual gradients backward path
        #attn_weights = probs2 + probs1 - probs1.detach()
        #
        #retrieved_representation = torch.bmm(attn_weights, context_embeds_sliced)
        return self.retriever(keys_sliced, query)
    def forward(self, idx, targets=None):
        batch_size, seq_len = idx.size()
        device = idx.device
        
        total_loss = torch.tensor(0.0, device=device)
        logits_history = []
        
        # Setup initial query results container [batch_size, 4, dim]
        R = torch.zeros(batch_size, self.num_queries, self.dim, device=device)
        
        # Dynamically reset node-routing states on incoming batches
        self.core.reset_states(batch_size)
        
        # Vectorized Precomputation (Global Context)
        all_context_embeds = self.embedding(idx)  # [batch_size, seq_len, dim]
        positions = torch.arange(seq_len, device=device)
        all_pos_embeds = self.pos_embedding(positions).unsqueeze(0)  # [1, seq_len, dim]
        all_keys = all_context_embeds + all_pos_embeds  # [batch_size, seq_len, dim]
        
        # --- TBPTT HORIZON CONFIGURATION ---
        # e.g., only calculate gradients on the last 10 tokens
         
        warmup_end = max(0, seq_len - tbptt_horizon)
        
        # Precompute normalization weights to bypass redundant calculation loop
        self.core.precompute_weights()
        
        try:
            # --- PHASE 1: RECURRENT STATE WARM-UP (no_grad) ---
            if warmup_end > 0:
                with torch.no_grad():
                    for t in range(warmup_end):
                        slot_0 = all_context_embeds[:, t]
                        
                        for think_step in range(self.thinking_steps):
                            flat_R = R.flatten(start_dim=1)
                            env_input = torch.cat([slot_0, flat_R], dim=-1)
                            
                            # Forward pass updates core.state and core.output, no gradients tracked
                            _ = self.core(env_input)
                            
                            read_wish = self.core.output[:, 1+self.num_queries:1+2*self.num_queries, :].detach()
                            keys_sliced = all_keys[:, :t+1]
                            context_embeds_sliced = all_context_embeds[:, :t+1]
                            R = self.retrieve_token(read_wish, keys_sliced, context_embeds_sliced)
                            
                        # Keep track of warm-up logits for shape-matching output checks
                        next_token_representation = self.core.output[:, 0, :]
                        step_logits = self.lm_head(next_token_representation)
                        logits_history.append(step_logits)
            
            # --- PHASE 2: BPTT ACTIVE TRAINING (with gradients) ---
            for t in range(warmup_end, seq_len):
                slot_0 = all_context_embeds[:, t]
                
                # --- ITERATIVE THINKING COMPUTE PHASE ---
                for think_step in range(self.thinking_steps):
                    flat_R = R.flatten(start_dim=1)
                    env_input = torch.cat([slot_0, flat_R], dim=-1)
                    
                    # Forward pass optimizing internal predictive coding states (gradients tracked)
                    core_loss = self.core(env_input)
                    
                    # Scale the step loss relative to our BPTT training window size
                    total_loss = total_loss + (core_loss / (tbptt_horizon * self.thinking_steps * batch_size))
                    
                    read_wish = self.core.output[:, 1+self.num_queries:1+2*self.num_queries, :].detach()
                    
                    # Can still retrieve from the entire history (all_keys goes up to t+1)
                    keys_sliced = all_keys[:, :t+1]
                    context_embeds_sliced = all_context_embeds[:, :t+1]
                    R = self.retrieve_token(read_wish, keys_sliced, context_embeds_sliced)
                    
                # --- PREDICTION PHASE ---
                next_token_representation = self.core.output[:, 0, :]
                step_logits = self.lm_head(next_token_representation)
                logits_history.append(step_logits)
                
                if targets is not None:
                    step_target = targets[:, t]
                    ce_loss = F.cross_entropy(step_logits, step_target)
                    total_loss = total_loss + (ce_loss / tbptt_horizon)
                    
        finally:
            self.core.clear_weight_caches()
            
        logits = torch.stack(logits_history, dim=1)
        return logits, total_loss

    def configure_optimizers(self, lr, wd):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = []
        nodecay_params = []
        for n, p in param_dict.items():
            if p.dim() >= 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)
        optim_groups = [
            {'params': nodecay_params, 'weight_decay': 0.0},
            {'params': decay_params,   'weight_decay': wd},
        ]
        return torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.999))

# =====================================================================
# 6. TRAINING INITIALIZATION & MAIN EXECUTION
# =====================================================================

# Build config and model
gptconf = CognitiveGPTConfig(vocab_size=vocab_size)
model = CognitiveGPT(gptconf)
model.to(device)

optimizer = model.configure_optimizers(lr=learning_rate, wd=weight_decay)

# Precision settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

try:
    model = torch.compile(model)
    print("Dynamically compiled the model using PyTorch Compiler.")
except Exception as e:
    print(f"[Warning] PyTorch compilation skipped/unsupported. Eager execution: {e}")

print(f"Starting training loop on {device}...")
print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    if it > lr_decay_iters:
        return min_lr
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
    return min_lr + coeff * (learning_rate - min_lr)

def eval_model(iter, t0):
    t2 = time.time()
    model.eval()
    losses = {'train': 0.0, 'val': 0.0}
    
    for split in ['train', 'val']:
        for _ in range(eval_iters):
            X_val, Y_val = get_batch(split)
            with torch.no_grad():
                _, loss = model(X_val, Y_val)
            losses[split] += loss.detach().item()
        losses[split] /= eval_iters
        
    optimizer.zero_grad(set_to_none=True)
    t1 = time.time()
    print(f"step {iter:{len(str(max_iters))}d}: ", end="")
    print(f"train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}, ", end="")
    print(f"time: {(t2 - t0)*1000:.0f} ms, Tval: {(t1 - t2)*1000:.0f} ms")
    model.train()
    return time.time()

t0 = time.time()

for iter in range(max_iters):
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    torch.compiler.cudagraph_mark_step_begin()
    
    X, Y = get_batch('train')
    logits, loss = model(X, Y)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    
    if iter % eval_interval == 0 or iter == max_iters - 1:
        t0 = eval_model(iter, t0)