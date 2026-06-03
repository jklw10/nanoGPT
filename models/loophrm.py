import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch._inductor.config.disable_cpp_codegen = True

# Custom import fallbacks
try:
    from parts.modules import HungryLinear, HungryPattLayer, PattLayer
    import parts.utils as utils
    Linear = HungryLinear
    Patt = HungryPattLayer
    HAS_PARTS = True
except ImportError:
    Linear = nn.Linear
    Patt = None
    HAS_PARTS = False

device          = 'cuda' if torch.cuda.is_available() else 'cpu'
bias            = False

# Configuration to target ~10M parameters
batch_size      = 64
block_size      = 128  # Bumped to 128 to better accommodate philosophical text sequences

# Configs for 10M parameters (3 tiers, each with 2 layers of 384 dim)
n_embd          = 256
n_head          = 8
n_layer_L1      = 2   
n_layer_L2      = 2   
n_layer_L3      = 2   
dropout         = 0.0
weight_decay    = 1e-1

# Cyclic Gating configuration
max_cycles      = 4     
gate_penalty    = 0.01  

# Two-Stage Training Schedule
pretrain_steps   = 30000  # Phase 1: OpenWebText
max_iters        = 40000  # Phase 2: 10,000 steps of Reasoning Fine-Tuning
warmup_iters     = 100    # Warmup for Phase 1
eval_interval    = 1000
eval_iters       = 100

learning_rate          = 5e-4  # Pretraining LR
learning_rate_finetune = 2e-4  # Fine-tuning LR (slightly lower for stability)
min_lr                 = 1e-5

# Flags from template
spl             = False
sspl            = False
wnll            = True
patt            = True and Patt

# Safe ASCII decoder helper that handles raw byte arrays safely
def safe_decode(indices):
    return "".join([chr(i) if 32 <= i < 127 or i == 10 else f"\\x{i:02x}" for i in indices])

class PhilosophyDataset:
    def __init__(self, block_size, device, file_path):
        self.block_size = block_size
        self.device = device
        self.vocab_size = 256  # Byte-level modeling
        
        # Binary tokenization at the byte level
        bin_path = file_path + ".bin"
        if not os.path.exists(bin_path):
            print(f"Tokenizing {file_path} to byte-level binaries (this happens once)...")
            with open(file_path, 'rb') as f:
                # Read the file in chunks and stream to a binary numpy array
                chunk_size = 1024 * 1024 * 64 # 64MB chunks
                with open(bin_path, 'wb') as bin_f:
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        # Convert raw byte chars to uint8 array
                        arr = np.frombuffer(chunk, dtype=np.uint8)
                        bin_f.write(arr.tobytes())
            print(f"Tokenization complete. Binary file created at {bin_path}")

        # Memory map the binary file so we don't load 6GB into active RAM
        self.data = np.memmap(bin_path, dtype=np.uint8, mode='r')
        self.data_len = len(self.data)

        print(f"Loaded memory-mapped dataset. Total tokens (bytes): {self.data_len:,}")

    def get_batch(self, split=None):
        # Sample random offsets across the massive file
        ix = torch.randint(0, self.data_len - self.block_size, (batch_size,))
        
        x_list = [torch.from_numpy((self.data[i:i+self.block_size]).astype(np.int64)) for i in ix]
        y_list = [torch.from_numpy((self.data[i+1:i+1+self.block_size]).astype(np.int64)) for i in ix]
        
        x = torch.stack(x_list).to(self.device)
        y = torch.stack(y_list).to(self.device)
        return x, y

if HAS_PARTS:
    # Attempt to load openwebtext
    try:
        decode, _ = utils.get_meta("openwebtext", always_byte=True)
        pretrain_get_batch = utils.get_get_batch(block_size, batch_size, "openwebtext", device, always_byte=True)
        print("Successfully configured Phase 1 (OpenWebText pretraining).")
    except Exception as e:
        print(f"Error loading OpenWebText: {e}. Skipping straight to Reasoning Phase.")
        pretrain_get_batch = None
        pretrain_steps = 0
else:
    # Fallback to ASCII decoder
    decode = safe_decode
    pretrain_get_batch = None
    pretrain_steps = 0
    print("Parts/Utils not found. Skipping Phase 1 (Pretraining) and starting on Reasoning task.")

# Stage-Aware Learning Rate Scheduler
def get_lr(it):
    if it < pretrain_steps:
        # Phase 1 Cosine Schedule
        if it < warmup_iters:
            return learning_rate * (it + 1) / (warmup_iters + 1)
        decay_ratio = (it - warmup_iters) / (pretrain_steps - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
        return min_lr + coeff * (learning_rate - min_lr)
    else:
        # Phase 2 Cosine Schedule (Warmup and decay reset at transition)
        it_phase2 = it - pretrain_steps
        warmup_phase2 = 500  # Give the model 500 steps to gently adjust to the new task
        total_phase2 = max_iters - pretrain_steps
        if it_phase2 < warmup_phase2:
            return learning_rate_finetune * (it_phase2 + 1) / (warmup_phase2 + 1)
        
        decay_ratio = (it_phase2 - warmup_phase2) / max(1, total_phase2 - warmup_phase2)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
        return min_lr + coeff * (learning_rate_finetune - min_lr)

philoset_train = None
philoset_val = None
# Wrapper to dynamically select the batch depending on progress
def dynamic_get_batch(split):
    # Fall back to task dataset if we are past pretraining steps OR if pretrain data is unavailable
    if iter_tracker < pretrain_steps and pretrain_get_batch is not None:
        return pretrain_get_batch(split)
    else:
        if split == "train":
            if not philoset_train:
                philoset_train = PhilosophyDataset(block_size=block_size, device=device, file_path="data/philo/train.txt")
            dataset= philoset_train
        else:
            if not philoset_val:
                philoset_val = PhilosophyDataset(block_size=block_size, device=device, file_path="data/philo/val.txt")
            dataset= philoset_val
        return dataset.get_batch(split)

@dataclass
class GPTConfig:
    block_size: int = 128
    vocab_size: int = 256
    n_embd: int     = 256
    dropout: float  = 0.0
    bias: bool      = False
    max_cycles: int = 4
    gate_penalty: float = 0.01
    
    device          = 'cuda'
    optimizer: str  = 'adam'
    batch_size      = 64
    step            = 0

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias_flag):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias_flag else None
       
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

# Rotary Positional Embedding (RoPE) Implementation
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Calculate theta frequency values
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings, device=device)

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
    # Splits the final dimension in half and rotates it
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # Reshape cos and sin for broadcasting to shape (1, 1, T, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % n_head == 0
        self.q_attn = Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_attn = Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_attn = Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = n_head 
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim = self.n_embd // self.n_head 
        
        self.qm = nn.Parameter(torch.ones(self.head_dim).normal_(mean=1, std=0.1))
        
        # Initialize the RoPE Module per attention layer
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=config.block_size)
    
    def forward(self, x, causal=True):
        B, T, C = x.size() 
        q = self.q_attn(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_attn(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_attn(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply RoPE on queries and keys
        cos, sin = self.rotary_emb(q, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        q_scale = math.log(T) * self.qm
        y = torch.nn.functional.scaled_dot_product_attention(
            q * q_scale, k, v, attn_mask=None, 
            dropout_p=self.dropout if self.training else 0, is_causal=causal
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C) 
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        up = 4
        self.c_fc    = Linear(config.n_embd, up * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = Linear(up * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = Attention(config)
        self.ln_1 = LayerNorm(config.n_embd, config.bias)
        self.ln_2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)
        
    def forward(self, x, causal=True):
        x = x + self.attn(self.ln_1(x), causal)
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.n_embd = config.n_embd
        self.config = config
        
        # Defining three distinct, stacked computational tiers
        self.L1_blocks = nn.ModuleList([Block(config) for _ in range(n_layer_L1)])
        self.L2_blocks = nn.ModuleList([Block(config) for _ in range(n_layer_L2)])
        self.L3_blocks = nn.ModuleList([Block(config) for _ in range(n_layer_L3)])
        
        # gate_head maps to [halt, continue] Q-values
        self.gate_head = nn.Linear(config.n_embd, 2, bias=config.bias)
        
        # wpe (absolute positional embedding) is removed to let RoPE manage sequence tracking
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            ln_f = LayerNorm(config.n_embd, config.bias),
        ))

        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight 
        
        if HAS_PARTS:
            self.wnl = utils.wnormlearnloss(config.n_embd)
        
        self.apply(self._init_weights)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        
        tok_emb = self.transformer.wte(idx)
        # Sequence input is dropped straight to the layers without additive absolute embeddings
        x = self.transformer.drop(tok_emb)
        
            
        active_mask = torch.ones(b, t, 1, device=device, dtype=x.dtype)
        final_output = torch.zeros_like(x)
        
        # To accumulate outputs and gate choices for Deep Supervision/ACT
        gate_logits_list = []
        cycle_logits_list = []
        active_mask_list = []
        
        for cycle in range(self.config.max_cycles):
            x_L1 = x
            for block in self.L1_blocks:
                x_L1 = block(x_L1)
            
            if self.training and HAS_PARTS and wnll:
                wnl = self.wnl(x_L1)
            x_L2 = x_L1
            for block in self.L2_blocks:
                x_L2 = block(x_L2)
                
            x_L3 = x_L2
            for block in self.L3_blocks:
                x_L3 = block(x_L3)
                
            gate_logits = self.gate_head(x_L3)  # (B, T, 2)
            gate_logits_list.append(gate_logits)
            active_mask_list.append(active_mask)
            
            # Projects to vocabulary at this cycle for Deep Supervision
            cycle_logits = self.lm_head(self.transformer.ln_f(x_L3))
            cycle_logits_list.append(cycle_logits)
            
            if cycle == self.config.max_cycles - 1:
                halt = torch.ones(b, t, 1, device=device, dtype=x.dtype)
                loop = torch.zeros(b, t, 1, device=device, dtype=x.dtype)
            else:
                if self.training:
                    # Gumbel-Softmax for differentiable routing during backprop
                    probs = F.gumbel_softmax(gate_logits, tau=1.0, hard=True, dim=-1)
                    halt = probs[..., 0:1] 
                    loop = probs[..., 1:2] 
                else:
                    # Deterministic early stopping based on Q-values
                    halt_decision = (gate_logits[..., 0:1] > gate_logits[..., 1:2]).to(x.dtype)
                    halt = halt_decision
                    loop = 1.0 - halt
                    
            newly_halted = active_mask * halt
            final_output = final_output + newly_halted * x_L3
            
            active_mask = active_mask * loop
            x = x_L3
            
            # Early break at inference time if all positions decided to halt
            if not self.training and (active_mask == 0.0).all():
                break
            
        x = self.transformer.ln_f(final_output)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            # 1. Deep Supervision Loss (CE at each cycle step)
            ce_losses = []
            for cycle in range(self.config.max_cycles):
                cycle_logits = cycle_logits_list[cycle]
                active_m = active_mask_list[cycle]
                
                # Element-wise cross-entropy across positions
                cycle_ce = F.cross_entropy(
                    cycle_logits.reshape(-1, cycle_logits.shape[-1]), 
                    targets.reshape(-1), 
                    reduction='none'
                ).reshape(b, t, 1)
                
                # Apply active step masking
                masked_ce = (cycle_ce * active_m).sum() / (active_m.sum() + 1e-8)
                ce_losses.append(masked_ce)
                
            loss = sum(ce_losses) / len(ce_losses)
            
            # 2. Adaptive Computation Time (ACT) Q-Learning Losses
            q_losses = []
            for cycle in range(self.config.max_cycles):
                gate_logits = gate_logits_list[cycle]
                cycle_logits = cycle_logits_list[cycle]
                active_m = active_mask_list[cycle]
                
                # Halt target: 1.0 if correct prediction, 0.0 otherwise
                correct_preds = (cycle_logits.argmax(dim=-1) == targets).float().unsqueeze(-1)
                
                # ACT Halt Loss (Binary Cross Entropy with Logits)
                halt_loss = F.binary_cross_entropy_with_logits(
                    gate_logits[..., 0:1], 
                    correct_preds, 
                    reduction='none'
                )
                masked_halt_loss = (halt_loss * active_m).sum() / (active_m.sum() + 1e-8)
                q_losses.append(0.5 * masked_halt_loss)
                
                # ACT Continue Loss (Temporal Difference Q-learning step)
                if cycle < self.config.max_cycles - 1:
                    next_gate_logits = gate_logits_list[cycle + 1]
                    
                    if cycle + 1 == self.config.max_cycles - 1:
                        # Final step's target is strictly the next step's halt value
                        target_continue = torch.sigmoid(next_gate_logits[..., 0:1])
                    else:
                        # Standard Bellman expectation step: max(Q(s', halt), Q(s', continue))
                        next_q_values = torch.sigmoid(next_gate_logits)
                        target_continue = torch.max(next_q_values[..., 0:1], next_q_values[..., 1:2])
                        
                    continue_loss = F.binary_cross_entropy_with_logits(
                        gate_logits[..., 1:2], 
                        target_continue.detach(),  # Detach target tensor for stability
                        reduction='none'
                    )
                    masked_continue_loss = (continue_loss * active_m).sum() / (active_m.sum() + 1e-8)
                    q_losses.append(0.5 * masked_continue_loss)
                    
            loss = loss + sum(q_losses)
            
            if self.training:
                if HAS_PARTS and wnll:
                    mean_surprise = ((1.0-wnl.detach())/2.0).mean()*0.99
                    loss = loss / (1 + wnl.detach()*mean_surprise)
                    loss = (loss + wnl).mean()
                
                # Active Loop Regularization (helps optimize resource usage)
                if len(active_mask_list) > 1:
                    loop_penalty = torch.stack(active_mask_list[:-1]).mean() * self.config.gate_penalty
                    loss = loss + loop_penalty
        else:
            # Yield final projection only for the target token during auto-regressive decoding
            logits = logits[:, [-1], :]
            
        return logits, loss
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, Linear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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
                {'params': nodecay_params,  'weight_decay': 0.0},
                {'params': decay_params,    'weight_decay': wd},
        ]
        return torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.999))
vocab_size = 256
# Initialize model
model_args = dict(n_embd=n_embd, block_size=block_size, bias=bias, vocab_size=vocab_size, 
                  dropout=dropout, max_cycles=max_cycles, gate_penalty=gate_penalty) 
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)
optimizer = model.configure_optimizers(lr=learning_rate, wd=weight_decay)

torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True 
torch.set_float32_matmul_precision('medium')

model.compile()
print(f"Starting curriculum training on {device}")
print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

t0 = time.time()
t1 = time.time()
t2 = time.time()

def eval(max_iters, get_batch_fn, model, optimizer, t0, t1, t2, iter):
    t2 = time.time()
    model.eval()
    losses = {}
    for split in ['train', 'val']:
        losses[split] = 0.0
        for k in range(eval_iters):
            X_val, Y_val = get_batch_fn(split)
            with torch.no_grad():
                _, loss = model(X_val, Y_val)
            losses[split] += loss.detach().item()
        losses[split] /= eval_iters
    optimizer.zero_grad(set_to_none=True)
    t1 = time.time()
    print(f"step {iter:{len(str(max_iters))}d}: ", end="")
    print(f"train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}, ", end="")
    print(f"time: {(t2-t0)*1000:.0f} ms, Tval: {(t1-t2)*1000:.0f} ms")
    model.train()
    t0 = time.time()
    return t0, t1, t2

# Autoregressive generation utility for the byte-level HRM model
@torch.no_grad()
def generate_text(model, prompt, max_new_tokens=64, temperature=0.8, top_k=None):
    model.eval()
    if not prompt:
        prompt = "\n"
        
    # Map prompt characters safely into 0-255 raw byte limits
    indices = [ord(char) % 256 for char in prompt]
    x = torch.tensor([indices], dtype=torch.long, device=device)
    
    for _ in range(max_new_tokens):
        # Crop context token limits according to the block size config
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]
        
        # Forward pass on evaluation: targets is None, which outputs logits shape (B, 1, Vocab_size)
        logits, _ = model(x_cond)
        
        # Retrieve final token prediction and scale by the generation temperature
        logits = logits[:, 0, :] / temperature
        
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, next_token), dim=1)
        
    return safe_decode(x[0].tolist())

# Evaluation script for multi-step generative tasks and persona mapping
def run_generation_suite(model, phase_name):
    print(f"\n" + "="*80)
    print(f" GENERATION EVALUATION FOR PHASE: {phase_name.upper()} ")
    print("="*80)
    
    # Task 1: Unconditioned Raw Dynamic Generation (Random Byte Seed)
    random_byte = torch.randint(0, 256, (1,)).item()
    # Normalize unprintable ASCII values to space, otherwise use its actual raw byte char representation
    seed_char = chr(random_byte) if (32 <= random_byte < 127 or random_byte == 10) else " "
    raw_output = generate_text(model, seed_char, max_new_tokens=64, temperature=0.8)
    print(f"\n[Test 1] Random Data Start (Seed byte value {random_byte}):")
    print(f"--> \"{raw_output}\"")
    
    # Task 2: Metaphysics Inquiry
    persona_output = generate_text(model, "Q: What is the nature of existence?\nA:", max_new_tokens=64, temperature=0.7)
    print(f"\n[Test 2] Metaphysical Prompt (\"What is the nature of existence?\"):")
    print(f"--> \"{persona_output}\"")
    
    # Task 3: Ethical Dilemmas & Definitions
    prompts = [
        "Q: Is lying ever justified?\nA:",
        "Q: What is justice?\nA:"
    ]
    for prompt in prompts:
        ethics_output = generate_text(model, prompt, max_new_tokens=64, temperature=0.7)
        print(f"\n[Test 3] Ethical/Moral Inquiry:")
        print(f"--> Prompt: \"{prompt}\"")
        print(f"--> Response: \"{ethics_output}\"")
        
    print("="*80 + "\n")
    model.train()

# Global reference for tracking current step within the generator wrapper
iter_tracker = 0

for iter in range(max_iters):
    iter_tracker = iter
    
    # Check for the task boundary transition
    if iter == pretrain_steps and pretrain_steps > 0:
        print("\n" + "="*50)
        print(">>> PRETRAINING PHASE COMPLETE. SWAPPING TO REASONING TASK <<<")
        print("="*50 + "\n")
        
        # Save Pretraining Checkpoint
        pretrain_checkpoint = "checkpoint_pretrain.pt"
        torch.save({
            'iter': iter,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': model_args
        }, pretrain_checkpoint)
        print(f"Saved Phase 1 pretraining checkpoint to: {pretrain_checkpoint}")
        
        # Run generation suite to see emergence of base linguistic capabilities
        run_generation_suite(model, "Phase 1 Pretraining")
        
        # Trigger an intermediate evaluation on the boundary
        t0, t1, t2 = eval(max_iters, dynamic_get_batch, model, optimizer, t0, t1, t2, iter)
    
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    torch.compiler.cudagraph_mark_step_begin()
    X, Y = dynamic_get_batch('train')
    logits, loss = model(X, Y)
    loss.backward()
    
    # Clip gradients to ensure stable finetuning transitions
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()
    
    if iter % eval_interval == 0 or iter == max_iters - 1:
        t0, t1, t2 = eval(max_iters, dynamic_get_batch, model, optimizer, t0, t1, t2, iter)
    optimizer.zero_grad(set_to_none=True)

# Final Post-Training Cleanup and Checklist
print("\n" + "="*50)
print(">>> TRAINING COMPLETE. SAVING FINAL CHECKPOINT & EVALUATING <<<")
print("="*50 + "\n")

# Save Fine-tuned Model Checkpoint
final_checkpoint = "checkpoint_finetune.pt"
torch.save({
    'iter': max_iters,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': model_args
}, final_checkpoint)
print(f"Saved Phase 2 reasoning checkpoint to: {final_checkpoint}")

# Final generation suite execution to observe changes in planning/entropy
run_generation_suite(model, "Phase 2 Reasoning Fine-Tuning")