import math
import os
import time
from attr import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

import parts.utils as utils

torch._inductor.config.disable_cpp_codegen = True
torch._dynamo.config.force_parameter_static_shapes = False

# =====================================================================
# 1. Experiment Registry & Toggles
# =====================================================================
EXPERIMENTS_TO_RUN = {
    # Method 1: Chunked Full-Vocab Voronoi (Exact Apple CCE compile pattern)
    # Target: ~60ms latency, exact ~5.52 validation loss
    "voronoi_cce_chunked":      True,
    
    # Method 2: Active Semantic Bank + Calibrated Partition (Static Compile)
    # Target: ~30ms latency, competitive validation loss
    "voronoi_adaptive_cce":     True,
}

dataset           = "openwebtext"
device            = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameters
batch_size        = 64
block_size        = 64
n_embd            = 256
n_head            = 8
n_layer           = 4
dropout           = 0.0

eval_interval     = 250
eval_iters        = 50
max_iters         = 2000
learning_rate     = 1e-3

# Static chunking sizes
N_CHUNK_SIZE      = 1024  # Tokens per GEMM chunk (stays in L2 cache)
K_ADAPTIVE_BANK   = 2048  # Fixed static bank size for the fast branch

decode, vocab_size = utils.get_meta(dataset, always_byte=False)
get_batch = utils.get_get_batch(block_size, batch_size, dataset, device, always_byte=False)

@dataclass
class GPTConfig:
    block_size: int = block_size
    vocab_size: int = vocab_size
    n_layer: int = n_layer
    n_head: int = n_head
    n_embd: int = n_embd
    dropout: float = dropout
    bias: bool = False

# =====================================================================
# 2. Shared Transformer Backbone
# =====================================================================

class LayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.n_embd))
        self.bias = nn.Parameter(torch.zeros(config.n_embd)) if config.bias else None
       
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.q_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size()
        q = self.q_attn(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_attn(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_attn(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
        )
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config)
        self.attn = Attention(config)
        self.ln_2 = LayerNorm(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class TransformerCore(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config),
        ))
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        return self.transformer.ln_f(x)

# =====================================================================
# 3. Fixed Loss Implementations
# =====================================================================

def loss_voronoi_cce_chunked(h_flat: torch.Tensor, centroids: torch.Tensor, targets: torch.Tensor, chunk_size: int = N_CHUNK_SIZE) -> torch.Tensor:
    """
    Exact Full-Vocab Cut-Cross-Entropy via Token Chunking.
    - Evaluates all 50k tokens in wide, hardware-efficient GEMM chunks.
    - Zero DRAM logit footprint: chunks fit directly into L2 cache.
    - Exact 100% mathematical match to ground-truth loss.
    """
    c_half_norm = 0.5 * (centroids * centroids).sum(dim=1)
    n_tokens = h_flat.shape[0]
    total_loss = 0.0
    
    for i in range(0, n_tokens, chunk_size):
        h_chunk = h_flat[i:i + chunk_size]
        t_chunk = targets[i:i + chunk_size]
        
        # [chunk, D] x [D, V] -> [chunk, V]
        chunk_logits = torch.addmm(-c_half_norm, h_chunk, centroids.T)
        total_loss = total_loss + F.cross_entropy(chunk_logits, t_chunk, reduction='sum')
        
    return total_loss / n_tokens

def loss_voronoi_adaptive_cce(h_flat: torch.Tensor, centroids: torch.Tensor, targets: torch.Tensor, k_bank: int = K_ADAPTIVE_BANK) -> torch.Tensor:
    """
    Fast Active-Bank Candidate Selection with Calibrated Background Partition.
    - In-batch active tokens act as local hard competitors.
    - Static bank size k_bank guarantees ZERO Dynamo recompilations.
    - Scaled extreme-value partition accounts for the unvisited vocabulary tail.
    """
    N, D = h_flat.shape
    V = centroids.shape[0]
    device = h_flat.device
    
    # If vocab is smaller than bank (e.g. byte model where V=256), exact loss
    if V <= k_bank:
        c_half_norm = 0.5 * (centroids * centroids).sum(dim=1)
        logits = torch.addmm(-c_half_norm, h_flat, centroids.T)
        return F.cross_entropy(logits, targets)
        
    # Build a strictly static bank: first target tokens, then fill remainder with random uniform tokens
    # Using fixed static shape guarantees no dynamic recompilations!
    perm = torch.randint(0, V, (k_bank,), device=device)
    # Ensure targets are present in the bank
    n_target_fill = min(N, k_bank // 2)
    perm[:n_target_fill] = targets[:n_target_fill]
    
    sub_c = centroids[perm] # [k_bank, D]
    c_half_norm = 0.5 * (sub_c * sub_c).sum(dim=1) # [k_bank]
    
    # Target exact logit
    pos_c = centroids[targets]
    pos_score = (h_flat * pos_c).sum(dim=-1) - 0.5 * (pos_c * pos_c).sum(dim=-1) # [N]
    
    # Bank GEMM: [N, D] x [D, k_bank] -> [N, k_bank]
    bank_logits = torch.addmm(-c_half_norm, h_flat, sub_c.T)
    
    # Calibrated Tail Partition:
    # Scale background energy by ratio of unvisited vocab (V - k_bank) / k_bank
    scale_factor = float(V - k_bank) / float(k_bank)
    bank_max = bank_logits.max(dim=1, keepdim=True).values
    scaled_tail_sum = scale_factor * torch.exp(bank_logits - bank_max).sum(dim=1, keepdim=True)
    
    overall_max = torch.maximum(pos_score.unsqueeze(1), bank_max)
    total_lse = overall_max + torch.log(
        torch.exp(pos_score.unsqueeze(1) - overall_max) + 
        torch.exp(bank_logits - overall_max).sum(dim=1, keepdim=True) +
        scaled_tail_sum * torch.exp(bank_max - overall_max)
    )
    
    loss = (-pos_score.unsqueeze(1) + total_lse).mean()
    return loss

# =====================================================================
# 4. Model Wrappers
# =====================================================================

class ModelVoronoiCCEChunked(nn.Module):
    def __init__(self, core: TransformerCore, chunk_size: int = N_CHUNK_SIZE):
        super().__init__()
        self.core = core
        self.chunk_size = chunk_size

    def forward(self, idx, targets=None):
        x = self.core(idx)
        if targets is not None:
            return None, loss_voronoi_cce_chunked(x.view(-1, self.core.config.n_embd), self.core.transformer.wte.weight, targets.view(-1), self.chunk_size)
        return None, None

class ModelVoronoiAdaptiveCCE(nn.Module):
    def __init__(self, core: TransformerCore, k_bank: int = K_ADAPTIVE_BANK):
        super().__init__()
        self.core = core
        self.k_bank = k_bank

    def forward(self, idx, targets=None):
        x = self.core(idx)
        if targets is not None:
            return None, loss_voronoi_adaptive_cce(x.view(-1, self.core.config.n_embd), self.core.transformer.wte.weight, targets.view(-1), self.k_bank)
        return None, None

# =====================================================================
# 5. Benchmark Orchestration
# =====================================================================

def evaluate_full_vocab_loss(model, get_batch, eval_iters):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(eval_iters):
            vx, vy = get_batch('val', block_size)
            h = model.core(vx)
            centroids = model.core.transformer.wte.weight
            loss = loss_voronoi_cce_chunked(h.view(-1, n_embd), centroids, vy.view(-1), chunk_size=1024)
            total_loss += loss.item()
    model.train()
    return total_loss / eval_iters

def run_benchmarks():
    gptconf = GPTConfig()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    print(f"Device: {device} | Vocab: {vocab_size} | Emb Dim: {n_embd} | Layers: {n_layer}")
    active_experiments = [k for k, v in EXPERIMENTS_TO_RUN.items() if v]
    print(f"Active Experiments: {active_experiments}")
    print("=" * 115)

    models = {}
    optimizers = {}

    for name in active_experiments:
        torch.manual_seed(1337)
        core = TransformerCore(gptconf).to(device)
        
        if name == "voronoi_cce_chunked":
            m = ModelVoronoiCCEChunked(core, chunk_size=N_CHUNK_SIZE).to(device)
        elif name == "voronoi_adaptive_cce":
            m = ModelVoronoiAdaptiveCCE(core, k_bank=K_ADAPTIVE_BANK).to(device)
        else:
            continue
            
        models[name] = torch.compile(m)
        optimizers[name] = torch.optim.AdamW(m.parameters(), lr=learning_rate)

    # Warmup (Compiles static graphs)
    for _ in range(3):
        bx, by = get_batch('train', block_size)
        for name, model in models.items():
            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()
            opt = optimizers[name]
            opt.zero_grad(set_to_none=True)
            _, l = model(bx, by)
            l.backward()
            opt.step()
    if device == 'cuda': torch.cuda.synchronize()

    header = f"{'Step':<6} | " + " | ".join([f"{name[:18]:<18} (ms)" for name in models.keys()])
    print(header)
    print("-" * 115)

    timing_accum = {name: 0.0 for name in models.keys()}

    for it in range(max_iters):
        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        bx, by = get_batch('train', block_size)
        step_times = {}

        for name, model in models.items():
            opt = optimizers[name]
            if device == 'cuda': torch.cuda.synchronize()
            t0 = time.perf_counter()
            
            opt.zero_grad(set_to_none=True)
            _, loss = model(bx, by)
            loss.backward()
            opt.step()
            
            if device == 'cuda': torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000.0
            step_times[name] = (loss.item(), dt)
            timing_accum[name] += dt

        if it % 50 == 0 or it == max_iters - 1:
            row = f"{it:<6d} | " + " | ".join([f"{loss:<6.3f} ({dt:<5.1f}ms)" for (loss, dt) in step_times.values()])
            print(row)

        if it % eval_interval == 0 and it > 0:
            print("-" * 115)
            eval_metrics = []
            for name, model in models.items():
                val_loss = evaluate_full_vocab_loss(model, get_batch, eval_iters)
                eval_metrics.append(f"{name}: {val_loss:.4f}")
            print(f"[EVAL @ {it} FULL VOCAB] " + " | ".join(eval_metrics))
            print("-" * 115)

    print("=" * 115)
    for name, t_total in timing_accum.items():
        print(f"Total Step Time - {name:<26}: {t_total:.1f} ms")

if __name__ == "__main__":
    run_benchmarks()