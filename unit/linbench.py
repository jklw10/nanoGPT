"""
Auto-Dropout Linear vs Standard & Blind Dropout Benchmark
Plugs into mlp.c_fc (256 -> 1024) to compare:
  1. dense_base        : Standard Linear (no dropout)
  2. dense_drop10      : Standard Linear + 10% Dropout
  3. dense_drop50      : Standard Linear + 50% Dropout (same active budget as auto50)
  4. auto_drop_v11_50  : 512/1024 active (50%) importance-sampled v11 clusters
  5. auto_drop_v11_75  : 768/1024 active (75%) importance-sampled v11 clusters
"""

from dataclasses import dataclass, replace
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.non_ml.triton_kmeans import run_engine_v11_triton, warmup_graphs
import parts.utils as utils

torch._inductor.config.disable_cpp_codegen = True
torch._inductor.config.triton.cudagraphs = False

# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GPTConfig:
    dataset: str = "openwebtext"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1337
    batch_size: int = 64
    block_size: int = 64
    vocab_size: int = 50304
    always_byte: bool = False

    n_layer: int = 4
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = False

    learning_rate: float = 1e-3
    min_lr: float = 1e-4
    warmup_iters: int = 100
    lr_decay_iters: int = 2000
    max_iters: int = 2000
    eval_interval: int = 500
    eval_iters: int = 50

    # Linear / Dropout settings for mlp.c_fc
    fc_type: str = "standard"       # "standard" | "dense_drop" | "auto_drop_v11"
    fc_drop_p: float = 0.0
    fc_k_active: int = 512
    fc_k_heavy: int = 128
    tree_refresh_steps: tuple[int, ...] = (50, 200, 500, 1000)


BASE_CFG = GPTConfig()

EXPERIMENTS = {
    "dense_base":       (True, replace(BASE_CFG, fc_type="standard")),
    "dense_drop10":     (True, replace(BASE_CFG, fc_type="dense_drop", fc_drop_p=0.10)),
    "dense_drop50":     (True, replace(BASE_CFG, fc_type="dense_drop", fc_drop_p=0.50)),
    "auto_drop_v11_50": (True, replace(BASE_CFG, fc_type="auto_drop_v11", fc_k_active=512, fc_k_heavy=128)),
    "auto_drop_v11_75": (True, replace(BASE_CFG, fc_type="auto_drop_v11", fc_k_active=768, fc_k_heavy=256)),
}

decode, vocab_size = utils.get_meta(BASE_CFG.dataset, always_byte=BASE_CFG.always_byte)
get_batch = utils.get_get_batch(
    BASE_CFG.block_size, BASE_CFG.batch_size, BASE_CFG.dataset, BASE_CFG.device, always_byte=BASE_CFG.always_byte
)

def get_lr(it: int, cfg: GPTConfig) -> float:
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / (cfg.warmup_iters + 1)
    if it > cfg.lr_decay_iters:
        return cfg.min_lr
    decay = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
    return cfg.min_lr + 0.5 * (1.0 + math.cos(math.pi * decay)) * (cfg.learning_rate - cfg.min_lr)

# ---------------------------------------------------------------------------
# 2. Auto-Dropout Linear (Importance-Sampled v11 Structured Dropout)
# ---------------------------------------------------------------------------

class AutoDropoutLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, k_active: int = 512, k_heavy: int = 128, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k_active = min(k_active, out_features)
        self.k_heavy = min(k_heavy, self.k_active)
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Clustering setup on neurons (rows of W)
        self.depth = 4                          # M = 16 neuron clusters
        self.M = 1 << self.depth
        self.k_coarse = 8
        self.D_tail = out_features - self.k_heavy
        self.S_bucket = math.ceil(1.10 * self.D_tail / self.M)

        K_rem = self.k_active - self.k_heavy
        self.P = max(1, min(self.M - 1, (2 * K_rem) // (3 * self.S_bucket)))
        self.K_routed = self.P * self.S_bucket
        self.K_tail = max(1, self.k_active - self.k_heavy - self.K_routed)

        dev = self.weight.device
        self.register_buffer("neuron_freq", torch.zeros(out_features, device=dev))
        self.register_buffer("heavy_idx", torch.arange(self.k_heavy, device=dev))
        self.register_buffer("leaf_neurons", torch.zeros((self.M, self.S_bucket), dtype=torch.long, device=dev))
        self.register_buffer("leaf_valid", torch.zeros((self.M, self.S_bucket), dtype=torch.bool, device=dev))

    @torch.no_grad()
    def track_activations(self, y_dense: torch.Tensor):
        # Track neuron firing magnitude to identify persistent heavy core
        mags = y_dense.abs().mean(dim=list(range(y_dense.dim() - 1)))
        self.neuron_freq.mul_(0.98).add_(mags)

    @torch.no_grad()
    def refresh_tree(self):
        # 1. Pin top heavy neurons
        order = torch.argsort(self.neuron_freq, descending=True)
        heavy = order[:self.k_heavy]
        tail_neurons = order[self.k_heavy:]
        self.heavy_idx.copy_(heavy)

        # 2. Cluster remaining tail neurons with v11
        W_tail = self.weight[tail_neurons].float()
        C_coarse = run_engine_v11_triton(W_tail, k=self.k_coarse, sub_n=min(1024, self.D_tail), half_x=True, use_graph=True)
        c_half = 0.5 * (C_coarse * C_coarse).sum(dim=1)
        assign = torch.argmax(torch.addmm(-c_half, W_tail, C_coarse.T), dim=1)
        
        # 3. Simple bucket packing
        buckets = [[] for _ in range(self.M)]
        for idx_local, c in enumerate(assign.cpu().tolist()):
            leaf_id = (c * 2) % self.M
            if len(buckets[leaf_id]) < self.S_bucket:
                buckets[leaf_id].append(idx_local)
            else:
                for alt in range(self.M):
                    if len(buckets[alt]) < self.S_bucket:
                        buckets[alt].append(idx_local)
                        break

        self.leaf_neurons.zero_()
        self.leaf_valid.zero_()
        for m, b in enumerate(buckets):
            if len(b) > 0:
                t_b = tail_neurons[torch.tensor(b, device=self.weight.device)]
                self.leaf_neurons[m, :len(b)] = t_b
                self.leaf_valid[m, :len(b)] = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # At test time: evaluate exact full dense linear layer!
        if not self.training or self.k_active >= self.out_features:
            return F.linear(x, self.weight, self.bias)

        B, T, C = x.shape
        x_flat = x.view(-1, C)
        N = x_flat.shape[0]

        # 1. Coarse routing against cluster centroids
        with torch.no_grad():
            leaf_embs = self.weight[self.leaf_neurons] * self.leaf_valid.unsqueeze(-1).to(self.weight.dtype)
            leaf_counts = self.leaf_valid.sum(dim=1, keepdim=True).clamp(min=1).to(self.weight.dtype)
            leaf_c = leaf_embs.sum(dim=1) / leaf_counts

            coarse = x_flat @ leaf_c.T
            heat = coarse.abs().sum(dim=0)
            sorted_c = torch.argsort(heat, descending=True)
            routed_c = sorted_c[:self.P]
            unrouted_c = sorted_c[self.P:]

            routed_idx = self.leaf_neurons[routed_c].reshape(-1)
            rand_u = torch.randint(0, self.M - self.P, (self.K_tail,), device=x.device)
            rand_s = torch.randint(0, self.S_bucket, (self.K_tail,), device=x.device)
            tail_idx = self.leaf_neurons[unrouted_c[rand_u], rand_s]

            bank_idx = torch.cat([self.heavy_idx, routed_idx, tail_idx], dim=0)

        # 2. Compute only active sub-matrix
        sub_w = self.weight[bank_idx]
        sub_b = self.bias[bank_idx] if self.bias is not None else None
        y_sub = F.linear(x_flat, sub_w, sub_b)

        # 3. Unbiased importance scaling on tail neurons
        tail_scale = float((self.M - self.P) * self.S_bucket) / float(self.K_tail)
        exact_cut = self.k_heavy + self.K_routed
        y_sub[:, exact_cut:] *= tail_scale

        # 4. Scatter back into dense output for GELU
        y_dense = torch.zeros(N, self.out_features, device=x.device, dtype=x.dtype)
        y_dense.scatter_(1, bank_idx.unsqueeze(0).expand(N, -1), y_sub)
        
        self.track_activations(y_dense)
        return y_dense.view(B, T, self.out_features)

# ---------------------------------------------------------------------------
# 3. Model Architecture
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.n_embd))

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, None, 1e-5)


class Attention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.q_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

    def forward(self, x):
        B, T, C = x.size()
        q = self.q_attn(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_attn(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_attn(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Build c_fc based on experiment
        if config.fc_type == "auto_drop_v11":
            self.c_fc = AutoDropoutLinear(
                config.n_embd, 4 * config.n_embd, 
                k_active=config.fc_k_active, k_heavy=config.fc_k_heavy, bias=config.bias
            )
        else:
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.fc_drop_p) if config.fc_type == "dense_drop" else nn.Identity()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config)
        self.attn = Attention(config)
        self.ln_2 = LayerNorm(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.cfg = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def refresh_trees(self):
        for block in self.transformer.h:
            if isinstance(block.mlp.c_fc, AutoDropoutLinear):
                block.mlp.c_fc.refresh_tree()

    def forward(self, idx, targets=None):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is None:
            return x, None
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# ---------------------------------------------------------------------------
# 4. Evaluation & Execution Engine
# ---------------------------------------------------------------------------

def evaluate_model(raw_model: GPT, val_batches):
    raw_model.eval()
    tot_ce, tot_gen = 0.0, 0.0
    with torch.no_grad():
        for vx, vy in val_batches:
            logits, _ = raw_model(vx, vy)
            tot_ce += F.cross_entropy(logits.view(-1, logits.size(-1)), vy.view(-1)).item()
            tot_gen += F.cross_entropy(logits[:, -1, :], vy[:, -1]).item()
    raw_model.train()
    n = len(val_batches)
    return tot_ce / n, tot_gen / n


def run_benchmarks():
    active = {k: replace(cfg, vocab_size=vocab_size) for k, (enabled, cfg) in EXPERIMENTS.items() if enabled}
    print(f"Device: {BASE_CFG.device} | Vocab: {vocab_size} | Active: {list(active.keys())}")

    val_batches = [get_batch("val", BASE_CFG.block_size) for _ in range(BASE_CFG.eval_iters)]
    warmup_graphs([(8, BASE_CFG.n_embd)], sub_n=min(1024, 4 * BASE_CFG.n_embd), half_x=True, device=BASE_CFG.device)

    raw_models, optimizers, cfgs = {}, {}, {}
    for name, cfg in active.items():
        torch.manual_seed(cfg.seed)
        m = GPT(cfg).to(cfg.device)
        m.refresh_trees()
        raw_models[name] = m
        cfgs[name] = cfg
        optimizers[name] = torch.optim.AdamW(m.parameters(), lr=cfg.learning_rate)

    col_widths = {n: max(len(n), 16) for n in raw_models.keys()}
    print("=" * 140)
    print(f"{'Step':<6} | " + " | ".join([f"{n:<{col_widths[n]}}" for n in raw_models.keys()]) + " | ms/step")
    print(f"{'':<6} | " + " | ".join([f"{' val_ce    gen':<{col_widths[n]}}" for n in raw_models.keys()]) + " |")
    print("-" * 140)

    interval_timing = {n: 0.0 for n in raw_models.keys()}

    for it in range(BASE_CFG.max_iters):
        bx, by = get_batch("train", BASE_CFG.block_size)
        step = it + 1
        is_eval = (step % BASE_CFG.eval_interval == 0 or step == BASE_CFG.max_iters)

        if step in BASE_CFG.tree_refresh_steps:
            for rm in raw_models.values():
                rm.refresh_trees()

        for name, rm in raw_models.items():
            cfg = cfgs[name]
            opt = optimizers[name]
            lr = get_lr(it, cfg)
            for pg in opt.param_groups:
                pg["lr"] = lr

            if cfg.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            opt.zero_grad(set_to_none=True)
            _, loss = rm(bx, by)
            loss.backward()
            opt.step()

            if cfg.device == "cuda":
                torch.cuda.synchronize()
            interval_timing[name] += (time.perf_counter() - t0) * 1000.0

        if is_eval:
            eval_cells, ms_cells = [], []
            for name, rm in raw_models.items():
                v_ce, v_gen = evaluate_model(rm, val_batches)
                eval_cells.append(f"{f'{v_ce:7.4f} {v_gen:7.4f}':<{col_widths[name]}}")
                ms_cells.append(f"{interval_timing[name] / BASE_CFG.eval_interval:5.1f}")
                interval_timing[name] = 0.0
            print(f"{step:<6d} | " + " | ".join(eval_cells) + " | " + " / ".join(ms_cells))


if __name__ == "__main__":
    run_benchmarks()