"""
Benchmark: Full-Vocab Baselines vs. Voronoi/Dot v11 Zipf-Routed Banks
Ablates logit geometry (Dot vs. Voronoi), tree partitioner (v11 vs. Power),
bank capacity (3k vs. 4k), and hierarchical softmax routing.
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
torch._dynamo.config.force_parameter_static_shapes = False
torch._dynamo.config.cache_size_limit = 128
torch._dynamo.config.accumulated_cache_size_limit = 1024

# =====================================================================
# 1. Unified Config & Experiment Registry
# =====================================================================

@dataclass(frozen=True)
class GPTConfig:
    # Data & Runtime
    dataset: str = "openwebtext"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1337
    batch_size: int = 64
    block_size: int = 64
    vocab_size: int = 50304
    always_byte: bool = False

    # Model Backbone
    n_layer: int = 4
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = False

    # Schedule & Optimizer
    learning_rate: float = 1e-3
    min_lr: float = 1e-4
    warmup_iters: int = 100
    lr_decay_iters: int = 2000
    max_iters: int = 50000
    eval_interval: int = 1000
    eval_iters: int = 50
    gen: bool = True
    gen_tokens: int = 100

    # Vocabulary Approximation Flags
    loss_mode: str = "dot_standard"      # "dot_standard" | "dot_chunked" | "voronoi_chunked" | "zipf_routed" | "tree_hsm"
    logit_mode: str = "dot"              # "dot" | "voronoi"
    tree_mode: str = "v11"               # "v11" | "power"
    k_bank: int = 3072
    k_heavy: int = 1024
    route_chunk_size: int = 512
    v11_k_coarse: int = 16
    v11_slack: float = 1.10
    v11_sub_n: int = 16384
    tree_depth: int = 8
    tree_refresh_steps: tuple[int, ...] = (25, 100, 250, 500, 1000, 1500)
    chunk_size: int = 1024


BASE_CFG = GPTConfig()

EXPERIMENTS = {
    # Full-Vocab Baselines (Exact References)
    "baseline_dot":         (True,  replace(BASE_CFG, loss_mode="dot_standard", logit_mode="dot")),
    "voronoi_cce":          (False, replace(BASE_CFG, loss_mode="voronoi_chunked", logit_mode="voronoi")),

    # Power-Iteration Baseline
    "power_zipf_voronoi":   (False, replace(BASE_CFG, loss_mode="zipf_routed", tree_mode="power", logit_mode="voronoi")),

    # Engine v11-Triton Approximations
    "v11_zipf_voronoi_3k":  (False,  replace(BASE_CFG, loss_mode="zipf_routed", tree_mode="v11",   logit_mode="voronoi")),
    "v11_zipf_dot_3k":      (False,  replace(BASE_CFG, loss_mode="zipf_routed", tree_mode="v11",   logit_mode="dot")),
    "v11_zipf_dot_4k":      (True, replace(BASE_CFG, loss_mode="zipf_routed", tree_mode="v11",   logit_mode="dot", k_bank=4096, k_heavy=1024)),
    "v11_tree_hsm":         (False, replace(BASE_CFG, loss_mode="tree_hsm",    tree_mode="v11",   logit_mode="voronoi")),
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
    decay_ratio = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)

# =====================================================================
# 2. Vocabulary Partitioners (Power-Iter vs. Engine v11-Triton)
# =====================================================================

def _pack_buckets_to_tensors(buckets: list[torch.Tensor], M: int, S_bucket: int, V: int, dev: torch.device):
    leaf_tokens = torch.zeros((M, S_bucket), device=dev, dtype=torch.long)
    leaf_valid = torch.zeros((M, S_bucket), device=dev, dtype=torch.bool)
    token_to_leaf = torch.zeros(V, device=dev, dtype=torch.long)
    token_to_slot = torch.zeros(V, device=dev, dtype=torch.long)

    for leaf_id, idxs in enumerate(buckets):
        cnt = idxs.shape[0]
        if cnt > 0:
            leaf_tokens[leaf_id, :cnt] = idxs
            leaf_valid[leaf_id, :cnt] = True
            token_to_leaf[idxs] = leaf_id
            token_to_slot[idxs] = torch.arange(cnt, device=dev)

    return leaf_tokens, leaf_valid, token_to_leaf, token_to_slot


def _principal_axis_order(sub: torch.Tensor) -> torch.Tensor:
    centered = sub - sub.mean(dim=0, keepdim=True)
    v = torch.randn(sub.shape[1], device=sub.device, dtype=sub.dtype)
    v = v / (torch.linalg.norm(v) + 1e-9)
    for _ in range(3):
        z = centered @ v
        v = centered.T @ z
        v = v / (torch.linalg.norm(v) + 1e-9)
    return torch.argsort(centered @ v)


def _split_into_m_leaves(W: torch.Tensor, idxs: torch.Tensor, m_leaves: int, out: list[torch.Tensor]):
    if m_leaves <= 1:
        out.append(idxs)
        return
    n = idxs.shape[0]
    if n <= 1:
        out.append(idxs)
        for _ in range(m_leaves - 1):
            out.append(idxs.new_empty(0))
        return
    m_left = m_leaves // 2
    m_right = m_leaves - m_left
    order = _principal_axis_order(W[idxs])
    half = (n * m_left) // m_leaves
    _split_into_m_leaves(W, idxs[order[:half]], m_left, out)
    _split_into_m_leaves(W, idxs[order[half:]], m_right, out)


@torch.no_grad()
def build_power_tree_buckets(W: torch.Tensor, depth: int):
    V, _ = W.shape
    M = 1 << depth
    S_bucket = math.ceil(V / M)

    buckets = [torch.arange(V, device=W.device)]
    for _ in range(depth):
        next_buckets = []
        for idxs in buckets:
            m = idxs.shape[0]
            if m <= 1:
                next_buckets.append(idxs)
                next_buckets.append(idxs.new_empty(0))
                continue
            order = _principal_axis_order(W[idxs])
            half = m // 2
            next_buckets.append(idxs[order[:half]])
            next_buckets.append(idxs[order[half:]])
        buckets = next_buckets

    return _pack_buckets_to_tensors(buckets, M, S_bucket, V, W.device)


@torch.no_grad()
def build_v11_tree_buckets(W: torch.Tensor, depth: int, k_coarse: int = 16, slack: float = 1.10, sub_n: int = 16384):
    V, _ = W.shape
    M = 1 << depth
    k_c = min(k_coarse, max(2, M // 4))
    S_bucket = math.ceil(slack * V / M)

    use_g = (V >= sub_n) and (W.device.type == "cuda")
    C_coarse = run_engine_v11_triton(W, k=k_c, sub_n=min(sub_n, V), half_x=True, use_graph=use_g)

    c_half = 0.5 * (C_coarse * C_coarse).sum(dim=1)
    coarse_assign = torch.argmax(torch.addmm(-c_half, W.float(), C_coarse.T), dim=1)
    counts = torch.bincount(coarse_assign, minlength=k_c).cpu().tolist()

    while True:
        m_alloc = [math.ceil(c / S_bucket) if c > 0 else 0 for c in counts]
        if sum(m_alloc) <= M:
            break
        S_bucket = math.ceil(S_bucket * 1.05)

    rem = M - sum(m_alloc)
    while rem > 0:
        best_c = max((c for c in range(k_c) if counts[c] > 0), key=lambda c: counts[c] / m_alloc[c])
        m_alloc[best_c] += 1
        rem -= 1

    buckets: list[torch.Tensor] = []
    for c in range(k_c):
        if m_alloc[c] == 0:
            continue
        idxs = (coarse_assign == c).nonzero(as_tuple=True)[0]
        _split_into_m_leaves(W, idxs, m_alloc[c], buckets)

    return _pack_buckets_to_tensors(buckets, M, S_bucket, V, W.device)

# =====================================================================
# 3. Model Architecture & Approximated Loss Dispatch
# =====================================================================

class LayerNorm(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.n_embd))
        self.bias = nn.Parameter(torch.zeros(config.n_embd)) if config.bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class Attention(nn.Module):
    def __init__(self, config: GPTConfig):
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
            q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


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
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config),
        ))

        # Setup routing buffers if routed approximation is chosen
        if config.loss_mode in ("zipf_routed", "tree_hsm"):
            V = config.vocab_size
            self.K_heavy = min(config.k_heavy, max(0, V - config.k_bank)) if config.loss_mode == "zipf_routed" else 0
            V_tail = max(1, V - self.K_heavy)
            self.depth = config.tree_depth if V_tail > 2048 else 4
            self.M = 1 << self.depth

            slack = config.v11_slack if config.tree_mode == "v11" else 1.0
            S_bucket = math.ceil(slack * V_tail / self.M)

            w_dev = self.transformer.wte.weight.device
            self.register_buffer("token_freq", torch.zeros(V, dtype=torch.float32, device=w_dev))
            self.register_buffer("heavy_idx", torch.arange(self.K_heavy, dtype=torch.long, device=w_dev))
            self.register_buffer("leaf_tokens", torch.zeros((self.M, S_bucket), dtype=torch.long, device=w_dev))
            self.register_buffer("leaf_valid", torch.zeros((self.M, S_bucket), dtype=torch.bool, device=w_dev))
            self.register_buffer("token_to_leaf", torch.zeros(V, dtype=torch.long, device=w_dev))
            self.register_buffer("token_to_slot", torch.zeros(V, dtype=torch.long, device=w_dev))

            if config.loss_mode == "zipf_routed":
                self._update_zipf_budget(S_bucket)
            #self.refresh_tree()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _update_zipf_budget(self, S_bucket: int):
        self.S_bucket = S_bucket
        K_rem = max(1, self.cfg.k_bank - self.K_heavy)
        max_p_for_tail = max(1, (K_rem - 1) // self.S_bucket)
        self.P = min(self.M - 1, max(1, min((2 * K_rem) // (3 * self.S_bucket), max_p_for_tail)))
        self.K_routed = self.P * self.S_bucket
        self.K_tail = max(1, self.cfg.k_bank - self.K_heavy - self.K_routed)

    @torch.no_grad()
    def track_targets(self, targets: torch.Tensor):
        if hasattr(self, "token_freq"):
            counts = torch.bincount(targets.reshape(-1), minlength=self.token_freq.shape[0]).float()
            self.token_freq.mul_(0.98).add_(counts)

    @torch.no_grad()
    def refresh_tree(self):
        if not hasattr(self, "leaf_tokens"):
            return
        V = self.cfg.vocab_size
        W = self.transformer.wte.weight

        if self.cfg.loss_mode == "zipf_routed":
            freq_order = torch.argsort(self.token_freq, descending=True)
            heavy = freq_order[:self.K_heavy]
            tail_vocab = freq_order[self.K_heavy:]

            if self.cfg.tree_mode == "v11":
                lt, lv, _, _ = build_v11_tree_buckets(
                    W[tail_vocab], self.depth, k_coarse=self.cfg.v11_k_coarse,
                    slack=self.cfg.v11_slack, sub_n=self.cfg.v11_sub_n
                )
            else:
                lt, lv, _, _ = build_power_tree_buckets(W[tail_vocab], self.depth)

            if lt.shape != self.leaf_tokens.shape:
                self.leaf_tokens = torch.zeros_like(lt)
                self.leaf_valid = torch.zeros_like(lv)
                self._update_zipf_budget(lt.shape[1])

            self.heavy_idx.copy_(heavy)
            self.leaf_tokens.copy_(tail_vocab[lt])
            self.leaf_valid.copy_(lv)

        elif self.cfg.loss_mode == "tree_hsm":
            if self.cfg.tree_mode == "v11":
                lt, lv, t2l, t2s = build_v11_tree_buckets(
                    W, self.depth, k_coarse=self.cfg.v11_k_coarse,
                    slack=self.cfg.v11_slack, sub_n=self.cfg.v11_sub_n
                )
            else:
                lt, lv, t2l, t2s = build_power_tree_buckets(W, self.depth)

            if lt.shape != self.leaf_tokens.shape:
                self.leaf_tokens = torch.zeros_like(lt)
                self.leaf_valid = torch.zeros_like(lv)
            self.leaf_tokens.copy_(lt)
            self.leaf_valid.copy_(lv)
            self.token_to_leaf.copy_(t2l)
            self.token_to_slot.copy_(t2s)

    def _loss_dot_standard(self, h_flat: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(F.linear(h_flat, self.transformer.wte.weight), targets)

    def _loss_dot_chunked(self, h_flat: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        centroids = self.transformer.wte.weight
        N = h_flat.shape[0]
        tot = 0.0
        for i in range(0, N, self.cfg.chunk_size):
            h_c = h_flat[i:i + self.cfg.chunk_size]
            t_c = targets[i:i + self.cfg.chunk_size]
            tot = tot + F.cross_entropy(h_c @ centroids.T, t_c, reduction="sum")
        return tot / N

    def _loss_voronoi_chunked(self, h_flat: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        centroids = self.transformer.wte.weight
        c_half = 0.5 * (centroids * centroids).sum(dim=1)
        N = h_flat.shape[0]
        tot = 0.0
        for i in range(0, N, self.cfg.chunk_size):
            h_c = h_flat[i:i + self.cfg.chunk_size]
            t_c = targets[i:i + self.cfg.chunk_size]
            logits = torch.addmm(-c_half, h_c, centroids.T)
            tot = tot + F.cross_entropy(logits, t_c, reduction="sum")
        return tot / N

    def _loss_tree_hsm(self, h_flat: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        centroids = self.transformer.wte.weight
        leaf_embs = centroids[self.leaf_tokens] * self.leaf_valid.unsqueeze(-1).to(centroids.dtype)
        leaf_counts = self.leaf_valid.sum(dim=1, keepdim=True).clamp(min=1).to(centroids.dtype)
        leaf_centroids = leaf_embs.sum(dim=1) / leaf_counts
        r_half = 0.5 * (leaf_centroids * leaf_centroids).sum(dim=1)

        t_leaf = self.token_to_leaf[targets]
        t_slot = self.token_to_slot[targets]
        N = h_flat.shape[0]
        tot = 0.0

        for i in range(0, N, self.cfg.chunk_size):
            h_c = h_flat[i:i + self.cfg.chunk_size]
            tl_c = t_leaf[i:i + self.cfg.chunk_size]
            ts_c = t_slot[i:i + self.cfg.chunk_size]

            coarse_logits = torch.addmm(-r_half, h_c, leaf_centroids.T)
            coarse_loss = F.cross_entropy(coarse_logits, tl_c, reduction="sum")

            cand_tokens = self.leaf_tokens[tl_c]
            cand_valid = self.leaf_valid[tl_c]
            cand_c = centroids[cand_tokens]
            cand_half = 0.5 * (cand_c * cand_c).sum(dim=-1)

            fine_logits = torch.bmm(h_c.unsqueeze(1), cand_c.transpose(1, 2)).squeeze(1) - cand_half
            fine_logits = fine_logits.masked_fill(~cand_valid, -1e9)
            fine_loss = F.cross_entropy(fine_logits, ts_c, reduction="sum")
            tot = tot + coarse_loss + fine_loss

        return tot / N

    def _loss_zipf_routed(self, h_flat: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        centroids = self.transformer.wte.weight
        V = centroids.shape[0]
        if V <= self.cfg.k_bank:
            if self.cfg.logit_mode == "dot":
                return self._loss_dot_chunked(h_flat, targets)
            return self._loss_voronoi_chunked(h_flat, targets)

        with torch.no_grad():
            leaf_embs = centroids[self.leaf_tokens] * self.leaf_valid.unsqueeze(-1).to(centroids.dtype)
            leaf_counts = self.leaf_valid.sum(dim=1, keepdim=True).clamp(min=1).to(centroids.dtype)
            leaf_centroids = leaf_embs.sum(dim=1) / leaf_counts
            r_half = 0.5 * (leaf_centroids * leaf_centroids).sum(dim=1) if self.cfg.logit_mode == "voronoi" else None

        N = h_flat.shape[0]
        tot = 0.0
        tail_scale = float((self.M - self.P) * self.S_bucket) / float(self.K_tail)
        heavy_valid = torch.ones(self.K_heavy, dtype=torch.bool, device=h_flat.device)
        exact_cut = self.K_heavy + self.K_routed

        for i in range(0, N, self.cfg.route_chunk_size):
            h_c = h_flat[i:i + self.cfg.route_chunk_size]
            t_c = targets[i:i + self.cfg.route_chunk_size]

            with torch.no_grad():
                if self.cfg.logit_mode == "dot":
                    coarse_scores = h_c @ leaf_centroids.T
                else:
                    coarse_scores = torch.addmm(-r_half, h_c, leaf_centroids.T)

                cluster_heat = F.softmax(coarse_scores, dim=-1).sum(dim=0)
                sorted_clusters = torch.argsort(cluster_heat, descending=True)

                routed_clusters = sorted_clusters[:self.P]
                unrouted_clusters = sorted_clusters[self.P:]

                routed_idx = self.leaf_tokens[routed_clusters].reshape(-1)
                routed_valid = self.leaf_valid[routed_clusters].reshape(-1)

                rand_u = torch.randint(0, self.M - self.P, (self.K_tail,), device=h_flat.device)
                rand_s = torch.randint(0, self.S_bucket, (self.K_tail,), device=h_flat.device)
                chosen_u = unrouted_clusters[rand_u]
                tail_idx = self.leaf_tokens[chosen_u, rand_s]
                tail_valid = self.leaf_valid[chosen_u, rand_s]

                bank_idx = torch.cat([self.heavy_idx, routed_idx, tail_idx], dim=0)
                bank_valid = torch.cat([heavy_valid, routed_valid, tail_valid], dim=0).unsqueeze(0)

            sub_c = centroids[bank_idx]
            pos_c = centroids[t_c]

            if self.cfg.logit_mode == "dot":
                bank_logits = h_c @ sub_c.T
                pos_score = (h_c * pos_c).sum(dim=-1)
            else:
                sub_half = 0.5 * (sub_c * sub_c).sum(dim=1)
                bank_logits = torch.addmm(-sub_half, h_c, sub_c.T)
                pos_score = (h_c * pos_c).sum(dim=-1) - 0.5 * (pos_c * pos_c).sum(dim=-1)

            is_target = (bank_idx.unsqueeze(0) == t_c.unsqueeze(1))
            masked_logits = bank_logits.masked_fill(is_target | (~bank_valid), -1e9)

            exact_logits = masked_logits[:, :exact_cut]
            tail_logits = masked_logits[:, exact_cut:]

            bank_max = masked_logits.max(dim=1, keepdim=True).values
            overall_max = torch.maximum(pos_score.unsqueeze(1), bank_max)

            pos_exp = torch.exp(pos_score.unsqueeze(1) - overall_max)
            exact_exp = torch.exp(exact_logits - overall_max).sum(dim=1, keepdim=True)
            tail_exp = tail_scale * torch.exp(tail_logits - overall_max).sum(dim=1, keepdim=True)

            total_lse = overall_max + torch.log(pos_exp + exact_exp + tail_exp)
            tot = tot + (-pos_score.unsqueeze(1) + total_lse).sum()

        return tot / N

    def forward(self, idx, targets=None):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        h = self.transformer.ln_f(x)

        if targets is None:
            return h, None

        h_flat = h.view(-1, self.cfg.n_embd)
        t_flat = targets.view(-1)

        if self.cfg.loss_mode == "dot_standard":
            loss = self._loss_dot_standard(h_flat, t_flat)
        elif self.cfg.loss_mode == "dot_chunked":
            loss = self._loss_dot_chunked(h_flat, t_flat)
        elif self.cfg.loss_mode == "voronoi_chunked":
            loss = self._loss_voronoi_chunked(h_flat, t_flat)
        elif self.cfg.loss_mode == "tree_hsm":
            loss = self._loss_tree_hsm(h_flat, t_flat)
        elif self.cfg.loss_mode == "zipf_routed":
            loss = self._loss_zipf_routed(h_flat, t_flat)
        else:
            raise ValueError(f"Unknown loss_mode: {self.cfg.loss_mode}")

        return h, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=5):
        centroids = self.transformer.wte.weight
        c_half = 0.5 * (centroids * centroids).sum(dim=1) if self.cfg.logit_mode == "voronoi" else None

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
            h_last, _ = self(idx_cond)
            h_vec = h_last[:, -1, :]

            if self.cfg.logit_mode == "voronoi":
                logits = torch.addmm(-c_half, h_vec, centroids.T)
            else:
                logits = F.linear(h_vec, centroids)

            logits = logits / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# =====================================================================
# 4. Benchmark Orchestrator
# =====================================================================

def evaluate_model(raw_model: GPT, val_batches):
    raw_model.eval()
    tot_ce, tot_gen = 0.0, 0.0
    centroids = raw_model.transformer.wte.weight
    c_half = 0.5 * (centroids * centroids).sum(dim=1) if raw_model.cfg.logit_mode == "voronoi" else None

    with torch.no_grad():
        for vx, vy in val_batches:
            h, _ = raw_model(vx)
            h_flat = h.view(-1, raw_model.cfg.n_embd)
            t_flat = vy.view(-1)

            # 1. Exact full-vocab CE across all tokens (chunked to prevent VRAM spikes)
            n_tokens = h_flat.shape[0]
            chunk_loss = 0.0
            for i in range(0, n_tokens, 1024):
                hc = h_flat[i:i + 1024]
                tc = t_flat[i:i + 1024]
                if raw_model.cfg.logit_mode == "voronoi":
                    logits_c = torch.addmm(-c_half, hc, centroids.T)
                else:
                    logits_c = hc @ centroids.T
                chunk_loss += F.cross_entropy(logits_c, tc, reduction="sum").item()
            tot_ce += chunk_loss / n_tokens

            # 2. Next-token prediction loss on the last context position
            h_last = h[:, -1, :]
            t_last = vy[:, -1]
            if raw_model.cfg.logit_mode == "voronoi":
                logits_last = torch.addmm(-c_half, h_last, centroids.T)
            else:
                logits_last = h_last @ centroids.T
            tot_gen += F.cross_entropy(logits_last, t_last).item()

    raw_model.train()
    n = len(val_batches)
    return tot_ce / n, tot_gen / n


def run_benchmarks():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    active = {
        k: replace(cfg, vocab_size=vocab_size)
        for k, (enabled, cfg) in EXPERIMENTS.items()
        if enabled
    }
    print(f"Device: {BASE_CFG.device} | Vocab: {vocab_size} | Experiments: {list(active.keys())}")

    val_batches = [get_batch("val", BASE_CFG.block_size) for _ in range(BASE_CFG.eval_iters)]

    # Pre-record Triton CUDA graphs for active v11 shapes
    if BASE_CFG.device == "cuda":
        uses_v11 = any(cfg.tree_mode == "v11" and cfg.loss_mode in ("zipf_routed", "tree_hsm") for cfg in active.values())
        if uses_v11 and vocab_size > BASE_CFG.k_bank:
            warmup_dims = sorted(list({(cfg.v11_k_coarse, cfg.n_embd) for cfg in active.values() if cfg.tree_mode == "v11"}))
            print(f"Pre-recording Triton CUDA graph for {warmup_dims}...")
            warmup_graphs(warmup_dims, sub_n=min(BASE_CFG.v11_sub_n, vocab_size - BASE_CFG.k_heavy), half_x=True, device=BASE_CFG.device)

    raw_models, models, optimizers, cfgs = {}, {}, {}, {}
    failed = set()
    bx_sanity, by_sanity = get_batch("train", BASE_CFG.block_size)

    for name, cfg in active.items():
        torch.manual_seed(cfg.seed)
        m = GPT(cfg).to(cfg.device)
        m.refresh_tree()
        raw_models[name] = m
        cfgs[name] = cfg
        optimizers[name] = torch.optim.AdamW(m.parameters(), lr=cfg.learning_rate)

        t0 = time.perf_counter()
        compiled_m = torch.compile(m)
        for _ in range(2):
            optimizers[name].zero_grad(set_to_none=True)
            _, loss = compiled_m(bx_sanity, by_sanity)
            loss.backward()
            optimizers[name].zero_grad(set_to_none=True)
        if cfg.device == "cuda":
            torch.cuda.synchronize()
        models[name] = compiled_m
        print(f"  [COMPILED]  {name:<22} in {(time.perf_counter() - t0):.2f} s")

    col_widths = {n: max(len(n), 16) for n in models.keys()}
    print("=" * 135)
    print(f"{'Step':<6} | " + " | ".join([f"{n:<{col_widths[n]}}" for n in models.keys()]) + " | ms/step")
    print(f"{'':<6} | " + " | ".join([f"{' val_ce    gen':<{col_widths[n]}}" for n in models.keys()]) + " |")
    print("-" * 135)

    interval_timing = {n: 0.0 for n in models.keys()}

    for it in range(BASE_CFG.max_iters):
        bx, by = get_batch("train", BASE_CFG.block_size)
        step = it + 1
        is_eval = (step % BASE_CFG.eval_interval == 0 or step == BASE_CFG.max_iters)

        # Track token frequencies & refresh trees at scheduled steps
        for rm in raw_models.values():
            rm.track_targets(by)

        if step in BASE_CFG.tree_refresh_steps:
            for rm in raw_models.values():
                rm.refresh_tree()

        for name, model in models.items():
            if name in failed:
                continue

            cfg = cfgs[name]
            opt = optimizers[name]
            lr = get_lr(it, cfg)
            for pg in opt.param_groups:
                pg["lr"] = lr

            if cfg.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            opt.zero_grad(set_to_none=True)
            _, loss = model(bx, by)

            if not torch.isfinite(loss):
                print(f"[NON-FINITE] {name} diverged at step {step}: loss={loss.item()}")
                failed.add(name)
                continue

            loss.backward()
            opt.step()

            if cfg.device == "cuda":
                torch.cuda.synchronize()
            interval_timing[name] += (time.perf_counter() - t0) * 1000.0

        if is_eval:
            eval_cells, ms_cells = [], []
            for name, rm in raw_models.items():
                if name in failed:
                    eval_cells.append(f"{'FAILED':<{col_widths[name]}}")
                    ms_cells.append("  ---")
                else:
                    v_ce, v_gen = evaluate_model(rm, val_batches)
                    eval_cells.append(f"{f'{v_ce:7.4f} {v_gen:7.4f}':<{col_widths[name]}}")
                    ms_cells.append(f"{interval_timing[name] / BASE_CFG.eval_interval:5.1f}")
                    interval_timing[name] = 0.0
            print(f"{step:<6d} | " + " | ".join(eval_cells) + " | " + " / ".join(ms_cells))

    if BASE_CFG.gen:
        print("\n=== Generation Sanity Checks ===")
        X_seed, _ = get_batch("train", BASE_CFG.block_size)
        seed = X_seed[0].unsqueeze(0)
        print(f"[PROMPT]: {decode(seed[0].detach().cpu().numpy().tolist())!r}\n")
        for name, rm in raw_models.items():
            if name in failed:
                continue
            rm.eval()
            with torch.no_grad():
                outp = rm.generate(seed, BASE_CFG.gen_tokens, temperature=1.0, top_k=5)
            rm.train()
            print(f"[{name:<22}]: {decode(outp[0, -BASE_CFG.gen_tokens:].detach().cpu().numpy().tolist())!r}")


if __name__ == "__main__":
    run_benchmarks()