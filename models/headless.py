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
    # Baselines (Exact Full-Vocab References)
    "baseline_dot_standard":    True,   # Unchunked F.linear + F.cross_entropy
    "baseline_dot_cce_chunked": False,   # Exact chunked dot-product CCE
    "voronoi_cce_chunked":      False,   # Exact full-vocab Voronoi CCE

    # Earlier Approximate Voronoi Methods
    "voronoi_adaptive_cce":     False,  # Original random static bank
    "voronoi_calibrated_bank":  False,  # Unbiased uniform static bank
    "voronoi_routed_bank":      False,  # Coarse routing (chunk=1024, no Zipf anchors)
    "voronoi_zipf_routed_v1":   False,  # k_bank=2048, K_heavy=256, chunk=256

    # Active Experiments
    "voronoi_power_tree_hsm":   False,   # Power-Iter 2-level Voronoi tree (early refresh)
    "voronoi_zipf_routed_v2":   False,   # k_bank=2560, K_heavy=768,  route_chunk=512
    "voronoi_zipf_routed_v2_3k":True,   # k_bank=3072, K_heavy=1024, route_chunk=512
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
gen               = True
# Static chunking & tree sizes
N_CHUNK_SIZE      = 1024  # Tokens per GEMM chunk for exact methods
K_ADAPTIVE_BANK   = 2048  # Default static bank size
TREE_REFRESH_STEPS = (25, 100, 250, 500)  # Refresh early while embeddings organize, then freeze

always_byte       = False
decode, vocab_size = utils.get_meta(dataset, always_byte=always_byte)
get_batch = utils.get_get_batch(block_size, batch_size, dataset, device, always_byte=always_byte)

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
# 3. Power-Iteration Anisotropic Vocabulary Partitioner
# =====================================================================

@torch.no_grad()
def build_power_tree_buckets(W: torch.Tensor, depth: int):
    """
    Partitions V embeddings into M = 2^depth balanced leaves of static max size
    S_bucket = ceil(V / M) using recursive Power-Iteration principal-axis splits.
    """
    V, D = W.shape
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
            sub = W[idxs]
            centered = sub - sub.mean(dim=0, keepdim=True)
            v = torch.randn(D, device=W.device, dtype=W.dtype)
            v = v / (torch.linalg.norm(v) + 1e-9)
            for _ in range(3):
                z = centered @ v
                v = centered.T @ z
                v = v / (torch.linalg.norm(v) + 1e-9)
            proj = centered @ v
            order = torch.argsort(proj)
            half = m // 2
            next_buckets.append(idxs[order[:half]])
            next_buckets.append(idxs[order[half:]])
        buckets = next_buckets

    leaf_tokens = torch.zeros((M, S_bucket), device=W.device, dtype=torch.long)
    leaf_valid = torch.zeros((M, S_bucket), device=W.device, dtype=torch.bool)
    token_to_leaf = torch.zeros(V, device=W.device, dtype=torch.long)
    token_to_slot = torch.zeros(V, device=W.device, dtype=torch.long)

    for leaf_id, idxs in enumerate(buckets):
        cnt = idxs.shape[0]
        if cnt > 0:
            leaf_tokens[leaf_id, :cnt] = idxs
            leaf_valid[leaf_id, :cnt] = True
            token_to_leaf[idxs] = leaf_id
            token_to_slot[idxs] = torch.arange(cnt, device=W.device)

    return leaf_tokens, leaf_valid, token_to_leaf, token_to_slot

# =====================================================================
# 4. Loss Implementations
# =====================================================================

def loss_dot_standard(h_flat: torch.Tensor, centroids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = F.linear(h_flat, centroids)
    return F.cross_entropy(logits, targets)

def loss_dot_cce_chunked(h_flat: torch.Tensor, centroids: torch.Tensor, targets: torch.Tensor, chunk_size: int = N_CHUNK_SIZE) -> torch.Tensor:
    n_tokens = h_flat.shape[0]
    total_loss = 0.0
    for i in range(0, n_tokens, chunk_size):
        h_chunk = h_flat[i:i + chunk_size]
        t_chunk = targets[i:i + chunk_size]
        chunk_logits = h_chunk @ centroids.T
        total_loss = total_loss + F.cross_entropy(chunk_logits, t_chunk, reduction='sum')
    return total_loss / n_tokens

def loss_voronoi_cce_chunked(h_flat: torch.Tensor, centroids: torch.Tensor, targets: torch.Tensor, chunk_size: int = N_CHUNK_SIZE) -> torch.Tensor:
    c_half_norm = 0.5 * (centroids * centroids).sum(dim=1)
    n_tokens = h_flat.shape[0]
    total_loss = 0.0

    for i in range(0, n_tokens, chunk_size):
        h_chunk = h_flat[i:i + chunk_size]
        t_chunk = targets[i:i + chunk_size]
        chunk_logits = torch.addmm(-c_half_norm, h_chunk, centroids.T)
        total_loss = total_loss + F.cross_entropy(chunk_logits, t_chunk, reduction='sum')

    return total_loss / n_tokens

def loss_voronoi_adaptive_cce(h_flat: torch.Tensor, centroids: torch.Tensor, targets: torch.Tensor, k_bank: int = K_ADAPTIVE_BANK) -> torch.Tensor:
    N, D = h_flat.shape
    V = centroids.shape[0]
    device = h_flat.device

    if V <= k_bank:
        c_half_norm = 0.5 * (centroids * centroids).sum(dim=1)
        logits = torch.addmm(-c_half_norm, h_flat, centroids.T)
        return F.cross_entropy(logits, targets)

    perm = torch.randint(0, V, (k_bank,), device=device)
    n_target_fill = min(N, k_bank // 2)
    perm[:n_target_fill] = targets[:n_target_fill]

    sub_c = centroids[perm]
    c_half_norm = 0.5 * (sub_c * sub_c).sum(dim=1)

    pos_c = centroids[targets]
    pos_score = (h_flat * pos_c).sum(dim=-1) - 0.5 * (pos_c * pos_c).sum(dim=-1)

    bank_logits = torch.addmm(-c_half_norm, h_flat, sub_c.T)

    scale_factor = float(V - k_bank) / float(k_bank)
    bank_max = bank_logits.max(dim=1, keepdim=True).values
    scaled_tail_sum = scale_factor * torch.exp(bank_logits - bank_max).sum(dim=1, keepdim=True)

    overall_max = torch.maximum(pos_score.unsqueeze(1), bank_max)
    total_lse = overall_max + torch.log(
        torch.exp(pos_score.unsqueeze(1) - overall_max) +
        torch.exp(bank_logits - overall_max).sum(dim=1, keepdim=True) +
        scaled_tail_sum * torch.exp(bank_max - overall_max)
    )
    return (-pos_score.unsqueeze(1) + total_lse).mean()

def loss_voronoi_calibrated_bank(h_flat: torch.Tensor, centroids: torch.Tensor, targets: torch.Tensor, k_bank: int = K_ADAPTIVE_BANK) -> torch.Tensor:
    N, D = h_flat.shape
    V = centroids.shape[0]
    device = h_flat.device

    if V <= k_bank:
        c_half_norm = 0.5 * (centroids * centroids).sum(dim=1)
        logits = torch.addmm(-c_half_norm, h_flat, centroids.T)
        return F.cross_entropy(logits, targets)

    bank_idx = torch.randint(0, V, (k_bank,), device=device)
    sub_c = centroids[bank_idx]
    c_half_norm = 0.5 * (sub_c * sub_c).sum(dim=1)

    pos_c = centroids[targets]
    pos_score = (h_flat * pos_c).sum(dim=-1) - 0.5 * (pos_c * pos_c).sum(dim=-1)

    bank_logits = torch.addmm(-c_half_norm, h_flat, sub_c.T)
    is_target = (bank_idx.unsqueeze(0) == targets.unsqueeze(1))
    masked_logits = bank_logits.masked_fill(is_target, -1e9)

    bank_max = masked_logits.max(dim=1, keepdim=True).values
    overall_max = torch.maximum(pos_score.unsqueeze(1), bank_max)

    scale_factor = float(V) / float(k_bank)
    pos_exp = torch.exp(pos_score.unsqueeze(1) - overall_max)
    neg_exp_sum = scale_factor * torch.exp(masked_logits - overall_max).sum(dim=1, keepdim=True)

    total_lse = overall_max + torch.log(pos_exp + neg_exp_sum)
    return (-pos_score.unsqueeze(1) + total_lse).mean()

# =====================================================================
# 5. Model Wrappers
# =====================================================================

class ModelDotStandard(nn.Module):
    def __init__(self, core: TransformerCore):
        super().__init__()
        self.core = core

    def forward(self, idx, targets=None):
        x = self.core(idx)
        if targets is not None:
            return None, loss_dot_standard(x.view(-1, self.core.config.n_embd), self.core.transformer.wte.weight, targets.view(-1))
        return None, None

class ModelDotCCEChunked(nn.Module):
    def __init__(self, core: TransformerCore, chunk_size: int = N_CHUNK_SIZE):
        super().__init__()
        self.core = core
        self.chunk_size = chunk_size

    def forward(self, idx, targets=None):
        x = self.core(idx)
        if targets is not None:
            return None, loss_dot_cce_chunked(x.view(-1, self.core.config.n_embd), self.core.transformer.wte.weight, targets.view(-1), self.chunk_size)
        return None, None

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

class ModelVoronoiCalibratedBank(nn.Module):
    def __init__(self, core: TransformerCore, k_bank: int = K_ADAPTIVE_BANK):
        super().__init__()
        self.core = core
        self.k_bank = k_bank

    def forward(self, idx, targets=None):
        x = self.core(idx)
        if targets is not None:
            return None, loss_voronoi_calibrated_bank(x.view(-1, self.core.config.n_embd), self.core.transformer.wte.weight, targets.view(-1), self.k_bank)
        return None, None

class ModelVoronoiPowerTreeHSM(nn.Module):
    """
    2-Level Hierarchical Voronoi Softmax built via Power-Iteration recursive bisection.
    """
    def __init__(self, core: TransformerCore, chunk_size: int = N_CHUNK_SIZE):
        super().__init__()
        self.core = core
        self.chunk_size = chunk_size
        V = core.config.vocab_size
        self.depth = 8 if V > 2048 else 4
        M = 1 << self.depth
        S_bucket = math.ceil(V / M)
        w_device = core.transformer.wte.weight.device

        self.register_buffer("leaf_tokens", torch.zeros((M, S_bucket), dtype=torch.long, device=w_device))
        self.register_buffer("leaf_valid", torch.zeros((M, S_bucket), dtype=torch.bool, device=w_device))
        self.register_buffer("token_to_leaf", torch.zeros(V, dtype=torch.long, device=w_device))
        self.register_buffer("token_to_slot", torch.zeros(V, dtype=torch.long, device=w_device))
        self.refresh_tree()

    @torch.no_grad()
    def refresh_tree(self):
        lt, lv, t2l, t2s = build_power_tree_buckets(self.core.transformer.wte.weight, self.depth)
        self.leaf_tokens.copy_(lt)
        self.leaf_valid.copy_(lv)
        self.token_to_leaf.copy_(t2l)
        self.token_to_slot.copy_(t2s)

    def compute_loss(self, h_flat: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        centroids = self.core.transformer.wte.weight
        leaf_embs = centroids[self.leaf_tokens] * self.leaf_valid.unsqueeze(-1).to(centroids.dtype)
        leaf_counts = self.leaf_valid.sum(dim=1, keepdim=True).clamp(min=1).to(centroids.dtype)
        leaf_centroids = leaf_embs.sum(dim=1) / leaf_counts
        r_half_norm = 0.5 * (leaf_centroids * leaf_centroids).sum(dim=1)

        t_leaf = self.token_to_leaf[targets]
        t_slot = self.token_to_slot[targets]

        n_tokens = h_flat.shape[0]
        total_loss = 0.0
        for i in range(0, n_tokens, self.chunk_size):
            h_chunk = h_flat[i:i + self.chunk_size]
            tl_chunk = t_leaf[i:i + self.chunk_size]
            ts_chunk = t_slot[i:i + self.chunk_size]

            coarse_logits = torch.addmm(-r_half_norm, h_chunk, leaf_centroids.T)
            coarse_loss = F.cross_entropy(coarse_logits, tl_chunk, reduction='sum')

            cand_tokens = self.leaf_tokens[tl_chunk]
            cand_valid = self.leaf_valid[tl_chunk]
            cand_c = centroids[cand_tokens]
            cand_half = 0.5 * (cand_c * cand_c).sum(dim=-1)

            fine_logits = torch.bmm(h_chunk.unsqueeze(1), cand_c.transpose(1, 2)).squeeze(1) - cand_half
            fine_logits = fine_logits.masked_fill(~cand_valid, -1e9)
            fine_loss = F.cross_entropy(fine_logits, ts_chunk, reduction='sum')

            total_loss = total_loss + coarse_loss + fine_loss

        return total_loss / n_tokens

    def forward(self, idx, targets=None):
        x = self.core(idx)
        if targets is not None:
            return None, self.compute_loss(x.view(-1, self.core.config.n_embd), targets.view(-1))
        return None, None

class ModelVoronoiRoutedBank(nn.Module):
    def __init__(self, core: TransformerCore, k_bank: int = K_ADAPTIVE_BANK, chunk_size: int = N_CHUNK_SIZE):
        super().__init__()
        self.core = core
        self.k_bank = k_bank
        self.chunk_size = chunk_size
        V = core.config.vocab_size
        self.depth = 8 if V > 2048 else 4
        self.M = 1 << self.depth
        self.S_bucket = math.ceil(V / self.M)
        self.P = min(self.M - 1, max(1, (3 * k_bank) // (4 * self.S_bucket)))
        self.K_routed = self.P * self.S_bucket
        self.K_tail = k_bank - self.K_routed
        w_device = core.transformer.wte.weight.device

        self.register_buffer("leaf_tokens", torch.zeros((self.M, self.S_bucket), dtype=torch.long, device=w_device))
        self.register_buffer("leaf_valid", torch.zeros((self.M, self.S_bucket), dtype=torch.bool, device=w_device))
        self.refresh_tree()

    @torch.no_grad()
    def refresh_tree(self):
        lt, lv, _, _ = build_power_tree_buckets(self.core.transformer.wte.weight, self.depth)
        self.leaf_tokens.copy_(lt)
        self.leaf_valid.copy_(lv)

    def forward(self, idx, targets=None):
        x = self.core(idx)
        if targets is None:
            return None, None

        h_flat = x.view(-1, self.core.config.n_embd)
        t_flat = targets.view(-1)
        centroids = self.core.transformer.wte.weight
        V = centroids.shape[0]

        if V <= self.k_bank:
            return None, loss_voronoi_cce_chunked(h_flat, centroids, t_flat, self.chunk_size)

        with torch.no_grad():
            leaf_embs = centroids[self.leaf_tokens] * self.leaf_valid.unsqueeze(-1).to(centroids.dtype)
            leaf_counts = self.leaf_valid.sum(dim=1, keepdim=True).clamp(min=1).to(centroids.dtype)
            leaf_centroids = leaf_embs.sum(dim=1) / leaf_counts
            r_half_norm = 0.5 * (leaf_centroids * leaf_centroids).sum(dim=1)

        N = h_flat.shape[0]
        total_loss = 0.0
        tail_scale = float((self.M - self.P) * self.S_bucket) / float(self.K_tail)

        for i in range(0, N, self.chunk_size):
            h_chunk = h_flat[i:i + self.chunk_size]
            t_chunk = t_flat[i:i + self.chunk_size]

            with torch.no_grad():
                coarse_scores = torch.addmm(-r_half_norm, h_chunk, leaf_centroids.T)
                cluster_heat = coarse_scores.max(dim=0).values
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

                bank_idx = torch.cat([routed_idx, tail_idx], dim=0)
                bank_valid = torch.cat([routed_valid, tail_valid], dim=0).unsqueeze(0)

            sub_c = centroids[bank_idx]
            sub_half_norm = 0.5 * (sub_c * sub_c).sum(dim=1)
            bank_logits = torch.addmm(-sub_half_norm, h_chunk, sub_c.T)

            pos_c = centroids[t_chunk]
            pos_score = (h_chunk * pos_c).sum(dim=-1) - 0.5 * (pos_c * pos_c).sum(dim=-1)

            is_target = (bank_idx.unsqueeze(0) == t_chunk.unsqueeze(1))
            masked_logits = bank_logits.masked_fill(is_target | (~bank_valid), -1e9)

            routed_logits = masked_logits[:, :self.K_routed]
            tail_logits = masked_logits[:, self.K_routed:]

            bank_max = masked_logits.max(dim=1, keepdim=True).values
            overall_max = torch.maximum(pos_score.unsqueeze(1), bank_max)

            pos_exp = torch.exp(pos_score.unsqueeze(1) - overall_max)
            routed_exp = torch.exp(routed_logits - overall_max).sum(dim=1, keepdim=True)
            tail_exp = tail_scale * torch.exp(tail_logits - overall_max).sum(dim=1, keepdim=True)

            total_lse = overall_max + torch.log(pos_exp + routed_exp + tail_exp)
            total_loss = total_loss + (-pos_score.unsqueeze(1) + total_lse).sum()

        return None, total_loss / N

class ModelVoronoiZipfRoutedBank(nn.Module):
    """
    Zipfian Anchor + Power-Iteration Coarse-Routed Bank (strictly disjoint):
    - Top k_heavy tokens (tracked via EMA of training targets) are always in the bank.
    - Remaining (V - k_heavy) tail tokens are partitioned into M=256 anisotropic clusters.
    - Each route_chunk_size token chunk routes by total coarse probability mass (softmax.sum).
    """
    def __init__(self, core: TransformerCore, k_bank: int = 2560, k_heavy: int = 768, route_chunk_size: int = 512):
        super().__init__()
        self.core = core
        self.k_bank = k_bank
        self.route_chunk_size = route_chunk_size
        V = core.config.vocab_size

        self.K_heavy = min(k_heavy, max(0, V - k_bank))
        V_tail = max(1, V - self.K_heavy)
        self.depth = 8 if V_tail > 2048 else 4
        self.M = 1 << self.depth
        self.S_bucket = math.ceil(V_tail / self.M)

        K_rem = max(1, k_bank - self.K_heavy)
        max_p_for_tail = max(1, (K_rem - 1) // self.S_bucket)
        self.P = min(self.M - 1, max(1, min((2 * K_rem) // (3 * self.S_bucket), max_p_for_tail)))
        self.K_routed = self.P * self.S_bucket
        self.K_tail = max(1, k_bank - self.K_heavy - self.K_routed)

        w_device = core.transformer.wte.weight.device
        self.register_buffer("token_freq", torch.zeros(V, dtype=torch.float32, device=w_device))
        self.register_buffer("heavy_idx", torch.arange(self.K_heavy, dtype=torch.long, device=w_device))
        self.register_buffer("leaf_tokens", torch.zeros((self.M, self.S_bucket), dtype=torch.long, device=w_device))
        self.register_buffer("leaf_valid", torch.zeros((self.M, self.S_bucket), dtype=torch.bool, device=w_device))
        self.refresh_tree()

    @torch.no_grad()
    def track_targets(self, targets: torch.Tensor):
        counts = torch.bincount(targets.reshape(-1), minlength=self.token_freq.shape[0]).float()
        self.token_freq.mul_(0.98).add_(counts)

    @torch.no_grad()
    def refresh_tree(self):
        V = self.core.config.vocab_size
        if V <= self.k_bank:
            return
        freq_order = torch.argsort(self.token_freq, descending=True)
        heavy = freq_order[:self.K_heavy]
        tail_vocab = freq_order[self.K_heavy:]  # Strictly disjoint from heavy

        W = self.core.transformer.wte.weight
        lt_local, lv, _, _ = build_power_tree_buckets(W[tail_vocab], self.depth)
        lt_global = tail_vocab[lt_local]

        self.heavy_idx.copy_(heavy)
        self.leaf_tokens.copy_(lt_global)
        self.leaf_valid.copy_(lv)

    def forward(self, idx, targets=None):
        x = self.core(idx)
        if targets is None:
            return None, None

        h_flat = x.view(-1, self.core.config.n_embd)
        t_flat = targets.view(-1)
        centroids = self.core.transformer.wte.weight
        V = centroids.shape[0]

        if V <= self.k_bank:
            return None, loss_voronoi_cce_chunked(h_flat, centroids, t_flat, N_CHUNK_SIZE)

        with torch.no_grad():
            leaf_embs = centroids[self.leaf_tokens] * self.leaf_valid.unsqueeze(-1).to(centroids.dtype)
            leaf_counts = self.leaf_valid.sum(dim=1, keepdim=True).clamp(min=1).to(centroids.dtype)
            leaf_centroids = leaf_embs.sum(dim=1) / leaf_counts
            r_half_norm = 0.5 * (leaf_centroids * leaf_centroids).sum(dim=1)

        N = h_flat.shape[0]
        total_loss = 0.0
        tail_scale = float((self.M - self.P) * self.S_bucket) / float(self.K_tail)
        heavy_valid = torch.ones(self.K_heavy, dtype=torch.bool, device=h_flat.device)
        exact_cut = self.K_heavy + self.K_routed

        for i in range(0, N, self.route_chunk_size):
            h_chunk = h_flat[i:i + self.route_chunk_size]
            t_chunk = t_flat[i:i + self.route_chunk_size]

            with torch.no_grad():
                coarse_scores = torch.addmm(-r_half_norm, h_chunk, leaf_centroids.T)
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
            sub_half_norm = 0.5 * (sub_c * sub_c).sum(dim=1)
            bank_logits = torch.addmm(-sub_half_norm, h_chunk, sub_c.T)

            pos_c = centroids[t_chunk]
            pos_score = (h_chunk * pos_c).sum(dim=-1) - 0.5 * (pos_c * pos_c).sum(dim=-1)

            is_target = (bank_idx.unsqueeze(0) == t_chunk.unsqueeze(1))
            masked_logits = bank_logits.masked_fill(is_target | (~bank_valid), -1e9)

            exact_logits = masked_logits[:, :exact_cut]
            tail_logits = masked_logits[:, exact_cut:]

            bank_max = masked_logits.max(dim=1, keepdim=True).values
            overall_max = torch.maximum(pos_score.unsqueeze(1), bank_max)

            pos_exp = torch.exp(pos_score.unsqueeze(1) - overall_max)
            exact_exp = torch.exp(exact_logits - overall_max).sum(dim=1, keepdim=True)
            tail_exp = tail_scale * torch.exp(tail_logits - overall_max).sum(dim=1, keepdim=True)

            total_lse = overall_max + torch.log(pos_exp + exact_exp + tail_exp)
            total_loss = total_loss + (-pos_score.unsqueeze(1) + total_lse).sum()

        return None, total_loss / N

# =====================================================================
# 6. Benchmark Orchestration
# =====================================================================

def evaluate_full_vocab_loss(name, raw_model, get_batch, eval_iters):
    raw_model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(eval_iters):
            vx, vy = get_batch('val', block_size)
            h = raw_model.core(vx).view(-1, n_embd)
            t = vy.view(-1)
            centroids = raw_model.core.transformer.wte.weight

            if "dot" in name:
                loss = loss_dot_cce_chunked(h, centroids, t, chunk_size=1024)
            elif name == "voronoi_power_tree_hsm":
                loss = raw_model.compute_loss(h, t)
            else:
                loss = loss_voronoi_cce_chunked(h, centroids, t, chunk_size=1024)
            total_loss += loss.item()
    raw_model.train()
    return total_loss / eval_iters

def run_benchmarks():
    gptconf = GPTConfig()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    print(f"Device: {device} | Vocab: {vocab_size} | Emb Dim: {n_embd} | Layers: {n_layer} | always_byte={always_byte}")
    if vocab_size <= K_ADAPTIVE_BANK:
        print(f"[NOTE] vocab_size ({vocab_size}) <= K_ADAPTIVE_BANK ({K_ADAPTIVE_BANK}): bank methods fall back to exact full-vocab loss. Set always_byte=False for 50k BPE.")
    active_experiments = [k for k, v in EXPERIMENTS_TO_RUN.items() if v]
    print(f"Active Experiments: {active_experiments}")
    print("=" * 130)

    raw_models = {}
    models = {}
    optimizers = {}

    for name in active_experiments:
        torch.manual_seed(1337)
        core = TransformerCore(gptconf).to(device)

        if name == "baseline_dot_standard":
            m = ModelDotStandard(core).to(device)
        elif name == "baseline_dot_cce_chunked":
            m = ModelDotCCEChunked(core, chunk_size=N_CHUNK_SIZE).to(device)
        elif name == "voronoi_cce_chunked":
            m = ModelVoronoiCCEChunked(core, chunk_size=N_CHUNK_SIZE).to(device)
        elif name == "voronoi_adaptive_cce":
            m = ModelVoronoiAdaptiveCCE(core, k_bank=K_ADAPTIVE_BANK).to(device)
        elif name == "voronoi_calibrated_bank":
            m = ModelVoronoiCalibratedBank(core, k_bank=K_ADAPTIVE_BANK).to(device)
        elif name == "voronoi_power_tree_hsm":
            m = ModelVoronoiPowerTreeHSM(core, chunk_size=N_CHUNK_SIZE).to(device)
        elif name == "voronoi_routed_bank":
            m = ModelVoronoiRoutedBank(core, k_bank=K_ADAPTIVE_BANK, chunk_size=N_CHUNK_SIZE).to(device)
        elif name == "voronoi_zipf_routed_v1":
            m = ModelVoronoiZipfRoutedBank(core, k_bank=2048, k_heavy=256, route_chunk_size=256).to(device)
        elif name == "voronoi_zipf_routed_v2":
            m = ModelVoronoiZipfRoutedBank(core, k_bank=2560, k_heavy=768, route_chunk_size=512).to(device)
        elif name == "voronoi_zipf_routed_v2_3k":
            m = ModelVoronoiZipfRoutedBank(core, k_bank=3072, k_heavy=1024, route_chunk_size=512).to(device)
        else:
            continue

        raw_models[name] = m
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
    if device == 'cuda':
        torch.cuda.synchronize()

    header = f"{'Step':<6} | " + " | ".join([f"{name[:17]:<17}" for name in models.keys()])
    print(header)
    print("-" * 130)

    timing_accum = {name: 0.0 for name in models.keys()}

    for it in range(max_iters):
        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        bx, by = get_batch('train', block_size)

        # Track target frequencies outside compiled graph
        for rm in raw_models.values():
            if hasattr(rm, "track_targets"):
                rm.track_targets(by)

        # Refresh early while embeddings organize, then freeze to preserve AdamW momentum
        if it in TREE_REFRESH_STEPS:
            for rm in raw_models.values():
                if hasattr(rm, "refresh_tree"):
                    rm.refresh_tree()

        step_times = {}

        for name, model in models.items():
            opt = optimizers[name]
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            opt.zero_grad(set_to_none=True)
            _, loss = model(bx, by)
            loss.backward()
            opt.step()

            if device == 'cuda':
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000.0
            step_times[name] = (loss.item(), dt)
            timing_accum[name] += dt

        if it % 50 == 0 or it == max_iters - 1:
            row = f"{it:<6d} | " + " | ".join([f"{loss:<5.3f}({dt:<4.1f}ms)" for (loss, dt) in step_times.values()])
            print(row)

        if it % eval_interval == 0 and it > 0:
            print("-" * 130)
            eval_metrics = []
            for name, rm in raw_models.items():
                val_loss = evaluate_full_vocab_loss(name, rm, get_batch, eval_iters)
                eval_metrics.append(f"{name}: {val_loss:.4f}")
            print(f"[EVAL @ {it} FULL VOCAB] " + " | ".join(eval_metrics))
            print("-" * 130)

    print("=" * 130)
    for name, t_total in timing_accum.items():
        avg_dt = t_total / max_iters
        print(f"Total Step Time - {name:<26}: {t_total:8.1f} ms ({avg_dt:5.2f} ms/step)")
    if gen:
        print("\n=== Generation Sanity Checks ===")
        X_seed, _ = get_batch('train')
        seed = X_seed[0].unsqueeze(0)
        for name, rm in raw_models.items():
            rm.eval()
            with torch.no_grad():
                outp = rm.generate(seed, 100, temperature=1.0, top_k=5)
            rm.train()
            print(f"[{name}]: {decode(outp[0, -100:].detach().cpu().numpy().tolist())!r}")

if __name__ == "__main__":
    run_benchmarks()