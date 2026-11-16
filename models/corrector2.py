import copy
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
# 1. Experiment Registry & Config
# =====================================================================
EXPERIMENTS_TO_RUN = {
    "ntp_only_zipf_3k":         True,   # Pure next-token baseline (voronoi_zipf_routed_v2_3k)
    "pred_topkhot_sig_0.15":    True,   # TopKHot (sigmoid - target) Two-Stream EMA, lambda=0.15
    "pred_top1_smx_0.15":       True,   # BankTop1SoftmaxSTE (softmax - target), lambda=0.15
    "pred_top1_smx_0.50":       True,   # BankTop1SoftmaxSTE (softmax - target), lambda=0.50
    "pred_top1_hybridv5_0.15":  True,   # BankTop1SoftmaxSTE (lambda=0.15) + HYBRID v5 clustering
}

dataset           = "openwebtext"
device            = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size        = 64
block_size        = 64
n_embd            = 256
n_head            = 8
n_layer           = 4
dropout           = 0.0

eval_interval     = 250
eval_iters        = 25
max_iters         = 2000
learning_rate     = 1e-3

N_CHUNK_SIZE       = 1024
TREE_REFRESH_STEPS = (25, 100, 250, 500)
EMA_DECAY          = 0.99

always_byte       = False
decode, vocab_size = utils.get_meta(dataset, always_byte=always_byte)
# Request block_size + 1 so (bx, by) spans x_0 .. x_{block_size+1} for 2-step predictive scoring
get_batch = utils.get_get_batch(block_size + 1, batch_size, dataset, device, always_byte=always_byte)

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
# 2. Transformer Core (Causal + Two-Stream Diagonal Mask Support)
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

    def forward(self, x, attn_mask: torch.Tensor = None):
        B, T, C = x.size()
        q = self.q_attn(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_attn(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_attn(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=(attn_mask is None),
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

    def forward(self, x, attn_mask: torch.Tensor = None):
        x = x + self.attn(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class TransformerCore(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size + 1, config.n_embd),
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
            x = block(x, attn_mask=None)
        return self.transformer.ln_f(x)

    def forward_two_stream(self, content_emb: torch.Tensor, query_emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        T_c = content_emb.shape[1]
        x = torch.cat([content_emb, query_emb], dim=1)
        for block in self.transformer.h:
            x = block(x, attn_mask=mask)
        return self.transformer.ln_f(x[:, T_c:, :])

def build_two_stream_mask(T_c: int, T_q: int, dev) -> torch.Tensor:
    L = T_c + T_q
    mask = torch.zeros((L, L), dtype=torch.bool, device=dev)
    mask[:T_c, :T_c] = torch.tril(torch.ones((T_c, T_c), dtype=torch.bool, device=dev))
    mask[T_c:, :T_c] = torch.tril(torch.ones((T_q, T_c), dtype=torch.bool, device=dev), diagonal=0)
    mask[T_c:, T_c:] = torch.eye(T_q, dtype=torch.bool, device=dev)
    return mask.unsqueeze(0).unsqueeze(0)

# =====================================================================
# 3. Target-Gradient STE Operators (TopKHot Sigmoid vs Shift-Invariant Softmax)
# =====================================================================

class BankTop1SigmoidSTE(torch.autograd.Function):
    """Your exact TopKHot (k=1, sigmoid - hard_target) adapted for masked banks & 1/N scaling."""
    @staticmethod
    def forward(ctx, logits: torch.Tensor, valid_mask: torch.Tensor, grad_scale: float):
        masked = logits.masked_fill(~valid_mask, -1e9)
        idx = torch.argmax(masked, dim=-1, keepdim=True)
        one_hot = torch.zeros_like(logits).scatter_(-1, idx, 1.0)
        ctx.save_for_backward(masked, valid_mask)
        ctx.grad_scale = grad_scale
        return one_hot

    @staticmethod
    def backward(ctx, g):
        masked, valid_mask = ctx.saved_tensors
        with torch.no_grad():
            neg_g = (-g).masked_fill(~valid_mask, -1e9)
            best_idx = torch.argmax(neg_g, dim=-1, keepdim=True)
            hard_target = torch.zeros_like(masked).scatter_(-1, best_idx, 1.0)
            sig = F.sigmoid(masked)
        grad_x = (ctx.grad_scale * (sig - hard_target)).masked_fill(~valid_mask, 0.0)
        return grad_x, None, None

class BankTop1SoftmaxSTE(torch.autograd.Function):
    """Shift-invariant 1-of-K counterpart (softmax - hard_target) matching cross_entropy scale."""
    @staticmethod
    def forward(ctx, logits: torch.Tensor, valid_mask: torch.Tensor, grad_scale: float):
        masked = logits.masked_fill(~valid_mask, -1e9)
        idx = torch.argmax(masked, dim=-1, keepdim=True)
        one_hot = torch.zeros_like(logits).scatter_(-1, idx, 1.0)
        ctx.save_for_backward(masked, valid_mask)
        ctx.grad_scale = grad_scale
        return one_hot

    @staticmethod
    def backward(ctx, g):
        masked, valid_mask = ctx.saved_tensors
        with torch.no_grad():
            neg_g = (-g).masked_fill(~valid_mask, -1e9)
            best_idx = torch.argmax(neg_g, dim=-1, keepdim=True)
            hard_target = torch.zeros_like(masked).scatter_(-1, best_idx, 1.0)
            probs = F.softmax(masked, dim=-1)
        grad_x = (ctx.grad_scale * (probs - hard_target)).masked_fill(~valid_mask, 0.0)
        return grad_x, None, None

# =====================================================================
# 4. Vocabulary Partitioners (Power-Tree vs Balanced HYBRID v5)
# =====================================================================

@torch.no_grad()
def build_power_tree_buckets(W: torch.Tensor, depth: int):
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
                v = centered.T @ (centered @ v)
                v = v / (torch.linalg.norm(v) + 1e-9)
            order = torch.argsort(centered @ v)
            half = m // 2
            next_buckets.append(idxs[order[:half]])
            next_buckets.append(idxs[order[half:]])
        buckets = next_buckets

    leaf_tokens = torch.zeros((M, S_bucket), device=W.device, dtype=torch.long)
    leaf_valid = torch.zeros((M, S_bucket), device=W.device, dtype=torch.bool)
    for leaf_id, idxs in enumerate(buckets):
        cnt = idxs.shape[0]
        if cnt > 0:
            leaf_tokens[leaf_id, :cnt] = idxs
            leaf_valid[leaf_id, :cnt] = True
    return leaf_tokens, leaf_valid

@torch.no_grad()
def _get_normal_pi(X: torch.Tensor, iters: int = 2) -> torch.Tensor:
    mu = X.mean(dim=0)
    v = torch.randn(X.shape[1], device=X.device, dtype=X.dtype)
    v = v / (torch.linalg.norm(v) + 1e-9)
    for _ in range(iters):
        z = torch.mv(X, v) - torch.dot(mu, v)
        v_next = torch.mv(X.T, z) - z.sum() * mu
        v = v_next / (torch.linalg.norm(v_next) + 1e-9)
    return v

@torch.no_grad()
def _split_dense_island(proj: torch.Tensor, num_bins: int = 64) -> torch.Tensor:
    n_pts = proj.shape[0]
    sample = proj[torch.randint(0, n_pts, (min(4096, n_pts),), device=proj.device)] if n_pts > 4096 else proj
    min_v, max_v = sample.aminmax()
    span = max_v - min_v
    if span <= 1e-6:
        return min_v
    norm_proj = ((sample - min_v) / (span + 1e-7) * (num_bins - 1)).clamp(0, num_bins - 1).long()
    counts = torch.bincount(norm_proj, minlength=num_bins).float()
    is_dense = counts > (0.35 * counts.mean())
    dense_idxs = torch.where(is_dense)[0]
    if dense_idxs.numel() < 2:
        return min_v + 0.5 * span
    first, last = dense_idxs[0], dense_idxs[-1]
    bin_idxs = torch.arange(num_bins, device=proj.device)
    empty_between = (~is_dense) & (bin_idxs >= first) & (bin_idxs <= last)
    padded = F.pad(empty_between.float(), (1, 1), value=0.0)
    diff = padded[1:] - padded[:-1]
    starts, ends = torch.where(diff == 1)[0], torch.where(diff == -1)[0]
    if starts.numel() > 0:
        lengths = ends - starts
        max_idx = torch.argmax(lengths)
        best_center = (ends[max_idx] - 1).float() - (lengths[max_idx].float() / 2.0)
        return min_v + best_center * (span / float(num_bins))
    return min_v + 0.5 * span

@torch.no_grad()
def build_hybrid_v5_buckets(W: torch.Tensor, depth: int):
    """Engine v5 Residual Seeding + Modified e-means (1ep) + Capacity-Balanced Leaf Packing."""
    V, D = W.shape
    M = 1 << depth
    S_bucket = math.ceil(V / M)
    dev = W.device

    # 1. Engine v5 Residual Power Seeding
    centroids = [W.mean(dim=0)]
    sub_n = min(16384, V)
    X_sub = W[torch.randint(0, V, (sub_n,), device=dev)]
    while len(centroids) < M:
        C_curr = torch.stack(centroids)
        c_half = 0.5 * (C_curr * C_curr).sum(dim=1)
        assign = torch.argmax(torch.addmm(-c_half, X_sub, C_curr.T), dim=1)
        target_c = torch.argmax(torch.bincount(assign, minlength=len(centroids))).item()
        pts = X_sub[assign == target_c]
        if pts.shape[0] < 8:
            centroids.append(W[torch.randint(0, V, (1,), device=dev)].squeeze(0))
            continue
        pts_samp = pts[torch.randint(0, pts.shape[0], (min(4096, pts.shape[0]),), device=dev)]
        v = _get_normal_pi(pts_samp, iters=2)
        proj = torch.mv(pts, v)
        thresh = _split_dense_island(proj)
        l_mask = proj < thresh
        if l_mask.sum() == 0 or l_mask.sum() == pts.shape[0]:
            l_mask = proj < torch.median(proj)
            if l_mask.sum() == 0 or l_mask.sum() == pts.shape[0]:
                l_mask = torch.zeros(pts.shape[0], dtype=torch.bool, device=dev)
                l_mask[:pts.shape[0] // 2] = True
        centroids[target_c] = pts[l_mask].mean(dim=0)
        centroids.append(pts[~l_mask].mean(dim=0))
    C = torch.stack(centroids)

    # 2. Modified e-means refinement (1.0 epoch)
    B = max(M * 2, min(V, ((int(round(0.035 * (M ** 0.13) * (V ** 0.88))) + 127) // 128) * 128))
    a0 = min(max(float(0.824 * (M ** -0.46) * (B ** 0.90) * (V ** -0.45)), 0.01), 1.0)
    n_batches = max(1, V // B)
    N_stat = torch.full((M, 1), 1e-10, device=dev, dtype=torch.float32)
    S = C.to(torch.float32) * 1e-10

    for step in range(n_batches):
        alpha = a0 * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * step / n_batches)))
        B_X = W[torch.randint(0, V, (B,), device=dev)]
        scores = torch.addmm(-0.5 * (C * C).sum(dim=1), B_X, C.T)
        assign = torch.argmax(scores, dim=1)
        O = torch.zeros(B, M, device=dev, dtype=torch.float32)
        O.scatter_(1, assign.unsqueeze(1), 1.0)
        S = (1.0 - alpha) * S + alpha * (O.T @ B_X.to(torch.float32))
        N_stat = (1.0 - alpha) * N_stat + alpha * O.sum(dim=0, keepdim=True).T
        valid = (N_stat.squeeze(1) > 1e-6)
        C = torch.where(valid.unsqueeze(1), (S / torch.clamp(N_stat, min=1e-6)).to(C.dtype), C)

    # 3. Capacity-balanced assignment into static (M, S_bucket) buckets
    scores_all = torch.addmm(-0.5 * (C * C).sum(dim=1), W, C.T)
    best_scores, assign = torch.max(scores_all, dim=1)

    leaf_tokens = torch.zeros((M, S_bucket), device=dev, dtype=torch.long)
    leaf_valid = torch.zeros((M, S_bucket), device=dev, dtype=torch.bool)
    counts = torch.zeros(M, dtype=torch.long, device=dev)
    overflow = []

    order = torch.argsort(best_scores, descending=True)
    assign_cpu = assign[order].tolist()
    order_cpu = order.tolist()

    for tok_id, cid in zip(order_cpu, assign_cpu):
        cnt = counts[cid].item()
        if cnt < S_bucket:
            leaf_tokens[cid, cnt] = tok_id
            leaf_valid[cid, cnt] = True
            counts[cid] = cnt + 1
        else:
            overflow.append(tok_id)

    if overflow:
        free_cids = []
        for cid in range(M):
            rem = S_bucket - counts[cid].item()
            if rem > 0:
                free_cids.extend([(cid, counts[cid].item() + s) for s in range(rem)])
        for tok_id, (cid, slot) in zip(overflow, free_cids):
            leaf_tokens[cid, slot] = tok_id
            leaf_valid[cid, slot] = True

    return leaf_tokens, leaf_valid

# =====================================================================
# 5. Exact Eval Helpers + Zipf Routed Head
# =====================================================================

def voronoi_cce_exact_per_token(h_seq: torch.Tensor, centroids: torch.Tensor, targets: torch.Tensor, chunk_size: int = 1024):
    B, T, D = h_seq.shape
    h_flat = h_seq.reshape(-1, D)
    t_flat = targets.reshape(-1)
    c_half_norm = 0.5 * (centroids * centroids).sum(dim=1)
    losses = []
    for i in range(0, h_flat.shape[0], chunk_size):
        logits = torch.addmm(-c_half_norm, h_flat[i:i + chunk_size], centroids.T)
        losses.append(F.cross_entropy(logits, t_flat[i:i + chunk_size], reduction='none'))
    return torch.cat(losses, dim=0).view(B, T)

@torch.no_grad()
def voronoi_argmax_tokens(h_seq: torch.Tensor, centroids: torch.Tensor, chunk_size: int = 1024) -> torch.Tensor:
    B, T, D = h_seq.shape
    h_flat = h_seq.reshape(-1, D)
    c_half_norm = 0.5 * (centroids * centroids).sum(dim=1)
    preds = []
    for i in range(0, h_flat.shape[0], chunk_size):
        logits = torch.addmm(-c_half_norm, h_flat[i:i + chunk_size], centroids.T)
        preds.append(torch.argmax(logits, dim=-1))
    return torch.cat(preds, dim=0).view(B, T)

class VoronoiZipfRoutedHead(nn.Module):
    def __init__(self, vocab_size: int, k_bank: int = 3072, k_heavy: int = 1024,
                 route_chunk_size: int = 512, use_hybrid_v5: bool = False):
        super().__init__()
        self.vocab_size = vocab_size
        self.k_bank = k_bank
        self.route_chunk_size = route_chunk_size
        self.use_hybrid_v5 = use_hybrid_v5

        self.K_heavy = min(k_heavy, max(0, vocab_size - k_bank))
        V_tail = max(1, vocab_size - self.K_heavy)
        self.depth = 8 if V_tail > 2048 else 4
        self.M = 1 << self.depth
        self.S_bucket = math.ceil(V_tail / self.M)

        K_rem = max(1, k_bank - self.K_heavy)
        max_p = max(1, (K_rem - 1) // self.S_bucket)
        self.P = min(self.M - 1, max(1, min((2 * K_rem) // (3 * self.S_bucket), max_p)))
        self.K_routed = self.P * self.S_bucket
        self.K_tail = max(1, k_bank - self.K_heavy - self.K_routed)

        self.tail_scale = float((self.M - self.P) * self.S_bucket) / float(self.K_tail)
        self.exact_cut = self.K_heavy + self.K_routed

        self.register_buffer("token_freq", torch.zeros(vocab_size, dtype=torch.float32))
        self.register_buffer("heavy_idx", torch.arange(self.K_heavy, dtype=torch.long))
        self.register_buffer("heavy_valid", torch.ones(self.K_heavy, dtype=torch.bool))
        self.register_buffer("leaf_tokens", torch.zeros((self.M, self.S_bucket), dtype=torch.long))
        self.register_buffer("leaf_valid", torch.zeros((self.M, self.S_bucket), dtype=torch.bool))

    @torch.no_grad()
    def track_targets(self, targets: torch.Tensor):
        counts = torch.bincount(targets.reshape(-1), minlength=self.vocab_size).float()
        self.token_freq.mul_(0.98).add_(counts)

    @torch.no_grad()
    def refresh_tree(self, W: torch.Tensor):
        if self.vocab_size <= self.k_bank:
            return
        freq_order = torch.argsort(self.token_freq, descending=True)
        heavy = freq_order[:self.K_heavy]
        tail_vocab = freq_order[self.K_heavy:]
        if self.use_hybrid_v5:
            lt_local, lv = build_hybrid_v5_buckets(W[tail_vocab].detach(), self.depth)
        else:
            lt_local, lv = build_power_tree_buckets(W[tail_vocab].detach(), self.depth)
        self.heavy_idx.copy_(heavy)
        self.leaf_tokens.copy_(tail_vocab[lt_local])
        self.leaf_valid.copy_(lv)

    @torch.no_grad()
    def get_coarse_centroids(self, W: torch.Tensor):
        leaf_embs = W[self.leaf_tokens] * self.leaf_valid.unsqueeze(-1).to(W.dtype)
        leaf_counts = self.leaf_valid.sum(dim=1, keepdim=True).clamp(min=1).to(W.dtype)
        leaf_centroids = leaf_embs.sum(dim=1) / leaf_counts
        r_half_norm = 0.5 * (leaf_centroids * leaf_centroids).sum(dim=1)
        return leaf_centroids, r_half_norm

    def _score_chunk(self, h_chunk: torch.Tensor, t_chunk: torch.Tensor, W: torch.Tensor,
                     leaf_centroids: torch.Tensor, r_half_norm: torch.Tensor):
        if self.vocab_size <= self.k_bank:
            c_half_norm = 0.5 * (W * W).sum(dim=1)
            bank_logits = torch.addmm(-c_half_norm, h_chunk, W.T)
            chunk_loss = F.cross_entropy(bank_logits, t_chunk, reduction='none')
            bank_valid = torch.ones_like(bank_logits, dtype=torch.bool)
            return chunk_loss, bank_logits, bank_valid, W

        with torch.no_grad():
            coarse_scores = torch.addmm(-r_half_norm, h_chunk, leaf_centroids.T)
            cluster_heat = F.softmax(coarse_scores, dim=-1).sum(dim=0)
            sorted_clusters = torch.argsort(cluster_heat, descending=True)

            routed_clusters = sorted_clusters[:self.P]
            unrouted_clusters = sorted_clusters[self.P:]

            routed_idx = self.leaf_tokens[routed_clusters].reshape(-1)
            routed_valid = self.leaf_valid[routed_clusters].reshape(-1)

            rand_u = torch.randint(0, self.M - self.P, (self.K_tail,), device=h_chunk.device)
            rand_s = torch.randint(0, self.S_bucket, (self.K_tail,), device=h_chunk.device)
            chosen_u = unrouted_clusters[rand_u]
            tail_idx = self.leaf_tokens[chosen_u, rand_s]
            tail_valid = self.leaf_valid[chosen_u, rand_s]

            bank_idx = torch.cat([self.heavy_idx, routed_idx, tail_idx], dim=0)
            bank_valid = torch.cat([self.heavy_valid, routed_valid, tail_valid], dim=0).unsqueeze(0)

        sub_c = W[bank_idx]
        sub_half_norm = 0.5 * (sub_c * sub_c).sum(dim=1)
        bank_logits = torch.addmm(-sub_half_norm, h_chunk, sub_c.T)

        pos_c = W[t_chunk]
        pos_score = (h_chunk * pos_c).sum(dim=-1) - 0.5 * (pos_c * pos_c).sum(dim=-1)

        is_target = (bank_idx.unsqueeze(0) == t_chunk.unsqueeze(1))
        masked_logits = bank_logits.masked_fill(is_target | (~bank_valid), -1e9)

        exact_logits = masked_logits[:, :self.exact_cut]
        tail_logits = masked_logits[:, self.exact_cut:]

        bank_max = masked_logits.max(dim=1, keepdim=True).values
        overall_max = torch.maximum(pos_score.unsqueeze(1), bank_max)

        pos_exp = torch.exp(pos_score.unsqueeze(1) - overall_max)
        exact_exp = torch.exp(exact_logits - overall_max).sum(dim=1, keepdim=True)
        tail_exp = self.tail_scale * torch.exp(tail_logits - overall_max).sum(dim=1, keepdim=True)

        total_lse = overall_max + torch.log(pos_exp + exact_exp + tail_exp)
        chunk_loss = (-pos_score.unsqueeze(1) + total_lse).squeeze(1)
        return chunk_loss, bank_logits, bank_valid.expand_as(bank_logits), sub_c

    def forward_head(self, h_seq: torch.Tensor, W: torch.Tensor, targets: torch.Tensor,
                     leaf_centroids: torch.Tensor, r_half_norm: torch.Tensor,
                     ste_mode: str = "none", grad_scale: float = 0.0):
        B, T, D = h_seq.shape
        h_flat = h_seq.reshape(-1, D)
        t_flat = targets.reshape(-1)
        N = h_flat.shape[0]

        losses, g2_chunks = [], []
        for i in range(0, N, self.route_chunk_size):
            l_c, b_logits, b_valid, sub_c = self._score_chunk(
                h_flat[i:i + self.route_chunk_size], t_flat[i:i + self.route_chunk_size],
                W, leaf_centroids, r_half_norm
            )
            losses.append(l_c)
            if ste_mode == "sigmoid":
                one_hot = BankTop1SigmoidSTE.apply(b_logits, b_valid, grad_scale)
                g2_chunks.append(one_hot @ sub_c.detach())
            elif ste_mode == "softmax":
                one_hot = BankTop1SoftmaxSTE.apply(b_logits, b_valid, grad_scale)
                g2_chunks.append(one_hot @ sub_c.detach())

        loss_map = torch.cat(losses, dim=0).view(B, T)
        g2_seq = torch.cat(g2_chunks, dim=0).view(B, T, D) if ste_mode != "none" else None
        return loss_map, g2_seq

# =====================================================================
# 6. Predictive Two-Stream Model Wrapper
# =====================================================================

class PredictiveVoronoiModel(nn.Module):
    def __init__(self, config: GPTConfig, lambda_guess: float = 0.0,
                 ste_mode: str = "none", use_hybrid_v5: bool = False):
        super().__init__()
        self.config = config
        self.lambda_guess = lambda_guess
        self.ste_mode = ste_mode

        self.core = TransformerCore(config)
        self.voronoi_head = VoronoiZipfRoutedHead(
            config.vocab_size, k_bank=3072, k_heavy=1024,
            route_chunk_size=512, use_hybrid_v5=use_hybrid_v5
        )

        # Frozen EMA shadow scorer prevents private-cipher cheating on L_guess
        self.scorer = copy.deepcopy(self.core)
        for p in self.scorer.parameters():
            p.requires_grad_(False)

        self.register_buffer("two_stream_mask", build_two_stream_mask(config.block_size + 1, config.block_size, torch.device('cpu')))

    @torch.no_grad()
    def sync_ema(self, decay: float = EMA_DECAY):
        for p_ema, p in zip(self.scorer.parameters(), self.core.parameters()):
            p_ema.lerp_(p.detach(), 1.0 - decay)

    def forward(self, bx: torch.Tensor, by: torch.Tensor):
        # bx: (B, 65) = x_0..x_64 ; by: (B, 65) = x_1..x_65
        B, T_c = bx.shape
        T = T_c - 1  # 64
        dev = bx.device
        W = self.core.transformer.wte.weight

        full_x = self.core(bx)  # (B, 65, D)
        leaf_c, r_half_c = self.voronoi_head.get_coarse_centroids(W)

        use_ste = self.training and (self.lambda_guess > 0.0) and (self.ste_mode != "none")
        # Normalize TopKHot/SoftmaxSTE gradient to match L_all.mean()'s 1/(B*T) per-token scale
        grad_scale = self.lambda_guess / float(B * T)

        L_all, g2_all = self.voronoi_head.forward_head(
            full_x, W, by, leaf_c, r_half_c,
            ste_mode=(self.ste_mode if use_ste else "none"),
            grad_scale=grad_scale
        )

        if not use_ste:
            return L_all.mean(), L_all.mean().detach(), torch.tensor(0.0, device=dev)

        # Pass 2: Two-Stream Counterfactual Scoring on frozen EMA scorer
        g2_emb = g2_all[:, :T, :]  # (B, 64, D)
        with torch.no_grad():
            pos_c = torch.arange(0, T_c, dtype=torch.long, device=dev)
            content_emb = self.scorer.transformer.wte(bx) + self.scorer.transformer.wpe(pos_c)
            pos_q = self.scorer.transformer.wpe(torch.arange(1, T + 1, dtype=torch.long, device=dev))

        query_emb = g2_emb + pos_q
        gen2_x = self.scorer.forward_two_stream(content_emb, query_emb, self.two_stream_mask)

        W_ema = self.scorer.transformer.wte.weight
        leaf_e, r_half_e = self.voronoi_head.get_coarse_centroids(W_ema)
        L_guess, _ = self.voronoi_head.forward_head(
            gen2_x, W_ema, by[:, 1:], leaf_e, r_half_e, ste_mode="none"
        )

        # Keep L_guess.mean() in graph so dL_guess/dg2_emb flows into BankTop1STE
        loss = L_all.mean() + self.lambda_guess * L_guess.mean()
        return loss, L_all.mean().detach(), L_guess.mean().detach()

    @torch.no_grad()
    def evaluate_predictive_metrics(self, bx: torch.Tensor, by: torch.Tensor):
        B, T_c = bx.shape
        T = T_c - 1
        dev = bx.device
        W = self.core.transformer.wte.weight
        W_ema = self.scorer.transformer.wte.weight

        full_x = self.core(bx)
        L_all_exact = voronoi_cce_exact_per_token(full_x, W, by)

        pred_tok = voronoi_argmax_tokens(full_x[:, :T, :], W)
        true_tok = by[:, :T]
        is_diff = (pred_tok != true_tok)

        pos_c = torch.arange(0, T_c, dtype=torch.long, device=dev)
        content_emb = self.scorer.transformer.wte(bx) + self.scorer.transformer.wpe(pos_c)
        pos_q = self.scorer.transformer.wpe(torch.arange(1, T + 1, dtype=torch.long, device=dev))

        q_gen = W_ema[pred_tok] + pos_q
        q_data = W_ema[true_tok] + pos_q
        q_skip = pos_q.unsqueeze(0).expand(B, -1, -1)

        t_future = by[:, 1:]
        h_gen = self.scorer.forward_two_stream(content_emb, q_gen, self.two_stream_mask)
        h_data = self.scorer.forward_two_stream(content_emb, q_data, self.two_stream_mask)
        h_skip = self.scorer.forward_two_stream(content_emb, q_skip, self.two_stream_mask)

        L_gen = voronoi_cce_exact_per_token(h_gen, W_ema, t_future)
        L_data = voronoi_cce_exact_per_token(h_data, W_ema, t_future)
        L_skip = voronoi_cce_exact_per_token(h_skip, W_ema, t_future)

        diff_cnt = is_diff.sum().clamp(min=1)
        return {
            "L_all": L_all_exact.mean().item(),
            "L_data": L_data.mean().item(),
            "L_gen": L_gen.mean().item(),
            "IG": (L_skip - L_gen).mean().item(),
            "Delta": (L_data - L_gen).mean().item(),
            "DiffWin": (((L_gen < L_data) & is_diff).sum().float() / diff_cnt.float()).item(),
            "Match": ((~is_diff).float().mean()).item(),
        }

# =====================================================================
# 7. Benchmark Runner (with Speed in Eval & Summary Reports)
# =====================================================================

def run_benchmarks():
    gptconf = GPTConfig()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    specs = {
        "ntp_only_zipf_3k":        {"lambda_guess": 0.0,  "ste_mode": "none",    "use_hybrid_v5": False},
        "pred_topkhot_sig_0.15":   {"lambda_guess": 0.15, "ste_mode": "sigmoid", "use_hybrid_v5": False},
        "pred_top1_smx_0.15":      {"lambda_guess": 0.15, "ste_mode": "softmax", "use_hybrid_v5": False},
        "pred_top1_smx_0.50":      {"lambda_guess": 0.50, "ste_mode": "softmax", "use_hybrid_v5": False},
        "pred_top1_hybridv5_0.15": {"lambda_guess": 0.15, "ste_mode": "softmax", "use_hybrid_v5": True},
    }
    active = [k for k, v in EXPERIMENTS_TO_RUN.items() if v]
    tokens_per_step = batch_size * block_size

    print(f"Device: {device} | Vocab: {vocab_size} | Block: {block_size} | Dim: {n_embd} | Layers: {n_layer} | always_byte={always_byte}")
    print(f"Active Experiments: {active}")

    models, optimizers = {}, {}
    bx0, by0 = get_batch('train', block_size + 1)

    print("\n--- Eager Sanity Checks ---")
    for name in active:
        torch.manual_seed(1337)
        m = PredictiveVoronoiModel(gptconf, **specs[name]).to(device)
        m.sync_ema(decay=0.0)
        m.voronoi_head.refresh_tree(m.core.transformer.wte.weight)

        m.train()
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=learning_rate)
        opt.zero_grad(set_to_none=True)
        loss, l_all, l_guess = m(bx0[:4], by0[:4])
        loss.backward()
        opt.zero_grad(set_to_none=True)

        m.eval()
        with torch.no_grad():
            met = m.evaluate_predictive_metrics(bx0[:4], by0[:4])
        m.train()
        print(f"  [SANITY OK] {name:<25} (L_all={met['L_all']:.3f}, IG={met['IG']:+.3f})")

        models[name] = m
        optimizers[name] = opt

    print("\n--- Compiling & Warming Up ---")
    for name, m in models.items():
        t_c = time.perf_counter()
        m.core.forward = torch.compile(m.core.forward)
        m.scorer.forward_two_stream = torch.compile(m.scorer.forward_two_stream)
        m.voronoi_head._score_chunk = torch.compile(m.voronoi_head._score_chunk)
        for _ in range(2):
            optimizers[name].zero_grad(set_to_none=True)
            loss, _, _ = m(bx0, by0)
            loss.backward()
            optimizers[name].zero_grad(set_to_none=True)
        if device == 'cuda':
            torch.cuda.synchronize()
        print(f"  [COMPILED]  {name:<25} in {(time.perf_counter() - t_c):.2f} s")

    print("=" * 155)
    print(f"{'Step':<6} | " + " | ".join([f"{n:<25}" for n in models.keys()]))
    print("-" * 155)

    timing_accum = {n: 0.0 for n in models.keys()}
    interval_timing = {n: 0.0 for n in models.keys()}
    interval_steps = 0
    base_name = active[0]

    for it in range(max_iters):
        bx, by = get_batch('train', block_size + 1)

        for m in models.values():
            m.voronoi_head.track_targets(by)
            if it in TREE_REFRESH_STEPS:
                m.voronoi_head.refresh_tree(m.core.transformer.wte.weight)

        step_stats = {}
        for name, m in models.items():
            opt = optimizers[name]
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            opt.zero_grad(set_to_none=True)
            loss, l_all, l_guess = m(bx, by)
            loss.backward()
            opt.step()
            m.sync_ema(decay=EMA_DECAY)

            if device == 'cuda':
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000.0
            timing_accum[name] += dt
            interval_timing[name] += dt
            step_stats[name] = (l_all.item(), l_guess.item(), dt)

        interval_steps += 1

        if it % 50 == 0 or it == max_iters - 1:
            row = f"{it:<6d} | " + " | ".join(
                [f"a:{la:<5.2f} g:{lg:<5.2f}({dt:<4.1f}ms)" for (la, lg, dt) in step_stats.values()]
            )
            print(row)

        if it % eval_interval == 0 and it > 0:
            val_batches = [get_batch('val', block_size + 1) for _ in range(eval_iters)]
            base_avg_ms = interval_timing[base_name] / max(1, interval_steps)
            print("-" * 155)
            print(f"[EVAL @ {it} EXACT FULL-VOCAB PREDICTIVE METRICS + SPEED]")
            for name, m in models.items():
                m.eval()
                agg = {k: 0.0 for k in ("L_all", "L_data", "L_gen", "IG", "Delta", "DiffWin", "Match")}
                for vx, vy in val_batches:
                    res = m.evaluate_predictive_metrics(vx, vy)
                    for k in agg:
                        agg[k] += res[k] / len(val_batches)
                m.train()

                avg_ms = interval_timing[name] / max(1, interval_steps)
                tok_s_k = (tokens_per_step / avg_ms)  # tokens/ms == k tokens/sec
                rel_spd = base_avg_ms / avg_ms if avg_ms > 0 else 0.0
                print(
                    f"  {name:<25} | L_all={agg['L_all']:.4f} | L_data={agg['L_data']:.4f} | "
                    f"L_gen={agg['L_gen']:.4f} | IG={agg['IG']:+.4f} | "
                    f"Delta={agg['Delta']:+.4f} | DiffWin={agg['DiffWin']*100:5.2f}% | Match={agg['Match']*100:5.2f}% | "
                    f"{avg_ms:5.1f}ms/st ({tok_s_k:5.1f}k tok/s, {rel_spd:.2f}x)"
                )
            print("-" * 155)
            interval_timing = {n: 0.0 for n in models.keys()}
            interval_steps = 0

    print("=" * 155)
    base_tot_ms = timing_accum[base_name]
    for name, t_tot in timing_accum.items():
        avg_ms = t_tot / max_iters
        tok_s_k = tokens_per_step / avg_ms
        rel_spd = base_tot_ms / t_tot if t_tot > 0 else 0.0
        print(
            f"Total Step Time - {name:<25}: {t_tot:9.1f} ms "
            f"({avg_ms:5.2f} ms/step | {tok_s_k:5.1f}k tok/s | {rel_spd:.2f}x)"
        )

if __name__ == "__main__":
    run_benchmarks()