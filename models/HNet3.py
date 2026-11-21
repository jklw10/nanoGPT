from dataclasses import dataclass, replace
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import parts.modules as modules
import parts.quantizer as quantizer
import parts.utils as utils
from cut_cross_entropy import linear_cross_entropy

torch._inductor.config.disable_cpp_codegen = True
torch._dynamo.config.force_parameter_static_shapes = False
torch._dynamo.config.cache_size_limit = 128
torch._dynamo.config.accumulated_cache_size_limit = 1024
if hasattr(torch._dynamo.config, "recompile_limit"):
    torch._dynamo.config.recompile_limit = 128
if hasattr(torch._dynamo.config, "accumulated_recompile_limit"):
    torch._dynamo.config.accumulated_recompile_limit = 1024

# =====================================================================
# 1. Global Training & Benchmark Config
# =====================================================================
dataset        = "openwebtext"
batch_size     = 64
block_size     = 256

max_iters      = 50000
eval_interval  = 1000
eval_iters     = 20
learning_rate  = 1e-3
warmup_iters   = 200
lr_decay_iters = 30000
decay_iters    = 5000
min_lr         = 1e-4

device         = 'cuda' if torch.cuda.is_available() else 'cpu'
gen            = True
g_log          = False

Linear = nn.Linear
wn = torch.nn.utils.parametrizations.weight_norm

decode, vocab_size = utils.get_meta(dataset, always_byte=True)
get_batch = utils.get_get_batch(block_size, batch_size, dataset, device, always_byte=True)

@dataclass(frozen=True)
class HNetConfig:
    patcher_type: str = "voronoi_conv_gpt"
    dim_l1: int = 128
    dim_l2: int = 128
    dim_l3: int = 128
    block_size: int = block_size
    k_gather_l1: int = block_size // 2
    k_gather_l2: int = block_size // 4
    n_head: int = 4
    n_inner: int = 4
    dropout: float = 0.0
    skiptok: int = 1

    # Ablation & architecture flags
    resid: bool = True
    ael: bool = False
    cce: bool = True
    winit: bool = True
    emb_wn: bool = False
    pattlin: bool = False
    wnll: bool = True
    patt_AE: bool = True
    pattcount: int = 256
    pds_skip: bool = True
    gpt_init: bool = False

    # --- Sparse / TopKHot / Clustering / FlexAttention switches ---
    inner_type: str = "standard"     # "standard" | "kattention" | "emeans_topk" | "emeans_flex"
    topk_mode: str = "dense"         # "dense" | "2k_sparse" | "k_sparse"
    cluster_algo: str = "qk_dot"     # "qk_dot" | "qk_mips" | "mag_only" | "engine_v5"
    cluster_on: str = "keys"         # "proj" (linear) | "keys" (learned k_attn(x))
    k_sparse: int = 4                # top-k tokens (kattention) or top-k blocks (cluster attn)
    emeans_clusters: int = 16        # number of centroids (M)
    emeans_block_size: int = 4       # semantic block size (must divide inner seq len)
    router_dim: int = 16             # projection dim for legacy engine_v5 cluster router
    cluster_norm: bool = False       # L2-normalize vectors for engine_v5 clustering & assignment
    mag_sort: bool = False           # Tie-break intra-cluster perm by descending magnitude

BASE_GPT = HNetConfig(patcher_type="voronoi_conv_gpt")

EXPERIMENTS = {
    # --- Active QK Co-Clustering vs Magnitude-Only Control ---
    "mag_only_keys":   (False,  replace(BASE_GPT, inner_type="emeans_topk", cluster_algo="mag_only", cluster_on="keys", topk_mode="dense", k_sparse=4, emeans_clusters=16, emeans_block_size=4)),
    "qk_dot_bs4":      (False,  replace(BASE_GPT, inner_type="emeans_topk", cluster_algo="qk_dot",   topk_mode="dense", k_sparse=4, emeans_clusters=16, emeans_block_size=4)),
    "qk_mips_bs4":     (False,  replace(BASE_GPT, inner_type="emeans_topk", cluster_algo="qk_mips",  topk_mode="dense", k_sparse=4, emeans_clusters=16, emeans_block_size=4)),
    "qk_dot_mag_bs4":  (False, replace(BASE_GPT, inner_type="emeans_topk", cluster_algo="qk_dot",   mag_sort=True, topk_mode="dense", k_sparse=4, emeans_clusters=16, emeans_block_size=4)),
    "qk_mips_mag_bs4": (False, replace(BASE_GPT, inner_type="emeans_topk", cluster_algo="qk_mips",  mag_sort=True, topk_mode="dense", k_sparse=4, emeans_clusters=16, emeans_block_size=4)),

    "qk_dot_flex":  (True, replace(BASE_GPT, inner_type="emeans_flex", cluster_algo="qk_dot",  mag_sort=True, topk_mode="dense", k_sparse=4, emeans_clusters=8, emeans_block_size=32)),
    "qk_mips_flex": (False, replace(BASE_GPT, inner_type="emeans_flex", cluster_algo="qk_mips", mag_sort=True, topk_mode="dense", k_sparse=4, emeans_clusters=8, emeans_block_size=32)),

    # --- Reference Baselines ---
    "conv_gpt_dim128":   (False, replace(BASE_GPT)),
    "kattn_dense":       (False, replace(BASE_GPT, inner_type="kattention", topk_mode="dense", k_sparse=4)),
    "v5_norm_keys":      (False, replace(BASE_GPT, inner_type="emeans_topk", cluster_algo="engine_v5", cluster_on="keys", cluster_norm=True,  mag_sort=False, topk_mode="dense", k_sparse=4, emeans_clusters=16, emeans_block_size=4)),
    "v5_magsort_keys":   (False, replace(BASE_GPT, inner_type="emeans_topk", cluster_algo="engine_v5", cluster_on="keys", cluster_norm=False, mag_sort=True,  topk_mode="dense", k_sparse=4, emeans_clusters=16, emeans_block_size=4)),
    "v5_bs4_keys_dense": (False, replace(BASE_GPT, inner_type="emeans_topk", cluster_algo="engine_v5", cluster_on="keys", topk_mode="dense", k_sparse=4, emeans_clusters=16, emeans_block_size=4)),
    "conv_gpt_pyramid":  (False, replace(BASE_GPT, dim_l1=64, dim_l2=128, dim_l3=256, n_head=4)),

    # --- FlexAttention Backend  ---
    "v5_flex_2k":        (False, replace(BASE_GPT, inner_type="emeans_flex", cluster_algo="engine_v5", cluster_on="keys", cluster_norm=True, topk_mode="2k_sparse", k_sparse=2, emeans_clusters=16, emeans_block_size=64)),
}

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# =====================================================================
# 2. Causal Conv Primitives
# =====================================================================

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        self.pad = kernel_size - 1
        self.kernel_size = kernel_size
        self.proj = nn.Linear(in_channels * kernel_size, out_channels, bias=bias)

    def forward(self, x):
        if self.pad > 0:
            x = F.pad(x, (0, 0, self.pad, 0))
        xw = x.unfold(1, self.kernel_size, 1).reshape(x.shape[0], -1, self.proj.in_features)
        return self.proj(xw)

class CausalConvBlock(nn.Module):
    def __init__(self, dim, kernel_size=7, mlp_ratio=4):
        super().__init__()
        self.ln1 = modules.LayerNorm(dim, False)
        self.conv = CausalConv1d(dim, dim, kernel_size=kernel_size, bias=False)
        self.ln2 = modules.LayerNorm(dim, False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio, bias=False),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim, bias=False),
        )

    def forward(self, x):
        x = x + self.conv(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

# =====================================================================
# 3. Patcher Family (Voronoi Segment & GPT Pre-LN Conv Patchers)
# =====================================================================

class VoronoiCausalCrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, num_heads=4):
        super().__init__()
        assert query_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.q_proj = nn.Linear(query_dim, query_dim, bias=False)
        self.k_proj = nn.Linear(key_dim, query_dim, bias=False)
        self.v_proj = nn.Linear(key_dim, query_dim, bias=False)
        self.out_proj = nn.Linear(query_dim, query_dim, bias=False)
        self.time_decay = nn.Parameter(torch.full((1, num_heads, 1, 1), 0.1))

    def forward(self, q_x, q_t, kv_x, kv_t):
        B, L_q, D = q_x.shape
        _, L_kv, _ = kv_x.shape
        H, d_h = self.num_heads, self.head_dim

        q = self.q_proj(q_x).view(B, L_q, H, d_h).transpose(1, 2)
        k = self.k_proj(kv_x).view(B, L_kv, H, d_h).transpose(1, 2)
        v = self.v_proj(kv_x).view(B, L_kv, H, d_h).transpose(1, 2)

        k_half = 0.5 * (k * k).sum(dim=-1, keepdim=True).transpose(-2, -1)
        logits = (torch.matmul(q, k.transpose(-2, -1)) - k_half) / math.sqrt(d_h)

        dt = (q_t.view(B, 1, L_q, 1) - kv_t.view(B, 1, 1, L_kv)).clamp(min=0.0)
        logits = logits - F.softplus(self.time_decay) * dt

        causal_mask = (kv_t.view(B, 1, 1, L_kv) <= q_t.view(B, 1, L_q, 1))
        logits = logits.masked_fill(~causal_mask, float('-inf'))

        attn = F.softmax(logits, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L_q, D)
        return self.out_proj(out)

class VoronoiPatcher(nn.Module):
    def __init__(self, outer_dim, inner_dim, outer_seq, inner_seq, cfg: HNetConfig):
        super().__init__()
        self.outer_dim = outer_dim
        self.inner_dim = inner_dim
        self.inner_seq = inner_seq
        self.outer_seq = outer_seq
        self.cfg = cfg

        self.up_proj = modules.PattLayer(outer_dim, inner_dim, cfg.pattcount) if cfg.pattlin else Linear(outer_dim, inner_dim, bias=False)
        self.AE = modules.PattLayer(inner_dim, inner_dim, cfg.pattcount) if cfg.patt_AE else modules.Autoencoder(
            inner_dim, inner_dim, inner_dim, inner_dim, quantizer.threshotSTE_softmax
        )

        self.output_pos_emb = nn.Embedding(outer_seq, inner_dim)
        self.down_block = modules.Block(inner_dim, cfg.n_head, 0.0, False)
        self.down_scatter2 = modules.CombineBlock(outer_dim, cfg.n_head, 0.0, False)
        self.down_proj = modules.PattLayer(inner_dim, outer_dim, cfg.pattcount) if cfg.pattlin else Linear(inner_dim, outer_dim, bias=False)
        self.down_scan = modules.Block(outer_dim, cfg.n_head, 0.0, False)

        self.up_gate = nn.Sequential(
            nn.Linear(outer_dim, max(1, outer_dim // 4), bias=False),
            nn.GELU(),
            nn.Linear(max(1, outer_dim // 4), 1)
        )
        self.novelty_scale = nn.Parameter(torch.tensor(0.5))
        self.pool_q = nn.Linear(outer_dim, outer_dim, bias=False)
        self.pool_k = nn.Linear(outer_dim, outer_dim, bias=False)
        self.pool_alpha = nn.Parameter(torch.tensor(0.5))

        self.res_norm = modules.LayerNorm(outer_dim, False)
        self.down_norm = modules.LayerNorm(inner_dim, False)
        self.query_block = modules.Block(outer_dim, cfg.n_head, 0.0, False)
        self.proj_block = modules.Block(inner_dim, cfg.n_head, 0.0, False)
        self.causal_upsampler = VoronoiCausalCrossAttention(inner_dim, inner_dim, num_heads=cfg.n_head)

    def abstract_up(self, x, conf=None):
        B, L, D = x.shape
        gathered = self.query_block(x)
        h_n = utils.rms_norm(gathered)
        gate_logits = self.up_gate(h_n).squeeze(-1)

        with torch.no_grad():
            sim = torch.bmm(h_n, h_n.transpose(1, 2)) / float(D)
            strict_past = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=-1)
            max_past_sim = sim.masked_fill(~strict_past.unsqueeze(0), -1e9).max(dim=-1).values
            novelty = (1.0 - max_past_sim).clamp(min=0.0, max=2.0)

            sel_score = gate_logits + self.novelty_scale * novelty
            if self.training:
                u = torch.rand_like(sel_score).clamp(1e-5, 1.0 - 1e-5)
                gumbel = -torch.log(-torch.log(u))
                sel_score = sel_score + 0.1 * gumbel
            sel_score[:, 0] = 1e9

            _, top_indices = torch.topk(sel_score, self.inner_seq, dim=-1)
            sorted_indices, _ = torch.sort(top_indices, dim=-1)

        anchor_h = torch.gather(h_n, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, D))
        anchor_probs = torch.sigmoid(gate_logits)
        sel_probs = torch.gather(anchor_probs, 1, sorted_indices).unsqueeze(-1)
        anchor_ste = anchor_h * (1.0 + sel_probs - sel_probs.detach())

        t_outer = torch.arange(L, device=x.device, dtype=x.dtype).view(1, -1, 1).expand(B, -1, -1)
        t_inner = torch.gather(t_outer, 1, sorted_indices.unsqueeze(-1))

        pq = self.pool_q(anchor_ste)
        pk = self.pool_k(h_n)
        pk_half = 0.5 * (pk * pk).sum(dim=-1).unsqueeze(1)
        voronoi_scores = (torch.bmm(pq, pk.transpose(1, 2)) - pk_half) / math.sqrt(D)

        pool_mask = (t_outer.squeeze(-1).unsqueeze(1) <= t_inner)
        voronoi_scores = voronoi_scores.masked_fill(~pool_mask, float('-inf'))
        pool_weights = F.softmax(voronoi_scores, dim=-1)
        pooled = torch.bmm(pool_weights, h_n)

        compressed = utils.rms_norm(anchor_ste + self.pool_alpha * pooled)
        up = self.proj_block(self.up_proj(compressed))

        hot, aux, ae_up = None, 0.0, up
        if self.cfg.ael:
            if self.cfg.patt_AE:
                ae_up = self.AE(up)
            else:
                ae_up, hot = self.AE(up)
            aux = F.mse_loss(up, ae_up)
            up = ae_up

        return up, [x, ae_up, hot, gate_logits, aux, t_outer, t_inner]

    def abstract_down(self, x, residual, decaying=None, conf=None):
        residual_x, up_ae, hot, gate, upaux, t_outer, t_inner = residual
        B, L, _ = residual_x.shape
        aux = 0.0
        x = self.down_norm(x)
        if self.cfg.ael:
            if self.cfg.patt_AE:
                aux = F.mse_loss(self.AE(x)[:, :-1, :], up_ae[:, 1:, :].detach())
            else:
                aux = F.binary_cross_entropy(self.AE.quant(x)[:, :-1, :], hot.detach()[:, 1:, :])

        pos = torch.arange(0, L, dtype=torch.long, device=x.device)
        pos_emb = self.output_pos_emb(pos).unsqueeze(0).expand(B, L, -1)

        scattered = self.causal_upsampler(pos_emb, t_outer, x, t_inner)
        pds = self.down_proj(self.down_block(scattered))

        if self.cfg.resid:
            pos_emb_outer = pos_emb if self.outer_dim == self.inner_dim else self.down_proj(pos_emb)
            query = self.res_norm(residual_x + pos_emb_outer)
            xout = self.down_scatter2(pds, query, causal=True)
            if self.cfg.pds_skip:
                xout = xout + pds
        else:
            xout = pds
        xout = self.down_scan(xout)
        return xout, aux + upaux

class VoronoiSegmentPatcher(VoronoiPatcher):
    def __init__(self, outer_dim, inner_dim, outer_seq, inner_seq, cfg: HNetConfig, pin_tail=False, novelty_window=None):
        super().__init__(outer_dim, inner_dim, outer_seq, inner_seq, cfg)
        self.pin_tail = pin_tail
        self.novelty_window = novelty_window

class VoronoiSegmentGPTPatcher(VoronoiSegmentPatcher):
    """
    GPT Pre-LN U-Net Patcher:
    - No Post-LN on the residual trunk; no CombineBlock outer attention bypass.
    - Pre-LN on pos_emb query (q_norm) and before down_proj (pds_norm).
    - Clean additive residual highway: xout = residual_x + pds.
    """
    def __init__(self, outer_dim, inner_dim, outer_seq, inner_seq, cfg: HNetConfig, pin_tail=True, novelty_window=16):
        super().__init__(outer_dim, inner_dim, outer_seq, inner_seq, cfg, pin_tail=pin_tail, novelty_window=novelty_window)
        self.q_norm = modules.LayerNorm(inner_dim, False)
        self.pds_norm = modules.LayerNorm(inner_dim, False)
        self.down_scatter2 = nn.Identity()
        self.res_norm = nn.Identity()

    def abstract_up(self, x, conf=None):
        B, L, D = x.shape
        gathered = self.query_block(x)
        h_n = utils.rms_norm(gathered)
        gate_logits = self.up_gate(h_n).squeeze(-1)

        with torch.no_grad():
            sim = torch.bmm(h_n, h_n.transpose(1, 2)) / float(D)
            if self.novelty_window is not None:
                pos = torch.arange(L, device=x.device)
                diff = pos.unsqueeze(1) - pos.unsqueeze(0)
                past_mask = (diff >= 1) & (diff <= self.novelty_window)
            else:
                past_mask = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=-1)

            max_past_sim = sim.masked_fill(~past_mask.unsqueeze(0), -1e9).max(dim=-1).values
            novelty = (1.0 - max_past_sim).clamp(min=0.0, max=2.0)

            sel_score = gate_logits + self.novelty_scale * novelty
            if self.training:
                u = torch.rand_like(sel_score).clamp(1e-5, 1.0 - 1e-5)
                gumbel = -torch.log(-torch.log(u))
                sel_score = sel_score + 0.1 * gumbel
            sel_score[:, 0] = 1e9
            if self.pin_tail:
                sel_score[:, -1] = 1e9

            _, top_indices = torch.topk(sel_score, self.inner_seq, dim=-1)
            sorted_indices, _ = torch.sort(top_indices, dim=-1)

        anchor_h = torch.gather(h_n, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, D))
        anchor_probs = torch.sigmoid(gate_logits)
        sel_probs = torch.gather(anchor_probs, 1, sorted_indices).unsqueeze(-1)
        anchor_ste = anchor_h * (1.0 + sel_probs - sel_probs.detach())

        t_outer = torch.arange(L, device=x.device, dtype=x.dtype).view(1, -1, 1).expand(B, -1, -1)
        t_inner = torch.gather(t_outer, 1, sorted_indices.unsqueeze(-1))

        pq = self.pool_q(anchor_ste)
        pk = self.pool_k(h_n)
        pk_half = 0.5 * (pk * pk).sum(dim=-1).unsqueeze(1)
        voronoi_scores = (torch.bmm(pq, pk.transpose(1, 2)) - pk_half) / math.sqrt(D)

        pos_l = torch.arange(L, device=x.device, dtype=torch.long).view(1, 1, L)
        k_end = sorted_indices.unsqueeze(-1)
        k_start = F.pad(sorted_indices[:, :-1], (1, 0), value=-1).unsqueeze(-1)
        seg_mask = (pos_l > k_start) & (pos_l <= k_end)

        voronoi_scores = voronoi_scores.masked_fill(~seg_mask, float('-inf'))
        pool_weights = F.softmax(voronoi_scores, dim=-1)
        pooled = torch.bmm(pool_weights, h_n)

        compressed = utils.rms_norm(anchor_ste + self.pool_alpha * pooled)
        up = self.proj_block(self.up_proj(compressed))

        hot, aux, ae_up = None, 0.0, up
        if self.cfg.ael:
            if self.cfg.patt_AE:
                ae_up = self.AE(up)
            else:
                ae_up, hot = self.AE(up)
            aux = F.mse_loss(up, ae_up)
            up = ae_up

        return up, [gathered, ae_up, hot, gate_logits, aux, t_outer, t_inner]

    def abstract_down(self, x, residual, decaying=None, conf=None):
        residual_x, up_ae, hot, gate, upaux, t_outer, t_inner = residual
        B, L, _ = residual_x.shape
        aux = 0.0
        x = self.down_norm(x)
        if self.cfg.ael:
            if self.cfg.patt_AE:
                aux = F.mse_loss(self.AE(x)[:, :-1, :], up_ae[:, 1:, :].detach())
            else:
                aux = F.binary_cross_entropy(self.AE.quant(x)[:, :-1, :], hot.detach()[:, 1:, :])

        pos = torch.arange(0, L, dtype=torch.long, device=x.device)
        pos_emb = self.output_pos_emb(pos).unsqueeze(0).expand(B, L, -1)

        scattered = self.causal_upsampler(self.q_norm(pos_emb), t_outer, x, t_inner)
        pds = self.down_proj(self.pds_norm(self.down_block(scattered)))

        xout = (residual_x + pds) if self.cfg.resid else pds
        xout = self.down_scan(xout)
        return xout, aux + upaux

class VoronoiConvGPTPatcher(VoronoiSegmentGPTPatcher):
    def __init__(self, outer_dim, inner_dim, outer_seq, inner_seq, cfg: HNetConfig, pin_tail=True, novelty_window=16):
        super().__init__(outer_dim, inner_dim, outer_seq, inner_seq, cfg, pin_tail=pin_tail, novelty_window=novelty_window)
        self.query_block = CausalConvBlock(outer_dim, kernel_size=7)
        self.up_gate = nn.Sequential(
            CausalConv1d(outer_dim, max(1, outer_dim // 2), kernel_size=5, bias=False),
            nn.GELU(),
            CausalConv1d(max(1, outer_dim // 2), 1, kernel_size=3, bias=True),
        )
        self.down_block = CausalConvBlock(inner_dim, kernel_size=5)
        self.down_scan = CausalConvBlock(outer_dim, kernel_size=7)

# =====================================================================
# 3b. TopKHot, Kattention, Engine v5/v10, QK Co-Clustering & FlexAttention
# =====================================================================

try:
    from torch.nn.attention.flex_attention import flex_attention, BlockMask
    HAS_FLEX_ATTENTION = True
except ImportError:
    flex_attention = None
    BlockMask = None
    HAS_FLEX_ATTENTION = False


class TopKHot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k, mode="dense"):
        k = min(k, x.size(-1))
        _, indices = torch.topk(x, k=k, dim=-1)
        active_mask = torch.zeros_like(x).scatter_(-1, indices, 1.0)
        valid_positions = (x > float('-inf')).to(x.dtype)
        active_mask = active_mask * valid_positions

        ctx.save_for_backward(x, active_mask, valid_positions)
        ctx.k = k
        ctx.mode = mode
        return active_mask

    @staticmethod
    def backward(ctx, g):
        x, active_mask, valid_positions = ctx.saved_tensors
        k = ctx.k
        mode = ctx.mode

        if mode == "k_sparse":
            grad_x = g * active_mask
            return grad_x, None, None

        with torch.no_grad():
            score_for_target = -g
            score_for_target = score_for_target.masked_fill(valid_positions == 0, float('-inf'))
            _, topk_indices = torch.topk(score_for_target, k=k, dim=-1)
            hard_target = torch.zeros_like(x).scatter_(-1, topk_indices, 1.0) * valid_positions

        valid_count = valid_positions.sum(dim=-1, keepdim=True).clamp(min=1.0)
        g_scale = (g * valid_positions).abs().sum(dim=-1, keepdim=True) / valid_count

        if mode == "dense":
            grad_x = g_scale * (torch.sigmoid(x) - hard_target) * valid_positions
        elif mode == "2k_sparse":
            grad_x = g_scale * (torch.sigmoid(x) * active_mask - hard_target) * valid_positions
        else:
            raise ValueError(f"Unknown TopKHot backward mode: {mode}")

        return grad_x, None, None


class Kattention(nn.Module):
    """Token-level Top-K sparse attention with configurable TopKHot backward mode."""
    def __init__(self, n_embd, n_head, k_sparse=4, topk_mode="dense", dropout=0.0, bias=False):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.k_sparse = k_sparse
        self.topk_mode = topk_mode

        self.q_attn = nn.Linear(n_embd, n_embd, bias=bias)
        self.k_attn = nn.Linear(n_embd, n_embd, bias=bias)
        self.v_attn = nn.Linear(n_embd, n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.qm = nn.Parameter(torch.ones(self.head_dim).normal_(mean=1.0, std=0.1))

    def forward(self, x):
        B, T, C = x.size()
        q = self.q_attn(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_attn(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_attn(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        k_val = min(self.k_sparse, T)
        scale_t = math.log(max(T, 2))
        att = ((q * (scale_t * self.qm)) @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool)).view(1, 1, T, T)
        att = att.masked_fill(~causal_mask, float('-inf'))

        k_hot_mask = TopKHot.apply(att, k_val, self.topk_mode)
        probs = F.softmax(att, dim=-1) * k_hot_mask
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)

        y = self.attn_dropout(probs) @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class KattentionBlock(nn.Module):
    def __init__(self, n_embd, n_head, k_sparse=4, topk_mode="dense", dropout=0.0, bias=False):
        super().__init__()
        self.ln_1 = modules.LayerNorm(n_embd, bias=bias)
        self.attn = Kattention(n_embd, n_head, k_sparse=k_sparse, topk_mode=topk_mode, dropout=dropout, bias=bias)
        self.ln_2 = modules.LayerNorm(n_embd, bias=bias)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=bias),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=bias),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@torch.no_grad()
def static_engine_v5(X: torch.Tensor, k: int, sub_n: int = 4096, power_iters: int = 2) -> torch.Tensor:
    """
    100% static-shape, graph-break-free GPU Engine v5 (Residual Power Seeding + Dense Island Slicing).
    Used for legacy engine_v5 / FlexAttention single-space clustering.
    """
    n, d = X.shape
    device = X.device
    orig_dtype = X.dtype

    if n > sub_n:
        idx = torch.randint(0, n, (sub_n,), device=device)
        X_sub = X[idx]
    else:
        X_sub = X

    X_sub = torch.nan_to_num(X_sub.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    dtype = torch.float32

    C = torch.zeros(k, d, device=device, dtype=dtype)
    C[0] = X_sub.mean(dim=0)

    num_bins = 64
    tri_upper = torch.triu(torch.ones(num_bins, num_bins, device=device, dtype=torch.bool))
    bin_arange = torch.arange(num_bins, device=device)
    len_matrix = bin_arange[None, :] - bin_arange[:, None] + 1
    c_arange = torch.arange(k, device=device)

    for m in range(1, k):
        c_half = 0.5 * (C * C).sum(dim=1)
        scores = torch.addmm(-c_half, X_sub, C.T)
        active_c_mask = (c_arange < m).unsqueeze(0)
        scores = scores.masked_fill(~active_c_mask, float('-inf'))
        assign = torch.argmax(scores, dim=1)

        O = F.one_hot(assign, num_classes=k).to(dtype)
        counts = O.sum(dim=0).masked_fill(c_arange >= m, -1.0)
        target_cluster = torch.argmax(counts)

        cluster_mask = (assign == target_cluster).to(dtype)
        cnt = cluster_mask.sum().clamp(min=1.0)
        w = cluster_mask.unsqueeze(1)
        mu = (w * X_sub).sum(dim=0) / cnt

        v = torch.randn(d, device=device, dtype=dtype)
        v = v / (torch.linalg.norm(v) + 1e-9)
        for _ in range(power_iters):
            z = ((X_sub @ v) - torch.dot(mu, v)) * cluster_mask
            v_next = (X_sub.T @ z) - z.sum() * mu
            v = v_next / (torch.linalg.norm(v_next) + 1e-9)

        proj = torch.nan_to_num(X_sub @ v, nan=0.0, posinf=1e4, neginf=-1e4)
        proj_for_min = torch.where(cluster_mask > 0, proj, torch.full_like(proj, 1e9))
        proj_for_max = torch.where(cluster_mask > 0, proj, torch.full_like(proj, -1e9))
        min_v = proj_for_min.min()
        max_v = proj_for_max.max()
        span = (max_v - min_v).clamp(min=1e-6)

        raw_bin = (proj - min_v) / span * float(num_bins - 1)
        raw_bin = torch.nan_to_num(raw_bin, nan=0.0, posinf=float(num_bins - 1), neginf=0.0)
        bin_idx = raw_bin.clamp(0, num_bins - 1).long()
        bin_counts = torch.zeros(num_bins, device=device, dtype=dtype).scatter_add_(0, bin_idx, cluster_mask)
        is_dense = bin_counts > (0.35 * bin_counts.mean())

        has_dense_left = is_dense.to(torch.int32).cumsum(dim=0) > 0
        has_dense_right = is_dense.flip(0).to(torch.int32).cumsum(dim=0).flip(0) > 0
        empty_between = (~is_dense) & (has_dense_left & has_dense_right)

        not_e = (~empty_between).to(torch.int32)
        zero_prefix = not_e.cumsum(dim=0)
        zp_before = F.pad(zero_prefix[:-1], (1, 0), value=0)
        all_empty_ij = tri_upper & ((zero_prefix[None, :] - zp_before[:, None]) == 0)
        masked_lens = torch.where(all_empty_ij, len_matrix, 0)

        flat_best = torch.argmax(masked_lens.view(-1))
        best_i = flat_best // num_bins
        best_j = flat_best % num_bins
        best_center_bin = 0.5 * (best_i + best_j).to(dtype)

        thresh_island = min_v + best_center_bin * (span / float(num_bins))
        thresh = torch.where(empty_between.any(), thresh_island, min_v + 0.5 * span)

        l_mask = (proj < thresh) & (cluster_mask > 0)
        r_mask = (proj >= thresh) & (cluster_mask > 0)

        mu_proj = (proj * cluster_mask).sum() / cnt
        bad_split = (l_mask.sum() < 1.0) | (r_mask.sum() < 1.0)
        thresh = torch.where(bad_split, mu_proj, thresh)

        l_mask = ((proj < thresh) & (cluster_mask > 0)).to(dtype)
        r_mask = ((proj >= thresh) & (cluster_mask > 0)).to(dtype)

        l_cnt = l_mask.sum()
        r_cnt = r_mask.sum()
        c_left = torch.where(l_cnt >= 1.0, (l_mask.unsqueeze(1) * X_sub).sum(dim=0) / l_cnt.clamp(min=1.0), mu)
        c_right = torch.where(r_cnt >= 1.0, (r_mask.unsqueeze(1) * X_sub).sum(dim=0) / r_cnt.clamp(min=1.0), mu)

        t_oh = (c_arange == target_cluster).unsqueeze(1)
        new_oh = (c_arange == m).unsqueeze(1)
        C = torch.where(t_oh, c_left.unsqueeze(0), C)
        C = torch.where(new_oh, c_right.unsqueeze(0), C)

    return C.to(orig_dtype)


@torch.no_grad()
def static_engine_v10_heads(
    X: torch.Tensor,
    k: int,
    sub_n: int = 4096,
    power_iters: int = 2,
    robust_cap_q: float = 0.95,
    x_sq: torch.Tensor = None,
) -> torch.Tensor:
    """
    Batched per-head Engine v10 (No-Island Batched Priority-SSE PDDP).
    X: (H, N, D) -> (H, k, D). Static shapes per round, torch.compile-safe.
    """
    H, n, d = X.shape
    device = X.device
    orig_dtype = X.dtype

    if n > sub_n:
        idx = torch.randint(0, n, (sub_n,), device=device)
        X_sub = X[:, idx, :]
        x_sub_sq = x_sq[:, idx].float() if x_sq is not None else None
    else:
        X_sub = X
        x_sub_sq = x_sq.float() if x_sq is not None else None

    X_sub = torch.nan_to_num(X_sub.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    S = X_sub.shape[1]
    dtype = torch.float32
    if x_sub_sq is None:
        x_sub_sq = (X_sub * X_sub).sum(dim=-1)

    C_buf = torch.zeros(H, k, d, device=device, dtype=dtype)
    C_buf[:, 0, :] = X_sub.mean(dim=1)

    m = 1
    while m < k:
        b = min(m, max(1, (2 * m + 2) // 3), k - m)
        C_curr = C_buf[:, :m, :]
        c_half = 0.5 * (C_curr * C_curr).sum(dim=-1)
        scores = torch.bmm(X_sub, C_curr.transpose(1, 2)) - c_half.unsqueeze(1)
        max_score, assign = torch.max(scores, dim=-1)

        pt_sse = torch.clamp(x_sub_sq - 2.0 * max_score, min=0.0)
        if robust_cap_q is not None:
            q_cap = torch.quantile(pt_sse, robust_cap_q, dim=1, keepdim=True)
            pt_sse = torch.clamp(pt_sse, max=q_cap)

        cluster_sse = torch.zeros(H, m, device=device, dtype=dtype).scatter_add_(1, assign, pt_sse)
        _, split_ids = torch.topk(cluster_sse, b, dim=1, largest=True)

        cluster_to_sel = torch.full((H, m), b, device=device, dtype=torch.long)
        b_arange = torch.arange(b, device=device).unsqueeze(0).expand(H, -1)
        cluster_to_sel.scatter_(1, split_ids, b_arange)
        sel_pos = torch.gather(cluster_to_sel, 1, assign)
        safe_pos = sel_pos.clamp(max=b - 1)
        in_sel = (sel_pos < b).to(dtype)

        O = F.one_hot(sel_pos, num_classes=b + 1).to(dtype)[:, :, :b]
        total_cnt = O.sum(dim=1)
        M_c = torch.bmm(O.transpose(1, 2), X_sub) / total_cnt.clamp(min=1.0).unsqueeze(-1)

        V = torch.randn(H, b, d, device=device, dtype=dtype)
        V = V / (V.norm(dim=-1, keepdim=True) + 1e-9)
        for _ in range(power_iters):
            xv_all = torch.bmm(X_sub, V.transpose(1, 2))
            dot_x_v = torch.gather(xv_all, 2, safe_pos.unsqueeze(-1)).squeeze(-1) * in_sel
            dot_mu_v = (M_c * V).sum(dim=-1)
            z = (dot_x_v - torch.gather(dot_mu_v, 1, safe_pos)) * in_sel
            Z_mat = O * z.unsqueeze(-1)
            V_next = torch.bmm(Z_mat.transpose(1, 2), X_sub) - Z_mat.sum(dim=1).unsqueeze(-1) * M_c
            V = V_next / (V_next.norm(dim=-1, keepdim=True) + 1e-9)

        proj = torch.gather(torch.bmm(X_sub, V.transpose(1, 2)), 2, safe_pos.unsqueeze(-1)).squeeze(-1)
        dot_mu_v = (M_c * V).sum(dim=-1)

        is_left = (proj < torch.gather(dot_mu_v, 1, safe_pos)) & (sel_pos < b)
        O_left = O * is_left.to(dtype).unsqueeze(-1)
        O_right = O - O_left
        cnt_left = O_left.sum(dim=1)
        cnt_right = O_right.sum(dim=1)

        C_left = torch.bmm(O_left.transpose(1, 2), X_sub) / cnt_left.clamp(min=1.0).unsqueeze(-1)
        C_right = torch.bmm(O_right.transpose(1, 2), X_sub) / cnt_right.clamp(min=1.0).unsqueeze(-1)

        valid = (cnt_left >= 1.0) & (cnt_right >= 1.0)
        C_sel_curr = torch.gather(C_curr, 1, split_ids.unsqueeze(-1).expand(-1, -1, d))
        C_left_safe = torch.where(valid.unsqueeze(-1), C_left, C_sel_curr)
        fallback_pts = X_sub[:, torch.arange(b, device=device) % S, :]
        C_right_safe = torch.where(valid.unsqueeze(-1), C_right, fallback_pts)

        C_buf.scatter_(1, split_ids.unsqueeze(-1).expand(-1, -1, d), C_left_safe)
        C_buf[:, m : m + b, :] = C_right_safe
        m += b

    return C_buf.to(orig_dtype)


@torch.no_grad()
def _damped_lloyd_step_heads(
    x_heads: torch.Tensor,
    C_heads: torch.Tensor,
    damp_scale: float = 2.0,
    x_sq: torch.Tensor = None,
    lam: torch.Tensor = None,
) -> torch.Tensor:
    H, N, D = x_heads.shape
    M = C_heads.shape[1]

    x_f32 = x_heads.float() if x_heads.dtype != torch.float32 else x_heads
    C_f32 = C_heads.float() if C_heads.dtype != torch.float32 else C_heads

    if x_sq is None:
        x_sq = (x_f32 * x_f32).sum(dim=-1)
    elif x_sq.dtype != torch.float32:
        x_sq = x_sq.float()
    c_half = 0.5 * (C_f32 * C_f32).sum(dim=-1)

    scores = torch.bmm(x_f32, C_f32.transpose(1, 2)) - c_half.unsqueeze(1)
    max_score, assign = torch.max(scores, dim=-1)

    dist_sq = torch.clamp(x_sq - 2.0 * max_score, min=0.0)
    if lam is None:
        step = max(1, N // 1024)
        if step > 1 and step % 2 == 0:
            step += 1
        sub = dist_sq[:, ::step]
        lam = torch.clamp(damp_scale * torch.median(sub, dim=1, keepdim=True).values, min=1e-4)

    bmm_dtype = torch.float32 if x_heads.dtype == torch.float16 else x_heads.dtype
    x_bmm = x_f32 if bmm_dtype == torch.float32 else x_heads
    w = (lam / (dist_sq + lam)).to(bmm_dtype)

    O_w = torch.zeros(H, N, M, device=x_heads.device, dtype=bmm_dtype)
    O_w.scatter_(2, assign.unsqueeze(-1), w.unsqueeze(-1))

    counts = O_w.float().sum(dim=1).unsqueeze(-1)
    new_C = torch.bmm(O_w.transpose(1, 2), x_bmm).float() / torch.clamp(counts, min=1e-6)
    return torch.where(counts < 1e-6, C_f32, new_C).to(C_heads.dtype)


class EmeansTopKAttention(nn.Module):
    """
    Semantic clustering + TopKHot block routing.
    Supports:
      cluster_algo = "qk_dot"    (per-head v10+damped Lloyd on K, Q @ block_C routing)
                   | "qk_mips"   (per-head joint [Q_tilde; K_tilde] MIPS co-clustering)
                   | "mag_only"  (chronological local + descending-||K|| global blocks)
                   | "engine_v5" (legacy single-space clustering + FlexAttention support)
    """
    def __init__(
        self,
        n_embd,
        n_head,
        k_blocks=4,
        block_size=8,
        n_clusters=16,
        router_dim=16,
        topk_mode="dense",
        cluster_algo="qk_dot",
        cluster_on="keys",
        cluster_norm=False,
        mag_sort=False,
        use_flex=False,
        dropout=0.0,
        bias=False,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.k_blocks = k_blocks
        self.block_size = block_size
        self.n_clusters = n_clusters
        self.router_dim = router_dim
        self.topk_mode = topk_mode
        self.cluster_algo = cluster_algo
        self.cluster_on = cluster_on
        self.cluster_norm = cluster_norm
        self.mag_sort = mag_sort
        self.use_flex = use_flex

        self.cluster_dim = n_embd if cluster_on == "keys" else router_dim

        if use_flex and not HAS_FLEX_ATTENTION:
            raise RuntimeError("use_flex=True requested but torch.nn.attention.flex_attention is unavailable.")

        self.q_attn = nn.Linear(n_embd, n_embd, bias=bias)
        self.k_attn = nn.Linear(n_embd, n_embd, bias=bias)
        self.v_attn = nn.Linear(n_embd, n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)

        if cluster_algo == "engine_v5":
            if cluster_on == "proj":
                self.cluster_proj = nn.Linear(n_embd, router_dim, bias=False)
            elif cluster_on == "keys":
                self.cluster_proj = None
            else:
                raise ValueError(f"Unknown cluster_on: {cluster_on}")
            self.router_q = nn.Linear(n_embd, router_dim, bias=False)
            self.router_k = nn.Linear(n_embd, router_dim, bias=False)
            self.register_buffer("C", torch.randn(n_clusters, self.cluster_dim) * 0.02)
        else:
            self.cluster_proj = None
            self.router_q = None
            self.router_k = None

        if cluster_algo in ("qk_dot", "qk_mips"):
            head_c_dim = self.head_dim + 1 if cluster_algo == "qk_mips" else self.head_dim
            self.register_buffer("C_heads", torch.randn(n_head, n_clusters, head_c_dim) * 0.02)
            self.register_buffer("k_max_buf", torch.ones(n_head, 1, 1, dtype=torch.float32))

    @staticmethod
    def _lloyd_step(x_flat: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        M = C.shape[0]
        c_half = 0.5 * (C * C).sum(dim=1)
        scores = torch.addmm(-c_half, x_flat, C.T)
        assign = torch.argmax(scores, dim=1)
        O = F.one_hot(assign, num_classes=M).to(torch.float32)
        counts = O.sum(dim=0, keepdim=True).T
        new_C = (O.T @ x_flat) / torch.clamp(counts, min=1.0)
        return torch.where(counts == 0, C, new_C)

    @torch.no_grad()
    def _update_clusters(self, x_flat: torch.Tensor):
        M = self.n_clusters
        x_flat = torch.nan_to_num(x_flat.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        if self.cluster_norm:
            x_flat = F.normalize(x_flat, dim=-1, eps=1e-8)

        C_v5 = static_engine_v5(x_flat, M, sub_n=4096, power_iters=2)
        C_next = self._lloyd_step(x_flat, C_v5)
        self.C.copy_(C_next.to(self.C.dtype))

    @torch.no_grad()
    def _update_qk_clusters(self, q: torch.Tensor, k: torch.Tensor):
        B, H, T, D = k.shape
        M = self.n_clusters

        if self.cluster_algo == "qk_dot":
            x_heads = torch.nan_to_num(
                k.detach().float().transpose(0, 1).reshape(H, B * T, D),
                nan=0.0, posinf=1e4, neginf=-1e4
            )
            x_sq = (x_heads * x_heads).sum(dim=-1)
            C_v10 = static_engine_v10_heads(x_heads, M, sub_n=4096, power_iters=2, x_sq=x_sq)
            C_next = _damped_lloyd_step_heads(x_heads, C_v10, damp_scale=2.0, x_sq=x_sq)
            self.C_heads.copy_(C_next.to(self.C_heads.dtype))
        else:
            k_f = torch.nan_to_num(k.detach().float(), nan=0.0, posinf=1e4, neginf=-1e4)
            q_f = torch.nan_to_num(q.detach().float(), nan=0.0, posinf=1e4, neginf=-1e4)

            new_kmax = k_f.norm(dim=-1).amax(dim=(0, 2)).clamp(min=1e-6)
            q_tilde = F.pad(F.normalize(q_f, dim=-1, eps=1e-8), (0, 1), value=0.0)

            k_scaled = k_f / new_kmax.view(1, H, 1, 1)
            k_sq = (k_scaled * k_scaled).sum(dim=-1, keepdim=True).clamp(max=1.0)
            k_tilde = torch.cat([k_scaled, torch.sqrt(1.0 - k_sq)], dim=-1)

            qt_heads = q_tilde.transpose(0, 1).reshape(H, B * T, D + 1)
            kt_heads = k_tilde.transpose(0, 1).reshape(H, B * T, D + 1)
            joint_heads = torch.cat([qt_heads, kt_heads], dim=1)
            x_sq = (joint_heads * joint_heads).sum(dim=-1)

            C_v10 = static_engine_v10_heads(joint_heads, M, sub_n=4096, power_iters=2, x_sq=x_sq)
            C_next = _damped_lloyd_step_heads(joint_heads, C_v10, damp_scale=2.0, x_sq=x_sq)
            self.C_heads.copy_(C_next.to(self.C_heads.dtype))
            self.k_max_buf.copy_(new_kmax.view(H, 1, 1))

    def _qk_cluster_forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, B: int, T: int, C: int):
        H, D = self.n_head, self.head_dim
        bs = self.block_size
        N_B = T // bs
        k_off = max(1, min(self.k_blocks - 1, N_B - 1))

        if self.cluster_algo == "qk_dot":
            C_buf = self.C_heads.to(k.dtype)
            c_half = 0.5 * (C_buf * C_buf).sum(dim=-1)
            k_scores = torch.einsum("bhtd,hmd->bhtm", k, C_buf) - c_half.view(1, H, 1, self.n_clusters)
            assign_k = torch.argmax(k_scores, dim=-1)
        else:
            k_max = self.k_max_buf.to(torch.float32)
            q_tilde = F.pad(F.normalize(q.float(), dim=-1, eps=1e-8), (0, 1), value=0.0)
            k_scaled = k.float() / k_max.view(1, H, 1, 1).clamp(min=1e-6)
            k_sq = (k_scaled * k_scaled).sum(dim=-1, keepdim=True).clamp(max=1.0)
            k_tilde = torch.cat([k_scaled, torch.sqrt(1.0 - k_sq)], dim=-1)

            C_buf = self.C_heads.float()
            c_half = 0.5 * (C_buf * C_buf).sum(dim=-1)
            k_scores = torch.einsum("bhtd,hmd->bhtm", k_tilde, C_buf) - c_half.view(1, H, 1, self.n_clusters)
            assign_k = torch.argmax(k_scores, dim=-1)

        if self.training:
            self._update_qk_clusters(q, k)

        if self.mag_sort:
            k_mag = k.float().norm(dim=-1)
            mag_rank = k_mag / (k_mag.amax(dim=-1, keepdim=True) + 1.0)
            sort_key = assign_k.float() * 2.0 + (1.0 - mag_rank)
            perm_k = torch.argsort(sort_key, dim=-1, stable=True)
        else:
            perm_k = torch.argsort(assign_k, dim=-1, stable=True)

        perm_kv = perm_k.unsqueeze(-1).expand(-1, -1, -1, D)
        k_s = torch.gather(k, 2, perm_kv)
        v_s = torch.gather(v, 2, perm_kv)
        assign_k_s = torch.gather(assign_k, 2, perm_k)

        k_blocks_all = k_s.view(B, H, N_B, bs, D)
        v_blocks_all = v_s.view(B, H, N_B, bs, D)
        orig_k_blocks = perm_k.view(B, H, N_B, bs)

        d_c = C_buf.shape[-1]
        C_exp = C_buf.unsqueeze(0).expand(B, -1, -1, -1)
        C_per_tok = torch.gather(C_exp, 2, assign_k_s.unsqueeze(-1).expand(-1, -1, -1, d_c))
        block_C = C_per_tok.view(B, H, N_B, bs, d_c).mean(dim=3)

        if self.cluster_algo == "qk_dot":
            router_logits = torch.matmul(q, block_C.transpose(-2, -1)) * (1.0 / math.sqrt(D))
        else:
            bc_half = 0.5 * (block_C * block_C).sum(dim=-1).unsqueeze(2)
            router_logits = (
                torch.matmul(q_tilde.to(q.dtype), block_C.transpose(-2, -1).to(q.dtype)) - bc_half.to(q.dtype)
            ) * 2.0

        q_block_id = torch.arange(T, device=q.device) // bs
        block_q_start = (q_block_id * bs).view(1, 1, T, 1)
        orig_k_min = orig_k_blocks.amin(dim=-1).unsqueeze(2)
        block_has_past = (orig_k_min < block_q_start)

        off_logits = router_logits.masked_fill(~block_has_past, float('-inf'))
        off_mask = TopKHot.apply(off_logits, k_off, self.topk_mode)
        _, off_idx = torch.topk(off_logits, k=k_off, dim=-1)
        off_valid = torch.gather(off_mask, -1, off_idx)

        flat_idx = off_idx.view(B, H, T * k_off)
        gather_kv = flat_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, bs, D)
        k_glob = torch.gather(k_blocks_all, 2, gather_kv).view(B, H, T, k_off * bs, D)
        v_glob = torch.gather(v_blocks_all, 2, gather_kv).view(B, H, T, k_off * bs, D)
        orig_k_glob = torch.gather(
            orig_k_blocks, 2, flat_idx.unsqueeze(-1).expand(-1, -1, -1, bs)
        ).view(B, H, T, k_off * bs)

        k_loc = k.view(B, H, N_B, bs, D)[:, :, q_block_id, :, :]
        v_loc = v.view(B, H, N_B, bs, D)[:, :, q_block_id, :, :]
        orig_loc = torch.arange(T, device=q.device).view(N_B, bs)[q_block_id, :].view(1, 1, T, bs)

        k_g = torch.cat([k_loc, k_glob], dim=3)
        v_g = torch.cat([v_loc, v_glob], dim=3)

        t_pos = torch.arange(T, device=q.device).view(1, 1, T, 1)
        local_causal = (t_pos >= orig_loc).expand(B, H, -1, -1)
        glob_valid_tok = off_valid.unsqueeze(-1).expand(-1, -1, -1, -1, bs).reshape(B, H, T, k_off * bs)
        glob_past = (orig_k_glob < block_q_start) & (glob_valid_tok > 0)
        full_mask = torch.cat([local_causal, glob_past], dim=-1)

        scores = torch.einsum("bhtd,bhtkd->bhtk", q, k_g) * (1.0 / math.sqrt(D))
        scores = scores.masked_fill(~full_mask, float('-inf'))
        probs = F.softmax(scores, dim=-1)

        gate = torch.cat([
            torch.ones(B, H, T, bs, device=q.device, dtype=q.dtype),
            glob_valid_tok.to(q.dtype),
        ], dim=-1)
        probs = probs * gate
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)

        out = torch.einsum("bhtk,bhtkd->bhtd", probs, v_g)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(out))

    def _qk_cluster_flex_forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, B: int, T: int, C: int):
        H, D = self.n_head, self.head_dim
        bs = self.block_size
        N_B = T // bs
        k_off = max(1, min(self.k_blocks - 1, N_B - 1))

        if self.cluster_algo == "qk_dot":
            C_buf = self.C_heads.to(k.dtype)
            c_half = 0.5 * (C_buf * C_buf).sum(dim=-1)
            k_scores = torch.einsum("bhtd,hmd->bhtm", k, C_buf) - c_half.view(1, H, 1, self.n_clusters)
            assign_k = torch.argmax(k_scores, dim=-1)
        else:
            k_max = self.k_max_buf.to(torch.float32)
            q_tilde = F.pad(F.normalize(q.float(), dim=-1, eps=1e-8), (0, 1), value=0.0)
            k_scaled = k.float() / k_max.view(1, H, 1, 1).clamp(min=1e-6)
            k_sq = (k_scaled * k_scaled).sum(dim=-1, keepdim=True).clamp(max=1.0)
            k_tilde = torch.cat([k_scaled, torch.sqrt(1.0 - k_sq)], dim=-1)

            C_buf = self.C_heads.float()
            c_half = 0.5 * (C_buf * C_buf).sum(dim=-1)
            k_scores = torch.einsum("bhtd,hmd->bhtm", k_tilde, C_buf) - c_half.view(1, H, 1, self.n_clusters)
            assign_k = torch.argmax(k_scores, dim=-1)

        if self.training:
            self._update_qk_clusters(q, k)

        if self.mag_sort:
            k_mag = k.float().norm(dim=-1)
            mag_rank = k_mag / (k_mag.amax(dim=-1, keepdim=True) + 1.0)
            sort_key = assign_k.float() * 2.0 + (1.0 - mag_rank)
            perm_k = torch.argsort(sort_key, dim=-1, stable=True)
        else:
            perm_k = torch.argsort(assign_k, dim=-1, stable=True)

        perm_kv = perm_k.unsqueeze(-1).expand(-1, -1, -1, D)
        k_s = torch.gather(k, 2, perm_kv)
        v_s = torch.gather(v, 2, perm_kv)
        assign_k_s = torch.gather(assign_k, 2, perm_k)

        orig_k_blocks = perm_k.view(B, H, N_B, bs)
        d_c = C_buf.shape[-1]
        C_exp = C_buf.unsqueeze(0).expand(B, -1, -1, -1)
        C_per_tok = torch.gather(C_exp, 2, assign_k_s.unsqueeze(-1).expand(-1, -1, -1, d_c))
        block_C = C_per_tok.view(B, H, N_B, bs, d_c).mean(dim=3)

        if self.cluster_algo == "qk_dot":
            q_b = q.view(B, H, N_B, bs, D).mean(dim=3)
            router_logits = torch.matmul(q_b, block_C.transpose(-2, -1)) * (1.0 / math.sqrt(D))
        else:
            q_tilde_b = q_tilde.view(B, H, N_B, bs, D + 1).mean(dim=3).to(q.dtype)
            bc_half = 0.5 * (block_C * block_C).sum(dim=-1).unsqueeze(2)
            router_logits = (
                torch.matmul(q_tilde_b, block_C.transpose(-2, -1).to(q.dtype)) - bc_half.to(q.dtype)
            ) * 2.0

        block_q_start = (torch.arange(N_B, device=q.device) * bs).view(1, 1, N_B, 1)
        orig_k_min = orig_k_blocks.amin(dim=-1).unsqueeze(2)
        block_has_past = (orig_k_min < block_q_start)

        off_logits = router_logits.masked_fill(~block_has_past, float('-inf'))
        off_mask = TopKHot.apply(off_logits, k_off, self.topk_mode)
        _, off_idx = torch.topk(off_logits, k=k_off, dim=-1)

        k_cat = torch.cat([k, k_s], dim=2).contiguous()
        v_cat = torch.cat([v, v_s], dim=2).contiguous()

        diag_idx = torch.arange(N_B, device=q.device).view(1, 1, N_B, 1).expand(B, H, N_B, 1)
        kv_indices = torch.cat([diag_idx, N_B + off_idx], dim=-1)
        if kv_indices.size(-1) < 2 * N_B:
            kv_indices = F.pad(kv_indices, (0, 2 * N_B - kv_indices.size(-1)), value=0)
        kv_indices = kv_indices.to(torch.int32).contiguous()
        kv_num_blocks = (1.0 + off_mask.sum(dim=-1)).to(torch.int32).contiguous()

        perm_k_c = perm_k.contiguous()

        def qk_cluster_mask(b, h, q_idx, kv_idx):
            safe_g = torch.clamp(kv_idx - T, min=0)
            orig_k = perm_k_c[b, h, safe_g]
            q_block_start = (q_idx // bs) * bs
            return torch.where(kv_idx < T, q_idx >= kv_idx, orig_k < q_block_start)

        bm = BlockMask.from_kv_blocks(
            kv_num_blocks=kv_num_blocks,
            kv_indices=kv_indices,
            full_kv_num_blocks=None,
            full_kv_indices=None,
            BLOCK_SIZE=(bs, bs),
            mask_mod=qk_cluster_mask,
            seq_lengths=(T, 2 * T),
        )

        bwd_sub = bs if bs <= 32 else max(32, bs // 2)
        k_opts = {
            "BACKEND": "TRITON",
            "BLOCK_M": bs,
            "BLOCK_N": bs,
            "BLOCK_M1": bwd_sub,
            "BLOCK_N1": bs,
            "BLOCK_M2": bs,
            "BLOCK_N2": bwd_sub,
        }
        out_s = flex_attention(q.contiguous(), k_cat, v_cat, block_mask=bm, kernel_options=k_opts)
        out_b = out_s.view(B, H, N_B, bs, D)

        if self.topk_mode in ("dense", "2k_sparse") and self.training:
            safe_logits = off_logits.masked_fill(~block_has_past, -1e4)
            router_probs = F.softmax(safe_logits, dim=-1) * block_has_past.to(q.dtype)
            masked_probs = router_probs * off_mask
            v_b = v_s.view(B, H, N_B, bs, D).mean(dim=3)
            proxy_ste = torch.einsum("bhqk,bhkd->bhqd", masked_probs, v_b).unsqueeze(3)
            out_b = out_b + (proxy_ste - proxy_ste.detach())
        elif self.topk_mode == "k_sparse" and self.training:
            proxy_sum = off_mask.sum(dim=-1, keepdim=True).unsqueeze(3)
            out_b = out_b * (1.0 + proxy_sum - proxy_sum.detach())

        out = out_b.view(B, H, T, D).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(out))

    def _mag_only_forward(self, q, k, v, r_x, B, T, C):
        """
        Clustering-free control: Q stays chronological (local block i),
        and attends to the top-k_off causally-past blocks of K,V sorted by descending magnitude.
        """
        H, D = self.n_head, self.head_dim
        bs = self.block_size
        N_B = T // bs
        k_off = max(1, min(self.k_blocks - 1, N_B - 1))

        k_mag = r_x.norm(dim=-1)
        perm_k = torch.argsort(k_mag, dim=-1, descending=True, stable=True)

        perm_kv = perm_k[:, None, :, None].expand(-1, H, -1, D)
        k_s = torch.gather(k, 2, perm_kv)
        v_s = torch.gather(v, 2, perm_kv)

        k_blocks_all = k_s.view(B, H, N_B, bs, D)
        v_blocks_all = v_s.view(B, H, N_B, bs, D)
        orig_k_blocks = perm_k.view(B, N_B, bs)

        orig_k_min = orig_k_blocks.amin(dim=-1).unsqueeze(1)
        block_q_start = (torch.arange(N_B, device=q.device) * bs).view(1, N_B, 1)
        block_has_past = (orig_k_min < block_q_start)

        block_priority = (-torch.arange(N_B, device=q.device, dtype=torch.float32)).view(1, 1, N_B).expand(B, N_B, N_B)
        block_scores = block_priority.masked_fill(~block_has_past, float('-inf'))

        top_vals, mag_idx = torch.topk(block_scores, k=k_off, dim=-1)
        mag_block_valid = (top_vals > float('-inf'))

        flat_idx = mag_idx.view(B, N_B * k_off)
        gather_kv = flat_idx[:, None, :, None, None].expand(-1, H, -1, bs, D)
        k_mag_g = torch.gather(k_blocks_all, 2, gather_kv).view(B, H, N_B, k_off * bs, D)
        v_mag_g = torch.gather(v_blocks_all, 2, gather_kv).view(B, H, N_B, k_off * bs, D)
        orig_k_mag_g = torch.gather(
            orig_k_blocks, 1, flat_idx[:, :, None].expand(-1, -1, bs)
        ).view(B, N_B, k_off * bs)

        k_loc = k.view(B, H, N_B, bs, D)
        v_loc = v.view(B, H, N_B, bs, D)
        orig_loc = torch.arange(T, device=q.device).view(1, N_B, bs).expand(B, -1, -1)

        k_g = torch.cat([k_loc, k_mag_g], dim=3)
        v_g = torch.cat([v_loc, v_mag_g], dim=3)

        local_causal = (orig_loc.unsqueeze(-1) >= orig_loc.unsqueeze(-2))
        mag_past = (orig_k_mag_g < block_q_start).unsqueeze(-2) & (
            mag_block_valid.unsqueeze(-1).expand(-1, -1, -1, bs).reshape(B, N_B, 1, k_off * bs)
        )
        full_mask = torch.cat([local_causal, mag_past.expand(-1, -1, bs, -1)], dim=-1).unsqueeze(1)

        q_chron = q.view(B, H, N_B, bs, D)
        scores = torch.matmul(q_chron, k_g.transpose(-2, -1)) * (1.0 / math.sqrt(D))
        scores = scores.masked_fill(~full_mask, float('-inf'))
        probs = F.softmax(scores, dim=-1)

        out_b = torch.matmul(probs, v_g)
        out = out_b.view(B, H, T, D).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(out))

    def forward(self, x):
        B, T, C = x.size()
        H, D = self.n_head, self.head_dim
        bs = self.block_size
        assert T % bs == 0, f"Sequence length {T} must be divisible by emeans_block_size {bs}"
        N_B = T // bs
        R = self.router_dim

        q_raw = self.q_attn(x)
        k_raw = self.k_attn(x)
        v_raw = self.v_attn(x)
        q = q_raw.view(B, T, H, D).transpose(1, 2)
        k = k_raw.view(B, T, H, D).transpose(1, 2)
        v = v_raw.view(B, T, H, D).transpose(1, 2)

        if self.cluster_algo in ("qk_dot", "qk_mips"):
            if self.use_flex:
                return self._qk_cluster_flex_forward(q, k, v, B, T, C)
            return self._qk_cluster_forward(q, k, v, B, T, C)

        if self.cluster_algo == "mag_only":
            return self._mag_only_forward(q, k, v, k_raw, B, T, C)

        # Legacy engine_v5 (also supports FlexAttention)
        r_x = k_raw if self.cluster_on == "keys" else self.cluster_proj(x)
        r_q = self.router_q(x)
        r_k = self.router_k(x)

        r_x_assign = F.normalize(r_x.float(), dim=-1, eps=1e-8).to(r_x.dtype) if self.cluster_norm else r_x
        C_buf = self.C.to(r_x.dtype)
        c_half = 0.5 * (C_buf * C_buf).sum(dim=1)
        assign_scores = F.linear(r_x_assign, C_buf, -c_half)
        assign = torch.argmax(assign_scores, dim=-1)

        if self.training:
            self._update_clusters(r_x.detach().reshape(B * T, self.cluster_dim))

        if self.mag_sort:
            k_mag = r_x.float().norm(dim=-1)
            mag_rank = k_mag / (k_mag.amax(dim=-1, keepdim=True) + 1.0)
            sort_key = assign.float() * 2.0 + (1.0 - mag_rank)
            perm = torch.argsort(sort_key, dim=-1, stable=True)
        else:
            perm = torch.argsort(assign, dim=-1, stable=True)
        inv_perm = torch.argsort(perm, dim=-1)

        perm_qkv = perm[:, None, :, None].expand(-1, H, -1, D)
        q_s = torch.gather(q, 2, perm_qkv)
        k_s = torch.gather(k, 2, perm_qkv)
        v_s = torch.gather(v, 2, perm_qkv)

        perm_r = perm[:, :, None].expand(-1, -1, R)
        rq_s = torch.gather(r_q, 1, perm_r)
        rk_s = torch.gather(r_k, 1, perm_r)

        rq_b = rq_s.view(B, N_B, bs, R).mean(dim=2)
        rk_b = rk_s.view(B, N_B, bs, R).mean(dim=2)
        router_logits = (rq_b @ rk_b.transpose(-2, -1)) * (1.0 / math.sqrt(R))

        orig_q_max = perm.view(B, N_B, bs).amax(dim=-1, keepdim=True)
        orig_k_min = perm.view(B, N_B, bs).amin(dim=-1).unsqueeze(1)
        block_has_causal = (orig_q_max >= orig_k_min)

        eye = torch.eye(N_B, device=x.device, dtype=torch.bool).unsqueeze(0)
        off_diag_valid = block_has_causal & (~eye)
        off_diag_logits = router_logits.masked_fill(~off_diag_valid, float('-inf'))

        k_off = max(1, min(self.k_blocks - 1, N_B - 1))
        off_diag_mask = TopKHot.apply(off_diag_logits, k_off, self.topk_mode)

        if self.use_flex:
            diag_idx = torch.arange(N_B, device=x.device).view(1, N_B, 1).expand(B, N_B, 1)
            _, off_idx = torch.topk(off_diag_logits, k=k_off, dim=-1)
            kv_indices = torch.cat([diag_idx, off_idx], dim=-1)
            if kv_indices.size(-1) < N_B:
                kv_indices = F.pad(kv_indices, (0, N_B - kv_indices.size(-1)), value=0)
            kv_indices = kv_indices.to(torch.int32).unsqueeze(1).contiguous()
            kv_num_blocks = (1.0 + off_diag_mask.sum(dim=-1)).to(torch.int32).unsqueeze(1).contiguous()

            def causal_perm_mask(b, h, q_idx, kv_idx):
                return perm[b, q_idx] >= perm[b, kv_idx]

            bm = BlockMask.from_kv_blocks(
                kv_num_blocks=kv_num_blocks,
                kv_indices=kv_indices,
                full_kv_num_blocks=None,
                full_kv_indices=None,
                BLOCK_SIZE=(bs, bs),
                mask_mod=causal_perm_mask,
                seq_lengths=(T, T),
            )
            k_opts = {
                "BLOCK_M": bs,
                "BLOCK_N": bs,
                "BLOCK_M1": max(16, bs // 2),
                "BLOCK_N1": bs,
                "BLOCK_M2": bs,
                "BLOCK_N2": max(16, bs // 2),
            }
            out_s = flex_attention(q_s, k_s, v_s, block_mask=bm, kernel_options=k_opts)
            out_b = out_s.view(B, H, N_B, bs, D)
        else:
            diag_idx = torch.arange(N_B, device=x.device).view(1, N_B, 1).expand(B, N_B, 1)
            _, off_idx = torch.topk(off_diag_logits, k=k_off, dim=-1)
            off_valid = torch.gather(off_diag_mask, -1, off_idx)

            sel_block_idx = torch.cat([diag_idx, off_idx], dim=-1)
            sel_block_valid = torch.cat([torch.ones_like(diag_idx, dtype=x.dtype), off_valid], dim=-1)
            k_eff = sel_block_idx.size(-1)

            k_blocks_all = k_s.view(B, H, N_B, bs, D)
            v_blocks_all = v_s.view(B, H, N_B, bs, D)
            orig_k_blocks = perm.view(B, N_B, bs)

            flat_idx = sel_block_idx.view(B, N_B * k_eff)
            gather_kv_idx = flat_idx[:, None, :, None, None].expand(-1, H, -1, bs, D)
            k_g = torch.gather(k_blocks_all, 2, gather_kv_idx).view(B, H, N_B, k_eff * bs, D)
            v_g = torch.gather(v_blocks_all, 2, gather_kv_idx).view(B, H, N_B, k_eff * bs, D)

            orig_k_g = torch.gather(
                orig_k_blocks, 1, flat_idx[:, :, None].expand(-1, -1, bs)
            ).view(B, N_B, k_eff * bs)

            q_b = q_s.view(B, H, N_B, bs, D)
            orig_q_b = perm.view(B, N_B, bs)

            scores = torch.matmul(q_b, k_g.transpose(-2, -1)) * (1.0 / math.sqrt(D))
            token_causal = (orig_q_b.unsqueeze(-1) >= orig_k_g.unsqueeze(-2))
            valid_k_tokens = (sel_block_valid.unsqueeze(-1).expand(-1, -1, -1, bs).reshape(B, N_B, 1, k_eff * bs) > 0)
            full_mask = (token_causal & valid_k_tokens).unsqueeze(1)

            scores = scores.masked_fill(~full_mask, float('-inf'))
            probs = F.softmax(scores, dim=-1)

            if self.topk_mode == "k_sparse":
                valid_scale = sel_block_valid.unsqueeze(-1).expand(-1, -1, -1, bs).reshape(B, 1, N_B, 1, k_eff * bs)
                probs = probs * valid_scale

            out_b = torch.matmul(probs, v_g)

        if self.topk_mode in ("dense", "2k_sparse") and self.training:
            safe_logits = off_diag_logits.masked_fill(~off_diag_valid, -1e4)
            router_probs = F.softmax(safe_logits, dim=-1) * off_diag_valid.to(x.dtype)
            masked_probs = router_probs * off_diag_mask
            v_b = v_s.view(B, H, N_B, bs, D).mean(dim=3)
            proxy_ste = torch.einsum("bqk,bhkd->bhqd", masked_probs, v_b).unsqueeze(3)
            out_b = out_b + (proxy_ste - proxy_ste.detach())
        elif self.use_flex and self.topk_mode == "k_sparse" and self.training:
            proxy_sum = off_diag_mask.sum(dim=-1, keepdim=True)[:, None, :, :, None]
            out_b = out_b * (1.0 + proxy_sum - proxy_sum.detach())

        out_s = out_b.view(B, H, T, D)
        out = torch.gather(out_s, 2, inv_perm[:, None, :, None].expand(-1, H, -1, D))
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(out))


class EmeansTopKBlock(nn.Module):
    def __init__(self, cfg: HNetConfig, use_flex=False):
        super().__init__()
        self.ln_1 = modules.LayerNorm(cfg.dim_l3, bias=False)
        self.attn = EmeansTopKAttention(
            n_embd=cfg.dim_l3,
            n_head=cfg.n_head,
            k_blocks=cfg.k_sparse,
            block_size=cfg.emeans_block_size,
            n_clusters=cfg.emeans_clusters,
            router_dim=cfg.router_dim,
            topk_mode=cfg.topk_mode,
            cluster_algo=cfg.cluster_algo,
            cluster_on=cfg.cluster_on,
            cluster_norm=cfg.cluster_norm,
            mag_sort=cfg.mag_sort,
            use_flex=use_flex,
            dropout=cfg.dropout,
            bias=False,
        )
        self.ln_2 = modules.LayerNorm(cfg.dim_l3, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.dim_l3, 4 * cfg.dim_l3, bias=False),
            nn.GELU(),
            nn.Linear(4 * cfg.dim_l3, cfg.dim_l3, bias=False),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# =====================================================================
# 4. Shared HNet Harness (Config-Driven)
# =====================================================================

class HNet(nn.Module):
    def __init__(self, vocab_size, cfg: HNetConfig):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.use_cce = cfg.cce and (vocab_size > 2048)

        self.tok_emb = nn.Embedding(vocab_size, cfg.dim_l1)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.dim_l1)
        self.wnll = utils.wnormlearnloss(cfg.dim_l1)
        self.h_ln = modules.LayerNorm(cfg.dim_l1, bias=False)

        if cfg.patcher_type == "voronoi_conv_gpt":
            self.sLayers = nn.ModuleList([
                VoronoiConvGPTPatcher(cfg.dim_l1, cfg.dim_l2, cfg.block_size, cfg.k_gather_l1, cfg, pin_tail=True, novelty_window=16),
                VoronoiSegmentGPTPatcher(cfg.dim_l2, cfg.dim_l3, cfg.k_gather_l1, cfg.k_gather_l2, cfg, pin_tail=True, novelty_window=16),
            ])
        else:
            patcher_cls = {
                "voronoi": VoronoiPatcher,
                "voronoi_segment": VoronoiSegmentPatcher,
            }[cfg.patcher_type]
            self.sLayers = nn.ModuleList([
                patcher_cls(cfg.dim_l1, cfg.dim_l2, cfg.block_size, cfg.k_gather_l1, cfg),
                patcher_cls(cfg.dim_l2, cfg.dim_l3, cfg.k_gather_l1, cfg.k_gather_l2, cfg),
            ])

        if cfg.inner_type == "kattention":
            self.inner = nn.ModuleList([
                KattentionBlock(cfg.dim_l3, cfg.n_head, k_sparse=cfg.k_sparse, topk_mode=cfg.topk_mode, dropout=cfg.dropout, bias=False)
                for _ in range(cfg.n_inner)
            ])
        elif cfg.inner_type == "emeans_topk":
            self.inner = nn.ModuleList([
                EmeansTopKBlock(cfg, use_flex=False) for _ in range(cfg.n_inner)
            ])
        elif cfg.inner_type == "emeans_flex":
            self.inner = nn.ModuleList([
                EmeansTopKBlock(cfg, use_flex=True) for _ in range(cfg.n_inner)
            ])
        else:
            self.inner = nn.ModuleList([
                modules.Block(cfg.dim_l3, cfg.n_head, 0.0, False) for _ in range(cfg.n_inner)
            ])

        if cfg.gpt_init:
            self.apply(self._init_gpt_weights)
            res_scale = 0.02 / math.sqrt(2.0 * (cfg.n_inner + 4))
            for pname, p in self.named_parameters():
                if pname.endswith(("mlp.2.weight", "conv.proj.weight", "c_proj.weight")):
                    torch.nn.init.normal_(p, mean=0.0, std=res_scale)
            for layer in self.sLayers:
                if hasattr(layer, "up_proj") and isinstance(layer.up_proj, nn.Linear):
                    torch.nn.init.normal_(layer.up_proj.weight, mean=0.0, std=1.0 / math.sqrt(layer.outer_dim))
                if hasattr(layer, "down_proj") and isinstance(layer.down_proj, nn.Linear):
                    torch.nn.init.normal_(layer.down_proj.weight, mean=0.0, std=1.0 / math.sqrt(2.0 * layer.inner_dim))
        elif cfg.winit:
            self.apply(self._init_weights)

        if cfg.emb_wn:
            torch.nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
            self.tok_emb = wn(self.tok_emb, dim=1)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_gpt_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, decaying=None, conf=None):
        b, t = idx.shape
        positions = torch.arange(0, t, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = utils.rms_norm(self.tok_emb(idx) + self.pos_emb(positions))

        residuals = []
        for layer in self.sLayers:
            x, res = layer.abstract_up(x)
            residuals.append(res)

        for block in self.inner:
            x = block(x)

        aux = 0.0
        for i in reversed(range(len(self.sLayers))):
            x, auxl = self.sLayers[i].abstract_down(x, residuals[i], decaying=decaying)
            aux = aux + auxl

        wnl = 0.0
        if self.training and self.cfg.wnll:
            wnl = self.wnll(x)
            aux = aux + wnl.mean()
        x = self.h_ln(x)

        if targets is None:
            return F.linear(x[:, -self.cfg.skiptok:, :], self.tok_emb.weight)

        T_out = x.shape[1]
        targets_aligned = targets[:, :T_out]
        red = 'none' if self.cfg.wnll else 'mean'

        if self.use_cce:
            ce_loss = linear_cross_entropy(
                x.to(torch.bfloat16),
                self.tok_emb.weight.to(torch.bfloat16),
                targets_aligned,
                impl='torch_compile',
                reduction=red
            )
            gloss = linear_cross_entropy(
                x.to(torch.bfloat16)[:, -1, :],
                self.tok_emb.weight.to(torch.bfloat16),
                targets_aligned[:, -1],
                impl='torch_compile'
            ).detach()
        else:
            logits = F.linear(x, self.tok_emb.weight)
            ce_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets_aligned.reshape(-1),
                reduction=red
            )
            if self.cfg.wnll:
                ce_loss = ce_loss.view(b, T_out)
            gloss = F.cross_entropy(
                logits[:, -1, :].reshape(-1, logits.shape[-1]),
                targets_aligned[:, -1].reshape(-1)
            ).detach()

        pure_ce = ce_loss.mean().detach()
        loss = ce_loss
        if self.training:
            if self.cfg.wnll:
                chaos = ((1.0 - wnl.detach()) / 2.0).mean() * 0.99
                loss = loss / (1.0 + wnl.detach().reshape(loss.shape) * chaos)
            loss = loss + aux

        if self.cfg.wnll:
            loss = loss.mean()
        return loss, pure_ce, gloss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        bs = self.cfg.block_size
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= bs else idx[:, -bs:]
            logits = self(idx_cond)[:, -self.cfg.skiptok:, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, :, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            B, T, C = probs.shape
            idx_next = torch.multinomial(probs.view(B * T, C), num_samples=1).view(B, T)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# =====================================================================
# 5. A/B Benchmark Runner
# =====================================================================

def module_grad_norm(mod):
    norms = [p.grad.detach().float().norm() for p in mod.parameters() if p.grad is not None]
    return torch.linalg.vector_norm(torch.stack(norms)).item() if norms else 0.0

def sanity_check_model(name, raw_model, bx, by, dec0):
    t0 = time.perf_counter()
    b_sub = min(4, bx.shape[0])
    tx, ty = bx[:b_sub], by[:b_sub]

    raw_model.train()
    raw_model.zero_grad(set_to_none=True)
    loss, pure_ce, gloss = raw_model(tx, ty, decaying=dec0)
    loss.backward()
    raw_model.zero_grad(set_to_none=True)

    raw_model.eval()
    with torch.no_grad():
        _ = raw_model(tx, ty, decaying=dec0)
        _ = raw_model.generate(tx[:1], max_new_tokens=2, temperature=1.0, top_k=5)
    raw_model.train()

    if device == 'cuda':
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1000.0
    print(f"  [SANITY OK] {name:<18} (ce={pure_ce.item():.3f}, gen={gloss.item():.3f}) in {dt:.1f} ms")

def evaluate_model(raw_model, decaying):
    raw_model.eval()
    tot_loss, tot_ce, tot_gen = 0.0, 0.0, 0.0
    with torch.no_grad():
        for _ in range(eval_iters):
            vx, vy = get_batch('val')
            loss, pure_ce, gloss = raw_model(vx, vy, decaying=decaying)
            tot_loss += loss.item()
            tot_ce += pure_ce.item()
            tot_gen += gloss.item()
    raw_model.train()
    return tot_loss / eval_iters, tot_ce / eval_iters, tot_gen / eval_iters

def run_benchmarks():
    torch.set_float32_matmul_precision('medium')
    active = {k: cfg for k, (enabled, cfg) in EXPERIMENTS.items() if enabled}
    print(f"Device: {device} | Vocab: {vocab_size} (Byte) | Active Experiments: {list(active.keys())}")

    raw_models, models, optimizers = {}, {}, {}
    bx_sanity, by_sanity = get_batch('train')
    dec0 = torch.tensor(1.0, device=device, dtype=torch.float32)

    print("\n--- Pre-Compile Eager Sanity Checks ---")
    for name, cfg in active.items():
        torch.manual_seed(1337)
        m = HNet(vocab_size=vocab_size, cfg=cfg).to(device)
        dims_str = f"{cfg.dim_l1}->{cfg.dim_l2}->{cfg.dim_l3}"
        print(f"  {name:<18} params: {sum(p.numel() for p in m.parameters()):,} | dims: {dims_str}")
        sanity_check_model(name, m, bx_sanity, by_sanity, dec0)
        raw_models[name] = m
        optimizers[name] = torch.optim.AdamW(m.parameters(), lr=learning_rate)

    print("\n--- Compiling & Warming Up ---")
    for name, m in raw_models.items():
        t_c = time.perf_counter()
        compiled_m = torch.compile(m)
        models[name] = compiled_m
        for _ in range(2):
            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()
            optimizers[name].zero_grad(set_to_none=True)
            loss, _, _ = compiled_m(bx_sanity, by_sanity, decaying=dec0)
            loss.backward()
            optimizers[name].zero_grad(set_to_none=True)
        if device == 'cuda':
            torch.cuda.synchronize()
        print(f"  [COMPILED]  {name:<18} in {(time.perf_counter() - t_c):.2f} s")

    col_widths = {n: max(len(n), 16) for n in models.keys()}

    print("=" * 120)
    print(f"{'Step':<6} | " + " | ".join([f"{n:<{col_widths[n]}}" for n in models.keys()]) + " | ms/step")
    print(f"{'':<6} | " + " | ".join([f"{' val_ce     gen':<{col_widths[n]}}" for n in models.keys()]) + " |")
    print("-" * 120)

    timing_accum = {n: 0.0 for n in models.keys()}
    interval_timing = {n: 0.0 for n in models.keys()}

    for it in range(max_iters):
        lr = get_lr(it)
        decaying = torch.clamp(torch.tensor(1.0 - (it / decay_iters), device=device, dtype=torch.float32), 0.0, 1.0)
        bx, by = get_batch('train')

        step = it + 1
        is_eval = (step % eval_interval == 0 or step == max_iters)
        grad_info = {}

        for name, model in models.items():
            opt = optimizers[name]
            for pg in opt.param_groups:
                pg['lr'] = lr

            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            opt.zero_grad(set_to_none=True)
            loss, _, _ = model(bx, by, decaying=decaying)
            loss.backward()

            if is_eval:
                rm = raw_models[name]
                grad_info[name] = (
                    module_grad_norm(rm.sLayers[0]),
                    module_grad_norm(rm.sLayers[1]),
                    module_grad_norm(rm.inner),
                )

            opt.step()

            if device == 'cuda':
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000.0
            timing_accum[name] += dt
            interval_timing[name] += dt

        if is_eval:
            eval_cells = []
            ms_cells = []
            for name, rm in raw_models.items():
                _, v_ce, v_gen = evaluate_model(rm, decaying)
                cell = f"{v_ce:7.4f}  {v_gen:7.4f}"
                eval_cells.append(f"{cell:<{col_widths[name]}}")
                ms_cells.append(f"{interval_timing[name] / eval_interval:5.1f}")
                interval_timing[name] = 0.0
            glog = ""
            if g_log:
                g_cells = [f"{g0:.2f}/{g1:.2f}/{gi:.2f}" for (g0, g1, gi) in grad_info.values()]
                glog = " | g(s0/s1/in): " + " ; ".join(g_cells)
            print(f"{step:<6d} | " + " | ".join(eval_cells) + " | " + " / ".join(ms_cells) + glog)

    print("=" * 120)
    for name, t_tot in timing_accum.items():
        print(f"Total Step Time - {name:<18}: {t_tot:9.1f} ms ({t_tot / max_iters:5.2f} ms/step)")

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