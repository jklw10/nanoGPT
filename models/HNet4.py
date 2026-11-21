"""
Benchmark: Kattention vs. Causal-Windowed & Amortized Engine v11 Variants
Trunk matches HNet3.py:
  - sLayers[0]: VoronoiConvGPTPatcher (fast conv on 256 raw bytes)
  - sLayers[1]: VoronoiSegmentGPTPatcher (attention on 128 compressed chunks)
  - gpt_init=False (Kaiming Uniform on linears; std=0.02 on embeddings only)

Active Experiments:
  - kattn_baseline: Exact Kattention (scale_t * qm + token Top-4 over T x T)
  - cw_l2_m16:      Causal-Windowed v11 + Stage-2 Kattention (l2_norm, W=8, bs=4, M=16)
  - cw_l2_m4:       Causal-Windowed v11 + Stage-2 Kattention (l2_norm, W=8, bs=4, M=4)
"""

from dataclasses import dataclass, replace
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.attention.flex_attention import flex_attention, create_block_mask

import parts.modules as modules
import parts.utils as utils
from cut_cross_entropy import linear_cross_entropy

torch._inductor.config.disable_cpp_codegen = True
torch._dynamo.config.force_parameter_static_shapes = False
torch._dynamo.config.cache_size_limit = 128
torch._dynamo.config.accumulated_cache_size_limit = 1024

# =====================================================================
# 1. Config & Hyperparameters
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
min_lr         = 1e-4

device         = 'cuda' if torch.cuda.is_available() else 'cpu'
decode, vocab_size = utils.get_meta(dataset, always_byte=True)
get_batch = utils.get_get_batch(block_size, batch_size, dataset, device, always_byte=True)

@dataclass(frozen=True)
class HNetConfig:
    dim_l1: int = 128
    dim_l2: int = 128
    dim_l3: int = 128
    block_size: int = block_size
    k_gather_l1: int = block_size // 2   # 128
    k_gather_l2: int = block_size // 4   # 64
    n_head: int = 4
    n_inner: int = 4
    dropout: float = 0.0
    skiptok: int = 1
    gpt_init: bool = False               # False = HNet3 Kaiming init on linears

    # Inner Attention Architecture
    inner_type: str = "cw_v11"           # "kattention" | "cw_v11" | "v11"
    variant: str = "l2_norm"             # "spherical_mips" | "l2_norm"
    use_flex: bool = False
    topk_mode: str = "dense"
    k_blocks: int = 4                    # Stage-1 coarse blocks
    k_sparse_tok: int = 4                # Stage-2 / Kattention token-level top-k
    emeans_clusters: int = 16
    emeans_block_size: int = 4
    win_size: int = 8                    # Causal window size for cw_v11
    update_every: int = 8                # Lloyd update cadence for cw_v11
    mag_sort: bool = True

    # Fixes & Optimizations
    use_qm_scale: bool = False           # Kattention's log(T) * qm query scaling
    two_stage_tok: bool = False          # Stage-2 token-level TopKHot on gathered blocks (for global v11)
    robust_cap_q: float | None = None    # None = preserve top 5% high-norm anchor/sink keys
    v11_refresh_interval: int = 50       # Full v11 bisecting split every N steps (amortized)
    ema_alpha: float = 0.2               # Temporal EMA smoothing for global v11 codebook

    cce: bool = True
    wnll: bool = True

BASE_CFG = HNetConfig()

EXPERIMENTS = {
    "kattn_baseline":   (True,  replace(BASE_CFG, inner_type="kattention", k_sparse_tok=4)),
    "cw_l2_m16":        (True,  replace(BASE_CFG, inner_type="cw_v11", variant="l2_norm", emeans_clusters=16, win_size=8, update_every=8)),
    "cw_l2_m4":         (True,  replace(BASE_CFG, inner_type="cw_v11", variant="l2_norm", emeans_clusters=4,  win_size=8, update_every=8)),

    "v11_sph_mag qm":   (False, replace(BASE_CFG, inner_type="v11", variant="spherical_mips", use_qm_scale=True,  two_stage_tok=False)),
    "v11_l2 qm":        (False, replace(BASE_CFG, inner_type="v11", variant="l2_norm",        use_qm_scale=True,  two_stage_tok=False)),
    "v11_sph_mag":      (False, replace(BASE_CFG, inner_type="v11", variant="spherical_mips", use_qm_scale=False, two_stage_tok=False)),
    "v11_l2_fixed":     (False, replace(BASE_CFG, inner_type="v11", variant="l2_norm",        use_qm_scale=False, two_stage_tok=False)),
    "v11_2stage_kattn": (False, replace(BASE_CFG, inner_type="v11", variant="spherical_mips", use_qm_scale=True,  two_stage_tok=True, k_sparse_tok=4)),
    "v11_flex_k4":      (False, replace(BASE_CFG, inner_type="v11", variant="spherical_mips", use_flex=True, emeans_block_size=32, k_blocks=4, use_qm_scale=True)),
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
# 2. Engine v11 (With Configurable Outlier Capping & Spherical Mode)
# =====================================================================

@torch.no_grad()
def _align_centroids(C_old: torch.Tensor, C_new: torch.Tensor) -> torch.Tensor:
    """Matches new v11 centroids (H, M, D) back to existing cluster slots via max cosine similarity."""
    H, M, _ = C_old.shape
    sim = torch.bmm(F.normalize(C_old, dim=-1), F.normalize(C_new, dim=-1).transpose(1, 2))
    sim_cpu = sim.cpu().tolist()
    perms = []
    for h in range(H):
        s = sim_cpu[h]
        used_r, used_c = set(), set()
        perm = [0] * M
        pairs = sorted(((s[i][j], i, j) for i in range(M) for j in range(M)), reverse=True)
        for _, i, j in pairs:
            if i not in used_r and j not in used_c:
                perm[i] = j
                used_r.add(i)
                used_c.add(j)
                if len(used_r) == M:
                    break
        perms.append(perm)
    perm_t = torch.tensor(perms, device=C_new.device, dtype=torch.long)
    return torch.gather(C_new, 1, perm_t.unsqueeze(-1).expand(-1, -1, C_new.size(-1)))

@torch.no_grad()
def run_engine_v11_aware(
    X: torch.Tensor,
    k: int,
    sub_n: int = 16384,
    use_sse: bool = True,
    valley_boost: float = 2.5,
    robust_cap_q: float | None = 0.95,
    bw_scale: float = 0.65,
    spherical: bool = False,
) -> torch.Tensor:
    orig_dtype = X.dtype
    X_in = torch.nan_to_num(X.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    if spherical:
        X_in = F.normalize(X_in, p=2, dim=-1)

    n, d = X_in.shape
    if n > sub_n:
        X_sub = X_in[torch.randint(0, n, (sub_n,), device=X.device)]
    else:
        sub_n = n
        X_sub = X_in[torch.randperm(n, device=X.device)]

    x_sub_sq = torch.ones(sub_n, device=X.device, dtype=torch.float32) if spherical else torch.linalg.vector_norm(X_sub, dim=1).square()
    is_B = (torch.arange(sub_n, device=X.device) & 1) == 1

    C_buf = torch.empty((k, d), device=X.device, dtype=torch.float32)
    init_c = X_sub.mean(dim=0)
    C_buf[0] = F.normalize(init_c, p=2, dim=-1) if spherical else init_c
    prev_V = torch.zeros((k, d), device=X.device, dtype=torch.float32)

    num_bins = 64
    idx = torch.arange(num_bins, device=X.device, dtype=torch.float32)
    dist2 = (idx.unsqueeze(0) - idx.unsqueeze(1)) ** 2
    sigma_bins = float(num_bins - 1) / 8.0

    m = 1
    while m < k:
        b_sched = min(m, max(1, (2 * m + 2) // 3), k - m)
        C_curr = C_buf[:m]

        if spherical:
            max_score, assign = torch.max(X_sub @ C_curr.T, dim=1)
            raw_sse = torch.clamp(1.0 - max_score, min=0.0)
        else:
            c_half = 0.5 * torch.linalg.vector_norm(C_curr, dim=1).square()
            max_score, assign = torch.max(torch.addmm(-c_half, X_sub, C_curr.T), dim=1)
            raw_sse = torch.clamp(x_sub_sq - 2.0 * max_score, min=0.0)

        if robust_cap_q is not None:
            cap = torch.quantile(raw_sse, robust_cap_q)
            w_pt = torch.clamp(cap / torch.clamp(raw_sse, min=1e-6), max=1.0)
        else:
            w_pt = torch.ones(sub_n, device=X.device, dtype=torch.float32)

        O = torch.zeros(sub_n, m, device=X.device, dtype=torch.float32).scatter_(1, assign.unsqueeze(1), 1.0)
        O_w = O * w_pt.unsqueeze(1)
        total_cnt = O.sum(dim=0)
        w_sum = O_w.sum(dim=0).clamp(min=1e-6)
        M = (O_w.T @ X_sub) / w_sum.unsqueeze(1)

        # 1. Dual-Axis Initialization (Deflated against prev_V)
        V1 = torch.randn(m, d, device=X.device, dtype=torch.float32)
        pv = prev_V[:m]
        V1 = V1 - (V1 * pv).sum(dim=1, keepdim=True) * pv
        V1 = V1 / (V1.norm(dim=1, keepdim=True) + 1e-9)

        V2 = torch.randn(m, d, device=X.device, dtype=torch.float32)
        V2 = V2 - (V2 * V1).sum(dim=1, keepdim=True) * V1
        V2 = V2 / (V2.norm(dim=1, keepdim=True) + 1e-9)

        # 2. Deflated Dual Power Iteration
        for _ in range(3):
            V_both = torch.cat([V1, V2], dim=0)
            dots = X_sub @ V_both.T
            dot1 = dots[:, :m].gather(1, assign.unsqueeze(1)).squeeze(1)
            dot2 = dots[:, m:].gather(1, assign.unsqueeze(1)).squeeze(1)

            mu1 = (M * V1).sum(dim=1)
            mu2 = (M * V2).sum(dim=1)

            z1 = dot1 - mu1[assign]
            z2 = dot2 - mu2[assign]

            Z1_mat = O_w * z1.unsqueeze(1)
            Z2_mat = O_w * z2.unsqueeze(1)
            Z_both = torch.cat([Z1_mat, Z2_mat], dim=1)

            V_next_both = Z_both.T @ X_sub - torch.cat([
                Z1_mat.sum(dim=0).unsqueeze(1) * M,
                Z2_mat.sum(dim=0).unsqueeze(1) * M,
            ], dim=0)

            V1 = V_next_both[:m]
            V1 = V1 / (V1.norm(dim=1, keepdim=True) + 1e-9)

            V2 = V_next_both[m:]
            V2 = V2 - (V2 * V1).sum(dim=1, keepdim=True) * V1
            V2 = V2 / (V2.norm(dim=1, keepdim=True) + 1e-9)

        # 3. 2D Coordinates & Mode Tracking
        V_both = torch.cat([V1, V2], dim=0)
        dots = X_sub @ V_both.T
        u1 = dots[:, :m].gather(1, assign.unsqueeze(1)).squeeze(1) - (M * V1).sum(dim=1)[assign]
        u2 = dots[:, m:].gather(1, assign.unsqueeze(1)).squeeze(1) - (M * V2).sum(dim=1)[assign]

        var1 = torch.zeros(m, device=X.device, dtype=torch.float32).scatter_add_(0, assign, w_pt * u1 * u1) / w_sum
        sig1 = torch.sqrt(var1.clamp(min=1e-12))
        cA_1, cA_2 = -0.75 * sig1, torch.zeros_like(sig1)
        cB_1, cB_2 = +0.75 * sig1, torch.zeros_like(sig1)

        for _ in range(3):
            dA2 = (u1 - cA_1[assign]) ** 2 + (u2 - cA_2[assign]) ** 2
            dB2 = (u1 - cB_1[assign]) ** 2 + (u2 - cB_2[assign]) ** 2
            is_B_pt = dB2 < dA2

            w_B_pt = w_pt * is_B_pt.float()
            w_A_pt = w_pt * (~is_B_pt).float()

            sum_wA = torch.zeros(m, device=X.device, dtype=torch.float32).scatter_add_(0, assign, w_A_pt).clamp(min=1e-6)
            sum_wB = torch.zeros(m, device=X.device, dtype=torch.float32).scatter_add_(0, assign, w_B_pt).clamp(min=1e-6)

            cA_1 = torch.zeros(m, device=X.device, dtype=torch.float32).scatter_add_(0, assign, w_A_pt * u1) / sum_wA
            cA_2 = torch.zeros(m, device=X.device, dtype=torch.float32).scatter_add_(0, assign, w_A_pt * u2) / sum_wA
            cB_1 = torch.zeros(m, device=X.device, dtype=torch.float32).scatter_add_(0, assign, w_B_pt * u1) / sum_wB
            cB_2 = torch.zeros(m, device=X.device, dtype=torch.float32).scatter_add_(0, assign, w_B_pt * u2) / sum_wB

        del_1, del_2 = cB_1 - cA_1, cB_2 - cA_2
        norm_del = torch.sqrt(del_1 * del_1 + del_2 * del_2).clamp(min=1e-9)
        V_star = (del_1 / norm_del).unsqueeze(1) * V1 + (del_2 / norm_del).unsqueeze(1) * V2
        V_star = V_star / (V_star.norm(dim=1, keepdim=True) + 1e-9)

        # 4. Project along V* & evaluate Split-Half KDE consensus
        proj = (X_sub @ V_star.T).gather(1, assign.unsqueeze(1)).squeeze(1)
        dot_mu_v = (M * V_star).sum(dim=1)

        diff = proj - dot_mu_v[assign]
        var_buf = torch.zeros(m, device=X.device, dtype=torch.float32).scatter_add_(0, assign, w_pt * diff * diff)
        var = var_buf / w_sum
        half_span = (4.0 * torch.sqrt(var.clamp(min=1e-12))).clamp(min=1e-4)
        lo_v = dot_mu_v - half_span
        span = 2.0 * half_span

        pos = ((proj - lo_v[assign]) / span[assign] * (num_bins - 1)).clamp(0.0, float(num_bins - 1) - 1e-4)
        b0 = pos.floor().long()
        b1 = (b0 + 1).clamp(max=num_bins - 1)
        frac = pos - b0.float()

        flat_base = is_B.long() * (m * num_bins) + assign * num_bins
        hist_ab = torch.zeros(2 * m * num_bins, device=X.device, dtype=torch.float32)
        hist_ab.scatter_add_(0, flat_base + b0, (1.0 - frac) * w_pt)
        hist_ab.scatter_add_(0, flat_base + b1, frac * w_pt)
        raw_ab = hist_ab.view(2, m, num_bins).permute(1, 2, 0)

        h_bins = bw_scale * sigma_bins * (0.5 * total_cnt).clamp(min=2.0).pow(-0.2)
        h2 = torch.clamp(h_bins * h_bins, min=0.05).view(m, 1, 1)
        K_kernel = torch.exp(-0.5 * dist2.unsqueeze(0) / h2)
        K_kernel = K_kernel / K_kernel.sum(dim=1, keepdim=True).clamp(min=1e-6)
        dens_ab = torch.bmm(K_kernel, raw_ab)
        dens_A, dens_B = dens_ab[:, :, 0], dens_ab[:, :, 1]

        lp_A = torch.cummax(dens_A, dim=1).values
        rp_A = torch.cummax(dens_A.flip(1), dim=1).values.flip(1)
        lp_B = torch.cummax(dens_B, dim=1).values
        rp_B = torch.cummax(dens_B.flip(1), dim=1).values.flip(1)

        val_persist = torch.minimum(
            torch.clamp(torch.minimum(lp_A, rp_A) - dens_A, min=0.0),
            torch.clamp(torch.minimum(lp_B, rp_B) - dens_B, min=0.0),
        )
        dens_mean = 0.5 * (dens_A + dens_B)
        cdf = torch.cumsum(dens_mean, dim=1) / dens_mean.sum(dim=1, keepdim=True).clamp(min=1e-6)
        v_max, v_idx = torch.max(val_persist * (4.0 * cdf * (1.0 - cdf)), dim=1)

        df_A = dens_A[:, 1:] - dens_A[:, :-1]
        df_B = dens_B[:, 1:] - dens_B[:, :-1]
        agree = (df_A * df_B).sum(dim=1) / ((df_A.norm(dim=1) * df_B.norm(dim=1)) + 1e-8)

        min_dens_max = torch.minimum(dens_A, dens_B).max(dim=1).values
        has_valley = (v_max > 0.03 * min_dens_max) & (agree > 0.35)
        v_cand = lo_v + (v_idx.float() / (num_bins - 1)) * span
        thresh = torch.where(has_valley, v_cand, dot_mu_v)

        is_left = proj < thresh[assign]
        cnt_left = (O * is_left.float().unsqueeze(1)).sum(dim=0)
        cnt_right = total_cnt - cnt_left
        thresh = torch.where((cnt_left < 1.0) | (cnt_right < 1.0), dot_mu_v, thresh)

        left_f = (proj < thresh[assign]).float().unsqueeze(1)
        Ow_left = O_w * left_f
        Ow_right = O_w - Ow_left
        cnt_left = (O * left_f).sum(dim=0)
        cnt_right = total_cnt - cnt_left

        wl = Ow_left.sum(dim=0)
        wr = Ow_right.sum(dim=0)
        C_left = (Ow_left.T @ X_sub) / wl.clamp(min=1e-6).unsqueeze(1)
        C_right = (Ow_right.T @ X_sub) / wr.clamp(min=1e-6).unsqueeze(1)
        if spherical:
            C_left = F.normalize(C_left, p=2, dim=-1)
            C_right = F.normalize(C_right, p=2, dim=-1)
        valid = (cnt_left >= 1.0) & (cnt_right >= 1.0)

        # 5. Split Ranking: Valley-Boosted Ward SSE
        if use_sse:
            delta_sse = (wl * wr / (wl + wr).clamp(min=1e-6)) * torch.linalg.vector_norm(C_left - C_right, dim=1).square()
            delta_sse = torch.where(valid, delta_sse, torch.zeros_like(delta_sse))
            ranking_score = delta_sse * torch.where(has_valley, valley_boost, 1.0)
        else:
            valley_rel_depth = torch.where(has_valley, v_max / min_dens_max.clamp(min=1e-6), torch.zeros_like(v_max))
            ranking_score = torch.where(valid, valley_rel_depth, torch.zeros_like(valley_rel_depth))
            if ranking_score.max() == 0:
                ranking_score = torch.where(valid, total_cnt, torch.zeros_like(total_cnt))

        active_cnt = int((ranking_score >= 0.05 * ranking_score.max()).sum().item())
        b = max(1, min(b_sched, active_cnt))
        _, top_ids = torch.topk(ranking_score, b, largest=True)

        ok = valid[top_ids]
        C_buf[top_ids] = torch.where(ok.unsqueeze(1), C_left[top_ids], C_curr[top_ids])
        C_buf[m : m + b] = torch.where(ok.unsqueeze(1), C_right[top_ids], X_sub[torch.arange(b, device=X.device) % sub_n])
        prev_V[top_ids] = V_star[top_ids]
        prev_V[m : m + b] = V_star[top_ids]
        m += b

    return C_buf.to(orig_dtype)


@torch.no_grad()
def static_engine_v11_heads(
    X: torch.Tensor,
    k: int,
    sub_n: int = 4096,
    use_sse: bool = True,
    valley_boost: float = 2.5,
    robust_cap_q: float | None = None,
    bw_scale: float = 0.65,
    spherical: bool = False,
) -> torch.Tensor:
    H, _, d = X.shape
    out = torch.empty(H, k, d, device=X.device, dtype=X.dtype)
    for h in range(H):
        out[h] = run_engine_v11_aware(
            X[h], k=k, sub_n=sub_n, use_sse=use_sse,
            valley_boost=valley_boost, robust_cap_q=robust_cap_q,
            bw_scale=bw_scale, spherical=spherical,
        )
    return out

@torch.no_grad()
def _damped_lloyd_step_heads(
    x_heads: torch.Tensor,
    C_heads: torch.Tensor,
    damp_scale: float = 2.0,
    x_sq: torch.Tensor = None,
    spherical: bool = False,
) -> torch.Tensor:
    H, N, D = x_heads.shape
    M = C_heads.shape[1]
    x_f32 = x_heads.float()
    C_f32 = C_heads.float()

    if spherical:
        x_f32 = F.normalize(x_f32, p=2, dim=-1)
        scores = torch.bmm(x_f32, C_f32.transpose(1, 2))
        max_score, assign = torch.max(scores, dim=-1)
        dist_sq = torch.clamp(1.0 - max_score, min=0.0)
    else:
        if x_sq is None:
            x_sq = (x_f32 * x_f32).sum(dim=-1)
        c_half = 0.5 * (C_f32 * C_f32).sum(dim=-1)
        scores = torch.bmm(x_f32, C_f32.transpose(1, 2)) - c_half.unsqueeze(1)
        max_score, assign = torch.max(scores, dim=-1)
        dist_sq = torch.clamp(x_sq - 2.0 * max_score, min=0.0)

    step = max(1, N // 1024)
    lam = torch.clamp(damp_scale * torch.median(dist_sq[:, ::step], dim=1, keepdim=True).values, min=1e-4)
    w = (lam / (dist_sq + lam)).to(x_heads.dtype)

    O_w = torch.zeros(H, N, M, device=x_heads.device, dtype=x_heads.dtype)
    O_w.scatter_(2, assign.unsqueeze(-1), w.unsqueeze(-1))

    counts = O_w.float().sum(dim=1).unsqueeze(-1)
    new_C = torch.bmm(O_w.transpose(1, 2), x_f32 if spherical else x_heads.float()) / torch.clamp(counts, min=1e-6)
    if spherical:
        new_C = F.normalize(new_C, p=2, dim=-1)
    return torch.where(counts < 1e-6, C_f32, new_C).to(C_heads.dtype)

# =====================================================================
# 3. TopKHot Straight-Through Estimator
# =====================================================================

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
        k, mode = ctx.k, ctx.mode

        if mode == "k_sparse":
            return g * active_mask, None, None

        with torch.no_grad():
            score_target = -g.masked_fill(valid_positions == 0, float('-inf'))
            _, topk_idx = torch.topk(score_target, k=k, dim=-1)
            hard_target = torch.zeros_like(x).scatter_(-1, topk_idx, 1.0) * valid_positions

        valid_cnt = valid_positions.sum(dim=-1, keepdim=True).clamp(min=1.0)
        g_scale = (g * valid_positions).abs().sum(dim=-1, keepdim=True) / valid_cnt

        if mode == "dense":
            grad_x = g_scale * (torch.sigmoid(x) - hard_target) * valid_positions
        elif mode == "2k_sparse":
            grad_x = g_scale * (torch.sigmoid(x) * active_mask - hard_target) * valid_positions
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return grad_x, None, None

# =====================================================================
# 4. Inner Attention Modules (Kattention, CausalWindowed v11, Global v11)
# =====================================================================

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


class CausalWindowedKAttention(nn.Module):
    """
    Causal-Windowed 2-Stage Engine v11 + Kattention:
    1. Updates spherical v11 codebook every `update_every` steps (copy_, zero EMA lag).
    2. Clusters & sorts keys inside causal windows of W tokens (0% future-token waste).
    3. Stage-1 coarse block gather + Stage-2 Kattention token Top-K (k=4).
    """
    def __init__(self, cfg: HNetConfig):
        super().__init__()
        assert cfg.dim_l3 % cfg.n_head == 0
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.head_dim = cfg.dim_l3 // cfg.n_head
        self.win_size = cfg.win_size
        self.update_every = cfg.update_every
        self.use_l2_norm = (cfg.variant == "l2_norm")
        self._step_cnt = 0
        self._last_refresh = 0

        self.q_attn = nn.Linear(cfg.dim_l3, cfg.dim_l3, bias=False)
        self.k_attn = nn.Linear(cfg.dim_l3, cfg.dim_l3, bias=False)
        self.v_attn = nn.Linear(cfg.dim_l3, cfg.dim_l3, bias=False)
        self.c_proj = nn.Linear(cfg.dim_l3, cfg.dim_l3, bias=False)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        init_c = F.normalize(
            torch.randn(self.n_head, cfg.emeans_clusters, self.head_dim) * 0.02,
            p=2,
            dim=-1,
        )
        self.register_buffer("C_heads", init_c)

        if self.use_l2_norm:
            self.logit_scale = nn.Parameter(torch.tensor(math.log(14.0)))
        else:
            self.qm = nn.Parameter(torch.ones(self.head_dim).normal_(mean=1.0, std=0.1))

    @torch._dynamo.disable
    @torch.no_grad()
    def _maybe_update_centroids(self, k: torch.Tensor):
        self._step_cnt += 1
        is_first = (self._step_cnt == 1)
        is_full_refresh = is_first or (
            (self._step_cnt - self._last_refresh) >= self.cfg.v11_refresh_interval
        )
        if not is_full_refresh and (self._step_cnt % self.update_every != 0):
            return

        B, H, T, D = k.shape
        x_heads = torch.nan_to_num(
            k.detach().float().transpose(0, 1).reshape(H, B * T, D),
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )
        if is_full_refresh:
            self._last_refresh = self._step_cnt
            C_seed = static_engine_v11_heads(
                x_heads,
                self.cfg.emeans_clusters,
                sub_n=4096,
                robust_cap_q=self.cfg.robust_cap_q,
                spherical=True,
            )
        else:
            C_seed = self.C_heads.float()

        C_next = _damped_lloyd_step_heads(
            x_heads, C_seed, damp_scale=2.0, spherical=True
        ).to(self.C_heads.dtype)

        self.C_heads.copy_(C_next)

    def forward(self, x):
        B, T, C_dim = x.size()
        H, D = self.n_head, self.head_dim
        W, bs = self.win_size, self.cfg.emeans_block_size
        N_W, B_W = T // W, W // bs
        N_B = N_W * B_W
        k_off = max(1, min(self.cfg.k_blocks - 1, N_B - B_W))

        q = self.q_attn(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_attn(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_attn(x).view(B, T, H, D).transpose(1, 2)

        k_mag = k.float().norm(dim=-1).view(B, H, N_W, W)

        if self.use_l2_norm:
            q_eff = F.normalize(q, p=2, dim=-1)
            k_eff = F.normalize(k, p=2, dim=-1)
            scale = self.logit_scale.exp().clamp(max=50.0)
        else:
            q_eff = q * (math.log(max(T, 2)) * self.qm)
            k_eff = k
            scale = 1.0 / math.sqrt(D)

        if self.training:
            self._maybe_update_centroids(k_eff)

        # 1. Assign on unit-norm k (dim=-1), sort k_eff by (cluster, k_mag) inside each W-window
        k_n = k_eff if self.use_l2_norm else F.normalize(k_eff, p=2, dim=-1)
        assign_k = torch.argmax(
            torch.einsum("bhtd,hmd->bhtm", k_n, self.C_heads.to(k_n.dtype)), dim=-1
        )
        assign_w = assign_k.view(B, H, N_W, W)
        if self.cfg.mag_sort:
            mag_rank = k_mag / (k_mag.amax(dim=-1, keepdim=True) + 1.0)
            perm_w = torch.argsort(assign_w.float() * 2.0 + (1.0 - mag_rank), dim=-1, stable=True)
        else:
            perm_w = torch.argsort(assign_w, dim=-1, stable=True)

        perm_kv = perm_w.unsqueeze(-1).expand(-1, -1, -1, -1, D)
        k_s = torch.gather(k_eff.view(B, H, N_W, W, D), 3, perm_kv).view(B, H, N_B, bs, D)
        v_s = torch.gather(v.view(B, H, N_W, W, D), 3, perm_kv).view(B, H, N_B, bs, D)

        # 2. Stage-1 Coarse Routing (past windows only -> 100% causally valid blocks)
        block_C = k_s.mean(dim=3)
        router_logits = torch.matmul(q_eff, block_C.transpose(-2, -1)) * scale

        block_win = (torch.arange(N_B, device=x.device) // B_W).view(1, 1, 1, N_B)
        q_win = (torch.arange(T, device=x.device) // W).view(1, 1, T, 1)
        past_win_mask = block_win < q_win

        off_logits = router_logits.masked_fill(~past_win_mask, float("-inf"))
        off_mask = TopKHot.apply(off_logits, k_off, self.cfg.topk_mode)
        _, off_idx = torch.topk(off_logits, k=k_off, dim=-1)
        off_valid = torch.gather(off_mask, -1, off_idx)

        flat_idx = off_idx.view(B, H, T * k_off)
        gather_kv = flat_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, bs, D)
        k_glob = torch.gather(k_s, 2, gather_kv).view(B, H, T, k_off * bs, D)
        v_glob = torch.gather(v_s, 2, gather_kv).view(B, H, T, k_off * bs, D)
        glob_gate = off_valid.unsqueeze(-1).expand(-1, -1, -1, -1, bs).reshape(B, H, T, k_off * bs)

        # 3. Local Causal Window (exact within current W window)
        q_win_1d = torch.arange(T, device=x.device) // W
        k_loc = k_eff.view(B, H, N_W, W, D)[:, :, q_win_1d, :, :]
        v_loc = v.view(B, H, N_W, W, D)[:, :, q_win_1d, :, :]
        loc_pos = (q_win_1d * W).view(T, 1) + torch.arange(W, device=x.device).view(1, W)
        t_pos = torch.arange(T, device=x.device).view(T, 1)
        local_causal = (t_pos >= loc_pos).view(1, 1, T, W).expand(B, H, -1, -1)

        k_g = torch.cat([k_loc, k_glob], dim=3)
        v_g = torch.cat([v_loc, v_glob], dim=3)
        full_mask = torch.cat([local_causal, glob_gate > 0], dim=-1)
        gate = torch.cat(
            [torch.ones(B, H, T, W, device=x.device, dtype=q.dtype), glob_gate.to(q.dtype)],
            dim=-1,
        )

        # 4. Stage-2 Kattention Token Top-K (k=4)
        scores = torch.einsum("bhtd,bhtkd->bhtk", q_eff, k_g) * scale
        scores = scores.masked_fill(~full_mask, float("-inf"))
        tok_mask = TopKHot.apply(
            scores, min(self.cfg.k_sparse_tok, scores.size(-1)), self.cfg.topk_mode
        )

        probs = F.softmax(scores, dim=-1) * tok_mask * gate
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)

        out = torch.einsum("bhtk,bhtkd->bhtd", probs, v_g)
        return self.resid_dropout(self.c_proj(out.transpose(1, 2).contiguous().view(B, T, C_dim)))


class UniversalEngineV11Attention(nn.Module):
    def __init__(self, cfg: HNetConfig):
        super().__init__()
        assert cfg.dim_l3 % cfg.n_head == 0
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.head_dim = cfg.dim_l3 // cfg.n_head
        self._step_cnt = 0

        self.q_attn = nn.Linear(cfg.dim_l3, cfg.dim_l3, bias=False)
        self.k_attn = nn.Linear(cfg.dim_l3, cfg.dim_l3, bias=False)
        self.v_attn = nn.Linear(cfg.dim_l3, cfg.dim_l3, bias=False)
        self.c_proj = nn.Linear(cfg.dim_l3, cfg.dim_l3, bias=False)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        init_c = F.normalize(torch.randn(self.n_head, cfg.emeans_clusters, self.head_dim) * 0.02, p=2, dim=-1)
        self.register_buffer("C_heads", init_c)

        if cfg.variant == "l2_norm":
            self.logit_scale = nn.Parameter(torch.tensor(math.log(14.0)))

        if cfg.use_qm_scale:
            self.qm = nn.Parameter(torch.ones(self.head_dim).normal_(mean=1.0, std=0.1))

    @torch._dynamo.disable
    @torch.no_grad()
    def _update_centroids(self, k: torch.Tensor):
        """Amortized codebook update: full bisecting v11 periodically, fast warm-started Lloyd + EMA otherwise."""
        self._step_cnt += 1
        B, H, T, D = k.shape
        x_heads = torch.nan_to_num(
            k.detach().float().transpose(0, 1).reshape(H, B * T, D),
            nan=0.0, posinf=1e4, neginf=-1e4
        )

        is_full_refresh = (self._step_cnt % self.cfg.v11_refresh_interval == 1)
        if is_full_refresh:
            C_seed = static_engine_v11_heads(
                x_heads,
                self.cfg.emeans_clusters,
                sub_n=4096,
                robust_cap_q=self.cfg.robust_cap_q,
                spherical=True,
            )
        else:
            C_seed = self.C_heads.float()

        C_next = _damped_lloyd_step_heads(
            x_heads, C_seed, damp_scale=2.0, spherical=True
        ).to(self.C_heads.dtype)

        if self._step_cnt == 1 or is_full_refresh:
            self.C_heads.copy_(C_next)
        else:
            self.C_heads.lerp_(C_next, self.cfg.ema_alpha)
            F.normalize(self.C_heads, p=2, dim=-1, out=self.C_heads)

    def forward(self, x):
        B, T, C_dim = x.size()
        H, D = self.n_head, self.head_dim
        bs = self.cfg.emeans_block_size
        N_B = T // bs
        k_off = max(1, min(self.cfg.k_blocks - 1, N_B - 1))

        q = self.q_attn(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_attn(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_attn(x).view(B, T, H, D).transpose(1, 2)

        k_mag = k.float().norm(dim=-1)

        if self.cfg.variant == "l2_norm":
            q_eff = F.normalize(q, p=2, dim=-1)
            k_eff = F.normalize(k, p=2, dim=-1)
            scale = self.logit_scale.exp().clamp(max=50.0)
        else:
            scale_t = math.log(max(T, 2)) if self.cfg.use_qm_scale else 1.0
            q_eff = q * (scale_t * self.qm) if self.cfg.use_qm_scale else q
            k_eff = k
            scale = 1.0 / math.sqrt(D)

        C_buf = self.C_heads.to(k_eff.dtype)
        k_scores = torch.einsum("bhtd,hmd->bhtm", F.normalize(k_eff, p=2, dim=-1) if self.cfg.variant != "l2_norm" else k_eff, C_buf)
        assign_k = torch.argmax(k_scores, dim=-1)

        if self.training:
            self._update_centroids(k_eff)

        if self.cfg.mag_sort:
            mag_rank = k_mag / (k_mag.amax(dim=-1, keepdim=True) + 1.0)
            sort_key = assign_k.float() * 2.0 + (1.0 - mag_rank)
            perm_k = torch.argsort(sort_key, dim=-1, stable=True)
        else:
            perm_k = torch.argsort(assign_k, dim=-1, stable=True)

        perm_kv = perm_k.unsqueeze(-1).expand(-1, -1, -1, D)
        k_s = torch.gather(k_eff, 2, perm_kv)
        v_s = torch.gather(v, 2, perm_kv)

        orig_k_blocks = perm_k.view(B, H, N_B, bs)
        block_C = k_s.view(B, H, N_B, bs, D).mean(dim=3)
        router_logits = torch.matmul(q_eff, block_C.transpose(-2, -1)) * scale

        q_block_id = torch.arange(T, device=q.device) // bs
        block_q_start = (q_block_id * bs).view(1, 1, T, 1)
        orig_k_min = orig_k_blocks.amin(dim=-1).unsqueeze(2)
        block_has_past = (orig_k_min < block_q_start)

        off_logits = router_logits.masked_fill(~block_has_past, float('-inf'))
        off_mask = TopKHot.apply(off_logits, k_off, self.cfg.topk_mode)
        _, off_idx = torch.topk(off_logits, k=k_off, dim=-1)
        off_valid = torch.gather(off_mask, -1, off_idx)

        if self.cfg.use_flex:
            off_valid_full = torch.zeros(B, H, T, N_B, device=q.device, dtype=torch.bool)
            off_valid_full.scatter_(3, off_idx, off_valid > 0)

            def mask_mod(b, h, q_idx, kv_idx):
                orig_pos = perm_k[b, h, kv_idx]
                causal = orig_pos <= q_idx
                in_local = (orig_pos // bs) == (q_idx // bs)
                in_glob = off_valid_full[b, h, q_idx, kv_idx // bs]
                return causal & (in_local | in_glob)

            block_mask = create_block_mask(mask_mod, B=B, H=H, Q_LEN=T, KV_LEN=T, device=q.device, BLOCK_SIZE=bs)
            q_flex = q_eff * (scale * math.sqrt(D)) if self.cfg.variant == "l2_norm" else q_eff
            out = flex_attention(q_flex, k_s, v_s, block_mask=block_mask)

            has_any = block_has_past.any(dim=-1, keepdim=True)
            safe_r = router_logits.masked_fill(~block_has_past, -1e9)
            s = (off_mask * F.softmax(safe_r, dim=-1)).sum(dim=-1, keepdim=True) * has_any.to(q.dtype)
            out = out * (1.0 + s - s.detach())

            out = out.transpose(1, 2).contiguous().view(B, T, C_dim)
            return self.resid_dropout(self.c_proj(out))

        k_blocks_all = k_s.view(B, H, N_B, bs, D)
        v_blocks_all = v_s.view(B, H, N_B, bs, D)

        flat_idx = off_idx.view(B, H, T * k_off)
        gather_kv = flat_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, bs, D)
        k_glob = torch.gather(k_blocks_all, 2, gather_kv).view(B, H, T, k_off * bs, D)
        v_glob = torch.gather(v_blocks_all, 2, gather_kv).view(B, H, T, k_off * bs, D)
        orig_k_glob = torch.gather(orig_k_blocks, 2, flat_idx.unsqueeze(-1).expand(-1, -1, -1, bs)).view(B, H, T, k_off * bs)

        k_loc = k_eff.view(B, H, N_B, bs, D)[:, :, q_block_id, :, :]
        v_loc = v.view(B, H, N_B, bs, D)[:, :, q_block_id, :, :]
        orig_loc = torch.arange(T, device=q.device).view(N_B, bs)[q_block_id, :].view(1, 1, T, bs)

        t_pos = torch.arange(T, device=q.device).view(1, 1, T, 1)
        local_causal = (t_pos >= orig_loc).expand(B, H, -1, -1)
        glob_valid_tok = off_valid.unsqueeze(-1).expand(-1, -1, -1, -1, bs).reshape(B, H, T, k_off * bs)
        glob_past = (orig_k_glob < block_q_start) & (glob_valid_tok > 0)

        k_g = torch.cat([k_loc, k_glob], dim=3)
        v_g = torch.cat([v_loc, v_glob], dim=3)
        full_mask = torch.cat([local_causal, glob_past], dim=-1)
        gate = torch.cat([torch.ones(B, H, T, bs, device=q.device, dtype=q.dtype), glob_valid_tok.to(q.dtype)], dim=-1)

        scores = torch.einsum("bhtd,bhtkd->bhtk", q_eff, k_g) * scale
        scores = scores.masked_fill(~full_mask, float('-inf'))

        if self.cfg.two_stage_tok:
            k_tok = min(self.cfg.k_sparse_tok, scores.size(-1))
            tok_mask = TopKHot.apply(scores, k_tok, self.cfg.topk_mode)
            probs = F.softmax(scores, dim=-1) * tok_mask * gate
        else:
            probs = F.softmax(scores, dim=-1) * gate

        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)

        out = torch.einsum("bhtk,bhtkd->bhtd", probs, v_g)
        out = out.transpose(1, 2).contiguous().view(B, T, C_dim)
        return self.resid_dropout(self.c_proj(out))


class InnerBlock(nn.Module):
    def __init__(self, cfg: HNetConfig):
        super().__init__()
        self.ln_1 = modules.LayerNorm(cfg.dim_l3, bias=False)
        if cfg.inner_type == "kattention":
            self.attn = Kattention(
                cfg.dim_l3, cfg.n_head, k_sparse=cfg.k_sparse_tok,
                topk_mode=cfg.topk_mode, dropout=cfg.dropout, bias=False
            )
        elif cfg.inner_type == "cw_v11":
            self.attn = CausalWindowedKAttention(cfg)
        else:
            self.attn = UniversalEngineV11Attention(cfg)
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
# 5. Patcher Trunk (VoronoiSegmentGPTPatcher & VoronoiConvGPTPatcher) & HNet
# =====================================================================

class CausalConv1d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, bias=True):
        super().__init__()
        self.pad = kernel_size - 1
        self.kernel_size = kernel_size
        self.proj = nn.Linear(in_c * kernel_size, out_c, bias=bias)

    def forward(self, x):
        if self.pad > 0:
            x = F.pad(x, (0, 0, self.pad, 0))
        xw = x.unfold(1, self.kernel_size, 1).reshape(x.shape[0], -1, self.proj.in_features)
        return self.proj(xw)

class CausalConvBlock(nn.Module):
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        self.ln1 = modules.LayerNorm(dim, False)
        self.conv = CausalConv1d(dim, dim, kernel_size=kernel_size, bias=False)
        self.ln2 = modules.LayerNorm(dim, False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4, bias=False),
            nn.GELU(),
            nn.Linear(dim * 4, dim, bias=False),
        )

    def forward(self, x):
        x = x + self.conv(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class VoronoiCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
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

class VoronoiSegmentGPTPatcher(nn.Module):
    """Attention-based GPT Pre-LN Patcher (used for sLayers[1]: L=128 -> 64)."""
    def __init__(self, outer_dim, inner_dim, outer_seq, inner_seq, cfg: HNetConfig, pin_tail=True, novelty_window=16):
        super().__init__()
        self.outer_dim, self.inner_dim = outer_dim, inner_dim
        self.outer_seq, self.inner_seq = outer_seq, inner_seq
        self.cfg = cfg
        self.pin_tail = pin_tail
        self.novelty_window = novelty_window

        self.query_block = modules.Block(outer_dim, cfg.n_head, 0.0, False)
        self.up_gate = nn.Sequential(
            nn.Linear(outer_dim, max(1, outer_dim // 4), bias=False),
            nn.GELU(),
            nn.Linear(max(1, outer_dim // 4), 1),
        )
        self.novelty_scale = nn.Parameter(torch.tensor(0.5))
        self.pool_q = nn.Linear(outer_dim, outer_dim, bias=False)
        self.pool_k = nn.Linear(outer_dim, outer_dim, bias=False)
        self.pool_alpha = nn.Parameter(torch.tensor(0.5))

        self.up_proj = nn.Linear(outer_dim, inner_dim, bias=False)
        self.proj_block = modules.Block(inner_dim, cfg.n_head, 0.0, False)

        self.output_pos_emb = nn.Embedding(outer_seq, inner_dim)
        self.q_norm = modules.LayerNorm(inner_dim, False)
        self.down_norm = modules.LayerNorm(inner_dim, False)
        self.causal_upsampler = VoronoiCrossAttention(inner_dim, num_heads=cfg.n_head)
        self.down_block = modules.Block(inner_dim, cfg.n_head, 0.0, False)
        self.pds_norm = modules.LayerNorm(inner_dim, False)
        self.down_proj = nn.Linear(inner_dim, outer_dim, bias=False)
        self.down_scan = modules.Block(outer_dim, cfg.n_head, 0.0, False)

    def abstract_up(self, x):
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
            novelty = (1.0 - sim.masked_fill(~past_mask.unsqueeze(0), -1e9).max(dim=-1).values).clamp(min=0.0, max=2.0)

            sel_score = gate_logits + self.novelty_scale * novelty
            if self.training:
                gumbel = -torch.log(-torch.log(torch.rand_like(sel_score).clamp(1e-5, 1.0 - 1e-5)))
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
        voronoi_scores = (torch.bmm(pq, pk.transpose(1, 2)) - 0.5 * (pk * pk).sum(dim=-1).unsqueeze(1)) / math.sqrt(D)

        pos_l = torch.arange(L, device=x.device, dtype=torch.long).view(1, 1, L)
        k_end = sorted_indices.unsqueeze(-1)
        k_start = F.pad(sorted_indices[:, :-1], (1, 0), value=-1).unsqueeze(-1)
        seg_mask = (pos_l > k_start) & (pos_l <= k_end)

        voronoi_scores = voronoi_scores.masked_fill(~seg_mask, float('-inf'))
        pooled = torch.bmm(F.softmax(voronoi_scores, dim=-1), h_n)
        compressed = utils.rms_norm(anchor_ste + self.pool_alpha * pooled)
        up = self.proj_block(self.up_proj(compressed))
        return up, [gathered, t_outer, t_inner]

    def abstract_down(self, x, residual):
        residual_x, t_outer, t_inner = residual
        B, L, _ = residual_x.shape
        x = self.down_norm(x)

        pos = torch.arange(0, L, dtype=torch.long, device=x.device)
        pos_emb = self.output_pos_emb(pos).unsqueeze(0).expand(B, L, -1)

        scattered = self.causal_upsampler(self.q_norm(pos_emb), t_outer, x, t_inner)
        pds = self.down_proj(self.pds_norm(self.down_block(scattered)))
        xout = self.down_scan(residual_x + pds)
        return xout

class VoronoiConvGPTPatcher(VoronoiSegmentGPTPatcher):
    """Conv-based GPT Pre-LN Patcher (used for sLayers[0]: L=256 raw bytes -> 128)."""
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

        self.sLayers = nn.ModuleList([
            VoronoiConvGPTPatcher(cfg.dim_l1, cfg.dim_l2, cfg.block_size, cfg.k_gather_l1, cfg, pin_tail=True, novelty_window=16),
            VoronoiSegmentGPTPatcher(cfg.dim_l2, cfg.dim_l3, cfg.k_gather_l1, cfg.k_gather_l2, cfg, pin_tail=True, novelty_window=16),
        ])

        self.inner = nn.ModuleList([InnerBlock(cfg) for _ in range(cfg.n_inner)])

        if cfg.gpt_init:
            self.apply(self._init_gpt_weights)
            res_scale = 0.02 / math.sqrt(2.0 * (cfg.n_inner + 4))
            for pname, p in self.named_parameters():
                if pname.endswith(("mlp.2.weight", "conv.proj.weight", "c_proj.weight")):
                    torch.nn.init.normal_(p, mean=0.0, std=res_scale)
        else:
            self.apply(self._init_weights)

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

    def forward(self, idx, targets=None):
        b, t = idx.shape
        positions = torch.arange(0, t, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = utils.rms_norm(self.tok_emb(idx) + self.pos_emb(positions))

        residuals = []
        for layer in self.sLayers:
            x, res = layer.abstract_up(x)
            residuals.append(res)

        for block in self.inner:
            x = block(x)

        for i in reversed(range(len(self.sLayers))):
            x = self.sLayers[i].abstract_down(x, residuals[i])

        wnl = 0.0
        if self.training and self.cfg.wnll:
            wnl = self.wnll(x)
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
            ce_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets_aligned.reshape(-1), reduction=red)
            if self.cfg.wnll:
                ce_loss = ce_loss.view(b, T_out)
            gloss = F.cross_entropy(logits[:, -1, :].reshape(-1, logits.shape[-1]), targets_aligned[:, -1].reshape(-1)).detach()

        pure_ce = ce_loss.mean().detach()
        loss = ce_loss
        if self.training and self.cfg.wnll:
            chaos = ((1.0 - wnl.detach()) / 2.0).mean() * 0.99
            loss = (loss / (1.0 + wnl.detach().reshape(loss.shape) * chaos)).mean() + wnl.mean()

        return loss, pure_ce, gloss

# =====================================================================
# 6. Benchmark Orchestrator
# =====================================================================

def evaluate_model(raw_model):
    raw_model.eval()
    tot_ce, tot_gen = 0.0, 0.0
    with torch.no_grad():
        for _ in range(eval_iters):
            vx, vy = get_batch('val')
            _, pure_ce, gloss = raw_model(vx, vy)
            tot_ce += pure_ce.item()
            tot_gen += gloss.item()
    raw_model.train()
    return tot_ce / eval_iters, tot_gen / eval_iters

def run_benchmarks():
    torch.set_float32_matmul_precision('high')
    active = {k: cfg for k, (enabled, cfg) in EXPERIMENTS.items() if enabled}
    print(f"Device: {device} | Vocab: {vocab_size} | Experiments: {list(active.keys())}")

    raw_models, models, optimizers = {}, {}, {}
    bx_sanity, by_sanity = get_batch('train')

    for name, cfg in active.items():
        torch.manual_seed(1337)
        m = HNet(vocab_size=vocab_size, cfg=cfg).to(device)
        print(f"  {name:<18} params: {sum(p.numel() for p in m.parameters()):,}")
        raw_models[name] = m
        optimizers[name] = torch.optim.AdamW(m.parameters(), lr=learning_rate)

        t0 = time.perf_counter()
        compiled_m = torch.compile(m)
        models[name] = compiled_m

        for _ in range(2):
            optimizers[name].zero_grad(set_to_none=True)
            loss, _, _ = compiled_m(bx_sanity, by_sanity)
            loss.backward()
            optimizers[name].zero_grad(set_to_none=True)
        if device == 'cuda':
            torch.cuda.synchronize()
        print(f"  [COMPILED]  {name:<18} in {(time.perf_counter() - t0):.2f} s")

    col_widths = {n: max(len(n), 16) for n in models.keys()}
    print("=" * 125)
    print(f"{'Step':<6} | " + " | ".join([f"{n:<{col_widths[n]}}" for n in models.keys()]) + " | ms/step")
    print(f"{'':<6} | " + " | ".join([f"{' val_ce    gen':<{col_widths[n]}}" for n in models.keys()]) + " |")
    print("-" * 125)

    interval_timing = {n: 0.0 for n in models.keys()}

    for it in range(max_iters):
        lr = get_lr(it)
        bx, by = get_batch('train')
        step = it + 1
        is_eval = (step % eval_interval == 0 or step == max_iters)

        for name, model in models.items():
            opt = optimizers[name]
            for pg in opt.param_groups:
                pg['lr'] = lr

            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            opt.zero_grad(set_to_none=True)
            loss, _, _ = model(bx, by)
            loss.backward()
            opt.step()

            if device == 'cuda':
                torch.cuda.synchronize()
            interval_timing[name] += (time.perf_counter() - t0) * 1000.0

        if is_eval:
            eval_cells, ms_cells = [], []
            for name, rm in raw_models.items():
                v_ce, v_gen = evaluate_model(rm)
                eval_cells.append(f"{f'{v_ce:7.4f} {v_gen:7.4f}':<{col_widths[name]}}")
                ms_cells.append(f"{interval_timing[name] / eval_interval:5.1f}")
                interval_timing[name] = 0.0
            print(f"{step:<6d} | " + " | ".join(eval_cells) + " | " + " / ".join(ms_cells))

if __name__ == "__main__":
    run_benchmarks()