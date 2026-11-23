"""
Standalone Reality Auditor + FlexAttention vs Gather Benchmark for models/HNet4.py
Compares (at the exact same W & bs):
  1. kattn_baseline : Exact Kattention (dense T x T + log(T)*qm + TopKHot k=4)
  2. cw_noflex      : CausalWindowedKAttention (explicit gather + Stage-2 TopKHot k=4)
  3. cw_flex        : CausalWindowedKAttention (BlockMask.from_kv_blocks + flex_attention, Stage-1 soft)

Usage:
    # Size 1024 (1024 -> 512 -> 256, B=16, auto W=32 bs=16):
    python hnetcluster.py --steps 5000 --audit-every 1000 --block-size 1024 --batch-size 16 --compile

    # Optional: 1:1 soft-attention parity on cw_noflex (skip Stage-2 TopKHot) or bs=32 W=64:
    python hnetcluster.py --steps 5000 --audit-every 1000 --block-size 1024 --batch-size 16 --no-stage2 --compile
"""

import argparse
from dataclasses import replace
import math
import time
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, BlockMask

from scipy.optimize import linear_sum_assignment

import models.HNet4 as hnet4


# =====================================================================
# 1. Compiled Batched Spherical Engine v11 (Zero .item() / Zero Loops)
# =====================================================================

@torch.no_grad()
def compiled_v11_batched_spherical(
    X_heads: torch.Tensor,
    k: int = 16,
    sub_n: int = 4096,
    valley_boost: float = 2.5,
    bw_scale: float = 0.65,
) -> torch.Tensor:
    orig_dtype = X_heads.dtype
    dev = X_heads.device
    X_in = F.normalize(
        torch.nan_to_num(X_heads.float(), nan=0.0, posinf=1e4, neginf=-1e4),
        p=2,
        dim=-1,
    )
    H, N, D = X_in.shape

    if N > sub_n:
        S = sub_n
        idx_sub = torch.randint(0, N, (H, S), device=dev)
        X_sub = torch.gather(X_in, 1, idx_sub.unsqueeze(-1).expand(-1, -1, D))
    else:
        S = N
        perm = torch.randperm(S, device=dev)
        X_sub = X_in[:, perm, :]

    is_B = (torch.arange(S, device=dev) & 1) == 1
    C_buf = torch.zeros((H, k, D), device=dev, dtype=torch.float32)
    init_c = F.normalize(X_sub.mean(dim=1), p=2, dim=-1)
    C_buf = torch.cat([init_c.unsqueeze(1), C_buf[:, 1:, :]], dim=1)
    prev_V = torch.zeros((H, k, D), device=dev, dtype=torch.float32)

    num_bins = 64
    idx_b = torch.arange(num_bins, device=dev, dtype=torch.float32)
    dist2 = (idx_b.unsqueeze(0) - idx_b.unsqueeze(1)) ** 2
    sigma_bins = float(num_bins - 1) / 8.0

    schedule = []
    m_cur = 1
    while m_cur < k:
        b_sched = min(m_cur, max(1, (2 * m_cur + 2) // 3), k - m_cur)
        schedule.append((m_cur, b_sched))
        m_cur += b_sched

    for m, b in schedule:
        C_curr = C_buf[:, :m, :]
        scores = torch.bmm(X_sub, C_curr.transpose(1, 2))
        _, assign = torch.max(scores, dim=-1)

        O = torch.zeros(H, S, m, device=dev, dtype=torch.float32).scatter_(
            2, assign.unsqueeze(-1), 1.0
        )
        total_cnt = O.sum(dim=1)
        w_sum = total_cnt.clamp(min=1e-6)
        M = torch.bmm(O.transpose(1, 2), X_sub) / w_sum.unsqueeze(-1)

        pv = prev_V[:, :m, :]
        V1 = torch.randn(H, m, D, device=dev, dtype=torch.float32)
        V1 = V1 - (V1 * pv).sum(dim=-1, keepdim=True) * pv
        V1 = F.normalize(V1, p=2, dim=-1, eps=1e-9)

        V2 = torch.randn(H, m, D, device=dev, dtype=torch.float32)
        V2 = V2 - (V2 * V1).sum(dim=-1, keepdim=True) * V1
        V2 = F.normalize(V2, p=2, dim=-1, eps=1e-9)

        for _ in range(3):
            V_both = torch.cat([V1, V2], dim=1)
            dots = torch.bmm(X_sub, V_both.transpose(1, 2))
            dot1 = dots[:, :, :m].gather(2, assign.unsqueeze(-1)).squeeze(-1)
            dot2 = dots[:, :, m:].gather(2, assign.unsqueeze(-1)).squeeze(-1)

            mu1 = (M * V1).sum(dim=-1)
            mu2 = (M * V2).sum(dim=-1)
            z1 = dot1 - mu1.gather(1, assign)
            z2 = dot2 - mu2.gather(1, assign)

            Z1_mat = O * z1.unsqueeze(-1)
            Z2_mat = O * z2.unsqueeze(-1)
            Z_both = torch.cat([Z1_mat, Z2_mat], dim=2)

            V_next_both = torch.bmm(Z_both.transpose(1, 2), X_sub) - torch.cat([
                Z1_mat.sum(dim=1).unsqueeze(-1) * M,
                Z2_mat.sum(dim=1).unsqueeze(-1) * M,
            ], dim=1)

            V1 = F.normalize(V_next_both[:, :m, :], p=2, dim=-1, eps=1e-9)
            V2 = V_next_both[:, m:, :]
            V2 = F.normalize(V2 - (V2 * V1).sum(dim=-1, keepdim=True) * V1, p=2, dim=-1, eps=1e-9)

        V_both = torch.cat([V1, V2], dim=1)
        dots = torch.bmm(X_sub, V_both.transpose(1, 2))
        u1 = dots[:, :, :m].gather(2, assign.unsqueeze(-1)).squeeze(-1) - (M * V1).sum(dim=-1).gather(1, assign)
        u2 = dots[:, :, m:].gather(2, assign.unsqueeze(-1)).squeeze(-1) - (M * V2).sum(dim=-1).gather(1, assign)

        var1 = torch.zeros(H, m, device=dev, dtype=torch.float32).scatter_add_(1, assign, u1 * u1) / w_sum
        sig1 = torch.sqrt(var1.clamp(min=1e-12))
        cA_1, cA_2 = -0.75 * sig1, torch.zeros_like(sig1)
        cB_1, cB_2 = +0.75 * sig1, torch.zeros_like(sig1)

        for _ in range(3):
            dA2 = (u1 - cA_1.gather(1, assign)) ** 2 + (u2 - cA_2.gather(1, assign)) ** 2
            dB2 = (u1 - cB_1.gather(1, assign)) ** 2 + (u2 - cB_2.gather(1, assign)) ** 2
            w_B = (dB2 < dA2).float()
            w_A = 1.0 - w_B

            sum_wA = torch.zeros(H, m, device=dev, dtype=torch.float32).scatter_add_(1, assign, w_A).clamp(min=1e-6)
            sum_wB = torch.zeros(H, m, device=dev, dtype=torch.float32).scatter_add_(1, assign, w_B).clamp(min=1e-6)

            cA_1 = torch.zeros(H, m, device=dev, dtype=torch.float32).scatter_add_(1, assign, w_A * u1) / sum_wA
            cA_2 = torch.zeros(H, m, device=dev, dtype=torch.float32).scatter_add_(1, assign, w_A * u2) / sum_wA
            cB_1 = torch.zeros(H, m, device=dev, dtype=torch.float32).scatter_add_(1, assign, w_B * u1) / sum_wB
            cB_2 = torch.zeros(H, m, device=dev, dtype=torch.float32).scatter_add_(1, assign, w_B * u2) / sum_wB

        del_1, del_2 = cB_1 - cA_1, cB_2 - cA_2
        norm_del = torch.sqrt(del_1 * del_1 + del_2 * del_2).clamp(min=1e-9)
        V_star = F.normalize(
            (del_1 / norm_del).unsqueeze(-1) * V1 + (del_2 / norm_del).unsqueeze(-1) * V2,
            p=2,
            dim=-1,
            eps=1e-9,
        )

        proj = torch.bmm(X_sub, V_star.transpose(1, 2)).gather(2, assign.unsqueeze(-1)).squeeze(-1)
        dot_mu_v = (M * V_star).sum(dim=-1)

        diff = proj - dot_mu_v.gather(1, assign)
        var = torch.zeros(H, m, device=dev, dtype=torch.float32).scatter_add_(1, assign, diff * diff) / w_sum
        half_span = (4.0 * torch.sqrt(var.clamp(min=1e-12))).clamp(min=1e-4)
        lo_v = dot_mu_v - half_span
        span = 2.0 * half_span

        pos = ((proj - lo_v.gather(1, assign)) / span.gather(1, assign) * (num_bins - 1)).clamp(
            0.0, float(num_bins - 1) - 1e-4
        )
        b0 = pos.floor().long()
        b1 = (b0 + 1).clamp(max=num_bins - 1)
        frac = pos - b0.float()

        flat_base = is_B.long().view(1, S) * (m * num_bins) + assign * num_bins
        hist_ab = torch.zeros(H, 2 * m * num_bins, device=dev, dtype=torch.float32)
        hist_ab.scatter_add_(1, flat_base + b0, 1.0 - frac)
        hist_ab.scatter_add_(1, flat_base + b1, frac)
        raw_ab = hist_ab.view(H, 2, m, num_bins).permute(0, 2, 3, 1)

        h_bins = bw_scale * sigma_bins * (0.5 * total_cnt).clamp(min=2.0).pow(-0.2)
        h2 = torch.clamp(h_bins * h_bins, min=0.05).view(H, m, 1, 1)
        K_kernel = torch.exp(-0.5 * dist2.view(1, 1, num_bins, num_bins) / h2)
        K_kernel = K_kernel / K_kernel.sum(dim=2, keepdim=True).clamp(min=1e-6)
        dens_ab = torch.matmul(K_kernel, raw_ab)
        dens_A, dens_B = dens_ab[..., 0], dens_ab[..., 1]

        lp_A = torch.cummax(dens_A, dim=-1).values
        rp_A = torch.cummax(dens_A.flip(-1), dim=-1).values.flip(-1)
        lp_B = torch.cummax(dens_B, dim=-1).values
        rp_B = torch.cummax(dens_B.flip(-1), dim=-1).values.flip(-1)

        val_persist = torch.minimum(
            (torch.minimum(lp_A, rp_A) - dens_A).clamp(min=0.0),
            (torch.minimum(lp_B, rp_B) - dens_B).clamp(min=0.0),
        )
        dens_mean = 0.5 * (dens_A + dens_B)
        cdf = torch.cumsum(dens_mean, dim=-1) / dens_mean.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        v_max, v_idx = torch.max(val_persist * (4.0 * cdf * (1.0 - cdf)), dim=-1)

        df_A = dens_A[..., 1:] - dens_A[..., :-1]
        df_B = dens_B[..., 1:] - dens_B[..., :-1]
        agree = (df_A * df_B).sum(dim=-1) / ((df_A.norm(dim=-1) * df_B.norm(dim=-1)) + 1e-8)

        min_dens_max = torch.minimum(dens_A, dens_B).amax(dim=-1)
        has_valley = (v_max > 0.03 * min_dens_max) & (agree > 0.35)
        v_cand = lo_v + (v_idx.float() / (num_bins - 1)) * span
        thresh = torch.where(has_valley, v_cand, dot_mu_v)

        is_left = proj < thresh.gather(1, assign)
        cnt_left = (O * is_left.float().unsqueeze(-1)).sum(dim=1)
        cnt_right = total_cnt - cnt_left
        thresh = torch.where((cnt_left < 1.0) | (cnt_right < 1.0), dot_mu_v, thresh)

        left_f = (proj < thresh.gather(1, assign)).float().unsqueeze(-1)
        O_left = O * left_f
        O_right = O - O_left
        cnt_left = O_left.sum(dim=1)
        cnt_right = total_cnt - cnt_left

        C_left = F.normalize(torch.bmm(O_left.transpose(1, 2), X_sub) / cnt_left.clamp(min=1e-6).unsqueeze(-1), p=2, dim=-1)
        C_right = F.normalize(torch.bmm(O_right.transpose(1, 2), X_sub) / cnt_right.clamp(min=1e-6).unsqueeze(-1), p=2, dim=-1)
        valid = (cnt_left >= 1.0) & (cnt_right >= 1.0)

        delta_sse = (cnt_left * cnt_right / (cnt_left + cnt_right).clamp(min=1e-6)) * (C_left - C_right).square().sum(dim=-1)
        delta_sse = torch.where(valid, delta_sse, torch.zeros_like(delta_sse))
        ranking_score = delta_sse * torch.where(has_valley, valley_boost, 1.0)

        _, top_ids = torch.topk(ranking_score, b, dim=-1, largest=True)
        ok = valid.gather(1, top_ids).unsqueeze(-1)
        top_ids_d = top_ids.unsqueeze(-1).expand(-1, -1, D)

        C_l_sel = torch.where(ok, C_left.gather(1, top_ids_d), C_curr.gather(1, top_ids_d))
        C_r_sel = torch.where(ok, C_right.gather(1, top_ids_d), X_sub[:, :b, :])
        V_sel = V_star.gather(1, top_ids_d)

        C_left_upd = C_curr.scatter(1, top_ids_d, C_l_sel)
        V_left_upd = pv.scatter(1, top_ids_d, V_sel)
        if m + b < k:
            C_buf = torch.cat([C_left_upd, C_r_sel, C_buf[:, m + b :, :]], dim=1)
            prev_V = torch.cat([V_left_upd, V_sel, prev_V[:, m + b :, :]], dim=1)
        else:
            C_buf = torch.cat([C_left_upd, C_r_sel], dim=1)
            prev_V = torch.cat([V_left_upd, V_sel], dim=1)

    return C_buf.to(orig_dtype)


# =====================================================================
# 2. Causal-Windowed Attention (Gather vs BlockMask.from_kv_blocks Flex)
# =====================================================================

class CausalWindowedKAttention(torch.nn.Module):
    def __init__(
        self,
        cfg: hnet4.HNetConfig,
        win_size: int = 32,
        update_every: int = 8,
        use_l2_norm: bool = True,
        update_mode: str = "ue8",
        v11_sub_n: int = 1024,
        use_flex: bool = False,
        two_stage_tok: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.head_dim = cfg.dim_l3 // cfg.n_head
        self.win_size = win_size
        self.update_every = update_every
        self.use_l2_norm = use_l2_norm
        self.update_mode = update_mode
        self.v11_sub_n = v11_sub_n
        self.use_flex = use_flex
        self.two_stage_tok = two_stage_tok
        self._step_cnt = 0
        self._last_refresh = 0

        if self.use_flex:
            assert cfg.emeans_block_size >= 16 and (cfg.emeans_block_size & (cfg.emeans_block_size - 1)) == 0, (
                f"FlexAttention requires bs >= 16 power of 2, got {cfg.emeans_block_size}"
            )

        self.q_attn = torch.nn.Linear(cfg.dim_l3, cfg.dim_l3, bias=False)
        self.k_attn = torch.nn.Linear(cfg.dim_l3, cfg.dim_l3, bias=False)
        self.v_attn = torch.nn.Linear(cfg.dim_l3, cfg.dim_l3, bias=False)
        self.c_proj = torch.nn.Linear(cfg.dim_l3, cfg.dim_l3, bias=False)
        self.resid_dropout = torch.nn.Dropout(cfg.dropout)

        init_c = F.normalize(
            torch.randn(self.n_head, cfg.emeans_clusters, self.head_dim) * 0.02,
            p=2,
            dim=-1,
        )
        self.register_buffer("C_heads", init_c)

        if use_l2_norm:
            self.logit_scale = torch.nn.Parameter(torch.tensor(math.log(14.0)))
        else:
            self.qm = torch.nn.Parameter(torch.ones(self.head_dim).normal_(mean=1.0, std=0.1))

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
            C_seed = hnet4.static_engine_v11_heads(
                x_heads,
                self.cfg.emeans_clusters,
                sub_n=self.v11_sub_n,
                robust_cap_q=self.cfg.robust_cap_q,
                spherical=True,
            )
        else:
            C_seed = self.C_heads.float()

        C_next = hnet4._damped_lloyd_step_heads(
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

        if self.use_l2_norm:
            q_eff = F.normalize(q, p=2, dim=-1)
            k_eff = F.normalize(k, p=2, dim=-1)
            scale = self.logit_scale.exp().clamp(max=50.0)
        else:
            q_eff = q * (math.log(max(T, 2)) * self.qm)
            k_eff = k
            scale = 1.0 / math.sqrt(D)

        # 1. Assign & sort k_eff by (cluster, k_mag) inside each W-window
        if self.cfg.emeans_clusters > 1:
            k_mag = k.float().norm(dim=-1).view(B, H, N_W, W)
            k_n = k_eff if self.use_l2_norm else F.normalize(k_eff, p=2, dim=-1)

            if self.training:
                if self.update_mode == "v11_every":
                    with torch.no_grad():
                        x_heads = torch.nan_to_num(
                            k_n.detach().float().transpose(0, 1).reshape(H, B * T, D),
                            nan=0.0,
                            posinf=1e4,
                            neginf=-1e4,
                        )
                        C_v11 = compiled_v11_batched_spherical(
                            x_heads, k=self.cfg.emeans_clusters, sub_n=self.v11_sub_n
                        )
                        C_next = hnet4._damped_lloyd_step_heads(
                            x_heads, C_v11, damp_scale=2.0, spherical=True
                        ).to(self.C_heads.dtype)
                        self.C_heads.copy_(C_next)
                    C_use = C_next.to(k_n.dtype)
                else:
                    self._maybe_update_centroids(k_n)
                    C_use = self.C_heads.to(k_n.dtype)
            else:
                C_use = self.C_heads.to(k_n.dtype)

            assign_k = torch.argmax(torch.einsum("bhtd,hmd->bhtm", k_n, C_use), dim=-1)
            assign_w = assign_k.view(B, H, N_W, W)
            mag_rank = k_mag / (k_mag.amax(dim=-1, keepdim=True) + 1.0)
            perm_w = torch.argsort(assign_w.float() * 2.0 + (1.0 - mag_rank), dim=-1, stable=True)

            perm_kv = perm_w.unsqueeze(-1).expand(-1, -1, -1, -1, D)
            k_s = torch.gather(k_eff.view(B, H, N_W, W, D), 3, perm_kv).view(B, H, N_B, bs, D)
            v_s = torch.gather(v.view(B, H, N_W, W, D), 3, perm_kv).view(B, H, N_B, bs, D)
        else:
            k_s = k_eff.view(B, H, N_B, bs, D)
            v_s = v.view(B, H, N_B, bs, D)

        block_C = k_s.mean(dim=3)

        # -----------------------------------------------------------------
        # 2A. FlexAttention Path (Q-block Stage-1 + BlockMask.from_kv_blocks)
        # -----------------------------------------------------------------
        if self.use_flex:
            q_blk = q_eff.view(B, H, N_B, bs, D).mean(dim=3)  # (B, H, N_B, D)
            router_logits = torch.matmul(q_blk, block_C.transpose(-2, -1)) * scale

            block_win = (torch.arange(N_B, device=x.device) // B_W).view(1, 1, 1, N_B)
            q_blk_win_1d = torch.arange(N_B, device=x.device) // B_W
            past_win_mask = block_win < q_blk_win_1d.view(1, 1, N_B, 1)

            off_logits = router_logits.masked_fill(~past_win_mask, float("-inf"))
            off_mask = hnet4.TopKHot.apply(off_logits, k_off, self.cfg.topk_mode)
            _, off_idx = torch.topk(off_logits, k=k_off, dim=-1)
            off_valid = torch.gather(off_mask, -1, off_idx) > 0

            # Concat [chronological local k_eff, window-sorted k_s] -> KV_LEN = 2 * T
            k_cat = torch.cat([k_eff, k_s.reshape(B, H, T, D)], dim=2)
            v_cat = torch.cat([v,     v_s.reshape(B, H, T, D)], dim=2)

            # Partial blocks: only the diagonal block per Q-block (needs causal mask_mod)
            kv_num_blocks = torch.ones((B, H, N_B), device=x.device, dtype=torch.int32)
            kv_indices = F.pad(
                torch.arange(N_B, device=x.device, dtype=torch.int32)
                .view(1, 1, N_B, 1)
                .expand(B, H, N_B, 1),
                (0, 2 * N_B - 1),
                value=0,
            ).contiguous()

            # Full blocks (skip mask_mod): prior blocks in current W-window + selected past sorted blocks
            past_cand = (N_B + off_idx).to(torch.int32)
            past_valid = (q_blk_win_1d >= 1).view(1, 1, N_B, 1) & off_valid
            if B_W > 1:
                loc_off = torch.arange(B_W - 1, device=x.device, dtype=torch.int32)
                loc_cand = (q_blk_win_1d.to(torch.int32) * B_W).unsqueeze(-1) + loc_off.unsqueeze(0)
                loc_valid = (
                    (loc_cand < torch.arange(N_B, device=x.device, dtype=torch.int32).unsqueeze(-1))
                    .view(1, 1, N_B, B_W - 1)
                    .expand(B, H, -1, -1)
                )
                loc_cand_bh = loc_cand.view(1, 1, N_B, B_W - 1).expand(B, H, -1, -1)
                all_cand = torch.cat([loc_cand_bh, past_cand], dim=-1)
                all_valid = torch.cat([loc_valid, past_valid], dim=-1)
            else:
                all_cand = past_cand
                all_valid = past_valid

            full_kv_num_blocks = all_valid.sum(dim=-1, dtype=torch.int32)
            cand_packed = torch.where(all_valid, all_cand, torch.full_like(all_cand, 2 * N_B))
            full_kv_indices, _ = torch.sort(cand_packed, dim=-1)
            full_kv_indices = torch.where(full_kv_indices >= 2 * N_B, 0, full_kv_indices)
            full_kv_indices = F.pad(
                full_kv_indices,
                (0, 2 * N_B - full_kv_indices.size(-1)),
                value=0,
            ).contiguous()

            def causal_local_mask(b, h, q_idx, kv_idx):
                return (kv_idx >= T) | (q_idx >= kv_idx)

            block_mask = BlockMask.from_kv_blocks(
                kv_num_blocks=kv_num_blocks,
                kv_indices=kv_indices,
                full_kv_num_blocks=full_kv_num_blocks,
                full_kv_indices=full_kv_indices,
                BLOCK_SIZE=(bs, bs),
                mask_mod=causal_local_mask,
                seq_lengths=(T, 2 * T),
            )

            q_flex = (q_eff * (scale * math.sqrt(D))).to(k_cat.dtype)
            flex_opts = {
                "BACKEND": "TRITON",
                "fwd_BLOCK_M": bs,
                "fwd_BLOCK_N": bs,
                "bwd_BLOCK_M1": bs,
                "bwd_BLOCK_N1": bs,
                "bwd_BLOCK_M2": bs,
                "bwd_BLOCK_N2": bs,
            }
            out = flex_attention(q_flex, k_cat, v_cat, block_mask=block_mask, kernel_options=flex_opts)

            # STE gradient back into router_logits / block_C
            #has_any = past_win_mask.any(dim=-1, keepdim=True)
            #safe_r = router_logits.masked_fill(~past_win_mask, -1e9)
            #s = (off_mask * F.softmax(safe_r, dim=-1)).sum(dim=-1, keepdim=True) * has_any.to(q.dtype)
            #out = (out.view(B, H, N_B, bs, D) * (1.0 + s - s.detach()).unsqueeze(3)).view(B, H, T, D)
            return self.resid_dropout(self.c_proj(out.transpose(1, 2).contiguous().view(B, T, C_dim)))

        # -----------------------------------------------------------------
        # 2B. Gather Path (No-Flex Baseline with Identical W, bs, k_off)
        # -----------------------------------------------------------------
        router_logits = torch.matmul(q_eff, block_C.transpose(-2, -1)) * scale

        block_win = (torch.arange(N_B, device=x.device) // B_W).view(1, 1, 1, N_B)
        q_win = (torch.arange(T, device=x.device) // W).view(1, 1, T, 1)
        past_win_mask = block_win < q_win

        off_logits = router_logits.masked_fill(~past_win_mask, float("-inf"))
        off_mask = hnet4.TopKHot.apply(off_logits, k_off, self.cfg.topk_mode)
        _, off_idx = torch.topk(off_logits, k=k_off, dim=-1)
        off_valid = torch.gather(off_mask, -1, off_idx)

        flat_idx = off_idx.view(B, H, T * k_off)
        gather_kv = flat_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, bs, D)
        k_glob = torch.gather(k_s, 2, gather_kv).view(B, H, T, k_off * bs, D)
        v_glob = torch.gather(v_s, 2, gather_kv).view(B, H, T, k_off * bs, D)
        glob_gate = off_valid.unsqueeze(-1).expand(-1, -1, -1, -1, bs).reshape(B, H, T, k_off * bs)

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

        scores = torch.einsum("bhtd,bhtkd->bhtk", q_eff, k_g) * scale
        scores = scores.masked_fill(~full_mask, float("-inf"))

        if self.two_stage_tok:
            tok_mask = hnet4.TopKHot.apply(
                scores, min(self.cfg.k_sparse_tok, scores.size(-1)), self.cfg.topk_mode
            )
            probs = F.softmax(scores, dim=-1) * tok_mask * gate
        else:
            probs = F.softmax(scores, dim=-1) * gate

        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)

        out = torch.einsum("bhtk,bhtkd->bhtd", probs, v_g)
        return self.resid_dropout(self.c_proj(out.transpose(1, 2).contiguous().view(B, T, C_dim)))


def build_cw_model(
    cfg: hnet4.HNetConfig,
    win_size: int = 32,
    update_every: int = 8,
    use_l2_norm: bool = True,
    update_mode: str = "ue8",
    v11_sub_n: int = 1024,
    use_flex: bool = False,
    two_stage_tok: bool = True,
) -> hnet4.HNet:
    model = hnet4.HNet(vocab_size=hnet4.vocab_size, cfg=cfg).to(hnet4.device)
    for block in model.inner:
        old_attn = block.attn
        new_attn = CausalWindowedKAttention(
            cfg,
            win_size=win_size,
            update_every=update_every,
            use_l2_norm=use_l2_norm,
            update_mode=update_mode,
            v11_sub_n=v11_sub_n,
            use_flex=use_flex,
            two_stage_tok=two_stage_tok,
        ).to(hnet4.device)
        new_attn.load_state_dict(old_attn.state_dict(), strict=False)
        block.attn = new_attn
    return model


# =====================================================================
# 3. Optimal Token-Overlap Hungarian Alignment & Oracle Audit
# =====================================================================

def _solve_max_assignment(mat_np):
    r, c = linear_sum_assignment(-mat_np)
    return r.tolist(), c.tolist()


@torch.no_grad()
def token_overlap_agreement(X_heads: torch.Tensor, C_old: torch.Tensor, C_new: torch.Tensor):
    H, N, _ = X_heads.shape
    M = C_old.shape[1]

    X_n = F.normalize(X_heads.float(), dim=-1)
    Co = F.normalize(C_old.float(), dim=-1)
    Cn = F.normalize(C_new.float(), dim=-1)

    a_old = torch.argmax(torch.bmm(X_n, Co.transpose(1, 2)), dim=-1)
    a_new = torch.argmax(torch.bmm(X_n, Cn.transpose(1, 2)), dim=-1)

    flat_idx = a_old * M + a_new
    overlap = torch.zeros(H, M * M, device=X_heads.device, dtype=torch.float32)
    overlap.scatter_add_(1, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
    overlap = overlap.view(H, M, M).cpu().numpy()

    asis_acc, hung_acc = [], []
    for h in range(H):
        asis_acc.append(float(overlap[h].trace()) / N)
        r_ind, c_ind = _solve_max_assignment(overlap[h])
        hung_acc.append(float(overlap[h][r_ind, c_ind].sum()) / N)

    return sum(asis_acc) / H, sum(hung_acc) / H


@torch.no_grad()
def eval_cw_router_oracle(
    q_eff: torch.Tensor,
    k_eff: torch.Tensor,
    k_mag: torch.Tensor,
    scale: float,
    C_use: torch.Tensor | None,
    win_size: int,
    bs: int,
    k_blocks: int,
    oracle_probs: torch.Tensor,
    oracle_top4: torch.Tensor,
):
    B, H, T, D = k_eff.shape
    W = win_size
    N_W, B_W = T // W, W // bs
    N_B = N_W * B_W
    k_off = max(1, min(k_blocks - 1, N_B - B_W))

    if C_use is None:
        perm_w = torch.arange(W, device=k_eff.device).view(1, 1, 1, W).expand(B, H, N_W, W)
    else:
        k_n = F.normalize(k_eff, p=2, dim=-1)
        assign_w = torch.argmax(
            torch.einsum("bhtd,hmd->bhtm", k_n, C_use.to(k_n.dtype)), dim=-1
        ).view(B, H, N_W, W)
        k_mag_w = k_mag.view(B, H, N_W, W)
        mag_rank = k_mag_w / (k_mag_w.amax(dim=-1, keepdim=True) + 1.0)
        perm_w = torch.argsort(assign_w.float() * 2.0 + (1.0 - mag_rank), dim=-1, stable=True)

    win_base = (torch.arange(N_W, device=k_eff.device) * W).view(1, 1, N_W, 1)
    orig_pos_w = (perm_w + win_base).view(B, H, N_B, bs)

    perm_kv = perm_w.unsqueeze(-1).expand(-1, -1, -1, -1, D)
    k_s = torch.gather(k_eff.view(B, H, N_W, W, D), 3, perm_kv).view(B, H, N_B, bs, D)
    block_C = k_s.mean(dim=3)
    router_logits = torch.matmul(q_eff, block_C.transpose(-2, -1)) * scale

    block_win = (torch.arange(N_B, device=k_eff.device) // B_W).view(1, 1, 1, N_B)
    q_win = (torch.arange(T, device=k_eff.device) // W).view(1, 1, T, 1)
    off_logits = router_logits.masked_fill(~(block_win < q_win), float("-inf"))
    _, off_idx = torch.topk(off_logits, k=k_off, dim=-1)
    off_valid_sel = torch.gather(off_logits, -1, off_idx) > float("-inf")

    flat_idx = off_idx.view(B, H, T * k_off)
    orig_k_glob = torch.gather(
        orig_pos_w, 2, flat_idx.unsqueeze(-1).expand(-1, -1, -1, bs)
    ).view(B, H, T, k_off * bs)
    glob_valid_tok = off_valid_sel.unsqueeze(-1).expand(-1, -1, -1, -1, bs).reshape(B, H, T, k_off * bs)

    q_win_1d = torch.arange(T, device=k_eff.device) // W
    loc_pos = (
        (q_win_1d * W).view(T, 1) + torch.arange(W, device=k_eff.device).view(1, W)
    ).view(1, 1, T, W).expand(B, H, -1, -1)
    local_causal = (torch.arange(T, device=k_eff.device).view(1, 1, T, 1) >= loc_pos)

    full_mask = torch.cat([local_causal, glob_valid_tok], dim=-1)
    cand_pos = torch.cat([loc_pos, orig_k_glob], dim=-1)

    gathered_probs = torch.gather(oracle_probs[:, :, 4:, :], -1, cand_pos[:, :, 4:, :].clamp(min=0))
    mass_captured = (gathered_probs * full_mask[:, :, 4:, :].float()).sum(dim=-1).mean().item()

    cand_masked = cand_pos[:, :, 4:, :].masked_fill(~full_mask[:, :, 4:, :], -1)
    hits = (oracle_top4.unsqueeze(-1) == cand_masked.unsqueeze(-2)).any(dim=-1)
    return hits.float().mean().item(), mass_captured


@torch.no_grad()
def audit_v11_layer(attn_mod, x_in: torch.Tensor) -> dict:
    B, T, _ = x_in.shape
    H, D = attn_mod.n_head, attn_mod.head_dim
    M = attn_mod.cfg.emeans_clusters
    bs = attn_mod.cfg.emeans_block_size
    W = attn_mod.win_size

    q = attn_mod.q_attn(x_in).view(B, T, H, D).transpose(1, 2)
    k = attn_mod.k_attn(x_in).view(B, T, H, D).transpose(1, 2)
    k_mag = k.float().norm(dim=-1)

    if attn_mod.use_l2_norm:
        q_eff = F.normalize(q, p=2, dim=-1)
        k_eff = F.normalize(k, p=2, dim=-1)
        scale = attn_mod.logit_scale.exp().clamp(max=50.0).item()
    else:
        q_eff = q * (math.log(max(T, 2)) * attn_mod.qm)
        k_eff = k
        scale = 1.0 / math.sqrt(D)

    x_heads = k_eff.float().transpose(0, 1).reshape(H, B * T, D)
    X_n = F.normalize(x_heads, dim=-1)
    C_tr = F.normalize(attn_mod.C_heads.float(), dim=-1)

    C_fr_25 = hnet4.static_engine_v11_heads(
        x_heads, M, sub_n=4096, valley_boost=2.5,
        robust_cap_q=attn_mod.cfg.robust_cap_q, spherical=True,
    )
    C_fr_10 = hnet4.static_engine_v11_heads(
        x_heads, M, sub_n=4096, valley_boost=1.0,
        robust_cap_q=attn_mod.cfg.robust_cap_q, spherical=True,
    )

    def quant_stats(C):
        sim, assign = torch.bmm(X_n, F.normalize(C, dim=-1).transpose(1, 2)).max(dim=-1)
        loss = (1.0 - sim).clamp(min=0.0).mean(dim=-1)
        cnt = torch.zeros(H, M, device=x_in.device).scatter_add_(
            1, assign, torch.ones_like(assign, dtype=torch.float32)
        )
        return loss, cnt

    loss_tr, cnt_tr = quant_stats(C_tr)
    loss_fr25, cnt_fr25 = quant_stats(C_fr_25)
    _, cnt_fr10 = quant_stats(C_fr_10)

    p_tr = (cnt_tr / (B * T)).clamp(min=1e-9)
    ent_tr = (-(p_tr * p_tr.log()).sum(dim=-1) / math.log(M)).mean().item()
    alive_tr = (cnt_tr >= 0.01 * (B * T)).sum(dim=-1).float().mean().item()
    top1_tr = (cnt_tr.max(dim=-1).values / (B * T)).mean().item()
    top1_fr25 = (cnt_fr25.max(dim=-1).values / (B * T)).mean().item()
    top1_fr10 = (cnt_fr10.max(dim=-1).values / (B * T)).mean().item()

    C_al_cos = hnet4._align_centroids(C_tr, C_fr_25)
    cos_ovl, hung_ovl = token_overlap_agreement(x_heads, C_tr, C_al_cos)
    cos_diag = (F.normalize(C_tr, dim=-1) * F.normalize(C_al_cos, dim=-1)).sum(dim=-1).mean().item()

    causal = torch.tril(torch.ones(T, T, device=x_in.device, dtype=torch.bool)).view(1, 1, T, T)
    oracle_logits = (torch.matmul(q_eff, k_eff.transpose(-2, -1)) * scale).masked_fill(~causal, float("-inf"))
    oracle_probs = F.softmax(oracle_logits, dim=-1)
    _, oracle_top4 = torch.topk(oracle_logits[:, :, 4:, :], k=4, dim=-1)

    rec_tr, mass_tr = eval_cw_router_oracle(q_eff, k_eff, k_mag, scale, C_tr, W, bs, attn_mod.cfg.k_blocks, oracle_probs, oracle_top4)
    rec_fr25, _ = eval_cw_router_oracle(q_eff, k_eff, k_mag, scale, C_fr_25, W, bs, attn_mod.cfg.k_blocks, oracle_probs, oracle_top4)
    rec_fr10, _ = eval_cw_router_oracle(q_eff, k_eff, k_mag, scale, C_fr_10, W, bs, attn_mod.cfg.k_blocks, oracle_probs, oracle_top4)
    rec_seq, mass_seq = eval_cw_router_oracle(q_eff, k_eff, k_mag, scale, None, W, bs, attn_mod.cfg.k_blocks, oracle_probs, oracle_top4)

    return {
        "drift": (loss_tr / loss_fr25.clamp(min=1e-6)).mean().item(),
        "alive": alive_tr,
        "M": M,
        "min_cnt": int(cnt_tr.min().item()),
        "max_cnt": int(cnt_tr.max().item()),
        "ent": ent_tr,
        "top1_tr": top1_tr,
        "top1_fr25": top1_fr25,
        "top1_fr10": top1_fr10,
        "cos_diag": cos_diag,
        "cos_ovl": cos_ovl,
        "hung_ovl": hung_ovl,
        "rec_tr": rec_tr,
        "mass_tr": mass_tr,
        "rec_fr25": rec_fr25,
        "rec_fr10": rec_fr10,
        "rec_seq": rec_seq,
        "mass_seq": mass_seq,
    }


@torch.no_grad()
def audit_model(name: str, raw_model: hnet4.HNet, vx: torch.Tensor):
    raw_model.eval()
    b, t = vx.shape
    positions = torch.arange(0, t, dtype=torch.long, device=vx.device).unsqueeze(0)
    x = hnet4.utils.rms_norm(raw_model.tok_emb(vx) + raw_model.pos_emb(positions))
    for layer in raw_model.sLayers:
        x, _ = layer.abstract_up(x)

    reports = []
    for l_idx, block in enumerate(raw_model.inner):
        x_in = block.ln_1(x)
        if isinstance(block.attn, CausalWindowedKAttention) and block.attn.cfg.emeans_clusters > 1:
            st = audit_v11_layer(block.attn, x_in)
            st["layer"] = l_idx
            reports.append(st)
        x = block(x)
    raw_model.train()

    if not reports:
        return
    print(f"  [AUDIT: {name}] (inner T={x.shape[1]}, B*T={b*x.shape[1]})")
    print(
        "    L | drift | alive   min..max  top1(tr/2.5/1.0)  ent  | cos_sim  cos_ovl/hung | "
        "Top4Rec(tr / fr2.5 / fr1.0 / seq) | Mass(tr / seq)"
    )
    for r in reports:
        print(
            f"    {r['layer']} | {r['drift']:5.3f} | {r['alive']:4.1f}/{r['M']:<2d} "
            f"{r['min_cnt']:3d}..{r['max_cnt']:<4d} "
            f"{r['top1_tr']*100:4.1f}/{r['top1_fr25']*100:4.1f}/{r['top1_fr10']*100:4.1f}% "
            f"{r['ent']:4.2f} |  {r['cos_diag']:5.3f}   {r['cos_ovl']*100:5.1f}%/{r['hung_ovl']*100:5.1f}% | "
            f"  {r['rec_tr']*100:4.1f} / {r['rec_fr25']*100:4.1f}  / {r['rec_fr10']*100:4.1f}  / {r['rec_seq']*100:4.1f}%  | "
            f"{r['mass_tr']*100:4.1f} / {r['mass_seq']*100:4.1f}%"
        )


@torch.no_grad()
def evaluate_model_local(raw_model: hnet4.HNet, get_batch_fn) -> tuple[float, float]:
    raw_model.eval()
    tot_ce, tot_gen = 0.0, 0.0
    for _ in range(hnet4.eval_iters):
        vx, vy = get_batch_fn("val")
        _, pure_ce, gloss = raw_model(vx, vy)
        tot_ce += pure_ce.item()
        tot_gen += gloss.item()
    raw_model.train()
    return tot_ce / hnet4.eval_iters, tot_gen / hnet4.eval_iters


# =====================================================================
# 4. Benchmark + Audit Runner
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--audit-every", type=int, default=1000)
    parser.add_argument("--compile", action="store_true", help="Compile models (recommended for FlexAttention)")
    parser.add_argument("--block-size", type=int, default=1024, help="Outer seq len (inner scales //2 -> //4)")
    parser.add_argument("--batch-size", type=int, default=16, help="Match ~16384 tok/step (16 for block-size 1024)")
    parser.add_argument("--win-size", type=int, default=None, help="Causal window size W (default: 32 if T_inner>=256 else 8)")
    parser.add_argument("--bs", type=int, default=None, help="Block size inside window (default: 16 if T_inner>=256 else 4)")
    parser.add_argument("--k-blocks", type=int, default=4, help="Stage-1 blocks (k_off = k_blocks - 1)")
    parser.add_argument("--update-mode", type=str, default="ue8", choices=["ue8", "v11_every"])
    parser.add_argument("--v11-sub-n", type=int, default=1024, help="Subsample size for v11 per head")
    parser.add_argument("--no-stage2", action="store_true", help="Disable Stage-2 TopKHot on cw_noflex for 1:1 soft-attn parity")
    parser.add_argument("--no-kattn", action="store_true", help="Skip kattn_baseline and only compare cw_noflex vs cw_flex")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    device = hnet4.device

    k_l1 = args.block_size // 2
    k_l2 = args.block_size // 4
    bs = args.bs if args.bs is not None else (16 if k_l2 >= 256 else 4)
    win_size = args.win_size if args.win_size is not None else (2 * bs)
    assert k_l2 % win_size == 0 and win_size % bs == 0, (
        f"Need T_inner ({k_l2}) % win_size ({win_size}) == 0 and win_size % bs ({bs}) == 0"
    )

    get_batch_fn = hnet4.utils.get_get_batch(
        args.block_size, args.batch_size, hnet4.dataset, device, always_byte=True
    )

    base_kattn = replace(
        hnet4.EXPERIMENTS["kattn_baseline"][1],
        block_size=args.block_size,
        k_gather_l1=k_l1,
        k_gather_l2=k_l2,
    )
    base_cw = replace(
        hnet4.EXPERIMENTS["v11_l2_fixed"][1],
        block_size=args.block_size,
        k_gather_l1=k_l1,
        k_gather_l2=k_l2,
        emeans_block_size=bs,
        k_blocks=args.k_blocks,
        emeans_clusters=16,
    )

    specs = {}
    #if not args.no_kattn:
    #    specs["kattn_baseline"] = (base_kattn, None)
    #specs["cw_noflex"] = (
    #    base_cw,
    #    dict(
    #        win_size=win_size, update_every=8, use_l2_norm=True,
    #        update_mode=args.update_mode, v11_sub_n=args.v11_sub_n,
    #        use_flex=False, two_stage_tok=(not args.no_stage2),
    #    ),
    #)
    specs["cw_flex"] = (
        base_cw,
        dict(
            win_size=win_size, update_every=8, use_l2_norm=True,
            update_mode=args.update_mode, v11_sub_n=args.v11_sub_n,
            use_flex=True, two_stage_tok=False,
        ),
    )

    raw_models, models, opts = {}, {}, {}
    bx0, by0 = get_batch_fn("train")

    for name, (cfg, cw_kw) in specs.items():
        torch.manual_seed(1337)
        if cw_kw is not None:
            m = build_cw_model(cfg, **cw_kw)
        else:
            m = hnet4.HNet(hnet4.vocab_size, cfg).to(device)
        raw_models[name] = m
        opts[name] = torch.optim.AdamW(m.parameters(), lr=hnet4.learning_rate)
        if args.compile:
            t0 = time.perf_counter()
            cm = torch.compile(m)
            for _ in range(2):
                opts[name].zero_grad(set_to_none=True)
                loss, _, _ = cm(bx0, by0)
                loss.backward()
                opts[name].zero_grad(set_to_none=True)
            if device == "cuda":
                torch.cuda.synchronize()
            print(f"  [COMPILED] {name:<16} in {time.perf_counter() - t0:.2f}s")
            models[name] = cm
        else:
            models[name] = m

    print(
        f"Auditor on {device} | B={args.batch_size} seq={args.block_size}->{k_l1}->{k_l2} "
        f"(W={win_size}, bs={bs}, k_blk={args.k_blocks}, mode={args.update_mode}, stage2={not args.no_stage2}) | "
        f"steps={args.steps} | compile={args.compile}"
    )

    if args.steps == 0:
        vx, _ = get_batch_fn("val")
        for name, rm in raw_models.items():
            audit_model(name, rm, vx)
        return

    timings = {n: 0.0 for n in models}

    for it in range(args.steps):
        lr = hnet4.get_lr(it)
        bx, by = get_batch_fn("train")
        step = it + 1

        for name, m in models.items():
            opt = opts[name]
            for pg in opt.param_groups:
                pg["lr"] = lr
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            opt.zero_grad(set_to_none=True)
            loss, _, _ = m(bx, by)
            loss.backward()
            opt.step()

            if device == "cuda":
                torch.cuda.synchronize()
            timings[name] += (time.perf_counter() - t0) * 1000.0

        if step % args.audit_every == 0 or step == args.steps:
            ce_line = []
            for name, rm in raw_models.items():
                v_ce, v_gen = evaluate_model_local(rm, get_batch_fn)
                ms = timings[name] / args.audit_every
                timings[name] = 0.0
                ce_line.append(f"{name}: val_ce={v_ce:.4f} (gen={v_gen:.4f}, {ms:.1f}ms)")
            print(f"STEP {step:<5d} | " + " | ".join(ce_line))
            print("-" * 125)
            vx, _ = get_batch_fn("val")
            for name, rm in raw_models.items():
                audit_model(name, rm, vx)


if __name__ == "__main__":
    main()