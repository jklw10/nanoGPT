"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import inspect
import math
from dataclasses import dataclass

import torch
from cut_cross_entropy import linear_cross_entropy
from torch import nn
from torch.nn import functional as F

from models.curlfad_bench import UnifiedKreinEntropicOptimizer
from models.curlfadam import CurlFreeAdamW, RowWiseCurlFreeAdamW
from parts import modules, optim, quantizer, utils, wackmodel
from unit import bench

#todo
#prediction machine

#Linear = optim.OptimizedLinear
#Linear = nn.Linear
Linear = nn.Linear
#Linear = modules.TensionLinear
Plin = modules.ProbingLinear
#Linear = modules.HungryGdiffLinear
#Linear = modules.HungryVGMLinear
#Linear = modules.HungryLinear
Orthlin = modules.OrthogonalLinear
#settings :)
#intest

cw_v11          = True


cardnewt        = False   
rootkan         = False  
rootkan_highdeg = False
rootkan_pade    = False
rootkan_glu     = False

qrot_radial     = False
qrot_pre_rootkan= False
qrot_angle      = False


lmsa            = False
msa             = False
kratt           = False
kreinSSM        = False
soloblock       = False
loopblock       = False
spl             = False

ukb             = False

#good in ssm
posembless      = False
#good in shkspr-char
mix_squish      = False

assocssm        = False
#good in owt-char
qrot            = False
diffac          = False
rms_lnorm       = False
#good in owt
OU_proj         = False
emb_OU_noise    = False
emb_norm        = False
wnHead          = False 
qksink          = False
qkfft           = False
qkwn            = False
pattmlp         = False

hcmlp           = False
v_patt          = False
v_patt_2        = False
hcnvc           = False
hcnHead         = False
testing_mlp     = False
#bad
mem_mix_squish  = False #or fftmem
causal_mem      = False
convemb         = False
diffembd        = False
qope            = False
side_squish     = False
#slow
think           = False
repeat_block    = False
repeat_center   = False

res_attn        = False
qrottention     = False

symloss         = False
midchaos        = False

specl           = False
topoloss        = False
acebtl          = False
acl1            = False # unimplemented

zeroinit        = False
#ungraded
fftmem          = False
qmem            = False
mmnorm          = False
normless        = False
orthqkvc        = False
orthHead        = False

plin            = False
nmix            = False
losspred        = False

dynskip         = False
k_atten         = False
h_k_atten       = False
p_atten         = False
sprs            = False
wnll            = False

csbr            = False

posembdrop      = False
exposemb        = False

residless       = False

Non_causal      = False

residcomp       = False
mhc_lr          = 0.01
mlp_ptok        = 4096
v_ptok          = 256
#known good
ssmax           = True
mtp             = False
headChange      = hcnHead or orthHead#  wnHead reparms apparently work?
CCE             = False and not headChange

auxmod          = acl1 or acebtl #or topoloss

blockin         = spl or symloss or midchaos or topoloss or specl or dynskip


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.n_embd))
        self.bias = nn.Parameter(torch.zeros(config.n_embd)) if config.bias else None
       

    def forward(self, input):
        #if(tests):
        #    return input
        if rms_lnorm:
            return self.weight * utils.rms_norm(input)
        if(normless):
            return input
        if(mmnorm):
            return utils.soft_range_clip_norm(input, dim=-1, scale=self.weight) 
            #return torch.tanh(input)*self.weight.norm(p=2,keepdim=True)
           # return utils.acwrap(input*self.weight, torch.tanh, self.bias)
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

def get_dct_matrix(N):
    dct_m = torch.zeros(N, N)
    for k in range(N):
        for n in range(N):
            weight = math.cos((math.pi / N) * (n + 0.5) * k)
            if k == 0:
                weight *= math.sqrt(1 / N)
            else:
                weight *= math.sqrt(2 / N)
            dct_m[k, n] = weight
    return dct_m

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        
        self.q_attn = (Linear(config.n_embd, config.n_embd, bias=config.bias))
        self.k_attn = (Linear(config.n_embd, config.n_embd, bias=config.bias))
        if v_patt_2:
            self.v_attn = modules.PattLayer(config.n_embd,  config.n_embd, v_ptok)# bias=config.bias)
            self.v_attn2 = Linear(config.n_embd,  config.n_embd)# bias=config.bias)
        
        
        if v_patt:
            self.v_attn = modules.PattLayer(config.n_embd,  config.n_embd, v_ptok)# bias=config.bias)
        else:
            self.v_attn = Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=config.bias)
        


        if hcnvc:
            self.v_attn = modules.HCLinear(config.n_embd, config.n_embd, 1.0, False)
            self.c_proj = modules.HCLinear(config.n_embd, config.n_embd, 1.0, False)
        

        if orthqkvc:
            self.q_attn = Orthlin(config.n_embd, config.n_embd, bias=config.bias)
            self.k_attn = Orthlin(config.n_embd, config.n_embd, bias=config.bias)
            self.v_attn = Orthlin(config.n_embd, config.n_embd, bias=config.bias)
            self.c_proj = Orthlin(config.n_embd, config.n_embd, bias=config.bias)
        
        if plin:
            self.q_attn = Plin(config.n_embd, config.n_embd, bias=config.bias)
            self.k_attn = Plin(config.n_embd, config.n_embd, bias=config.bias)
            self.v_attn = Plin(config.n_embd, config.n_embd, bias=config.bias)
            self.c_proj = Plin(config.n_embd, config.n_embd, bias=config.bias)
        
        if qkwn:
            self.q_attn = wn(self.q_attn)
            self.k_attn = wn(self.k_attn)
            #self.q_attn = modules.HCLinear(config.n_embd, config.n_embd, 1.0, True)
            #self.k_attn = modules.HCLinear(config.n_embd, config.n_embd, 1.0, True)
        
        if qksink:
            self.energy_bind1 = modules.SinkhornLinear(config.n_embd)
            self.energy_bind2 = modules.SinkhornLinear(config.n_embd)
        
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head 
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim =  self.n_embd //  self.n_head 
        
        self.qm = nn.Parameter(torch.ones(self.head_dim).normal_(mean=1, std=0.1))
    
    def apply_fft_processing(self, x, sinkhorn_layer):
        x_f = torch.fft.rfft(x.float(), n=x.shape[-1], dim=-1, norm='ortho')
        
        mag = x_f.abs() + 1e-8
        phase = x_f.angle()
        
        norm_mag= utils.rms_norm(mag,dim=-1)
        processed_mag = sinkhorn_layer(norm_mag)
        processed_complex = processed_mag * torch.exp(torch.complex(torch.zeros_like(phase),phase))

        return torch.fft.irfft(processed_complex, n=x.shape[-1], dim=-1, norm='ortho')
    def forward(self, x, causal=True):
        B, T, C = x.size() 
        
        qkx= x
        if qkfft:
            qkx = torch.fft.fft(x, dim=-1).abs()
            qkx = utils.rms_norm(qkx)

        if qksink:
            q = (self.q_attn(self.energy_bind1(qkx)))
            k = (self.k_attn(self.energy_bind2(qkx)))
        else:
            q = self.q_attn(qkx)
            k = self.k_attn(qkx)
            
        v = (self.v_attn(x))
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)

        if ssmax:
            q_scale = math.log(T)*self.qm
        else:
            q_scale = 1.0
        y = torch.nn.functional.scaled_dot_product_attention(q*q_scale,k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal= causal)
        #y2 = torch.nn.functional.scaled_dot_product_attention(q*q_scale,k, v2, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal= causal)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) 
        if v_patt_2:
            v2 = (self.v_attn2(x))
            y=y+v2
        
        y = self.resid_dropout(self.c_proj(y))

        return y



class _TopKHot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k, mode="dense"):
        k = min(k, x.size(-1))
        _, indices = torch.topk(x, k=k, dim=-1)
        active_mask = torch.zeros_like(x).scatter_(-1, indices, 1.0)
        valid_positions = (x > float("-inf")).to(x.dtype)
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
            score_target = -g.masked_fill(valid_positions == 0, float("-inf"))
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


@torch.no_grad()
def _v11_batched_spherical(
    X_heads: torch.Tensor,
    k: int = 16,
    sub_n: int = 1024,
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
        X_sub = X_in[:, torch.randperm(S, device=dev), :]

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
        _, assign = torch.max(torch.bmm(X_sub, C_curr.transpose(1, 2)), dim=-1)

        O = torch.zeros(H, S, m, device=dev, dtype=torch.float32).scatter_(
            2, assign.unsqueeze(-1), 1.0
        )
        total_cnt = O.sum(dim=1)
        w_sum = total_cnt.clamp(min=1e-6)
        M = torch.bmm(O.transpose(1, 2), X_sub) / w_sum.unsqueeze(-1)

        pv = prev_V[:, :m, :]
        V1 = torch.randn(H, m, D, device=dev, dtype=torch.float32)
        V1 = F.normalize(V1 - (V1 * pv).sum(dim=-1, keepdim=True) * pv, p=2, dim=-1, eps=1e-9)

        V2 = torch.randn(H, m, D, device=dev, dtype=torch.float32)
        V2 = F.normalize(V2 - (V2 * V1).sum(dim=-1, keepdim=True) * V1, p=2, dim=-1, eps=1e-9)

        for _ in range(3):
            V_both = torch.cat([V1, V2], dim=1)
            dots = torch.bmm(X_sub, V_both.transpose(1, 2))
            dot1 = dots[:, :, :m].gather(2, assign.unsqueeze(-1)).squeeze(-1)
            dot2 = dots[:, :, m:].gather(2, assign.unsqueeze(-1)).squeeze(-1)

            z1 = dot1 - (M * V1).sum(dim=-1).gather(1, assign)
            z2 = dot2 - (M * V2).sum(dim=-1).gather(1, assign)

            Z1_mat = O * z1.unsqueeze(-1)
            Z2_mat = O * z2.unsqueeze(-1)
            V_next_both = torch.bmm(torch.cat([Z1_mat, Z2_mat], dim=2).transpose(1, 2), X_sub) - torch.cat([
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
        ranking_score = torch.where(valid, delta_sse, torch.zeros_like(delta_sse)) * torch.where(has_valley, valley_boost, 1.0)

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


@torch.no_grad()
def _damped_lloyd_spherical(x_heads: torch.Tensor, C_heads: torch.Tensor, damp_scale: float = 2.0) -> torch.Tensor:
    H, N, _ = x_heads.shape
    M = C_heads.shape[1]
    x_f32 = F.normalize(x_heads.float(), p=2, dim=-1)
    C_f32 = C_heads.float()

    max_score, assign = torch.max(torch.bmm(x_f32, C_f32.transpose(1, 2)), dim=-1)
    dist_sq = torch.clamp(1.0 - max_score, min=0.0)

    step = max(1, N // 1024)
    lam = torch.clamp(damp_scale * torch.median(dist_sq[:, ::step], dim=1, keepdim=True).values, min=1e-4)
    w = lam / (dist_sq + lam)

    O_w = torch.zeros(H, N, M, device=x_heads.device, dtype=torch.float32)
    O_w.scatter_(2, assign.unsqueeze(-1), w.unsqueeze(-1))

    counts = O_w.sum(dim=1).unsqueeze(-1)
    new_C = F.normalize(torch.bmm(O_w.transpose(1, 2), x_f32) / torch.clamp(counts, min=1e-6), p=2, dim=-1)
    return torch.where(counts < 1e-6, C_f32, new_C).to(C_heads.dtype)


class CausalWindowedV11Attention(nn.Module):
    """Drop-in NanoGPT attention: Causal-Windowed v11 + Stage-2 Kattention (cw_l2)."""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        bias = getattr(config, "bias", False)
        dropout = getattr(config, "dropout", 0.0)

        # Optional config overrides (safe defaults match cw_l2_m16)
        self.win_size = getattr(config, "cw_win_size", 8)
        self.bs = getattr(config, "cw_block_size", 4)
        self.k_blocks = getattr(config, "cw_k_blocks", 4)
        self.k_sparse_tok = getattr(config, "cw_k_sparse_tok", 4)
        self.n_clusters = getattr(config, "cw_n_clusters", 16)
        self.update_every = getattr(config, "cw_update_every", 8)
        self.refresh_interval = getattr(config, "cw_refresh_interval", 50)
        self.sub_n = getattr(config, "cw_sub_n", 1024)
        self.update_mode = getattr(config, "cw_update_mode", "ue8")  # "ue8" | "v11_every"
        self.topk_mode = getattr(config, "cw_topk_mode", "dense")
        assert self.win_size % self.bs == 0

        self._step_cnt = 0
        self._last_refresh = 0

        self.q_attn = nn.Linear(self.n_embd, self.n_embd, bias=bias)
        self.k_attn = nn.Linear(self.n_embd, self.n_embd, bias=bias)
        self.v_attn = nn.Linear(self.n_embd, self.n_embd, bias=bias)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)

        self.logit_scale = nn.Parameter(torch.tensor(math.log(14.0)))
        init_c = F.normalize(torch.randn(self.n_head, self.n_clusters, self.head_dim) * 0.02, p=2, dim=-1)
        self.register_buffer("C_heads", init_c)

    @torch._dynamo.disable
    @torch.no_grad()
    def _maybe_update_centroids(self, k_n: torch.Tensor):
        self._step_cnt += 1
        is_first = (self._step_cnt == 1)
        is_full_refresh = is_first or ((self._step_cnt - self._last_refresh) >= self.refresh_interval)
        if not is_full_refresh and (self._step_cnt % self.update_every != 0):
            return

        B, H, T, D = k_n.shape
        x_heads = torch.nan_to_num(
            k_n.detach().float().transpose(0, 1).reshape(H, B * T, D),
            nan=0.0, posinf=1e4, neginf=-1e4,
        )
        if is_full_refresh:
            self._last_refresh = self._step_cnt
            C_seed = _v11_batched_spherical(x_heads, k=self.n_clusters, sub_n=self.sub_n)
        else:
            C_seed = self.C_heads.float()

        C_next = _damped_lloyd_spherical(x_heads, C_seed, damp_scale=2.0).to(self.C_heads.dtype)
        self.C_heads.copy_(C_next)

    def forward(self, x, causal=True):
        B, T, C = x.size()
        H, D = self.n_head, self.head_dim
        W, bs = self.win_size, self.bs

        q = self.q_attn(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_attn(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_attn(x).view(B, T, H, D).transpose(1, 2)

        q_eff = F.normalize(q, p=2, dim=-1)
        k_eff = F.normalize(k, p=2, dim=-1)
        scale = self.logit_scale.exp().clamp(max=50.0)

        # Short-sequence fallback (e.g., generate() when T <= W)
        if T <= W:
            att = torch.matmul(q_eff, k_eff.transpose(-2, -1)) * scale
            if causal:
                cmask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool)).view(1, 1, T, T)
                att = att.masked_fill(~cmask, float("-inf"))
            tok_mask = _TopKHot.apply(att, min(self.k_sparse_tok, T), self.topk_mode)
            probs = F.softmax(att, dim=-1) * tok_mask
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-6)
            out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, T, C)
            return self.resid_dropout(self.c_proj(out))

        # Pad on right only during eval/generate if T % W != 0 (never triggers when T = block_size)
        pad = (W - (T % W)) % W
        if pad > 0:
            q_eff = F.pad(q_eff, (0, 0, 0, pad))
            k_eff = F.pad(k_eff, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))
            k_mag = F.pad(k.float().norm(dim=-1), (0, pad))
        else:
            k_mag = k.float().norm(dim=-1)

        T_pad = T + pad
        N_W, B_W = T_pad // W, W // bs
        N_B = N_W * B_W
        k_off = max(1, min(self.k_blocks - 1, N_B - B_W))

        if self.training:
            if self.update_mode == "v11_every":
                with torch.no_grad():
                    x_heads = torch.nan_to_num(
                        k_eff.detach().float().transpose(0, 1).reshape(H, B * T_pad, D),
                        nan=0.0, posinf=1e4, neginf=-1e4,
                    )
                    C_v11 = _v11_batched_spherical(x_heads, k=self.n_clusters, sub_n=self.sub_n)
                    C_next = _damped_lloyd_spherical(x_heads, C_v11, damp_scale=2.0).to(self.C_heads.dtype)
                    self.C_heads.copy_(C_next)
                C_use = C_next.to(k_eff.dtype)
            else:
                self._maybe_update_centroids(k_eff)
                C_use = self.C_heads.to(k_eff.dtype)
        else:
            C_use = self.C_heads.to(k_eff.dtype)

        # 1. Cluster & mag-sort inside each W-window
        assign_w = torch.argmax(torch.einsum("bhtd,hmd->bhtm", k_eff, C_use), dim=-1).view(B, H, N_W, W)
        k_mag_w = k_mag.view(B, H, N_W, W)
        mag_rank = k_mag_w / (k_mag_w.amax(dim=-1, keepdim=True) + 1.0)
        perm_w = torch.argsort(assign_w.float() * 2.0 + (1.0 - mag_rank), dim=-1, stable=True)

        perm_kv = perm_w.unsqueeze(-1).expand(-1, -1, -1, -1, D)
        k_s = torch.gather(k_eff.view(B, H, N_W, W, D), 3, perm_kv).view(B, H, N_B, bs, D)
        v_s = torch.gather(v.view(B, H, N_W, W, D), 3, perm_kv).view(B, H, N_B, bs, D)

        # 2. Stage-1 block router
        block_C = k_s.mean(dim=3)
        router_logits = torch.matmul(q_eff, block_C.transpose(-2, -1)) * scale

        block_win = (torch.arange(N_B, device=x.device) // B_W).view(1, 1, 1, N_B)
        q_win = (torch.arange(T_pad, device=x.device) // W).view(1, 1, T_pad, 1)
        if causal:
            win_mask = block_win < q_win
        else:
            valid_win = block_win < ((T + W - 1) // W if pad == 0 else N_W - 1)
            win_mask = (block_win != q_win) & valid_win

        off_logits = router_logits.masked_fill(~win_mask, float("-inf"))
        off_mask = _TopKHot.apply(off_logits, k_off, self.topk_mode)
        _, off_idx = torch.topk(off_logits, k=k_off, dim=-1)
        off_valid = torch.gather(off_mask, -1, off_idx)

        flat_idx = off_idx.view(B, H, T_pad * k_off)
        gather_kv = flat_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, bs, D)
        k_glob = torch.gather(k_s, 2, gather_kv).view(B, H, T_pad, k_off * bs, D)
        v_glob = torch.gather(v_s, 2, gather_kv).view(B, H, T_pad, k_off * bs, D)
        glob_gate = off_valid.unsqueeze(-1).expand(-1, -1, -1, -1, bs).reshape(B, H, T_pad, k_off * bs)

        # 3. Local window
        q_win_1d = torch.arange(T_pad, device=x.device) // W
        k_loc = k_eff.view(B, H, N_W, W, D)[:, :, q_win_1d, :, :]
        v_loc = v.view(B, H, N_W, W, D)[:, :, q_win_1d, :, :]
        loc_pos = (q_win_1d * W).view(T_pad, 1) + torch.arange(W, device=x.device).view(1, W)
        if causal:
            t_pos = torch.arange(T_pad, device=x.device).view(T_pad, 1)
            local_mask = (t_pos >= loc_pos).view(1, 1, T_pad, W).expand(B, H, -1, -1)
        else:
            local_mask = (loc_pos < T).view(1, 1, T_pad, W).expand(B, H, -1, -1)

        k_g = torch.cat([k_loc, k_glob], dim=3)
        v_g = torch.cat([v_loc, v_glob], dim=3)
        full_mask = torch.cat([local_mask, glob_gate > 0], dim=-1)
        gate = torch.cat(
            [torch.ones(B, H, T_pad, W, device=x.device, dtype=q_eff.dtype), glob_gate.to(q_eff.dtype)],
            dim=-1,
        )

        # 4. Stage-2 token Top-K
        scores = torch.einsum("bhtd,bhtkd->bhtk", q_eff, k_g) * scale
        scores = scores.masked_fill(~full_mask, float("-inf"))
        tok_mask = _TopKHot.apply(scores, min(self.k_sparse_tok, scores.size(-1)), self.topk_mode)

        probs = F.softmax(scores, dim=-1) * tok_mask * gate
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-6)

        out = torch.einsum("bhtk,bhtkd->bhtd", probs, v_g)[:, :, :T, :]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(out))

class KreinAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head 
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim = self.n_embd // self.n_head 
        assert self.head_dim % 2 == 0, "head_dim must be even to pair channels for Lorentz boosts."
        
        # Hyperparameters for the Krein/Lorentzian coordinate manifold
        self.pos_dim = getattr(config, 'pos_dim', 16)
        self.delta = getattr(config, 'delta', 1e-4)
        self.tau = getattr(config, 'tau', 0.01)
        init_c = getattr(config, 'init_c', 1.0)
        self.block_size = getattr(config, 'block_size', 1024)
        self.phi_max = getattr(config, 'phi_max', 2.0)  # Maximum rapidity boundary
        
        # Feature projections
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Projections from high-dimensional representations to coordinate space
        self.to_pos_q = nn.Linear(config.n_embd, self.pos_dim, bias=False)
        self.to_pos_kv = nn.Linear(config.n_embd, self.pos_dim, bias=False)
        
        # Self-adjoint coordinate matrix (Stable 0.02 initialization)
        self.pos_proj = nn.Parameter(torch.randn(self.pos_dim, self.pos_dim) * 0.02)
        
        # Learnable baseline speed of causality propagation c
        self.log_c = nn.Parameter(torch.tensor(math.log(init_c)))
        
        # --- Content-dependent speed of causality projections ---
        self.to_c_q = nn.Linear(config.n_embd, 1, bias=True)
        self.to_c_k = nn.Linear(config.n_embd, 1, bias=True)
        
        # --- Relativistic Spacetime Lorentz Boost Projections ---
        # Shared rapidities mapping
        self.to_rapidity_q = nn.Linear(self.pos_dim + 1, self.head_dim // 2, bias=False)
        self.to_rapidity_k = nn.Linear(self.pos_dim + 1, self.head_dim // 2, bias=False)
        
        # Initializations
        nn.init.normal_(self.to_c_q.weight, std=0.01)
        nn.init.normal_(self.to_c_k.weight, std=0.01)
        nn.init.zeros_(self.to_c_q.bias)
        nn.init.zeros_(self.to_c_k.bias)
        
        nn.init.normal_(self.to_rapidity_q.weight, std=0.02)
        nn.init.normal_(self.to_rapidity_k.weight, std=0.02)
        
        # Learnable Head-wise Spacetime Mixing Coefficient
        self.mix_coef = nn.Parameter(torch.zeros(1, self.n_head, 1, 1))
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, causal=True):
        B, T, C = x.size()
        H = self.n_head
        d_k = self.head_dim
        
        # 1. Dynamically generate sequential temporal coordinates and compute temporal distance
        t = torch.arange(T, dtype=x.dtype, device=x.device).view(1, T, 1)  # (1, T, 1)
        t = t.expand(B, T, 1) / self.block_size
        
        q_x, kv_x = x, x
        q_t, kv_t = t, t
        
        # Temporal distance: dT = t_q - t_kv
        dT = q_t - kv_t.transpose(1, 2)  # (B, T, T)
            
        # 2. Map features to raw spatial coordinates in the manifold
        q_pos_raw = self.to_pos_q(q_x)
        kv_pos_raw = self.to_pos_kv(kv_x)
            
        # 3. Symmetrize weight matrix to enforce self-adjointness (prevents coordinate curl)
        W_symm = 0.5 * (self.pos_proj + self.pos_proj.t())
        q_pos_proj = F.linear(q_pos_raw, W_symm)
        kv_pos_proj = F.linear(kv_pos_raw, W_symm)
        
        # 4. Standard multi-head feature mapping
        q = self.q_proj(q_x).view(B, T, H, d_k).transpose(1, 2)
        k = self.k_proj(kv_x).view(B, T, H, d_k).transpose(1, 2)
        v = self.v_proj(kv_x).view(B, T, H, d_k).transpose(1, 2)
        
        # --- 5. Curved Spacetime Geometry: Hyperboloid Projection ---
        q_norm2 = torch.sum(q_pos_proj**2, dim=-1, keepdim=True)
        q_0 = torch.sqrt(1.0 + q_norm2)  # (B, T, 1)
        
        kv_norm2 = torch.sum(kv_pos_proj**2, dim=-1, keepdim=True)
        kv_0 = torch.sqrt(1.0 + kv_norm2)  # (B, T, 1)
        
        term_time = -torch.matmul(q_0, kv_0.transpose(1, 2))  # (B, T, T)
        term_space = torch.bmm(q_pos_proj, kv_pos_proj.transpose(1, 2))  # (B, T, T)
        minkowski_dot = term_time + term_space  # (B, T, T)
        
        # Hyperbolic distance calculation: dS = arcosh(-minkowski_dot)
        u = -minkowski_dot
        u = torch.clamp(u, min=1.0)
        
        # Stable arcosh
        dS = torch.log(u + torch.sqrt(u**2 - 1.0 + self.delta))
        dS2 = dS ** 2
        
        # Apply smooth softsign distance scaling
        dS2_smooth = dS2 / (torch.sqrt(dS2 + self.delta) + 1.0)
        
        # --- Spacetime Metric Modulation: Variable Speed of Causality ---
        c_global = torch.exp(torch.clamp(self.log_c, min=-5.0, max=5.0))
        
        # Local temporal metric scaling
        c_mod_q = 0.1 + 1.9 * torch.sigmoid(self.to_c_q(q_x))  # (B, T, 1)
        c_mod_kv = 0.1 + 1.9 * torch.sigmoid(self.to_c_k(kv_x))  # (B, T, 1)
        
        # Pairwise variable speed of light: C_ts = c_global * c_q(t) * c_kv(s)
        c_matrix = c_global * torch.matmul(c_mod_q, c_mod_kv.transpose(1, 2))  # (B, T, T)
        
        # Curved spacetime interval
        lorentzian_interval = dS2_smooth - (c_matrix * dT) ** 2  # (B, T, T)
        
        # --- Relativistic Spacetime Lorentz Boosts (L-RoPE) ---
        # Concatenate spatial and temporal coordinates
        q_coord = torch.cat([q_pos_proj, q_t], dim=-1)  # (B, T, pos_dim + 1)
        kv_coord = torch.cat([kv_pos_proj, kv_t], dim=-1)  # (B, T, pos_dim + 1)
        
        # Map coordinates to rapidities (tanh limits growth for numerical safety)
        phi_q = self.phi_max * torch.tanh(self.to_rapidity_q(q_coord))  # (B, T, d_k // 2)
        phi_k = self.phi_max * torch.tanh(self.to_rapidity_k(kv_coord))  # (B, T, d_k // 2)
        
        # Reshape for multi-head rapidities
        phi_q = phi_q.unsqueeze(1).expand(-1, H, -1, -1)
        phi_k = phi_k.unsqueeze(1).expand(-1, H, -1, -1)
        
        # Split features into 2D pairs (spacelike and timelike halves)
        q1, q2 = torch.split(q, d_k // 2, dim=-1)
        k1, k2 = torch.split(k, d_k // 2, dim=-1)
        
        # Apply Lorentz Boost transform
        cosh_q, sinh_q = torch.cosh(phi_q), torch.sinh(phi_q)
        cosh_k, sinh_k = torch.cosh(phi_k), torch.sinh(phi_k)
        
        # Euclidean representation of the boost
        q = torch.cat([q1 * cosh_q + q2 * sinh_q, q1 * sinh_q + q2 * cosh_q], dim=-1)
        k = torch.cat([k1 * cosh_k + k2 * sinh_k, k1 * sinh_k + k2 * cosh_k], dim=-1)
        
        # Euclidean inner product calculation
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 6. Continuous Spacetime Gating
        gate_temporal = torch.sigmoid(dT / self.tau)
        gate_lorentzian = torch.sigmoid(-lorentzian_interval / self.tau)
        
        if causal:
            gate_space = (gate_temporal * (1.0 - gate_lorentzian)).unsqueeze(1)  # Spacelike channel (B, 1, T, T)
            gate_time = (gate_temporal * gate_lorentzian).unsqueeze(1)           # Timelike channel (B, 1, T, T)
        else:
            gate_space = (1.0 - gate_lorentzian).unsqueeze(1)
            gate_time = gate_lorentzian.unsqueeze(1)
        
        # Unified Gating with Single Softmax
        mix = torch.sigmoid(self.mix_coef)  # (1, H, 1, 1)
        gate_unified = (1.0 - mix) * gate_space + mix * gate_time  # (B, H, T, T)
        
        gated_logits = attn_logits + torch.log(gate_unified + 1e-6)
        
        # STRICT CAUSAL HARD-MASK: Prevents future leakage
        if causal:
            hard_mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1).view(1, 1, T, T)
            gated_logits = gated_logits + hard_mask
        
        # Single Softmax pass (drastically lowers VRAM footprint and stabilizes gradients)
        attn_weights = F.softmax(gated_logits, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Single multiplication by the full V matrix (No representational capacity drop)
        context = torch.matmul(attn_weights, v)
        
        # Symmetrical transposition back to query_dim (C)
        context = context.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.resid_dropout(self.out_proj(context))


class MetricSimplexAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert (
            config.n_embd % config.n_head == 0
        ), "n_embd must be divisible by n_head."
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.epsilon = getattr(config, "epsilon", 0.05)
        self.delta = getattr(config, "delta", 1e-4)

        # Linear projections
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(
            config.n_embd, config.n_embd, bias=config.bias
        )

        # --- Dynamic Content-Dependent Scale Projections ---
        self.to_scale_q = nn.Linear(config.n_embd, config.n_head, bias=False)
        self.to_scale_k = nn.Linear(config.n_embd, config.n_head, bias=False)
        nn.init.normal_(self.to_scale_q.weight, std=0.01)
        nn.init.normal_(self.to_scale_k.weight, std=0.01)

        # --- Self-Adjoint Metric Projection Matrix ---
        self.metric_proj = nn.Parameter(
            torch.randn(self.head_dim, self.head_dim) * 0.02
        )

        # Smooth learnable scale initialized near ~0.5
        self.temporal_lambda = nn.Parameter(torch.zeros(1, self.n_head, 1, 1))

        # --- Log-Scale Initialization for Gradient Stability ---
        self.log_scale = nn.Parameter(torch.zeros(1, config.n_head, 1, 1))

        # Causal mask buffer pre-allocation
        block_size = 128 #getattr(config, "block_size", 2048)
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1
            ),
            persistent=False,
        )

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, causal=True):
        B, T, C = x.size()
        H = self.n_head
        d_k = self.head_dim
        device = x.device

        # 1. Standard projection and reshaping
        q = self.q_proj(x).view(B, T, H, d_k).transpose(1, 2)  # (B, H, T, d_k)
        k = self.k_proj(x).view(B, T, H, d_k).transpose(1, 2)  # (B, H, T, d_k)
        v = self.v_proj(x).view(B, T, H, d_k).transpose(1, 2)  # (B, H, T, d_k)

        # 2. Self-Adjoint Riemannian metric projection: (q M)(k M)^T = q M^2 k^T
        M_symm = 0.5 * (self.metric_proj + self.metric_proj.t())
        q = torch.matmul(q, M_symm)
        k = torch.matmul(k, M_symm)

        # --- Safe Numerics Block (Execute Attention Map in FP32) ---
        with torch.autocast(device_type=device.type, enabled=False):
            q_32 = F.normalize(q.float(), p=2, dim=-1)
            k_32 = F.normalize(k.float(), p=2, dim=-1)

            # 4. Cosine similarity matrix
            cos_sim = torch.matmul(q_32, k_32.transpose(-2, -1))

            # 5. Boundary clamping
            boundary = 1.0 - self.epsilon
            cos_sim = torch.clamp(cos_sim, min=-boundary, max=boundary)

            # 6. Dynamic Scaling with Safe Log-Clamping
            sq = self.to_scale_q(x).transpose(1, 2).unsqueeze(-1).float()
            sk = self.to_scale_k(x).transpose(1, 2).unsqueeze(-2).float()

            # Bound scale exponents to range [-10.0, 5.0] to avoid FP16/FP32 overflow
            total_log_scale = torch.clamp(
                self.log_scale.float() + sq + sk, min=-10.0, max=5.0
            )
            dynamic_scale = torch.exp(total_log_scale)

            # 7. Entropic Potential mapping
            denom = 1.0 - torch.square(cos_sim) + self.delta
            gated_similarities = (cos_sim / denom) * dynamic_scale

            # 8. Simplex Mapping via Softplus
            positive_similarities = F.softplus(gated_similarities)

            # 9. Power-Law Temporal Decay Gate: (1 + dT)^(-scale)
            t_seq = torch.arange(T, device=device, dtype=torch.float32)
            dT = (t_seq.unsqueeze(1) - t_seq.unsqueeze(0)).clamp(min=0.0)
            dT_log = torch.log1p(dT).unsqueeze(0).unsqueeze(0)

            # Smoothly bound decay scale to (0, 3.0) maintaining non-zero gradients
            scale = 3.0 * torch.sigmoid(self.temporal_lambda.float())
            decay_gate = torch.exp(-scale * dT_log)

            positive_similarities = positive_similarities * decay_gate

            # 10. Causal Masking
            if causal:
                if T <= self.causal_mask.size(0):
                    mask_slice = self.causal_mask[:T, :T].view(1, 1, T, T)
                else:
                    mask_slice = torch.triu(
                        torch.ones(T, T, device=device, dtype=torch.bool),
                        diagonal=1,
                    ).view(1, 1, T, T)
                positive_similarities = positive_similarities.masked_fill(
                    mask_slice, 0.0
                )

            # 11. Row Normalization
            row_sums = positive_similarities.sum(dim=-1, keepdim=True).clamp(
                min=1e-9
            )
            attn_weights = positive_similarities / row_sums

        # Cast back to original input precision (e.g., FP16/BF16)
        attn_weights = attn_weights.to(dtype=x.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # 12. Value Aggregation & Output Projection
        out = torch.matmul(attn_weights, v)  # (B, H, T, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.out_proj(out))

class LinearMetricSimplexAttention(nn.Module):

    def __init__(self, config, num_decay_branches=4):
        super().__init__()
        assert (
            config.n_embd % config.n_head == 0
        ), "n_embd must be divisible by n_head."

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.M = num_decay_branches

        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.to_scale_q = nn.Linear(config.n_embd, config.n_head, bias=False)
        self.to_scale_k = nn.Linear(config.n_embd, config.n_head, bias=False)

        eye = torch.eye(self.head_dim).repeat(config.n_head, 1, 1)
        noise = torch.randn(config.n_head, self.head_dim, self.head_dim) * 0.02
        self.metric_proj = nn.Parameter(eye + noise)

        init_decay_rates = torch.logspace(-3, -0.3, steps=self.M).unsqueeze(0)
        inv_softplus_rates = torch.log(torch.expm1(init_decay_rates))
        self.log_decay_rates = nn.Parameter(
            inv_softplus_rates.repeat(config.n_head, 1)
        )

        self.to_branch_logits = nn.Linear(
            config.n_embd, config.n_head * self.M, bias=False
        )
        nn.init.normal_(self.to_branch_logits.weight, std=0.01)

        self.attn_dropout = nn.Dropout(0.0)
        self.resid_dropout = nn.Dropout(config.dropout)

    def _get_decay_parameters(self, x):
        B, T, _ = x.size()
        decay_rates = F.softplus(self.log_decay_rates) + 1e-5

        branch_logits = (
            self.to_branch_logits(x)
            .view(B, T, self.n_head, self.M)
            .transpose(1, 2)
        )
        branch_weights = F.softmax(branch_logits, dim=-1) + 1e-4

        return decay_rates, branch_weights

    def _prepare_qkv(self, x):
        B, T, C = x.size()
        H, d_k = self.n_head, self.head_dim

        q = self.q_proj(x).view(B, T, H, d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, d_k).transpose(1, 2)

        M_symm = 0.5 * (self.metric_proj + self.metric_proj.transpose(-1, -2))
        q = torch.matmul(q, M_symm)
        k = torch.matmul(k, M_symm)

        q_pos = F.normalize(F.softplus(q) + 1e-4, p=2, dim=-1)
        k_pos = F.normalize(F.softplus(k) + 1e-4, p=2, dim=-1)

        sq = (
            torch.sigmoid(self.to_scale_q(x))
            .transpose(1, 2)
            .unsqueeze(-1)
            .to(q.dtype)
        )
        sk = (
            torch.sigmoid(self.to_scale_k(x))
            .transpose(1, 2)
            .unsqueeze(-1)
            .to(k.dtype)
        )

        q_gate = q_pos * (0.5 + sq)
        k_gate = k_pos * (0.5 + sk)

        return q_gate, k_gate, v

    def forward(self, x, causal=True):
        B, T, C = x.size()
        H, d_k = self.n_head, self.head_dim
        device = x.device

        # Dynamically match computation precision
        calc_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32

        q, k, v = self._prepare_qkv(x)
        decay_rates, branch_weights = self._get_decay_parameters(x)

        with torch.autocast(device_type=device.type, enabled=False):
            q_calc = q.to(dtype=calc_dtype)
            k_calc = k.to(dtype=calc_dtype)
            v_calc = v.to(dtype=calc_dtype)

            t_seq = torch.arange(T, device=device, dtype=calc_dtype)
            dT = t_seq.unsqueeze(1) - t_seq.unsqueeze(0)
            if causal:
                dT = torch.tril(dT)
            else:
                dT = dT.abs()

            rates = decay_rates.view(1, H, 1, 1, self.M).to(dtype=calc_dtype)
            weights = branch_weights.unsqueeze(-2).to(dtype=calc_dtype)
            dT_5d = dT.view(1, 1, T, T, 1)

            decay_components = torch.exp(-rates * dT_5d)
            L_combined = (weights * decay_components).sum(dim=-1)

            S = torch.matmul(q_calc, k_calc.transpose(-2, -1))

            if causal:
                causal_mask = torch.tril(
                    torch.ones(T, T, device=device, dtype=torch.bool)
                ).view(1, 1, T, T)
                S_decayed = S * L_combined * causal_mask
            else:
                S_decayed = S * L_combined

            denom = S_decayed.sum(dim=-1, keepdim=True)
            attn_weights = S_decayed / denom

            attn_weights = self.attn_dropout(attn_weights)
            out_calc = torch.matmul(attn_weights, v_calc)

        out = out_calc.to(dtype=x.dtype)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(out))

    def init_recurrent_state(self, batch_size, device, dtype=None):
        """Allocates state-space memory buffers for streaming generation."""
        if dtype is None:
            dtype = self.metric_proj.dtype
        H, d_k = self.n_head, self.head_dim
        return {
            "S": torch.zeros(
                batch_size, H, self.M, d_k, d_k, device=device, dtype=dtype
            ),
            "z": torch.zeros(
                batch_size, H, self.M, d_k, device=device, dtype=dtype
            ),
        }

    @torch.no_grad()
    def step(self, x_t, state):
        """Processes a single token x_t in O(1) memory."""
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(1)

        B, T, C = x_t.size()
        assert T == 1, "step() processes exactly 1 token at a time."
        H, d_k = self.n_head, self.head_dim
        device = x_t.device

        calc_dtype = torch.float64 if x_t.dtype == torch.float64 else torch.float32

        branch_logits = (
            self.to_branch_logits(x_t)
            .view(B, 1, H, self.M)
            .transpose(1, 2)
        )
        branch_weights = (
            F.softmax(branch_logits, dim=-1).squeeze(2).to(dtype=calc_dtype) + 1e-4
        )

        q_t, k_t, v_t = self._prepare_qkv(x_t)
        decay_rates = F.softplus(self.log_decay_rates) + 1e-5

        with torch.autocast(device_type=device.type, enabled=False):
            q_t = q_t.squeeze(2).to(dtype=calc_dtype)
            k_t = k_t.squeeze(2).to(dtype=calc_dtype)
            v_t = v_t.squeeze(2).to(dtype=calc_dtype)

            gamma = (
                torch.exp(-decay_rates)
                .view(1, H, self.M, 1, 1)
                .to(dtype=calc_dtype)
            )
            gamma_z = gamma.squeeze(-1)

            kv_outer = torch.einsum("bhk,bhv->bhkv", k_t, v_t).unsqueeze(2)

            state["S"] = gamma * state["S"] + kv_outer
            state["z"] = gamma_z * state["z"] + k_t.unsqueeze(2)

            q_S = torch.einsum("bhk,bhmkv->bhmv", q_t, state["S"])
            q_z = torch.einsum("bhk,bhmk->bhm", q_t, state["z"])

            w = branch_weights.unsqueeze(-1)
            w_z = branch_weights

            o_combined = (w * q_S).sum(dim=2)
            z_combined = (w_z * q_z).sum(dim=2, keepdim=True)

            out_t = o_combined / z_combined

        out_t = out_t.to(dtype=x_t.dtype)
        out_t = out_t.view(B, 1, C)

        return self.resid_dropout(self.out_proj(out_t)), state

class LinearMetricSimplexAttention5(nn.Module):

    def __init__(self, config, num_decay_branches=4):
        super().__init__()
        assert (
            config.n_embd % config.n_head == 0
        ), "n_embd must be divisible by n_head."

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.M = num_decay_branches  # Number of exponential states per head

        # Linear projections
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(
            config.n_embd, config.n_embd, bias=config.bias
        )

        # Dynamic Content-Dependent Scale Gates
        self.to_scale_q = nn.Linear(config.n_embd, config.n_head, bias=False)
        self.to_scale_k = nn.Linear(config.n_embd, config.n_head, bias=False)

        # Per-Head Self-Adjoint Metric Projection Matrix
        eye = torch.eye(self.head_dim).repeat(config.n_head, 1, 1)
        noise = torch.randn(config.n_head, self.head_dim, self.head_dim) * 0.02
        self.metric_proj = nn.Parameter(eye + noise)

        # Log-spaced decay initialization using Inverse Softplus
        init_decay_rates = torch.logspace(-3, -0.3, steps=self.M).unsqueeze(0)
        inv_softplus_rates = torch.log(torch.expm1(init_decay_rates))
        self.log_decay_rates = nn.Parameter(
            inv_softplus_rates.repeat(config.n_head, 1)
        )  # (H, M)

        # Simple Linear Head for Dynamic Token Branch Weights (No Conv needed)
        self.to_branch_logits = nn.Linear(
            config.n_embd, config.n_head * self.M, bias=False
        )
        nn.init.normal_(self.to_branch_logits.weight, std=0.01)

        self.attn_dropout = nn.Dropout(0.0)
        self.resid_dropout = nn.Dropout(config.dropout)

    def _get_decay_parameters(self, x):
        """Computes decay rates and strictly positive softmax branch weights."""
        B, T, _ = x.size()

        # 1. Non-negative decay rates (Softplus is smooth and > 0)
        decay_rates = F.softplus(self.log_decay_rates) + 1e-5  # (H, M)

        # 2. Dynamic Token-Dependent Branch Weights (Softmax ensures > 0)
        branch_logits = (
            self.to_branch_logits(x)
            .view(B, T, self.n_head, self.M)
            .transpose(1, 2)
        )
        branch_weights = F.softmax(branch_logits, dim=-1) + 1e-4  # (B, H, T, M)

        return decay_rates, branch_weights

    def _prepare_qkv(self, x):
        """Projects Q, K, V and applies Riemannian Metric Projection (M^2) and
        Strictly Positive Feature Mapping.
        """
        B, T, C = x.size()
        H, d_k = self.n_head, self.head_dim

        q = self.q_proj(x).view(B, T, H, d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, d_k).transpose(1, 2)

        # Per-head Self-Adjoint Riemannian Metric
        M_symm = 0.5 * (self.metric_proj + self.metric_proj.transpose(-1, -2))
        q = torch.matmul(q, M_symm)
        k = torch.matmul(k, M_symm)

        # Strictly Positive Feature Map (Softplus + eps > 0 everywhere)
        q_pos = F.normalize(F.softplus(q) + 1e-4, p=2, dim=-1)
        k_pos = F.normalize(F.softplus(k) + 1e-4, p=2, dim=-1)

        # Dynamic Content Scaling
        sq = (
            torch.sigmoid(self.to_scale_q(x))
            .transpose(1, 2)
            .unsqueeze(-1)
            .to(q.dtype)
        )
        sk = (
            torch.sigmoid(self.to_scale_k(x))
            .transpose(1, 2)
            .unsqueeze(-1)
            .to(k.dtype)
        )

        q_gate = q_pos * (0.5 + sq)
        k_gate = k_pos * (0.5 + sk)

        return q_gate, k_gate, v

    def forward(self, x, causal=True):
        B, T, C = x.size()
        H, d_k = self.n_head, self.head_dim
        device = x.device

        q, k, v = self._prepare_qkv(x)
        decay_rates, branch_weights = self._get_decay_parameters(x)

        with torch.autocast(device_type=device.type, enabled=False):
            q_32 = q.float()
            k_32 = k.float()
            v_32 = v.float()

            t_seq = torch.arange(T, device=device, dtype=torch.float32)
            dT = t_seq.unsqueeze(1) - t_seq.unsqueeze(0)
            if causal:
                dT = torch.tril(dT)  # Zeroes upper triangle without clamp
            else:
                dT = dT.abs()

            rates = decay_rates.view(1, H, 1, 1, self.M).float()  # (1, H, 1, 1, M)
            weights = branch_weights.unsqueeze(-2).float()        # (B, H, T, 1, M)
            dT_5d = dT.view(1, 1, T, T, 1)                        # (1, 1, T, T, 1)

            decay_components = torch.exp(-rates * dT_5d)
            L_combined = (weights * decay_components).sum(dim=-1) # (B, H, T, T)

            S = torch.matmul(q_32, k_32.transpose(-2, -1))

            if causal:
                causal_mask = torch.tril(
                    torch.ones(T, T, device=device, dtype=torch.bool)
                ).view(1, 1, T, T)
                S_decayed = S * L_combined * causal_mask
            else:
                S_decayed = S * L_combined

            # Pure Division: denom is strictly > 0 due to positive feature maps
            denom = S_decayed.sum(dim=-1, keepdim=True)
            attn_weights = S_decayed / denom

            attn_weights = self.attn_dropout(attn_weights)
            out_32 = torch.matmul(attn_weights, v_32)

        out = out_32.to(dtype=x.dtype)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(out))

    def init_recurrent_state(self, batch_size, device, dtype=torch.float32):
        """Allocates state-space memory buffers for streaming generation."""
        H, d_k = self.n_head, self.head_dim
        return {
            "S": torch.zeros(
                batch_size, H, self.M, d_k, d_k, device=device, dtype=dtype
            ),
            "z": torch.zeros(
                batch_size, H, self.M, d_k, device=device, dtype=dtype
            ),
        }

    @torch.no_grad()
    def step(self, x_t, state):
        """Processes a single token x_t in O(1) memory."""
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(1)

        B, T, C = x_t.size()
        assert T == 1, "step() processes exactly 1 token at a time."
        H, d_k = self.n_head, self.head_dim
        device = x_t.device

        branch_logits = (
            self.to_branch_logits(x_t)
            .view(B, 1, H, self.M)
            .transpose(1, 2)
        )
        branch_weights = (
            F.softmax(branch_logits, dim=-1).squeeze(2).float() + 1e-4
        )  # (B, H, M)

        q_t, k_t, v_t = self._prepare_qkv(x_t)
        decay_rates = F.softplus(self.log_decay_rates) + 1e-5

        with torch.autocast(device_type=device.type, enabled=False):
            q_t = q_t.squeeze(2).float()  # (B, H, d_k)
            k_t = k_t.squeeze(2).float()  # (B, H, d_k)
            v_t = v_t.squeeze(2).float()  # (B, H, d_k)

            gamma = (
                torch.exp(-decay_rates)
                .view(1, H, self.M, 1, 1)
                .to(dtype=torch.float32)
            )
            gamma_z = gamma.squeeze(-1)

            kv_outer = torch.einsum("bhk,bhv->bhkv", k_t, v_t).unsqueeze(2)

            state["S"] = gamma * state["S"] + kv_outer
            state["z"] = gamma_z * state["z"] + k_t.unsqueeze(2)

            q_S = torch.einsum("bhk,bhmkv->bhmv", q_t, state["S"])
            q_z = torch.einsum("bhk,bhmk->bhm", q_t, state["z"])

            w = branch_weights.unsqueeze(-1)  # (B, H, M, 1)
            w_z = branch_weights              # (B, H, M)

            o_combined = (w * q_S).sum(dim=2)
            z_combined = (w_z * q_z).sum(dim=2, keepdim=True)

            out_t = o_combined / z_combined

        out_t = out_t.to(dtype=x_t.dtype)
        out_t = out_t.view(B, 1, C)

        return self.resid_dropout(self.out_proj(out_t)), state
   
class ShiftedReLU(nn.Module):
    def __init__(self, num_neurons):
        super().__init__()
        shifts = 0.02/torch.sqrt(torch.arange(0, num_neurons))
        self.register_buffer('shift', shifts) 

    def forward(self, x):
        return F.relu(x - self.shift)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        up = 4
        hidden_dim = up * config.n_embd
        qrot_channels = self.n_embd // 2

        if pattmlp:
            self.thing = modules.PattLayer(
                config.n_embd,
                config.n_embd,
                mlp_ptok,
            )
            return

        self.c_fc = Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.gelu = nn.GELU()

        # Output projection dimension: qrot variants reduce 4*n_embd to 2*n_embd
        uses_qrot_dim = (
            qrot or qrot_radial or qrot_pre_rootkan or qrot_angle
        )
        proj_in_dim = 2 * config.n_embd if uses_qrot_dim else hidden_dim
        self.c_proj = Linear(proj_in_dim, config.n_embd, bias=config.bias)

        if hcmlp:
            self.c_fc = modules.HCLinear(config.n_embd, hidden_dim, 1.0, False)
            self.c_proj = modules.HCLinear(proj_in_dim, config.n_embd, 1.0, False)

        # -------------------------------------------------------------
        # 1. PARAMETERS FOR ITERLESS ROOTKAN (Degree 3)
        # -------------------------------------------------------------
        k3 = torch.arange(1, 4, dtype=torch.float32)
        cheb_d3 = torch.cos((2 * k3 - 1) / (2 * 3) * math.pi)  # [+0.866, 0.0, -0.866]

        if rootkan:
            self.roots_d3 = nn.Parameter(cheb_d3.view(3, 1).repeat(1, hidden_dim))
            self.scale_d3 = nn.Parameter(torch.randn(hidden_dim) * 0.1)

        if rootkan_glu:
            self.glu_roots = nn.Parameter(cheb_d3.view(3, 1).repeat(1, hidden_dim))
            self.glu_scale = nn.Parameter(torch.randn(hidden_dim) * 0.1)

        # -------------------------------------------------------------
        # 2. HYBRID 1: RADIAL ROOTKAN PARAMETERS (Modulates norm on S^3)
        # -------------------------------------------------------------
        if qrot_radial:
            # Operates on the scalar norm of each 4D quaternion channel
            self.radial_roots = nn.Parameter(cheb_d3.view(3, 1, 1).repeat(1, qrot_channels, 1))
            self.radial_scale = nn.Parameter(torch.randn(qrot_channels, 1) * 0.1)

        # -------------------------------------------------------------
        # 3. HYBRID 2: UNITARY PRE-ALIGNMENT PARAMETERS
        # -------------------------------------------------------------
        if qrot_pre_rootkan:
            # Evaluates RootKAN on the 2*n_embd unitarily rotated channels
            self.pre_roots = nn.Parameter(cheb_d3.view(3, 1).repeat(1, 2 * config.n_embd))
            self.pre_scale = nn.Parameter(torch.randn(2 * config.n_embd) * 0.1)

        # -------------------------------------------------------------
        # 4. HYBRID 3: ANGLE-MODULATED ROTOR PARAMETERS
        # -------------------------------------------------------------
        if qrot_angle:
            self.angle_roots = nn.Parameter(cheb_d3.view(3, 1, 1).repeat(1, qrot_channels, 1))
            self.angle_scale = nn.Parameter(torch.randn(qrot_channels, 1) * 0.1)

        self.dropout = nn.Dropout(config.dropout)
        self.dac = ShiftedReLU(hidden_dim)
        self.univ = utils.Univfun(config.n_embd)

    # -----------------------------------------------------------------
    # Helper: Clean 3rd-Degree Factorized Root Product
    # -----------------------------------------------------------------
    def _eval_root_product(self, u, roots, scale):
        denom = torch.sqrt(1.0 + roots.pow(2))
        prod = (u - roots[0]) / denom[0]
        prod = prod * ((u - roots[1]) / denom[1])
        prod = prod * ((u - roots[2]) / denom[2])
        return prod * scale

    # -----------------------------------------------------------------
    # HYBRID 1: qrot Phase Routing + RootKAN Radial Amplitude Gating
    # -----------------------------------------------------------------
    def _forward_qrot_radial(self, x):
        b, t, _ = x.shape
        x4d = x.view(b, t, self.n_embd, 4)

        # 1. Decompose into Magnitude and S^3 Direction
        x_norm = torch.clamp_min(x4d.norm(dim=-1, keepdim=True), 1e-10)
        x_dir = x4d / x_norm

        # 2. Pure Unitary Phase Rotation
        rotors, rotated = x_dir.chunk(2, dim=2)
        x_rotated = utils.quaternion_multiply(rotors, rotated)  # (b, t, n_embd//2, 4)

        # 3. RootKAN Amplitude Gating on the Magnitude
        norm_rotors, norm_rotated = x_norm.chunk(2, dim=2)
        # Evaluate root polynomial on the radial distance
        poly_r = self._eval_root_product(norm_rotated, self.radial_roots, self.radial_scale)
        # One-sided gating: zeroes out irrelevant keys without destroying unit rotation
        r_gate = norm_rotated * torch.sigmoid(norm_rotated + poly_r)

        # 4. Recombine Amplitude * Phase
        x_out = x_rotated * r_gate
        x_flat = x_out.view(b, t, 2 * self.n_embd)
        return self.dropout(self.c_proj(x_flat))

    # -----------------------------------------------------------------
    # HYBRID 2: Unitary Pre-Alignment + Iterless Degree-3 RootKAN
    # -----------------------------------------------------------------
    def _forward_qrot_pre_rootkan(self, x):
        b, t, _ = x.shape
        x4d = x.view(b, t, self.n_embd, 4)

        # 1. Rotate coordinates unitarily on S^3 (zero variance explosion, sigma == 1)
        x_norm = torch.clamp_min(x4d.norm(dim=-1, keepdim=True), 1e-10)
        x_dir = x4d / x_norm
        rotors, rotated = x_dir.chunk(2, dim=2)
        x_rotated = utils.quaternion_multiply(rotors, rotated)

        # Preserve energy, but in unitarily pre-aligned coordinate frame
        _, norm_rotated = x_norm.chunk(2, dim=2)
        x_aligned = (x_rotated * norm_rotated).view(b, t, 2 * self.n_embd)

        # 2. Evaluate Iterless RootKAN on the Rotated Coordinates
        y = self._eval_root_product(
            x_aligned, self.pre_roots.unsqueeze(1), self.pre_scale.view(1, 1, -1)
        )
        return self.dropout(self.c_proj(y))

    # -----------------------------------------------------------------
    # HYBRID 3: RootKAN Angle-Modulated Rotor Gating
    # -----------------------------------------------------------------
    def _forward_qrot_angle(self, x):
        b, t, _ = x.shape
        x4d = x.view(b, t, self.n_embd, 4)

        x_norm = torch.clamp_min(x4d.norm(dim=-1, keepdim=True), 1e-10)
        x_dir = x4d / x_norm
        rotors, rotated = x_dir.chunk(2, dim=2)
        _, norm_rotated = x_norm.chunk(2, dim=2)

        # 1. Extract unit 3D spatial rotation axis n = (x, y, z)
        axis = rotors[..., 1:]
        axis_norm = torch.clamp_min(axis.norm(dim=-1, keepdim=True), 1e-10)
        axis_unit = axis / axis_norm

        # 2. Modulate rotation angle theta via RootKAN on the token norm
        poly_theta = self._eval_root_product(norm_rotated, self.angle_roots, self.angle_scale)
        theta = math.pi * torch.tanh(poly_theta)

        # 3. Construct dynamic angle-modulated rotor: q = (cos(theta/2), n * sin(theta/2))
        w = torch.cos(theta / 2.0)
        xyz = axis_unit * torch.sin(theta / 2.0)
        rotor_dynamic = torch.cat([w, xyz], dim=-1)

        # 4. Rotate and re-apply magnitude
        x_rotated = utils.quaternion_multiply(rotor_dynamic, rotated) * norm_rotated
        x_flat = x_rotated.view(b, t, 2 * self.n_embd)
        return self.dropout(self.c_proj(x_flat))

    # -----------------------------------------------------------------
    # BASELINE QROT
    # -----------------------------------------------------------------
    def _forward_qrot_base(self, x):
        b, t, _ = x.shape
        x4d = x.view(b, t, self.n_embd, 4)
        x_norm = torch.clamp_min(x4d.norm(dim=-1, keepdim=True), 1e-10)
        x_normalized = x4d / x_norm
        rotors, rotated = x_normalized.chunk(2, dim=2)
        x_rotated = utils.quaternion_multiply(rotors, rotated)
        x = x_rotated.view(b, t, 2 * self.n_embd)
        return self.dropout(self.c_proj(x))

    # -----------------------------------------------------------------
    # Forward Pass Routing
    # -----------------------------------------------------------------
    def forward(self, x):
        if testing_mlp:
            return self.univ(x)
        if pattmlp:
            return self.thing(x)

        # 1. Linear expansion projection
        x = self.c_fc(x)

        # 2. Active Model Routing
        if qrot_radial:
            return self._forward_qrot_radial(x)
        elif qrot_pre_rootkan:
            return self._forward_qrot_pre_rootkan(x)
        elif qrot_angle:
            return self._forward_qrot_angle(x)
        elif qrot:
            return self._forward_qrot_base(x)
        elif rootkan:
            y = self._eval_root_product(
                x, self.roots_d3.unsqueeze(1), self.scale_d3.view(1, 1, -1)
            )
            return self.dropout(self.c_proj(y))
        elif rootkan_glu:
            poly = self._eval_root_product(
                x, self.glu_roots.unsqueeze(1), self.glu_scale.view(1, 1, -1)
            )
            gate = x * torch.sigmoid(x + poly)
            return self.dropout(self.c_proj(gate))
        elif diffac:
            return self.dropout(self.c_proj(self.dac(x)))
        else:
            return self.dropout(self.c_proj(self.gelu(x)))
class KreinSelfRelation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = getattr(config, 'pos_dim', config.n_embd // config.n_head)
        
        # 1. STRICTLY ONE projection matrix for sequence features
        self.to_pos = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        
        # 2. Observer scale projection (1 scalar per head) to create asymmetry
        self.to_scale = nn.Linear(self.n_embd, self.n_head, bias=False)
        
        # 3. Learnable temporal decay factor per head
        self.lambda_time = nn.Parameter(torch.ones(1, self.n_head, 1, 1) * 0.1)
        
        # Initializing weights
        nn.init.normal_(self.to_pos.weight, std=0.02)
        nn.init.zeros_(self.to_scale.weight)

    def forward(self, x: torch.Tensor, causal=True) -> torch.Tensor:
        B, T, C = x.size()
        H, D = self.n_head, self.head_dim
        
        # --- A. Map to Spatial Manifold (Single Projection) ---
        # Shape: (B, H, T, D)
        p = self.to_pos(x).view(B, T, H, D).transpose(1, 2)
        
        # Compute Pairwise Squared Euclidean Distance: ||p_i - p_j||^2
        # ||p_i - p_j||^2 = ||p_i||^2 + ||p_j||^2 - 2 <p_i, p_j>
        p_norm2 = torch.sum(p**2, dim=-1, keepdim=True)  # (B, H, T, 1)
        dist_space = p_norm2 + p_norm2.transpose(-2, -1) - 2.0 * torch.matmul(p, p.transpose(-2, -1))
        dist_space = torch.clamp(dist_space, min=0.0)  # (B, H, T, T)
        
        # --- B. Correct Temporal Distance ---
        t_seq = torch.arange(T, device=x.device, dtype=x.dtype)
        dT = (t_seq.unsqueeze(1) - t_seq.unsqueeze(0))  # (T, T), where dT[i, j] = i - j
        dT2 = (dT ** 2).unsqueeze(0).unsqueeze(1)       # (1, 1, T, T)
        
        # --- C. Total Spacetime Distance ---
        # Both spatial distance and temporal distance penalize attention
        time_penalty = F.softplus(self.lambda_time) * dT2
        total_distance = dist_space + time_penalty  # (B, H, T, T)
        
        # --- D. Asymmetric Observer Metric Scale sigma(x_i) ---
        # Token i acts as the observer; sigma_i determines its receptive field width
        # Shape: (B, H, T, 1)
        sigma_i = F.softplus(self.to_scale(x)).transpose(1, 2).unsqueeze(-1) + 0.1
        
        # --- E. Logits Calculation ---
        # Negated distance ensures closer tokens in spacetime receive HIGHER attention
        attn_logits = -total_distance / (sigma_i * math.sqrt(D))
        
        # Causal Masking
        causal_mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        attn_logits = attn_logits + causal_mask.view(1, 1, T, T)
        
        return F.softmax(attn_logits, dim=-1)

class UnifiedKreinBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.bias = getattr(config, 'bias', False)
        
        self.p_dim = getattr(config, 'p_dim', 4 * config.n_embd)
        assert self.p_dim % self.n_head == 0, "p_dim must be divisible by n_head"
        self.head_p_dim = self.p_dim // self.n_head
        
        self.ln_1 = nn.LayerNorm(self.n_embd, bias=self.bias)
        self.ln_2 = nn.LayerNorm(self.head_p_dim, bias=self.bias)
        self.ln_p = nn.LayerNorm(self.p_dim, bias=self.bias)
        
        self.krein = KreinSelfRelation(config)
        
        self.up_proj = nn.Sequential(
            nn.Linear(self.n_embd, self.p_dim, bias=self.bias),
            nn.GELU()  
        )
        self.ac = nn.GELU()  
        self.down_proj = nn.Linear(self.p_dim, self.n_embd, bias=self.bias)
        
        self.dropout = nn.Dropout(getattr(config, 'dropout', 0.1))

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        B, T, C = x.size()
        H = self.n_head
        
        norm_x = self.ln_1(x)
        
        A = (self.krein(norm_x, causal=causal))
        
        U = (self.up_proj(norm_x))
        
        U = U.view(B, T, H, self.head_p_dim).transpose(1, 2)
        
        Y = self.ln_2(U + torch.matmul(A, U))
        
        Y = Y.transpose(1, 2).contiguous().view(B, T, self.p_dim)
        
        out = self.down_proj(Y)
        
        return x + self.dropout(out)

       
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = Attention(config)
        if(qrottention):
            self.attn = modules.SSM(config.n_embd,config.dropout,config.bias)
        elif cw_v11: 
            self.attn = CausalWindowedV11Attention(config)
        elif assocssm:
            self.attn = modules.AssociativeSSM(config.n_embd,config.dropout,config.bias)
        elif kreinSSM:
            self.attn = modules.KreinSSM(config.n_embd,config.dropout,config.bias)
        elif kratt:
            self.attn = KreinAttention(config)
        elif msa:
            self.attn = MetricSimplexAttention(config)
        elif lmsa:
            self.attn = LinearMetricSimplexAttention(config)
        elif k_atten:
            self.attn = modules.FKattention(config)
        elif h_k_atten:
            self.attn = modules.HydraKattention(config)
        elif p_atten:
            self.attn = modules.Pattention(config,64)
        if residcomp:
            self.shl = modules.SinkhornLinear(config.n_embd)
            self.shl2 = modules.SinkhornLinear(config.n_embd)
            with torch.no_grad():
                self.shl.weight.data = (torch.eye(config.n_embd))
                self.shl2.weight.data = (torch.eye(config.n_embd))
           
        self.ln_1 = LayerNorm(config)
        self.ln_2 = LayerNorm(config)
        self.mlp = MLP(config)
        self.register_buffer("block_input",torch.zeros([config.block_size, config.n_embd]))
        if fftmem:
            self.register_buffer("mem_input",torch.zeros([config.block_size, config.n_embd]))
        

    def forward(self, x, causal = True, fwd=True):
        if spl and fwd:
            self.block_input = x
        
        if residless:
            x = self.attn(self.ln_1(x), causal)
            x = self.mlp(self.ln_2(x))
        elif residcomp:
            #y = self.attn(self.ln_3(x), causal)
            y = self.shl(x) 
            x = y + self.attn(self.ln_1(x), causal)
            #y = self.mlp(self.ln_4(x))
            y = self.shl2(x) 
            x = y + self.mlp(self.ln_2(x))
        else:
            x = x + self.attn(self.ln_1(x), causal)
            x = x + self.mlp(self.ln_2(x))
        return x

    
    def spl(self, x, mem = None):
        if mem is not None:
            x = torch.cat([mem, x],dim=1)
        predicted_output = self(-x, causal = False,fwd =False)
        
        if mem is not None:
            predicted_output = predicted_output[:,mem.size(1):,:]
        sploss = F.mse_loss(
            utils.range_norm(predicted_output.flatten()),
            utils.range_norm(self.block_input.flatten())
        ) 
        return sploss

class Block2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = Attention(config)
        
        if(qrottention):
            self.attn = modules.SSM(config.n_embd,config.dropout,config.bias)
        self.ln_1 = LayerNorm(config)
        self.ln_2 = LayerNorm(config)
        self.mlp = MLP(config)
        self.past_attn = modules.CrossAttention(config.n_embd,config.n_head,config.dropout,config.bias)
        self.past_attn2 = modules.CrossAttention(config.n_embd,config.n_head,config.dropout,config.bias)
        self.ln_3_1 = LayerNorm(config)
        self.ln_4_1 = LayerNorm(config)
        self.ln_3_2 = LayerNorm(config)
        self.ln_4_2 = LayerNorm(config)
    def forward(self, x, past=None, causal=True):
        if isinstance(x,tuple):
            past = x[1]
            x=x[0]
        B, T, D = x.shape
        
        if past is not None:
            P = past.shape[2]
            
            q_flat = x.view(B * T, 1, D)
            
            kv_flat = past.view(B * T, P, D)
            
            resid_flat = self.past_attn(self.ln_3_1(kv_flat), self.ln_3_2(q_flat), causal=False)
            
            resid = (resid_flat.view(B, T, D))
            
            x = x + resid + self.attn(self.ln_1(x), causal)
            
            past = torch.cat([past, resid.unsqueeze(2)], dim=2)
            
            P2 = past.shape[2] 
            q_flat2 = x.view(B * T, 1, D)
            kv_flat2 = past.view(B * T, P2, D)
            
            resid_flat2 = self.past_attn2(self.ln_4_1(kv_flat2), self.ln_4_2(q_flat2), causal=False)
            resid2 = (resid_flat2.view(B, T, D))
            
            x = x + resid2 + self.mlp(self.ln_2(x))
            
            past = torch.cat([past, resid2.unsqueeze(2)], dim=2)
            
            return x, past
            
        else:
            resid = x.unsqueeze(2)
            x = x + self.attn(self.ln_1(x), causal)
            
            resid2 = x.unsqueeze(2)
            x = x + self.mlp(self.ln_2(x))
            
            past = torch.cat([resid, resid2], dim=2)
            
            return x, past

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    internal_vocab_size_multi: int = 10
    internal_vocab_size: int = 0
    internal_block_size: int = 30
    device='cuda'
    optimizer: str = 'adam'
    mem_block_size: int = 64
    batch_size = 64
    step = 0

wn = torch.nn.utils.parametrizations.weight_norm

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.n_embd = config.n_embd
        self.config = config
        if res_attn:
            blocks = nn.ModuleList([Block2(config) for _ in range(config.n_layer)])
        elif ukb:
            blocks = nn.ModuleList([UnifiedKreinBlock(config) for _ in range(config.n_layer)])  
        else:
            blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        

        if(exposemb):
            wpe = nn.Embedding(config.block_size//2, config.n_embd)
        else:
            wpe = nn.Embedding(config.block_size, config.n_embd)

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            
            wpe = wpe,
            drop = nn.Dropout(config.dropout),
            h = blocks,
            ln_f = LayerNorm(config),
        ))

        if orthHead:
            self.lm_head = Orthlin(config.n_embd, config.vocab_size, bias=False)
        else:
            self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)
            self.transformer.wte.weight = self.lm_head.weight 
        
        if wnHead:
            self.lm_head = wn(self.lm_head)
                
        if hcnHead:
            self.lm_head = modules.HCLinear(config.n_embd, config.vocab_size)
            self.transformer.wte.weight = self.lm_head.weight 

        self.wnl = utils.wnormlearnloss(config.n_embd)
        
        self.sphead = (Linear(config.n_embd, 1))
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, module in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(module, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        if OU_proj:
            init_proj = torch.randn(self.n_embd, self.n_embd) / math.sqrt(self.n_embd)
            self.register_buffer('ou_proj', init_proj)
            self.ema_alpha = 0.99999       
            self.proj_strength  = 0.9   
        
        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def forward(self, idx, targets=None, decaying = 0.0, get_emb = False):
        
        device = idx.device
        b, t = idx.size()
        
        if not (exposemb or posembless):
            assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        
        tok_emb = self.transformer.wte(idx)

        if OU_proj:
            if self.training:
                fresh_proj = torch.randn(self.n_embd, self.n_embd, device=idx.device) / math.sqrt(self.n_embd)

                with torch.no_grad():
                    self.ou_proj.copy_(
                        self.ema_alpha * self.ou_proj + 
                        math.sqrt(1.0 - self.ema_alpha**2) * fresh_proj
                    )


                #tok_emb =  (tok_emb @ self.ou_proj)

                scale = math.sqrt(1.0 - self.proj_strength **2)
                tok_emb = scale * tok_emb + self.proj_strength  * (tok_emb @ self.ou_proj)
        
        pos_emb = None
        if exposemb:
            pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
            pos = pos%(self.config.block_size//2)
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (b, t, n_embd)
        elif posembless:
            pass
        else:
            pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (b, t, n_embd)
        
        if(posembdrop):
            noiseramp = torch.randn_like(pos_emb) * (1-decaying)
            demb = (pos_emb+noiseramp) * decaying
            x = tok_emb + demb 
        elif posembless:
            x = tok_emb
        else:
            x = tok_emb + pos_emb
        if get_emb:
            x.retain_grad()
            f_emb =  x
        
        x = self.transformer.drop(x)

        if emb_norm:
            x = x-x.mean()
            #x = utils.rms(x)
            #x = utils.quatnorm(x)
        
        
        
        for i, block in enumerate(self.transformer.h):
            if self.training:
                x = block(x,not Non_causal) 
            else:
                x = block(x) 

        if isinstance(x,tuple):
            x = x[0]
        wnl = 0
        if self.training:
            if wnll:
                wnl = self.wnl(x)
        
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = None# self.lm_head(x)
            reduce = 'mean'
            if wnll or Non_causal:
                reduce = 'none'
            if CCE:
                loss = linear_cross_entropy(x.to(torch.bfloat16), 
                                            self.lm_head.weight.to(torch.bfloat16), 
                                            targets, 
                                            impl='torch_compile',
                                            reduction=reduce,)
            else:
                
                logits = self.lm_head(x)
                
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),reduction=reduce)
            
            
            if self.training:
                if Non_causal and not wnll:
                    loss = loss.reshape(-1, logits.shape[-1])[:,-1].contiguous().mean()
                if spl:
                    loss = loss+ self.transformer.h[0].spl(x)
                if wnll:
                    mean_surprise = ((1.0-wnl.detach())/2.0).mean()*0.99
                    loss = loss / (1+ wnl.detach()*mean_surprise)
                    loss = (loss + wnl).mean()
                if losspred:
                    loss = (loss + F.mse_loss(self.botl(self.lpb(x)).squeeze(-1), loss.detach()))/2.0
            else:
                if wnll or Non_causal:
                    loss = loss.mean()
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
        if get_emb:
            return logits, loss, f_emb, x
        
        return logits, loss
    def loopfwd(self, x, targets):
        b,t,c = x.shape
        blocks=10
        blockouts=torch.zeros(b,t,c,blocks,device=x.device,dtype=x.dtype)
        block_lps=torch.zeros(b,t,blocks,device=x.device,dtype=x.dtype)
        
        for i in range(blocks):
            x = self.transformer.h[0](x) 
            
            lp = self.sphead(x).squeeze(-1) 
            blockouts[...,i] = x
            block_lps[...,i] = lp
        
        bm = block_lps.mean(dim=-2)
        onehot = quantizer.TopKHot.apply(-bm,1)
        x = (blockouts * onehot[:, None, None, :]).sum(dim=-1)
        x_flat = blockouts.permute(0, 1, 3, 2).reshape(-1, blockouts.shape[2])
        x_flat = self.transformer.ln_f(x_flat)
        t_flat = targets.unsqueeze(-1).expand(-1, -1, blocks).reshape(-1)
        
        future_ce_loss = linear_cross_entropy(
            x_flat.to(torch.bfloat16), 
            self.lm_head.weight.to(torch.bfloat16), 
            t_flat, 
            impl='torch_compile', 
            reduction='none'
        ).detach()
        loopaux = F.mse_loss(
            block_lps[..., -1], 
            future_ce_loss[..., 1:],
        )
        return x, loopaux
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        
        mhc_params = []
        decay_params = []
        nodecay_params = []

        for n, p in param_dict.items():
            # shl named modules have slower lr
            if 'shl' in n:
                mhc_params.append(p)
            elif p.dim() >= 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)
        if 'muon' in self.config.optimizer:
            other_groups = [
                {'params': nodecay_params,  'weight_decay': 0.0},
            ]
            muon_groups = [
                {'params': decay_params,    'weight_decay': weight_decay},
                {'params': mhc_params,      'weight_decay': 0.0, 'lr_scale': mhc_lr},
                #{'params': self_lr, 'lr': 1.0},
            ]
        else:
            optim_groups = [
                {'params': nodecay_params,  'weight_decay': 0.0},
                {'params': decay_params,    'weight_decay': weight_decay},
                {'params': mhc_params,      'weight_decay': 0.0, 'lr_scale': mhc_lr},
                #{'params': self_lr, 'lr': 1.0},
            ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        match self.config.optimizer:

            case "adam":
                optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
            case "sgd":
                optimizer = torch.optim.SGD(optim_groups, lr=learning_rate, **extra_args)
            case "phi2":
                optimizer = wackmodel.Phi2Optimizer(optim_groups,lr=learning_rate, **extra_args)
            case "bubbles":
                base = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
                optimizer = optim.ChaosRollbackOptimizer(self, base_optimizer=base)
            case "muon":
                optimizer = torch.optim.Muon(muon_groups, lr=learning_rate)
                optimizer2 = torch.optim.AdamW(other_groups, lr=learning_rate, betas=betas, **extra_args)
                return optimizer,optimizer2
            case "kksm":
                optimizer = UnifiedKreinEntropicOptimizer(optim_groups, lr=learning_rate)
            case "curlfadamw":
                optimizer = RowWiseCurlFreeAdamW(optim_groups, betas=betas, lr=learning_rate)
            #optimizer =torch.optim.Adadelta(optim_groups)#, **extra_args)# lr=learning_rate, **extra_args)
        print(f"using fused AdamW: {use_fused}")
                        
        return optimizer, None

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        
        if isinstance(module, Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            if posembless:
                idx_cond = idx
            else:
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            #         record for super pos
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    