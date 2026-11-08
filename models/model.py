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

cardnewt        = False   
rootkan         = True  
rootkan_highdeg = False
rootkan_pade    = False
rootkan_glu     = False


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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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

        # -------------------------------------------------------------
        # TOGGLES FOR THE 3 TESTS (Set exactly one to True)
        # -------------------------------------------------------------
        # rootkan_highdeg = True   # Option 1: Degree d=5 RootKAN + Safe Halley Solve
        # rootkan_pade    = False  # Option 2: Padé Rational Root Function (bounded tails)
        # rootkan_glu     = False  # Option 3: Root-Gated GLU (SwiGLU-style, one-sided gate)

        if pattmlp:
            self.thing = modules.PattLayer(
                config.n_embd,
                config.n_embd,
                mlp_ptok,
            )
            return

        self.c_fc    = Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = Linear(hidden_dim, config.n_embd, bias=config.bias)

        if hcmlp:
            self.c_fc = modules.HCLinear(config.n_embd, hidden_dim, 1.0, False)
            self.c_proj = modules.HCLinear(hidden_dim, config.n_embd, 1.0, False)
        if qrot:
            self.c_proj  = Linear(2 * config.n_embd, config.n_embd, bias=config.bias)

        # -------------------------------------------------------------
        # 1. OPTION 1: HIGH DEGREE (d=5) ROOT-KAN PARAMETERS
        # -------------------------------------------------------------
        if rootkan_highdeg:
            d = 5
            k = torch.arange(1, d + 1, dtype=torch.float32)
            cheb_nodes = torch.cos((2 * k - 1) / (2 * d) * math.pi)
            self.kan_roots_d5 = nn.Parameter(cheb_nodes.view(d, 1).repeat(1, hidden_dim))
            self.kan_scale_d5 = nn.Parameter(torch.randn(hidden_dim) * 0.0)

        # -------------------------------------------------------------
        # 2. OPTION 2: PADÉ RATIONAL ROOT PARAMETERS
        # -------------------------------------------------------------
        if rootkan_pade:
            # Degree 2 numerator roots + learnable denominator stiffness
            k = torch.arange(1, 3, dtype=torch.float32)
            cheb_nodes = torch.cos((2 * k - 1) / (2 * 2) * math.pi)  # [+0.7071, -0.7071]
            self.pade_roots = nn.Parameter(cheb_nodes.view(2, 1).repeat(1, hidden_dim))
            self.pade_scale = nn.Parameter(torch.randn(hidden_dim) * 0.1)
            self.pade_alpha = nn.Parameter(torch.zeros(hidden_dim))  # softplus(0) ~ 0.69

        # -------------------------------------------------------------
        # 3. OPTION 3: ROOT-GATED GLU (SwiGLU Style)
        # -------------------------------------------------------------
        if rootkan_glu:
            k = torch.arange(1, 4, dtype=torch.float32)
            cheb_nodes = torch.cos((2 * k - 1) / (2 * 3) * math.pi)  # [+0.866, 0.0, -0.866]
            self.glu_roots = nn.Parameter(cheb_nodes.view(3, 1).repeat(1, hidden_dim))
            self.glu_scale = nn.Parameter(torch.randn(hidden_dim) * 0.1)

        # (Existing parameters for previous runs)
        if cardnewt:
            self.cardnewt_a = nn.Parameter(torch.zeros(hidden_dim))
            self.cardnewt_sigma = nn.Parameter(torch.ones(hidden_dim) * 0.5)
            k = torch.arange(1, 4, dtype=torch.float32)        
            cheb_nodes = torch.cos((2 * k - 1) / (2 * 3) * math.pi)
            self.kan_roots = nn.Parameter(cheb_nodes.view(3, 1).repeat(1, hidden_dim))
            self.kan_scale = nn.Parameter(torch.randn(hidden_dim) * 0.1)
            
        if rootkan:
            k = torch.arange(1, 4, dtype=torch.float32)
            cheb_nodes = torch.cos((2 * k - 1) / (2 * 3) * math.pi)
            self.kan_roots = nn.Parameter(cheb_nodes.view(3, 1).repeat(1, hidden_dim))
            self.kan_scale = nn.Parameter(torch.randn(hidden_dim) * 0.1)

        self.dropout = nn.Dropout(config.dropout)
        self.dac = ShiftedReLU(hidden_dim)
        self.univ = utils.Univfun(config.n_embd)

    
    def _rootkan_forward(self, u):
        orig_dtype = u.dtype
        u_f32 = u.float()

        
        roots = self.kan_roots.float()
        scale = self.kan_scale.view(1, 1, -1).float()
        d = 2

        denom = torch.sqrt(1.0 + roots.pow(2))
        prod = (u_f32 - roots[0].view(1, 1, -1)) / denom[0].view(1, 1, -1)
        for k in range(1, d):
            prod = prod * ((u_f32 - roots[k].view(1, 1, -1)) / denom[k].view(1, 1, -1))

        v = prod * scale

        return v.to(orig_dtype)

    # -------------------------------------------------------------
    # CUBIC ROOT-KAN
    # -------------------------------------------------------------
    def _rootkan_forward2(self, u):
        orig_dtype = u.dtype
        u_f32 = u.float()
        roots = self.kan_roots.float()
        scale = self.kan_scale.view(1, 1, -1).float()

        denom = torch.sqrt(1.0 + roots.pow(2))
        prod = (u_f32 - roots[0].view(1, 1, -1)) / denom[0].view(1, 1, -1)
        for k in range(1, 3):
            prod = prod * ((u_f32 - roots[k].view(1, 1, -1)) / denom[k].view(1, 1, -1))

        v = prod * scale

        cbrt_v_sq = (v.pow(2) + 1e-5).pow(1.0 / 3.0)
        y = v / (1.0 + cbrt_v_sq)

        #for _ in range(2):
        #    y_sq = y.pow(2)
        #    G = y_sq * y + y - v
        #    G_prime = 3.0 * y_sq + 1.0
        #    G_double_prime = 6.0 * y
        #    num = 2.0 * G * G_prime
        #    denom_h = 2.0 * G_prime.pow(2) - G * G_double_prime
        #    y = y - num / denom_h

        return y.to(orig_dtype)

    # -----------------------------------------------------------------
    # OPTION 1 FORWARD: Degree d=5 RootKAN with NaN-Safe Halley Solve
    # -----------------------------------------------------------------
    
    def _highdeg_forward(self, u):
        orig_dtype = u.dtype
        u_f32 = u.float()
        roots = self.kan_roots_d5.float()
        scale = self.kan_scale_d5.view(1, 1, -1).float()

        # 1. Bounded polynomial fluctuation
        u_bounded = torch.tanh(u_f32)
        denom = torch.sqrt(1.0 + roots.pow(2))
        prod = (u_bounded - roots[0].view(1, 1, -1)) / denom[0].view(1, 1, -1)
        for k in range(1, 5):
            prod = prod * ((u_bounded - roots[k].view(1, 1, -1)) / denom[k].view(1, 1, -1))

        # 2. Feed ONLY the polynomial fluctuation into the cubic compressor
        v_poly = scale * prod

        cbrt_v_sq = (v_poly.pow(2) + 1e-5).pow(1.0 / 3.0)
        y = v_poly / (1.0 + cbrt_v_sq)

        for _ in range(3):
            y_sq = y.pow(2)
            G = y_sq * y + y - v_poly
            G_prime = 3.0 * y_sq + 1.0
            G_double_prime = 6.0 * y
            num = 2.0 * G * G_prime
            denom_h = 2.0 * G_prime.pow(2) - G * G_double_prime
            y = y - num / denom_h

        # 3. True residual addition AFTER the solve
        out = u_f32 + y
        out = y
        return out.to(orig_dtype)

    # -----------------------------------------------------------------
    # OPTION 2 FORWARD: Padé Rational Root Function (Bounded Asymptote)
    # -----------------------------------------------------------------
    def _pade_forward(self, u):
        roots = self.pade_roots
        scale = self.pade_scale.view(1, 1, -1)
        alpha = F.softplus(self.pade_alpha).view(1, 1, -1)

        denom_r = torch.sqrt(1.0 + roots.pow(2))
        f1 = (u - roots[0].view(1, 1, -1)) / denom_r[0].view(1, 1, -1)
        f2 = (u - roots[1].view(1, 1, -1)) / denom_r[1].view(1, 1, -1)

        # Pure polynomial numerator (degree 2) / linear denominator (degree 1) => Linear Asymptote
        num = scale * (f1 * f2)
        den = torch.sqrt(1.0 + alpha * u.pow(2))
        return num / den

    # -----------------------------------------------------------------
    # OPTION 3 FORWARD: Root-Gated Linear Unit (SwiGLU Style)
    # -----------------------------------------------------------------
    def _glu_forward(self, u):
        roots = self.glu_roots
        scale = self.glu_scale.view(1, 1, -1)

        denom_r = torch.sqrt(1.0 + roots.pow(2))
        prod = (u - roots[0].view(1, 1, -1)) / denom_r[0].view(1, 1, -1)
        for k in range(1, 3):
            prod = prod * ((u - roots[k].view(1, 1, -1)) / denom_r[k].view(1, 1, -1))

        # Root polynomial modulates the opening threshold of the gate
        gate = u + scale * prod
        # u * sigmoid(gate) guarantees 0 on the left, linear identity on the right!
        return u * torch.sigmoid(gate)

    
    def _cardnewt_forward(self, u):
        orig_dtype = u.dtype
        u_f32 = u.float()
        
        # 1. Cubic factor product (provides the u^2 numerator gradient!)
        roots = self.kan_roots.float()
        scale = self.kan_scale.view(1, 1, -1).float()
        denom = torch.sqrt(1.0 + roots.pow(2))
        
        prod = (u_f32 - roots[0].view(1, 1, -1)) / denom[0].view(1, 1, -1)
        for k in range(1, 3):
            prod = prod * ((u_f32 - roots[k].view(1, 1, -1)) / denom[k].view(1, 1, -1))

        # Target B is cubic in u: B ~ u^3
        B = prod * scale

        # 2. Cardano parameters
        a = self.cardnewt_a.view(1, 1, -1).float()
        sigma = self.cardnewt_sigma.view(1, 1, -1).float()
        r_sq = (a.pow(2) / 3.0) + sigma.pow(2) + 1e-4

        # 3. Pade seed: matches linear and cubic regimes
        cbrt_B_sq = (B.pow(2) + 1e-5).pow(1.0 / 3.0)
        y = B / (r_sq + cbrt_B_sq)

        # 4. 3 unrolled Newton steps: G'(y) >= sigma^2 + 1e-4 > 0 strictly
        for _ in range(3):
            y_sq = y.pow(2)
            G = (y_sq + a * y + r_sq) * y - B
            G_prime = 3.0 * y_sq + 2.0 * a * y + r_sq
            y = y - (G / G_prime)

        return y.to(orig_dtype)
    # -----------------------------------------------------------------
    # Forward Pass Routing
    # -----------------------------------------------------------------
    def forward(self, x):
        if testing_mlp:
            return self.univ(x)
        if pattmlp:
            return self.thing(x)
        if qrot:
            return self.qrot(x)

        x = self.c_fc(x)

        # Execute chosen test
        if rootkan_highdeg:
            x = self._highdeg_forward(x)
        elif rootkan_pade:
            x = self._pade_forward(x)
        elif rootkan_glu:
            x = self._glu_forward(x)
        elif cardnewt:
            x = self._cardnewt_forward(x)
        elif rootkan:
            x = self._rootkan_forward(x)
        elif diffac:
            x = self.dac(x)
        else:
            x = self.gelu(x)

        x = self.c_proj(x)
        x = self.dropout(x)
        return x
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
    