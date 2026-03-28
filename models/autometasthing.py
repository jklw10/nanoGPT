
"""
pruned_binary_gpt2_enhanced.py

GPT-2 → PWL Surrogate → Additive (2^k) signed-bitplane export (+ improved reporting).

What changed vs earlier versions:
- Export weights as *signed additive bitplanes*: W_int = Σ_k 2^k * D_k, with D_k ∈ {-1,0,+1}.
  (binary scheme is a special case; optional NAF/CSD-style signed digits reduce active bits).
- Per-plane storage auto-selects **bitmap** (bitpacked) or **index list** (sparse), so 99% sparse
  planes are stored as (delta-compressible) indices instead of dense bitmaps.
- Expanded reports:
  * per-tensor digit counts, densities, estimated storage, and diamond gadget nnz bounds
  * optional bit-clipping study (drop low bits) to trade accuracy for sparsity/storage
- Diamond forward is now OFF by default (it is enormous); only analytic size + small layer checks
  are run unless explicitly enabled.

Notes:
- This script still uses float activations (and PWL) for inference. The exported representation
  targets *weight storage/computation*. Full XNOR-popcount requires binarized activations too.
- For "random base + mask" deployments: this exporter produces the *mask/digit structure* you
  can reuse; mapping it to a hashed PRNG base is a separate approximation problem.

Dependencies: torch, numpy, scipy, transformers, datasets, tqdm
"""

import os
import math
import time
import argparse
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel


# =============================================================================
# Utilities
# =============================================================================

class Timer:
    def __init__(self):
        self.t0 = time.time()
        self.last = self.t0

    def lap(self, msg: str):
        t = time.time()
        print(f"[time] {msg}: {t - self.last:.2f}s (total {t - self.t0:.2f}s)")
        self.last = t


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def human_bytes(n: float) -> str:
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024.0:
            return f"{n:.2f}{unit}"
        n /= 1024.0
    return f"{n:.2f}PB"


def delta_encode_indices(idx: np.ndarray) -> Dict[str, Any]:
    """
    Delta-encode a sorted uint32 index array.
    Returns a dict with first + deltas tensor (uint16 if possible else uint32).
    """
    idx = idx.astype(np.uint32, copy=False)
    n = int(idx.size)
    if n == 0:
        return {"encoding": "delta", "first": 0, "deltas": torch.empty(0, dtype=torch.uint16)}
    first = int(idx[0])
    deltas = idx[1:].astype(np.uint32) - idx[:-1].astype(np.uint32)
    max_delta = int(deltas.max()) if deltas.size else 0
    if max_delta <= 0xFFFF:
        dt = torch.uint16
        deltas_t = torch.from_numpy(deltas.astype(np.uint16, copy=False))
    else:
        dt = torch.uint32
        deltas_t = torch.from_numpy(deltas.astype(np.uint32, copy=False))
    return {"encoding": "delta", "first": first, "deltas": deltas_t}

def delta_decode_indices(payload: Dict[str, Any]) -> np.ndarray:
    """Inverse of delta_encode_indices."""
    if payload.get("encoding") != "delta":
        raise ValueError("payload is not delta-encoded")
    first = int(payload["first"])
    deltas = payload["deltas"].detach().cpu().numpy().astype(np.uint64, copy=False)
    if deltas.size == 0:
        return np.array([], dtype=np.uint32)
    idx = np.empty(deltas.size + 1, dtype=np.uint64)
    idx[0] = first
    idx[1:] = deltas
    idx = np.cumsum(idx, dtype=np.uint64)
    return idx.astype(np.uint32)

# =============================================================================
# Data Loading
# =============================================================================

def build_lm_contexts(
    tokenizer, ctx_len: int, stride: int, max_ctx: int, seed: int,
    dataset_name: str = "wikitext", dataset_config: str = "wikitext-2-raw-v1",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load train/test contexts as token-id matrices."""

    def load_split(split: str, max_n: int):
        ds = load_dataset(dataset_name, dataset_config, split=split)
        rng = np.random.default_rng(seed)
        idxs = np.arange(len(ds))
        rng.shuffle(idxs)

        ids = []
        eos = tokenizer.eos_token_id
        pbar = tqdm(idxs, desc=f"[data] {split}", ncols=80, leave=False)
        for ii in pbar:
            txt = ds[int(ii)].get("text", "")
            if not isinstance(txt, str) or not txt.strip():
                continue
            enc = tokenizer(txt, add_special_tokens=False)["input_ids"]
            if enc:
                ids.extend(enc)
                ids.append(eos)
            if len(ids) >= max_n * stride + ctx_len + 100:
                break

        contexts = []
        for start in range(0, len(ids) - ctx_len, stride):
            contexts.append(ids[start:start + ctx_len])
            if len(contexts) >= max_n:
                break
        return torch.tensor(contexts, dtype=torch.long)

    X_train = load_split("train", max_ctx)
    X_test = load_split("test", max(1, max_ctx // 8))
    return X_train, X_test


# =============================================================================
# PWL with Bucketize (fast inference) + Hinge export
# =============================================================================

class PWL(nn.Module):
    """Fast PWL using bucketize for inference. Stores knots for hinge export."""
    def __init__(self, x_knots: torch.Tensor, y_knots: torch.Tensor):
        super().__init__()
        xk = x_knots.detach().float()
        yk = y_knots.detach().float()

        dx = xk[1:] - xk[:-1]
        dy = yk[1:] - yk[:-1]
        m = dy / dx
        b = yk[:-1] - m * xk[:-1]

        self.register_buffer("x0", xk[0:1])
        self.register_buffer("xN", xk[-1:])
        self.register_buffer("xk", xk)
        self.register_buffer("yk", yk)
        self.register_buffer("knots_interior", xk[1:-1].contiguous())
        self.register_buffer("m", m.contiguous())
        self.register_buffer("b", b.contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xc = x.clamp(self.x0, self.xN)
        if self.knots_interior.numel() == 0:
            idx = torch.zeros_like(xc, dtype=torch.long)
        else:
            idx = torch.bucketize(xc, self.knots_interior)
        return self.m[idx] * xc + self.b[idx]

    def to_hinge_params(self) -> Dict[str, torch.Tensor]:
        """Convert to explicit hinge representation y = base + m0*x + Σ d_i ReLU(x - h_i)."""
        m = self.m
        m0 = m[0]
        base = self.yk[0] - m0 * self.xk[0]

        if self.xk.numel() > 2:
            hinges = self.xk[1:-1]
            d = m[1:] - m[:-1]
        else:
            hinges = torch.empty(0, device=self.xk.device)
            d = torch.empty(0, device=self.xk.device)

        return {"base": base, "m0": m0, "hinges": hinges, "d": d, "x0": self.x0, "xN": self.xN}


def build_pwl_uniform(func, lo: float, hi: float, segments: int, device) -> PWL:
    x = torch.linspace(lo, hi, segments + 1, device=device)
    y = func(x)
    return PWL(x, y)


def build_pwl_logx(func, lo: float, hi: float, segments: int, device) -> PWL:
    lo = max(lo, 1e-12)
    u = torch.linspace(math.log(lo), math.log(hi), segments + 1, device=device)
    x = torch.exp(u)
    y = func(x)
    return PWL(x, y)


# =============================================================================
# Surrogate GPT-2
# =============================================================================

@dataclass
class SurrogateCfg:
    exp_clip_min: float = -30.0
    ln_eps: float = 1e-5


class SurrogateBlock(nn.Module):
    def __init__(self, teacher_block, pwl_gelu, pwl_exp, pwl_inv, pwl_rsqrt, cfg: SurrogateCfg):
        super().__init__()

        # Store weights as buffers (frozen)
        self.register_buffer("ln_1_w", teacher_block.ln_1.weight.detach().clone())
        self.register_buffer("ln_1_b", teacher_block.ln_1.bias.detach().clone())
        self.register_buffer("ln_2_w", teacher_block.ln_2.weight.detach().clone())
        self.register_buffer("ln_2_b", teacher_block.ln_2.bias.detach().clone())

        attn = teacher_block.attn
        self.register_buffer("c_attn_w", attn.c_attn.weight.detach().clone())
        self.register_buffer("c_attn_b", attn.c_attn.bias.detach().clone())
        self.register_buffer("c_proj_w", attn.c_proj.weight.detach().clone())
        self.register_buffer("c_proj_b", attn.c_proj.bias.detach().clone())

        self.n_head = attn.num_heads
        self.head_dim = attn.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        mlp = teacher_block.mlp
        self.register_buffer("fc_w", mlp.c_fc.weight.detach().clone())
        self.register_buffer("fc_b", mlp.c_fc.bias.detach().clone())
        self.register_buffer("proj_w", mlp.c_proj.weight.detach().clone())
        self.register_buffer("proj_b", mlp.c_proj.bias.detach().clone())

        self.pwl_gelu = pwl_gelu
        self.pwl_exp = pwl_exp
        self.pwl_inv = pwl_inv
        self.pwl_rsqrt = pwl_rsqrt
        self.cfg = cfg

    def layernorm_approx(self, x, w, b):
        mu = x.mean(dim=-1, keepdim=True)
        xc = x - mu
        var = (xc * xc).mean(dim=-1, keepdim=True)
        inv_std = self.pwl_rsqrt(var + self.cfg.ln_eps)
        return (xc * inv_std) * w + b

    def attn_approx(self, x):
        B, T, D = x.shape
        qkv = x @ self.c_attn_w + self.c_attn_b
        q, k, v = qkv.split(D, dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * self.scale
        causal = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal, float("-inf"))

        m = scores.max(dim=-1, keepdim=True).values
        s = torch.clamp(scores - m, min=self.cfg.exp_clip_min, max=0.0)

        e = self.pwl_exp(s)
        p = e * self.pwl_inv(e.sum(dim=-1, keepdim=True))

        out = (p @ v).transpose(1, 2).contiguous().view(B, T, D)
        return out @ self.c_proj_w + self.c_proj_b

    def mlp_approx(self, x):
        h = self.pwl_gelu(x @ self.fc_w + self.fc_b)
        return h @ self.proj_w + self.proj_b

    def forward(self, x):
        x = x + self.attn_approx(self.layernorm_approx(x, self.ln_1_w, self.ln_1_b))
        x = x + self.mlp_approx(self.layernorm_approx(x, self.ln_2_w, self.ln_2_b))
        return x


class SurrogateGPT2(nn.Module):
    def __init__(self, teacher, pwl_gelu, pwl_exp, pwl_inv, pwl_rsqrt, cfg: SurrogateCfg):
        super().__init__()
        self.cfg = cfg
        self.n_layer = teacher.config.n_layer

        self.register_buffer("wte", teacher.transformer.wte.weight.detach().clone())
        self.register_buffer("wpe", teacher.transformer.wpe.weight.detach().clone())

        self.blocks = nn.ModuleList([
            SurrogateBlock(teacher.transformer.h[i], pwl_gelu, pwl_exp, pwl_inv, pwl_rsqrt, cfg)
            for i in range(self.n_layer)
        ])

        self.register_buffer("ln_f_w", teacher.transformer.ln_f.weight.detach().clone())
        self.register_buffer("ln_f_b", teacher.transformer.ln_f.bias.detach().clone())

    def forward_hidden(self, input_ids):
        """Return final hidden state (after ln_f), shape [B, T, d_model]."""
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.wte[input_ids] + self.wpe[pos]

        for blk in self.blocks:
            x = blk(x)

        # Final LN exact
        mu = x.mean(dim=-1, keepdim=True)
        xc = x - mu
        var = (xc * xc).mean(dim=-1, keepdim=True)
        x = (xc * torch.rsqrt(var + 1e-5)) * self.ln_f_w + self.ln_f_b
        return x

    def forward(self, input_ids):
        x = self.forward_hidden(input_ids)
        return x @ self.wte.t()


# =============================================================================
# Calibration (teacher forward to get activation ranges)
# =============================================================================

@torch.no_grad()
def calibrate_activation_ranges(teacher, X_train, device, batch_size=16, max_batches=128):
    """Get min/max of GELU preacts and LN variances."""
    teacher.eval()

    gelu_min, gelu_max = float('inf'), float('-inf')
    var_min, var_max = float('inf'), float('-inf')

    pbar = tqdm(range(0, min(len(X_train), max_batches * batch_size), batch_size),
                desc="[calib]", ncols=80)
    for i in pbar:
        xb = X_train[i:i+batch_size].to(device)
        B, T = xb.shape
        x = teacher.transformer.wte(xb) + teacher.transformer.wpe(torch.arange(T, device=device))

        for block in teacher.transformer.h:
            # LN1 variance
            mu = x.mean(dim=-1, keepdim=True)
            var = ((x - mu) ** 2).mean(dim=-1, keepdim=True)
            var_min = min(var_min, var.min().item())
            var_max = max(var_max, var.max().item())

            x = x + block.attn(block.ln_1(x))[0]

            # LN2 variance
            mu2 = x.mean(dim=-1, keepdim=True)
            var2 = ((x - mu2) ** 2).mean(dim=-1, keepdim=True)
            var_min = min(var_min, var2.min().item())
            var_max = max(var_max, var2.max().item())

            # GELU preact
            ln2_out = block.ln_2(x)
            preact = block.mlp.c_fc(ln2_out)
            gelu_min = min(gelu_min, preact.min().item())
            gelu_max = max(gelu_max, preact.max().item())

            x = x + block.mlp(ln2_out)

    return {'gelu': (gelu_min, gelu_max), 'var': (var_min, var_max)}


# =============================================================================
# Evaluation (top-k metrics + logits cosine + optional hidden procrustes)
# =============================================================================

def parse_positions_arg(pos_str: str, ctx_len: int) -> List[int]:
    """Parse eval positions: '-2', '-1,-2', 'all', 'last32'."""
    s = str(pos_str).strip().lower()
    if s == "all":
        return list(range(ctx_len))
    if s.startswith("last"):
        n = int(s[4:])
        n = max(1, min(n, ctx_len))
        return list(range(-n, 0))
    if not s:
        return [-2]
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def _normalize_positions(positions: List[int], T: int) -> List[int]:
    out, seen = [], set()
    for p in positions:
        p2 = p if p >= 0 else (T + p)
        if 0 <= p2 < T and p2 not in seen:
            out.append(p2)
            seen.add(p2)
    return out


@torch.no_grad()
def eval_agreement(teacher, surrogate, X, device, batch_size=16, positions=(-2,), topk: int = 1,
                   return_details: bool = False):
    """Agreement metrics between teacher logits and surrogate logits at selected positions."""
    teacher.eval()
    surrogate.eval()

    # Try compile for speed
    try:
        t_fwd = torch.compile(teacher, mode="reduce-overhead", fullgraph=False)
        s_fwd = torch.compile(surrogate, mode="reduce-overhead", fullgraph=False)
        _xw = X[:2].to(device)
        _ = t_fwd(_xw).logits[:, -2, :].argmax(-1)
        _ = s_fwd(_xw)[:, -2, :].argmax(-1)
        compiled = True
    except Exception:
        t_fwd = teacher
        s_fwd = surrogate
        compiled = False

    topk = int(topk)
    if topk < 1:
        raise ValueError("--eval_topk must be >= 1")

    agree_top1 = 0
    agree_t1_in_sk = 0
    sum_topk_overlap = 0.0
    sum_logits_cos = 0.0
    total = 0

    tag = "[eval-compiled]" if compiled else "[eval]"
    pbar = tqdm(range(0, len(X), batch_size), desc=tag, ncols=90)
    for i in pbar:
        xb = X[i:i + batch_size].to(device)
        B, T = xb.shape

        pos_idx = _normalize_positions(list(positions), T)
        if not pos_idx:
            raise ValueError(f"No valid positions for T={T}: {positions}")
        pos_t = torch.tensor(pos_idx, device=xb.device, dtype=torch.long)

        t_logits = t_fwd(xb).logits.index_select(1, pos_t)  # [B,P,V]
        s_logits = s_fwd(xb).index_select(1, pos_t)         # [B,P,V]

        t_top1 = t_logits.argmax(dim=-1)                    # [B,P]
        s_top1 = s_logits.argmax(dim=-1)                    # [B,P]
        agree_top1 += (t_top1 == s_top1).sum().item()

        sum_logits_cos += F.cosine_similarity(t_logits.float(), s_logits.float(), dim=-1).sum().item()

        if topk > 1:
            t_topk = torch.topk(t_logits, k=topk, dim=-1).indices
            s_topk = torch.topk(s_logits, k=topk, dim=-1).indices
            agree_t1_in_sk += (t_top1.unsqueeze(-1) == s_topk).any(dim=-1).sum().item()
            inter = (t_topk.unsqueeze(-1) == s_topk.unsqueeze(-2)).any(dim=-1).sum(dim=-1)
            sum_topk_overlap += (inter.float() / topk).sum().item()

        total += B * len(pos_idx)
        pbar.set_postfix(top1=f"{agree_top1/total:.4f}")

    details = {
        "top1": agree_top1 / total,
        "logits_cos": sum_logits_cos / total,
        "n_pairs": total,
        "positions": list(positions),
        "topk": topk,
        "compiled": compiled,
    }
    if topk > 1:
        details["t1_in_sK"] = agree_t1_in_sk / total
        details["topk_overlap"] = sum_topk_overlap / total

    return details if return_details else details["top1"]


def _orthogonal_procrustes(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Solve argmin_R ||A R - B||_F with R orthogonal. A,B: [N,d]."""
    M = A.T @ B
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    return U @ Vh


@torch.no_grad()
def eval_hidden_procrustes(teacher, surrogate, X, device, batch_size=16, positions=(-2,), max_samples: int = 512):
    """Compare final hidden states with an orthogonal Procrustes alignment."""
    teacher.eval()
    surrogate.eval()

    Ht_list, Hs_list = [], []
    n_take = min(int(max_samples), len(X))
    for i in range(0, n_take, batch_size):
        xb = X[i:i + batch_size].to(device)
        B, T = xb.shape
        pos_idx = _normalize_positions(list(positions), T)
        pos_t = torch.tensor(pos_idx, device=xb.device, dtype=torch.long)

        ht = teacher.transformer(input_ids=xb).last_hidden_state.index_select(1, pos_t)
        hs = surrogate.forward_hidden(xb).index_select(1, pos_t)

        Ht_list.append(ht.reshape(-1, ht.shape[-1]).float())
        Hs_list.append(hs.reshape(-1, hs.shape[-1]).float())

    Ht = torch.cat(Ht_list, dim=0)
    Hs = torch.cat(Hs_list, dim=0)

    Ht0 = Ht - Ht.mean(dim=0, keepdim=True)
    Hs0 = Hs - Hs.mean(dim=0, keepdim=True)

    mse_pre = torch.mean((Hs0 - Ht0) ** 2).item()
    cos_pre = torch.mean(F.cosine_similarity(Hs0, Ht0, dim=-1)).item()

    R = _orthogonal_procrustes(Hs0, Ht0)
    Hs1 = Hs0 @ R

    mse_post = torch.mean((Hs1 - Ht0) ** 2).item()
    cos_post = torch.mean(F.cosine_similarity(Hs1, Ht0, dim=-1)).item()

    return {"mse_pre": mse_pre, "mse_post": mse_post, "cos_pre": cos_pre, "cos_post": cos_post,
            "n_pairs": int(Ht0.shape[0]), "positions": list(positions)}


# =============================================================================
# Quantization + Signed Bitplane Decomposition
# =============================================================================

def quantize_weight(W: torch.Tensor, precision: int = 16) -> Tuple[np.ndarray, float]:
    """
    Quantize float weights to fixed-point integers:
      W_int = round(W * scale)
      scale = (2^(p-1)-1) / max(|W|)
    Returns (W_int int64 numpy, scale float).
    """
    Wc = W.detach().cpu().float()
    max_val = float(Wc.abs().max().item())
    if max_val == 0.0:
        return np.zeros(Wc.numel(), dtype=np.int64).reshape(tuple(Wc.shape)), 1.0
    scale = (2 ** (precision - 1) - 1) / max_val
    W_int = torch.round(Wc * scale).to(torch.int64).numpy()
    return W_int, float(scale)


def _required_bits_from_abs(abs_max: int) -> int:
    if abs_max <= 0:
        return 1
    return int(abs_max.bit_length())


def decompose_signed_bitplanes(W_int: np.ndarray, scheme: str = "binary",
                              clip_low_bits: int = 0) -> Tuple[List[int], List[np.ndarray], List[np.ndarray]]:
    """
    Decompose integer matrix into signed additive planes:
      W_int = Σ_k 2^k * D_k, with D_k ∈ {-1,0,+1}
    Returns (bit_positions, pos_masks, neg_masks), where masks are bool arrays with same shape as W_int.
    - scheme='binary': digits are just the binary bits, sign folded into pos/neg.
    - scheme='naf': non-adjacent form digits (CSD-like), typically fewer nonzeros than binary.
    clip_low_bits drops all planes with k < clip_low_bits.
    """
    assert scheme in ("binary", "naf")
    sgn = np.sign(W_int).astype(np.int8)
    absW = np.abs(W_int).astype(np.int64)
    abs_max = int(absW.max())
    n_bits = _required_bits_from_abs(abs_max)
    n_bits = max(n_bits, 1)

    bit_positions: List[int] = []
    pos_masks: List[np.ndarray] = []
    neg_masks: List[np.ndarray] = []

    if scheme == "binary":
        for k in range(clip_low_bits, n_bits):
            bit = ((absW >> k) & 1).astype(bool)
            if not bit.any():
                continue
            pos = bit & (sgn > 0)
            neg = bit & (sgn < 0)
            bit_positions.append(k)
            pos_masks.append(pos)
            neg_masks.append(neg)
        return bit_positions, pos_masks, neg_masks

    # NAF/CSD-like signed digits on magnitudes, then apply sign
    n = absW.copy()
    for k in range(n_bits):  # upper bound; n will reach 0 early for most entries
        if k < clip_low_bits:
            # still need to update n to keep correctness for higher bits
            odd = (n & 1).astype(bool)
            mod4 = (n & 3)
            z = np.zeros_like(n, dtype=np.int64)
            z[odd] = (2 - mod4[odd]).astype(np.int64)  # 1 or -1
            n = (n - z) >> 1
            continue

        odd = (n & 1).astype(bool)
        mod4 = (n & 3)
        z = np.zeros_like(n, dtype=np.int64)
        z[odd] = (2 - mod4[odd]).astype(np.int64)  # 1 or -1
        # signed digit for full weight
        d = z * sgn.astype(np.int64)  # {-1,0,+1}
        pos = (d == 1)
        neg = (d == -1)
        if pos.any() or neg.any():
            bit_positions.append(k)
            pos_masks.append(pos)
            neg_masks.append(neg)
        n = (n - z) >> 1

        if n.max() == 0:
            break

    return bit_positions, pos_masks, neg_masks


# =============================================================================
# Sparse mask storage (bitmap vs indices) + PackedWeight
# =============================================================================

class MaskStore:
    """
    Stores a boolean mask over N entries as either:
      - bitmap: packed bits (np.uint8)
      - indices: sorted uint32 indices of ones
    Optionally stores complement indices with invert=True.
    """
    def __init__(self, mask_bool: np.ndarray, force: str = "auto"):
        assert mask_bool.dtype == np.bool_
        self.N = int(mask_bool.size)
        self.format = None
        self.invert = False
        self.packed: Optional[np.ndarray] = None
        self.indices: Optional[np.ndarray] = None
        self.nnz = int(mask_bool.sum())

        bitmap_bytes = (self.N + 7) // 8
        idx_bytes = self.nnz * 4

        if force == "bitmap":
            choose = "bitmap"
        elif force == "indices":
            choose = "indices"
        else:
            choose = "indices" if idx_bytes < bitmap_bytes else "bitmap"

        # Optional complement indices if that is smaller (rare but cheap to support)
        if choose == "indices":
            nnz0 = self.N - self.nnz
            idx0_bytes = nnz0 * 4
            if idx0_bytes < idx_bytes:
                self.invert = True
                idx = np.flatnonzero(~mask_bool.ravel()).astype(np.uint32)
                self.indices = idx
                self.format = "indices"
                return
            idx = np.flatnonzero(mask_bool.ravel()).astype(np.uint32)
            self.indices = idx
            self.format = "indices"
            return

        self.packed = np.packbits(mask_bool.ravel()).astype(np.uint8)
        self.format = "bitmap"

    def nbytes(self) -> int:
        if self.format == "bitmap":
            return int(self.packed.nbytes)
        if self.format == "indices":
            return int(self.indices.nbytes)
        return 0

    def to_bool(self, shape: Tuple[int, ...]) -> np.ndarray:
        if self.format == "bitmap":
            bits = np.unpackbits(self.packed)[:self.N].astype(bool)
            return bits.reshape(shape)
        assert self.format == "indices"
        out = np.zeros(self.N, dtype=bool)
        if self.indices.size:
            out[self.indices.astype(np.int64)] = True
        if self.invert:
            out = ~out
        return out.reshape(shape)


class PackedWeight:
    """
    Stores a weight tensor as signed additive planes:
      W_int = Σ_k 2^k (pos_k - neg_k)
      W = W_int / scale
    Each plane stores pos/neg masks via MaskStore.
    """
    def __init__(self, name: str, W: torch.Tensor, precision: int = 16,
                 scheme: str = "binary", mask_format: str = "auto",
                 clip_low_bits: int = 0):
        self.name = name
        self.shape = tuple(W.shape)
        self.n_params = int(W.numel())
        self.precision = int(precision)
        self.scheme = scheme
        self.clip_low_bits = int(clip_low_bits)

        W_int, self.scale = quantize_weight(W, precision=precision)
        self.W_int = W_int  # kept only for diagnostics; can be dropped if memory matters

        bits, pos_masks, neg_masks = decompose_signed_bitplanes(W_int, scheme=scheme, clip_low_bits=clip_low_bits)
        self.bits = bits
        self.n_planes = len(bits)

        self.pos: List[MaskStore] = []
        self.neg: List[MaskStore] = []
        self.nnz_pos: List[int] = []
        self.nnz_neg: List[int] = []
        self.bytes_pos: List[int] = []
        self.bytes_neg: List[int] = []

        for pm, nm in zip(pos_masks, neg_masks):
            sp = MaskStore(pm.astype(bool), force=mask_format)
            sn = MaskStore(nm.astype(bool), force=mask_format)
            self.pos.append(sp)
            self.neg.append(sn)
            self.nnz_pos.append(int(pm.sum()))
            self.nnz_neg.append(int(nm.sum()))
            self.bytes_pos.append(sp.nbytes())
            self.bytes_neg.append(sn.nbytes())

        self._recon_cache: Optional[torch.Tensor] = None

    def packed_bytes(self) -> int:
        # masks only; metadata excluded
        return int(sum(self.bytes_pos) + sum(self.bytes_neg))

    def digits(self) -> int:
        return int(sum(self.nnz_pos) + sum(self.nnz_neg))

    def avg_digits_per_weight(self) -> float:
        return float(self.digits()) / max(1, self.n_params)

    def bpw(self) -> float:
        return (self.packed_bytes() * 8) / max(1, self.n_params)

    def reconstruct(self, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Reconstruct float weights from compressed planes (cached)."""
        if self._recon_cache is not None:
            out = self._recon_cache
            if device is not None and out.device != device:
                out = out.to(device)
            return out.to(dtype)

        N = self.n_params
        W_flat = np.zeros(N, dtype=np.int64)
        base_shape = self.shape

        for bit, sp, sn in zip(self.bits, self.pos, self.neg):
            coeff = (1 << int(bit))
            if sp.nnz:
                if sp.format == "indices":
                    idx = sp.indices.astype(np.int64)
                    if sp.invert:
                        # indices store zeros; fill ones then zero out
                        W_flat += coeff
                        W_flat[idx] -= coeff
                    else:
                        W_flat[idx] += coeff
                else:
                    m = sp.to_bool(base_shape).ravel()
                    W_flat[m] += coeff
            if sn.nnz:
                if sn.format == "indices":
                    idx = sn.indices.astype(np.int64)
                    if sn.invert:
                        W_flat -= coeff
                        W_flat[idx] += coeff
                    else:
                        W_flat[idx] -= coeff
                else:
                    m = sn.to_bool(base_shape).ravel()
                    W_flat[m] -= coeff

        Wf = (W_flat.reshape(base_shape).astype(np.float32) / float(self.scale))
        out = torch.from_numpy(Wf)
        self._recon_cache = out  # cache on CPU
        if device is not None:
            out = out.to(device)
        return out.to(dtype)

    def summary(self) -> str:
        dense_bytes = (self.n_params + 7) // 8  # single bitmap
        return (f"{self.name}: shape={list(self.shape)}, planes={self.n_planes}, "
                f"digits={self.digits():,} ({100.0*self.digits()/max(1,self.n_params):.2f}% of weights), "
                f"packed={human_bytes(self.packed_bytes())} ({self.bpw():.2f} bpw) "
                f"[dense_bitmap_one_plane={human_bytes(dense_bytes)}]")

    def plane_table(self, max_rows: int = 32) -> List[Dict[str, Any]]:
        rows = []
        for bit, sp, sn, npz, nnz in zip(self.bits, self.pos, self.neg, self.nnz_pos, self.nnz_neg):
            rows.append({
                "bit": int(bit),
                "nnz_pos": int(npz),
                "nnz_neg": int(nnz),
                "bytes_pos": int(sp.nbytes()),
                "bytes_neg": int(sn.nbytes()),
                "fmt_pos": sp.format + ("~" if sp.invert else ""),
                "fmt_neg": sn.format + ("~" if sn.invert else ""),
            })
        rows.sort(key=lambda r: r["bit"])
        return rows[:max_rows]


# =============================================================================
# Export + Reports
# =============================================================================

def get_all_weight_tensors(surrogate: SurrogateGPT2):
    yield ("wte", surrogate.wte)
    yield ("wpe", surrogate.wpe)
    for i, block in enumerate(surrogate.blocks):
        for wname in ["ln_1_w", "ln_1_b", "ln_2_w", "ln_2_b",
                      "c_attn_w", "c_attn_b", "c_proj_w", "c_proj_b",
                      "fc_w", "fc_b", "proj_w", "proj_b"]:
            yield (f"block_{i}.{wname}", getattr(block, wname))
    yield ("ln_f_w", surrogate.ln_f_w)
    yield ("ln_f_b", surrogate.ln_f_b)


def export_to_bitplanes(surrogate: SurrogateGPT2, precision: int, scheme: str,
                        mask_format: str, clip_low_bits: int) -> Dict[str, Any]:
    print("\n" + "="*80)
    print("EXPORT: SIGNED ADDITIVE BITPLANES")
    print("="*80)
    print(f"  precision={precision} bits, digit_scheme={scheme}, mask_format={mask_format}, clip_low_bits={clip_low_bits}")

    pw_dict: Dict[str, PackedWeight] = {}
    total_params = 0
    total_bytes = 0
    total_digits = 0

    for name, tensor in tqdm(list(get_all_weight_tensors(surrogate)), desc="[export]", ncols=90):
        pw = PackedWeight(name, tensor.cpu(), precision=precision, scheme=scheme,
                          mask_format=mask_format, clip_low_bits=clip_low_bits)
        pw_dict[name] = pw
        total_params += pw.n_params
        total_bytes += pw.packed_bytes()
        total_digits += pw.digits()

    bpw = (total_bytes * 8) / max(1, total_params)
    print(f"\n[export summary]")
    print(f"  tensors:          {len(pw_dict)}")
    print(f"  total params:     {total_params:,}")
    print(f"  total digits:     {total_digits:,}  (avg {total_digits/max(1,total_params):.4f} digits/weight)")
    print(f"  mask bytes:       {human_bytes(total_bytes)}  ({bpw:.2f} bpw masks-only)")

    # PWL params (small)
    block0 = surrogate.blocks[0]
    pwl_params = {}
    pwl_bytes = 0
    for pname, pm in [("gelu", block0.pwl_gelu), ("exp", block0.pwl_exp), ("inv", block0.pwl_inv), ("rsqrt", block0.pwl_rsqrt)]:
        hp = pm.to_hinge_params()
        pwl_params[pname] = {k: v.detach().cpu() for k, v in hp.items()}
        pwl_bytes += sum(int(v.numel()) * 4 for v in hp.values() if isinstance(v, torch.Tensor))

    print(f"  pwl params bytes: {human_bytes(pwl_bytes)}")

    return {"pw_dict": pw_dict, "pwl_params": pwl_params,
            "precision": int(precision), "scheme": scheme, "mask_format": mask_format,
            "clip_low_bits": int(clip_low_bits),
            "total_params": int(total_params), "total_mask_bytes": int(total_bytes), "total_digits": int(total_digits),
            "pwl_bytes": int(pwl_bytes), "n_layer": int(surrogate.n_layer)}


def verify_reconstruction(surrogate: SurrogateGPT2, export: Dict[str, Any], max_tensors: int = 999):
    """Verify reconstructed weights match original (up to quantization error)."""
    print("\n" + "="*80)
    print("VERIFYING RECONSTRUCTION")
    print("="*80)

    max_err = 0.0
    count = 0
    for name, tensor in get_all_weight_tensors(surrogate):
        pw: PackedWeight = export["pw_dict"][name]
        recon = pw.reconstruct(device=torch.device("cpu"))
        err = (tensor.cpu().float() - recon.float()).abs().max().item()
        max_err = max(max_err, float(err))
        count += 1
        if count <= 8 or err > 1e-3:
            print(f"  {name:<22s} err_inf = {err:.6e}  planes={pw.n_planes:2d}  bpw={pw.bpw():5.2f}  digits/weight={pw.avg_digits_per_weight():.4f}")
        if count >= max_tensors:
            break

    print(f"\n  Max reconstruction error (∞-norm): {max_err:.6e}")
    return max_err


def bitplane_report(export: Dict[str, Any], topn: int = 30):
    """Print per-tensor digit/sparsity/storage report."""
    print("\n" + "="*80)
    print("BITPLANE STORAGE REPORT")
    print("="*80)

    rows = []
    for name, pw in export["pw_dict"].items():
        rows.append({
            "name": name,
            "shape": "x".join(map(str, pw.shape)),
            "planes": pw.n_planes,
            "digits": pw.digits(),
            "digits_per_w": pw.avg_digits_per_weight(),
            "mask_bytes": pw.packed_bytes(),
            "bpw": pw.bpw(),
        })
    rows.sort(key=lambda r: r["mask_bytes"], reverse=True)

    header = f"{'tensor':<24} {'shape':>12} {'pl':>3} {'digits':>12} {'dig/w':>8} {'mask':>10} {'bpw':>6}"
    print(header)
    print("-"*len(header))
    for r in rows[:topn]:
        print(f"{r['name'][:24]:<24} {r['shape']:>12} {r['planes']:>3d} {r['digits']:>12,d} {r['digits_per_w']:>8.4f} {human_bytes(r['mask_bytes']):>10} {r['bpw']:>6.2f}")
    if len(rows) > topn:
        print(f"... ({len(rows) - topn} more tensors omitted)")

    total_mask = export["total_mask_bytes"]
    total = total_mask + export["pwl_bytes"]
    print("\n[totals]")
    print(f"  mask bytes:  {human_bytes(total_mask)}")
    print(f"  + PWL bytes: {human_bytes(export['pwl_bytes'])}")
    print(f"  total:       {human_bytes(total)}")



def global_bit_hist(export: Dict[str, Any], topn: int = 32):
    """Aggregate digits/bytes by bit position across all tensors (weight planes only)."""
    agg: Dict[int, Dict[str, int]] = {}
    for pw in export["pw_dict"].values():
        for bit, sp, sn in zip(pw.bits, pw.pos, pw.neg):
            b = int(bit)
            if b not in agg:
                agg[b] = {"digits": 0, "bytes": 0}
            agg[b]["digits"] += int(sp.nnz) + int(sn.nnz)
            agg[b]["bytes"] += int(sp.nbytes()) + int(sn.nbytes())

    bits = sorted(agg.keys())
    if not bits:
        print("\n[global bit hist] no planes")
        return

    print("\n" + "="*80)
    print("GLOBAL BIT HISTOGRAM (ALL TENSORS)")
    print("="*80)
    header = f"{'bit':>4} {'digits':>14} {'bytes':>12}"
    print(header)
    print("-"*len(header))
    for b in bits[:topn]:
        print(f"{b:>4d} {agg[b]['digits']:>14,d} {human_bytes(agg[b]['bytes']):>12}")
    if len(bits) > topn:
        print(f"... ({len(bits)-topn} more bits omitted)")

def diamond_size_report(export: Dict[str, Any], topn: int = 40) -> Dict[str, Any]:
    """
    Analytic nnz report for the explicit diamond gadget.
    We do NOT build the giant sparse matrices; we use closed-form counts.
    Each nonzero digit at plane k yields TWO combine edges (mirror trick).
    """
    print("\n" + "="*80)
    print("DIAMOND GADGET ANALYTIC SIZE REPORT")
    print("="*80)

    rows = []
    totals = dict(expand_nnz=0, chain_nnz=0, combine_nnz=0, total_nnz=0, tensors_2d=0)

    for name, pw in export["pw_dict"].items():
        if len(pw.shape) != 2:
            continue
        d_in, d_out = pw.shape
        if pw.n_planes == 0:
            continue
        n_bits = int(max(pw.bits)) + 1 if pw.bits else 1
        width = 4 * n_bits * d_in

        # Closed-form:
        expand_nnz = 4 * d_in * n_bits
        chain_nnz = 6 * d_in * n_bits * (n_bits - 1) if n_bits >= 2 else 0

        # Exact digits from export
        digits = pw.digits()
        combine_nnz = 2 * digits

        total_nnz = expand_nnz + chain_nnz + combine_nnz
        comb_dens = combine_nnz / max(1, d_out * width)

        rows.append({
            "name": name, "shape": f"{d_in}x{d_out}", "n_bits": n_bits, "width": width,
            "expand": expand_nnz, "chain": chain_nnz, "combine": combine_nnz, "total": total_nnz,
            "comb_dens": comb_dens,
        })

        totals["expand_nnz"] += int(expand_nnz)
        totals["chain_nnz"] += int(chain_nnz)
        totals["combine_nnz"] += int(combine_nnz)
        totals["total_nnz"] += int(total_nnz)
        totals["tensors_2d"] += 1

    rows.sort(key=lambda r: r["total"], reverse=True)

    header = f"{'tensor':<24} {'shape':>11} {'bits':>4} {'width':>10} {'expand':>12} {'chain':>12} {'combine':>12} {'total':>12} {'dens':>9}"
    print(header)
    print("-"*len(header))
    for r in rows[:topn]:
        print(f"{r['name'][:24]:<24} {r['shape']:>11} {r['n_bits']:>4d} {r['width']:>10,d} {r['expand']:>12,d} {r['chain']:>12,d} {r['combine']:>12,d} {r['total']:>12,d} {r['comb_dens']:>9.6f}")
    if len(rows) > topn:
        print(f"... ({len(rows) - topn} more 2D tensors omitted)")

    nnz = totals["total_nnz"]
    print("\n[diamond totals]")
    print(f"  2D tensors:  {totals['tensors_2d']}")
    print(f"  expand nnz:   {totals['expand_nnz']:,}")
    print(f"  chain nnz:    {totals['chain_nnz']:,}")
    print(f"  combine nnz:  {totals['combine_nnz']:,}")
    print(f"  total nnz:    {totals['total_nnz']:,}")
    # Rough CSR bytes per nnz (col idx int32 + val int8) + rowptr ignored
    print(f"  CSR-ish bytes: ~{human_bytes(nnz * 5)} (col int32 + val int8; rowptr excluded)")

    return {"per_tensor": rows, "totals": totals}


# =============================================================================
# Binary Surrogate GPT-2 (weights reconstructed once)
# =============================================================================

class BinarySurrogateBlock(nn.Module):
    """Same compute graph, but weights loaded from PackedWeight reconstruction (cached)."""
    def __init__(self, block: SurrogateBlock, pw_dict: Dict[str, PackedWeight], cfg: SurrogateCfg, idx: int, device):
        super().__init__()
        prefix = f"block_{idx}."
        for wname in ["ln_1_w", "ln_1_b", "ln_2_w", "ln_2_b",
                      "c_attn_w", "c_attn_b", "c_proj_w", "c_proj_b",
                      "fc_w", "fc_b", "proj_w", "proj_b"]:
            pw = pw_dict[prefix + wname]
            self.register_buffer(wname, pw.reconstruct(device=device))

        self.n_head = block.n_head
        self.head_dim = block.head_dim
        self.scale = block.scale
        self.pwl_gelu = block.pwl_gelu
        self.pwl_exp = block.pwl_exp
        self.pwl_inv = block.pwl_inv
        self.pwl_rsqrt = block.pwl_rsqrt
        self.cfg = cfg

    def layernorm_approx(self, x, w, b):
        mu = x.mean(dim=-1, keepdim=True)
        xc = x - mu
        var = (xc * xc).mean(dim=-1, keepdim=True)
        inv_std = self.pwl_rsqrt(var + self.cfg.ln_eps)
        return (xc * inv_std) * w + b

    def attn_approx(self, x):
        B, T, D = x.shape
        qkv = x @ self.c_attn_w + self.c_attn_b
        q, k, v = qkv.split(D, dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * self.scale
        causal = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal, float("-inf"))

        m = scores.max(dim=-1, keepdim=True).values
        s = torch.clamp(scores - m, min=self.cfg.exp_clip_min, max=0.0)
        e = self.pwl_exp(s)
        p = e * self.pwl_inv(e.sum(dim=-1, keepdim=True))

        out = (p @ v).transpose(1, 2).contiguous().view(B, T, D)
        return out @ self.c_proj_w + self.c_proj_b

    def mlp_approx(self, x):
        h = self.pwl_gelu(x @ self.fc_w + self.fc_b)
        return h @ self.proj_w + self.proj_b

    def forward(self, x):
        x = x + self.attn_approx(self.layernorm_approx(x, self.ln_1_w, self.ln_1_b))
        x = x + self.mlp_approx(self.layernorm_approx(x, self.ln_2_w, self.ln_2_b))
        return x


class BinarySurrogateGPT2(nn.Module):
    def __init__(self, surrogate: SurrogateGPT2, export: Dict[str, Any], cfg: SurrogateCfg, device):
        super().__init__()
        self.cfg = cfg
        self.n_layer = surrogate.n_layer
        pw_dict = export["pw_dict"]

        self.register_buffer("wte", pw_dict["wte"].reconstruct(device=device))
        self.register_buffer("wpe", pw_dict["wpe"].reconstruct(device=device))
        self.register_buffer("ln_f_w", pw_dict["ln_f_w"].reconstruct(device=device))
        self.register_buffer("ln_f_b", pw_dict["ln_f_b"].reconstruct(device=device))

        self.blocks = nn.ModuleList([
            BinarySurrogateBlock(surrogate.blocks[i], pw_dict, cfg, i, device=device)
            for i in range(self.n_layer)
        ])

    def forward_hidden(self, input_ids):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.wte[input_ids] + self.wpe[pos]

        for blk in self.blocks:
            x = blk(x)

        mu = x.mean(dim=-1, keepdim=True)
        xc = x - mu
        var = (xc * xc).mean(dim=-1, keepdim=True)
        x = (xc * torch.rsqrt(var + 1e-5)) * self.ln_f_w + self.ln_f_b
        return x

    def forward(self, input_ids):
        x = self.forward_hidden(input_ids)
        return x @ self.wte.t()


# =============================================================================
# Save / Load (torch.save, masks already compressed)
# =============================================================================

def save_export(export: Dict[str, Any], output_path: str, delta_encode: bool = False) -> int:
    """
    Save export (masks + metadata + PWL params).
    Note: indices/bitmaps are stored as torch tensors.
    """
    save = {
        "precision": export["precision"],
        "scheme": export["scheme"],
        "mask_format": export["mask_format"],
        "clip_low_bits": export["clip_low_bits"],
        "n_layer": export["n_layer"],
        "pwl_params": export["pwl_params"],
        "weights": {},
    }
    for name, pw in export["pw_dict"].items():
        planes = []
        for bit, sp, sn in zip(pw.bits, pw.pos, pw.neg):
            planes.append({
                "bit": int(bit),
                "pos": {"format": sp.format, "invert": bool(sp.invert),
                        "data": (torch.from_numpy(sp.packed.copy()) if sp.format == "bitmap" else (delta_encode_indices(sp.indices) if delta_encode else torch.from_numpy(sp.indices.copy())))},
                "neg": {"format": sn.format, "invert": bool(sn.invert),
                        "data": (torch.from_numpy(sn.packed.copy()) if sn.format == "bitmap" else (delta_encode_indices(sn.indices) if delta_encode else torch.from_numpy(sn.indices.copy())))},
                "nnz_pos": int(sp.nnz),
                "nnz_neg": int(sn.nnz),
            })
        save["weights"][name] = {"shape": list(pw.shape), "scale": float(pw.scale), "planes": planes}

    torch.save(save, output_path)
    return int(os.path.getsize(output_path))


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--ctx_len", type=int, default=128)
    ap.add_argument("--max_train", type=int, default=8000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)

    # Eval
    ap.add_argument("--eval_positions", default="-2",
                    help="Comma-separated positions, or 'all', or 'lastN' (e.g. last32).")
    ap.add_argument("--eval_topk", type=int, default=1)
    ap.add_argument("--eval_hidden_procrustes", action="store_true")
    ap.add_argument("--eval_hidden_samples", type=int, default=512)

    # PWL segments
    ap.add_argument("--gelu_seg", type=int, default=2048)
    ap.add_argument("--exp_seg", type=int, default=4096)
    ap.add_argument("--inv_seg", type=int, default=4096)
    ap.add_argument("--rsqrt_seg", type=int, default=2048)

    # Quant/planes
    ap.add_argument("--precision", type=int, default=16)
    ap.add_argument("--digit_scheme", choices=["binary", "naf"], default="binary",
                    help="binary=standard bits; naf=signed non-adjacent form (usually fewer digits)")
    ap.add_argument("--mask_format", choices=["auto", "bitmap", "indices"], default="auto",
                    help="plane mask storage format")
    ap.add_argument("--clip_low_bits", type=int, default=0,
                    help="drop planes with bit position < k (approximation that increases sparsity)")

    # Reports
    ap.add_argument("--report_topn", type=int, default=30)
    ap.add_argument("--diamond_report", action="store_true", help="print analytic diamond gadget size report")

    # Export
    ap.add_argument("--output", default="pruned_binary_gpt2_export.pt")
    ap.add_argument("--skip_export", action="store_true")
    ap.add_argument("--skip_binary_eval", action="store_true")
    ap.add_argument("--save_export", action="store_true")
    ap.add_argument("--delta_encode_indices", action="store_true", help="delta-encode index planes when saving")

    args = ap.parse_args()

    tmr = Timer()
    set_seed(args.seed)
    device = torch.device(args.device)

    # Tokenizer + teacher
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    teacher = GPT2LMHeadModel.from_pretrained(args.model).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    tmr.lap("load teacher")

    # Data
    X_train, X_test = build_lm_contexts(tokenizer, args.ctx_len, stride=64, max_ctx=args.max_train, seed=args.seed)
    print(f"[data] train={len(X_train)} test={len(X_test)} ctx_len={args.ctx_len}")
    tmr.lap("load data")

    # Calibrate
    ranges = calibrate_activation_ranges(teacher, X_train, device, args.batch_size)
    gelu_lo, gelu_hi = ranges["gelu"]
    var_lo, var_hi = ranges["var"]
    var_lo = max(var_lo, 1e-8) + 1e-5
    var_hi = var_hi + 1e-5
    print(f"\n[calib] GELU preact: [{gelu_lo:.4f}, {gelu_hi:.4f}]")
    print(f"[calib] LN var:      [{var_lo:.3e}, {var_hi:.3e}]")
    tmr.lap("calibrate")

    # PWL
    cfg = SurrogateCfg()
    pwl_gelu = build_pwl_uniform(F.gelu, gelu_lo, gelu_hi, args.gelu_seg, device)
    pwl_exp = build_pwl_uniform(torch.exp, -30, 0, args.exp_seg, device)
    pwl_inv = build_pwl_logx(lambda x: 1/x, 1, args.ctx_len, args.inv_seg, device)
    pwl_rsqrt = build_pwl_logx(torch.rsqrt, var_lo, var_hi, args.rsqrt_seg, device)
    tmr.lap("build PWL")

    # Surrogate
    surrogate = SurrogateGPT2(teacher, pwl_gelu, pwl_exp, pwl_inv, pwl_rsqrt, cfg).to(device).eval()
    tmr.lap("build surrogate")

    # Evaluate float surrogate
    eval_positions = parse_positions_arg(args.eval_positions, args.ctx_len)
    float_details = eval_agreement(teacher, surrogate, X_test, device,
                                   batch_size=args.batch_size,
                                   positions=eval_positions,
                                   topk=args.eval_topk,
                                   return_details=True)
    print(f"\n[result] FLOAT SURROGATE top1={float_details['top1']:.4f}  logits_cos={float_details['logits_cos']:.4f}")
    if float_details.get("topk", 1) > 1:
        print(f"         teacher_top1_in_surrogate_topk={float_details['t1_in_sK']:.4f}  topk_overlap={float_details['topk_overlap']:.4f}")
    print(f"         positions={float_details['positions']}  compiled={float_details['compiled']}")
    tmr.lap("evaluate float surrogate")

    if args.eval_hidden_procrustes:
        hrep = eval_hidden_procrustes(teacher, surrogate, X_test, device,
                                      batch_size=max(1, args.batch_size // 4),
                                      positions=eval_positions,
                                      max_samples=args.eval_hidden_samples)
        print("\n[hidden Procrustes] teacher.transformer vs surrogate.forward_hidden")
        print(f"  mse_pre={hrep['mse_pre']:.6e}  cos_pre={hrep['cos_pre']:.4f}")
        print(f"  mse_post={hrep['mse_post']:.6e} cos_post={hrep['cos_post']:.4f}")
        print(f"  n_pairs={hrep['n_pairs']}  positions={hrep['positions']}")

    if args.skip_export:
        return

    # Export weights to signed bitplanes
    export = export_to_bitplanes(surrogate, precision=args.precision, scheme=args.digit_scheme,
                                 mask_format=args.mask_format, clip_low_bits=args.clip_low_bits)
    tmr.lap("export bitplanes")

    # Verify reconstruction
    max_err = verify_reconstruction(surrogate, export)
    tmr.lap("verify reconstruction")

    # Reports
    bitplane_report(export, topn=args.report_topn)
    if args.diamond_report:
        diamond_size_report(export)

    # Optional: evaluate the reconstructed-weight surrogate (should match float surrogate closely)
    if not args.skip_binary_eval:
        binary_model = BinarySurrogateGPT2(surrogate, export, cfg, device=device).to(device).eval()
        bin_details = eval_agreement(teacher, binary_model, X_test, device,
                                     batch_size=args.batch_size,
                                     positions=eval_positions,
                                     topk=args.eval_topk,
                                     return_details=True)
        print(f"\n[result] RECONSTRUCTED-WEIGHT model top1={bin_details['top1']:.4f}  logits_cos={bin_details['logits_cos']:.4f}")
        if bin_details.get("topk", 1) > 1:
            print(f"         teacher_top1_in_binary_topk={bin_details['t1_in_sK']:.4f}  topk_overlap={bin_details['topk_overlap']:.4f}")
        tmr.lap("evaluate reconstructed-weight model")

    if args.save_export:
        fsz = save_export(export, args.output, delta_encode=args.delta_encode_indices)
        print(f"\n[saved] {args.output}  size={human_bytes(fsz)}")

    # Final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"float surrogate top1:           {float_details['top1']:.4f}")
    if not args.skip_binary_eval:
        print(f"reconstructed-weight top1:      {bin_details['top1']:.4f}")
    print(f"max weight recon error (inf):   {max_err:.3e}")
    print(f"mask bytes (total):             {human_bytes(export['total_mask_bytes'])}  ({export['total_mask_bytes']*8/export['total_params']:.2f} bpw)")
    print(f"+ PWL bytes:                    {human_bytes(export['pwl_bytes'])}")
    print(f"digit_scheme={args.digit_scheme}  mask_format={args.mask_format}  clip_low_bits={args.clip_low_bits}")
    print("="*80)


if __name__ == "__main__":
    main()
