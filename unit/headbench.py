"""
Isolated Vocabulary Head Microbenchmark with CCE Memory Cuts
Compares:
  1. baseline_dot_standard    : Exact PyTorch full-vocab (F.linear + F.cross_entropy)
  2. baseline_dot_chunked     : Chunked full-vocab (1024-token chunks)
  3. v11_zipf_dot_3k          : Original 3k bank (Autograd saves all 8 chunks)
  4. v11_zipf_dot_cce_ckpt    : 3k bank + CCE checkpointing (Throws away logits per chunk)
  5. v11_zipf_dot_cce_manual  : 3k bank + Custom Autograd (Saves only 1D LSE, in-place index_add_)
"""

import gc
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.non_ml.triton_kmeans import run_engine_v11_triton, warmup_graphs

torch._inductor.config.disable_cpp_codegen = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

N_TOKENS = 4096       # B * T (64 * 64)
EMB_DIM = 256
VOCAB_SIZE = 50304
K_BANK = 3072
K_HEAVY = 1024
ROUTE_CHUNK = 512
WARMUP_STEPS = 15
BENCH_STEPS = 40
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Tree Builder
# ---------------------------------------------------------------------------

def _principal_axis_order(sub: torch.Tensor) -> torch.Tensor:
    centered = sub - sub.mean(dim=0, keepdim=True)
    v = torch.randn(sub.shape[1], device=sub.device, dtype=sub.dtype)
    v = v / (torch.linalg.norm(v) + 1e-9)
    for _ in range(3):
        z = centered @ v
        v = centered.T @ z
        v = v / (torch.linalg.norm(v) + 1e-9)
    return torch.argsort(centered @ v)


def _split_into_m_leaves(W: torch.Tensor, idxs: torch.Tensor, m_leaves: int, out: list):
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
def build_v11_tree(W: torch.Tensor, depth: int = 8, k_coarse: int = 16, slack: float = 1.10):
    V = W.shape[0]
    M = 1 << depth
    k_c = min(k_coarse, max(2, M // 4))
    S_bucket = math.ceil(slack * V / M)

    C_coarse = run_engine_v11_triton(W, k=k_c, sub_n=min(16384, V), half_x=True, use_graph=True)
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

    buckets = []
    for c in range(k_c):
        if m_alloc[c] == 0:
            continue
        idxs = (coarse_assign == c).nonzero(as_tuple=True)[0]
        _split_into_m_leaves(W, idxs, m_alloc[c], buckets)

    leaf_tokens = torch.zeros((M, S_bucket), device=W.device, dtype=torch.long)
    leaf_valid = torch.zeros((M, S_bucket), device=W.device, dtype=torch.bool)
    for leaf_id, idxs in enumerate(buckets):
        cnt = idxs.shape[0]
        if cnt > 0:
            leaf_tokens[leaf_id, :cnt] = idxs
            leaf_valid[leaf_id, :cnt] = True
    return leaf_tokens, leaf_valid, S_bucket

# ---------------------------------------------------------------------------
# Custom Autograd CCE Kernel (Recomputes logits in backward, in-place index_add_)
# ---------------------------------------------------------------------------

class FusedZipfChunkCCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h_c, centroids, bank_idx, t_c, exact_cut, tail_scale, bank_valid):
        sub_c = centroids[bank_idx]
        pos_c = centroids[t_c]

        bank_logits = h_c @ sub_c.T
        pos_score = (h_c * pos_c).sum(dim=-1, keepdim=True)

        is_target = (bank_idx.unsqueeze(0) == t_c.unsqueeze(1))
        masked_logits = bank_logits.masked_fill(is_target | (~bank_valid.unsqueeze(0)), -1e9)

        exact_logits = masked_logits[:, :exact_cut]
        tail_logits = masked_logits[:, exact_cut:]

        overall_max = torch.maximum(pos_score, masked_logits.max(dim=1, keepdim=True).values)
        pos_exp = torch.exp(pos_score - overall_max)
        exact_exp = torch.exp(exact_logits - overall_max).sum(dim=1, keepdim=True)
        tail_exp = tail_scale * torch.exp(tail_logits - overall_max).sum(dim=1, keepdim=True)

        total_lse = overall_max + torch.log(pos_exp + exact_exp + tail_exp)
        loss = (-pos_score + total_lse).sum()

        # CCE CUT: Save only h_c, t_c, bank_idx, and 1D total_lse vector (2 KB!)
        ctx.save_for_backward(h_c, centroids, bank_idx, t_c, total_lse, bank_valid)
        ctx.exact_cut = exact_cut
        ctx.tail_scale = tail_scale
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        h_c, centroids, bank_idx, t_c, total_lse, bank_valid = ctx.saved_tensors

        with torch.no_grad():
            sub_c = centroids[bank_idx]
            pos_c = centroids[t_c]

            # Recompute logits in backward (takes ~0.1 ms, frees 100% of forward activation VRAM)
            bank_logits = h_c @ sub_c.T
            pos_score = (h_c * pos_c).sum(dim=-1, keepdim=True)

            is_target = (bank_idx.unsqueeze(0) == t_c.unsqueeze(1))
            masked_logits = bank_logits.masked_fill(is_target | (~bank_valid.unsqueeze(0)), -1e9)

            probs = torch.exp(masked_logits - total_lse)
            probs[:, ctx.exact_cut:] *= ctx.tail_scale
            probs = probs.masked_fill(is_target | (~bank_valid.unsqueeze(0)), 0.0)

            pos_prob = torch.exp(pos_score - total_lse)
            d_pos = -(1.0 - pos_prob) * grad_output

            # Accumulate dh
            grad_h = (probs @ sub_c) * grad_output + d_pos * pos_c

            # In-place streaming accumulation into d_centroids (bypasses Autograd graph)
            grad_centroids = torch.zeros_like(centroids)
            grad_sub_c = (probs.T @ (h_c * grad_output))
            grad_centroids.index_add_(0, bank_idx, grad_sub_c)
            grad_centroids.index_add_(0, t_c, d_pos * h_c)

        return grad_h, grad_centroids, None, None, None, None, None

# ---------------------------------------------------------------------------
# Candidate Heads
# ---------------------------------------------------------------------------

class StandardDotHead(nn.Module):
    def __init__(self, W: nn.Parameter):
        super().__init__()
        self.W = W

    def forward(self, h: torch.Tensor, targets: torch.Tensor):
        logits = F.linear(h, self.W)
        return F.cross_entropy(logits, targets)


class ChunkedDotHead(nn.Module):
    def __init__(self, W: nn.Parameter, chunk_size: int = 1024):
        super().__init__()
        self.W = W
        self.chunk_size = chunk_size

    def forward(self, h: torch.Tensor, targets: torch.Tensor):
        N = h.shape[0]
        tot = 0.0
        for i in range(0, N, self.chunk_size):
            h_c = h[i:i + self.chunk_size]
            t_c = targets[i:i + self.chunk_size]
            tot = tot + F.cross_entropy(h_c @ self.W.T, t_c, reduction="sum")
        return tot / N


class ZipfRoutedHeadBase(nn.Module):
    def __init__(self, W: nn.Parameter):
        super().__init__()
        self.W = W
        self.depth = 8
        self.M = 1 << self.depth
        self.K_heavy = K_HEAVY

        tail_vocab = torch.arange(self.K_heavy, VOCAB_SIZE, device=W.device)
        lt, lv, S_bucket = build_v11_tree(W[tail_vocab].detach(), depth=self.depth)
        self.S_bucket = S_bucket

        K_rem = max(1, K_BANK - self.K_heavy)
        max_p = max(1, (K_rem - 1) // self.S_bucket)
        self.P = min(self.M - 1, max(1, min((2 * K_rem) // (3 * self.S_bucket), max_p)))
        self.K_routed = self.P * self.S_bucket
        self.K_tail = max(1, K_BANK - self.K_heavy - self.K_routed)

        self.register_buffer("heavy_idx", torch.arange(self.K_heavy, device=W.device))
        self.register_buffer("leaf_tokens", tail_vocab[lt])
        self.register_buffer("leaf_valid", lv)

    def _route_chunk(self, h_c, leaf_centroids):
        with torch.no_grad():
            coarse = h_c @ leaf_centroids.T
            cluster_heat = F.softmax(coarse, dim=-1).sum(dim=0)
            sorted_c = torch.argsort(cluster_heat, descending=True)
            routed_c = sorted_c[:self.P]
            unrouted_c = sorted_c[self.P:]

            routed_idx = self.leaf_tokens[routed_c].reshape(-1)
            routed_valid = self.leaf_valid[routed_c].reshape(-1)

            rand_u = torch.randint(0, self.M - self.P, (self.K_tail,), device=h_c.device)
            rand_s = torch.randint(0, self.S_bucket, (self.K_tail,), device=h_c.device)
            tail_idx = self.leaf_tokens[unrouted_c[rand_u], rand_s]
            tail_valid = self.leaf_valid[unrouted_c[rand_u], rand_s]

            bank_idx = torch.cat([self.heavy_idx, routed_idx, tail_idx], dim=0)
            heavy_valid = torch.ones(self.K_heavy, dtype=torch.bool, device=h_c.device)
            bank_valid = torch.cat([heavy_valid, routed_valid, tail_valid], dim=0)
        return bank_idx, bank_valid


# 1. Standard Autograd (saves all 8 chunks in memory)
class ZipfRoutedHeadAutograd(ZipfRoutedHeadBase):
    def forward(self, h: torch.Tensor, targets: torch.Tensor):
        centroids = self.W
        with torch.no_grad():
            leaf_embs = centroids[self.leaf_tokens] * self.leaf_valid.unsqueeze(-1).to(centroids.dtype)
            leaf_counts = self.leaf_valid.sum(dim=1, keepdim=True).clamp(min=1).to(centroids.dtype)
            leaf_centroids = leaf_embs.sum(dim=1) / leaf_counts

        N = h.shape[0]
        tot = 0.0
        tail_scale = float((self.M - self.P) * self.S_bucket) / float(self.K_tail)
        exact_cut = self.K_heavy + self.K_routed

        for i in range(0, N, ROUTE_CHUNK):
            h_c = h[i:i + ROUTE_CHUNK]
            t_c = targets[i:i + ROUTE_CHUNK]
            bank_idx, bank_valid = self._route_chunk(h_c, leaf_centroids)

            sub_c = centroids[bank_idx]
            pos_c = centroids[t_c]
            bank_logits = h_c @ sub_c.T
            pos_score = (h_c * pos_c).sum(dim=-1)

            is_target = (bank_idx.unsqueeze(0) == t_c.unsqueeze(1))
            masked_logits = bank_logits.masked_fill(is_target | (~bank_valid.unsqueeze(0)), -1e9)

            exact_logits = masked_logits[:, :exact_cut]
            tail_logits = masked_logits[:, exact_cut:]

            overall_max = torch.maximum(pos_score.unsqueeze(1), masked_logits.max(dim=1, keepdim=True).values)
            pos_exp = torch.exp(pos_score.unsqueeze(1) - overall_max)
            exact_exp = torch.exp(exact_logits - overall_max).sum(dim=1, keepdim=True)
            tail_exp = tail_scale * torch.exp(tail_logits - overall_max).sum(dim=1, keepdim=True)

            total_lse = overall_max + torch.log(pos_exp + exact_exp + tail_exp)
            tot = tot + (-pos_score.unsqueeze(1) + total_lse).sum()

        return tot / N


# 2. Checkpointed CCE (discards chunk logits immediately, recomputes during backward)
class ZipfRoutedHeadCCE_Ckpt(ZipfRoutedHeadBase):
    @staticmethod
    def _chunk_eval(h_c, centroids, bank_idx, t_c, exact_cut, tail_scale, bank_valid):
        sub_c = centroids[bank_idx]
        pos_c = centroids[t_c]
        bank_logits = h_c @ sub_c.T
        pos_score = (h_c * pos_c).sum(dim=-1)

        is_target = (bank_idx.unsqueeze(0) == t_c.unsqueeze(1))
        masked_logits = bank_logits.masked_fill(is_target | (~bank_valid.unsqueeze(0)), -1e9)

        exact_logits = masked_logits[:, :exact_cut]
        tail_logits = masked_logits[:, exact_cut:]

        overall_max = torch.maximum(pos_score.unsqueeze(1), masked_logits.max(dim=1, keepdim=True).values)
        pos_exp = torch.exp(pos_score.unsqueeze(1) - overall_max)
        exact_exp = torch.exp(exact_logits - overall_max).sum(dim=1, keepdim=True)
        tail_exp = tail_scale * torch.exp(tail_logits - overall_max).sum(dim=1, keepdim=True)

        total_lse = overall_max + torch.log(pos_exp + exact_exp + tail_exp)
        return (-pos_score.unsqueeze(1) + total_lse).sum()

    def forward(self, h: torch.Tensor, targets: torch.Tensor):
        centroids = self.W
        with torch.no_grad():
            leaf_embs = centroids[self.leaf_tokens] * self.leaf_valid.unsqueeze(-1).to(centroids.dtype)
            leaf_counts = self.leaf_valid.sum(dim=1, keepdim=True).clamp(min=1).to(centroids.dtype)
            leaf_centroids = leaf_embs.sum(dim=1) / leaf_counts

        N = h.shape[0]
        tot = 0.0
        tail_scale = float((self.M - self.P) * self.S_bucket) / float(self.K_tail)
        exact_cut = self.K_heavy + self.K_routed

        for i in range(0, N, ROUTE_CHUNK):
            h_c = h[i:i + ROUTE_CHUNK]
            t_c = targets[i:i + ROUTE_CHUNK]
            bank_idx, bank_valid = self._route_chunk(h_c, leaf_centroids)

            # Checkpoint per chunk: Discards logits immediately, recomputes 3k chunk in backward
            chunk_loss = checkpoint(
                self._chunk_eval, h_c, centroids, bank_idx, t_c, exact_cut, tail_scale, bank_valid,
                use_reentrant=False
            )
            tot = tot + chunk_loss

        return tot / N


# 3. Manual Autograd CCE (Saves only 1D LSE, in-place index_add_)
class ZipfRoutedHeadCCE_Manual(ZipfRoutedHeadBase):
    def forward(self, h: torch.Tensor, targets: torch.Tensor):
        centroids = self.W
        with torch.no_grad():
            leaf_embs = centroids[self.leaf_tokens] * self.leaf_valid.unsqueeze(-1).to(centroids.dtype)
            leaf_counts = self.leaf_valid.sum(dim=1, keepdim=True).clamp(min=1).to(centroids.dtype)
            leaf_centroids = leaf_embs.sum(dim=1) / leaf_counts

        N = h.shape[0]
        tot = 0.0
        tail_scale = float((self.M - self.P) * self.S_bucket) / float(self.K_tail)
        exact_cut = self.K_heavy + self.K_routed

        for i in range(0, N, ROUTE_CHUNK):
            h_c = h[i:i + ROUTE_CHUNK]
            t_c = targets[i:i + ROUTE_CHUNK]
            bank_idx, bank_valid = self._route_chunk(h_c, leaf_centroids)

            chunk_loss = FusedZipfChunkCCE.apply(h_c, centroids, bank_idx, t_c, exact_cut, tail_scale, bank_valid)
            tot = tot + chunk_loss

        return tot / N

# ---------------------------------------------------------------------------
# Isolated Profiling Engine
# ---------------------------------------------------------------------------

def profile_isolated(make_fn, h_pool, t_pool, compile_it=True):
    # Enforce fresh garbage collection & purge CUDA caching allocator
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    base_mem = torch.cuda.memory_allocated() / (1024 ** 2)
    module, W = make_fn()
    fn = torch.compile(module) if compile_it else module

    # Warmup
    for i in range(WARMUP_STEPS):
        h, t = h_pool[i % len(h_pool)], t_pool[i % len(t_pool)]
        W.grad = None
        h.grad = None
        loss = fn(h, t)
        loss.backward()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    fwd_times, bwd_times = [], []
    start_event = torch.cuda.Event(enable_timing=True)
    mid_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for i in range(BENCH_STEPS):
        h, t = h_pool[i % len(h_pool)], t_pool[i % len(t_pool)]
        W.grad = None
        h.grad = None

        start_event.record()
        loss = fn(h, t)
        mid_event.record()
        loss.backward()
        end_event.record()

        torch.cuda.synchronize()
        fwd_times.append(start_event.elapsed_time(mid_event))
        bwd_times.append(mid_event.elapsed_time(end_event))

    fwd_ms = sum(fwd_times) / len(fwd_times)
    bwd_ms = sum(bwd_times) / len(bwd_times)
    tot_ms = fwd_ms + bwd_ms

    peak_total_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
    act_delta_vram = max(0.0, peak_total_vram - base_mem)

    # Cleanup after candidate finishes
    del fn, module, W
    gc.collect()
    torch.cuda.empty_cache()

    return fwd_ms, bwd_ms, tot_ms, peak_total_vram, act_delta_vram


def main():
    print(f"=== Isolated Head Profiler: CCE Memory Cuts Comparison ===")
    print(f"Tokens: {N_TOKENS} | Dim: {EMB_DIM} | Vocab: {VOCAB_SIZE} | Bank: {K_BANK}")
    print(f"Warmup: {WARMUP_STEPS} steps | Measure: {BENCH_STEPS} steps\n")

    warmup_graphs([(16, EMB_DIM)], sub_n=min(16384, VOCAB_SIZE - K_HEAVY), half_x=True, device=DEVICE)

    torch.manual_seed(42)
    h_pool = [torch.randn(N_TOKENS, EMB_DIM, device=DEVICE, requires_grad=True) for _ in range(8)]
    t_pool = [torch.randint(0, VOCAB_SIZE, (N_TOKENS,), device=DEVICE) for _ in range(8)]

    # Builders to ensure clean memory per method
    experiments = {
        "baseline_dot_standard":   (lambda: (StandardDotHead(nn.Parameter(torch.randn(VOCAB_SIZE, EMB_DIM, device=DEVICE) * 0.02)), None), True),
        "baseline_dot_chunked":    (lambda: (ChunkedDotHead(nn.Parameter(torch.randn(VOCAB_SIZE, EMB_DIM, device=DEVICE) * 0.02), chunk_size=1024), None), True),
        "v11_zipf_dot_3k":         (lambda: (ZipfRoutedHeadAutograd(nn.Parameter(torch.randn(VOCAB_SIZE, EMB_DIM, device=DEVICE) * 0.02)), None), True),
        "v11_zipf_dot_cce_ckpt":   (lambda: (ZipfRoutedHeadCCE_Ckpt(nn.Parameter(torch.randn(VOCAB_SIZE, EMB_DIM, device=DEVICE) * 0.02)), None), True),
        "v11_zipf_dot_cce_manual": (lambda: (ZipfRoutedHeadCCE_Manual(nn.Parameter(torch.randn(VOCAB_SIZE, EMB_DIM, device=DEVICE) * 0.02)), None), False),
    }

    # Fix returning W for gradient clearing
    def make_wrapper(cls):
        def _build():
            W = nn.Parameter(torch.randn(VOCAB_SIZE, EMB_DIM, device=DEVICE) * 0.02)
            return cls(W), W
        return _build

    experiments["baseline_dot_standard"]   = (make_wrapper(StandardDotHead), True)
    experiments["baseline_dot_chunked"]    = (make_wrapper(lambda W: ChunkedDotHead(W, 1024)), True)
    experiments["v11_zipf_dot_3k"]         = (make_wrapper(ZipfRoutedHeadAutograd), True)
    experiments["v11_zipf_dot_cce_ckpt"]   = (make_wrapper(ZipfRoutedHeadCCE_Ckpt), True)
    experiments["v11_zipf_dot_cce_manual"] = (make_wrapper(ZipfRoutedHeadCCE_Manual), False)

    results = {}
    for name, (builder, should_compile) in experiments.items():
        print(f"Profiling {name:<26} (compiled={should_compile})...")
        fwd, bwd, tot, peak, act = profile_isolated(builder, h_pool, t_pool, compile_it=should_compile)
        results[name] = (fwd, bwd, tot, peak, act)

    base_tot = results["baseline_dot_standard"][2]
    print("\n" + "=" * 108)
    print(f"{'Method':<26} | {'Fwd (ms)':<9} | {'Bwd (ms)':<9} | {'Total (ms)':<10} | {'Peak VRAM':<10} | {'Act VRAM':<10} | {'Speedup'}")
    print("-" * 108)
    for name, (fwd, bwd, tot, peak, act) in results.items():
        sp = base_tot / tot
        print(f"{name:<26} | {fwd:7.2f} ms | {bwd:7.2f} ms | {tot:8.2f} ms | {peak:7.1f} MB | {act:7.1f} MB | {sp:5.2f}x")
    print("=" * 108)


if __name__ == "__main__":
    torch._inductor.config.triton.cudagraphs = False
    main()