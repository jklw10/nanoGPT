import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

# -----------------------------------------------------------------------------
# TRITON KERNEL: Fused Top-K Attention
# -----------------------------------------------------------------------------

@triton.jit
def _fused_topk_fwd_kernel(
    Q, K, V, 
    Out, Indices, Weights,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_on,
    sm_scale,
    H, N_CTX, 
    HEAD_DIM: tl.constexpr, 
    K_SPARSE: tl.constexpr,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_K_SPARSE: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(N_CTX, BLOCK_M)
    
    # Grid logic
    pid_batch = pid // (num_pid_m * H)
    pid_head = (pid // num_pid_m) % H
    pid_m = pid % num_pid_m

    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    
    # Load Q
    Q_base = Q + (pid_batch * stride_qb) + (pid_head * stride_qh)
    q_ptrs = Q_base + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    mask_q = (offs_m[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM)
    q = tl.load(q_ptrs, mask=mask_q, other=0.0)

    # --- Initialize Running Top-K Registers ---
    # Init with -inf. 
    # CRITICAL: We rely on these being -inf for logic, but must handle them carefully in math.
    run_vals = tl.full([BLOCK_M, BLOCK_K_SPARSE], float("-inf"), dtype=tl.float32)
    run_inds = tl.full([BLOCK_M, BLOCK_K_SPARSE], -1, dtype=tl.int32)

    # K Pointers
    K_base = K + (pid_batch * stride_kb) + (pid_head * stride_kh)
    offs_n = tl.arange(0, BLOCK_N)
    k_range = tl.arange(0, BLOCK_K_SPARSE)

    # Iterate over K blocks
    for start_n in range(0, N_CTX, BLOCK_N):
        k_ptr = K_base + (start_n + offs_n[None, :]) * stride_kn + offs_k[:, None] * stride_kk
        mask_k = ((start_n + offs_n[None, :]) < N_CTX) & (offs_k[:, None] < HEAD_DIM)
        k_block = tl.load(k_ptr, mask=mask_k, other=0.0)
        
        # Compute Scores
        qk = tl.dot(q, k_block)
        qk *= sm_scale
        
        # Causal Masking
        global_n = start_n + offs_n
        is_causal = global_n[None, :] > offs_m[:, None]
        qk = tl.where(is_causal, float("-inf"), qk)
        
        # --- Update Top-K Candidates from this Block ---
        cur_scores = qk 
        
        # We extract up to K_SPARSE candidates from the current block to ensure we don't miss any.
        for cand_k in range(K_SPARSE):
            # 1. Find max in current remaining scores
            block_max, block_idx_local = tl.max(cur_scores, 1, return_indices=True)
            block_idx = block_idx_local + start_n
            
            # 2. Mask picked so we find the next highest in the next iteration
            mask_picked = (tl.arange(0, BLOCK_N)[None, :] == block_idx_local[:, None])
            cur_scores = tl.where(mask_picked, float("-inf"), cur_scores)
            
            # 3. Bubble Insertion
            candidate_val = block_max
            candidate_ind = block_idx
            
            for rank in range(K_SPARSE):
                mask_col = (k_range == rank)[None, :]
                
                # --- FIXED EXTRACTION LOGIC ---
                # OLD BROKEN WAY: curr_r_val = tl.sum(run_vals * mask_col, 1) 
                # This caused NaN because -inf * 0 = NaN.
                
                # NEW SAFE WAY: Use tl.where to zero out non-target columns safely
                # If mask_col is 0, we take 0.0. If mask_col is 1, we take run_vals (even if -inf).
                # Summing 0.0 + (-inf) = -inf. Safe.
                curr_r_val = tl.sum(tl.where(mask_col, run_vals, 0.0), 1)
                curr_r_ind = tl.sum(run_inds * mask_col, 1) # Integers are safe to multiply
                
                swap = candidate_val > curr_r_val
                
                new_val_for_pos = tl.where(swap, candidate_val, curr_r_val)
                new_ind_for_pos = tl.where(swap, candidate_ind, curr_r_ind)
                
                candidate_val = tl.where(swap, curr_r_val, candidate_val)
                candidate_ind = tl.where(swap, curr_r_ind, candidate_ind)
                
                run_vals = tl.where(mask_col, new_val_for_pos[:, None], run_vals)
                run_inds = tl.where(mask_col, new_ind_for_pos[:, None], run_inds)
                
    # --- Softmax on Top K ---
    max_val = tl.max(run_vals, 1)
    
    # SAFETY: Ensure we don't do -inf - (-inf) -> NaN
    max_is_inf = max_val == float("-inf")
    safe_max = tl.where(max_is_inf, 0.0, max_val)
    
    num = tl.exp(run_vals - safe_max[:, None])
    denom = tl.sum(num, 1)
    
    # SAFETY: Avoid division by zero if entire row was masked
    probs = num / tl.where(denom == 0, 1.0, denom)[:, None]

    # --- Save Indices & Weights ---
    base_idx_ptr = Indices + (pid_batch * H * N_CTX * K_SPARSE) + \
                   (pid_head * N_CTX * K_SPARSE) + \
                   (offs_m[:, None] * K_SPARSE) + \
                   k_range[None, :]
                   
    base_w_ptr = Weights + (pid_batch * H * N_CTX * K_SPARSE) + \
                 (pid_head * N_CTX * K_SPARSE) + \
                 (offs_m[:, None] * K_SPARSE) + \
                 k_range[None, :]

    mask_store = (offs_m[:, None] < N_CTX) & (k_range[None, :] < K_SPARSE)
    
    tl.store(base_idx_ptr, run_inds, mask=mask_store)
    tl.store(base_w_ptr, probs, mask=mask_store)

    # --- Gather V ---
    V_base = V + (pid_batch * stride_vb) + (pid_head * stride_vh)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    for i in range(K_SPARSE):
        mask_col = (k_range == i)[None, :]
        idx = tl.sum(run_inds * mask_col, 1)
        w = tl.sum(probs * mask_col, 1)
        
        valid = idx != -1
        safe_idx = tl.where(valid, idx, 0)
        
        v_ptr = V_base + safe_idx[:, None] * stride_vn + offs_k[None, :] * stride_vk
        mask_v = valid[:, None] & (offs_k[None, :] < HEAD_DIM)
        val = tl.load(v_ptr, mask=mask_v, other=0.0)
        
        acc += val * w[:, None]

    # Store Out
    O_base = Out + (pid_batch * stride_ob) + (pid_head * stride_oh)
    O_ptr = O_base + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_on)
    mask_out = (offs_m[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM)
    tl.store(O_ptr, acc, mask=mask_out)

# -----------------------------------------------------------------------------
# PYTHON HELPERS
# -----------------------------------------------------------------------------

def run_fused_forward(q, k, v, k_sparse):
    B, H, T, D = q.shape
    scale = 1.0 / math.sqrt(D)
    
    out = torch.zeros_like(q)
    indices = torch.full((B, H, T, k_sparse), -1, dtype=torch.int32, device=q.device)
    weights = torch.zeros((B, H, T, k_sparse), dtype=torch.float32, device=q.device)
    
    BLOCK_DMODEL = triton.next_power_of_2(D)
    BLOCK_K_SPARSE = triton.next_power_of_2(k_sparse)
    
    grid = (B * H * triton.cdiv(T, 64), 1, 1)
    
    _fused_topk_fwd_kernel[grid](
        q, k, v, out, indices, weights,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        scale, H, T, 
        HEAD_DIM=D, K_SPARSE=k_sparse,
        BLOCK_M=64, BLOCK_N=64,
        BLOCK_DMODEL=BLOCK_DMODEL, 
        BLOCK_K_SPARSE=BLOCK_K_SPARSE
    )
    return out, indices, weights

# -----------------------------------------------------------------------------
# AUTOGRAD FUNCTION
# -----------------------------------------------------------------------------

class TopKHotFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, k_sparse):
        out, indices, weights = run_fused_forward(q, k, v, k_sparse)
        ctx.save_for_backward(q, k, v, indices, weights)
        ctx.k_sparse = k_sparse
        ctx.scale = 1.0 / math.sqrt(q.size(-1))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, indices, weights = ctx.saved_tensors
        k_sparse = ctx.k_sparse
        scale = ctx.scale
        
        grad_output = grad_output.contiguous()
        
        B, H, T, D = grad_output.shape
        grad_flat = grad_output.view(B*H, T, D)
        q_flat = q.contiguous().view(B*H, T, D)
        k_flat = k.contiguous().view(B*H, T, D)
        v_flat = v.contiguous().view(B*H, T, D)
        
        indices_flat = indices.view(B*H, T, k_sparse).long()
        weights_flat = weights.view(B*H, T, k_sparse)
        
        dQ_flat = torch.zeros_like(q_flat)
        dK_flat = torch.zeros_like(k_flat)
        dV_flat = torch.zeros_like(v_flat)
        
        valid_mask = (indices_flat != -1)

        # Precompute Term: sum(P * (dO . V))
        row_dot_sum = torch.zeros(B*H, T, 1, device=q.device, dtype=q.dtype)
        
        for i in range(k_sparse):
            idx = indices_flat[..., i]
            w   = weights_flat[..., i]
            mask = valid_mask[..., i]
            safe_idx = idx.masked_fill(~mask, 0)
            
            # Gather V: [BH, T, D]
            v_selected = v_flat.gather(1, safe_idx.unsqueeze(-1).expand(-1, -1, D))
            
            # dV Accumulation
            grad_w = grad_flat * w.unsqueeze(-1)
            grad_w = grad_w.masked_fill(~mask.unsqueeze(-1), 0)
            dV_flat.scatter_add_(1, safe_idx.unsqueeze(-1).expand(-1, -1, D), grad_w)
            
            # Softmax Gradient Prep
            dot_val = (grad_flat * v_selected).sum(dim=-1, keepdim=True)
            row_dot_sum += (dot_val * w.unsqueeze(-1)).masked_fill(~mask.unsqueeze(-1), 0)

        # Compute dQ and dK
        for i in range(k_sparse):
            idx = indices_flat[..., i]
            w   = weights_flat[..., i]
            mask = valid_mask[..., i]
            safe_idx = idx.masked_fill(~mask, 0)

            v_selected = v_flat.gather(1, safe_idx.unsqueeze(-1).expand(-1, -1, D))
            k_selected = k_flat.gather(1, safe_idx.unsqueeze(-1).expand(-1, -1, D))
            
            dot_val = (grad_flat * v_selected).sum(dim=-1, keepdim=True)
            
            dS_val = (w.unsqueeze(-1) * (dot_val - row_dot_sum)) * scale
            dS_val = dS_val.masked_fill(~mask.unsqueeze(-1), 0)
            
            dQ_flat += dS_val * k_selected
            
            dK_val = dS_val * q_flat
            dK_flat.scatter_add_(1, safe_idx.unsqueeze(-1).expand(-1, -1, D), dK_val)

        return dQ_flat.view(B, H, T, D), dK_flat.view(B, H, T, D), dV_flat.view(B, H, T, D), None

class KattentionFused(nn.Module):
    def __init__(self, config, k_sparse=4):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.k_sparse = k_sparse

    def forward(self, x, causal=True):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2).contiguous()
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2).contiguous()
        
        y = TopKHotFused.apply(q, k, v, self.k_sparse)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)