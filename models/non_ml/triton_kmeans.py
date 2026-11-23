# v11_triton.py
from dataclasses import dataclass
import time
import torch
import triton
import triton.language as tl


# =====================================================================
# 1. Triton Kernels
# =====================================================================

@triton.jit
def _k_dot2_weighted(
    X_ptr, assign_ptr, V1_ptr, V2_ptr, mv1_ptr, mv2_ptr, w_ptr, out1_ptr, out2_ptr,
    N, D, stride_xn, stride_xd, stride_vm, stride_vd,
    APPLY_W: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    n0 = pid * BLOCK_N
    n_offs = n0 + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N
    n_last = tl.minimum(n0 + BLOCK_N - 1, N - 1)

    c_min = tl.load(assign_ptr + n0)
    c_max = tl.load(assign_ptr + n_last)

    acc1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_N,), dtype=tl.float32)

    if c_min == c_max:
        for d0 in range(0, D, BLOCK_D):
            d_offs = d0 + tl.arange(0, BLOCK_D)
            d_mask = d_offs < D
            m2 = n_mask[:, None] & d_mask[None, :]
            x = tl.load(X_ptr + n_offs[:, None] * stride_xn + d_offs[None, :] * stride_xd, mask=m2, other=0.0).to(tl.float32)
            v1 = tl.load(V1_ptr + c_min * stride_vm + d_offs * stride_vd, mask=d_mask, other=0.0)
            v2 = tl.load(V2_ptr + c_min * stride_vm + d_offs * stride_vd, mask=d_mask, other=0.0)
            acc1 += tl.sum(x * v1[None, :], axis=1)
            acc2 += tl.sum(x * v2[None, :], axis=1)
        mv1 = tl.zeros((BLOCK_N,), dtype=tl.float32) + tl.load(mv1_ptr + c_min)
        mv2 = tl.zeros((BLOCK_N,), dtype=tl.float32) + tl.load(mv2_ptr + c_min)
    else:
        c = tl.load(assign_ptr + n_offs, mask=n_mask, other=0)
        for d0 in range(0, D, BLOCK_D):
            d_offs = d0 + tl.arange(0, BLOCK_D)
            d_mask = d_offs < D
            m2 = n_mask[:, None] & d_mask[None, :]
            x = tl.load(X_ptr + n_offs[:, None] * stride_xn + d_offs[None, :] * stride_xd, mask=m2, other=0.0).to(tl.float32)
            v1 = tl.load(V1_ptr + c[:, None] * stride_vm + d_offs[None, :] * stride_vd, mask=m2, other=0.0)
            v2 = tl.load(V2_ptr + c[:, None] * stride_vm + d_offs[None, :] * stride_vd, mask=m2, other=0.0)
            acc1 += tl.sum(x * v1, axis=1)
            acc2 += tl.sum(x * v2, axis=1)
        mv1 = tl.load(mv1_ptr + c, mask=n_mask, other=0.0)
        mv2 = tl.load(mv2_ptr + c, mask=n_mask, other=0.0)

    z1 = acc1 - mv1
    z2 = acc2 - mv2
    if APPLY_W:
        w = tl.load(w_ptr + n_offs, mask=n_mask, other=0.0)
        z1 = z1 * w
        z2 = z2 * w

    tl.store(out1_ptr + n_offs, z1, mask=n_mask)
    tl.store(out2_ptr + n_offs, z2, mask=n_mask)


@triton.jit
def _k_dot1(
    X_ptr, assign_ptr, V_ptr, dot_ptr,
    N, D, stride_xn, stride_xd, stride_vm, stride_vd,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    n0 = pid * BLOCK_N
    n_offs = n0 + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N
    n_last = tl.minimum(n0 + BLOCK_N - 1, N - 1)

    c_min = tl.load(assign_ptr + n0)
    c_max = tl.load(assign_ptr + n_last)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    if c_min == c_max:
        for d0 in range(0, D, BLOCK_D):
            d_offs = d0 + tl.arange(0, BLOCK_D)
            d_mask = d_offs < D
            m2 = n_mask[:, None] & d_mask[None, :]
            x = tl.load(X_ptr + n_offs[:, None] * stride_xn + d_offs[None, :] * stride_xd, mask=m2, other=0.0).to(tl.float32)
            v = tl.load(V_ptr + c_min * stride_vm + d_offs * stride_vd, mask=d_mask, other=0.0)
            acc += tl.sum(x * v[None, :], axis=1)
    else:
        c = tl.load(assign_ptr + n_offs, mask=n_mask, other=0)
        for d0 in range(0, D, BLOCK_D):
            d_offs = d0 + tl.arange(0, BLOCK_D)
            d_mask = d_offs < D
            m2 = n_mask[:, None] & d_mask[None, :]
            x = tl.load(X_ptr + n_offs[:, None] * stride_xn + d_offs[None, :] * stride_xd, mask=m2, other=0.0).to(tl.float32)
            v = tl.load(V_ptr + c[:, None] * stride_vm + d_offs[None, :] * stride_vd, mask=m2, other=0.0)
            acc += tl.sum(x * v, axis=1)

    tl.store(dot_ptr + n_offs, acc, mask=n_mask)


@triton.jit
def _k_segmented_accum(
    X_ptr, assign_ptr, a1_ptr, a2_ptr, Out1_ptr, Out2_ptr,
    N, D, stride_xn, stride_xd, stride_om, stride_od,
    HAS_TWO: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)

    n0 = pid_n * BLOCK_N
    n_offs = n0 + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N
    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    n_last = tl.minimum(n0 + BLOCK_N - 1, N - 1)
    c_min = tl.load(assign_ptr + n0)
    c_max = tl.load(assign_ptr + n_last)

    m2 = n_mask[:, None] & d_mask[None, :]
    x = tl.load(X_ptr + n_offs[:, None] * stride_xn + d_offs[None, :] * stride_xd, mask=m2, other=0.0).to(tl.float32)
    a1 = tl.load(a1_ptr + n_offs, mask=n_mask, other=0.0)

    if c_min == c_max:
        s1 = tl.sum(a1[:, None] * x, axis=0)
        tl.atomic_add(Out1_ptr + c_min * stride_om + d_offs * stride_od, s1, mask=d_mask)
        if HAS_TWO:
            a2 = tl.load(a2_ptr + n_offs, mask=n_mask, other=0.0)
            s2 = tl.sum(a2[:, None] * x, axis=0)
            tl.atomic_add(Out2_ptr + c_min * stride_om + d_offs * stride_od, s2, mask=d_mask)
    else:
        c_vec = tl.load(assign_ptr + n_offs, mask=n_mask, other=-1)
        if HAS_TWO:
            a2 = tl.load(a2_ptr + n_offs, mask=n_mask, other=0.0)
        else:
            a2 = a1
        c = c_min
        while c <= c_max:
            mask_c = n_mask & (c_vec == c)
            if tl.sum(mask_c.to(tl.int32), axis=0) > 0:
                w1 = tl.where(mask_c, a1, 0.0)
                s1 = tl.sum(w1[:, None] * x, axis=0)
                tl.atomic_add(Out1_ptr + c * stride_om + d_offs * stride_od, s1, mask=d_mask)
                if HAS_TWO:
                    w2 = tl.where(mask_c, a2, 0.0)
                    s2 = tl.sum(w2[:, None] * x, axis=0)
                    tl.atomic_add(Out2_ptr + c * stride_om + d_offs * stride_od, s2, mask=d_mask)
            c = c + 1


@triton.jit
def _k_2d_mode_all(
    u1_ptr, u2_ptr, w_ptr, starts_ptr, ends_ptr, wsum_ptr, del1_ptr, del2_ptr,
    BLOCK_N: tl.constexpr,
):
    c = tl.program_id(0)
    start = tl.load(starts_ptr + c)
    end = tl.load(ends_ptr + c)

    if end <= start:
        tl.store(del1_ptr + c, 1.5e-6)
        tl.store(del2_ptr + c, 0.0)
    else:
        w_sum = tl.load(wsum_ptr + c)
        var1_num = tl.sum(tl.zeros((1,), dtype=tl.float32), axis=0)

        n0 = start
        while n0 < end:
            offs = n0 + tl.arange(0, BLOCK_N)
            mask = offs < end
            u1 = tl.load(u1_ptr + offs, mask=mask, other=0.0)
            w = tl.load(w_ptr + offs, mask=mask, other=0.0)
            var1_num = var1_num + tl.sum(w * u1 * u1, axis=0)
            n0 = n0 + BLOCK_N

        sig1 = tl.sqrt(tl.maximum(var1_num / tl.maximum(w_sum, 1e-6), 1e-12))
        cA1 = -0.75 * sig1
        cA2 = tl.sum(tl.zeros((1,), dtype=tl.float32), axis=0)
        cB1 = 0.75 * sig1
        cB2 = tl.sum(tl.zeros((1,), dtype=tl.float32), axis=0)

        for _ in range(3):
            swA = tl.sum(tl.zeros((1,), dtype=tl.float32), axis=0)
            swB = tl.sum(tl.zeros((1,), dtype=tl.float32), axis=0)
            sA1 = tl.sum(tl.zeros((1,), dtype=tl.float32), axis=0)
            sA2 = tl.sum(tl.zeros((1,), dtype=tl.float32), axis=0)
            sB1 = tl.sum(tl.zeros((1,), dtype=tl.float32), axis=0)
            sB2 = tl.sum(tl.zeros((1,), dtype=tl.float32), axis=0)

            n0 = start
            while n0 < end:
                offs = n0 + tl.arange(0, BLOCK_N)
                mask = offs < end
                u1 = tl.load(u1_ptr + offs, mask=mask, other=0.0)
                u2 = tl.load(u2_ptr + offs, mask=mask, other=0.0)
                w = tl.load(w_ptr + offs, mask=mask, other=0.0)

                dA2 = (u1 - cA1) * (u1 - cA1) + (u2 - cA2) * (u2 - cA2)
                dB2 = (u1 - cB1) * (u1 - cB1) + (u2 - cB2) * (u2 - cB2)
                wB = tl.where(dB2 < dA2, w, 0.0)
                wA = w - wB

                swA = swA + tl.sum(wA, axis=0)
                swB = swB + tl.sum(wB, axis=0)
                sA1 = sA1 + tl.sum(wA * u1, axis=0)
                sA2 = sA2 + tl.sum(wA * u2, axis=0)
                sB1 = sB1 + tl.sum(wB * u1, axis=0)
                sB2 = sB2 + tl.sum(wB * u2, axis=0)
                n0 = n0 + BLOCK_N

            swA = tl.maximum(swA, 1e-6)
            swB = tl.maximum(swB, 1e-6)
            cA1 = sA1 / swA
            cA2 = sA2 / swA
            cB1 = sB1 / swB
            cB2 = sB2 / swB

        tl.store(del1_ptr + c, cB1 - cA1)
        tl.store(del2_ptr + c, cB2 - cA2)


@triton.jit
def _k_var_and_bounds(
    proj_ptr, w_ptr, starts_ptr, ends_ptr, dot_mu_v_ptr, wsum_ptr, lo_ptr, span_ptr,
    BLOCK_N: tl.constexpr,
):
    c = tl.program_id(0)
    start = tl.load(starts_ptr + c)
    end = tl.load(ends_ptr + c)
    mu_v = tl.load(dot_mu_v_ptr + c)

    if end <= start:
        tl.store(lo_ptr + c, mu_v - 1e-4)
        tl.store(span_ptr + c, 2e-4)
    else:
        w_sum = tl.load(wsum_ptr + c)
        var_num = tl.sum(tl.zeros((1,), dtype=tl.float32), axis=0)
        n0 = start
        while n0 < end:
            offs = n0 + tl.arange(0, BLOCK_N)
            mask = offs < end
            p = tl.load(proj_ptr + offs, mask=mask, other=0.0)
            w = tl.load(w_ptr + offs, mask=mask, other=0.0)
            diff = p - mu_v
            var_num = var_num + tl.sum(w * diff * diff, axis=0)
            n0 = n0 + BLOCK_N

        var = var_num / tl.maximum(w_sum, 1e-6)
        half_span = tl.maximum(4.0 * tl.sqrt(tl.maximum(var, 1e-12)), 1e-4)
        tl.store(lo_ptr + c, mu_v - half_span)
        tl.store(span_ptr + c, 2.0 * half_span)


@triton.jit
def _k_split_hist(
    proj_ptr, w_ptr, assign_ptr, isB_ptr, lo_ptr, span_ptr, rawA_ptr, rawB_ptr,
    N, NUM_BINS: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    n_offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    p = tl.load(proj_ptr + n_offs, mask=n_mask, other=0.0)
    w = tl.load(w_ptr + n_offs, mask=n_mask, other=0.0)
    c = tl.load(assign_ptr + n_offs, mask=n_mask, other=0)
    is_b = tl.load(isB_ptr + n_offs, mask=n_mask, other=0.0)
    lo = tl.load(lo_ptr + c, mask=n_mask, other=0.0)
    sp = tl.load(span_ptr + c, mask=n_mask, other=1.0)

    max_bin = float(NUM_BINS - 1)
    pos = tl.minimum(tl.maximum((p - lo) / sp * max_bin, 0.0), max_bin - 1e-4)
    b0 = pos.to(tl.int32)
    b1 = tl.minimum(b0 + 1, NUM_BINS - 1)
    frac = pos - b0.to(tl.float32)

    w0 = w * (1.0 - frac)
    w1 = w * frac
    is_a = 1.0 - is_b
    base = c * NUM_BINS

    tl.atomic_add(rawA_ptr + base + b0, w0 * is_a, mask=n_mask)
    tl.atomic_add(rawA_ptr + base + b1, w1 * is_a, mask=n_mask)
    tl.atomic_add(rawB_ptr + base + b0, w0 * is_b, mask=n_mask)
    tl.atomic_add(rawB_ptr + base + b1, w1 * is_b, mask=n_mask)


@triton.jit
def _k_split_weights_per_cluster(
    proj_ptr, w_ptr, starts_ptr, ends_ptr, thresh_ptr, mu_v_ptr, tot_ptr, wsum_ptr,
    wl_pt_ptr, wr_pt_ptr, cnt_left_ptr, wl_ptr, wr_ptr,
    BLOCK_N: tl.constexpr,
):
    c = tl.program_id(0)
    start = tl.load(starts_ptr + c)
    end = tl.load(ends_ptr + c)

    if end <= start:
        tl.store(cnt_left_ptr + c, 0.0)
        tl.store(wl_ptr + c, 0.0)
        tl.store(wr_ptr + c, 0.0)
    else:
        th = tl.load(thresh_ptr + c)
        mu_v = tl.load(mu_v_ptr + c)
        tot = tl.load(tot_ptr + c)
        wsum = tl.load(wsum_ptr + c)

        cnt_l = tl.sum(tl.zeros((1,), dtype=tl.float32), axis=0)
        wl_s = tl.sum(tl.zeros((1,), dtype=tl.float32), axis=0)
        n0 = start
        while n0 < end:
            offs = n0 + tl.arange(0, BLOCK_N)
            mask = offs < end
            p = tl.load(proj_ptr + offs, mask=mask, other=0.0)
            w = tl.load(w_ptr + offs, mask=mask, other=0.0)
            is_l = mask & (p < th)
            cnt_l = cnt_l + tl.sum(is_l.to(tl.float32), axis=0)
            wl_s = wl_s + tl.sum(tl.where(is_l, w, 0.0), axis=0)
            n0 = n0 + BLOCK_N

        bad = (cnt_l < 1.0) | ((tot - cnt_l) < 1.0)
        th_use = tl.where(bad, mu_v, th)
        if bad:
            cnt_l = tl.sum(tl.zeros((1,), dtype=tl.float32), axis=0)
            wl_s = tl.sum(tl.zeros((1,), dtype=tl.float32), axis=0)
            n0 = start
            while n0 < end:
                offs = n0 + tl.arange(0, BLOCK_N)
                mask = offs < end
                p = tl.load(proj_ptr + offs, mask=mask, other=0.0)
                w = tl.load(w_ptr + offs, mask=mask, other=0.0)
                is_l = mask & (p < th_use)
                cnt_l = cnt_l + tl.sum(is_l.to(tl.float32), axis=0)
                wl_s = wl_s + tl.sum(tl.where(is_l, w, 0.0), axis=0)
                n0 = n0 + BLOCK_N

        n0 = start
        while n0 < end:
            offs = n0 + tl.arange(0, BLOCK_N)
            mask = offs < end
            p = tl.load(proj_ptr + offs, mask=mask, other=0.0)
            w = tl.load(w_ptr + offs, mask=mask, other=0.0)
            wl_i = tl.where(p < th_use, w, 0.0)
            tl.store(wl_pt_ptr + offs, wl_i, mask=mask)
            tl.store(wr_pt_ptr + offs, w - wl_i, mask=mask)
            n0 = n0 + BLOCK_N

        tl.store(cnt_left_ptr + c, cnt_l)
        tl.store(wl_ptr + c, wl_s)
        tl.store(wr_ptr + c, tl.maximum(wsum - wl_s, 0.0))


def _triton_accum1(X_ord, assign_i32, a1, m):
    N, D = X_ord.shape
    out1 = torch.zeros((m, D), device=X_ord.device, dtype=torch.float32)
    grid = (triton.cdiv(N, 64), triton.cdiv(D, 128))
    _k_segmented_accum[grid](
        X_ord, assign_i32, a1, a1, out1, out1,
        N, D, X_ord.stride(0), X_ord.stride(1), out1.stride(0), out1.stride(1),
        HAS_TWO=False, BLOCK_N=64, BLOCK_D=128,
    )
    return out1


def _triton_accum2(X_ord, assign_i32, a1, a2, m):
    N, D = X_ord.shape
    out1 = torch.zeros((m, D), device=X_ord.device, dtype=torch.float32)
    out2 = torch.zeros((m, D), device=X_ord.device, dtype=torch.float32)
    grid = (triton.cdiv(N, 64), triton.cdiv(D, 128))
    _k_segmented_accum[grid](
        X_ord, assign_i32, a1, a2, out1, out2,
        N, D, X_ord.stride(0), X_ord.stride(1), out1.stride(0), out1.stride(1),
        HAS_TWO=True, BLOCK_N=64, BLOCK_D=128,
    )
    return out1, out2


# =====================================================================
# 2. Static-Shape Core Function
# =====================================================================

def _engine_v11_core(
    X_sub: torch.Tensor,
    k: int,
    use_sse: bool = True,
    valley_boost: float = 2.5,
    robust_cap_q: float | None = 0.95,
    bw_scale: float = 0.65,
    half_x: bool = False,
) -> torch.Tensor:
    sub_n, d = X_sub.shape
    dev = X_sub.device

    x_sub_sq = torch.linalg.vector_norm(X_sub.float(), dim=1).square()

    mean_0 = X_sub.float().mean(dim=0, keepdim=True)
    C_buf = torch.cat([mean_0, torch.zeros((k - 1, d), device=dev, dtype=torch.float32)], dim=0)
    prev_V = torch.zeros((k, d), device=dev, dtype=torch.float32)
    active = (torch.arange(k, device=dev) == 0)

    num_bins = 64
    idx = torch.arange(num_bins, device=dev, dtype=torch.float32)
    dist2 = (idx.unsqueeze(0) - idx.unsqueeze(1)) ** 2
    sigma_bins = float(num_bins - 1) / 8.0

    b_max = min(16, max(1, k // 2))
    schedule = []
    m_hi = 1
    while m_hi < k:
        b = min(m_hi, max(1, (2 * m_hi + 2) // 3), b_max, k - m_hi)
        schedule.append((m_hi, b, True))
        m_hi += b
    b_tail = min(12, max(1, k // 4))
    schedule.append((k, b_tail, True))
    schedule.append((k, b_tail, True))
    schedule.append((k, b_tail, False))

    buf1 = torch.empty(sub_n, device=dev, dtype=torch.float32)
    buf2 = torch.empty(sub_n, device=dev, dtype=torch.float32)
    proj = torch.empty(sub_n, device=dev, dtype=torch.float32)
    wl_pt = torch.empty(sub_n, device=dev, dtype=torch.float32)
    wr_pt = torch.empty(sub_n, device=dev, dtype=torch.float32)
    ones_n = torch.ones(sub_n, device=dev, dtype=torch.float32)

    dot_block_n = 32 if sub_n <= 4096 else 64
    dot_grid = (triton.cdiv(sub_n, dot_block_n),)
    hist_grid = (triton.cdiv(sub_n, 256),)

    for m, b_step, use_gate in schedule:
        m_next = min(k, m + b_step)
        C_curr = C_buf[:m]
        pv_curr = prev_V[:m]
        act_m = active[:m]

        c_half = 0.5 * torch.linalg.vector_norm(C_curr, dim=1).square()
        if half_x:
            scores = (X_sub @ C_curr.half().T).float() - c_half
        else:
            scores = torch.addmm(-c_half, X_sub, C_curr.T)
        scores = scores.masked_fill(~act_m.unsqueeze(0), -1e9)
        max_score, assign = torch.max(scores, dim=1)
        raw_sse = torch.clamp(x_sub_sq - 2.0 * max_score, min=0.0)

        if robust_cap_q is not None:
            k_top = max(1, int(round((1.0 - robust_cap_q) * sub_n)))
            cap = torch.topk(raw_sse, k_top, largest=True).values[-1:]
            w_pt = torch.clamp(cap / torch.clamp(raw_sse, min=1e-6), max=1.0)
        else:
            w_pt = ones_n

        order = torch.argsort(assign)
        X_ord = X_sub[order].contiguous()
        assign_ord = assign[order]
        assign_i32 = assign_ord.to(torch.int32).contiguous()
        w_ord = w_pt[order].contiguous()
        is_B_ord = ((order & 1) == 1).float().contiguous()

        counts = torch.zeros(m, device=dev, dtype=torch.int64).scatter_add_(
            0, assign_ord, torch.ones(sub_n, device=dev, dtype=torch.int64)
        )
        ends = torch.cumsum(counts, dim=0)
        starts_i32 = (ends - counts).to(torch.int32).contiguous()
        ends_i32 = ends.to(torch.int32).contiguous()
        total_cnt = counts.float()

        z_m = torch.zeros(m, device=dev, dtype=torch.float32)
        w_sum = z_m.scatter_add(0, assign_ord, w_ord).clamp(min=1e-6)
        M = _triton_accum1(X_ord, assign_i32, w_ord, m) / w_sum.unsqueeze(1)

        # 1. Dual-Axis Initialization
        V1 = torch.randn(m, d, device=dev, dtype=torch.float32)
        V1 = V1 - (V1 * pv_curr).sum(dim=1, keepdim=True) * pv_curr
        V1 = (V1 / (V1.norm(dim=1, keepdim=True) + 1e-9)).contiguous()

        V2 = torch.randn(m, d, device=dev, dtype=torch.float32)
        V2 = V2 - (V2 * V1).sum(dim=1, keepdim=True) * V1
        V2 = (V2 / (V2.norm(dim=1, keepdim=True) + 1e-9)).contiguous()

        # 2. Deflated Dual Power Iteration
        for _ in range(3):
            mv1 = (M * V1).sum(dim=1).contiguous()
            mv2 = (M * V2).sum(dim=1).contiguous()
            _k_dot2_weighted[dot_grid](
                X_ord, assign_i32, V1, V2, mv1, mv2, w_ord, buf1, buf2,
                sub_n, d, X_ord.stride(0), X_ord.stride(1), V1.stride(0), V1.stride(1),
                APPLY_W=True, BLOCK_N=dot_block_n, BLOCK_D=128,
            )
            raw_V1, raw_V2 = _triton_accum2(X_ord, assign_i32, buf1, buf2, m)
            s1 = z_m.scatter_add(0, assign_ord, buf1)
            s2 = z_m.scatter_add(0, assign_ord, buf2)

            V1 = raw_V1 - s1.unsqueeze(1) * M
            V1 = (V1 / (V1.norm(dim=1, keepdim=True) + 1e-9)).contiguous()

            V2 = raw_V2 - s2.unsqueeze(1) * M
            V2 = V2 - (V2 * V1).sum(dim=1, keepdim=True) * V1
            V2 = (V2 / (V2.norm(dim=1, keepdim=True) + 1e-9)).contiguous()

        # 3. 2D Mode Tracking
        mv1 = (M * V1).sum(dim=1).contiguous()
        mv2 = (M * V2).sum(dim=1).contiguous()
        _k_dot2_weighted[dot_grid](
            X_ord, assign_i32, V1, V2, mv1, mv2, w_ord, buf1, buf2,
            sub_n, d, X_ord.stride(0), X_ord.stride(1), V1.stride(0), V1.stride(1),
            APPLY_W=False, BLOCK_N=dot_block_n, BLOCK_D=128,
        )

        del_1 = torch.empty(m, device=dev, dtype=torch.float32)
        del_2 = torch.empty(m, device=dev, dtype=torch.float32)
        _k_2d_mode_all[(m,)](
            buf1, buf2, w_ord, starts_i32, ends_i32, w_sum, del_1, del_2,
            BLOCK_N=256,
        )

        norm_del = torch.sqrt(del_1 * del_1 + del_2 * del_2).clamp(min=1e-9)
        V_star = (del_1 / norm_del).unsqueeze(1) * V1 + (del_2 / norm_del).unsqueeze(1) * V2
        V_star = (V_star / (V_star.norm(dim=1, keepdim=True) + 1e-9)).contiguous()

        # 4. Split-Half KDE Valley Consensus
        _k_dot1[dot_grid](
            X_ord, assign_i32, V_star, proj,
            sub_n, d, X_ord.stride(0), X_ord.stride(1), V_star.stride(0), V_star.stride(1),
            BLOCK_N=dot_block_n, BLOCK_D=128,
        )
        dot_mu_v = (M * V_star).sum(dim=1).contiguous()
        lo_v = torch.empty(m, device=dev, dtype=torch.float32)
        span = torch.empty(m, device=dev, dtype=torch.float32)
        _k_var_and_bounds[(m,)](
            proj, w_ord, starts_i32, ends_i32, dot_mu_v, w_sum, lo_v, span,
            BLOCK_N=256,
        )

        raw_A = torch.zeros((m, num_bins), device=dev, dtype=torch.float32)
        raw_B = torch.zeros((m, num_bins), device=dev, dtype=torch.float32)
        _k_split_hist[hist_grid](
            proj, w_ord, assign_i32, is_B_ord, lo_v, span, raw_A, raw_B,
            sub_n, NUM_BINS=num_bins, BLOCK_N=256,
        )
        raw_ab = torch.stack([raw_A, raw_B], dim=2)

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
        thresh = torch.where(has_valley, v_cand, dot_mu_v).contiguous()

        cnt_left = torch.empty(m, device=dev, dtype=torch.float32)
        wl = torch.empty(m, device=dev, dtype=torch.float32)
        wr = torch.empty(m, device=dev, dtype=torch.float32)
        _k_split_weights_per_cluster[(m,)](
            proj, w_ord, starts_i32, ends_i32, thresh, dot_mu_v, total_cnt, w_sum,
            wl_pt, wr_pt, cnt_left, wl, wr,
            BLOCK_N=256,
        )
        cnt_right = total_cnt - cnt_left

        sum_left, sum_right = _triton_accum2(X_ord, assign_i32, wl_pt, wr_pt, m)
        C_left = sum_left / wl.clamp(min=1e-6).unsqueeze(1)
        C_right = sum_right / wr.clamp(min=1e-6).unsqueeze(1)
        valid = act_m & (cnt_left >= 1.0) & (cnt_right >= 1.0)

        # 5. Zero-Sync Throttled Split
        if use_sse:
            delta_sse = (wl * wr / (wl + wr).clamp(min=1e-6)) * torch.linalg.vector_norm(C_left - C_right, dim=1).square()
            delta_sse = torch.where(valid, delta_sse, torch.zeros_like(delta_sse))
            ranking_score = delta_sse * torch.where(has_valley, valley_boost, 1.0)
        else:
            valley_rel_depth = torch.where(has_valley, v_max / min_dens_max.clamp(min=1e-6), torch.zeros_like(v_max))
            ranking_score = torch.where(valid, valley_rel_depth, torch.zeros_like(valley_rel_depth))
            fallback = torch.where(valid, total_cnt, torch.zeros_like(total_cnt))
            ranking_score = torch.where(ranking_score.max() > 0, ranking_score, fallback)

        b_pick = min(b_step, m)
        top_vals, top_ids = torch.topk(ranking_score, b_pick, largest=True)
        free_mask = ~active[:m_next]
        _, free_ids = torch.topk(free_mask.float(), b_pick, largest=True)

        thresh_gate = (0.05 * top_vals[:1]) if use_gate else torch.zeros(1, device=dev, dtype=torch.float32)
        ok = valid[top_ids] & free_mask[free_ids] & (top_vals > 0.0) & (top_vals >= thresh_gate)
        ok_2d = ok.unsqueeze(1)

        C_buf = C_buf.scatter(0, top_ids.unsqueeze(1).expand(-1, d), torch.where(ok_2d, C_left[top_ids], C_buf[top_ids]))
        C_buf = C_buf.scatter(0, free_ids.unsqueeze(1).expand(-1, d), torch.where(ok_2d, C_right[top_ids], C_buf[free_ids]))
        active = active.scatter(0, free_ids, active[free_ids] | ok)

        V_pick = V_star[top_ids]
        prev_V = prev_V.scatter(0, top_ids.unsqueeze(1).expand(-1, d), torch.where(ok_2d, V_pick, prev_V[top_ids]))
        prev_V = prev_V.scatter(0, free_ids.unsqueeze(1).expand(-1, d), torch.where(ok_2d, V_pick, prev_V[free_ids]))

    return C_buf


# =====================================================================
# 3. CUDA Graph Cache & Public API
# =====================================================================

@dataclass
class _GraphEntry:
    graph: torch.cuda.CUDAGraph
    static_in: torch.Tensor
    static_out: torch.Tensor


_GRAPH_CACHE: dict[tuple, _GraphEntry] = {}


def _get_or_capture_graph(
    k: int,
    d: int,
    sub_n: int,
    use_sse: bool,
    valley_boost: float,
    robust_cap_q: float | None,
    bw_scale: float,
    half_x: bool,
    dev: torch.device,
) -> _GraphEntry:
    dev_idx = dev.index if dev.index is not None else torch.cuda.current_device()
    key = (k, d, sub_n, use_sse, valley_boost, robust_cap_q, bw_scale, half_x, dev_idx)
    entry = _GRAPH_CACHE.get(key)
    if entry is not None:
        return entry

    in_dtype = torch.float16 if half_x else torch.float32
    static_in = torch.randn(sub_n, d, device=dev, dtype=in_dtype)

    s = torch.cuda.Stream(device=dev)
    s.wait_stream(torch.cuda.current_stream(device=dev))
    with torch.cuda.stream(s):
        # 3 warmup passes to initialize memory pool & compile Triton kernels
        for _ in range(3):
            _ = _engine_v11_core(
                static_in, k=k, use_sse=use_sse, valley_boost=valley_boost,
                robust_cap_q=robust_cap_q, bw_scale=bw_scale, half_x=half_x,
            )
        s.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            static_out = _engine_v11_core(
                static_in, k=k, use_sse=use_sse, valley_boost=valley_boost,
                robust_cap_q=robust_cap_q, bw_scale=bw_scale, half_x=half_x,
            )
    torch.cuda.current_stream(device=dev).wait_stream(s)

    entry = _GraphEntry(graph=g, static_in=static_in, static_out=static_out)
    _GRAPH_CACHE[key] = entry
    return entry


def warmup_graphs(
    specs: list[tuple[int, int]],
    sub_n: int = 16384,
    half_x: bool = False,
    device: str = "cuda",
):
    dev = torch.device(device)
    for (k, d) in specs:
        _ = _get_or_capture_graph(
            k=k, d=d, sub_n=sub_n, use_sse=True, valley_boost=2.5,
            robust_cap_q=0.95, bw_scale=0.65, half_x=half_x, dev=dev,
        )
    torch.cuda.synchronize(dev)


@torch.no_grad
def run_engine_v11_triton(
    X: torch.Tensor,
    k: int,
    sub_n: int = 16384,
    use_sse: bool = True,
    valley_boost: float = 2.5,
    robust_cap_q: float | None = 0.95,
    bw_scale: float = 0.65,
    half_x: bool = False,
    use_graph: bool = True,
) -> torch.Tensor:
    n, d = X.shape
    dev = X.device
    if n > sub_n:
        X_sub_f = X[torch.randint(0, n, (sub_n,), device=dev)]
    else:
        sub_n = n
        X_sub_f = X[torch.randperm(n, device=dev)]
    X_sub_f = torch.nan_to_num(X_sub_f.float(), nan=0.0, posinf=1e4, neginf=-1e4)

    if not use_graph:
        X_sub = X_sub_f.half() if half_x else X_sub_f
        return _engine_v11_core(
            X_sub, k=k, use_sse=use_sse, valley_boost=valley_boost,
            robust_cap_q=robust_cap_q, bw_scale=bw_scale, half_x=half_x,
        )

    entry = _get_or_capture_graph(
        k=k, d=d, sub_n=sub_n, use_sse=use_sse, valley_boost=valley_boost,
        robust_cap_q=robust_cap_q, bw_scale=bw_scale, half_x=half_x, dev=dev,
    )
    if half_x:
        entry.static_in.copy_(X_sub_f.half())
    else:
        entry.static_in.copy_(X_sub_f)

    entry.graph.replay()
    return entry.static_out.clone()


# =====================================================================
# 4. Standalone Test & Comparison
# =====================================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda"

    print("Pre-capturing CUDA graphs...")
    warmup_graphs([(16, 1024), (64, 1024)], sub_n=16384, half_x=False, device=dev)
    warmup_graphs([(16, 1024), (64, 1024)], sub_n=16384, half_x=True, device=dev)
    warmup_graphs([(8, 256)], sub_n=1024, half_x=False, device=dev)
    print("Graphs ready.\n")

    bench_configs = [
        # (N, D, k, sub_n, half_x, tag)
        (640000, 1024, 64, 16384, False, "kmeans_clean (k=64 fp32)"),
        (640000, 1024, 64, 16384, True,  "kmeans_clean (k=64 hx=True)"),
        (640000, 1024, 16, 16384, False, "kmeans_clean (k=16 fp32)"),
        (4096,   256,  8,  1024,  False, "GPT batch (sub_n=1024)"),
    ]

    for N, D, k, sub_n, half_x, tag in bench_configs:
        X = torch.randn(N, D, device=dev) * 0.5
        centers = torch.randn(k, D, device=dev) * 2.5
        X = X + centers[torch.randint(0, k, (N,), device=dev)]

        iters = 10 if N > 10000 else 50

        # Eager (no graph)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = run_engine_v11_triton(X, k=k, sub_n=sub_n, half_x=half_x, use_graph=False)
        torch.cuda.synchronize()
        ms_eager = (time.perf_counter() - t0) * 1000.0 / iters

        # Replay (CUDA Graph)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            C = run_engine_v11_triton(X, k=k, sub_n=sub_n, half_x=half_x, use_graph=True)
        torch.cuda.synchronize()
        ms_graph = (time.perf_counter() - t0) * 1000.0 / iters

        c_half = 0.5 * (C * C).sum(dim=1)
        assign = torch.argmax(torch.addmm(-c_half, X, C.T), dim=1)
        mse = (X - C[assign]).pow(2).sum(dim=1).mean().item()

        speedup = ms_eager / ms_graph
        print(f"{tag:<30} | Eager: {ms_eager:6.2f} ms | Graph: {ms_graph:6.2f} ms | Speedup: {speedup:4.1f}x | MSE: {mse:.4f}")