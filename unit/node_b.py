import torch

B, N, D, O = 64, 1024, 16, 32
torch.set_float32_matmul_precision('high')
# Baseline: (B, N, D) layout with transposes
x = torch.randn(B, N, D, device='cuda')
w = torch.randn(N, D, O, device='cuda')

# 1. Baseline: Your original bmm with transposes
start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
torch.cuda.synchronize()
start.record()
for _ in range(1000):
    _ = torch.bmm(x.transpose(0, 1), w).transpose(0, 1)
end.record(); torch.cuda.synchronize()
print(f"1. Baseline bmm (strided transposes): {start.elapsed_time(end):.2f} ms")

# 2. Contiguous Node-First Layout (N, B, D)
# If state/output are stored natively as (N, B, D), zero transposes are needed:
x_node_first = torch.randn(N, B, D, device='cuda')
start.record()
for _ in range(1000):
    _ = torch.bmm(x_node_first, w)
end.record(); torch.cuda.synchronize()
print(f"2. Native (N, B, D) bmm (zero transposes): {start.elapsed_time(end):.2f} ms")

# 3. Torch.compile with Triton fusion (fuses transpose + GEMM + epilogue)
def bmm_transposed(a, b):
    return torch.bmm(a.transpose(0, 1), b).transpose(0, 1)

compiled_bmm = torch.compile(bmm_transposed)
# Warmup
for _ in range(10): _ = compiled_bmm(x, w)
torch.cuda.synchronize()

start.record()
for _ in range(1000):
    _ = compiled_bmm(x, w)
end.record(); torch.cuda.synchronize()
print(f"3. Compiled bmm (Triton fused layout): {start.elapsed_time(end):.2f} ms")

# 4. Fused Q+K projection vs 2 separate bmm calls
K = 8
w_q = torch.randn(N, D, K, device='cuda')
w_k = torch.randn(N, D, K, device='cuda')
w_qk = torch.randn(N, D, 2 * K, device='cuda')

start.record()
for _ in range(1000):
    q = torch.bmm(x.transpose(0, 1), w_q).transpose(0, 1)
    k = torch.bmm(x.transpose(0, 1), w_k).transpose(0, 1)
end.record(); torch.cuda.synchronize()
print(f"4a. Unfused Q & K (two bmm calls):   {start.elapsed_time(end):.2f} ms")

start.record()
for _ in range(1000):
    qk = torch.bmm(x.transpose(0, 1), w_qk).transpose(0, 1)
    q, k = qk.chunk(2, dim=-1)
end.record(); torch.cuda.synchronize()
print(f"4b. Fused QK (one bmm call):         {start.elapsed_time(end):.2f} ms")