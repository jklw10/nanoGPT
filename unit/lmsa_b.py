import time
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model import LinearMetricSimplexAttention

# Import your fixed LinearMetricSimplexAttention module here
# from module import LinearMetricSimplexAttention

@dataclass
class Config:
    n_embd: int = 256
    n_head: int = 8
    dropout: float = 0.0
    bias: bool = False

def run_precision_benchmark(use_fp64=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64 if use_fp64 else torch.float32
    config = Config()
    
    model = LinearMetricSimplexAttention(config).to(device=device, dtype=dtype)
    model.eval()

    batch_size = 4
    seq_len = 128
    x = torch.randn(batch_size, seq_len, config.n_embd, device=device, dtype=dtype)

    # 1. Parallel Forward
    with torch.no_grad():
        out_forward = model(x, causal=True)

    # 2. Recurrent Step
    with torch.no_grad():
        state = model.init_recurrent_state(batch_size, device=device, dtype=dtype)
        out_steps = []
        for t in range(seq_len):
            x_t = x[:, t:t+1, :]
            out_t, state = model.step(x_t, state)
            out_steps.append(out_t)
            
        out_step_concat = torch.cat(out_steps, dim=1)

    abs_diff = (out_forward - out_step_concat).abs()
    max_err = abs_diff.max().item()
    mean_err = abs_diff.mean().item()
    
    atol = 1e-10 if use_fp64 else 1e-3
    is_close = torch.allclose(out_forward, out_step_concat, atol=atol, rtol=1e-3)

    print("=" * 65)
    print(f"Precision Mode       : {'FP64 (Double)' if use_fp64 else 'FP32 (Single)'}")
    print(f"Device               : {device}")
    print(f"Max Absolute Error   : {max_err:.8e}")
    print(f"Mean Absolute Error  : {mean_err:.8e}")
    print(f"Forward == Step?     : {'✅ MATCH (Verified)' if is_close else '❌ MISMATCH'}")
    print("=" * 65)

if __name__ == "__main__":
    print("\n--- FP32 RUN ---")
    run_precision_benchmark(use_fp64=False)

    print("\n--- FP64 RUN ---")
    run_precision_benchmark(use_fp64=True)