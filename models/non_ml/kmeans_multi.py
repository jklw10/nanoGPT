import math
import time
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch._inductor.config.disable_cpp_codegen = True

from models.non_ml.kmeans_clean import run_lloyd_kmeans,run_original_emeans, run_engine_v10_aware, run_engine_v11_aware, run_hybrid_v11_aware_emeans
# =====================================================================
# 1. Core Task Generators
# =====================================================================

def make_task_1b_blobs(N, D, K, seed):
    torch.manual_seed(seed)
    total_n = N + 20
    X = torch.empty((total_n, D), device=device, dtype=torch.float32)
    pts_per_k = N // K
    for i in range(K):
        center = torch.randn(D, device=device) * 4.0
        torch.randn(pts_per_k, D, device=device, out=X[i * pts_per_k : (i + 1) * pts_per_k])
        X[i * pts_per_k : (i + 1) * pts_per_k].add_(center)
    torch.randn(20, D, device=device, out=X[N:]).mul_(35.0)
    core_mask = torch.zeros(total_n, dtype=torch.bool, device=device)
    core_mask[:N] = True
    return X, core_mask

def make_task_2_l2(N, D, K, seed):
    torch.manual_seed(seed)
    latent_rank = 64
    num_topics = 256
    Q, _ = torch.linalg.qr(torch.randn(D, latent_rank, device=device))
    U = Q.T
    topic_centers = torch.randn(num_topics, latent_rank, device=device)
    topic_centers /= topic_centers.norm(dim=1, keepdim=True)
    probs = (torch.arange(num_topics, device=device, dtype=torch.float32) + 2.0) ** -1.1
    probs /= probs.sum()
    topic_ids = torch.multinomial(probs, N, replacement=True)
    Z = topic_centers[topic_ids] + torch.randn(N, latent_rank, device=device) * 0.42
    X = Z @ U + torch.randn(N, D, device=device) * 0.015
    X /= torch.clamp(X.norm(dim=1, keepdim=True), min=1e-8)
    return X, None

def make_task_3_anisotropic(N, D, K, seed):
    torch.manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(D, D, device=device))
    eig_scales = 10.0 * (0.985 ** torch.arange(D, device=device, dtype=torch.float32)) + 0.15
    X = torch.empty((N, D), device=device, dtype=torch.float32)
    pts_per_k = N // K
    for i in range(K):
        center = (torch.randn(D, device=device) * eig_scales) @ Q
        local_scales = eig_scales * 0.25
        noise = (torch.randn(pts_per_k, D, device=device) * local_scales) @ Q
        X[i * pts_per_k : (i + 1) * pts_per_k] = noise + center
    return X[torch.randperm(N, device=device)], None

def make_task_4_heavy_tail(N, D, K, seed):
    torch.manual_seed(seed)
    n_core = int(0.96 * N)
    n_noise = N - n_core
    X = torch.empty((N, D), device=device, dtype=torch.float32)
    pts_per_k = n_core // K
    for i in range(K):
        center = torch.randn(D, device=device) * 3.0
        torch.randn(pts_per_k, D, device=device, out=X[i * pts_per_k : (i + 1) * pts_per_k])
        X[i * pts_per_k : (i + 1) * pts_per_k].add_(center)
    torch.randn(n_noise, D, device=device, out=X[n_core:]).mul_(10.0)
    core_mask = torch.zeros(N, dtype=torch.bool, device=device)
    core_mask[:n_core] = True
    perm = torch.randperm(N, device=device)
    return X[perm], core_mask[perm]

def make_task_5_imbalance(N, D, K, seed):
    torch.manual_seed(seed)
    n_giants = int(0.75 * N)
    n_dwarfs = N - n_giants
    X = torch.empty((N, D), device=device, dtype=torch.float32)
    g_pts = n_giants // 2
    for g in range(2):
        center = torch.randn(D, device=device) * 4.0
        start = g * g_pts
        end = (g + 1) * g_pts if g == 0 else n_giants
        torch.randn(end - start, D, device=device, out=X[start:end]).mul_(3.2).add_(center)
    num_dwarfs = K - 2
    d_pts = n_dwarfs // num_dwarfs
    for i in range(num_dwarfs):
        center = torch.randn(D, device=device) * 4.0
        start = n_giants + i * d_pts
        end = n_giants + (i + 1) * d_pts if i < num_dwarfs - 1 else N
        torch.randn(end - start, D, device=device, out=X[start:end]).mul_(0.6).add_(center)
    return X[torch.randperm(N, device=device)], None


# =====================================================================
# 2. Evaluation Helper
# =====================================================================

@torch.no_grad()
def evaluate_distortion(X, centroids, core_mask=None, chunk_size=32768):
    total_sq = 0.0
    core_sq = 0.0
    n = X.shape[0]
    for i in range(0, n, chunk_size):
        chunk = X[i : i + chunk_size]
        min_sq = torch.cdist(chunk, centroids).min(dim=1).values.square()
        total_sq += min_sq.sum().item()
        if core_mask is not None:
            core_sq += min_sq[core_mask[i : i + chunk_size]].sum().item()
    total_mse = total_sq / n
    core_mse = (core_sq / core_mask.sum().item()) if core_mask is not None else None
    return total_mse, core_mse


# =====================================================================
# 3. Master 4-Seed Benchmark Runner
# =====================================================================

def run_4seed_benchmark():
    SEEDS = [42, 1337, 2026, 7777]
    N, D, K = 640000, 1024, 64

    tasks = [
        {"name": "Task 1b: Isolated Blobs + Outliers", "fn": make_task_1b_blobs},
        {"name": "Task 2: L2-Normalized Manifold",      "fn": make_task_2_l2},
        {"name": "Task 3: Rotated Anisotropic",         "fn": make_task_3_anisotropic},
        {"name": "Task 4: 96% Core + 4% Clutter",       "fn": make_task_4_heavy_tail},
        {"name": "Task 5: Extreme Imbalance (2G + 62D)","fn": make_task_5_imbalance},
    ]

    # Import engines from your script or section definitions:
    algos = [
        ("Classic Lloyd (15 iters)",        lambda x, k: run_lloyd_kmeans(x, k, iters=15)),
        ("Original e-means (1.0 ep)",       lambda x, k: run_original_emeans(x, k, epochs=1.0)),
        ("Engine v10-aware",                lambda x, k: run_engine_v10_aware(x, k)),
        ("Engine v11-aware",                lambda x, k: run_engine_v11_aware(x, k)),
        ("HYBRID v11-aware -> e-means",     lambda x, k: run_hybrid_v11_aware_emeans(x, k, epochs=0.5)),
    ]

    print(f"=== Running 4-Seed Benchmark (Seeds: {SEEDS}) on {device.type.upper()} ===")
    print(f"N={N:,} | D={D} | K={K}\n")

    # Warmup GPU
    _ = torch.randn(4096, D, device=device) @ torch.randn(D, 64, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()

    for task_cfg in tasks:
        print(f"\n" + "=" * 92)
        print(f" {task_cfg['name']} (Averaged across 4 Seeds)")
        print("=" * 92)
        print(f"{'Algorithm':<32} | {'Time (ms)':<16} | {'Total MSE':<20} | {'Core MSE':<16}")
        print("-" * 92)

        for algo_name, algo_fn in algos:
            times = []
            total_mses = []
            core_mses = []

            for s in SEEDS:
                X, core_mask = task_cfg["fn"](N, D, K, seed=s)
                
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                
                centroids = algo_fn(X, K)
                
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = (time.perf_counter() - t0) * 1000.0

                tot_mse, c_mse = evaluate_distortion(X, centroids, core_mask)
                
                times.append(elapsed)
                total_mses.append(tot_mse)
                if c_mse is not None:
                    core_mses.append(c_mse)

                del X, centroids, core_mask

            t_mean, t_std = np.mean(times), np.std(times)
            m_mean, m_std = np.mean(total_mses), np.std(total_mses)
            
            t_str = f"{t_mean:6.1f} ± {t_std:4.1f}"
            m_str = f"{m_mean:9.3f} ± {m_std:5.3f}"
            c_str = f"{np.mean(core_mses):8.2f} ± {np.std(core_mses):4.2f}" if core_mses else "N/A"

            print(f"{algo_name:<32} | {t_str:<16} | {m_str:<20} | {c_str:<16}")


if __name__ == "__main__":
    run_4seed_benchmark()