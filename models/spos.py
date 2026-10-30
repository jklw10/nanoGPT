import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

# ==============================================================================
# 1. DATA GENERATOR & MODEL (Detached Head2)
# ==============================================================================
def generate_sparse_data(batch_size=1024, n_features=5, prob=0.05):
    mask = (torch.rand(batch_size, n_features) < prob).float()
    values = torch.rand(batch_size, n_features)
    return mask * values

class TwoHeadToyModel(nn.Module):
    def __init__(self, n_features=5, n_hidden=2):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_hidden, n_features) * 0.1)
        self.head2 = nn.Sequential(nn.Linear(n_hidden, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        h = torch.matmul(x, self.W.T)
        head1 = torch.matmul(h, self.W)
        head2 = self.head2(h.detach())  # Detached to prevent NLL gradient corruption
        return head1, head2

# ==============================================================================
# 2. NO-CLAMP LOSS FUNCTIONS (Smooth C^infinity Scaling)
# ==============================================================================

# A. Standard EKNLL
class EntropicKreinNLLLoss(nn.Module):
    def __init__(self, S0_fallback: float = 10.0, eps: float = 1e-6):
        super().__init__()
        self.S0_fallback, self.eps = S0_fallback, eps

    def forward(self, y_pred, y_true, log_var_raw):
        error = y_pred - y_true
        residual_var = torch.var(error.detach()) + self.eps
        # Smooth Softplus S0 bound (No hard clamp)
        S0 = self.S0_fallback + F.softplus(0.5 * torch.log(residual_var / self.eps) - self.S0_fallback)
        log_var = log_var_raw / torch.sqrt(1.0 + (log_var_raw / S0) ** 2)
        var = torch.exp(log_var) + self.eps
        return (0.5 * (error ** 2) / var + 0.5 * log_var).mean()

# B. Gram NLL
class GramNLLLoss(nn.Module):
    def __init__(self, lambda_gram: float = 0.1, lambda_norm: float = 0.1, temperature: float = 10.0):
        super().__init__()
        self.lambda_gram, self.lambda_norm, self.temperature = lambda_gram, lambda_norm, temperature

    def forward(self, y_pred, y_true, log_var_raw, W):
        error = y_pred - y_true
        log_var = 10.0 * torch.tanh(log_var_raw / 10.0)
        var = torch.exp(log_var) + 1e-6
        recon = (torch.abs(error) / var.detach()).mean()
        nll = (0.5 * (error.detach() ** 2) / var + 0.5 * log_var).mean()
        
        W_norms = torch.sqrt(torch.sum(W ** 2, dim=0, keepdim=True) + 1e-6)
        W_normed = W / W_norms
        G = torch.matmul(W_normed.T, W_normed)
        N = W.shape[1]
        off_diag_mask = ~torch.eye(N, dtype=torch.bool, device=W.device)
        r_sq = (G[off_diag_mask] ** 2) * 0.999
        gram_penalty = (1.0 / self.temperature) * torch.logsumexp(self.temperature * r_sq, dim=0)
        norm_penalty = torch.mean((W_norms.squeeze(0) - 1.0) ** 2)
        return recon + nll + self.lambda_gram * gram_penalty + self.lambda_norm * norm_penalty

# C. Modular NLL
class ModularL4NLLLoss(nn.Module):
    def __init__(self, S0_fallback: float = 10.0, eps: float = 1e-6):
        super().__init__()
        self.S0_fallback, self.eps = S0_fallback, eps

    def forward(self, y_pred, y_true, log_var_raw):
        error = y_pred - y_true
        residual_var = torch.var(error.detach()) + self.eps
        S0 = self.S0_fallback + F.softplus(0.5 * torch.log(residual_var / self.eps) - self.S0_fallback)
        log_var = log_var_raw / torch.sqrt(1.0 + (log_var_raw / S0) ** 2)
        var = torch.exp(log_var) + self.eps
        l4_loss = (error ** 4).mean()
        nll_loss = (0.5 * (error.detach() ** 2) / var + 0.5 * log_var).mean()
        return l4_loss + nll_loss

# D. Original ekl4_nll (Mean-based Gram)
class EntropicKreinL4NLLLoss(nn.Module):
    def __init__(self, S0_fallback: float = 10.0, eps: float = 1e-6):
        super().__init__()
        self.S0_fallback, self.eps = S0_fallback, eps

    def forward(self, y_pred, y_true, log_var_raw, W):
        error = y_pred - y_true
        residual_var = torch.var(error.detach()) + self.eps
        S0 = self.S0_fallback + F.softplus(0.5 * torch.log(residual_var / self.eps) - self.S0_fallback)
        log_var = log_var_raw / torch.sqrt(1.0 + (log_var_raw / S0) ** 2)
        var = torch.exp(log_var) + self.eps

        l4_geometry_loss = torch.log1p(error ** 4).mean()
        nll_loss = (0.5 * (error.detach() ** 2) / var + 0.5 * log_var).mean()

        W_norms = torch.sqrt(torch.sum(W ** 2, dim=0, keepdim=True) + self.eps)
        W_normed = W / W_norms
        gram_matrix = torch.matmul(W_normed.T, W_normed)
        N = W.shape[1]
        off_diag_mask = ~torch.eye(N, dtype=torch.bool, device=W.device)
        r_sq = (gram_matrix[off_diag_mask] ** 2) * 0.999
        gram_penalty = -torch.mean(torch.log(1.0 - r_sq + self.eps))
        return l4_geometry_loss + nll_loss + gram_penalty

# E. NEW: ekgnll_softmax (L2 + Softmax Gram + Norm Anchor, No Clamps)
class EntropicKreinGramSoftmaxLoss(nn.Module):
    def __init__(self, S0_fallback: float = 10.0, tau: float = 15.0, eps: float = 1e-6):
        super().__init__()
        self.S0_fallback, self.tau, self.eps = S0_fallback, tau, eps

    def forward(self, y_pred, y_true, log_var_raw, W):
        error = y_pred - y_true
        residual_var = torch.var(error.detach()) + self.eps
        S0 = self.S0_fallback + F.softplus(0.5 * torch.log(residual_var / self.eps) - self.S0_fallback)
        log_var = log_var_raw / torch.sqrt(1.0 + (log_var_raw / S0) ** 2)
        var = torch.exp(log_var) + self.eps

        recon_loss = (error ** 2).mean()  # L2 Reconstruction
        nll_loss = (0.5 * (error.detach() ** 2) / var + 0.5 * log_var).mean()

        W_norms = torch.sqrt(torch.sum(W ** 2, dim=0, keepdim=True) + self.eps)
        W_normed = W / W_norms
        gram_matrix = torch.matmul(W_normed.T, W_normed)
        
        N = W.shape[1]
        off_diag_mask = ~torch.eye(N, dtype=torch.bool, device=W.device)
        r_sq = gram_matrix[off_diag_mask] ** 2
        
        # Softmax weighting + Smooth 0.95 scaling factor (No hard clamp)
        weights = F.softmax(self.tau * r_sq, dim=0)
        smooth_max_r_sq = torch.sum(weights * r_sq)
        gram_penalty = -torch.log(1.0 - 0.95 * smooth_max_r_sq + self.eps)

        norm_penalty = torch.mean((W_norms.squeeze(0) - 1.0) ** 2)
        return recon_loss + nll_loss + gram_penalty + norm_penalty

# F. NEW: ekl4_softmax_gram (L4 + Softmax Gram + Norm Anchor, No Clamps)
class EntropicKreinL4SoftmaxGramLoss(nn.Module):
    def __init__(self, S0_fallback: float = 10.0, tau: float = 15.0, eps: float = 1e-6):
        super().__init__()
        self.S0_fallback, self.tau, self.eps = S0_fallback, tau, eps

    def forward(self, y_pred, y_true, log_var_raw, W):
        error = y_pred - y_true
        residual_var = torch.var(error.detach()) + self.eps
        S0 = self.S0_fallback + F.softplus(0.5 * torch.log(residual_var / self.eps) - self.S0_fallback)
        log_var = log_var_raw / torch.sqrt(1.0 + (log_var_raw / S0) ** 2)
        var = torch.exp(log_var) + self.eps

        l4_geometry_loss = torch.log1p(error ** 4).mean()  # Soft-L4 Reconstruction
        nll_loss = (0.5 * (error.detach() ** 2) / var + 0.5 * log_var).mean()

        W_norms = torch.sqrt(torch.sum(W ** 2, dim=0, keepdim=True) + self.eps)
        W_normed = W / W_norms
        gram_matrix = torch.matmul(W_normed.T, W_normed)
        
        N = W.shape[1]
        off_diag_mask = ~torch.eye(N, dtype=torch.bool, device=W.device)
        r_sq = gram_matrix[off_diag_mask] ** 2
        
        # Softmax weighting + Smooth 0.95 scaling factor (No hard clamp)
        weights = F.softmax(self.tau * r_sq, dim=0)
        smooth_max_r_sq = torch.sum(weights * r_sq)
        gram_penalty = -torch.log(1.0 - 0.95 * smooth_max_r_sq + self.eps)

        norm_penalty = torch.mean((W_norms.squeeze(0) - 1.0) ** 2)
        return l4_geometry_loss + nll_loss + gram_penalty + norm_penalty

# ==============================================================================
# 3. BENCHMARK EXECUTION SUITE
# ==============================================================================
def run_experiment(loss_mode, num_steps=3000):
    model = TwoHeadToyModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    loss_instances = {
        "eknll": EntropicKreinNLLLoss(),
        "gram": GramNLLLoss(),
        "mknll": ModularL4NLLLoss(),
        "ekl4_nll": EntropicKreinL4NLLLoss(),
        "ekgnll_softmax": EntropicKreinGramSoftmaxLoss(tau=15.0),
        "ekl4_softmax_gram": EntropicKreinL4SoftmaxGramLoss(tau=15.0)
    }

    grad_norms = []

    for step in range(num_steps):
        x = generate_sparse_data()
        optimizer.zero_grad()
        head1, head2 = model(x)

        if loss_mode == "standard_mse":
            loss = torch.mean((head1 - x) ** 2)
        elif loss_mode == "l4_loss":
            loss = torch.mean((head1 - x) ** 4)
        else:
            loss_fn = loss_instances[loss_mode]
            if loss_mode in ["gram", "ekl4_nll", "ekgnll_softmax", "ekl4_softmax_gram"]:
                loss = loss_fn(head1, x, head2, model.W)
            else:
                loss = loss_fn(head1, x, head2)

        loss.backward()

        if model.W.grad is not None:
            g_norm = model.W.grad.norm().item()
            grad_norms.append(g_norm)

        optimizer.step()

    W_final = model.W.detach().cpu().numpy()
    norms = np.linalg.norm(W_final, axis=0)
    W_normed = W_final / (norms + 1e-8)
    gram_mat = np.matmul(W_normed.T, W_normed)
    max_off_diag = np.max(np.abs(gram_mat - np.eye(5)))
    alive_features = int(np.sum(norms > 0.05))

    return {
        "model": model,
        "W": W_final,
        "norms": norms,
        "max_off_diag": max_off_diag,
        "alive": alive_features,
        "max_grad": max(grad_norms) if grad_norms else 0.0,
        "avg_grad": sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
    }

# ==============================================================================
# 4. BAYER DITHERING RENDERER & VISUALIZATION
# ==============================================================================
def compute_dithered_rgb(x_hat, res):
    bayer8 = np.array([
        [ 0, 32,  8, 40,  2, 34, 10, 42],
        [48, 16, 56, 24, 50, 18, 58, 26],
        [12, 44,  4, 36, 14, 46,  6, 38],
        [60, 28, 52, 20, 62, 30, 54, 22],
        [ 3, 35, 11, 43,  1, 33,  9, 41],
        [51, 19, 59, 27, 49, 17, 57, 25],
        [15, 47,  7, 39, 13, 45,  5, 37],
        [63, 31, 55, 23, 61, 29, 53, 21]
    ], dtype=np.float32) / 64.0

    bayer_tile = np.tile(bayer8, (res // 8 + 1, res // 8 + 1))[:res, :res]
    feature_colors = np.array([
        [0.90, 0.10, 0.10], [0.10, 0.40, 0.95], [0.10, 0.80, 0.20],
        [0.70, 0.10, 0.85], [0.95, 0.65, 0.00]
    ], dtype=np.float32)

    x_np = x_hat.numpy().reshape(res, res, 5)
    densities = 1.0 - np.exp(-1.8 * x_np)
    rgb = np.ones((res, res, 3), dtype=np.float32)

    for i in range(5):
        shift_x, shift_y = (i * 3) % 8, (i * 5) % 8
        dither_threshold = np.roll(bayer_tile, (shift_x, shift_y), axis=(0, 1))
        mask = densities[:, :, i] > dither_threshold
        rgb[mask] = feature_colors[i]

    return rgb

# ==============================================================================
# 5. EXECUTION ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    modes = [
        #"standard_mse",
        #"l4_loss",
        #"eknll",
        #"gram",
        #"mknll",
        #"ekl4_nll",
        "ekgnll_softmax",     # L2 + Softmax Gram + Norm Anchor
        "ekl4_softmax_gram"   # L4 + Softmax Gram + Norm Anchor
    ]

    results = {}
    print("=" * 110)
    print(f"{'LOSS MODE':<20} | {'MAX OFF-DIAG':<12} | {'ALIVE (5)':<10} | {'MAX GRAD NORM':<14} | {'AVG GRAD NORM':<14} | {'STATUS':<10}")
    print("=" * 110)

    for mode in modes:
        res = run_experiment(mode, num_steps=3000)
        results[mode] = res

        status = "🔴 EXTREME" if res['max_grad'] > 50.0 else ("🟡 MODERATE" if res['max_grad'] > 10.0 else "🟢 STABLE")
        print(f"{mode:<20} | {res['max_off_diag']:<12.3f} | {res['alive']:<10d} | {res['max_grad']:<14.2f} | {res['avg_grad']:<14.4f} | {status:<10}")
        print(f"  └─ Feature Norms: {np.round(res['norms'], 3)}")

    print("=" * 110)

    # --- RENDER 2x4 SUBPLOT GRID ---
    grid_res = 320
    x_range = np.linspace(-2.0, 2.0, grid_res)
    y_range = np.linspace(-2.0, 2.0, grid_res)
    xx, yy = np.meshgrid(x_range, y_range)
    grid_h = torch.tensor(np.column_stack([xx.ravel(), yy.ravel()]), dtype=torch.float32)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.ravel()
    arrow_colors = ['#cc0000', '#0055cc', '#009900', '#6600cc', '#cc9900']

    for idx, mode in enumerate(modes):
        ax = axes[idx]
        W_tensor = torch.tensor(results[mode]["W"], dtype=torch.float32)
        x_hat = torch.relu(grid_h @ W_tensor)

        rgb_image = compute_dithered_rgb(x_hat, grid_res)
        ax.imshow(rgb_image, origin='lower', extent=[-2.0, 2.0, -2.0, 2.0])

        W, norms, off_diag = results[mode]["W"], results[mode]["norms"], results[mode]["max_off_diag"]
        for i in range(5):
            if norms[i] > 0.05:
                ax.quiver(0, 0, W[0, i], W[1, i], angles='xy', scale_units='xy', scale=1,
                          color=arrow_colors[i], width=0.015, label=f"F{i} ({norms[i]:.2f})")

        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.axvline(0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-2.0, 2.0)
        ax.set_aspect('equal')
        ax.set_title(f"{mode}\nMax Off-Diag: {off_diag:.3f}", fontsize=10, fontweight='bold')
        ax.legend(loc='upper right', fontsize=7, framealpha=0.85)

    plt.suptitle("Comparison: Standard MSE vs L4 vs EKNLL vs Softmax-Gram Formulations (5 Features in 2D)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()