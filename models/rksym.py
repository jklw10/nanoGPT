import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.parametrizations import weight_norm as wn
import torch._functorch.config

# --- Performance & stability ---
torch._functorch.config.donated_buffer = False
torch.set_float32_matmul_precision('high')

# --- 1. Model & Projection Head (unchanged) ---
class DynamicSymmetryNet(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 7x7
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(latent_dim, 10)

    def forward(self, x):
        latent = self.encoder(x)
        logits = self.classifier(latent)
        return logits, latent


class SymmetryProjectionHead(nn.Module):
    def __init__(self, embed_dim=64, proj_dim=32):
        super().__init__()
        self.proj = nn.Sequential(
            wn(nn.Linear(embed_dim, embed_dim * 2, bias=False), dim=None),
            nn.GELU(),
            wn(nn.Linear(embed_dim * 2, proj_dim, bias=False), dim=None)
        )
        self.stability_head = wn(nn.Linear(proj_dim, 1, bias=False), dim=None)

    def forward(self, x):
        subspace_vec = self.proj(x)
        stability = torch.tanh(self.stability_head(subspace_vec))
        return subspace_vec, stability


# --- 2. Krein‑Entropic ES Optimizer (from first code, adapted for any shape) ---
def compute_soft_kronecker_matrix(Cov, epsilon=1e-5, steps=2, power_iters=2, alpha=0.5):
    d = Cov.shape[0]
    device, dtype = Cov.device, Cov.dtype
    eye_d = torch.eye(d, device=device, dtype=dtype)
    Cov_reg = Cov + epsilon * eye_d

    v = torch.ones((d, 1), device=device, dtype=dtype)
    for _ in range(power_iters):
        u = torch.matmul(Cov_reg, v)
        u = u / (u.norm() + 1e-12)
        v = torch.matmul(Cov_reg.T, u)
        v = v / (v.norm() + 1e-12)

    s = torch.matmul(u.T, torch.matmul(Cov_reg, v)).squeeze().clamp(min=1e-8)
    B = Cov_reg / s
    Y = eye_d.clone()

    for _ in range(steps):
        Y = 0.5 * torch.matmul(Y, 3.0 * eye_d - torch.matmul(Y, torch.matmul(B, Y)))

    W_hard = Y / torch.sqrt(s)
    return alpha * W_hard + (1.0 - alpha) * eye_d


def generate_low_rank_spectral_noise(shape, rank=2, device=None):
    if len(shape) < 2:
        return torch.randn(shape, device=device)
    d_in, d_out = shape[-2], shape[-1]
    batch_shape = shape[:-2]
    A = torch.randn(*batch_shape, d_in, rank, device=device)
    B = torch.randn(*batch_shape, d_out, rank, device=device)
    if len(shape) == 2:
        noise = (A @ B.mT) / math.sqrt(rank)
    else:
        noise = torch.matmul(A, B.transpose(-1, -2)) / math.sqrt(rank)
    return noise


def update_krein_es_step(p, p_plus, p_minus, u_grad_precond, lr, gamma, scale_ref):
    eps_geom = 1e-6
    R_t = 6.0 * scale_ref + eps_geom

    u_plus = torch.clamp(-u_grad_precond, min=0.0)
    u_minus = torch.clamp(u_grad_precond, min=0.0)

    x_norm_plus = torch.clamp(p_plus / R_t, min=eps_geom, max=0.99)
    K_plus = x_norm_plus / (1.0 + x_norm_plus)
    K_prime_plus = 1.0 / (1.0 + x_norm_plus).square()
    force_plus = gamma * (K_plus * K_prime_plus) / (1.0 - K_plus.square() + eps_geom)

    x_norm_minus = torch.clamp(p_minus / R_t, min=eps_geom, max=0.99)
    K_minus = x_norm_minus / (1.0 + x_norm_minus)
    K_prime_minus = 1.0 / (1.0 + x_norm_minus).square()
    force_minus = gamma * (K_minus * K_prime_minus) / (1.0 - K_minus.square() + eps_geom)

    p_plus_raw = torch.clamp(p_plus + lr * (u_plus - force_plus), min=eps_geom)
    p_minus_raw = torch.clamp(p_minus + lr * (u_minus - force_minus), min=eps_geom)

    p_data_next = p_plus_raw - p_minus_raw
    p_plus_next = torch.clamp(p_data_next, min=0.0) + eps_geom
    p_minus_next = torch.clamp(-p_data_next, min=0.0) + eps_geom

    return p_plus_next, p_minus_next, p_data_next


class KreinEntropicESOptimizer:
    def __init__(self, target_tensor, pop_size=16, rank=2, sigma=0.05, lr=1e-2,
                 gamma=1e-4, beta_cov=0.1, damping=1e-5):
        self.target = target_tensor
        self.pop_size = pop_size
        self.half_pop = pop_size // 2
        self.rank = rank
        self.sigma = sigma
        self.lr = lr
        self.gamma = gamma
        self.beta_cov = beta_cov
        self.damping = damping
        self.step_idx = 0

        device = target_tensor.device
        self.p_plus = torch.clamp(target_tensor.data.clone(), min=0.0).add_(1e-4)
        self.p_minus = torch.clamp(-target_tensor.data.clone(), min=0.0).add_(1e-4)
        self.scale_ref = float(torch.nan_to_num(torch.std(target_tensor.data), nan=1e-4).item())
        self.L_ema = None
        self.R_ema = None
        self.W_L = None
        self.W_R = None

    @torch.no_grad()
    def step(self, fitness_fn):
        self.step_idx += 1
        device = self.target.device
        target_shape = self.target.shape

        # Generate low‑rank noise perturbations (half_pop, *target_shape)
        white_noise = torch.stack([
            generate_low_rank_spectral_noise(target_shape, rank=self.rank, device=device)
            for _ in range(self.half_pop)
        ], dim=0)

        # Antithetic population
        w_pos = self.target + self.sigma * white_noise
        w_neg = self.target - self.sigma * white_noise
        pop_weights = torch.cat([w_pos, w_neg], dim=0)  # (pop_size, *target_shape)

        # Fitness for each perturbed population
        surprises = fitness_fn(pop_weights)  # shape (pop_size,)
        surprises_pos = surprises[:self.half_pop]
        surprises_neg = surprises[self.half_pop:]

        mean_s = surprises.mean()
        std_s = surprises.std() + 1e-8
        fit_weights = ((surprises_pos - mean_s) / std_s) - ((surprises_neg - mean_s) / std_s)

        # ES gradient (same shape as target)
        raw_es_grad = torch.einsum('i,i...->...', fit_weights, white_noise) / (self.pop_size * self.sigma)
        raw_es_grad = raw_es_grad.view_as(self.target)

        # Soft‑Kronecker preconditioning (if enough dimensions)
        if raw_es_grad.ndim >= 2:
            m = raw_es_grad.shape[0]
            n = raw_es_grad.numel() // m
            g_2d = raw_es_grad.view(m, n)

            L_batch = (1.0 / n) * (g_2d @ g_2d.T)
            R_batch = (1.0 / m) * (g_2d.T @ g_2d)

            if self.L_ema is None:
                self.L_ema = torch.zeros_like(L_batch)
                self.R_ema = torch.zeros_like(R_batch)

            self.L_ema.mul_(self.beta_cov).add_(L_batch, alpha=1.0 - self.beta_cov)
            self.R_ema.mul_(self.beta_cov).add_(R_batch, alpha=1.0 - self.beta_cov)

            bias_corr = 1.0 - (self.beta_cov ** self.step_idx)
            L_hat = self.L_ema / bias_corr
            R_hat = self.R_ema / bias_corr

            if self.W_L is None or self.step_idx % 10 == 0:
                self.W_L = compute_soft_kronecker_matrix(L_hat, epsilon=self.damping)
                self.W_R = compute_soft_kronecker_matrix(R_hat, epsilon=self.damping)

            u_kron_2d = self.W_L @ g_2d @ self.W_R
            scale = g_2d.norm() / (u_kron_2d.norm() + 1e-8)
            precond_grad = (u_kron_2d * scale).view_as(self.target)
        else:
            precond_grad = raw_es_grad

        # Dual‑channel Krein step + gauge projection
        self.p_plus, self.p_minus, p_next = update_krein_es_step(
            self.target.data,
            self.p_plus,
            self.p_minus,
            precond_grad,
            self.lr,
            self.gamma,
            self.scale_ref
        )

        self.target.data.copy_(p_next)
        return mean_s.item()


# --- 3. Main training loop ---
if __name__ == '__main__':
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # Dataset
    BATCH_SIZE = 64
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    full_train = torchvision.datasets.MNIST("./", train=True, download=True, transform=transform)
    train_size = 50000
    val_size = len(full_train) - train_size
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2, pin_memory=True)

    # Models
    raw_model = DynamicSymmetryNet().to(device)
    raw_proj = SymmetryProjectionHead(embed_dim=64, proj_dim=32).to(device)
    model = torch.compile(raw_model)
    proj_head = torch.compile(raw_proj)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(proj_head.parameters()),
        lr=1e-3
    )
    criterion = nn.CrossEntropyLoss()

    # --- Kernel population (16 kernels) ---
    KERNEL_POP = 16
    MAX_SAFE_CAP = 6
    KERNEL_SIZE = 3
    PROMOTION_COOLDOWN = 10
    MUTATION_COOLDOWN = 10

    kernels = torch.randn(KERNEL_POP, 1, KERNEL_SIZE, KERNEL_SIZE, device=device)
    kernels = kernels / (kernels.abs().sum(dim=(-2, -1), keepdim=True) + 1e-6)
    # Index 0 = identity
    kernels[0] = torch.tensor([[[[0,0,0],[0,1,0],[0,0,0]]]], device=device)

    ages = torch.ones(KERNEL_POP, device=device)
    ages[0] = 100.0
    ema_wobbles = torch.zeros(KERNEL_POP, device=device)
    is_safe = torch.zeros(KERNEL_POP, dtype=torch.bool, device=device)
    is_safe[0] = True
    cooldowns = torch.zeros(KERNEL_POP, dtype=torch.int32, device=device)

    # --- Krein‑Entropic ES optimizer for the entire population ---
    es_optimizer = KreinEntropicESOptimizer(
        target_tensor=kernels,
        pop_size=8,          # number of perturbations per ES step
        rank=2,
        sigma=0.02,          # exploration noise
        lr=5e-3,
        gamma=1e-1,
        damping=1e-3,
    )

    EPOCHS = 5
    LAMBDA_WOBBLE = 0.5
    EMA_ALPHA = 0.15
    print(f"Combined Krein‑ES + Population Cooldowns on {device}\n")

    for epoch in range(EPOCHS):
        model.train()
        proj_head.train()
        total_loss = 0.0
        promoted = demoted = pruned = 0

        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            B = images.size(0)
            optimizer.zero_grad(set_to_none=True)

            # ----- 1. Raw forward pass -----
            logits_raw, latent_raw = model(images)
            task_loss = criterion(logits_raw, labels)
            p_raw, stab_raw = proj_head(latent_raw)
            chaos_factor = ((1.0 - stab_raw.detach()) / 2.0).mean() * 0.99

            # ----- 2. Compute wobbles for current kernels (for model loss & statuses) -----
            # Vectorized forward for all kernels
            norm_kernels = kernels / (kernels.abs().sum(dim=(-2,-1), keepdim=True) + 1e-6)
            filtered = F.conv2d(images, norm_kernels, padding=KERNEL_SIZE//2)
            filtered_reshaped = filtered.permute(1, 0, 2, 3).reshape(KERNEL_POP * B, 1, 28, 28)

            with torch.no_grad():
                _, latents_all = model(filtered_reshaped)
                p_all, _ = proj_head(latents_all)
            p_all = p_all.view(KERNEL_POP, B, -1)
            subspace_wobbles = (p_all - p_raw.detach().unsqueeze(0)).pow(2).mean(dim=(1,2))
            weighted_wobbles = subspace_wobbles / (1.0 + chaos_factor)

            # Model loss: task + safe wobble
            safe_mask = is_safe
            safe_wobble_loss = weighted_wobbles[safe_mask].mean() if safe_mask.any() else torch.tensor(0.0, device=device)
            model_loss = task_loss + LAMBDA_WOBBLE * safe_wobble_loss
            model_loss.backward()
            optimizer.step()
            total_loss += model_loss.item()

            # ----- 3. ES update of the entire kernel population -----
            def es_fitness_fn(pop_kernels):
                # pop_kernels shape: (es_pop_size, KERNEL_POP, 1, KERNEL_SIZE, KERNEL_SIZE)
                es_pop, kp = pop_kernels.shape[0], pop_kernels.shape[1]
                # Normalise and reshape for conv2d
                norm_pop = pop_kernels / (pop_kernels.abs().sum(dim=(-2,-1), keepdim=True) + 1e-6)
                # Combine batch dims: (es_pop * kp, 1, 28, 28)
                flat_kernels = norm_pop.view(es_pop * kp, 1, KERNEL_SIZE, KERNEL_SIZE)
                filtered_batch = F.conv2d(images, flat_kernels, padding=KERNEL_SIZE//2)
                # Shape: (es_pop * kp, 1, 28, 28)
                filtered_reshaped = filtered_batch.view(es_pop * kp * B, 1, 28, 28)

                with torch.no_grad():
                    _, latents_all = model(filtered_reshaped)
                    p_all, _ = proj_head(latents_all)
                p_all = p_all.view(es_pop, kp, B, -1)  # (es_pop, kp, B, proj_dim)
                # Compute wobble per kernel per perturbation
                p_raw_expanded = p_raw.detach().unsqueeze(0).unsqueeze(1)  # (1,1,B,proj_dim)
                wobbles = (p_all - p_raw_expanded).pow(2).mean(dim=(2,3))   # (es_pop, kp)
                # Weighted by chaos factor
                weighted = wobbles / (1.0 + chaos_factor)
                # Fitness = sum over kernels (or mean) – we want to minimise total wobble
                fitness = weighted.sum(dim=1)  # (es_pop,)
                return fitness

            es_loss = es_optimizer.step(es_fitness_fn)

            # ----- 4. After ES, recompute wobbles for updated kernels -----
            with torch.no_grad():
                norm_kernels_new = kernels / (kernels.abs().sum(dim=(-2,-1), keepdim=True) + 1e-6)
                filtered_new = F.conv2d(images, norm_kernels_new, padding=KERNEL_SIZE//2)
                filtered_reshaped_new = filtered_new.permute(1, 0, 2, 3).reshape(KERNEL_POP * B, 1, 28, 28)
                _, latents_all_new = model(filtered_reshaped_new)
                p_all_new, _ = proj_head(latents_all_new)
                p_all_new = p_all_new.view(KERNEL_POP, B, -1)
                subspace_wobbles_new = (p_all_new - p_raw.detach().unsqueeze(0)).pow(2).mean(dim=(1,2))
                weighted_wobbles_new = subspace_wobbles_new / (1.0 + chaos_factor)

            # ----- 5. Update ages, EMAs, cooldowns and apply rules -----
            ages += 1.0
            cooldowns = torch.clamp(cooldowns - 1, min=0)
            ready_mask = (cooldowns == 0)

            wobbles_detached = weighted_wobbles_new.detach()
            ema_wobbles = (EMA_ALPHA * wobbles_detached) + ((1 - EMA_ALPHA) * ema_wobbles)
            deltas = torch.abs(ema_wobbles - wobbles_detached)

            identity_wobble = wobbles_detached[0]
            safe_wobbles = wobbles_detached[safe_mask]
            safe_deltas = deltas[safe_mask]
            medoid_wobble = torch.median(safe_wobbles) if safe_mask.any() else identity_wobble
            medoid_delta = torch.median(safe_deltas) if safe_mask.any() else torch.tensor(0.001, device=device)

            current_safe_count = safe_mask.sum().item()

            # Demotion
            if current_safe_count > 1:
                safe_indices = torch.where(is_safe & ready_mask)[0]
                for s_idx in safe_indices:
                    if s_idx == 0: continue
                    if wobbles_detached[s_idx] > (medoid_wobble * 1.5):
                        is_safe[s_idx] = False
                        cooldowns[s_idx] = MUTATION_COOLDOWN
                        demoted += 1

            # Promotion
            if current_safe_count < MAX_SAFE_CAP:
                exp_ready_mask = (~is_safe) & ready_mask
                to_promote = exp_ready_mask & (wobbles_detached <= medoid_wobble) & (wobbles_detached <= identity_wobble * 1.2)
                promote_indices = torch.where(to_promote)[0]
                slots_left = MAX_SAFE_CAP - current_safe_count
                if len(promote_indices) > 0:
                    chosen = promote_indices[:slots_left]
                    is_safe[chosen] = True
                    cooldowns[chosen] = PROMOTION_COOLDOWN
                    promoted += len(chosen)

            # Pruning & mutation
            exp_ready_mask = (~is_safe) & ready_mask
            to_prune = exp_ready_mask & (deltas < medoid_delta) & (wobbles_detached > medoid_wobble)
            if to_prune.any():
                prune_indices = torch.where(to_prune)[0]
                pruned += len(prune_indices)
                safe_indices = torch.where(is_safe)[0]
                for idx in prune_indices:
                    parent_idx = safe_indices[torch.randint(0, len(safe_indices), (1,)).item()]
                    noise = torch.randn(1, 1, KERNEL_SIZE, KERNEL_SIZE, device=device) * 0.1
                    mutated = kernels[parent_idx] + noise
                    kernels[idx] = mutated / (mutated.abs().sum() + 1e-6)
                    ages[idx] = 1.0
                    ema_wobbles[idx] = 0.0
                    is_safe[idx] = False
                    cooldowns[idx] = MUTATION_COOLDOWN

        # ----- Validation -----
        model.eval()
        proj_head.eval()
        correct = total = 0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                logits, _ = model(val_images)
                preds = logits.argmax(dim=1)
                correct += (preds == val_labels).sum().item()
                total += val_labels.size(0)
        acc = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        safe_count = is_safe.sum().item()
        exp_count = KERNEL_POP - safe_count

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | ES Wobble: {es_loss:.6f} "
              f"| Safe: {safe_count}/{MAX_SAFE_CAP} | Exp: {exp_count} "
              f"| Prom: {promoted} | Dem: {demoted} | Prune: {pruned} | Val Acc: {acc:.2f}%")

    print("\nTraining complete!")