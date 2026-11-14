import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")

# =====================================================================
# 1. Topological Primitives (Divisive Tree)
# =====================================================================

def power_iteration_step(X: torch.Tensor, v: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    z = torch.mv(X, v) - torch.dot(mu, v)
    v_next = torch.mv(X.T, z) - z.sum() * mu
    return v_next / (torch.linalg.norm(v_next) + 1e-9)

def get_normal_power_iteration(X: torch.Tensor, iters: int = 2) -> torch.Tensor:
    n, d = X.shape
    mu = X.mean(dim=0)
    v = torch.randn(d, device=X.device, dtype=X.dtype)
    v = v / (torch.linalg.norm(v) + 1e-9)
    for _ in range(iters):
        v = power_iteration_step(X, v, mu)
    return v

def split_dense_island(proj: torch.Tensor, num_bins: int = 64) -> torch.Tensor:
    span = proj.max() - proj.min()
    if span <= 1e-7:
        return proj.median()
    
    n_pts = proj.shape[0]
    sample_proj = proj[torch.randint(0, n_pts, (min(4096, n_pts),), device=proj.device)]
    min_v, max_v = sample_proj.min(), sample_proj.max()
    
    counts = torch.histc(sample_proj, bins=num_bins, min=min_v.item(), max=max_v.item())
    is_dense = counts > (0.35 * counts.mean())
    dense_idxs = torch.where(is_dense)[0]
    
    if dense_idxs.numel() < 2:
        return torch.median(sample_proj)
    
    first, last = dense_idxs[0], dense_idxs[-1]
    bin_idxs = torch.arange(num_bins, device=proj.device)
    valid_mask = (bin_idxs >= first) & (bin_idxs <= last)
    empty_between = (~is_dense) & valid_mask
    bin_width = (max_v - min_v) / num_bins
    
    padded = F.pad(empty_between.float(), (1, 1), value=0.0)
    diff = padded[1:] - padded[:-1]
    starts = torch.where(diff == 1)[0]
    ends = torch.where(diff == -1)[0]
    
    if starts.numel() > 0:
        lengths = ends - starts
        max_idx = torch.argmax(lengths)
        best_end = (ends[max_idx] - 1).float()
        best_center = best_end - (lengths[max_idx].float() / 2.0)
        return min_v + best_center * bin_width
            
    return torch.median(sample_proj)

def find_topological_centroids(X: torch.Tensor, y: torch.Tensor, k: int) -> torch.Tensor:
    clusters = [torch.arange(X.shape[0], device=X.device)]
    while len(clusters) < k:
        clusters.sort(key=lambda idxs: len(idxs), reverse=True)
        idxs = clusters.pop(0)
        if len(idxs) <= 4:
            clusters.append(idxs)
            break
            
        sub = idxs[torch.randint(0, len(idxs), (min(2048, len(idxs)),), device=X.device)]
        w = get_normal_power_iteration(X[sub])
        proj = torch.mv(X[idxs], w)
        thresh = split_dense_island(proj)
        
        l_mask = proj < thresh
        r_mask = ~l_mask
        if l_mask.sum() == 0 or r_mask.sum() == 0:
            thresh = torch.median(proj)
            l_mask, r_mask = proj < thresh, proj >= thresh
            
        clusters.append(idxs[l_mask])
        clusters.append(idxs[r_mask])
        
    raw_C = torch.stack([X[c].mean(dim=0) for c in clusters[:k]])
    
    # Class alignment
    ordered_C = torch.zeros(k, X.shape[1], device=X.device)
    for c in range(k):
        mask = (y == c)
        if mask.sum() > 0:
            class_mean = X[mask].mean(dim=0)
            closest_idx = torch.argmin(torch.norm(raw_C - class_mean, dim=1))
            ordered_C[c] = raw_C[closest_idx]
        else:
            ordered_C[c] = raw_C[c]
    return ordered_C

# =====================================================================
# 2. Architectures
# =====================================================================

class StandardModel(nn.Module):
    def __init__(self, in_dim=784, hidden_dim=256, emb_dim=128, num_classes=10):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
            nn.LayerNorm(emb_dim)
        )
        self.head = nn.Linear(emb_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats)

class VoronoiModel(nn.Module):
    def __init__(self, in_dim=784, hidden_dim=256, emb_dim=128, num_classes=10, alpha=0.10):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
            nn.LayerNorm(emb_dim)
        )
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.alpha = alpha
        self.register_buffer("centroids", torch.randn(num_classes, emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        c_half_norm = 0.5 * (self.centroids * self.centroids).sum(dim=1)
        return torch.addmm(-c_half_norm, feats, self.centroids.T)

    def training_step(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        
        with torch.no_grad():
            s_batch = torch.zeros_like(self.centroids)
            n_batch = torch.zeros(self.num_classes, 1, device=x.device, dtype=x.dtype)
            s_batch.scatter_add_(0, targets.unsqueeze(1).expand(-1, self.emb_dim), feats)
            n_batch.scatter_add_(0, targets.unsqueeze(1), torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype))
            
            dead_mask = (n_batch == 0)
            fallback = feats[:self.num_classes]
            s_batch = torch.where(dead_mask, fallback, s_batch)
            n_batch = torch.where(dead_mask, torch.ones_like(n_batch), n_batch)
            
            batch_means = s_batch / n_batch
            self.centroids.mul_(1.0 - self.alpha).add_(batch_means, alpha=self.alpha)

        c_half_norm = 0.5 * (self.centroids * self.centroids).sum(dim=1)
        return torch.addmm(-c_half_norm, feats, self.centroids.T)

# =====================================================================
# 3. Data Loader with Exact Batch Multiples
# =====================================================================

def load_mnist_exact_batches(batch_size=512):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST("./data", train=False, download=True, transform=transform)
    
    X_train = ((train_set.data.view(-1, 784).float().to(device) / 255.0) - 0.1307) / 0.3081
    y_train = train_set.targets.to(device)
    X_test = ((test_set.data.view(-1, 784).float().to(device) / 255.0) - 0.1307) / 0.3081
    y_test = test_set.targets.to(device)
    
    # Drop remainder samples to guarantee zero shape-mismatch recompilations
    n_train_exact = (X_train.shape[0] // batch_size) * batch_size
    n_test_exact = (X_test.shape[0] // batch_size) * batch_size
    
    return X_train[:n_train_exact], y_train[:n_train_exact], X_test[:n_test_exact], y_test[:n_test_exact]

def evaluate_acc(model, X, y, batch_size=512):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()
            bx = X[i:i + batch_size]
            by = y[i:i + batch_size]
            preds = torch.argmax(model(bx), dim=1)
            correct += (preds == by).sum().item()
    return (correct / X.shape[0]) * 100.0

# =====================================================================
# 4. Main Benchmark Runner
# =====================================================================

def run_mnist_experiment():
    batch_size = 512
    epochs = 5
    D, K = 128, 10
    
    X_train, y_train, X_test, y_test = load_mnist_exact_batches(batch_size=batch_size)
    n_batches = X_train.shape[0] // batch_size
    
    print(f"Device: {device.type.upper()} | Exact Batches: Train={n_batches}x{batch_size} ({X_train.shape[0]:,}), Test={X_test.shape[0]//batch_size}x{batch_size} ({X_test.shape[0]:,})")
    print("=" * 88)

    compile_mode = "reduce-overhead" if device.type == "cuda" else None

    # -----------------------------------------------------------------
    # 1. BASELINE: Standard Linear Backprop
    # -----------------------------------------------------------------
    torch.manual_seed(42)
    model_base = torch.compile(StandardModel(emb_dim=D).to(device), mode=compile_mode)
    opt_base = torch.optim.Adam(model_base.parameters(), lr=1e-3)

    # Method-specific warmup for full forward + backward graph
    for _ in range(3):
        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        opt_base.zero_grad()
        loss = F.cross_entropy(model_base(X_train[:batch_size]), y_train[:batch_size])
        loss.backward()
        opt_base.step()
    evaluate_acc(model_base, X_test[:batch_size], y_test[:batch_size], batch_size=batch_size)
    if device.type == "cuda": torch.cuda.synchronize()

    acc0 = evaluate_acc(model_base, X_test, y_test, batch_size=batch_size)
    print("--- 1. Baseline Backprop (Standard Linear Head) ---")
    print(f"  Epoch 0 (Untrained Baseline): {acc0:.2f}%")
    
    if device.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model_base.train()
        for b in range(n_batches):
            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()
            bx = X_train[b * batch_size:(b + 1) * batch_size]
            by = y_train[b * batch_size:(b + 1) * batch_size]
            opt_base.zero_grad()
            loss = F.cross_entropy(model_base(bx), by)
            loss.backward()
            opt_base.step()
        acc1 = evaluate_acc(model_base, X_test, y_test, batch_size=batch_size)
        print(f"  Epoch {ep}: Test Acc = {acc1:.2f}%")
    if device.type == "cuda": torch.cuda.synchronize()
    t_base = (time.perf_counter() - t0) * 1000

    # -----------------------------------------------------------------
    # 2. PERSPECTIVE A: Divisive Island Tree Seeding (Cold-Start)
    # -----------------------------------------------------------------
    torch.manual_seed(42)
    model_a_raw = StandardModel(emb_dim=D).to(device)
    
    with torch.no_grad():
        init_feats = model_a_raw.backbone(X_train[:4096])
        seeded_C = find_topological_centroids(init_feats, y_train[:4096], k=K)
        model_a_raw.head.weight.copy_(seeded_C)
        model_a_raw.head.bias.copy_(-0.5 * (seeded_C * seeded_C).sum(dim=1))
        
    model_a = torch.compile(model_a_raw, mode=compile_mode)
    opt_a = torch.optim.Adam(model_a.parameters(), lr=1e-3)

    # Method-specific warmup
    for _ in range(3):
        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        opt_a.zero_grad()
        loss = F.cross_entropy(model_a(X_train[:batch_size]), y_train[:batch_size])
        loss.backward()
        opt_a.step()
    evaluate_acc(model_a, X_test[:batch_size], y_test[:batch_size], batch_size=batch_size)
    if device.type == "cuda": torch.cuda.synchronize()

    acc0_a = evaluate_acc(model_a, X_test, y_test, batch_size=batch_size)
    print(f"\n--- 2. Perspective A (Topological Cold-Start Head) ---")
    print(f"  Epoch 0 (Cold-Start Initialized): {acc0_a:.2f}% (vs {acc0:.2f}% random)")

    if device.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model_a.train()
        for b in range(n_batches):
            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()
            bx = X_train[b * batch_size:(b + 1) * batch_size]
            by = y_train[b * batch_size:(b + 1) * batch_size]
            opt_a.zero_grad()
            loss = F.cross_entropy(model_a(bx), by)
            loss.backward()
            opt_a.step()
        acc2 = evaluate_acc(model_a, X_test, y_test, batch_size=batch_size)
        print(f"  Epoch {ep}: Test Acc = {acc2:.2f}%")
    if device.type == "cuda": torch.cuda.synchronize()
    t_a = (time.perf_counter() - t0) * 1000

    # -----------------------------------------------------------------
    # 3. PERSPECTIVE B: Fused Gradient-Free Voronoi Head
    # -----------------------------------------------------------------
    torch.manual_seed(42)
    model_b_raw = VoronoiModel(emb_dim=D, num_classes=K, alpha=0.10).to(device)

    with torch.no_grad():
        init_f = model_b_raw.backbone(X_train[:4096])
        for c in range(K):
            mask = (y_train[:4096] == c)
            if mask.sum() > 0:
                model_b_raw.centroids[c] = init_f[mask].mean(dim=0)

    model_b = torch.compile(model_b_raw, mode=compile_mode)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=1e-3)

    # Method-specific warmup for the training_step graph
    for _ in range(3):
        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        opt_b.zero_grad()
        logits = model_b.training_step(X_train[:batch_size], y_train[:batch_size])
        loss = F.cross_entropy(logits, y_train[:batch_size])
        loss.backward()
        opt_b.step()
    evaluate_acc(model_b, X_test[:batch_size], y_test[:batch_size], batch_size=batch_size)
    if device.type == "cuda": torch.cuda.synchronize()

    acc0_b = evaluate_acc(model_b, X_test, y_test, batch_size=batch_size)
    print(f"\n--- 3. Perspective B (Gradient-Free Head: No Autograd on Classifier) ---")
    print(f"  Epoch 0 (Direct Centroid Means): {acc0_b:.2f}%")

    if device.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model_b.train()
        for b in range(n_batches):
            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()
            bx = X_train[b * batch_size:(b + 1) * batch_size]
            by = y_train[b * batch_size:(b + 1) * batch_size]
            opt_b.zero_grad()
            
            logits = model_b.training_step(bx, by)
            loss = F.cross_entropy(logits, by)
            loss.backward()
            opt_b.step()
            
        acc3 = evaluate_acc(model_b, X_test, y_test, batch_size=batch_size)
        print(f"  Epoch {ep}: Test Acc = {acc3:.2f}%")
    if device.type == "cuda": torch.cuda.synchronize()
    t_b = (time.perf_counter() - t0) * 1000

    # -----------------------------------------------------------------
    # Summary Table
    # -----------------------------------------------------------------
    print("\n" + "=" * 88)
    print(f"{'Method':<40} | {'Epoch 0 Acc':<12} | {'Epoch 5 Acc':<12} | {'Time (5 eps)'}")
    print("-" * 88)
    print(f"{'1. Standard Linear (Backprop)':<40} | {acc0:<11.2f}% | {acc1:<11.2f}% | {t_base:.1f} ms")
    print(f"{'2. Perspective A (Topological Seed)':<40} | {acc0_a:<11.2f}% | {acc2:<11.2f}% | {t_a:.1f} ms")
    print(f"{'3. Perspective B (Grad-Free Voronoi)':<40} | {acc0_b:<11.2f}% | {acc3:<11.2f}% | {t_b:.1f} ms")
    print("=" * 88)

if __name__ == "__main__":
    run_mnist_experiment()