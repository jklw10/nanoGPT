import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import math
import time

try:
    import torchvision
    import torchvision.transforms as transforms
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


# =====================================================================
# 1. NEW V6 ARCHITECTURES
# =====================================================================

class AsymmetricImplicitNewton(nn.Module):
    """
    CONTENDER 1: Native Cardano Rectifier (Zero SiLU, Zero Helper Activations)
    Solves G(y) = y^3 + a*y^2 + r^2*y - B = 0.
    Setting r^2 = a^2/3 + sigma^2 + 1e-4 mathematically guarantees that
    G'(y) = 3y^2 + 2ay + r^2 > 0 strictly everywhere (Delta < 0).
    Produces an asymmetric, rectifier-like saturation curve natively!
    """
    def __init__(self, in_dim, hidden_dim, out_dim, steps=4):
        super().__init__()
        self.steps = steps
        self.w_proj1 = nn.Linear(in_dim, hidden_dim)
        self.a1 = nn.Parameter(torch.zeros(hidden_dim))
        self.sigma1 = nn.Parameter(torch.ones(hidden_dim) * 0.5)
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.w_proj2 = nn.Linear(hidden_dim, out_dim)
        self.a2 = nn.Parameter(torch.zeros(out_dim))
        self.sigma2 = nn.Parameter(torch.ones(out_dim) * 0.5)

    def _solve_layer(self, B, a, sigma):
        y = torch.zeros_like(B)
        # Guarantees strictly positive derivative: 3*r^2 > a^2
        r_sq = (a.pow(2) / 3.0) + sigma.pow(2) + 1e-4
        a_vec = a.unsqueeze(0)
        r_sq_vec = r_sq.unsqueeze(0)

        for _ in range(self.steps):
            G = y.pow(3) + a_vec * y.pow(2) + r_sq_vec * y - B
            G_prime = 3.0 * y.pow(2) + 2.0 * a_vec * y + r_sq_vec
            y = y - (G / G_prime)
        return y

    def forward(self, x):
        # 100% pure zero-finding: NO SiLU anywhere!
        B1 = self.w_proj1(x)
        h1 = self.ln1(self._solve_layer(B1, self.a1, self.sigma1))
        B2 = self.w_proj2(h1)
        return self._solve_layer(B2, self.a2, self.sigma2)

    def inspect_root_topology(self):
        with torch.no_grad():
            r1 = torch.sqrt((self.a1.pow(2) / 3.0) + self.sigma1.pow(2) + 1e-4).cpu().flatten()
            r2 = torch.sqrt((self.a2.pow(2) / 3.0) + self.sigma2.pow(2) + 1e-4).cpu().flatten()
            all_r = torch.cat([r1, r2])
            total = len(all_r)
            active = (all_r.abs() <= 1.0).sum().item()
            ghost = (all_r.abs() > 1.0).sum().item()
            s_r = torch.sort(all_r).values
            q = [s_r[min(int(total * p), total - 1)].item() for p in [0.0, 0.25, 0.5, 0.75, 1.0]]
            return active, ghost, total, q


class RegCompressiveRootKAN(nn.Module):
    """
    CONTENDER 2: Regularized Compressive RootKAN
    Takes the 64% training power of Compressive_RootKAN and adds
    channel dropout + root damping to conquer test-set generalization.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, degree=3, dropout_p=0.15):
        super().__init__()
        self.degree = degree
        self.proj1 = nn.Linear(in_dim, hidden_dim)
        self.drop = nn.Dropout(dropout_p)

        k = torch.arange(1, degree + 1, dtype=torch.float32)
        cheb_nodes = torch.cos((2 * k - 1) / (2 * degree) * math.pi)
        self.roots1 = nn.Parameter(cheb_nodes.view(-1, 1).repeat(1, hidden_dim))
        self.scale1 = nn.Parameter(torch.randn(hidden_dim) * 0.1)

        self.ln = nn.LayerNorm(hidden_dim)
        self.act = nn.SiLU()

        self.proj2 = nn.Linear(hidden_dim, out_dim)
        self.roots2 = nn.Parameter(cheb_nodes.view(-1, 1).repeat(1, out_dim))
        self.scale2 = nn.Parameter(torch.randn(out_dim) * 0.1)

    def _eval_and_compress(self, u, roots, scale):
        prod = (u - roots[0])
        for k in range(1, self.degree):
            prod = prod * (u - roots[k])
        v = prod * scale + u

        # 2-step implicit contraction: solves y^3 + y - v = 0
        y = torch.zeros_like(v)
        for _ in range(2):
            G = y.pow(3) + y - v
            G_prime = 3.0 * y.pow(2) + 1.0
            y = y - (G / G_prime)
        return y

    def forward(self, x):
        h1 = self._eval_and_compress(self.proj1(x), self.roots1, self.scale1)
        h1 = self.drop(self.act(self.ln(h1)))
        return self._eval_and_compress(self.proj2(h1), self.roots2, self.scale2)

    def inspect_root_topology(self):
        r1 = self.roots1.detach().cpu().flatten()
        r2 = self.roots2.detach().cpu().flatten()
        all_r = torch.cat([r1, r2])
        total = len(all_r)
        active = (all_r.abs() <= 1.0).sum().item()
        ghost = (all_r.abs() > 1.0).sum().item()
        s_r = torch.sort(all_r).values
        q = [s_r[min(int(total * p), total - 1)].item() for p in [0.0, 0.25, 0.5, 0.75, 1.0]]
        return active, ghost, total, q


class FastPerWeightRootKAN(nn.Module):
    """
    CONTENDER 3: Fast Dual-GEMM Per-Weight RootKAN
    Every single weight connection (j, i) gets its own learnable root r_ji!
    phi_{ji}(x_i) = W_{ji} * x_i * (x_i - r_{ji})
    Evaluated with ZERO 3D tensor overhead using the dual-GEMM identity:
    y = (x^2) @ W.T - x @ (W * R).T + bias
    """
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        # Layer 1
        self.W1 = nn.Parameter(torch.randn(hidden_dim, in_dim) / math.sqrt(in_dim))
        self.R1 = nn.Parameter(torch.randn(hidden_dim, in_dim) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act = nn.SiLU()

        # Layer 2
        self.W2 = nn.Parameter(torch.randn(out_dim, hidden_dim) / math.sqrt(hidden_dim))
        self.R2 = nn.Parameter(torch.randn(out_dim, hidden_dim) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        # 100% cuBLAS dual GEMM execution
        x_sq = x.pow(2)
        WR1 = self.W1 * self.R1
        h1 = torch.mm(x_sq, self.W1.T) - torch.mm(x, WR1.T) + self.b1
        h1 = self.act(self.ln1(h1))

        h1_sq = h1.pow(2)
        WR2 = self.W2 * self.R2
        return torch.mm(h1_sq, self.W2.T) - torch.mm(h1, WR2.T) + self.b2

    def inspect_root_topology(self):
        all_r = torch.cat([self.R1.detach().cpu().flatten(), self.R2.detach().cpu().flatten()])
        total = len(all_r)
        active = (all_r.abs() <= 1.0).sum().item()
        ghost = (all_r.abs() > 1.0).sum().item()
        s_r = torch.sort(all_r).values
        q = [s_r[min(int(total * p), total - 1)].item() for p in [0.0, 0.25, 0.5, 0.75, 1.0]]
        return active, ghost, total, q


# =====================================================================
# 2. CONTENDER 4 & BASELINES (CONTROLS)
# =====================================================================

class ModernMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class FactoredImplicitNewtonControl(nn.Module):
    """Control model: Factored Implicit Newton WITHOUT SiLU (the spittling king)"""
    def __init__(self, in_dim, hidden_dim, out_dim, steps=4):
        super().__init__()
        self.steps = steps
        self.w_proj1 = nn.Linear(in_dim, hidden_dim)
        self.r1 = nn.Parameter(torch.ones(hidden_dim) * 0.5)
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.w_proj2 = nn.Linear(hidden_dim, out_dim)
        self.r2 = nn.Parameter(torch.ones(out_dim) * 0.5)

    def _solve_factored_layer(self, B, r):
        y = torch.zeros_like(B)
        r_sq = r.pow(2).unsqueeze(0) + 1e-4
        for _ in range(self.steps):
            G = y.pow(3) + r_sq * y - B
            G_prime = 3.0 * y.pow(2) + r_sq
            y = y - (G / G_prime)
        return y

    def forward(self, x):
        B1 = self.w_proj1(x)
        h1 = self.ln1(self._solve_factored_layer(B1, self.r1))
        B2 = self.w_proj2(h1)
        return self._solve_factored_layer(B2, self.r2)

    def inspect_root_topology(self):
        all_r = torch.cat([self.r1.detach().cpu().flatten(), self.r2.detach().cpu().flatten()])
        total = len(all_r)
        active = (all_r.abs() <= 1.0).sum().item()
        ghost = (all_r.abs() > 1.0).sum().item()
        s_r = torch.sort(all_r).values
        q = [s_r[min(int(total * p), total - 1)].item() for p in [0.0, 0.25, 0.5, 0.75, 1.0]]
        return active, ghost, total, q


# =====================================================================
# 3. BENCHMARK HARNESS (TF32 + WARMUP + MEDIAN/P90)
# =====================================================================

def load_dataset(dataset_name: str, batch_size: int = 128, max_train_samples: int = 12000):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) if dataset_name == "MNIST" else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    try:
        if dataset_name == "MNIST":
            train_set = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
            test_set = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
            in_dim = 784
        elif dataset_name == "CIFAR10":
            train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
            test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
            in_dim = 3072
        else:
            raise ValueError(dataset_name)

        train_loader = DataLoader(torch.utils.data.Subset(train_set, range(min(len(train_set), max_train_samples))), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(torch.utils.data.Subset(test_set, range(min(len(test_set), 2500))), batch_size=batch_size, shuffle=False)
    except Exception as e:
        print(f"Synthetic fallback for {dataset_name} ({e})")
        in_dim = 784 if dataset_name == "MNIST" else 3072
        x_tr = torch.randn(max_train_samples, in_dim).clamp(-1.0, 1.0)
        y_tr = torch.randint(0, 10, (max_train_samples,))
        x_te = torch.randn(2500, in_dim).clamp(-1.0, 1.0)
        y_te = torch.randint(0, 10, (2500,))
        train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(x_te, y_te), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, in_dim


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_batch_latency_median(model, sample_batch, device, warmup_iters=50, bench_iters=150):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(sample_batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        latencies = []
        for _ in range(bench_iters):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(sample_batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)

    lat_tensor = torch.tensor(latencies, dtype=torch.float32)
    return torch.median(lat_tensor).item(), torch.quantile(lat_tensor, 0.90).item()


# =====================================================================
# 4. EXECUTION TOURNAMENT (V6)
# =====================================================================

def run_v6_tournament():
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    USE_TORCH_COMPILE = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on Device: {device} | torch.compile: {USE_TORCH_COMPILE} | TF32: True")

    datasets = ["MNIST", "CIFAR10"]
    epochs = 10
    lr = 0.010

    for dname in datasets:
        train_loader, test_loader, in_dim = load_dataset(dname, batch_size=128, max_train_samples=12000)

        # Matched parameter budgets (~39.8k MNIST, ~61.7k CIFAR10)
        if dname == "MNIST":
            mlp_h = 50
            newton_h = 50
            comp_h = 50
            pw_h = 25       # Dual-GEMM uses 2x weights per connection
            degree = 3
        else:  # CIFAR10
            mlp_h = 20
            newton_h = 20
            comp_h = 20
            pw_h = 10
            degree = 2

        models = {
            # --- Anchors & Baselines ---
            #"Modern_MLP":                     ModernMLP(in_dim, mlp_h, 10),
            "Factored_Implicit_Newton_noSilu": FactoredImplicitNewtonControl(in_dim, newton_h, 10, steps=4),

            # --- NEW V6 CONTENDERS ---
            # 1. Asymmetric Newton: Breaks odd-symmetry via algebraic Cardano rectification (NO SiLU!)
            "Asymmetric_Implicit_Newton":     AsymmetricImplicitNewton(in_dim, newton_h, 10, steps=4),

            # 2. Regularized Compressive RootKAN: 15% channel dropout to harness the 64% training power
            "Reg_Compressive_RootKAN":        RegCompressiveRootKAN(in_dim, comp_h, 10, degree=degree, dropout_p=0.15),

            # 3. Fast Dual-GEMM Per-Weight RootKAN: Every connection gets its own root via pure 2D GEMM!
            "Fast_PerWeight_RootKAN":         FastPerWeightRootKAN(in_dim, pw_h, 10),

            # =========================================================
            # COMMENTED OUT (ALREADY INSPECTED)
            # =========================================================
            # "Cheby_KAN_v2":                 ChebyKAN_v2(...),
            # "RootKAN_v4_PurePoly":          RootFactoredKAN_v4(...),
            # "Projected_RootKAN":            ProjectedRootKAN(...),
            # "Compressive_RootKAN_Unreg":    CompressiveRootKAN(...),
        }

        print("\n" + "=" * 132)
        print(f"DATASET: {dname} (Input Dim: {in_dim}, Epochs: {epochs})")
        print(f"{'MODEL / V6 CONTENDER':<32} | {'PARAMS':<8} | {'TRAIN ACC':<11} | {'TEST ACC':<11} | {'TEST LOSS':<11} | {'MEDIAN LAT':<11} | {'P90 LAT'}")
        print("-" * 132)

        root_reports = {}

        for name, model in models.items():
            torch.manual_seed(42)
            model = model.to(device)
            params = count_parameters(model)

            compiled_model = torch.compile(model) if (USE_TORCH_COMPILE and hasattr(torch, "compile")) else model

            optimizer = torch.optim.AdamW(compiled_model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader), eta_min=1e-5)
            loss_fn = nn.CrossEntropyLoss()

            compiled_model.train()
            for ep in range(epochs):
                for x_b, y_b in train_loader:
                    x_b = x_b.view(x_b.size(0), -1).to(device)
                    y_b = y_b.to(device)

                    optimizer.zero_grad()
                    out = compiled_model(x_b)
                    loss = loss_fn(out, y_b)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

            compiled_model.eval()
            correct, total, total_loss = 0, 0, 0.0
            with torch.no_grad():
                for x_b, y_b in test_loader:
                    x_b = x_b.view(x_b.size(0), -1).to(device)
                    y_b = y_b.to(device)
                    logits = compiled_model(x_b)
                    total_loss += loss_fn(logits, y_b).item() * x_b.size(0)
                    correct += (logits.argmax(dim=-1) == y_b).sum().item()
                    total += x_b.size(0)

            test_acc = (correct / total) * 100.0
            test_loss = total_loss / total

            sample_x, sample_y = next(iter(train_loader))
            sample_x = sample_x.view(sample_x.size(0), -1).to(device)
            train_acc = (compiled_model(sample_x).argmax(dim=-1) == sample_y.to(device)).float().mean().item() * 100.0

            med_ms, p90_ms = benchmark_batch_latency_median(compiled_model, sample_x, device, warmup_iters=50, bench_iters=150)

            if hasattr(model, "inspect_root_topology"):
                root_reports[name] = model.inspect_root_topology()

            print(f"{name:<32} | {params:<8d} | {train_acc:<10.2f}% | {test_acc:<10.2f}% | {test_loss:<11.5f} | {med_ms:<8.3f} ms | {p90_ms:.3f} ms")

        if root_reports:
            print("\n" + "-" * 132)
            print(f"ROOT TOPOLOGY DIAGNOSTICS FOR {dname}:")
            for m_name, (active, ghost, tot, q) in root_reports.items():
                print(f"  {m_name:<32} -> Active: {active}/{tot} ({active/tot*100:.1f}%) | "
                      f"Ghosts: {ghost}/{tot} ({ghost/tot*100:.1f}%) | "
                      f"Spread [Min➔Q1➔Med➔Q3➔Max]: [{q[0]:+.2f} ➔ {q[1]:+.2f} ➔ {q[2]:+.2f} ➔ {q[3]:+.2f} ➔ {q[4]:+.2f}]")
        print("=" * 132)


if __name__ == "__main__":
    run_v6_tournament()