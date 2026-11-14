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
# 1. NEW HYBRID ARCHITECTURES ("BEST OF BOTH WORLDS")
# =====================================================================

class ProjectedRootKAN(nn.Module):
    """
    HYBRID 1: Projected RootKAN
    - Dense spatial projection (W1) solves the vision width-starvation bottleneck:
      Hidden dimension expands from h=10 to h=50 (MNIST) and h=5 to h=20 (CIFAR-10).
    - Evaluates fused, unnormalized root polynomials along mixed latent channels.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, degree=3):
        super().__init__()
        self.degree = degree
        self.proj1 = nn.Linear(in_dim, hidden_dim)
        
        # Channel-wise roots: (degree, hidden_dim)
        k = torch.arange(1, degree + 1, dtype=torch.float32)
        cheb_nodes = torch.cos((2 * k - 1) / (2 * degree) * math.pi)
        self.roots1 = nn.Parameter(cheb_nodes.view(-1, 1).repeat(1, hidden_dim))
        self.scale1 = nn.Parameter(torch.randn(hidden_dim) * 0.1)

        self.ln = nn.LayerNorm(hidden_dim)
        self.act = nn.SiLU()

        self.proj2 = nn.Linear(hidden_dim, out_dim)
        self.roots2 = nn.Parameter(cheb_nodes.view(-1, 1).repeat(1, out_dim))
        self.scale2 = nn.Parameter(torch.randn(out_dim) * 0.1)

    def _eval_root_poly(self, u, roots, scale):
        # Fused unnormalized factor product
        prod = (u - roots[0])
        for k in range(1, self.degree):
            prod = prod * (u - roots[k])
        return prod * scale + u  # Residual linear bypass

    def forward(self, x):
        u1 = self.proj1(x)
        h1 = self._eval_root_poly(u1, self.roots1, self.scale1)
        h1 = self.act(self.ln(h1))

        u2 = self.proj2(h1)
        return self._eval_root_poly(u2, self.roots2, self.scale2)

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


class FactoredImplicitNewton(nn.Module):
    """
    HYBRID 2: Factored Implicit Newton Zero-Finder
    - Solves an implicit zero-finding equation where curvature is parameterized by
      learnable roots rather than arbitrary cubic coefficients:
      G(y) = y * (y^2 + r^2) - B(x) = 0
    - The positive term (y^2 + r^2) guarantees G'(y) > 0 strictly, eliminating
      division-by-zero singularities without heuristic clamps.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, steps=4):
        super().__init__()
        self.steps = steps
        self.w_proj1 = nn.Linear(in_dim, hidden_dim)
        # Learnable root parameters controlling saturation curvature
        self.r1 = nn.Parameter(torch.ones(hidden_dim) * 0.5)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act = nn.SiLU()

        self.w_proj2 = nn.Linear(hidden_dim, out_dim)
        self.r2 = nn.Parameter(torch.ones(out_dim) * 0.5)

    def _solve_factored_layer(self, B, r):
        y = torch.zeros_like(B)
        r_sq = r.pow(2).unsqueeze(0) + 1e-4

        for _ in range(self.steps):
            # G(y) = y^3 + r^2 * y - B = 0
            G = y.pow(3) + r_sq * y - B
            # G'(y) = 3*y^2 + r^2 > 0 (Strictly positive denominator)
            G_prime = 3.0 * y.pow(2) + r_sq
            y = y - (G / G_prime)
        return y

    def forward(self, x):
        B1 = self.w_proj1(x)
        h1 = self._solve_factored_layer(B1, self.r1)
        h1 = self.ln1(h1)
        #h1 = self.act(self.ln1(h1))

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


class CompressiveRootKAN(nn.Module):
    """
    HYBRID 3: Compressive RootKAN
    - Evaluates high-frequency root products on mixed projections, then immediately
      pipes the output into a 2-step implicit contraction:
      y^3 + y - P(u) = 0  => y ~= cbrt(P(u))
    - Algebraically compresses runaway polynomial outputs into smooth, bounded features.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, degree=3):
        super().__init__()
        self.degree = degree
        self.proj1 = nn.Linear(in_dim, hidden_dim)

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
        # 1. Forward Root Product
        prod = (u - roots[0])
        for k in range(1, self.degree):
            prod = prod * (u - roots[k])
        v = prod * scale + u

        # 2. 2-Step Implicit Contraction: solves y^3 + y - v = 0
        y = torch.zeros_like(v)
        for _ in range(2):
            G = y.pow(3) + y - v
            G_prime = 3.0 * y.pow(2) + 1.0
            y = y - (G / G_prime)
        return y

    def forward(self, x):
        h1 = self._eval_and_compress(self.proj1(x), self.roots1, self.scale1)
        h1 = self.act(self.ln(h1))
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


# =====================================================================
# 2. ESTABLISHED ANCHORS & BASELINES (PRESERVED)
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


class ChebyKANLayer_v2(nn.Module):
    def __init__(self, in_features, out_features, degree=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.weight = nn.Parameter(torch.randn(out_features, in_features * (degree + 1)) * 0.1)
        self.base_linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x_c = torch.clamp(x, -1.0, 1.0)
        T = [torch.ones_like(x_c), x_c]
        for _ in range(2, self.degree + 1):
            T.append(2.0 * x_c * T[-1] - T[-2])
        T_cat = torch.cat(T, dim=-1)
        return torch.mm(T_cat, self.weight.T) / math.sqrt(self.in_features) + self.base_linear(x)


class ChebyKAN_v2(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, degree=3):
        super().__init__()
        self.layer1 = ChebyKANLayer_v2(in_dim, hidden_dim, degree=degree)
        self.ln = nn.LayerNorm(hidden_dim)
        self.act = nn.SiLU()
        self.layer2 = ChebyKANLayer_v2(hidden_dim, out_dim, degree=degree)

    def forward(self, x):
        return self.layer2(self.act(self.ln(self.layer1(x))))


class ImplicitNewtonKAN_v3(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, steps=4):
        super().__init__()
        self.steps = steps
        self.w0_1 = nn.Linear(in_dim, hidden_dim)
        self.w1_1 = nn.Linear(in_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act = nn.SiLU()
        self.w0_2 = nn.Linear(hidden_dim, out_dim)
        self.w1_2 = nn.Linear(hidden_dim, out_dim)

    def _solve(self, x, w0, w1):
        A = torch.relu(w1(x)) + 0.1
        B = w0(x)
        y = torch.zeros_like(B)
        for _ in range(self.steps):
            G = y.pow(3) + A * y - B
            G_prime = 3.0 * y.pow(2) + A + 1e-6
            y = y - (G / G_prime)
        return y

    def forward(self, x):
        return self._solve(self.act(self.ln1(self._solve(x, self.w0_1, self.w1_1))), self.w0_2, self.w1_2)


class RootKANLayer_v4(nn.Module):
    def __init__(self, in_features, out_features, degree=3, use_factor_norm=False, use_base_linear=True):
        super().__init__()
        self.in_features = in_features
        self.degree = degree
        self.use_factor_norm = use_factor_norm
        self.use_base_linear = use_base_linear

        k = torch.arange(1, degree + 1, dtype=torch.float32)
        nodes = torch.cos((2 * k - 1) / (2 * degree) * math.pi)
        self.roots = nn.Parameter(nodes.view(-1, 1, 1).repeat(1, out_features, in_features))
        self.scale = nn.Parameter(torch.randn(out_features, in_features) * 0.1)

        if use_base_linear:
            self.base_linear = nn.Linear(in_features, out_features)
        else:
            self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        x_in = x.unsqueeze(1)
        if self.use_factor_norm:
            denoms = torch.sqrt(1.0 + self.roots.pow(2))
            prod = (x_in - self.roots[0]) / denoms[0]
            for k in range(1, self.degree):
                prod = prod * ((x_in - self.roots[k]) / denoms[k])
        else:
            prod = (x_in - self.roots[0])
            for k in range(1, self.degree):
                prod = prod * (x_in - self.roots[k])

        kan_out = torch.sum(prod * self.scale, dim=-1) / math.sqrt(self.in_features)
        return kan_out + (self.base_linear(x) if self.use_base_linear else self.bias)


class RootFactoredKAN_v4(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, degree=3, use_factor_norm=False, use_base_linear=True):
        super().__init__()
        self.layer1 = RootKANLayer_v4(in_dim, hidden_dim, degree=degree, use_factor_norm=use_factor_norm, use_base_linear=use_base_linear)
        self.ln = nn.LayerNorm(hidden_dim)
        self.act = nn.SiLU()
        self.layer2 = RootKANLayer_v4(hidden_dim, out_dim, degree=degree, use_factor_norm=use_factor_norm, use_base_linear=use_base_linear)

    def forward(self, x):
        return self.layer2(self.act(self.ln(self.layer1(x))))

    def inspect_root_topology(self):
        all_r = torch.cat([self.layer1.roots.detach().cpu().flatten(), self.layer2.roots.detach().cpu().flatten()])
        total = len(all_r)
        active = (all_r.abs() <= 1.0).sum().item()
        ghost = (all_r.abs() > 1.0).sum().item()
        s_r = torch.sort(all_r).values
        q = [s_r[min(int(total * p), total - 1)].item() for p in [0.0, 0.25, 0.5, 0.75, 1.0]]
        return active, ghost, total, q


# =====================================================================
# 3. DATA LOADER & BENCHMARK HARNESS
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
        print(f"Warning: Synthetic fallback for {dataset_name} ({e})")
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
# 4. EXECUTION TOURNAMENT (V5)
# =====================================================================

def run_v5_tournament():
    USE_TORCH_COMPILE = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on Device: {device} | torch.compile: {USE_TORCH_COMPILE}")

    datasets = ["MNIST", "CIFAR10"]
    epochs = 10
    lr = 0.010

    for dname in datasets:
        train_loader, test_loader, in_dim = load_dataset(dname, batch_size=128, max_train_samples=12000)

        # Budget-matched parameters (~39.8k MNIST, ~61.7k CIFAR10)
        if dname == "MNIST":
            mlp_h = 50
            kan_h = 10
            proj_h = 50       # Projected hybrids get full width!
            newton_h = 25
            factored_newton_h = 50  # 1 projection matrix instead of 2!
            degree = 3
        else:  # CIFAR10
            mlp_h = 20
            kan_h = 5
            proj_h = 20       # Projected hybrids get full width!
            newton_h = 10
            factored_newton_h = 20
            degree = 2

        # =============================================================
        # CONTENDERS FOR V5 TOURNAMENT
        # =============================================================
        models = {
            # --- Anchors / Previous Champions ---
            #"Modern_MLP":               ModernMLP(in_dim, mlp_h, 10),
            #"Cheby_KAN_v2":             ChebyKAN_v2(in_dim, kan_h, 10, degree=degree),
            #"Implicit_Newton_v3":       ImplicitNewtonKAN_v3(in_dim, newton_h, 10, steps=4),
            #"RootKAN_v4_Lean":          RootFactoredKAN_v4(in_dim, kan_h, 10, degree=degree, use_factor_norm=False, use_base_linear=True),
            #"RootKAN_v4_PurePoly":      RootFactoredKAN_v4(in_dim, kan_h, 10, degree=degree, use_factor_norm=False, use_base_linear=False),

            # --- NEW V5 HYBRIDS ("BEST OF BOTH WORLDS") ---
            # 1. Projected RootKAN: Dense spatial mixing gives full width (h=50/20) + fused root products
            #"Projected_RootKAN":        ProjectedRootKAN(in_dim, proj_h, 10, degree=degree),

            # 2. Factored Implicit Newton: Implicit zero-finding equation written in root form
            "Factored_Implicit_Newton": FactoredImplicitNewton(in_dim, factored_newton_h, 10, steps=4),

            # 3. Compressive RootKAN: High-frequency root product + 2-step implicit contraction
            #"Compressive_RootKAN":      CompressiveRootKAN(in_dim, proj_h, 10, degree=degree),

            # =========================================================
            # COMMENTED OUT (ALREADY INSPECTED)
            # =========================================================
            # "RootKAN_v4_Full":        RootFactoredKAN_v4(in_dim, kan_h, 10, degree=degree, use_factor_norm=True, use_base_linear=True),
            # "RootKAN_v4_Uniform_Lean":(RootFactoredKAN_v4(in_dim, kan_h, 10, degree=degree, use_factor_norm=False, use_base_linear=True),
            # "RootKAN_v4_NoNorm_Repel":(RootFactoredKAN_v4(in_dim, kan_h, 10, degree=degree, use_factor_norm=False, use_base_linear=True),
        }

        print("\n" + "=" * 132)
        print(f"DATASET: {dname} (Input Dim: {in_dim}, Epochs: {epochs})")
        print(f"{'MODEL / HYBRID':<26} | {'PARAMS':<8} | {'TRAIN ACC':<11} | {'TEST ACC':<11} | {'TEST LOSS':<11} | {'MEDIAN LAT':<11} | {'P90 LAT'}")
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

            print(f"{name:<26} | {params:<8d} | {train_acc:<10.2f}% | {test_acc:<10.2f}% | {test_loss:<11.5f} | {med_ms:<8.3f} ms | {p90_ms:.3f} ms")

        if root_reports:
            print("\n" + "-" * 132)
            print(f"ROOT TOPOLOGY DIAGNOSTICS FOR {dname}:")
            for m_name, (active, ghost, tot, q) in root_reports.items():
                print(f"  {m_name:<26} -> Active: {active}/{tot} ({active/tot*100:.1f}%) | "
                      f"Ghosts: {ghost}/{tot} ({ghost/tot*100:.1f}%) | "
                      f"Spread [Min➔Q1➔Med➔Q3➔Max]: [{q[0]:+.2f} ➔ {q[1]:+.2f} ➔ {q[2]:+.2f} ➔ {q[3]:+.2f} ➔ {q[4]:+.2f}]")
        print("=" * 132)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    run_v5_tournament()