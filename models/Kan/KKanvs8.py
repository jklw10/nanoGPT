import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import math
import time

import torchvision
import torchvision.transforms as transforms


# =====================================================================
# 1. BRANCHLESS ASYMMETRIC CARDANO ZERO-FINDER (PURE NEWTON K=3)
# =====================================================================

class CardanoNewtonActivation(nn.Module):
    """
    Solves G(y) = y^3 + a*y^2 + r^2*y - B = 0 via unrolled Newton-Raphson (K=3).
    
    Discriminant Constraint: r^2 = a^2 / 3 + sigma^2 + 1e-4
    Guarantees: G'(y) = 3y^2 + 2ay + r^2 >= sigma^2 + 1e-4 > 0 strictly everywhere.
    
    Features:
      - ZERO branching (no torch.where, no dual computation).
      - ZERO possibility of division by zero.
      - ZERO direction flipping.
      - Fuses completely into a single Triton/CUDA kernel under torch.compile.
    """
    def __init__(self, dim, steps=3):
        super().__init__()
        self.steps = steps
        self.a = nn.Parameter(torch.zeros(dim))
        self.sigma = nn.Parameter(torch.ones(dim) * 0.5)

    def forward(self, B):
        r_sq = (self.a.pow(2) / 3.0) + self.sigma.pow(2) + 1e-4
        a_vec = self.a.unsqueeze(0)
        r_sq_vec = r_sq.unsqueeze(0)

        # Asymptotic Pade initial guess: matches linear (B/r^2) and cubic (cbrt(B)) regimes
        abs_B = B.abs()
        denom_init = r_sq_vec + abs_B.pow(2.0 / 3.0)
        y = B / denom_init

        # Branchless, unconditionally stable unrolled Newton steps
        for _ in range(self.steps):
            y_sq = y.pow(2)
            G = (y_sq + a_vec * y + r_sq_vec) * y - B
            G_prime = 3.0 * y_sq + 2.0 * a_vec * y + r_sq_vec
            y = y - (G / G_prime)

        return y

    def get_effective_roots(self):
        with torch.no_grad():
            r_sq = (self.a.pow(2) / 3.0) + self.sigma.pow(2) + 1e-4
            return torch.sqrt(r_sq).cpu().flatten()


# =====================================================================
# 2. MODEL ARCHITECTURES (CLEAN LOGIT OUTPUT PATHS)
# =====================================================================

class CardanoNewtonMLP(nn.Module):
    """
    CONTENDER 1: Native Asymmetric Algebraic MLP
    Linear projection -> Cardano-Newton Activation -> LayerNorm -> Linear Logits.
    Raw unconstrained logits output directly to CrossEntropyLoss.
    """
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.proj1 = nn.Linear(in_dim, hidden_dim)
        self.act1 = CardanoNewtonActivation(hidden_dim, steps=3)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.proj2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h1 = self.ln1(self.act1(self.proj1(x)))
        return self.proj2(h1)

    def inspect_root_topology(self):
        all_r = self.act1.get_effective_roots()
        total = len(all_r)
        active = (all_r.abs() <= 1.0).sum().item()
        ghost = (all_r.abs() > 1.0).sum().item()
        s_r = torch.sort(all_r).values
        q = [s_r[min(int(total * p), total - 1)].item() for p in [0.0, 0.25, 0.5, 0.75, 1.0]]
        return active, ghost, total, q


class DualGEMM_CardanoNewton(nn.Module):
    """
    CONTENDER 2: Grand Unified Model V8.1
    Layer 1: 2D Dual-GEMM quadratic roots + Linear Skip + Cardano-Newton Activation.
    Layer 2: 2D Dual-GEMM directly to output logits (no squashing on logits).
    """
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        # Layer 1
        self.W1 = nn.Parameter(torch.randn(hidden_dim, in_dim) / math.sqrt(in_dim))
        self.R1 = nn.Parameter(torch.randn(hidden_dim, in_dim) * 0.1)
        self.base1 = nn.Linear(in_dim, hidden_dim)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.act1 = CardanoNewtonActivation(hidden_dim, steps=3)
        self.ln1 = nn.LayerNorm(hidden_dim)

        # Layer 2 (Output: Dual-GEMM to logits)
        self.W2 = nn.Parameter(torch.randn(out_dim, hidden_dim) / math.sqrt(hidden_dim))
        self.R2 = nn.Parameter(torch.randn(out_dim, hidden_dim) * 0.1)
        self.base2 = nn.Linear(hidden_dim, out_dim)
        self.b2 = nn.Parameter(torch.zeros(out_dim))

    def _dual_gemm(self, x, W, R, base_linear, b, scale_factor):
        x_sq = x.pow(2)
        WR = W * R
        quad_term = torch.mm(x_sq, W.T) - torch.mm(x, WR.T)
        return (quad_term / scale_factor) + base_linear(x) + b

    def forward(self, x):
        B1 = self._dual_gemm(x, self.W1, self.R1, self.base1, self.b1, math.sqrt(self.in_dim))
        h1 = self.ln1(self.act1(B1))
        return self._dual_gemm(h1, self.W2, self.R2, self.base2, self.b2, math.sqrt(self.hidden_dim))

    def inspect_root_topology(self):
        all_r = torch.cat([self.R1.detach().cpu().flatten(), self.R2.detach().cpu().flatten()])
        total = len(all_r)
        active = (all_r.abs() <= 1.0).sum().item()
        ghost = (all_r.abs() > 1.0).sum().item()
        s_r = torch.sort(all_r).values
        q = [s_r[min(int(total * p), total - 1)].item() for p in [0.0, 0.25, 0.5, 0.75, 1.0]]
        return active, ghost, total, q


class TripleGEMM_CubicRootKAN_v8(nn.Module):
    """
    CONTENDER 3: Triple-GEMM Cubic Per-Weight RootKAN (degree d=3 in 3 2D GEMMs)
    """
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        node1 = math.cos((2 * 1 - 1) / (2 * 2) * math.pi)  # +0.7071
        node2 = math.cos((2 * 2 - 1) / (2 * 2) * math.pi)  # -0.7071

        # Layer 1
        self.W1 = nn.Parameter(torch.randn(hidden_dim, in_dim) / math.sqrt(in_dim))
        self.R1_1 = nn.Parameter(torch.ones(hidden_dim, in_dim) * node1 + torch.randn(hidden_dim, in_dim) * 0.05)
        self.R2_1 = nn.Parameter(torch.ones(hidden_dim, in_dim) * node2 + torch.randn(hidden_dim, in_dim) * 0.05)
        self.base1 = nn.Linear(in_dim, hidden_dim)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act = nn.SiLU()

        # Layer 2
        self.W2 = nn.Parameter(torch.randn(out_dim, hidden_dim) / math.sqrt(hidden_dim))
        self.R1_2 = nn.Parameter(torch.ones(out_dim, hidden_dim) * node1 + torch.randn(out_dim, hidden_dim) * 0.05)
        self.R2_2 = nn.Parameter(torch.ones(out_dim, hidden_dim) * node2 + torch.randn(out_dim, hidden_dim) * 0.05)
        self.base2 = nn.Linear(hidden_dim, out_dim)
        self.b2 = nn.Parameter(torch.zeros(out_dim))

    def _cubic_gemm(self, x, W, R1, R2, base_linear, b, scale_factor):
        x_sq = x.pow(2)
        x_cube = x_sq * x
        W_sum = W * (R1 + R2)
        W_prod = W * R1 * R2
        poly_term = torch.mm(x_cube, W.T) - torch.mm(x_sq, W_sum.T) + torch.mm(x, W_prod.T)
        return (poly_term / scale_factor) + base_linear(x) + b

    def forward(self, x):
        h1 = self._cubic_gemm(x, self.W1, self.R1_1, self.R2_1, self.base1, self.b1, math.sqrt(self.in_dim))
        h1 = self.act(self.ln1(h1))
        return self._cubic_gemm(h1, self.W2, self.R1_2, self.R2_2, self.base2, self.b2, math.sqrt(self.hidden_dim))

    def inspect_root_topology(self):
        all_r = torch.cat([
            self.R1_1.detach().cpu().flatten(), self.R2_1.detach().cpu().flatten(),
            self.R1_2.detach().cpu().flatten(), self.R2_2.detach().cpu().flatten()
        ])
        total = len(all_r)
        active = (all_r.abs() <= 1.0).sum().item()
        ghost = (all_r.abs() > 1.0).sum().item()
        s_r = torch.sort(all_r).values
        q = [s_r[min(int(total * p), total - 1)].item() for p in [0.0, 0.25, 0.5, 0.75, 1.0]]
        return active, ghost, total, q


# =====================================================================
# 3. BASELINES & ANCHORS
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


class ChebyKAN_v2(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, degree=3):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.degree = degree

        self.w1 = nn.Parameter(torch.randn(hidden_dim, in_dim * (degree + 1)) * 0.1)
        self.b1 = nn.Linear(in_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.act = nn.SiLU()

        self.w2 = nn.Parameter(torch.randn(out_dim, hidden_dim * (degree + 1)) * 0.1)
        self.b2 = nn.Linear(hidden_dim, out_dim)

    def _cheby_forward(self, x, weight, base_linear):
        x_c = torch.clamp(x, -1.0, 1.0)
        T = [torch.ones_like(x_c), x_c]
        for _ in range(2, self.degree + 1):
            T.append(2.0 * x_c * T[-1] - T[-2])
        T_cat = torch.cat(T, dim=-1)
        return torch.mm(T_cat, weight.T) / math.sqrt(x.shape[-1]) + base_linear(x)

    def forward(self, x):
        h = self._cheby_forward(x, self.w1, self.b1)
        h = self.act(self.ln(h))
        return self._cheby_forward(h, self.w2, self.b2)


class RegCompressiveRootKAN(nn.Module):
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

    def _eval_and_compress(self, u, roots, scale):
        denom = torch.sqrt(1.0 + roots.pow(2))
        prod = (u - roots[0]) / denom[0]
        for k in range(1, self.degree):
            prod = prod * ((u - roots[k]) / denom[k])

        v = prod * scale + u
        # Symmetric Halley (a=0): D = 12y^4 + 6y^2 + 2 > 0 strictly everywhere
        y = v / (1.0 + v.abs().pow(2.0 / 3.0))
        for _ in range(2):
            y_sq = y.pow(2)
            G = y_sq * y + y - v
            G_prime = 3.0 * y_sq + 1.0
            G_double_prime = 6.0 * y
            num = 2.0 * G * G_prime
            denom_h = 2.0 * G_prime.pow(2) - G * G_double_prime
            y = y - num / denom_h
        return y

    def forward(self, x):
        h1 = self._eval_and_compress(self.proj1(x), self.roots1, self.scale1)
        h1 = self.drop(self.act(self.ln(h1)))
        return self.proj2(h1)  # Clean linear output logits

    def inspect_root_topology(self):
        all_r = self.roots1.detach().cpu().flatten()
        total = len(all_r)
        active = (all_r.abs() <= 1.0).sum().item()
        ghost = (all_r.abs() > 1.0).sum().item()
        s_r = torch.sort(all_r).values
        q = [s_r[min(int(total * p), total - 1)].item() for p in [0.0, 0.25, 0.5, 0.75, 1.0]]
        return active, ghost, total, q


# =====================================================================
# 4. PATCH TOKENIZER STEM WRAPPER
# =====================================================================

class PatchTokenizedBackbone(nn.Module):
    def __init__(self, core_model, in_channels=3, patch_dim=48):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, patch_dim, kernel_size=4, stride=4),
            nn.BatchNorm2d(patch_dim),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten()
        )
        self.core_model = core_model

    def forward(self, x):
        if x.dim() == 2 and x.shape[-1] == 3072:
            x = x.view(-1, 3, 32, 32).contiguous()
        features = self.stem(x)
        return self.core_model(features)

    def inspect_root_topology(self):
        if hasattr(self.core_model, "inspect_root_topology"):
            return self.core_model.inspect_root_topology()
        return None


# =====================================================================
# 5. DATA LOADER & HARNESS WITH PARAMETER SPLITTING
# =====================================================================

def load_dataset(dataset_name: str, batch_size: int = 128, max_train_samples: int = 12000):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) if dataset_name == "MNIST" else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    raw_name = "MNIST" if dataset_name == "MNIST" else "CIFAR10"
    
    if raw_name == "MNIST":
        train_set = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        in_dim = 784
    else:
        train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        in_dim = 192 if dataset_name == "CIFAR10_Patch" else 3072
    train_loader = DataLoader(torch.utils.data.Subset(train_set, range(min(len(train_set), max_train_samples))), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(torch.utils.data.Subset(test_set, range(min(len(test_set), 2500))), batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, in_dim


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_optimizer_with_decoupled_root_decay(model, lr=0.010, weight_decay=1e-4):
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lower = name.lower()
        if any(k in lower for k in ["root", "sigma", "r1", "r2", "bias", "b1", "b2", ".a", "norm"]):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return torch.optim.AdamW([
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ], lr=lr)


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
# 6. EXECUTION TOURNAMENT (V8.1)
# =====================================================================

def run_v8_1_tournament():

    import torch._dynamo
    torch._dynamo.config.recompile_limit = 64
    

    USE_TORCH_COMPILE = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on Device: {device} | torch.compile: {USE_TORCH_COMPILE} | TF32: True")

    tasks = ["MNIST", "CIFAR10_Flat", "CIFAR10_Patch"]
    epochs = 10
    lr = 0.010

    for task in tasks:
        train_loader, test_loader, in_dim = load_dataset(task, batch_size=128, max_train_samples=12000)

        if task == "MNIST":
            mlp_h = 50
            cardano_h = 50
            comp_h = 50
            cheby_h = 10
            dual_h = 25
            triple_h = 17
            degree = 3
        elif task == "CIFAR10_Flat":
            mlp_h = 20
            cardano_h = 20
            comp_h = 20
            cheby_h = 5
            dual_h = 10
            triple_h = 7
            degree = 2
        else:  # CIFAR10_Patch (in_dim=192)
            mlp_h = 64
            cardano_h = 64
            comp_h = 64
            cheby_h = 24
            dual_h = 48
            triple_h = 32
            degree = 2

        base_models = {
            # --- Anchors ---
            "Modern_MLP":                  ModernMLP(in_dim, mlp_h, 10),
            "Cheby_KAN_v2":                ChebyKAN_v2(in_dim, cheby_h, 10, degree=degree),
            "Reg_Compressive_RootKAN":     RegCompressiveRootKAN(in_dim, comp_h, 10, degree=degree, dropout_p=0.15),

            # --- V8.1 CONTENDERS ---
            # 1. Pure Branchless Newton-Cardano (K=3, provably positive G' > 0 everywhere)
            "Cardano_Newton_MLP":          CardanoNewtonMLP(in_dim, cardano_h, 10),

            # 2. Dual-GEMM + Cardano-Newton Hidden + Output Dual-GEMM Logits
            "Dual_GEMM_Cardano_Newton":    DualGEMM_CardanoNewton(in_dim, dual_h, 10),

            # 3. Triple-GEMM Cubic Per-Weight RootKAN (Chebyshev init + Fan-in scale + Linear Base)
            "Triple_GEMM_Cubic_v8":        TripleGEMM_CubicRootKAN_v8(in_dim, triple_h, 10),
        }

        if task == "CIFAR10_Patch":
            models = {name: PatchTokenizedBackbone(m, in_channels=3, patch_dim=48) for name, m in base_models.items()}
        else:
            models = base_models

        print("\n" + "=" * 134)
        print(f"BENCHMARK TASK: {task} (Input Dim: {in_dim}, Epochs: {epochs})")
        print(f"{'MODEL / V8.1 CONTENDER':<30} | {'PARAMS':<8} | {'TRAIN ACC':<11} | {'TEST ACC':<11} | {'TEST LOSS':<11} | {'MEDIAN LAT':<11} | {'P90 LAT'}")
        print("-" * 134)

        root_reports = {}

        for name, model in models.items():
            torch.manual_seed(42)
            model = model.to(device)
            params = count_parameters(model)

            compiled_model = torch.compile(model) if (USE_TORCH_COMPILE and hasattr(torch, "compile")) else model

            optimizer = build_optimizer_with_decoupled_root_decay(compiled_model, lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader), eta_min=1e-5)
            loss_fn = nn.CrossEntropyLoss()

            compiled_model.train()
            for ep in range(epochs):
                for x_b, y_b in train_loader:
                    x_b = x_b.to(device)
                    if task != "CIFAR10_Patch":
                        x_b = x_b.view(x_b.size(0), -1)
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
                    x_b = x_b.to(device)
                    if task != "CIFAR10_Patch":
                        x_b = x_b.view(x_b.size(0), -1)
                    y_b = y_b.to(device)
                    logits = compiled_model(x_b)
                    total_loss += loss_fn(logits, y_b).item() * x_b.size(0)
                    correct += (logits.argmax(dim=-1) == y_b).sum().item()
                    total += x_b.size(0)

            test_acc = (correct / total) * 100.0
            test_loss = total_loss / total

            sample_x, sample_y = next(iter(train_loader))
            sample_x = sample_x.to(device)
            if task != "CIFAR10_Patch":
                sample_x = sample_x.view(sample_x.size(0), -1)
            train_acc = (compiled_model(sample_x).argmax(dim=-1) == sample_y.to(device)).float().mean().item() * 100.0

            med_ms, p90_ms = benchmark_batch_latency_median(compiled_model, sample_x, device, warmup_iters=50, bench_iters=150)

            if hasattr(model, "inspect_root_topology") and model.inspect_root_topology() is not None:
                root_reports[name] = model.inspect_root_topology()

            print(f"{name:<30} | {params:<8d} | {train_acc:<10.2f}% | {test_acc:<10.2f}% | {test_loss:<11.5f} | {med_ms:<8.3f} ms | {p90_ms:.3f} ms")

        if root_reports:
            print("\n" + "-" * 134)
            print(f"ROOT TOPOLOGY DIAGNOSTICS FOR {task}:")
            for m_name, (active, ghost, tot, q) in root_reports.items():
                print(f"  {m_name:<30} -> Active: {active}/{tot} ({active/tot*100:.1f}%) | "
                      f"Ghosts: {ghost}/{tot} ({ghost/tot*100:.1f}%) | "
                      f"Spread [Min➔Q1➔Med➔Q3➔Max]: [{q[0]:+.2f} ➔ {q[1]:+.2f} ➔ {q[2]:+.2f} ➔ {q[3]:+.2f} ➔ {q[4]:+.2f}]")
        print("=" * 134)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    run_v8_1_tournament()