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
# 1. MODULAR V3 ARCHITECTURES (ABLATION-READY)
# =====================================================================

class RootKANLayer_v3(nn.Module):
    """
    Modular Root-Factored Layer for controlled ablation testing:
    - use_factor_norm: divides by sqrt(1 + r^2)
    - use_cheb_init:   initializes at Chebyshev nodes vs uniform linspace
    - use_base_linear: includes parallel linear skip connection
    """
    def __init__(self, in_features, out_features, degree=3,
                 use_factor_norm=True, use_cheb_init=True, use_base_linear=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.use_factor_norm = use_factor_norm
        self.use_base_linear = use_base_linear

        # 1. Root Initialization Ablation
        if use_cheb_init:
            k = torch.arange(1, degree + 1, dtype=torch.float32)
            nodes = torch.cos((2 * k - 1) / (2 * degree) * math.pi)
        else:
            nodes = torch.linspace(-0.6, 0.6, degree)

        self.roots = nn.Parameter(nodes.view(-1, 1, 1).repeat(1, out_features, in_features))
        self.scale = nn.Parameter(torch.randn(out_features, in_features) * 0.1)

        # 2. Linear Bypass Ablation
        if use_base_linear:
            self.base_linear = nn.Linear(in_features, out_features)
        else:
            self.base_linear = None
            self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        x_in = x.unsqueeze(1)  # (batch, 1, in)

        # 3. Factor Normalization Ablation
        if self.use_factor_norm:
            denoms = torch.sqrt(1.0 + self.roots.pow(2))
            prod = (x_in - self.roots[0]) / denoms[0]
            for k in range(1, self.degree):
                prod = prod * ((x_in - self.roots[k]) / denoms[k])
        else:
            # Raw polynomial factors (unbounded)
            prod = (x_in - self.roots[0])
            for k in range(1, self.degree):
                prod = prod * (x_in - self.roots[k])

        # Kaiming fan-in scaling: preserves O(1) variance
        kan_out = torch.sum(prod * self.scale, dim=-1) / math.sqrt(self.in_features)

        if self.use_base_linear:
            return kan_out + self.base_linear(x)
        else:
            return kan_out + self.bias

    def repulsion_loss(self, eps=1e-4):
        d = self.degree
        if d <= 1:
            return torch.tensor(0.0, device=self.roots.device)
        r_diff = self.roots.unsqueeze(1) - self.roots.unsqueeze(0)
        dist_sq = r_diff.pow(2) + eps
        eye = torch.eye(d, device=self.roots.device).view(d, d, 1, 1)
        dist_sq = dist_sq + eye * 1e6
        return (1.0 / dist_sq).sum() / (self.out_features * self.in_features * d * (d - 1))


class RootFactoredKAN_v3(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, degree=3,
                 use_factor_norm=True, use_cheb_init=True, use_base_linear=True):
        super().__init__()
        self.layer1 = RootKANLayer_v3(in_dim, hidden_dim, degree=degree,
                                     use_factor_norm=use_factor_norm,
                                     use_cheb_init=use_cheb_init,
                                     use_base_linear=use_base_linear)
        self.ln = nn.LayerNorm(hidden_dim)
        self.act = nn.SiLU()
        self.layer2 = RootKANLayer_v3(hidden_dim, out_dim, degree=degree,
                                     use_factor_norm=use_factor_norm,
                                     use_cheb_init=use_cheb_init,
                                     use_base_linear=use_base_linear)

    def forward(self, x):
        h = self.layer1(x)
        h = self.act(self.ln(h))
        return self.layer2(h)

    def repulsion_loss(self):
        return self.layer1.repulsion_loss() + self.layer2.repulsion_loss()

    def inspect_root_topology(self):
        """Extracts active vs. ghost roots and quartile spread across all layers"""
        r1 = self.layer1.roots.detach().cpu().flatten()
        r2 = self.layer2.roots.detach().cpu().flatten()
        all_r = torch.cat([r1, r2])
        total = len(all_r)
        active = (all_r.abs() <= 1.0).sum().item()
        ghost  = (all_r.abs() >  1.0).sum().item()

        sorted_r = torch.sort(all_r).values
        q = [sorted_r[min(int(total * p), total - 1)].item() for p in [0.0, 0.25, 0.5, 0.75, 1.0]]
        return active, ghost, total, q


# =====================================================================
# 2. IMPLICIT NEWTON KAN V3 (FORWARD PASS IS 0-FINDING)
# =====================================================================

class ImplicitNewtonKAN_v3(nn.Module):
    """
    Modernized Implicit Newton KAN:
    Forward pass solves G(y) = y^3 + A(x)*y - B(x) = 0 via 4 unrolled Newton steps.
    Scaled with LayerNorm and Kaiming initialization.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, steps=4):
        super().__init__()
        self.steps = steps
        self.w0_1 = nn.Linear(in_dim, hidden_dim)
        self.w1_1 = nn.Linear(in_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act = nn.SiLU()

        self.w0_2 = nn.Linear(hidden_dim, out_dim)
        self.w1_2 = nn.Linear(hidden_dim, out_dim)

    def _solve_layer(self, x, w0, w1):
        # A > 0 strictly guarantees a single real root
        A = torch.relu(w1(x)) + 0.1
        B = w0(x)
        y = torch.zeros_like(B)

        # 4 static unrolled Newton-Raphson steps
        for _ in range(self.steps):
            G = y.pow(3) + A * y - B
            G_prime = 3.0 * y.pow(2) + A + 1e-6
            y = y - (G / G_prime)
        return y

    def forward(self, x):
        h = self._solve_layer(x, self.w0_1, self.w1_1)
        h = self.act(self.ln1(h))
        return self._solve_layer(h, self.w0_2, self.w1_2)


# =====================================================================
# 3. BASELINES & LEGACY RETENTION
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
        out_poly = torch.mm(T_cat, self.weight.T) / math.sqrt(self.in_features)
        return out_poly + self.base_linear(x)


class ChebyKAN_v2(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, degree=3):
        super().__init__()
        self.layer1 = ChebyKANLayer_v2(in_dim, hidden_dim, degree=degree)
        self.ln = nn.LayerNorm(hidden_dim)
        self.act = nn.SiLU()
        self.layer2 = ChebyKANLayer_v2(hidden_dim, out_dim, degree=degree)

    def forward(self, x):
        h = self.layer1(x)
        h = self.act(self.ln(h))
        return self.layer2(h)


# =====================================================================
# 4. DATA LOADER & BENCHMARK HARNESS
# =====================================================================

def load_dataset(dataset_name: str, batch_size: int = 128, max_train_samples: int = 12000):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) if dataset_name == "MNIST" else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    try:
        if dataset_name == "MNIST":
            train_set = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
            test_set  = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
            in_dim = 784
        elif dataset_name == "CIFAR10":
            train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
            test_set  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
            in_dim = 3072
        else:
            raise ValueError(dataset_name)

        train_loader = DataLoader(
            torch.utils.data.Subset(train_set, range(min(len(train_set), max_train_samples))),
            batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            torch.utils.data.Subset(test_set, range(min(len(test_set), 2500))),
            batch_size=batch_size, shuffle=False
        )
    except Exception as e:
        print(f"Warning: Falling back to synthetic {dataset_name} ({e})")
        in_dim = 784 if dataset_name == "MNIST" else 3072
        x_tr = torch.randn(max_train_samples, in_dim).clamp(-1.0, 1.0)
        y_tr = torch.randint(0, 10, (max_train_samples,))
        x_te = torch.randn(2500, in_dim).clamp(-1.0, 1.0)
        y_te = torch.randint(0, 10, (2500,))
        train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True)
        test_loader  = DataLoader(TensorDataset(x_te, y_te), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, in_dim


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_batch_latency(model, sample_batch, device, iters=100):
    model.eval()
    for _ in range(15):
        _ = model(sample_batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = model(sample_batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return (time.perf_counter() - t0) * 1000.0 / iters


def run_ablation_tournament():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on Device: {device}")

    datasets = ["MNIST", "CIFAR10"]
    epochs = 10
    lr = 0.010

    for dname in datasets:
        train_loader, test_loader, in_dim = load_dataset(dname, batch_size=128, max_train_samples=12000)

        # Matched parameter budgets:
        # MNIST: ~39.8k params | CIFAR10: ~61.7k params
        if dname == "MNIST":
            kan_hidden = 10
            mlp_hidden = 50
            newton_hidden = 25  # 2 * 784 * 25 + ... ~ 39.7k params
            degree = 3
        else:  # CIFAR10
            kan_hidden = 5
            mlp_hidden = 20
            newton_hidden = 10  # 2 * 3072 * 10 + ... ~ 61.6k params
            degree = 2

        # =============================================================
        # ACTIVE CONTENDERS: ABLATIONS & V3 ZERO-FINDING
        # =============================================================
        models = {
            # --- Anchors ---
            "Modern_MLP":             (ModernMLP(in_dim, mlp_hidden, 10), 0.0),
            "Cheby_KAN_v2":           (ChebyKAN_v2(in_dim, kan_hidden, 10, degree=degree), 0.0),

            # --- RootKAN Ablations ---
            "RootKAN_v3_Full":        (RootFactoredKAN_v3(in_dim, kan_hidden, 10, degree=degree,
                                                          use_factor_norm=True, use_cheb_init=True, use_base_linear=True), 1e-4),
            "RootKAN_v3_NoNorm":      (RootFactoredKAN_v3(in_dim, kan_hidden, 10, degree=degree,
                                                          use_factor_norm=False, use_cheb_init=True, use_base_linear=True), 1e-4),
            "RootKAN_v3_NoLinear":    (RootFactoredKAN_v3(in_dim, kan_hidden, 10, degree=degree,
                                                          use_factor_norm=True, use_cheb_init=True, use_base_linear=False), 1e-4),
            "RootKAN_v3_UniformInit": (RootFactoredKAN_v3(in_dim, kan_hidden, 10, degree=degree,
                                                          use_factor_norm=True, use_cheb_init=False, use_base_linear=True), 1e-4),
            "RootKAN_v3_NoRepel":     (RootFactoredKAN_v3(in_dim, kan_hidden, 10, degree=degree,
                                                          use_factor_norm=True, use_cheb_init=True, use_base_linear=True), 0.0),

            # --- Zero-Finding Forward Pass Model ---
            "Implicit_Newton_v3":     (ImplicitNewtonKAN_v3(in_dim, newton_hidden, 10, steps=4), 0.0),

            # =========================================================
            # OLD / SUPERSEDED VERSIONS (UNCOMMENT TO TEST LEGACY V1/V2)
            # =========================================================
            # "Root_Factored_KAN_v2": (RootFactoredKAN_v2(in_dim, kan_hidden, 10, degree=degree), 1e-4),
            # "Cheby_KAN_v1":         (ChebyKAN(in_dim, kan_hidden, 10, degree=degree), 0.0),
            # "Root_Factored_KAN_v1": (RootFactoredKAN(in_dim, kan_hidden, 10, degree=degree), 0.0),
        }

        print("\n" + "=" * 122)
        print(f"DATASET: {dname} (Input Dim: {in_dim}, Epochs: {epochs})")
        print(f"{'MODEL / ABLATION':<24} | {'PARAMS':<8} | {'TRAIN ACC':<11} | {'TEST ACC':<11} | {'TEST LOSS':<11} | {'BATCH LATENCY'}")
        print("-" * 122)

        root_reports = {}

        for name, (model, repel_lambda) in models.items():
            torch.manual_seed(42)
            model = torch.compile(model.to(device))
            params = count_parameters(model)

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader), eta_min=1e-5)
            loss_fn = nn.CrossEntropyLoss()

            # Train Loop
            model.train()
            for ep in range(epochs):
                for x_b, y_b in train_loader:
                    x_b = x_b.view(x_b.size(0), -1).to(device)
                    y_b = y_b.to(device)

                    optimizer.zero_grad()
                    out = model(x_b)
                    loss = loss_fn(out, y_b)
                    if repel_lambda > 0.0 and hasattr(model, "repulsion_loss"):
                        loss = loss + repel_lambda * model.repulsion_loss()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

            # Test Loop
            model.eval()
            correct, total, total_loss = 0, 0, 0.0
            with torch.no_grad():
                for x_b, y_b in test_loader:
                    x_b = x_b.view(x_b.size(0), -1).to(device)
                    y_b = y_b.to(device)
                    logits = model(x_b)
                    loss = loss_fn(logits, y_b)

                    total_loss += loss.item() * x_b.size(0)
                    correct += (logits.argmax(dim=-1) == y_b).sum().item()
                    total += x_b.size(0)

            test_acc = (correct / total) * 100.0
            test_loss = total_loss / total

            sample_x, sample_y = next(iter(train_loader))
            sample_x = sample_x.view(sample_x.size(0), -1).to(device)
            train_acc = (model(sample_x).argmax(dim=-1) == sample_y.to(device)).float().mean().item() * 100.0

            latency_ms = benchmark_batch_latency(model, sample_x, device)

            # Collect root topology if model supports it
            if hasattr(model, "inspect_root_topology"):
                root_reports[name] = model.inspect_root_topology()

            print(f"{name:<24} | {params:<8d} | {train_acc:<10.2f}% | {test_acc:<10.2f}% | {test_loss:<11.5f} | {latency_ms:.3f} ms/batch")

        # =============================================================
        # ROOT TOPOLOGY INSPECTION REPORT
        # =============================================================
        if root_reports:
            print("\n" + "-" * 122)
            print(f"ROOT TOPOLOGY DIAGNOSTICS FOR {dname}:")
            for m_name, (active, ghost, tot, q) in root_reports.items():
                print(f"  {m_name:<24} -> Active: {active}/{tot} ({active/tot*100:.1f}%) | "
                      f"Ghosts: {ghost}/{tot} ({ghost/tot*100:.1f}%) | "
                      f"Spread [Min➔Q1➔Med➔Q3➔Max]: [{q[0]:+.2f} ➔ {q[1]:+.2f} ➔ {q[2]:+.2f} ➔ {q[3]:+.2f} ➔ {q[4]:+.2f}]")
        print("=" * 122)


if __name__ == "__main__":
    run_ablation_tournament()