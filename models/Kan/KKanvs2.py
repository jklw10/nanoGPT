import torch
import torch.nn as nn
import math
import time

# =====================================================================
# 1. CONTENDER ARCHITECTURES
# =====================================================================

class RootKANLayer(nn.Module):
    def __init__(self, in_features, out_features, degree=6):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree

        # Chebyshev node initialization
        k = torch.arange(1, degree + 1, dtype=torch.float32)
        cheb_nodes = torch.cos((2 * k - 1) / (2 * degree) * math.pi)
        self.roots = nn.Parameter(cheb_nodes.view(-1, 1, 1).repeat(1, out_features, in_features))
        self.scale = nn.Parameter(torch.randn(out_features, in_features) * 0.2)
        self.base_linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        diff = x.unsqueeze(0).unsqueeze(2) - self.roots.unsqueeze(1)
        denom = torch.sqrt(1.0 + self.roots.unsqueeze(1).pow(2))
        norm_factors = diff / denom
        prod = torch.prod(norm_factors, dim=0)
        kan_out = torch.sum(prod * self.scale.unsqueeze(0), dim=-1)
        return kan_out + self.base_linear(x)


class RootFactoredKAN(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=8, out_dim=1, degree=6):
        super().__init__()
        self.layer1 = RootKANLayer(in_dim, hidden_dim, degree=degree)
        self.layer2 = RootKANLayer(hidden_dim, out_dim, degree=degree)

    def forward(self, x):
        return self.layer2(torch.tanh(self.layer1(x)))

    def repulsion_loss(self, eps=1e-4):
        """Electrostatic Coulomb repulsion to prevent root clustering"""
        loss = 0.0
        for layer in [self.layer1, self.layer2]:
            roots = layer.roots  # (degree, out, in)
            d, out_f, in_f = roots.shape
            r_diff = roots.unsqueeze(1) - roots.unsqueeze(0)  # (d, d, out, in)
            dist_sq = r_diff.pow(2) + eps
            eye = torch.eye(d, device=roots.device).view(d, d, 1, 1)
            dist_sq = dist_sq + eye * 1e6  # Ignore self-distance
            loss += (1.0 / dist_sq).sum() / (out_f * in_f * d * (d - 1))
        return loss


class ChebyKANLayer(nn.Module):
    def __init__(self, in_features, out_features, degree=6):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.weight = nn.Parameter(torch.randn(out_features, in_features * (degree + 1)) * 0.1)
        self.base_linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x_c = torch.tanh(x)
        T = [torch.ones_like(x_c), x_c]
        for _ in range(2, self.degree + 1):
            T.append(2.0 * x_c * T[-1] - T[-2])
        T_cat = torch.cat(T, dim=-1)
        return torch.mm(T_cat, self.weight.T) + self.base_linear(x)


class ChebyKAN(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=8, out_dim=1, degree=6):
        super().__init__()
        self.layer1 = ChebyKANLayer(in_dim, hidden_dim, degree=degree)
        self.layer2 = ChebyKANLayer(hidden_dim, out_dim, degree=degree)

    def forward(self, x):
        return self.layer2(torch.tanh(self.layer1(x)))


class FastKANLayer(nn.Module):
    def __init__(self, in_features, out_features, num_grids=7):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_grids = num_grids
        grid = torch.linspace(-1.0, 1.0, num_grids)
        self.register_buffer("grid", grid)
        self.h = 2.0 / (num_grids - 1)
        self.rbf_weights = nn.Parameter(torch.randn(out_features, in_features * num_grids) * 0.1)
        self.base_linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x_c = torch.tanh(x)
        diff = x_c.unsqueeze(-1) - self.grid.view(1, 1, -1)
        rbf = torch.exp(-(diff / self.h).pow(2)).view(x.shape[0], -1)
        return torch.mm(rbf, self.rbf_weights.T) + self.base_linear(x)


class FastKAN(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=8, out_dim=1, num_grids=7):
        super().__init__()
        self.layer1 = FastKANLayer(in_dim, hidden_dim, num_grids=num_grids)
        self.layer2 = FastKANLayer(hidden_dim, out_dim, num_grids=num_grids)

    def forward(self, x):
        return self.layer2(torch.tanh(self.layer1(x)))


class ModernMLP(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=12, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# =====================================================================
# 2. RUNGE-STIFF BENCHMARK DATASET
# =====================================================================
def get_runge_dataset(n_samples=1024):
    """
    2D Runge-Stiff Surface:
    Steep rational Lorentzian peak in the center + high-frequency ripple.
    Requires d >= 5 to fit inside [-1, 1], but violently whips outside [-1, 1].
    """
    torch.manual_seed(1337)
    x = torch.rand(n_samples, 2) * 2.0 - 1.0  # [-1, 1]
    r_sq = x[:, 0:1]**2 + x[:, 1:2]**2
    y = (1.0 / (1.0 + 20.0 * r_sq)) + 0.3 * torch.sin(2.5 * math.pi * x[:, 0:1]) * x[:, 1:2]

    # Split: 60% Train, 20% Val (for sweep), 20% ID Test
    train_end = int(0.60 * n_samples)
    val_end   = int(0.80 * n_samples)

    x_train, y_train = x[:train_end], y[:train_end]
    x_val,   y_val   = x[train_end:val_end], y[train_end:val_end]
    x_test,  y_test  = x[val_end:], y[val_end:]

    # Out-Of-Distribution (OOD) Test Set: [-1.5, 1.5]
    x_ood = torch.rand(512, 2) * 3.0 - 1.5
    r_sq_ood = x_ood[:, 0:1]**2 + x_ood[:, 1:2]**2
    y_ood = (1.0 / (1.0 + 20.0 * r_sq_ood)) + 0.3 * torch.sin(2.5 * math.pi * x_ood[:, 0:1]) * x_ood[:, 1:2]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test), (x_ood, y_ood)


# =====================================================================
# 3. AUTOMATED PARAMETER SWEEP & EVALUATION
# =====================================================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def build_model(model_name, device):
    if model_name in ["RootKAN_d6_Raw", "RootKAN_d6_Fixed"]:
        return RootFactoredKAN(in_dim=2, hidden_dim=8, out_dim=1, degree=6).to(device)
    elif model_name == "Cheby_KAN_d6":
        return ChebyKAN(in_dim=2, hidden_dim=8, out_dim=1, degree=6).to(device)
    elif model_name == "Fast_KAN":
        return FastKAN(in_dim=2, hidden_dim=8, out_dim=1, num_grids=7).to(device)
    elif model_name == "Modern_MLP":
        return ModernMLP(in_dim=2, hidden_dim=12, out_dim=1).to(device)
    raise ValueError(model_name)

def train_epochs(model, model_name, x_train, y_train, epochs, lr, lambda_repel):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(x_train)
        loss = loss_fn(pred, y_train)
        if model_name == "RootKAN_d6_Fixed":
            loss = loss + lambda_repel * model.repulsion_loss()
        loss.backward()
        opt.step()
        scheduler.step()

def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        return nn.MSELoss()(model(x), y).item()


def run_tournament():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (x_tr, y_tr), (x_val, y_val), (x_te, y_te), (x_ood, y_ood) = get_runge_dataset()
    x_tr, y_tr   = x_tr.to(device), y_tr.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)
    x_te, y_te   = x_te.to(device), y_te.to(device)
    x_ood, y_ood = x_ood.to(device), y_ood.to(device)

    contenders = [
        "RootKAN_d6_Raw",
        "RootKAN_d6_Fixed",
        "Cheby_KAN_d6",
        "Fast_KAN",
        "Modern_MLP"
    ]

    lr_candidates = [0.008, 0.015, 0.030]
    lambda_candidates = [1e-4, 5e-4]

    print("=" * 125)
    print("STEP 1: RUNNING AUTOMATED HYPERPARAMETER SWEEP (ON VALIDATION SET)")
    print("=" * 125)
    best_configs = {}

    for name in contenders:
        best_val = float('inf')
        best_cfg = {}
        
        # Grid search over candidate hyperparams
        for lr in lr_candidates:
            lambdas = lambda_candidates if name == "RootKAN_d6_Fixed" else [0.0]
            for l_repel in lambdas:
                torch.manual_seed(42)
                model = build_model(name, device)
                train_epochs(model, name, x_tr, y_tr, epochs=100, lr=lr, lambda_repel=l_repel)
                val_loss = evaluate(model, x_val, y_val)

                if val_loss < best_val:
                    best_val = val_loss
                    best_cfg = {"lr": lr, "lambda_repel": l_repel}

        best_configs[name] = best_cfg
        cfg_str = f"lr={best_cfg['lr']}" + (f", λ={best_cfg['lambda_repel']:.1e}" if name == "RootKAN_d6_Fixed" else "")
        print(f"  {name:<18} -> Best Val MSE: {best_val:.5f} | Optimal [{cfg_str}]")

    print("\n" + "=" * 125)
    print("STEP 2: FULL TOURNAMENT (250 EPOCHS ON OPTIMAL HYPERPARAMETERS)")
    print("=" * 125)
    print(f"{'MODEL':<18} | {'PARAMS':<7} | {'OPTIMAL CONFIG':<18} | {'TRAIN MSE':<11} | {'TEST ID MSE':<11} | {'TEST OOD MSE':<12} | {'RUNGE RATIO'}")
    print("-" * 125)

    for name in contenders:
        torch.manual_seed(42)
        model = build_model(name, device)
        params = count_parameters(model)
        cfg = best_configs[name]

        # Train full 250 epochs with optimal hyperparameters
        train_epochs(model, name, x_tr, y_tr, epochs=250, lr=cfg['lr'], lambda_repel=cfg['lambda_repel'])

        train_mse = evaluate(model, x_tr, y_tr)
        test_id   = evaluate(model, x_te, y_te)
        test_ood  = evaluate(model, x_ood, y_ood)
        runge_ratio = test_ood / (test_id + 1e-12)

        cfg_str = f"lr={cfg['lr']}" + (f", λ={cfg['lambda_repel']:.0e}" if name == "RootKAN_d6_Fixed" else "")
        print(f"{name:<18} | {params:<7d} | {cfg_str:<18} | {train_mse:<11.5f} | {test_id:<11.5f} | {test_ood:<12.5f} | {runge_ratio:<8.1f}x")
    print("=" * 125)


if __name__ == "__main__":
    run_tournament()