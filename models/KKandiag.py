# run_v9_patch_breakthrough.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import time

# IMPORT MODULAR ARCHITECTURES
from models.KKanvs10 import (
    ModularRootKAN_Deep,
    ModernMLP_Deep
)


# =====================================================================
# 1. 4x4 PATCH TOKENIZER STEM WRAPPER
# =====================================================================

class PatchTokenizedBackbone(nn.Module):
    """
    Turns 3x32x32 raw images into 192-D spatial patch tokens:
    Conv2d(stride=4, k=4) -> BatchNorm -> SiLU -> AdaptiveAvgPool(2,2) -> Flatten(192)
    """
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
# 2. DATA LOADER & BENCHMARK HARNESS
# =====================================================================

def get_cifar10_patch_loaders(batch_size=128, max_train=12000, max_test=2500):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_sub = torch.utils.data.Subset(train_set, range(min(len(train_set), max_train)))
    test_sub = torch.utils.data.Subset(test_set, range(min(len(test_set), max_test)))

    return (
        DataLoader(train_sub, batch_size=batch_size, shuffle=True),
        DataLoader(test_sub, batch_size=batch_size, shuffle=False)
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_and_eval(model, train_loader, test_loader, epochs=10, lr=0.010, device="cuda"):
    model = model.to(device)
    compiled_model = torch.compile(model)

    # Decouple weight decay from roots and norms
    decay_params, no_decay_params = [], []
    for name, p in compiled_model.named_parameters():
        if not p.requires_grad:
            continue
        lower = name.lower()
        if any(k in lower for k in ["root", "sigma", "a", "bias", "norm"]):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": 1e-4},
        {"params": no_decay_params, "weight_decay": 0.0}
    ], lr=lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(train_loader), eta_min=1e-5
    )
    loss_fn = nn.CrossEntropyLoss()

    compiled_model.train()
    for _ in range(epochs):
        for x_b, y_b in train_loader:
            x_b = x_b.to(device)
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
            y_b = y_b.to(device)
            logits = compiled_model(x_b)
            total_loss += loss_fn(logits, y_b).item() * x_b.size(0)
            correct += (logits.argmax(dim=-1) == y_b).sum().item()
            total += x_b.size(0)

    test_acc = (correct / total) * 100.0
    test_loss = total_loss / total

    # Latency Benchmark
    sample_x = next(iter(train_loader))[0].to(device)
    with torch.no_grad():
        for _ in range(50):
            _ = compiled_model(sample_x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        latencies = []
        for _ in range(150):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = compiled_model(sample_x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)

    lat_tensor = torch.tensor(latencies)
    return test_acc, test_loss, torch.median(lat_tensor).item(), torch.quantile(lat_tensor, 0.90).item()


# =====================================================================
# 3. RUNNER
# =====================================================================

def run_patch_tournament():
    import torch._dynamo
    torch._dynamo.config.recompile_limit = 64
    torch.set_float32_matmul_precision('high')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_cifar10_patch_loaders()
    in_dim, hidden_dim, out_dim = 192, 64, 10
    epochs = 10
    depth = 3

    print("=" * 130)
    print(f"BREAKTHROUGH EXPERIMENT: CIFAR10_PATCH (Token Dim: {in_dim}, Hidden: {hidden_dim}, Depth: {depth})")
    print("=" * 130)
    print(f"{'MODEL CONFIGURATION':<46} | {'PARAMS':<8} | {'TEST ACC':<10} | {'TEST LOSS':<10} | {'MEDIAN LAT':<10} | {'P90 LAT'}")
    print("-" * 130)

    models_to_test = {
        # Baseline Reference
        "Modern_MLP_Deep [Baseline]": ModernMLP_Deep(in_dim, hidden_dim, out_dim, num_layers=depth),

        # The Dimensionally Matched Champion
        "RootKAN_Deep [deg=3, comp=2, drop=0.00]": ModularRootKAN_Deep(
            in_dim, hidden_dim, out_dim, num_layers=depth, degree=3, comp_steps=2, dropout_p=0.00
        ),

        # Mild Regularization Variant
        "RootKAN_Deep [deg=3, comp=2, drop=0.05]": ModularRootKAN_Deep(
            in_dim, hidden_dim, out_dim, num_layers=depth, degree=3, comp_steps=2, dropout_p=0.05
        ),

        # The Quadratic Speed Champion
        "RootKAN_Deep [deg=2, comp=0 (No Solve), drop=0.00]": ModularRootKAN_Deep(
            in_dim, hidden_dim, out_dim, num_layers=depth, degree=2, comp_steps=0, dropout_p=0.00
        ),
    }

    root_reports = {}

    for name, core_model in models_to_test.items():
        torch.manual_seed(42)
        wrapper = PatchTokenizedBackbone(core_model, in_channels=3, patch_dim=48)
        params = count_parameters(wrapper)

        acc, loss, med_lat, p90_lat = train_and_eval(
            wrapper, train_loader, test_loader, epochs=epochs, lr=0.010, device=device
        )

        print(f"{name:<46} | {params:<8d} | {acc:<9.2f}% | {loss:<10.4f} | {med_lat:<7.3f} ms | {p90_lat:.3f} ms")

        if wrapper.inspect_root_topology() is not None:
            root_reports[name] = wrapper.inspect_root_topology()

    if root_reports:
        print("\n" + "-" * 130)
        print("ROOT TOPOLOGY DIAGNOSTICS FOR PATCH CHAMPIONS:")
        for m_name, (active, ghost, tot, q) in root_reports.items():
            print(f"  {m_name:<46} -> Active: {active}/{tot} ({active/tot*100:.1f}%) | "
                  f"Ghosts: {ghost}/{tot} ({ghost/tot*100:.1f}%) | "
                  f"Spread [Min➔Q1➔Med➔Q3➔Max]: [{q[0]:+.2f} ➔ {q[1]:+.2f} ➔ {q[2]:+.2f} ➔ {q[3]:+.2f} ➔ {q[4]:+.2f}]")
    print("=" * 130)


if __name__ == "__main__":
    run_patch_tournament()