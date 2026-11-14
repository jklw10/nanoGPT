import torch
import torch.nn as nn
import math

# Use the same model definitions from the previous benchmark
from models.Kan.KKan2 import (
    TaylorCoeffKAN, RootFactoredKAN, ImplicitNewtonKAN,
    FixedKreinOptimizer, get_data
)

def compute_gradient_condition_number(model, x, y):
    """
    Computes condition number kappa = s_max / s_min of the empirical 
    gradient covariance to measure coordinate-space distortion.
    """
    model.zero_grad()
    loss_fn = nn.MSELoss()
    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()

    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().flatten())
    g_vec = torch.cat(grads)
    
    # Empirical gradient outer-product spectrum
    G = g_vec.unsqueeze(1)
    s = torch.linalg.svdvals(G)
    s_max = s.max().item()
    s_mean = s.mean().item()
    return s_max, s_mean


def test_extrapolation_runge(model, task: str):
    """
    Tests model on Out-Of-Distribution (OOD) domain [-1.5, 1.5]
    to measure Runge's phenomenon / edge explosion.
    """
    device = next(model.parameters()).device
    # Generate OOD data outside [-1, 1]
    x_ood = torch.rand(512, 2, device=device) * 3.0 - 1.5  # [-1.5, 1.5]
    if task == "MultiScale_Poly":
        y_ood = 3.5 * (x_ood[:, 0:1]**4) - 1.2 * (x_ood[:, 0:1]**2) + 0.5 * torch.sin(math.pi * x_ood[:, 1:2])
    elif task == "Saddle_Valley":
        y_ood = (1.0 - x_ood[:, 0:1])**2 + 4.0 * (x_ood[:, 1:2] - x_ood[:, 0:1]**2)**2
    
    with torch.no_grad():
        pred = model(x_ood)
        ood_loss = nn.MSELoss()(pred, y_ood).item()
    return ood_loss


def run_deep_diagnostics():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks = ["MultiScale_Poly", "Saddle_Valley"]
    models = ["Taylor_Coeff", "Root_Factored", "Implicit_Newton"]
    epochs = 120

    print("=" * 115)
    print(f"{'TASK':<16} | {'MODEL':<16} | {'TRAIN LOSS':<10} | {'OOD LOSS (1.5x)':<15} | {'GRAD PEAK / MEAN':<18} | {'RUNGE RATIO'}")
    print("=" * 115)

    all_roots_summary = {}

    for task in tasks:
        x_train, y_train = get_data(task)
        x_train, y_train = x_train.to(device), y_train.to(device)

        for m_name in models:
            torch.manual_seed(42)
            if m_name == "Taylor_Coeff":
                model = TaylorCoeffKAN(degree=3).to(device)
            elif m_name == "Root_Factored":
                model = RootFactoredKAN(degree=3).to(device)
            elif m_name == "Implicit_Newton":
                model = ImplicitNewtonKAN(steps=4).to(device)

            # Use Krein_NGD as the baseline probe
            opt = FixedKreinOptimizer(model.parameters(), lr=0.02, use_adam=True, 
                                      use_kron=True, use_krein_space=True, damping=1e-3, gamma=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)
            loss_fn = nn.MSELoss()

            # Train
            for epoch in range(epochs):
                opt.zero_grad()
                out = model(x_train)
                loss = loss_fn(out, y_train)
                loss.backward()
                opt.step()
                scheduler.step()

            # 1. Measure Training and OOD Loss
            train_loss = loss.item()
            ood_loss = test_extrapolation_runge(model, task)
            runge_ratio = ood_loss / (train_loss + 1e-12)

            # 2. Gradient condition stats
            s_max, s_mean = compute_gradient_condition_number(model, x_train, y_train)
            grad_stat = f"{s_max:.2e} / {s_mean:.2e}"

            # 3. Collect Root distribution for Root_Factored
            if m_name == "Root_Factored":
                all_r1 = model.roots1.detach().cpu().flatten()
                all_r2 = model.roots2.detach().cpu().flatten()
                all_r = torch.cat([all_r1, all_r2])
                active = (all_r.abs() <= 1.0).sum().item()
                ghost = (all_r.abs() > 1.0).sum().item()
                all_roots_summary[task] = (all_r, active, ghost)

            print(f"{task:<16} | {m_name:<16} | {train_loss:<10.4f} | {ood_loss:<15.4f} | {grad_stat:<18} | {runge_ratio:<10.1f}x")
        print("-" * 115)

    # =============================================================
    # DEEP ROOT SPECTRUM ANALYSIS
    # =============================================================
    print("\n" + "=" * 80)
    print("INTERNAL ROOT TOPOLOGY ANALYSIS (Root_Factored)")
    print("=" * 80)
    for task, (roots, active, ghost) in all_roots_summary.items():
        total = active + ghost
        print(f"\nTask: {task}")
        print(f"  Total Learned Roots: {total}")
        print(f"  Active Roots  (Inside [-1, 1]): {active:2d} ({active/total*100:.1f}%) -> Actively carving local non-linearities")
        print(f"  Ghost Roots   (Outside [-1, 1]): {ghost:2d} ({ghost/total*100:.1f}%) -> Deflated degrees (Linearizing edges)")
        
        # Sort and show quartiles
        sorted_r = torch.sort(roots).values
        q = [sorted_r[min(int(len(sorted_r) * p), len(sorted_r) - 1)].item() for p in [0.0, 0.25, 0.5, 0.75, 1.0]]
        print(f"  Root Spread (Min ➔ Q1 ➔ Median ➔ Q3 ➔ Max):")
        print(f"    [{q[0]:+.2f} ➔ {q[1]:+.2f} ➔ {q[2]:+.2f} ➔ {q[3]:+.2f} ➔ {q[4]:+.2f}]")


if __name__ == "__main__":
    run_deep_diagnostics()