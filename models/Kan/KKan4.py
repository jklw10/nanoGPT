import torch
import torch.nn as nn
import math

from models.Kan.KKan2 import (
    TaylorCoeffKAN, RootFactoredKAN, FixedKreinOptimizer, get_data
)

def run_degree_stress_test():
    """
    TEST 1: DEGREE OVERPARAMETERIZATION STRESS TEST
    Hypothesis: As degree d scales from 2 to 6 on MultiScale_Poly:
      - Taylor_Coeff will degrade in conditioning and Runge ratio.
      - Root_Factored will maintain stability by ejecting surplus roots into Ghost status.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y = get_data("MultiScale_Poly", n=512)
    x, y = x.to(device), y.to(device)

    degrees = [2, 4, 6]
    epochs = 120

    print("=" * 105)
    print("TEST 1: DEGREE SCALING & AUTOMATED PRUNING TEST (MultiScale_Poly)")
    print("=" * 105)
    print(f"{'DEGREE':<8} | {'MODEL':<16} | {'TRAIN LOSS':<12} | {'OOD LOSS (1.5x)':<16} | {'GHOST ROOTS (%)':<16} | {'STATUS'}")
    print("-" * 105)

    for d in degrees:
        for m_name in ["Taylor_Coeff", "Root_Factored"]:
            torch.manual_seed(42)
            if m_name == "Taylor_Coeff":
                model = TaylorCoeffKAN(degree=d).to(device)
            else:
                model = RootFactoredKAN(degree=d).to(device)

            opt = FixedKreinOptimizer(model.parameters(), lr=0.02, use_adam=True, use_kron=True, 
                                      use_krein_space=True, damping=1e-3, gamma=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)
            loss_fn = nn.MSELoss()

            for _ in range(epochs):
                opt.zero_grad()
                out = model(x)
                loss = loss_fn(out, y)
                loss.backward()
                opt.step()
                scheduler.step()

            train_loss = loss.item()

            # OOD evaluation [-1.5, 1.5]
            x_ood = torch.rand(512, 2, device=device) * 3.0 - 1.5
            y_ood = 3.5 * (x_ood[:, 0:1]**4) - 1.2 * (x_ood[:, 0:1]**2) + 0.5 * torch.sin(math.pi * x_ood[:, 1:2])
            with torch.no_grad():
                ood_loss = loss_fn(model(x_ood), y_ood).item()

            if m_name == "Root_Factored":
                all_r = torch.cat([model.roots1.flatten(), model.roots2.flatten()]).detach()
                ghost_cnt = (all_r.abs() > 1.0).sum().item()
                ghost_pct = f"{ghost_cnt}/{len(all_r)} ({ghost_cnt/len(all_r)*100:.1f}%)"
                status = "Auto-Deflated" if ghost_cnt > 0 else "All Active"
            else:
                ghost_pct = "N/A"
                status = "Wiggling" if ood_loss > 50.0 else "Stable"

            print(f"d = {d:<4} | {m_name:<16} | {train_loss:<12.5f} | {ood_loss:<16.4f} | {ghost_pct:<16} | {status}")
        print("-" * 105)


def run_root_sensitivity_ablation():
    """
    TEST 2: FUNCTIONAL SENSITIVITY OF ACTIVE VS GHOST ROOTS
    Hypothesis: The loss gradient magnitude |dL/dr| for Ghost Roots (|r| > 1) 
    will be significantly lower than for Active Roots (|r| <= 1), proving 
    ghost roots have decoupled from the active non-linear dynamics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y = get_data("MultiScale_Poly", n=512)
    x, y = x.to(device), y.to(device)

    torch.manual_seed(42)
    model = RootFactoredKAN(degree=4).to(device)
    opt = FixedKreinOptimizer(model.parameters(), lr=0.02, use_adam=True, use_kron=True, 
                              use_krein_space=True, damping=1e-3, gamma=1e-4)
    loss_fn = nn.MSELoss()

    # Train to convergence
    for _ in range(120):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()

    # Compute exact parameter gradients at convergence
    opt.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()

    roots = torch.cat([model.roots1.flatten(), model.roots2.flatten()])
    roots_grad = torch.cat([model.roots1.grad.flatten(), model.roots2.grad.flatten()]).abs()

    active_mask = roots.abs() <= 1.0
    ghost_mask = roots.abs() > 1.0

    active_grads = roots_grad[active_mask]
    ghost_grads = roots_grad[ghost_mask]

    print("\n" + "=" * 80)
    print("TEST 2: ROOT SENSITIVITY ABLATION (|dL/dr|)")
    print("=" * 80)
    print(f"Active Roots Count (|r| <= 1.0): {active_mask.sum().item()}")
    print(f"  Mean Gradient Sensitivity    : {active_grads.mean().item():.6f}")
    print(f"Ghost Roots Count  (|r| >  1.0): {ghost_mask.sum().item()}")
    print(f"  Mean Gradient Sensitivity    : {ghost_grads.mean().item():.6f}")
    
    ratio = active_grads.mean().item() / (ghost_grads.mean().item() + 1e-12)
    print(f"\nSENSITIVITY RATIO (Active / Ghost): {ratio:.2f}x")
    if ratio > 2.0:
        print("VERDICT: CONFIRMED. Active roots dominate functional fitting; ghost roots have functionally decoupled.")
    else:
        print("VERDICT: UNCONFIRMED. Ghost roots still exert strong functional pressure.")
    print("=" * 80)


if __name__ == "__main__":
    run_degree_stress_test()
    run_root_sensitivity_ablation()