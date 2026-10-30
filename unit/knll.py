import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set seed for reproducibility
torch.manual_seed(42)
class EntropicKreinL4NLLLoss2(nn.Module):
    def __init__(self, S0: float = 10.0, eps: float = 1e-8):
        super().__init__()
        self.S0 = S0
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, log_var_raw: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        # Compute S0 from residual error variance (y_pred - y_true), OR use self.S0 floor
        error = y_pred - y_true
        residual_var = torch.var(error) + self.eps
        dynamic_S0 = 0.5 * torch.log(residual_var / self.eps)
        S0 = torch.clamp(dynamic_S0, min=self.S0) # Never let S0 fall below self.S0 (e.g. 10.0)

        log_var = log_var_raw / torch.sqrt(1.0 + (log_var_raw / S0) ** 2)
        var = torch.exp(log_var) + self.eps

        nll_loss = 0.5 * (error ** 2) / var + 0.5 * log_var
        l4_geometry_loss = error ** 4
        
        W_norms = torch.norm(W, dim=0, keepdim=True) + self.eps
        W_normed = W / W_norms
        gram_matrix = torch.matmul(W_normed.T, W_normed)
        off_diag_gram = gram_matrix - torch.eye(W.shape[1], device=W.device)
        gram_penalty = torch.mean(off_diag_gram ** 4)

        total_loss = nll_loss.mean() + l4_geometry_loss.mean() + gram_penalty
        return total_loss


class EntropicKreinL4NLLLoss(nn.Module):
    def __init__(self, S0_fallback: float = 10.0, eps: float = 1e-6):
        super().__init__()
        self.S0_fallback = S0_fallback
        self.eps = eps

    def forward(
        self, 
        y_pred: torch.Tensor, 
        y_true: torch.Tensor, 
        log_var_raw: torch.Tensor, 
        W: torch.Tensor
    ) -> torch.Tensor:
        error = y_pred - y_true  # [B, N]
        
        # 1. Softsign Constrained Uncertainty Scaling (S0)
        residual_var = torch.var(error.detach()) + self.eps
        dynamic_S0 = 0.5 * torch.log(residual_var / self.eps)
        S0 = torch.clamp(dynamic_S0, min=self.S0_fallback)

        log_var = log_var_raw / torch.sqrt(1.0 + (log_var_raw / S0) ** 2)
        var = torch.exp(log_var) + self.eps

        # 2. Pure L4 Geometry Loss for Representation (Drives W)
        l4_geometry_loss = (error ** 4).mean()

        # 3. Uncertainty NLL Loss (Drives head2)
        # Using error.detach() ensures uncalibrated variance NEVER explodes gradients on W,
        # preventing Feature Death whether head2 is attached or detached!
        nll_uncertainty_loss = (0.5 * (error.detach() ** 2) / var + 0.5 * log_var).mean()

        # 4. Normalized Gram Matrix Log-Barrier Penalty (Forces Pentagram Geometry)
        W_norms = torch.norm(W, dim=0, keepdim=True) + self.eps
        W_normed = W / W_norms
        gram_matrix = torch.matmul(W_normed.T, W_normed)
        
        N = W.shape[1]
        off_diag_mask = ~torch.eye(N, dtype=torch.bool, device=W.device)
        r = gram_matrix[off_diag_mask]
        
        # Barrier at r^2 -> 1 penalizes collinearity (both parallel and anti-parallel)
        r_sq = torch.clamp(r ** 2, max=1.0 - self.eps)
        gram_penalty = -torch.mean(torch.log(1.0 - r_sq + self.eps))

        total_loss = l4_geometry_loss + nll_uncertainty_loss + gram_penalty
        return total_loss

class GramL4NLLLoss(nn.Module):
    def __init__(self, S0_fallback: float = 10.0, eps: float = 1e-6):
        super().__init__()
        self.S0_fallback = S0_fallback
        self.eps = eps

    def forward(
        self, 
        y_pred: torch.Tensor, 
        y_true: torch.Tensor, 
        log_var_raw: torch.Tensor, 
        W: torch.Tensor
    ) -> torch.Tensor:
        error = y_pred - y_true  # [B, N]
        
        # --- Step 1: Dynamic Softsign Uncertainty Scaling (S0 Bound) ---
        residual_var = torch.var(error.detach()) + self.eps
        dynamic_S0 = 0.5 * torch.log(residual_var / self.eps)
        S0 = torch.clamp(dynamic_S0, min=self.S0_fallback)

        log_var = log_var_raw / torch.sqrt(1.0 + (log_var_raw / S0) ** 2)
        var = torch.exp(log_var) + self.eps

        # --- Step 2: Optimal L2 Precision Reconstruction Loss ---
        recon_loss = (error ** 2).mean()

        # --- Step 3: Decoupled Heteroscedastic NLL Loss (Drives head2) ---
        nll_uncertainty_loss = (0.5 * (error.detach() ** 2) / var + 0.5 * log_var).mean()

        # --- Step 4: Max-Based Log-Barrier Gram Penalty (Enforces ETF Pentagram) ---
        W_norms = torch.norm(W, dim=0, keepdim=True) + self.eps
        W_normed = W / W_norms
        gram_matrix = torch.matmul(W_normed.T, W_normed)
        
        N = W.shape[1]
        off_diag_mask = ~torch.eye(N, dtype=torch.bool, device=W.device)
        r = gram_matrix[off_diag_mask]
        
        r_sq = torch.clamp(r ** 2, max=1.0 - self.eps)
        
        if r_sq.numel() == 0:
            gram_penalty = torch.tensor(0.0, device=W.device)
        else:
            # -log(1 - max(r^2)) forces tightest frame line packing (r_max <= 0.8090)
            gram_penalty = -torch.log(1.0 - torch.max(r_sq) + self.eps)

        # --- Step 5: Feature Unit-Norm Penalty (Prevents Vector Shrinkage) ---
        norm_penalty = torch.mean((W_norms.squeeze(0) - 1.0) ** 2)

        total_loss = recon_loss + nll_uncertainty_loss + gram_penalty + norm_penalty
        return total_loss
    
class EntropicKreinL4NLLLoss4(nn.Module):
    def __init__(self, S0: float = 10.0, eps: float = 1e-6):
        super().__init__()
        self.S0 = S0
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, log_var_raw: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        # 1. Uncertainty & Error
        error = y_pred - y_true
        residual_var = torch.var(error) + self.eps
        dynamic_S0 = 0.5 * torch.log(residual_var / self.eps)
        S0 = torch.clamp(dynamic_S0, min=self.S0)

        log_var = log_var_raw / torch.sqrt(1.0 + (log_var_raw / S0) ** 2)
        var = torch.exp(log_var) + self.eps

        nll_loss = 0.5 * (error ** 2) / var + 0.5 * log_var
        l4_geometry_loss = torch.log1p(error ** 4).mean()
        
        # 2. Normalized Gram Matrix
        W_norms = torch.norm(W, dim=0, keepdim=True) + self.eps
        W_normed = W / W_norms
        gram_matrix = torch.matmul(W_normed.T, W_normed)
        
        # Mask out diagonal (i == j)
        N = W.shape[1]
        off_diag_mask = ~torch.eye(N, dtype=torch.bool, device=W.device)
        r = gram_matrix[off_diag_mask]
        
        # 3. Log-Barrier Penalty: -log(1 - r^2 + eps)
        # Prevents r -> 1 with infinite barrier, but maintains gradient down to r = 0.31
        r_sq = torch.clamp(r ** 2, max=1.0 - self.eps)
        log_barrier_gram = -torch.log(1.0 - r_sq + self.eps)
        gram_penalty = torch.mean(log_barrier_gram)

        total_loss = nll_loss.mean() + l4_geometry_loss.mean() + gram_penalty
        return total_loss

class EntropicKreinL4NLLLoss3(nn.Module):
    def __init__(self, l4_weight: float = 0.1, gram_weight: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.l4_weight = l4_weight
        self.gram_weight = gram_weight
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, log_var_raw: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        error = y_pred - y_true
        
        # 1. Softsign log_var with numerical safeguard against log(1.0) = 0
        sigma_max_sq = torch.var(y_true) + self.eps
        ratio = torch.clamp(sigma_max_sq / self.eps, min=1.0 + 1e-5)
        S0 = 0.5 * torch.log(ratio)
        log_var = log_var_raw / torch.sqrt(1.0 + (log_var_raw / S0) ** 2)
        var = torch.exp(log_var) + self.eps

        # 2. Heteroscedastic NLL
        nll_loss = 0.5 * (error ** 2) / var + 0.5 * log_var

        # 3. Variance-Normalized L4 Loss (keeps units consistent with NLL)
        l4_geometry_loss = (error ** 4) / (var ** 2)

        # 4. Normalized Gram Matrix Penalty (averaged, not summed, so scale is N-invariant)
        W_norms = torch.norm(W, dim=0, keepdim=True) + self.eps
        W_normed = W / W_norms
        gram_matrix = torch.matmul(W_normed.T, W_normed)
        off_diag_gram = gram_matrix - torch.eye(W.shape[1], device=W.device)
        gram_penalty = torch.mean(off_diag_gram ** 4)  # .mean() instead of .sum()

        total_loss = nll_loss.mean() + self.l4_weight * l4_geometry_loss.mean() + self.gram_weight * gram_penalty
        return total_loss
    
class EntropicKreinL4NLLLoss5(nn.Module):
    def __init__(self, S0_fallback: float = 10.0, eps: float = 1e-6, l4_delta: float = 2.0):
        super().__init__()
        self.S0_fallback = S0_fallback
        self.eps = eps
        self.l4_delta = l4_delta  # Transition to linear tail

    def forward(self, y_pred, y_true, log_var_raw, W):
        error = y_pred - y_true
        
        residual_var = torch.var(error.detach()) + self.eps
        dynamic_S0 = 0.5 * torch.log(residual_var / self.eps)
        S0 = torch.clamp(dynamic_S0, min=self.S0_fallback)

        log_var = log_var_raw / torch.sqrt(1.0 + (log_var_raw / S0) ** 2)
        var = torch.exp(log_var) + self.eps

        # Smooth L4 — raw character inside |e| < 2, vanishing tail beyond
        u = error / self.l4_delta
        l4_geometry_loss = (self.l4_delta ** 4 * torch.log1p(u ** 4)).mean()

        nll_uncertainty_loss = (0.5 * (error.detach() ** 2) / var + 0.5 * log_var).mean()

        W_norms = torch.norm(W, dim=0, keepdim=True) + self.eps
        W_normed = W / W_norms
        gram_matrix = torch.matmul(W_normed.T, W_normed)
        
        N = W.shape[1]
        off_diag_mask = ~torch.eye(N, dtype=torch.bool, device=W.device)
        r = gram_matrix[off_diag_mask]
        
        r_sq = torch.clamp(r ** 2, max=1.0 - self.eps)
        gram_penalty = -torch.mean(torch.log(1.0 - r_sq + self.eps))

        return l4_geometry_loss + nll_uncertainty_loss + gram_penalty
# =====================================================================
# TEST 1: Numerical Stability & Edge Cases
# =====================================================================

in_use_loss = EntropicKreinL4NLLLoss5
def test_numerical_stability():
    print("--- TEST 1: Numerical Stability & Edge Cases ---")
    loss_fn = in_use_loss()

    # Case A: Zero variance target (constant output)
    y_pred = torch.zeros(32, 5, requires_grad=True)
    y_true = torch.zeros(32, 5) # variance is 0
    log_var_raw = torch.zeros(32, 1, requires_grad=True)
    W = torch.randn(2, 5, requires_grad=True)

    loss = loss_fn(y_pred, y_true, log_var_raw, W)
    is_nan = torch.isnan(loss).any().item()
    print(f"Zero-variance target test -> Loss: {loss.item():.4f} | Contains NaN: {is_nan}")
    assert not is_nan, "FAILED: Loss produced NaN on zero-variance batch!"

    # Case B: Large errors (Testing L4 gradient explosion risk)
    y_pred_large = torch.tensor([[100.0, -100.0]], requires_grad=True)
    y_true_large = torch.tensor([[0.0, 0.0]])
    log_var_large = torch.tensor([[0.0]], requires_grad=True)
    W_large = torch.eye(2, requires_grad=True)

    loss_large = loss_fn(y_pred_large, y_true_large, log_var_large, W_large)
    loss_large.backward()
    grad_norm = y_pred_large.grad.norm().item()
    print(f"Large Residual (100.0) -> Loss: {loss_large.item():.2e} | Grad Norm: {grad_norm:.2e}")
    if grad_norm > 1e6:
        print("  ⚠️ WARNING: Extreme gradient explosion due to unscaled L4 error penalty!")

    print("✅ TEST 1 Passed.\n")


# =====================================================================
# TEST 2: Undercomplete vs. Overcomplete Gram Penalty Behavior
# =====================================================================
def test_gram_penalty_regimes():
    print("--- TEST 2: Gram Penalty in Different Capacity Regimes ---")
    loss_fn = in_use_loss()

    # Scenario A: Undercomplete (N_features=3 <= N_hidden=5) -> Perfect orthogonality achievable
    W_under = torch.randn(5, 3, requires_grad=True)
    optimizer = optim.Adam([W_under], lr=0.01)

    for _ in range(500):
        optimizer.zero_grad()
        y_dummy = torch.randn(16, 3)
        log_var_dummy = torch.zeros(16, 1)
        loss = loss_fn(y_dummy, y_dummy, log_var_dummy, W_under)
        loss.backward()
        optimizer.step()

    W_norm = W_under / torch.norm(W_under, dim=0, keepdim=True)
    gram_under = (W_norm.T @ W_norm).detach().numpy()
    max_off_diag_under = np.max(np.abs(gram_under - np.eye(3)))
    print(f"Undercomplete (3 features in 5D) Max Off-Diag Gram: {max_off_diag_under:.4f}")

    # Scenario B: Overcomplete (N_features=5 > N_hidden=2) -> Superposition required
    W_over = torch.randn(2, 5, requires_grad=True)
    optimizer = optim.Adam([W_over], lr=0.01)

    for _ in range(500):
        optimizer.zero_grad()
        y_dummy = torch.randn(16, 5)
        log_var_dummy = torch.zeros(16, 1)
        loss = loss_fn(y_dummy, y_dummy, log_var_dummy, W_over)
        loss.backward()
        optimizer.step()

    W_norm = W_over / torch.norm(W_over, dim=0, keepdim=True)
    gram_over = (W_norm.T @ W_norm).detach().numpy()
    max_off_diag_over = np.max(np.abs(gram_over - np.eye(5)))
    print(f"Overcomplete (5 features in 2D) Max Off-Diag Gram: {max_off_diag_over:.4f}")
    print("  ℹ️ Interpretation: In 2D, min theoretical max off-diag for 5 vectors is ~0.809 (Penta-gon geometry).")
    print("✅ TEST 2 Completed.\n")


# =====================================================================
# TEST 3: Heteroscedastic Noise Estimation Accuracy
# =====================================================================
def test_heteroscedastic_calibration():
    print("--- TEST 3: Heteroscedastic Noise & Uncertainty Calibration ---")
    loss_fn = in_use_loss()

    batch_size = 1000
    y_true = torch.zeros(batch_size, 1)
    
    true_sigma = torch.cat([torch.full((500, 1), 0.1), torch.full((500, 1), 1.0)])
    noise = torch.randn(batch_size, 1) * true_sigma
    y_pred = y_true + noise

    log_var_low = torch.nn.Parameter(torch.tensor([0.0]))
    log_var_high = torch.nn.Parameter(torch.tensor([0.0]))
    W_dummy = torch.eye(1)

    optimizer = optim.Adam([log_var_low, log_var_high], lr=0.05)

    for step in range(1000):
        optimizer.zero_grad()
        log_var_pred = torch.cat([log_var_low.repeat(500, 1), log_var_high.repeat(500, 1)])
        loss = loss_fn(y_pred, y_true, log_var_pred, W_dummy)
        loss.backward()
        optimizer.step()

    # FIX: Evaluate using actual error variance or loss_fn S0 logic (not y_true variance)
    with torch.no_grad():
        error = y_pred - y_true
        residual_var = torch.var(error) + loss_fn.eps
        dynamic_S0 = 0.5 * torch.log(residual_var / loss_fn.eps)
        S0 = dynamic_S0.item()

        v_low = log_var_low / torch.sqrt(1.0 + (log_var_low / S0) ** 2)
        v_high = log_var_high / torch.sqrt(1.0 + (log_var_high / S0) ** 2)

        pred_var_low = torch.exp(v_low).item()
        pred_var_high = torch.exp(v_high).item()

    print(f"Low Noise Group  -> True Var: 0.0100 | Fitted Var: {pred_var_low:.4f}")
    print(f"High Noise Group -> True Var: 1.0000 | Fitted Var: {pred_var_high:.4f}")

    if abs(pred_var_low - 0.01) > 0.1 or abs(pred_var_high - 1.0) > 0.3:
        print("  ⚠️ WARNING: Distortion in variance estimation!")
    else:
        print("  ✅ Uncertainty estimation calibrated well under noise.")
    print("✅ TEST 3 Completed.\n")


# =====================================================================
# TEST 4: Reconstruction Fidelity vs Standard Loss Functions
# =====================================================================
def test_reconstruction_fidelity():
    print("--- TEST 4: Sparse Reconstruction Benchmark ---")
    
    def generate_sparse_data(batch_size=512, n_features=5, prob=0.1):
        mask = (torch.rand(batch_size, n_features) < prob).float()
        values = torch.rand(batch_size, n_features)
        return mask * values

    class SimpleAE(nn.Module):
        def __init__(self, n_features=5, n_hidden=2):
            super().__init__()
            self.W = nn.Parameter(torch.randn(n_hidden, n_features) * 0.1)
            self.head2 = nn.Sequential(nn.Linear(n_hidden, 8), nn.ReLU(), nn.Linear(8, 1))

        def forward(self, x):
            h = torch.matmul(x, self.W.T)
            head1 = torch.matmul(h, self.W)
            head2 = self.head2(h)
            return head1, head2

    # Benchmark EntropicKreinL4 vs standard MSE
    for name, is_l4 in [("Standard MSE Loss", False), ("EntropicKrein L4 Loss", True)]:
        model = SimpleAE()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        loss_l4_fn = in_use_loss()

        for step in range(3000):
            x = generate_sparse_data()
            optimizer.zero_grad()
            head1, head2 = model(x)
            
            if not is_l4:
                loss = torch.mean((head1 - x) ** 2)
            else:
                loss = loss_l4_fn(head1, x, head2, model.W)

            loss.backward()
            optimizer.step()

        # Evaluate final reconstruction MSE
        test_x = generate_sparse_data(batch_size=2000)
        with torch.no_grad():
            pred_x, _ = model(test_x)
            rec_mse = torch.mean((pred_x - test_x) ** 2).item()
            
            W_norm = model.W / (torch.norm(model.W, dim=0, keepdim=True) + 1e-8)
            gram = (W_norm.T @ W_norm).cpu().numpy()
            max_off_diag = np.max(np.abs(gram - np.eye(5)))

        print(f"[{name}] Final Rec MSE: {rec_mse:.6f} | Max Off-Diag Interference: {max_off_diag:.4f}")

    print("✅ TEST 4 Completed.\n")


if __name__ == "__main__":
    test_numerical_stability()
    test_gram_penalty_regimes()
    test_heteroscedastic_calibration()
    test_reconstruction_fidelity()