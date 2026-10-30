import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================
# 1. DISCRETE BOTTLENECK & SYMMETRIC ACTIVATION
# =========================================================================
class Puntan(nn.Module):
    """Sign-symmetric activation function: f(-x) = -f(x)."""
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((dim,), 0.5))

    def forward(self, x):
        return x - torch.tanh(x) * self.alpha

def topkhotSTE(x, k):
    """Straight-Through Estimator for Top-K discrete code generation."""
    probs = torch.sigmoid(x)
    _, indices = torch.topk(x, k=k, dim=-1)
    hard = torch.zeros_like(x).scatter_(-1, indices, 1.0)
    return hard + probs - probs.detach()


# =========================================================================
# 2. INVOLUTIONARY HAIS BLOCK (WITH SELF-CONSISTENCY LOSS)
# =========================================================================
class HAISBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        
        # Linear layers without bias to preserve origin-symmetry
        self.w1 = nn.Linear(dim, dim * 2, bias=False)
        self.act = Puntan(dim * 2)
        self.w2 = nn.Linear(dim * 2, dim, bias=False)
        self.proj_attn = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        # Odd-symmetric residual transition
        h = self.w2(self.act(self.w1(self.ln1(x))))
        h_attn = torch.tanh(self.proj_attn(self.ln2(x))) * x
        return x + h + h_attn

    def compute_inversion_loss(self, x_in, x_out):
        """
        Involutionary Self-Consistency Check:
        Passes negated output -x_out back through the block.
        Satisfies: F(-F(x)) ≈ x
        """
        reconstructed_in = self(-x_out)
        
        # Normalize to measure structural shape consistency
        norm_rec = F.normalize(reconstructed_in, dim=-1)
        norm_orig = F.normalize(x_in, dim=-1)
        
        inv_loss = F.mse_loss(norm_rec, norm_orig)
        return inv_loss


# =========================================================================
# 3. HIERARCHICAL ACTIVE INFERENCE LEVEL
# =========================================================================
class HAISLevel(nn.Module):
    def __init__(self, dim, k_active):
        super().__init__()
        self.dim = dim
        self.k_active = k_active
        self.block = HAISBlock(dim)
        self.predictor = nn.Linear(dim, dim, bias=False)

    def forward(self, z_curr, top_down_goal=None):
        # 1. Forward transition
        z_next_raw = self.block(z_curr)
        
        # Add top-down goal influence if available
        if top_down_goal is not None:
            z_next_raw = z_next_raw + 0.2 * top_down_goal

        # 2. Discrete bottleneck (kills floating-point drift)
        z_next_discrete = topkhotSTE(z_next_raw, k=self.k_active)

        # 3. Local Involutionary Consistency Loss
        inv_loss = self.block.compute_inversion_loss(z_curr, z_next_raw)

        return z_next_discrete, inv_loss


# =========================================================================
# 4. FULL HIERARCHICAL ACTIVE INFERENCE SYSTEM (HAIS)
# =========================================================================
class HAISAgent(nn.Module):
    def __init__(self, dim=64, k_active=16):
        super().__init__()
        self.dim = dim
        # Level 0: Fast Perception/Action Loop
        self.level0 = HAISLevel(dim, k_active)
        # Level 1: Slow Abstract Thought / Goal Loop
        self.level1 = HAISLevel(dim, k_active)
        
        # Bottom-Up Abstraction Head (L0 -> L1 summary)
        self.bottom_up = nn.Linear(dim, dim, bias=False)
        # Top-Down Goal Head (L1 -> L0 guidance)
        self.top_down = nn.Linear(dim, dim, bias=False)

    def step(self, z0_curr, z1_curr, macro_step=False):
        """
        Executes one Active Inference step across the hierarchy.
        """
        # --- Level 1 (Slow Thought Stream) ---
        if macro_step:
            # L1 updates its abstract thought vector
            summary_from_l0 = self.bottom_up(z0_curr)
            z1_next, l1_inv_loss = self.level1(z1_curr + summary_from_l0)
        else:
            z1_next = z1_curr
            l1_inv_loss = torch.tensor(0.0, device=z0_curr.device)

        # Top-down goal emitted by Level 1 to guide Level 0
        goal_signal = self.top_down(z1_next)

        # --- Level 0 (Fast Action Stream) ---
        z0_next, l0_inv_loss = self.level0(z0_curr, top_down_goal=goal_signal)

        # --- Hierarchical Free Energy / Prediction Error ---
        # Measures if L0's state is aligned with L1's goal expectation
        l0_summary = self.bottom_up(z0_next)
        free_energy = F.mse_loss(l0_summary, z1_next.detach())

        # Combined Total Self-Consistency Metric
        total_consistency_loss = l0_inv_loss + l1_inv_loss + free_energy

        metrics = {
            'l0_inversion_loss': l0_inv_loss.item(),
            'l1_inversion_loss': l1_inv_loss.item(),
            'free_energy': free_energy.item(),
            'total_consistency_score': total_consistency_loss.item()
        }

        return z0_next, z1_next, total_consistency_loss, metrics


# =========================================================================
# 5. EXECUTION & VERIFICATION AUDIT
# =========================================================================
if __name__ == "__main__":
    print("=== Initializing Hierarchical Active Inference System (HAIS) ===")
    agent = HAISAgent(dim=64, k_active=16)
    
    # Dummy Latent Thought Vectors
    z0 = torch.randn(1, 64)  # Fast state
    z1 = torch.randn(1, 64)  # Slow macro-thought

    print("\nExecuting 5-Step Latent Thought Stream:")
    print(f"{'Step':>5} | {'Macro Update':>12} | {'L0 Inv Loss':>12} | {'Free Energy':>12} | {'Consistency Score':>18}")
    print("-" * 70)

    for step in range(1, 6):
        # Update high-level thought stream every 2 steps
        is_macro = (step % 2 == 0)
        
        z0, z1, loss, metrics = agent.step(z0, z1, macro_step=is_macro)
        
        print(f"{step:5d} | {str(is_macro):>12} | {metrics['l0_inversion_loss']:12.5f} | "
              f"{metrics['free_energy']:12.5f} | {metrics['total_consistency_score']:18.5f}")

    print("\n✅ HAIS Execution Complete!")