# models/kkanvs9.py
import torch
import torch.nn as nn
import math

# =====================================================================
# 1. PARAMETRIC CARDANO-NEWTON ZERO-FINDER
# =====================================================================

class CardanoNewtonActivation(nn.Module):
    """
    Solves G(y) = y^3 + a*y^2 + r^2*y - B = 0 via unrolled Newton-Raphson (K=3).
    Guarantees: G'(y) = 3y^2 + 2ay + r^2 >= sigma^2 + 1e-4 > 0 strictly everywhere.
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

        abs_B = B.abs()
        denom_init = r_sq_vec + abs_B.pow(2.0 / 3.0)
        y = B / denom_init

        for _ in range(self.steps):
            y_sq = y.pow(2)
            G = (y_sq + a_vec * y + r_sq_vec) * y - B
            G_prime = 3.0 * y_sq + 2.0 * a_vec * y + r_sq_vec
            y = y - (G / G_prime)

        return y

    def get_diagnostics(self):
        with torch.no_grad():
            r_sq = (self.a.pow(2) / 3.0) + self.sigma.pow(2) + 1e-4
            r_eff = torch.sqrt(r_sq).cpu().flatten()
            a_val = self.a.detach().cpu().flatten()
            return {
                "r_eff": r_eff,
                "a": a_val,
                "r_mean": r_eff.mean().item(),
                "a_mean": a_val.mean().item(),
                "a_abs_mean": a_val.abs().mean().item()
            }


# =====================================================================
# 2. MODULAR COMPRESSIVE ROOT-KAN BLOCK (CONFIGURABLE)
# =====================================================================

class ModularCompressiveBlock(nn.Module):
    """
    Configurable RootKAN Block:
      - degree: polynomial factor count
      - comp_steps: 0 (no solve), 1, or 2 Halley compression steps
      - dropout_p: dropout rate
      - use_linear_skip: whether to add u directly to polynomial product
    """
    def __init__(self, in_features, out_features, degree=3, comp_steps=2, dropout_p=0.0, use_linear_skip=True):
        super().__init__()
        self.degree = degree
        self.comp_steps = comp_steps
        self.use_linear_skip = use_linear_skip

        self.linear = nn.Linear(in_features, out_features)
        self.drop = nn.Dropout(dropout_p) if dropout_p > 0.0 else nn.Identity()

        k = torch.arange(1, degree + 1, dtype=torch.float32)
        cheb_nodes = torch.cos((2 * k - 1) / (2 * degree) * math.pi)
        self.roots = nn.Parameter(cheb_nodes.view(-1, 1).repeat(1, out_features))
        self.scale = nn.Parameter(torch.randn(out_features) * 0.1)

        self.ln = nn.LayerNorm(out_features)
        self.act = nn.SiLU()

    def _eval_and_compress(self, u):
        denom = torch.sqrt(1.0 + self.roots.pow(2))
        prod = (u - self.roots[0]) / denom[0]
        for k in range(1, self.degree):
            prod = prod * ((u - self.roots[k]) / denom[k])

        v = prod * self.scale
        if self.use_linear_skip:
            v = v + u

        if self.comp_steps <= 0:
            return v

        # Symmetric Halley solve (a=0): D = 12y^4 + 6y^2 + 2 > 0 strictly
        y = v / (1.0 + v.abs().pow(2.0 / 3.0))
        for _ in range(self.comp_steps):
            y_sq = y.pow(2)
            G = y_sq * y + y - v
            G_prime = 3.0 * y_sq + 1.0
            G_double_prime = 6.0 * y
            num = 2.0 * G * G_prime
            denom_h = 2.0 * G_prime.pow(2) - G * G_double_prime
            y = y - num / denom_h
        return y

    def forward(self, x):
        u = self.linear(x)
        y = self._eval_and_compress(u)
        return self.drop(self.act(self.ln(y)))


# =====================================================================
# 3. COMPLETE MODEL WRAPPERS
# =====================================================================

class ModularRootKAN_Deep(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, degree=3, comp_steps=2, dropout_p=0.0, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.in_block = ModularCompressiveBlock(in_dim, hidden_dim, degree=degree, comp_steps=comp_steps, dropout_p=dropout_p)
        self.hidden_blocks = nn.ModuleList([
            ModularCompressiveBlock(hidden_dim, hidden_dim, degree=degree, comp_steps=comp_steps, dropout_p=dropout_p)
            for _ in range(num_layers - 1)
        ])
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h = self.in_block(x)
        for block in self.hidden_blocks:
            h = (h + block(h)) if self.use_residual else block(h)
        return self.out_proj(h)

    def inspect_root_topology(self):
        roots = [self.in_block.roots.detach().cpu().flatten()]
        for b in self.hidden_blocks:
            roots.append(b.roots.detach().cpu().flatten())
        all_r = torch.cat(roots)
        total = len(all_r)
        active = (all_r.abs() <= 1.0).sum().item()
        ghost = (all_r.abs() > 1.0).sum().item()
        s_r = torch.sort(all_r).values
        q = [s_r[min(int(total * p), total - 1)].item() for p in [0.0, 0.25, 0.5, 0.75, 1.0]]
        return active, ghost, total, q


class CardanoNewtonMLP_Deep(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.in_act = CardanoNewtonActivation(hidden_dim, steps=3)
        self.in_ln = nn.LayerNorm(hidden_dim)

        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "linear": nn.Linear(hidden_dim, hidden_dim),
                "act": CardanoNewtonActivation(hidden_dim, steps=3),
                "ln": nn.LayerNorm(hidden_dim)
            }) for _ in range(num_layers - 1)
        ])
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h = self.in_ln(self.in_act(self.in_proj(x)))
        for block in self.blocks:
            res = h
            out = block["ln"](block["act"](block["linear"](h)))
            h = (res + out) if self.use_residual else out
        return self.out_proj(h)

    def inspect_diagnostics(self):
        diag = [self.in_act.get_diagnostics()]
        for b in self.blocks:
            diag.append(b["act"].get_diagnostics())
        return diag


class ModernMLP_Deep(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU()
            ) for _ in range(num_layers - 1)
        ])
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h = self.in_proj(x)
        for block in self.blocks:
            h = (h + block(h)) if self.use_residual else block(h)
        return self.out_proj(h)