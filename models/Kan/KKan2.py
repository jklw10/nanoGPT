import torch
import torch.nn as nn
import math
import time

# =====================================================================
# 1. OPTIMIZER WITH DYNAMIC HORIZON & UNCENTERED FISHER
# =====================================================================

def compute_soft_kronecker_matrix(Cov, epsilon=1e-4, steps=2, power_iters=2, alpha=0.5):
    d = Cov.shape[0]
    device, dtype = Cov.device, Cov.dtype
    eye_d = torch.eye(d, device=device, dtype=dtype)
    Cov_reg = Cov + epsilon * eye_d

    v = torch.ones((d, 1), device=device, dtype=dtype)
    for _ in range(power_iters):
        u = torch.matmul(Cov_reg, v)
        u = u / (u.norm() + 1e-12)
        v = torch.matmul(Cov_reg.T, u)
        v = v / (v.norm() + 1e-12)

    s = torch.matmul(u.T, torch.matmul(Cov_reg, v)).squeeze().clamp(min=1e-8)
    B = Cov_reg / s
    Y = eye_d.clone()

    for _ in range(steps):
        Y = 0.5 * torch.matmul(Y, 3.0 * eye_d - torch.matmul(Y, torch.matmul(B, Y)))

    W_hard = Y / torch.sqrt(s)
    return alpha * W_hard + (1.0 - alpha) * eye_d


def update_entropic_krein_step(
    p: torch.Tensor,
    p_plus: torch.Tensor,
    p_minus: torch.Tensor,
    u_grad_precond: torch.Tensor,
    lr: float,
    gamma: float,
    use_krein_space: bool
):
    eps_geom = 1e-6
    current_max = torch.max(torch.abs(p)).item()
    R_t = max(15.0, current_max * 2.5) + eps_geom

    if use_krein_space:
        u_plus = torch.clamp(-u_grad_precond, min=0.0)
        u_minus = torch.clamp(u_grad_precond, min=0.0)

        x_norm_plus = torch.clamp(p_plus / R_t, min=eps_geom, max=0.98)
        K_plus = x_norm_plus / (1.0 + x_norm_plus)
        K_prime_plus = 1.0 / (1.0 + x_norm_plus).square()
        force_plus = gamma * (K_plus * K_prime_plus) / (1.0 - K_plus.square() + eps_geom)

        x_norm_minus = torch.clamp(p_minus / R_t, min=eps_geom, max=0.98)
        K_minus = x_norm_minus / (1.0 + x_norm_minus)
        K_prime_minus = 1.0 / (1.0 + x_norm_minus).square()
        force_minus = gamma * (K_minus * K_prime_minus) / (1.0 - K_minus.square() + eps_geom)

        p_plus_raw = torch.clamp(p_plus + lr * (u_plus - force_plus), min=eps_geom)
        p_minus_raw = torch.clamp(p_minus + lr * (u_minus - force_minus), min=eps_geom)
        
        p_data_next = p_plus_raw - p_minus_raw

        p_plus_next = torch.clamp(p_data_next, min=0.0) + eps_geom
        p_minus_next = torch.clamp(-p_data_next, min=0.0) + eps_geom

        return p_plus_next, p_minus_next, p_data_next
    else:
        p_abs = torch.abs(p)
        x_norm = torch.clamp(p_abs / R_t, min=eps_geom, max=0.98)
        K_val = x_norm / (1.0 + x_norm)
        K_prime = 1.0 / (1.0 + x_norm).square()
        force_direct = torch.sign(p) * gamma * (K_val * K_prime) / (1.0 - K_val.square() + eps_geom)

        p_data_next = p - lr * (u_grad_precond + force_direct)
        return p_plus, p_minus, p_data_next


class FixedKreinOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, beta_cov=0.1,
                 alpha=0.5, eps=1e-8, gamma=1e-4, damping=1e-3, weight_decay=1e-4,
                 use_adam=True, use_kron=False, use_krein_space=False):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, beta_cov=beta_cov, alpha=alpha,
                        eps=eps, gamma=gamma, damping=damping, weight_decay=weight_decay,
                        use_adam=use_adam, use_kron=use_kron, use_krein_space=use_krein_space)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['beta1'], group['beta2']
            beta_cov, alpha = group['beta_cov'], group['alpha']
            eps, gamma = group['gamma'], group['eps']
            damping, weight_decay = group['damping'], group['weight_decay']
            use_adam, use_kron = group['use_adam'], group['use_kron']
            use_krein_space = group['use_krein_space']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                orig_shape = p.shape
                is_multidim = p.dim() >= 2

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data, dtype=torch.float32)
                    state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=torch.float32)
                    state['p_plus'] = torch.clamp(p.data, min=0.0).add_(1e-4)
                    state['p_minus'] = torch.clamp(-p.data, min=0.0).add_(1e-4)
                    state['L_ema'] = None
                    state['R_ema'] = None
                    state['W_L'] = None
                    state['W_R'] = None

                state['step'] += 1
                step_idx = state['step']
                p_plus, p_minus = state['p_plus'], state['p_minus']

                if weight_decay != 0:
                    p.data.mul_(1.0 - lr * weight_decay)

                grad_fp32 = grad.to(torch.float32)

                exp_avg = state['exp_avg']
                if beta1 > 0:
                    exp_avg.mul_(beta1).add_(grad_fp32, alpha=1.0 - beta1)
                    m_hat = exp_avg / (1.0 - beta1 ** step_idx)
                else:
                    m_hat = grad_fp32

                if use_adam:
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg_sq.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1.0 - beta2)
                    v_hat = exp_avg_sq / (1.0 - beta2 ** step_idx)
                    d_inv = 1.0 / (torch.sqrt(v_hat) + eps)
                    g_drive = m_hat * d_inv
                else:
                    g_drive = m_hat

                if use_kron and is_multidim:
                    m = orig_shape[0]
                    n = p.numel() // m
                    g_2d = g_drive.view(m, n)
                    
                    L_batch = (1.0 / n) * torch.matmul(g_2d, g_2d.T)
                    R_batch = (1.0 / m) * torch.matmul(g_2d.T, g_2d)

                    if state['L_ema'] is None:
                        state['L_ema'] = torch.zeros_like(L_batch)
                        state['R_ema'] = torch.zeros_like(R_batch)

                    state['L_ema'].mul_(beta_cov).add_(L_batch, alpha=1.0 - beta_cov)
                    state['R_ema'].mul_(beta_cov).add_(R_batch, alpha=1.0 - beta_cov)
                    bias_corr = 1.0 - (beta_cov ** step_idx)

                    if state['W_L'] is None or step_idx % 5 == 0:
                        state['W_L'] = compute_soft_kronecker_matrix(
                            state['L_ema'] / bias_corr, epsilon=damping, alpha=alpha
                        )
                        state['W_R'] = compute_soft_kronecker_matrix(
                            state['R_ema'] / bias_corr, epsilon=damping, alpha=alpha
                        )

                    u_kron_2d = torch.matmul(state['W_L'], torch.matmul(g_2d, state['W_R']))
                    u_grad_precond = u_kron_2d.view_as(p.data)
                else:
                    u_grad_precond = g_drive

                p_plus_next, p_minus_next, p_data_next = update_entropic_krein_step(
                    p.data, p_plus, p_minus, u_grad_precond,
                    lr, gamma, use_krein_space
                )

                if use_krein_space:
                    p_plus.copy_(p_plus_next)
                    p_minus.copy_(p_minus_next)

                p.data.copy_(p_data_next.to(p.dtype))


# =====================================================================
# 2. THE THREE ARCHITECTURES
# =====================================================================

class TaylorCoeffKAN(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=6, out_dim=1, degree=3):
        super().__init__()
        self.degree = degree
        fact = torch.tensor([math.factorial(k) for k in range(degree + 1)], dtype=torch.float32)
        self.register_buffer("inv_fact", 1.0 / fact)
        self.c1 = nn.Parameter(torch.randn(degree + 1, hidden_dim, in_dim) * 0.1)
        self.c2 = nn.Parameter(torch.randn(degree + 1, out_dim, hidden_dim) * 0.1)

    def _eval_layer(self, x, coeffs):
        powers = [torch.ones_like(x)]
        for _ in range(self.degree):
            powers.append(powers[-1] * x)
        poly = torch.stack(powers, dim=0) * self.inv_fact.view(-1, 1, 1)
        return torch.einsum('doi,dbi->bo', coeffs, poly)

    def forward(self, x):
        h = torch.tanh(self._eval_layer(x, self.c1))
        return self._eval_layer(h, self.c2)


class RootFactoredKAN(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=6, out_dim=1, degree=3):
        super().__init__()
        self.degree = degree
        self.roots1 = nn.Parameter(torch.linspace(-0.6, 0.6, degree).view(-1, 1, 1).repeat(1, hidden_dim, in_dim))
        self.scale1 = nn.Parameter(torch.randn(hidden_dim, in_dim) * 0.2)
        self.bias1  = nn.Parameter(torch.zeros(hidden_dim))

        self.roots2 = nn.Parameter(torch.linspace(-0.6, 0.6, degree).view(-1, 1, 1).repeat(1, out_dim, hidden_dim))
        self.scale2 = nn.Parameter(torch.randn(out_dim, hidden_dim) * 0.2)
        self.bias2  = nn.Parameter(torch.zeros(out_dim))

    def _eval_layer(self, x, roots, scale, bias):
        factors = x.unsqueeze(0).unsqueeze(2) - roots.unsqueeze(1)
        prod = torch.prod(factors, dim=0)
        return torch.sum(prod * scale.unsqueeze(0), dim=-1) + bias

    def forward(self, x):
        h = torch.tanh(self._eval_layer(x, self.roots1, self.scale1, self.bias1))
        return self._eval_layer(h, self.roots2, self.scale2, self.bias2)


class ImplicitNewtonKAN(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=6, out_dim=1, steps=4):
        super().__init__()
        self.steps = steps
        self.w0_1 = nn.Linear(in_dim, hidden_dim)
        self.w1_1 = nn.Linear(in_dim, hidden_dim)
        self.w0_2 = nn.Linear(hidden_dim, out_dim)
        self.w1_2 = nn.Linear(hidden_dim, out_dim)

    def _solve_layer(self, x, w0, w1):
        A = torch.relu(w1(x)) + 0.1
        B = w0(x)
        y = torch.zeros_like(B)
        for _ in range(self.steps):
            G = y.pow(3) + A * y - B
            G_prime = 3.0 * y.pow(2) + A + 1e-6
            y = y - (G / G_prime)
        return y

    def forward(self, x):
        h = torch.tanh(self._solve_layer(x, self.w0_1, self.w1_1))
        return self._solve_layer(h, self.w0_2, self.w1_2)


# =====================================================================
# 3. BENCHMARK WITH SCHEDULING & ROOT TRACKING
# =====================================================================

def get_data(task: str, n=512):
    torch.manual_seed(1337)
    x = torch.rand(n, 2) * 2.0 - 1.0
    if task == "MultiScale_Poly":
        y = 3.5 * (x[:, 0:1]**4) - 1.2 * (x[:, 0:1]**2) + 0.5 * torch.sin(math.pi * x[:, 1:2])
    elif task == "Saddle_Valley":
        y = (1.0 - x[:, 0:1])**2 + 4.0 * (x[:, 1:2] - x[:, 0:1]**2)**2
    return x, y


def format_trajectory(losses):
    l_init = losses[0]
    l_q1 = losses[len(losses)//4]
    l_mid = losses[len(losses)//2]
    l_final = losses[-1]

    pct_drop = ((l_init - l_final) / (l_init + 1e-12)) * 100.0
    indicator = "📉" if pct_drop > 25.0 else ("📈" if pct_drop < -10.0 else "⏸ ")
    return f"[{l_init:.2f}➔{l_q1:.2f}➔{l_mid:.2f}➔{l_final:.3f}] ({indicator} {pct_drop:+5.1f}%)"


def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks = ["MultiScale_Poly", "Saddle_Valley"]
    models = ["Taylor_Coeff", "Root_Factored", "Implicit_Newton"]
    optimizers = ["SGD_Adam", "Standard_NGD", "Krein_NGD"]
    epochs = 120

    print("=" * 118)
    print(f"{'TASK':<16} | {'MODEL':<16} | {'OPTIMIZER':<13} | {'MIN':<7} | {'FINAL':<7} | {'TRAJECTORY [Init➔Q1➔Mid➔Final]'}")
    print("=" * 118)

    sample_roots = {}

    for task in tasks:
        x, y = get_data(task)
        x, y = x.to(device), y.to(device)

        for m_name in models:
            for opt_name in optimizers:
                torch.manual_seed(42)

                if m_name == "Taylor_Coeff":
                    model = TaylorCoeffKAN(degree=3).to(device)
                elif m_name == "Root_Factored":
                    model = RootFactoredKAN(degree=3).to(device)
                elif m_name == "Implicit_Newton":
                    model = ImplicitNewtonKAN(steps=4).to(device)

                lr = 0.02
                if opt_name == "SGD_Adam":
                    opt = FixedKreinOptimizer(model.parameters(), lr=lr, use_adam=True, use_kron=False, use_krein_space=False)
                elif opt_name == "Standard_NGD":
                    opt = FixedKreinOptimizer(model.parameters(), lr=lr, use_adam=True, use_kron=True, use_krein_space=False, damping=1e-3)
                elif opt_name == "Krein_NGD":
                    opt = FixedKreinOptimizer(model.parameters(), lr=lr, use_adam=True, use_kron=True, use_krein_space=True, damping=1e-3, gamma=1e-4)

                # COSINE ANNEALING: Slingshot Brake
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)

                loss_fn = nn.MSELoss()
                losses = []

                for _ in range(epochs):
                    opt.zero_grad()
                    out = model(x)
                    loss = loss_fn(out, y)
                    loss.backward()
                    opt.step()
                    scheduler.step()
                    losses.append(loss.item())

                min_l = min(losses)
                fin_l = losses[-1]
                traj_str = format_trajectory(losses)

                if m_name == "Root_Factored" and opt_name == "Krein_NGD":
                    sample_roots[task] = model.roots1[:, 0, 0].detach().cpu().tolist()

                print(f"{task:<16} | {m_name:<16} | {opt_name:<13} | {min_l:<7.4f} | {fin_l:<7.4f} | {traj_str}")
            print("-" * 118)

    print("\n" + "=" * 60)
    print("ROOT INSPECTION (Initial roots were [-0.60, 0.00, +0.60]):")
    print("=" * 60)
    for task, roots in sample_roots.items():
        formatted = [f"{r:+.3f}" for r in sorted(roots)]
        print(f"Learned Roots for {task:<16}: {formatted}")


if __name__ == "__main__":
    run_benchmark()