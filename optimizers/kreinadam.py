import torch

@torch.compile(backend="inductor", options={"triton.cudagraphs": False})
def compute_soft_kronecker_matrix(Cov, epsilon=1e-5, steps=2, power_iters=2, alpha=0.5):
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


@torch.compile(backend="inductor", options={"triton.cudagraphs": False})
def update_entropic_krein_step(
    p_fp32: torch.Tensor,
    p_plus: torch.Tensor,
    p_minus: torch.Tensor,
    u_grad_precond: torch.Tensor,
    lr: float,
    gamma: float,
    scale_ref: torch.Tensor,
    use_krein_space: bool,
):
    eps_geom = 1e-6
    R_t = 6.0 * scale_ref + eps_geom

    if use_krein_space:
        u_plus = torch.clamp(-u_grad_precond, min=0.0)
        u_minus = torch.clamp(u_grad_precond, min=0.0)

        x_norm_plus = torch.clamp(p_plus / R_t, min=eps_geom, max=0.99)
        K_plus = x_norm_plus / (1.0 + x_norm_plus)
        K_prime_plus = 1.0 / (1.0 + x_norm_plus).square()
        force_plus = gamma * (K_plus * K_prime_plus) / (1.0 - K_plus.square() + eps_geom)

        x_norm_minus = torch.clamp(p_minus / R_t, min=eps_geom, max=0.99)
        K_minus = x_norm_minus / (1.0 + x_norm_minus)
        K_prime_minus = 1.0 / (1.0 + x_norm_minus).square()
        force_minus = gamma * (K_minus * K_prime_minus) / (1.0 - K_minus.square() + eps_geom)

        p_plus_raw = torch.clamp(p_plus + lr * (u_plus - force_plus), min=eps_geom)
        p_minus_raw = torch.clamp(p_minus + lr * (u_minus - force_minus), min=eps_geom)
        p_data_next = p_plus_raw - p_minus_raw

        p_plus_next = torch.clamp(p_data_next, min=0.0) + eps_geom
        p_minus_next = torch.clamp(-p_data_next, min=0.0) + eps_geom
        return p_plus_next, p_minus_next, p_data_next

    x_norm = torch.clamp(torch.abs(p_fp32) / R_t, min=eps_geom, max=0.99)
    K_val = x_norm / (1.0 + x_norm)
    K_prime = 1.0 / (1.0 + x_norm).square()
    force_direct = torch.sign(p_fp32) * gamma * (K_val * K_prime) / (1.0 - K_val.square() + eps_geom)

    p_data_next = p_fp32 - lr * (u_grad_precond + force_direct)
    return p_plus, p_minus, p_data_next


class UnifiedKreinEntropicOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        beta_cov: float = 0.1,
        alpha: float = 1.0,
        eps: float = 1e-8,
        gamma: float = 1e-4,
        damping: float = 1e-5,
        weight_decay: float = 1e-2,
        max_kron_dim: int = 2048,
        kron_freq: int = 10,
        steps_ns: int = 2,
        power_iters: int = 2,
        use_adam: bool = True,
        use_kron: bool = False,
        use_krein_space: bool = False,
        use_grms: bool = True,
    ):
        defaults = dict(
            lr=lr, beta1=beta1, beta2=beta2, beta_cov=beta_cov, alpha=alpha,
            eps=eps, gamma=gamma, damping=damping, weight_decay=weight_decay,
            max_kron_dim=max_kron_dim, kron_freq=kron_freq,
            steps_ns=steps_ns, power_iters=power_iters,
            use_adam=use_adam, use_kron=use_kron,
            use_krein_space=use_krein_space, use_grms=use_grms,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            beta_cov = group["beta_cov"]
            alpha = group["alpha"]
            eps = group["eps"]
            gamma = group["gamma"]
            damping = group["damping"]
            weight_decay = group["weight_decay"]
            max_kron_dim = group["max_kron_dim"]
            kron_freq = group["kron_freq"]
            steps_ns = group["steps_ns"]
            power_iters = group["power_iters"]
            use_adam = group["use_adam"]
            use_kron = group["use_kron"]
            use_krein_space = group["use_krein_space"]
            use_grms = group["use_grms"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad_fp32 = p.grad.to(torch.float32)
                state = self.state[p]
                orig_shape = p.shape
                is_multidim = p.dim() > 1
                m, n = (orig_shape[0], p.numel() // orig_shape[0]) if is_multidim else (p.numel(), 1)

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data, dtype=torch.float32)
                    if use_adam:
                        state["exp_avg_sq"] = torch.zeros_like(p.data, dtype=torch.float32)
                    if use_krein_space:
                        state["p_plus"] = torch.clamp(p.data.float(), min=0.0).add_(1e-4)
                        state["p_minus"] = torch.clamp(-p.data.float(), min=0.0).add_(1e-4)
                    else:
                        state["p_plus"] = p.data.new_empty(0)
                        state["p_minus"] = p.data.new_empty(0)

                    std_raw = torch.std(p.data.float()) if p.numel() > 1 else torch.abs(p.data.float())
                    state["std_init"] = torch.clamp(torch.nan_to_num(std_raw, nan=1e-4), min=1e-4)
                    state["L_ema"] = None
                    state["R_ema"] = None
                    state["W_L"] = None
                    state["W_R"] = None

                state["step"] += 1
                step_idx = state["step"]
                p_fp32 = p.data.float()
                p_plus = state["p_plus"]
                p_minus = state["p_minus"]

                # Decoupled weight decay
                if weight_decay != 0.0:
                    decay_factor = 1.0 - lr * weight_decay
                    p_fp32.mul_(decay_factor)
                    if use_krein_space:
                        p_plus.mul_(decay_factor)
                        p_minus.mul_(decay_factor)

                # 1. Per-matrix Gradient RMS normalization
                if use_grms:
                    nd = (-1, -2) if grad_fp32.ndim >= 2 else (-1,)
                    rms = torch.rsqrt(grad_fp32.pow(2).mean(dim=nd, keepdim=True) + 1e-8)
                    grad_fp32 = grad_fp32 * rms

                # 2. Momentum
                if beta1 > 0.0:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(beta1).add_(grad_fp32, alpha=1.0 - beta1)
                    m_hat = exp_avg / (1.0 - (beta1 ** step_idx))
                else:
                    m_hat = grad_fp32

                # 3. Diagonal second moment A^(1/2) = v^(-1/4)
                if use_adam:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1.0 - beta2)
                    v_hat = exp_avg_sq / (1.0 - (beta2 ** step_idx))
                    d_half = (torch.sqrt(v_hat) + eps).rsqrt()
                    g_drive = m_hat * d_half
                else:
                    d_half = None
                    g_drive = m_hat

                # 4. Symmetric Kronecker sandwich (optional)
                if use_kron and is_multidim and m <= max_kron_dim and n <= max_kron_dim:
                    g_2d = g_drive.view(m, n)
                    g_centered = g_2d - g_2d.mean(dim=0, keepdim=True)

                    L_batch = (1.0 / n) * torch.matmul(g_centered, g_centered.T)
                    R_batch = (1.0 / m) * torch.matmul(g_centered.T, g_centered)

                    if state["L_ema"] is None:
                        state["L_ema"] = torch.zeros_like(L_batch)
                        state["R_ema"] = torch.zeros_like(R_batch)

                    state["L_ema"].mul_(beta_cov).add_(L_batch, alpha=1.0 - beta_cov)
                    state["R_ema"].mul_(beta_cov).add_(R_batch, alpha=1.0 - beta_cov)

                    bias_corr_cov = 1.0 - (beta_cov ** step_idx)
                    L_hat = state["L_ema"] / bias_corr_cov
                    R_hat = state["R_ema"] / bias_corr_cov

                    if state["W_L"] is None or step_idx % kron_freq == 0:
                        state["W_L"] = compute_soft_kronecker_matrix(
                            L_hat, epsilon=damping, steps=steps_ns, power_iters=power_iters, alpha=alpha
                        )
                        state["W_R"] = compute_soft_kronecker_matrix(
                            R_hat, epsilon=damping, steps=steps_ns, power_iters=power_iters, alpha=alpha
                        )

                    # Apply W_L, W_R to uncentered g_2d so column-mean gradients aren't dropped
                    u_kron_2d = torch.matmul(state["W_L"], torch.matmul(g_2d, state["W_R"]))
                    scale = g_2d.norm() / (u_kron_2d.norm() + 1e-8)
                    u_whitened = (u_kron_2d * scale).view_as(p.data)
                    u_grad_precond = u_whitened * d_half if d_half is not None else u_whitened
                else:
                    u_grad_precond = g_drive * d_half if d_half is not None else g_drive

                # 5. Entropic update
                p_plus_next, p_minus_next, p_data_next = update_entropic_krein_step(
                    p_fp32, p_plus, p_minus, u_grad_precond,
                    lr, gamma, state["std_init"], use_krein_space
                )

                if use_krein_space:
                    p_plus.copy_(p_plus_next)
                    p_minus.copy_(p_minus_next)

                p.data.copy_(p_data_next.to(p.dtype))

        return loss