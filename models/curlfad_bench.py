import torch


@torch.compile(backend='inductor', options={"triton.cudagraphs": False})
def update_entropic_krein_step(
    p: torch.Tensor,
    p_plus: torch.Tensor,
    p_minus: torch.Tensor,
    u_grad_precond: torch.Tensor,
    lr: float,
    gamma: float,
    scale_ref: float,
    use_krein_space: bool
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

    else:
        p_abs = torch.abs(p)
        x_norm = torch.clamp(p_abs / R_t, min=eps_geom, max=0.99)
        K_val = x_norm / (1.0 + x_norm)
        K_prime = 1.0 / (1.0 + x_norm).square()

        force_direct = torch.sign(p) * gamma * (K_val * K_prime) / (1.0 - K_val.square() + eps_geom)
        p_data_next = p - lr * (u_grad_precond + force_direct)

        return p_plus, p_minus, p_data_next


@torch.compile(backend='inductor', options={"triton.cudagraphs": False})
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
    W_soft = alpha * W_hard + (1.0 - alpha) * eye_d
    
    # Enforce strict self-adjoint symmetry down to float precision
    W_soft = 0.5 * (W_soft + W_soft.T)
    return W_soft


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
        gamma_jordan: float = 0.5,
        damping: float = 1e-5,
        weight_decay: float = 1e-2,
        max_kron_dim: int = 2048,
        kron_freq: int = 10,
        # Toggles
        use_adam: bool = True,
        use_kron: bool = False,
        use_krein_space: bool = True,
        use_grms: bool = True,
        use_curl_free_jordan: bool = True,
        use_nesterov: bool = True,  # Enables Nesterov look-ahead across all EMAs
    ):
        defaults = {
            'lr': lr, 'beta1': beta1, 'beta2': beta2, 'beta_cov': beta_cov, 'alpha': alpha,
            'eps': eps, 'gamma': gamma, 'gamma_jordan': gamma_jordan, 'damping': damping,
            'weight_decay': weight_decay, 'max_kron_dim': max_kron_dim, 'kron_freq': kron_freq,
            'use_adam': use_adam, 'use_kron': use_kron,
            'use_krein_space': use_krein_space, 'use_grms': use_grms,
            'use_curl_free_jordan': use_curl_free_jordan,
            'use_nesterov': use_nesterov
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            beta_cov = group['beta_cov']
            alpha = group['alpha']
            eps = group['eps']
            gamma = group['gamma']
            gamma_jordan = group['gamma_jordan']
            damping = group['damping']
            weight_decay = group['weight_decay']
            max_kron_dim = group['max_kron_dim']
            kron_freq = group['kron_freq']
            
            use_adam = group['use_adam']
            use_kron = group['use_kron']
            use_krein_space = group['use_krein_space']
            use_grms = group['use_grms']
            use_curl_free_jordan = group['use_curl_free_jordan']
            use_nesterov = group['use_nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                orig_shape = p.shape
                is_multidim = p.dim() > 1

                if is_multidim:
                    m = orig_shape[0]
                    n = p.numel() // m
                else:
                    m, n = p.numel(), 1

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data, dtype=torch.float32)
                    state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=torch.float32)

                    state['p_plus'] = torch.clamp(p.data, min=0.0).add_(1e-4)
                    state['p_minus'] = torch.clamp(-p.data, min=0.0).add_(1e-4)
                    state['std_init'] = float(torch.nan_to_num(torch.std(p.data), nan=1e-4).item())

                    state['grms_scale_ema'] = 1.0
                    state['kron_scale_ema'] = 1.0

                    state['L_ema'] = None
                    state['R_ema'] = None
                    state['W_L'] = None
                    state['W_R'] = None

                state['step'] += 1
                step_idx = state['step']

                p_plus = state['p_plus']
                p_minus = state['p_minus']

                # Decoupled Weight Decay
                if weight_decay != 0:
                    decay_factor = 1.0 - lr * weight_decay
                    p.data.mul_(decay_factor)
                    if use_krein_space:
                        p_plus.mul_(decay_factor)
                        p_minus.mul_(decay_factor)

                grad_fp32 = grad.to(torch.float32)

                # =============================================================
                # 1. GRMS NORMALIZATION (With optional Nesterov Look-Ahead)
                # =============================================================
                if use_grms:
                    d2 = not (grad_fp32.ndim < 2)
                    nd = [-1, -2] if d2 else [-1]
                    rms_inst = torch.rsqrt(grad_fp32.pow(2).mean(dim=nd, keepdim=True) + 1e-8)
                    
                    if step_idx == 1:
                        state['grms_scale_ema'] = rms_inst
                    else:
                        state['grms_scale_ema'].mul_(0.9).add_(rms_inst, alpha=0.1)

                    if use_nesterov:
                        # Nesterov look-ahead scale prediction
                        grms_scale_eff = 0.9 * state['grms_scale_ema'] + 0.1 * rms_inst
                    else:
                        grms_scale_eff = state['grms_scale_ema']

                    grad_fp32 = grad_fp32 * grms_scale_eff

                exp_avg = state['exp_avg']

                # =============================================================
                # 2. FIRST MOMENT (Nadam Look-Ahead Momentum)
                # =============================================================
                if beta1 > 0:
                    exp_avg.mul_(beta1).add_(grad_fp32, alpha=1.0 - beta1)
                    bias_correction1 = 1.0 - (beta1 ** step_idx)

                    if use_nesterov:
                        # Nadam look-ahead momentum: beta1 * exp_avg + (1 - beta1) * grad
                        m_lookahead = beta1 * exp_avg + (1.0 - beta1) * grad_fp32
                        m_hat = m_lookahead / bias_correction1
                    else:
                        m_hat = exp_avg / bias_correction1
                else:
                    m_hat = grad_fp32

                # =============================================================
                # 3. SECOND MOMENT (Nesterov Variance Look-Ahead)
                # =============================================================
                if use_adam:
                    exp_avg_sq = state['exp_avg_sq']
                    grad_sq = grad_fp32.pow(2)
                    exp_avg_sq.mul_(beta2).add_(grad_sq, alpha=1.0 - beta2)
                    bias_correction2 = 1.0 - (beta2 ** step_idx)

                    if use_nesterov:
                        # Nesterov look-ahead variance prediction
                        v_lookahead = beta2 * exp_avg_sq + (1.0 - beta2) * grad_sq
                        v_hat = v_lookahead / bias_correction2
                    else:
                        v_hat = exp_avg_sq / bias_correction2

                    denom = torch.sqrt(v_hat) + eps
                    d_half = denom.rsqrt()          # A^(1/2) = v^(-1/4)
                else:
                    d_half = torch.ones_like(m_hat)

                # =============================================================
                # 4. ROW-WISE (CHANNEL-WISE) JORDAN FILTERING
                # =============================================================
                if is_multidim and use_curl_free_jordan and p.dim() >= 2:
                    g_2d = grad_fp32.view(m, n)
                    m_2d = m_hat.view(m, n)
                    d_half_2d = d_half.view(m, n)

                    g_row_norm = torch.norm(g_2d, dim=-1, keepdim=True) + 1e-8
                    u_2d = g_2d / g_row_norm

                    dot_u_m = torch.sum(u_2d * m_2d, dim=-1, keepdim=True)
                    m_parallel = dot_u_m * u_2d
                    m_ortho = m_2d - m_parallel
                    m_ortho_norm = torch.norm(m_ortho, dim=-1, keepdim=True) + 1e-8
                    v_2d = m_ortho / m_ortho_norm

                    x_half_2d = d_half_2d * m_2d
                    dot_v_x = torch.sum(v_2d * x_half_2d, dim=-1, keepdim=True)
                    h_out_2d = x_half_2d - gamma_jordan * dot_v_x * v_2d
                    g_drive = h_out_2d.view_as(p.data)
                else:
                    g_drive = m_hat * d_half

                # =============================================================
                # 5. KRONECKER ZCA (Nesterov Look-Ahead Covariance)
                # =============================================================
                if use_kron and is_multidim and m <= max_kron_dim and n <= max_kron_dim:
                    g_drive_2d = g_drive.view(m, n)

                    L_batch = (1.0 / n) * torch.matmul(g_drive_2d, g_drive_2d.T)
                    R_batch = (1.0 / m) * torch.matmul(g_drive_2d.T, g_drive_2d)

                    if state['L_ema'] is None:
                        state['L_ema'] = torch.zeros_like(L_batch)
                        state['R_ema'] = torch.zeros_like(R_batch)

                    state['L_ema'].mul_(beta_cov).add_(L_batch, alpha=1.0 - beta_cov)
                    state['R_ema'].mul_(beta_cov).add_(R_batch, alpha=1.0 - beta_cov)

                    bias_corr_cov = 1.0 - (beta_cov ** step_idx)

                    if use_nesterov:
                        # Nesterov look-ahead covariance matrices
                        L_lookahead = beta_cov * state['L_ema'] + (1.0 - beta_cov) * L_batch
                        R_lookahead = beta_cov * state['R_ema'] + (1.0 - beta_cov) * R_batch
                        L_hat = L_lookahead / bias_corr_cov
                        R_hat = R_lookahead / bias_corr_cov
                    else:
                        L_hat = state['L_ema'] / bias_corr_cov
                        R_hat = state['R_ema'] / bias_corr_cov

                    if state['W_L'] is None or step_idx % kron_freq == 0:
                        state['W_L'] = compute_soft_kronecker_matrix(
                            L_hat, epsilon=damping, alpha=alpha
                        )
                        state['W_R'] = compute_soft_kronecker_matrix(
                            R_hat, epsilon=damping, alpha=alpha
                        )

                    u_kron_2d = torch.matmul(state['W_L'], torch.matmul(g_drive_2d, state['W_R']))
                    
                    scale_inst = g_drive_2d.norm() / (u_kron_2d.norm() + 1e-8)
                    if step_idx == 1:
                        state['kron_scale_ema'] = scale_inst
                    else:
                        state['kron_scale_ema'].mul_(0.9).add_(scale_inst, alpha=0.1)

                    if use_nesterov:
                        kron_scale_eff = 0.9 * state['kron_scale_ema'] + 0.1 * scale_inst
                    else:
                        kron_scale_eff = state['kron_scale_ema']

                    u_whitened = (u_kron_2d * kron_scale_eff).view_as(p.data)
                    u_grad_precond = u_whitened * d_half
                else:
                    u_grad_precond = g_drive * d_half

                # =============================================================
                # 6. ENTROPIC KREIN STEP
                # =============================================================
                p_plus_next, p_minus_next, p_data_next = update_entropic_krein_step(
                    p.data, p_plus, p_minus, u_grad_precond,
                    lr, gamma, state['std_init'], use_krein_space
                )

                if use_krein_space:
                    p_plus.copy_(p_plus_next)
                    p_minus.copy_(p_minus_next)

                p.data.copy_(p_data_next.to(p.dtype))

        return loss