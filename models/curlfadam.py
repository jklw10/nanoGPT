import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Optimizer

# =========================================================================
# 1. Custom Optimizers
# =========================================================================

class CurlFreeAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, gamma=0.5):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, gamma=gamma)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, (beta1, beta2), eps, wd = group['lr'], group['betas'], group['eps'], group['weight_decay']
            gamma = group['gamma']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                # 1. Decoupled weight decay
                if wd != 0:
                    p.mul_(1.0 - lr * wd)

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1
                step = state['step']
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # 2. Standard Adam moment updates
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).add_(grad.pow(2), alpha=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step

                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                # 3. Square-root preconditioner A^{1/2}
                d_scale = 1.0 / (torch.sqrt(v_hat) + eps)
                d_half = torch.sqrt(d_scale)

                # 4. Unit Gradient Direction u
                g_norm = torch.norm(grad)
                if g_norm > 1e-8:
                    u = grad / g_norm
                    
                    # 5. Orthogonal Momentum Direction v (v ⊥ u)
                    m_parallel = torch.sum(u * m_hat) * u
                    m_ortho = m_hat - m_parallel
                    m_ortho_norm = torch.norm(m_ortho)

                    if m_ortho_norm > 1e-8:
                        v = m_ortho / m_ortho_norm
                        
                        # 6. Contractive Sandwich: H(x) = x - gamma * <v, x> * v
                        x_half = d_half * m_hat
                        dot_v = torch.sum(v * x_half)
                        h_out = x_half - gamma * dot_v * v  # <-- Contractive minus sign
                        jordan_dir = d_half * h_out
                    else:
                        jordan_dir = d_scale * m_hat
                else:
                    jordan_dir = d_scale * m_hat

                # 7. Parameter update
                p.add_(jordan_dir, alpha=-lr)

        return loss


class RowWiseCurlFreeAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, gamma=0.5):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, gamma=gamma)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, (beta1, beta2), eps, wd = group['lr'], group['betas'], group['eps'], group['weight_decay']
            gamma = group['gamma']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                # Decoupled Isotropic Weight Decay
                if wd != 0:
                    p.mul_(1.0 - lr * wd)

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1
                step = state['step']
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # Adam moments
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).add_(grad.pow(2), alpha=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step

                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                d_scale = 1.0 / (torch.sqrt(v_hat) + eps)
                d_half = torch.sqrt(d_scale)

                # =============================================================
                # ROW-WISE (CHANNEL-WISE) CURL-FREE FILTERING
                # =============================================================
                if p.dim() >= 2:
                    # Flatten into (Channels, Features) e.g., (out_channels, -1)
                    g_2d = grad.view(grad.size(0), -1)
                    m_2d = m_hat.view(m_hat.size(0), -1)
                    d_half_2d = d_half.view(d_half.size(0), -1)

                    # 1. Compute per-row unit gradient u_i
                    g_row_norm = torch.norm(g_2d, dim=-1, keepdim=True) + 1e-8
                    u_2d = g_2d / g_row_norm

                    # 2. Compute per-row orthogonal momentum v_i
                    dot_u_m = torch.sum(u_2d * m_2d, dim=-1, keepdim=True)
                    m_parallel = dot_u_m * u_2d
                    m_ortho = m_2d - m_parallel
                    m_ortho_norm = torch.norm(m_ortho, dim=-1, keepdim=True) + 1e-8
                    v_2d = m_ortho / m_ortho_norm

                    # 3. Apply Row-Wise Self-Adjoint Sandwich
                    x_half_2d = d_half_2d * m_2d
                    dot_v_x = torch.sum(v_2d * x_half_2d, dim=-1, keepdim=True)
                    h_out_2d = x_half_2d - gamma * dot_v_x * v_2d
                    jordan_dir = (d_half_2d * h_out_2d).view_as(p)
                else:
                    # 1D Vectors (biases, layernorms) use standard scale
                    jordan_dir = d_scale * m_hat

                # Parameter update
                p.add_(jordan_dir, alpha=-lr)

        return loss

class StandardRMSSGD(Optimizer):
    def __init__(self, params, lr=1e-2, momentum=0.9, eps=1e-8, weight_decay=1e-4):
        defaults = dict(lr=lr, momentum=momentum, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, momentum, eps, wd = group['lr'], group['momentum'], group['eps'], group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                if wd != 0:
                    p.mul_(1 - lr * wd)

                d2 = not (grad.ndim < 2)
                nd = [-1, -2] if d2 else [-1]
                rms = torch.rsqrt(grad.pow(2).mean(dim=nd, keepdim=True) + eps)
                preconditioned_grad = grad * rms

                state = self.state[p]
                if momentum != 0:
                    if 'momentum_buffer' not in state:
                        buf = state['momentum_buffer'] = torch.clone(preconditioned_grad).detach()
                    else:
                        buf = state['momentum_buffer']
                        buf.mul_(momentum).add_(preconditioned_grad)
                    preconditioned_grad = buf

                p.add_(preconditioned_grad, alpha=-lr)

        return loss


class CurlFreeRMSSGD(Optimizer):
    def __init__(self, params, lr=1e-2, momentum=0.9, eps=1e-8, weight_decay=1e-4):
        defaults = dict(lr=lr, momentum=momentum, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, momentum, eps, wd = group['lr'], group['momentum'], group['eps'], group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                if wd != 0:
                    p.mul_(1 - lr * wd)

                if p.ndim >= 2:
                    grad_2d = grad.view(grad.size(0), -1)
                    mean_row_sq = grad_2d.pow(2).mean(dim=1, keepdim=True)
                    mean_col_sq = grad_2d.pow(2).mean(dim=0, keepdim=True)

                    row_scale = mean_row_sq.add(eps).rsqrt().sqrt()
                    col_scale = mean_col_sq.add(eps).rsqrt().sqrt()

                    preconditioned_grad = (row_scale * grad_2d * col_scale).view_as(grad)
                else:
                    mean_sq = grad.pow(2).mean()
                    iso_scale = mean_sq.add(eps).rsqrt()
                    preconditioned_grad = grad * iso_scale

                state = self.state[p]
                if momentum != 0:
                    if 'momentum_buffer' not in state:
                        buf = state['momentum_buffer'] = torch.clone(preconditioned_grad).detach()
                    else:
                        buf = state['momentum_buffer']
                        buf.mul_(momentum).add_(preconditioned_grad)
                    preconditioned_grad = buf

                p.add_(preconditioned_grad, alpha=-lr)

        return loss

# =========================================================================
# 2. CIFAR-100 ConvNet Model
# =========================================================================

class CIFAR100ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 100) # 100 CIFAR-100 Classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # 16x16
        x = self.pool(F.relu(self.conv2(x))) # 8x8
        x = self.pool(F.relu(self.conv3(x))) # 4x4
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def evaluate(model, test_loader, device):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            val_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

    return val_loss / total, (correct / total) * 100.0

def train_cifar100(opt_name, opt_fn, train_loader, test_loader, device, epochs=5):
    torch.manual_seed(42)
    model = CIFAR100ConvNet().to(device).compile()
    optimizer = opt_fn(model.parameters())
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    val_loss, val_acc = evaluate(model, test_loader, device)
    elapsed = time.time() - start_time
    return val_loss, val_acc, elapsed

# =========================================================================
# 3. CIFAR-100 Benchmark Runner
# =========================================================================

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running CIFAR-100 Benchmark on Device: {device}")

    # Standard CIFAR-100 Normalization
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)

    EPOCHS = 5

    benchmark_grid = [
        ("Regular SGD (m=0.9)",   lambda lr: lambda p: torch.optim.SGD(p, lr=lr, momentum=0.9),       [0.01, 0.05]),
        ("Standard RMS-SGD",      lambda lr: lambda p: StandardRMSSGD(p, lr=lr, momentum=0.9),       [0.0003, 0.0008]),
        ("CurlFreeRMSSGD",       lambda lr: lambda p: CurlFreeRMSSGD(p, lr=lr, momentum=0.9),       [0.0003, 0.0008]),
        ("Standard AdamW",       lambda lr: lambda p: torch.optim.AdamW(p, lr=lr),                   [0.001, 0.003]),
        ("CurlFreeAdamW",        lambda lr: lambda p: CurlFreeAdamW(p, lr=lr),                       [0.001, 0.003]),
    ]

    print("\n" + "=" * 85)
    print(f"{'Optimizer':22s} | {'LR':<7s} | {'Val Loss':<10s} | {'Val Acc (%)':<12s} | {'Time (s)':<8s}")
    print("=" * 85)

    best_results = {}

    for opt_name, opt_factory, lr_list in benchmark_grid:
        for lr in lr_list:
            opt_fn = opt_factory(lr)
            val_loss, val_acc, elapsed = train_cifar100(opt_name, opt_fn, train_loader, test_loader, device, epochs=EPOCHS)
            
            print(f"{opt_name:22s} | {lr:<7.4f} | {val_loss:<10.4f} | {val_acc:<12.2f} | {elapsed:<8.2f}")
            
            if opt_name not in best_results or val_acc > best_results[opt_name]['acc']:
                best_results[opt_name] = {'acc': val_acc, 'loss': val_loss, 'lr': lr, 'time': elapsed}

    print("\n" + "=" * 85)
    print("CIFAR-100 SUMMARY: BEST ACCURACY PER OPTIMIZER (5 Epochs)")
    print("=" * 85)
    for opt_name, res in best_results.items():
        print(f"{opt_name:22s} | Best Acc: {res['acc']:6.2f}% | Val Loss: {res['loss']:.4f} (at lr={res['lr']})")