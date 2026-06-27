import torch
from torch.optim import Optimizer
# --- Lean-Verified Metric Functions ---

def K_fn(theta):
    # Softsign kernel
    return theta / (1.0 + torch.abs(theta))

def K_prime_fn(theta):
    # Derivative of softsign
    return 1.0 / (1.0 + torch.abs(theta))**2

class EntropicKFACOptimizer(Optimizer):
    """
    A whole-model hybrid optimizer combining Kronecker-factored Approximate 
    Curvature (K-FAC) for structural weight dependencies with Lean-verified 
    Entropic Natural Gradient Descent for parameter-wise boundary safety.
    """
    def __init__(self, model, lr=1e-3, damping_kfac=1e-3, damping_entropic=1e-5,
                 ema_decay=0.95, invert_interval=10, weight_decay=1e-4):
        self.model = model
        self.damping_kfac = damping_kfac
        self.damping_entropic = damping_entropic
        self.ema_decay = ema_decay
        self.invert_interval = invert_interval
        self.weight_decay = weight_decay
        
        # Track statistics for linear layers
        self.modules = []
        self.stats = {} # Stores covariance matrices: 'A' and 'S'
        self.inverses = {} # Stores inverse covariances: 'A_inv' and 'S_inv'
        self.steps = 0
        
        # 1. Register hooks for all nn.Linear layers
        self._register_hooks()
        
        # 2. Extract parameters for the base Optimizer initialization
        # Separate K-FAC parameters (Linear weights) from non-KFAC parameters
        kfac_params = []
        non_kfac_params = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                kfac_params.append(module.weight)
                if module.bias is not None:
                    non_kfac_params.append(module.bias)
            else:
                for p in module.parameters(recurse=False):
                    non_kfac_params.append(p)
                    
        # Filter out duplicates
        kfac_params = list(set(kfac_params))
        non_kfac_params = list(set(non_kfac_params))
        
        defaults = dict(lr=lr)
        super().__init__([
            {'params': kfac_params, 'is_kfac': True},
            {'params': non_kfac_params, 'is_kfac': False}
        ], defaults)

    def _register_hooks(self):
        """Finds all nn.Linear layers and attaches forward/backward hooks."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                self.modules.append(module)
                self.stats[module] = {'A': None, 'S': None}
                self.inverses[module] = {'A_inv': None, 'S_inv': None}
                
                # Attach forward and backward hooks to dynamically capture covariance
                module.register_forward_hook(self._save_forward_stats)
                module.register_backward_hook(self._save_backward_stats)

    def _save_forward_stats(self, module, input, output):
        """Saves activation covariance A = E[X X^T] using EMA."""
        X = input[0].detach()
        if len(X.shape) > 2:
            # Flatten batch and sequence dimensions for Transformers [batch, seq, features]
            X = X.view(-1, X.shape[-1])
        
        batch_size = X.size(0)
        A_curr = (X.t() @ X) / batch_size
        
        if self.stats[module]['A'] is None:
            self.stats[module]['A'] = A_curr
        else:
            self.stats[module]['A'].mul_(self.ema_decay).add_(A_curr, alpha=1.0 - self.ema_decay)

    def _save_backward_stats(self, module, grad_input, grad_output):
        """Saves gradient covariance S = E[G G^T] using EMA."""
        G = grad_output[0].detach()
        if len(G.shape) > 2:
            G = G.view(-1, G.shape[-1])
            
        batch_size = G.size(0)
        S_curr = (G.t() @ G) / batch_size
        
        if self.stats[module]['S'] is None:
            self.stats[module]['S'] = S_curr
        else:
            self.stats[module]['S'].mul_(self.ema_decay).add_(S_curr, alpha=1.0 - self.ema_decay)

    def _update_inverses(self):
        """Computes inverses of the K-FAC covariance matrices."""
        for module in self.modules:
            A = self.stats[module]['A']
            S = self.stats[module]['S']
            
            if A is None or S is None:
                continue
                
            # A_inv = (A + damping * I)⁻¹
            A_inv = torch.inverse(A + self.damping_kfac * torch.eye(A.size(0), device=A.device))
            # S_inv = (S + damping * I)⁻¹
            S_inv = torch.inverse(S + self.damping_kfac * torch.eye(S.size(0), device=S.device))
            
            self.inverses[module]['A_inv'] = A_inv
            self.inverses[module]['S_inv'] = S_inv

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.steps += 1
        
        # 1. Periodically update the expensive matrix inverses to save compute
        if self.steps == 1 or self.steps % self.invert_interval == 0:
            self._update_inverses()

        for group in self.param_groups:
            lr = group['lr']
            is_kfac = group['is_kfac']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                
                # Apply optional weight decay
                if self.weight_decay != 0:
                    grad = grad.add(p, alpha=self.weight_decay)

                # Find which module this parameter belongs to
                module = None
                if is_kfac:
                    for m in self.modules:
                        if m.weight is p:
                            module = m
                            break

                # 2. Compute Entropic Boundary Scale (Softsign)
                # K(p) = p / (1 + |p|)
                denom_k = p.abs() + 1.0
                K = p / denom_k
                K_prime = 1.0 / denom_k.square()

                one_minus_K2 = 1.0 - K.square()
                num = one_minus_K2.square()
                denom_metric = K_prime.square() + self.damping_entropic * num
                denom_metric = torch.clamp(denom_metric, min=1e-15)
                entropic_scale = num / denom_metric

                # 3. Apply Update Step
                if is_kfac and module is not None:
                    A_inv = self.inverses[module]['A_inv']
                    S_inv = self.inverses[module]['S_inv']
                    
                    if A_inv is not None and S_inv is not None:
                        # Compute K-FAC step: S_inv * grad * A_inv
                        kfac_grad = S_inv @ grad @ A_inv
                        
                        # Apply K-FAC step combined element-wise with Entropic Scaling
                        update = kfac_grad * entropic_scale
                        p.add_(update, alpha=-lr)
                    else:
                        # Fallback if statistics aren't populated yet
                        p.add_(grad * entropic_scale, alpha=-lr)
                else:
                    # Non-KFAC parameters (biases, layernorms, etc.) use 1D Entropic NGD
                    p.add_(grad * entropic_scale, alpha=-lr)

        return loss
class EntropicNGDOptimizer(Optimizer):
    """
    A mathematically rigorous PyTorch optimizer implementing the 
    Lean-verified Damped Natural Gradient update.
    
    Transforms standard gradients using the inverse Fisher Information Metric:
        θ_next = θ - lr * G_d⁻¹ * g
    """
    def __init__(self, params, lr=1e-3, damping=1e-5):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, damping=damping)
        super(EntropicNGDOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            damping = group['damping']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Get the total gradient g computed by PyTorch's autograd
                grad = p.grad
                
                K_val = K_fn(p)
                K_prime_val = K_prime_fn(p)

                # Compute the Lean-verified inverse damped metric:
                # G_d⁻¹ = (1 - K²)² / (K'² + λ * (1 - K²)²)
                numerator = (1.0 - K_val**2)**2
                denominator = K_prime_val**2 + damping * (1.0 - K_val**2)**2
                
                # Prevent numerical underflow at the absolute boundary
                denominator = torch.clamp(denominator, min=1e-15)
                
                # Project the gradient onto the natural Riemannian manifold
                natural_grad = (numerator / denominator) * grad
                
                # Update parameter: θ = θ - lr * natural_grad
                p.add_(natural_grad, alpha=-lr)

        return loss


# --- Running the Tracking Task ---

# 1. Initialize parameter θ near 0
theta = torch.tensor([0.1], requires_grad=True)

# 2. Instantiate the Lean-verified NGD Optimizer
optimizer = EntropicNGDOptimizer([theta], lr=0.05, damping=1e-5)
#optimizer = torch.optim.SGD([theta], lr=0.05)

# 3. Define the Environment / Task
target = torch.tensor([0.9])
gamma = 0.05  # Weight of the entropic potential (restoring force)

print("--- Starting Corrected Optimization Loop ---")
for step in range(101):
    optimizer.zero_grad()
    
    K_val = K_fn(theta)
    
    # Task Loss: Minimize tracking error
    task_loss = 0.5 * (K_val - target)**2
    
    # Entropic Loss: V(θ) = -γ * log(1 - K²) (minimizing this maximizes entropy)
    entropic_loss = -gamma * torch.log(1.0 - K_val**2 + 1e-15)
    
    # Total Objective
    total_loss = task_loss + entropic_loss
    
    # Backpropagation computes the total gradient g = grad_task + grad_entropic
    total_loss.backward()
    
    # Optimizer projects the total gradient through the inverse Fisher Metric
    optimizer.step()
    
    if step % 10 == 0:
        print(f"Step {step:02d} | θ: {theta.item():.4f} | K(θ): {K_fn(theta).item():.4f} | Total Loss: {total_loss.item():.4f}")
