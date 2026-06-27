import torch
import torch.nn as nn
import math
import time

# --- 1. Original Newton-Schulz (with O(N^3) Residual GEMM) ---
@torch.compile(backend='inductor', mode='max-autotune')
def NS_inverse_update_original(A_reg, V_old, steps=3, power_iters=5):
    device, dtype = A_reg.device, A_reg.dtype
    d = A_reg.size(0)
    I = torch.eye(d, device=device, dtype=dtype)
    
    # Estimate spectral norm of A_reg
    v = torch.ones(d, 1, device=device, dtype=dtype)
    for _ in range(power_iters):
        v = torch.matmul(A_reg, v)
        v = v / (v.norm() + 1e-15)
    s = torch.matmul(v.T, torch.matmul(A_reg, v)).squeeze()
    
    # Check convergence of warm-start (O(N^3) GEMM)
    R = I - torch.matmul(A_reg, V_old)
    v_r = torch.ones(d, 1, device=device, dtype=dtype)
    for _ in range(power_iters):
        v_r = torch.matmul(R, v_r)
        v_r = v_r / (v_r.norm() + 1e-15)
    rho_R = torch.matmul(R, v_r).norm().squeeze()
    
    is_safe = (rho_R < 0.95) & (~torch.isnan(rho_R)) & (~torch.isinf(rho_R))
    V0_fallback = I / (s + 1e-15)
    V = torch.where(is_safe, V_old, V0_fallback)
    
    for _ in range(steps):
        V = torch.matmul(V, 2 * I - torch.matmul(A_reg, V))
    return V

# --- 2. Optimized Newton-Schulz (Zero GEMMs in Safety Check, O(N^2) Vector Iteration) ---
@torch.compile(backend='inductor', mode='max-autotune')
def NS_inverse_update_optimized(A_reg, V_old, steps=2, power_iters=3):
    device, dtype = A_reg.device, A_reg.dtype
    d = A_reg.size(0)
    I = torch.eye(d, device=device, dtype=dtype)
    
    # Estimate spectral norm of A_reg in O(N^2)
    v = torch.ones(d, 1, device=device, dtype=dtype)
    for _ in range(power_iters):
        v = torch.matmul(A_reg, v)
        v = v / (v.norm() + 1e-15)
    s = torch.matmul(v.T, torch.matmul(A_reg, v)).squeeze()
    
    # Check convergence in O(N^2) by avoiding R matrix creation
    # R x = x - A_reg @ (V_old @ x)
    v_r = torch.ones(d, 1, device=device, dtype=dtype)
    for _ in range(power_iters):
        v_r = v_r - torch.matmul(A_reg, torch.matmul(V_old, v_r))
        v_r = v_r / (v_r.norm() + 1e-15)
    
    rho_R = (v_r - torch.matmul(A_reg, torch.matmul(V_old, v_r))).norm().squeeze()
    
    is_safe = (rho_R < 0.95) & (~torch.isnan(rho_R)) & (~torch.isinf(rho_R))
    V0_fallback = I / (s + 1e-15)
    V = torch.where(is_safe, V_old, V0_fallback)
    
    # Only computation with O(N^3) GEMMs is the refinement itself
    for _ in range(steps):
        V = torch.matmul(V, 2 * I - torch.matmul(A_reg, V))
    return V


# --- 3. Optimizer Class ---
class EntropicKFACOptimizer(torch.optim.Optimizer):
    def __init__(self, model, params=None, lr=1e-3, damping_kfac=1e-3, use_optimized=False):
        self.model = model
        self.damping_kfac = damping_kfac
        self.use_optimized = use_optimized
        self.modules = []
        self.stats = {}
        self.inverses = {}
        self.steps = 0
        
        self.param_to_module = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                self.param_to_module[module.weight] = module
        
        self._register_hooks()
        if params is None:
            params = [p for p in self.model.parameters() if p.requires_grad]
        super().__init__(params, dict(lr=lr))

    def _register_hooks(self):
        for module in self.param_to_module.values():
            self.modules.append(module)
            out_features, in_features = module.weight.shape
            device = module.weight.device
            dtype = module.weight.dtype
            
            self.stats[module] = {
                'A': torch.zeros(in_features, in_features, device=device, dtype=dtype),
                'S': torch.zeros(out_features, out_features, device=device, dtype=dtype),
                'initialized': torch.tensor(0.0, device=device)
            }
            self.inverses[module] = {
                'A_inv': torch.eye(in_features, device=device, dtype=dtype) / self.damping_kfac,
                'S_inv': torch.eye(out_features, device=device, dtype=dtype) / self.damping_kfac
            }
            module.register_forward_hook(self._save_forward_stats)
            module.register_full_backward_hook(self._save_backward_stats)

    def _save_forward_stats(self, module, input, output):
        X = input[0].detach()
        if len(X.shape) > 2: X = X.view(-1, X.shape[-1])
        A_curr = (X.t() @ X) / X.size(0)
        self.stats[module]['A'].copy_(A_curr)

    def _save_backward_stats(self, module, grad_input, grad_output):
        G = grad_output[0].detach()
        if len(G.shape) > 2: G = G.view(-1, G.shape[-1])
        S_curr = (G.t() @ G) / G.size(0)
        self.stats[module]['S'].copy_(S_curr)

    def _update_inverses(self):
        for module in self.modules:
            A = self.stats[module]['A']
            S = self.stats[module]['S']
            
            dim_A = A.size(0)
            dim_S = S.size(0)
            
            A_reg = A + self.damping_kfac * torch.eye(dim_A, dtype=A.dtype, device=A.device)
            S_reg = S + self.damping_kfac * torch.eye(dim_S, dtype=S.dtype, device=S.device)

            A_inv_old = self.inverses[module]['A_inv']
            S_inv_old = self.inverses[module]['S_inv']
            
            if self.use_optimized:
                A_inv = NS_inverse_update_optimized(A_reg, A_inv_old, steps=1, power_iters=1).clone()
                S_inv = NS_inverse_update_optimized(S_reg, S_inv_old, steps=1, power_iters=1).clone()
            else:
                A_inv = NS_inverse_update_original(A_reg, A_inv_old, steps=3, power_iters=5).clone()
                S_inv = NS_inverse_update_original(S_reg, S_inv_old, steps=3, power_iters=5).clone()
            
            self.inverses[module]['A_inv'] = A_inv
            self.inverses[module]['S_inv'] = S_inv

    @torch.no_grad()
    def step(self):
        self.steps += 1
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                module = self.param_to_module.get(p, None)
                if module is not None:
                    A_inv = self.inverses[module]['A_inv']
                    S_inv = self.inverses[module]['S_inv']
                    g_eff = S_inv @ p.grad @ A_inv
                    p.add_(g_eff, alpha=-group['lr'])


# --- 4. Timing Harness ---
def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # MLP with 4 layers (generating shapes of 1024x1024 and 512x512)
    model = nn.Sequential(
        nn.Linear(1024, 1024), nn.ReLU(),
        nn.Linear(1024, 1024), nn.ReLU(),
        nn.Linear(1024, 1024), nn.ReLU(),
        nn.Linear(1024, 512)
    ).to(device)

    x = torch.randn(256, 1024, device=device)
    y = torch.randn(256, 512, device=device)
    criterion = nn.MSELoss()

    optimizer = EntropicKFACOptimizer(model, lr=1e-3, use_optimized=False)

    print("="*60)
    print("STRICT WARM-UP PHASE (Baking Compiler & Instantiating Kernels)")
    print("="*60)
    
    # 1. Warm up the forward/backward/step path 15 times
    for _ in range(15):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
    # 2. Warm up both compilation paths of the Inversion Step explicitly
    print("Pre-compiling and warming up 'Original NS Inversion'...")
    optimizer.use_optimized = False
    for _ in range(15):
        optimizer._update_inverses()
        
    print("Pre-compiling and warming up 'Optimized NS Inversion'...")
    optimizer.use_optimized = True
    for _ in range(15):
        optimizer._update_inverses()
        
    torch.cuda.synchronize()
    print("Warm-up complete. All kernels are 100% compiled.\n")

    # --- BENCHMARKING ACTIVE STEP ---
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    steps = 50

    # A. Time Forward + Backward Pass
    start_evt.record()
    for _ in range(steps):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
    end_evt.record()
    torch.cuda.synchronize()
    fwd_bwd_time = start_evt.elapsed_time(end_evt) / steps

    # B. Time Warmed-Up Original Matrix Inversion
    optimizer.use_optimized = False
    start_evt.record()
    for _ in range(steps):
        optimizer._update_inverses()
    end_evt.record()
    torch.cuda.synchronize()
    inv_time_original = start_evt.elapsed_time(end_evt) / steps

    # C. Time Warmed-Up Optimized Matrix Inversion
    optimizer.use_optimized = True
    start_evt.record()
    for _ in range(steps):
        optimizer._update_inverses()
    end_evt.record()
    torch.cuda.synchronize()
    inv_time_optimized = start_evt.elapsed_time(end_evt) / steps

    # D. Time Optimizer Parameter Step
    # Populate gradients once
    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    
    start_evt.record()
    for _ in range(steps):
        optimizer.step()
    end_evt.record()
    torch.cuda.synchronize()
    step_time = start_evt.elapsed_time(end_evt) / steps

    print("="*60)
    print("           TRUE WARMED-UP GPU EXECUTION SPEED           ")
    print("="*60)
    print(f"Forward + Backward Pass (with hooks): {fwd_bwd_time:10.3f} ms")
    print(f"Original Inversion (steps=3, O(N^3)): {inv_time_original:10.3f} ms")
    print(f"Optimized Inversion (steps=2, O(N^2)):{inv_time_optimized:10.3f} ms")
    print(f"Optimizer Parameter Step:             {step_time:10.3f} ms")
    print("="*60)

if __name__ == "__main__":
    run_benchmark()