import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import time

# Set reproducible seeds
torch.manual_seed(42)

# ==============================================================================
# 1. SYNTHETIC 2D DATA GENERATION (Caustic Cusp + Waves + Heavy Shot Noise)
# ==============================================================================
H, W = 128, 128
grid_y, grid_x = torch.meshgrid(
    torch.linspace(0.0, 1.0, H), torch.linspace(0.0, 1.0, W), indexing="ij"
)
grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0) # [1, 2, H, W]

# Ground Truth: Smooth Wave + Sharp Caustic Focal Point at center (0.5, 0.5)
y_smooth = torch.sin(2.0 * math.pi * grid_x) * torch.cos(2.0 * math.pi * grid_y)
dist_to_center = torch.sqrt((grid_x - 0.5)**2 + (grid_y - 0.5)**2)
y_cusp = 2.5 * torch.exp(-dist_to_center / 0.03)

y_clean = (y_smooth + y_cusp).unsqueeze(0).unsqueeze(0) # [1, 1, H, W]

# Add Heavy Shot Noise
noise = torch.randn_like(y_clean) * 0.35
y_noisy = y_clean + noise

# ==============================================================================
# 2. 2D FITTING ARCHITECTURE
# ==============================================================================
class FittingNet2D(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU()
        )
        self.out_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, grid):
        embed = self.stem(grid)
        y_pred = self.out_head(embed)
        return y_pred, embed

# ==============================================================================
# 3. METRIC / LOSS FUNCTION DEFINITIONS
# ==============================================================================

# --- Helper: 2D Gaussian Kernel ---
def gaussian_kernel2d(kernel_size=11, sigma=1.5):
    coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    g2d = g.unsqueeze(1) @ g.unsqueeze(0)
    return g2d.unsqueeze(0).unsqueeze(0)

# 1. Standard MSE Loss (L2)
class MSELoss2D(nn.Module):
    def forward(self, pred, target):
        return F.mse_loss(pred, target)

# 2. L4 Loss
class L4Loss2D(nn.Module):
    def forward(self, pred, target):
        return torch.mean((pred - target) ** 4)

# 3. SSIM Loss (Pure PyTorch)
class SSIMLoss2D(nn.Module):
    def __init__(self, kernel_size=11, sigma=1.5):
        super().__init__()
        kernel = gaussian_kernel2d(kernel_size, sigma)
        self.register_buffer('kernel', kernel)
        self.c1 = (0.01 * 3.5) ** 2
        self.c2 = (0.03 * 3.5) ** 2

    def forward(self, img1, img2):
        channels = img1.shape[1]
        kernel = self.kernel.repeat(channels, 1, 1, 1)
        
        mu1 = F.conv2d(img1, kernel, padding=5, groups=channels)
        mu2 = F.conv2d(img2, kernel, padding=5, groups=channels)
        
        mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, kernel, padding=5, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, kernel, padding=5, groups=channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, kernel, padding=5, groups=channels) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + self.c1) * (2 * sigma12 + self.c2)) / \
                   ((mu1_sq + mu2_sq + self.c1) * (sigma1_sq + sigma2_sq + self.c2))
        return 1.0 - ssim_map.mean()

# 4. Perceptual CNN Feature Loss (LPIPS-like)
class PerceptualLoss2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat_net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GELU()
        )
        for param in self.feat_net.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        f_pred = self.feat_net(pred)
        f_target = self.feat_net(target)
        return F.mse_loss(f_pred, f_target) + F.l1_loss(pred, target)

# 5. NVIDIA FLIP-inspired Metric
class FLIPLoss2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.lp_kernel = gaussian_kernel2d(7, 1.0)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred, target):
        pred_lp = F.conv2d(pred, self.lp_kernel, padding=3)
        target_lp = F.conv2d(target, self.lp_kernel, padding=3)
        l1_smooth = torch.abs(pred_lp - target_lp)
        
        pred_gx = F.conv2d(pred, self.sobel_x, padding=1)
        pred_gy = F.conv2d(pred, self.sobel_y, padding=1)
        target_gx = F.conv2d(target, self.sobel_x, padding=1)
        target_gy = F.conv2d(target, self.sobel_y, padding=1)
        
        grad_diff = torch.abs(torch.sqrt(pred_gx**2 + pred_gy**2 + 1e-6) - 
                              torch.sqrt(target_gx**2 + target_gy**2 + 1e-6))
        
        return (0.6 * l1_smooth + 0.4 * grad_diff).mean()

# 6. Learned WNorm Predictability Loss Head (WNL 2D)
wn = torch.nn.utils.parametrizations.weight_norm
class WNormLearnLoss2D(nn.Module):
    def __init__(self, embed_dim=32):
        super().__init__()
        self.head = nn.Sequential(
            wn(nn.Conv2d(embed_dim, embed_dim, 1, bias=False)),
            nn.GELU(),
            wn(nn.Conv2d(embed_dim, 1, 1, bias=False))
        )

    def forward(self, embed):
        return torch.tanh(self.head(embed))

# 7. Spatial Structure Coherence Loss (Analytic Learnability)
class StructureCoherenceLoss2D(nn.Module):
    def __init__(self, kernel_size=7, gamma=2.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.gamma = gamma
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def compute_coherence(self, target):
        Ix = F.conv2d(target, self.sobel_x, padding=1)
        Iy = F.conv2d(target, self.sobel_y, padding=1)
        J11, J22, J12 = Ix ** 2, Iy ** 2, Ix * Iy
        
        pad = self.kernel_size // 2
        J11 = F.avg_pool2d(J11, self.kernel_size, stride=1, padding=pad)
        J22 = F.avg_pool2d(J22, self.kernel_size, stride=1, padding=pad)
        J12 = F.avg_pool2d(J12, self.kernel_size, stride=1, padding=pad)
        
        trace = J11 + J22
        det_term = torch.sqrt((J11 - J22) ** 2 + 4.0 * J12 ** 2 + 1e-8)
        l1, l2 = 0.5 * (trace + det_term), 0.5 * (trace - det_term)
        coherence = ((l1 - l2) / (l1 + l2 + 1e-6)) ** self.gamma
        return coherence

    def forward(self, pred, target):
        raw_error = (pred - target) ** 2
        with torch.no_grad():
            coherence = self.compute_coherence(target)
        weight = 0.2 + 0.8 * coherence
        return (raw_error * weight).mean()


class NonLearningWNLLoss2D(nn.Module):
    """
    Non-Learning (Deterministic) Approximation of Weight-Norm Predictability Loss.
    Requires 0 trainable parameters and 0 gradient updates for the loss head.
    """
    def __init__(self, kernel_size=7, sigma=1.5, lambda_edge=3.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.lambda_edge = lambda_edge
        
        # Gaussian Kernel for Spatial Pooling
        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        g2d = (g.unsqueeze(1) @ g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        self.register_buffer('g2d', g2d)
        
        # Sobel Filters for Structural Gradient
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, y_pred, y_noisy):
        pad = self.kernel_size // 2
        
        # 1. Smooth noisy target to estimate signal structure
        y_smooth = F.conv2d(y_noisy, self.g2d, padding=pad)
        
        # 2. Local Residual Variance Map
        raw_sq_err = (y_pred - y_noisy) ** 2
        v_res = F.conv2d(raw_sq_err, self.g2d, padding=pad)
        
        # 3. Structural Gradient Strength on Smooth Target
        gx = F.conv2d(y_smooth, self.sobel_x, padding=1)
        gy = F.conv2d(y_smooth, self.sobel_y, padding=1)
        edge_mag = torch.sqrt(gx**2 + gy**2 + 1e-6)
        
        # 4. Noise Probability Score (Suppress edge regions from being marked as noise)
        noise_score = v_res / (1.0 + self.lambda_edge * edge_mag)
        
        # 5. Standardize & Map to [-1, 1] range via Tanh
        mean_n = noise_score.mean()
        std_n = noise_score.std() + 1e-6
        p_score = torch.tanh((noise_score - mean_n) / std_n)
        
        # 6. Chaos Factor Calculation
        chaos = ((1.0 - p_score.detach()) / 2.0).mean() * 0.99
        
        # 7. Modulated Loss
        modulated_loss = (raw_sq_err / (1.0 + p_score * chaos)).mean()
        return modulated_loss
# ==============================================================================
# 4. BENCHMARK EXECUTION SUITE
# ==============================================================================
loss_configs = [
    ("nlwnl", NonLearningWNLLoss2D(), 0.005, False),
    ("Standard MSE (L2)", MSELoss2D(), 0.005, False),
    ("L4 Loss", L4Loss2D(), 0.002, False),
    ("SSIM Loss", SSIMLoss2D(), 0.005, False),
    ("Perceptual Feature Loss", PerceptualLoss2D(), 0.005, False),
    ("Simplified FLIP Loss", FLIPLoss2D(), 0.005, False),
    ("Learned WNL Head 2D", WNormLearnLoss2D(), 0.005, True),
    ("Structure Coherence (Analytic)", StructureCoherenceLoss2D(), 0.005, False)
]

results = []
epochs = 1000
eval_ssim = SSIMLoss2D()

print(f"=== Starting 2D Benchmark across {len(loss_configs)} Loss Functions ({epochs} Epochs Each) ===\n")

for name, loss_fn, lr, is_wnl in loss_configs:
    model = FittingNet2D(hidden_dim=32)
    
    if is_wnl:
        opt = optim.Adam(list(model.parameters()) + list(loss_fn.parameters()), lr=lr)
    else:
        opt = optim.Adam(model.parameters(), lr=lr)
        
    start_time = time.time()
    
    for epoch in range(epochs):
        opt.zero_grad()
        y_pred, embed = model(grid)
        
        if is_wnl:
            raw_error = (y_pred - y_noisy) ** 2
            wnl = loss_fn(embed)
            chaos = ((1.0 - wnl.detach()) / 2.0).mean() * 0.99
            loss = (raw_error / (1.0 + wnl * chaos)).mean()
        else:
            loss = loss_fn(y_pred, y_noisy)
            
        loss.backward()
        opt.step()
        
    elapsed = time.time() - start_time
    
    # --- EVALUATION AGAINST GROUND TRUTH (y_clean) ---
    with torch.no_grad():
        y_pred_clean, _ = model(grid)
        
        # 1. Overall GT MSE & PSNR
        gt_mse = F.mse_loss(y_pred_clean, y_clean).item()
        gt_psnr = 10.0 * math.log10((3.5 ** 2) / (gt_mse + 1e-8))
        
        # 2. GT SSIM
        gt_ssim = 1.0 - eval_ssim(y_pred_clean, y_clean).item()
        
        # 3. Cusp Peak Error at center (64, 64)
        cusp_err = (y_pred_clean[0, 0, 64, 64] - y_clean[0, 0, 64, 64]).abs().item()
        
        # 4. Smooth Region Ripple (Corner area X < 0.3, Y < 0.3)
        smooth_mask = (grid_x < 0.3) & (grid_y < 0.3)
        ripple = (y_pred_clean[0, 0][smooth_mask] - y_clean[0, 0][smooth_mask]).std().item()
        
        results.append((name, gt_mse, gt_psnr, gt_ssim, cusp_err, ripple, elapsed))

# ==============================================================================
# 5. DISPLAY RESULTS TABLE
# ==============================================================================
print("="*115)
print(f"{'LOSS FUNCTION / METRIC':<32} | {'GT MSE (↓)':<12} | {'GT PSNR (↑)':<12} | {'GT SSIM (↑)':<12} | {'Cusp Err (↓)':<12} | {'Ripple (↓)':<12}")
print("="*115)
for name, gt_mse, gt_psnr, gt_ssim, cusp_err, ripple, elapsed in results:
    print(f"{name:<32} | {gt_mse:<12.6f} | {gt_psnr:<12.2f} | {gt_ssim:<12.4f} | {cusp_err:<12.6f} | {ripple:<12.6f}")
print("="*115)