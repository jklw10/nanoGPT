import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import seaborn as sns


class SinkhornLinear2(nn.Module):
    def __init__(self, size, n_iter=10, eps=1e-6, project=True):
        super().__init__()
        self.size = size
        self.n_iter = n_iter
        self.eps = eps
        self.project_on_run = project
        self.weight = nn.Parameter(torch.eye(size))
        
        with torch.no_grad():
            self.weight.add_(torch.randn(size, size) * 1e-5)
            
    def forward(self, x):
        #if self.project_on_run:
        #    with torch.no_grad():
        #        self.weight.copy_(self.project())
        W = self.project()
        #W = self.weight
        return F.linear(x, W), W
    
    def project(self):
        W = self.weight.clamp(min=1e-8)
        for _ in range(self.n_iter):
            W = W / W.sum(dim=1, keepdim=True)
            W = W / W.sum(dim=0, keepdim=True)
        return W
    
class fftSinkhornLinear(nn.Module):
    def __init__(self, dim, n_iters=5, temperature=1.0):
        super().__init__()
        self.n_iters = n_iters
        self.temperature = temperature # Controls how "sharp" the matrix is allowed to be
        
        self.log_weight = nn.Parameter(torch.eye(dim)+torch.randn(dim, dim))

    def forward(self, x):
        
        xfft = torch.fft.fft(x, dim=1).abs()
        xfft = xfft / (xfft.max(dim=1, keepdim=True)[0] + 1e-6)
        x = xfft
        log_P = self.log_weight / self.temperature

        # Sinkhorn Iterations (Normalization in Log-Space)
        for _ in range(self.n_iters):
            # Row Norm: log(P) - log(sum(P, row))
            log_P = log_P - torch.logsumexp(log_P, dim=-1, keepdim=True)
            # Col Norm: log(P) - log(sum(P, col))
            log_P = log_P - torch.logsumexp(log_P, dim=-2, keepdim=True)

        # Exponentiate to get the final Doubly Stochastic Matrix
        P = log_P.exp()
        
        # Apply
        return torch.matmul(x, P), P    
# ==========================================
# 2. HELL MODE DATASET: Phase-Shifted Waves
# ==========================================
def generate_hell_mode_data(n_samples=1000, dim=100, n_clusters=3):
    """
    Cluster 0: Low Frequency Sine Wave
    Cluster 1: Medium Frequency Sine Wave
    Cluster 2: High Frequency Sine Wave
    
    THE TRAP: Random Phase Shifts. 
    If you shift a sine wave, Euclidean distance thinks it's a totally different vector.
    """
    X = np.zeros((n_samples, dim))
    y = np.zeros(n_samples)
    t = np.linspace(0, 4 * np.pi, dim) # Time axis
    
    for i in range(n_samples):
        cluster_id = np.random.randint(0, n_clusters)
        y[i] = cluster_id
        
        # Define Frequencies
        freqs = [1.0, 2.5, 4.0] 
        freq = freqs[cluster_id]
        
        # RANDOM PHASE SHIFT (The Euclidean Killer)
        # We shift the wave randomly. 
        # A shift of pi causes correlation to go from +1 to -1.
        phase = np.random.uniform(0, 2 * np.pi)
        
        signal = np.sin(freq * t + phase)
        
        # Add Heavy Noise
        # Noise level 1.0 means noise is as strong as signal
        noise = np.random.normal(0, 0.9, dim) 
        
        X[i] = signal + noise

    return torch.FloatTensor(X).to("cuda"), y

# ==========================================
# 3. Training & Evaluation
# ==========================================
torch.set_float32_matmul_precision('high')
DIM = 100
CLUSTERS = 3
BATCH_SIZE = 256
EPOCHS = 100
SAMPLES = 1000
print(f"Generating Hell Mode Data (Dim={DIM})...")
X, y_true = generate_hell_mode_data(SAMPLES, DIM, CLUSTERS)

# --- Metric 1: Standard K-Means ---
print("Running Raw K-Means...")
kmeans_raw = KMeans(n_clusters=CLUSTERS, random_state=42)
y_pred_raw = kmeans_raw.fit_predict(X.cpu().numpy())
score_raw = adjusted_rand_score(y_true, y_pred_raw)

# --- Metric 2: PCA ---
print("Running PCA + K-Means...")
pca = PCA(n_components=20)
X_pca = pca.fit_transform(X.cpu().numpy())
kmeans_pca = KMeans(n_clusters=CLUSTERS, random_state=42)
y_pred_pca = kmeans_pca.fit_predict(X_pca)
score_pca = adjusted_rand_score(y_true, y_pred_pca)

#X_fft = torch.fft.rfft(X, dim=1).abs()
## Normalize to keep math stable
#X_fft = X_fft / (X_fft.max(dim=1, keepdim=True)[0] + 1e-6)
## Update DIM to FFT size (usually dim/2 + 1)
#DIM = X_fft.shape[1] 
#X = X_fft
# --- Metric 3: Learned Sinkhorn ---
print("Training Sinkhorn Layer...")
model = torch.compile(SinkhornLinear(DIM).to("cuda"))
optimizer = optim.Adam(model.parameters(),betas=[0.85,0.999], lr=0.002)
triplet_loss_fn = nn.TripletMarginLoss(margin=0.5, p=2)

# Use a DataLoader
dataset = torch.utils.data.TensorDataset(X, torch.LongTensor(y_true))
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

loss_history = []

for epoch in range(EPOCHS): 
    total_loss = 0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        projected_x, _ = model(batch_x)
        
        # HARD MINING (Simple version)
        # For every item, find a positive (same cluster) and negative (diff cluster)
        anchors, positives, negatives = [], [], []
        
        for i in range(len(batch_x)):
            # Find indices of same class
            pos_indices = (batch_y == batch_y[i]).nonzero(as_tuple=True)[0]
            # Find indices of different class
            neg_indices = (batch_y != batch_y[i]).nonzero(as_tuple=True)[0]
            
            if len(pos_indices) > 1 and len(neg_indices) > 0:
                # Pick random positive that isn't self
                p_idx = pos_indices[torch.randint(0, len(pos_indices), (1,)).item()]
                while p_idx == i:
                    p_idx = pos_indices[torch.randint(0, len(pos_indices), (1,)).item()]
                
                n_idx = neg_indices[torch.randint(0, len(neg_indices), (1,)).item()]
                
                anchors.append(projected_x[i])
                positives.append(projected_x[p_idx])
                negatives.append(projected_x[n_idx])
        
        if anchors:
            loss = triplet_loss_fn(torch.stack(anchors), torch.stack(positives), torch.stack(negatives))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    loss_history.append(total_loss)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss {total_loss:.4f}")

with torch.no_grad():
    X_sinkhorn, P_matrix = model(X)
    X_sinkhorn = X_sinkhorn.cpu().numpy()

kmeans_sinkhorn = KMeans(n_clusters=CLUSTERS, random_state=42)
y_pred_sinkhorn = kmeans_sinkhorn.fit_predict(X_sinkhorn)
score_sinkhorn = adjusted_rand_score(y_true, y_pred_sinkhorn)

# ==========================================
# 4. Visualization
# ==========================================
print("\n" + "="*40)
print(f"HELL MODE RESULTS (ARI Score)")
print("="*40)
print(f"1. Raw Euclidean:      {score_raw:.4f} (Should be near 0)")
print(f"2. PCA:                {score_pca:.4f}")
print(f"3. Sinkhorn Learned:   {score_sinkhorn:.4f}")
print("="*40)

plt.figure(figsize=(12, 5))

# Plot 1: The Heatmap
plt.subplot(1, 3, 1)
# Focus on the diagonal to see the "smearing"
sns.heatmap(P_matrix.detach().cpu().numpy()[:30, :30], cmap="inferno", square=True)
plt.title("Learned Matrix (First 30 dims)\nLook for diagonal broadening")

# Plot 2: Why Euclidean Failed
plt.subplot(1, 3, 2)
plt.title("Why Raw Distance Failed\n(Same Cluster, Phase Shifted)")
# Plot two samples from Cluster 0
c0_indices = np.where(y_true == 0)[0]
plt.plot(X[c0_indices[0]].cpu().numpy(), label="Sample A", alpha=0.7)
plt.plot(X[c0_indices[1]].cpu().numpy(), label="Sample B", alpha=0.7)
plt.legend()

# Plot 3: Loss
plt.subplot(1, 3, 3)
plt.plot(loss_history)
plt.title("Training Loss")

plt.tight_layout()
plt.show()