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

import utils


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
            dirx = W.sum(dim=1, keepdim=True)
            diry = W.sum(dim=0, keepdim=True)
            W = W / (dirx+diry)*0.5
        return W
#todo nuclear norm,
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
            dirx = torch.logsumexp(log_P, dim=-1, keepdim=True)
            diry =  torch.logsumexp(log_P, dim=-2, keepdim=True)
            
            log_P = log_P - (diry+dirx)*0.5
        P = log_P.exp()
        
        # Apply
        return torch.matmul(x, P), P    

class fftLinear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim,dim, bias=False)

    def forward(self, x):
        
        xfft = torch.fft.fft(x, dim=1).abs()
        #xfft = utils.rms(xfft)
        xfft = xfft / (xfft.max(dim=1, keepdim=True)[0] + 1e-6)
        
        return self.lin(xfft) , self.lin.weight
    
class lineargelu(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim,dim, bias=False)
        self.lin2 = nn.Linear(dim,dim, bias=False)

    def forward(self, x):
        x =self.lin(x)
        x =F.gelu(x)
        x =self.lin2(x)
        return x , self.lin.weight
    
class linear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim,dim, bias=False)

    def forward(self, x):
        x =self.lin(x)
        return x , self.lin.weight
def generate_freq_clusters(n_samples=1000, dim=100, n_clusters=3):
    X = np.zeros((n_samples, dim))
    y = np.zeros(n_samples)
    t = np.linspace(0, 4 * np.pi, dim)
    for i in range(n_samples):
        cluster_id = np.random.randint(0, n_clusters)
        y[i] = cluster_id
        
        freqs = [1.0, 2.5, 4.0] 
        freq = freqs[cluster_id]
        
        phase = np.random.uniform(0, 2 * np.pi*0.6)
        
        signal = np.sin(freq * t + phase)
        
        noise = np.random.normal(0, 0.9, dim) 
        
        X[i] = signal + noise

    return torch.FloatTensor(X).to("cuda"), y

def batch_loss(model, batch_x, batch_y, loss_fn):
    projected_x, _ = model(batch_x)
    anchors, positives, negatives = [], [], []
    for i in range(len(batch_x)):
        pos_indices = (batch_y == batch_y[i]).nonzero(as_tuple=True)[0]
        neg_indices = (batch_y != batch_y[i]).nonzero(as_tuple=True)[0]
        
        if len(pos_indices) > 1 and len(neg_indices) > 0:
            p_idx = pos_indices[torch.randint(0, len(pos_indices), (1,)).item()]
            while p_idx == i:
                p_idx = pos_indices[torch.randint(0, len(pos_indices), (1,)).item()]
            
            n_idx = neg_indices[torch.randint(0, len(neg_indices), (1,)).item()]
            
            anchors.append(projected_x[i])
            positives.append(projected_x[p_idx])
            negatives.append(projected_x[n_idx])
    
    if anchors:
        stack_a = torch.stack(anchors)
        stack_p = torch.stack(positives)
        stack_n = torch.stack(negatives)
        return loss_fn(stack_a, stack_p, stack_n)
    else:
        return torch.tensor(0.0, device=batch_x.device, requires_grad=True)

torch.set_float32_matmul_precision('high')
DIM = 100
CLUSTERS = 3
BATCH_SIZE = 256
EPOCHS = 1000
SAMPLES = 1000

print("Training Sinkhorn Layer...")
model = torch.compile(SinkhornLinear2(DIM).to("cuda"))
optimizer = optim.Adam(model.parameters(),betas=[0.85,0.999], lr=0.002)
triplet_loss_fn = nn.TripletMarginLoss(margin=0.5, p=2)

X_val, y_val = generate_freq_clusters(BATCH_SIZE, DIM, CLUSTERS)
X_train, y_train = generate_freq_clusters(SAMPLES, DIM, CLUSTERS)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True) 

train_history = []
val_history = []

for epoch in range(EPOCHS): 
    train_loss = 0
    val_loss = 0
    tbs = 1
    vbs = 1
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        loss = batch_loss(model, batch_x, batch_y, triplet_loss_fn)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        tbs +=1
    
    for batch_x, batch_y in val_loader:
        model.eval()
        with torch.no_grad(): 
            loss = batch_loss(model, batch_x, batch_y, triplet_loss_fn)
        model.train()
        val_loss += loss.item()
        vbs +=1
    val_loss = val_loss / vbs
    train_loss = train_loss / tbs
    val_history.append(val_loss)
    train_history.append(train_loss)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss {train_loss:.4f}, val {val_loss:.4f}")


X_cpu = X_val.cpu().numpy()

with torch.no_grad():
    X_sinkhorn, P_matrix = model(X_val)
    X_sinkhorn = X_sinkhorn.cpu().numpy()

kmeans_sinkhorn = KMeans(n_clusters=CLUSTERS, random_state=42)
y_pred_sinkhorn = kmeans_sinkhorn.fit_predict(X_sinkhorn)
score_sinkhorn = adjusted_rand_score(y_val, y_pred_sinkhorn)

kmeans_raw = KMeans(n_clusters=CLUSTERS, random_state=42)
y_pred_raw = kmeans_raw.fit_predict(X_cpu)
score_raw = adjusted_rand_score(y_val, y_pred_raw)

pca = PCA(n_components=20)
X_pca = pca.fit_transform(X_cpu)
kmeans_pca = KMeans(n_clusters=CLUSTERS, random_state=42)
y_pred_pca = kmeans_pca.fit_predict(X_pca)
score_pca = adjusted_rand_score(y_val, y_pred_pca)

print("\n" + "="*40)
print("ARI Score")
print("="*40)
print(f"1. Raw Euclidean:      {score_raw:.4f}")
print(f"2. PCA:                {score_pca:.4f}")
print(f"3. Learned:            {score_sinkhorn:.4f}")
print("="*40)

plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
sns.heatmap(P_matrix.detach().cpu().numpy()[:30, :30], cmap="inferno", square=True)
plt.title("Learned Matrix (First 30 dims)")

plt.subplot(1, 3, 2)
plt.title("Why Raw Distance Failed\n(Same Cluster, Phase Shifted)")
c0_indices = np.where(y_train == 0)[0]
plt.plot(X_train[c0_indices[0]].cpu().numpy(), label="Sample A", alpha=0.7)
plt.plot(X_train[c0_indices[1]].cpu().numpy(), label="Sample B", alpha=0.7)
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(train_history)
plt.title("Training Loss")

plt.subplot(1, 3, 3)
plt.plot(val_history)
plt.title("validation Loss")
plt.tight_layout()
plt.show()