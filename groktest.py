import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import numpy as np

import modules
import utils
torch._inductor.config.disable_cpp_codegen = True

HungryLinear = modules.HungryLinear
ProbingLinear = modules.ProbingLinear

class Model(nn.Module):
    def __init__(self, p, d_model, is_hungry=False):
        super().__init__()
        self.embedding = nn.Embedding(p, d_model)
        self.p = p
        
        in_dim = 2 * d_model
        hidden_dim = 128
        
        if is_hungry:
            # Zero Init is the ultimate test for your Spark Plug
            self.fc1 = ProbingLinear(in_dim, hidden_dim)
            
            self.fc2 = ProbingLinear(hidden_dim, p)
            self.act = utils.synthrelu
        else:
            self.fc1 = nn.Linear(in_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, p)
            self.act = utils.synthrelu
            

    def forward(self, x):
        # x: [Batch, 2] containing integers (a, b)
        embeds = self.embedding(x) # [Batch, 2, d_model]
        flat = embeds.view(embeds.size(0), -1) # [Batch, 2*d_model]
        
        out = self.fc1(flat)
        out = self.act(out)
        out = self.fc2(out)
        return out # Logits over p classes

# ==========================================
# 3. Data & Training Harness
# ==========================================
def run_experiment():
    # Settings
    P = 97       # Prime number for modulo
    TRAIN_FRAC = 0.5
    EPOCHS = 3000 
    LR = 1e-3
    WD = 1.0     # High weight decay is crucial for Grokking
    SEED = 42

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # --- Generate Data (a + b) % P ---
    pairs = [(i, j) for i in range(P) for j in range(P)]
    labels = [(i + j) % P for i, j in pairs]
    
    data_idx = list(range(len(pairs)))
    random.shuffle(data_idx)
    cut = int(len(data_idx) * TRAIN_FRAC)
    
    train_idx = data_idx[:cut]
    test_idx = data_idx[cut:]
    
    X = torch.tensor(pairs)
    Y = torch.tensor(labels)
    
    # --- Helper to Train one Model ---
    def train_model(model_name, model):
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        criterion = nn.CrossEntropyLoss()
        
        history = {'train_acc': [], 'test_acc': [], 'train_loss': [], 'test_loss': []}
        
        pbar = tqdm(range(EPOCHS), desc=model_name)
        for epoch in pbar:
            torch.compiler.cudagraph_mark_step_begin()
            # Train
            model.train()
            optimizer.zero_grad()
            logits = model(X[train_idx])
            loss = criterion(logits, Y[train_idx])
            loss.backward()
            optimizer.step()
            
            # Record Metrics
            if epoch % 10 == 0:
                with torch.no_grad():
                    model.eval()
                    # Train metrics
                    pred_train = logits.argmax(dim=1)
                    acc_train = (pred_train == Y[train_idx]).float().mean().item()

                    # Test metrics
                    test_logits = model(X[test_idx])
                    test_loss = criterion(test_logits, Y[test_idx]).item()
                    pred_test = test_logits.argmax(dim=1)
                    acc_test = (pred_test == Y[test_idx]).float().mean().item()

                    history['train_loss'].append(loss.item())
                    history['train_acc'].append(acc_train)
                    history['test_loss'].append(test_loss)
                    history['test_acc'].append(acc_test)

                    if epoch % 100 == 0:
                        pbar.set_postfix({'T-Acc': f"{acc_test:.2f}", 'Loss': f"{loss.item():.4f}"})
                    
        return history

    # --- Run Models ---
    print("\nTraining Hungry (Zero Init + Spark Plug)...")
    hungry_model = torch.compile(Model(P, d_model=32, is_hungry=True))
    hungry_hist = train_model("Hungry", hungry_model)
    
    print("Training Baseline (Standard Linear)...")
    baseline_model = torch.compile(Model(P, d_model=32, is_hungry=False))
    base_hist = train_model("Baseline", baseline_model)
    

    # ==========================================
    # 4. Visualization
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy Plot
    ax1.plot(base_hist['train_acc'], label='Base Train', color='blue', alpha=0.3)
    ax1.plot(base_hist['test_acc'], label='Base Test', color='blue')
    ax1.plot(hungry_hist['train_acc'], label='Hungry Train', color='red', alpha=0.3)
    ax1.plot(hungry_hist['test_acc'], label='Hungry Test', color='red')
    ax1.set_title(f"Accuracy (Grokking = Late Test Jump)")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss Plot (Log Scale)
    ax2.plot(base_hist['train_loss'], label='Base Train', color='blue', alpha=0.3)
    ax2.plot(base_hist['test_loss'], label='Base Test', color='blue')
    ax2.plot(hungry_hist['train_loss'], label='Hungry Train', color='red', alpha=0.3)
    ax2.plot(hungry_hist['test_loss'], label='Hungry Test', color='red')
    ax2.set_title("Loss (Log Scale)")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Cross Entropy")
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Grokking: Standard vs Hungry Linear (P={P})")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()