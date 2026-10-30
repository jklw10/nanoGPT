import torch

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import model
from parts import utils

class MockConfig:
    """Mock configuration to instantiate KreinAttention."""
    n_embd = 64      # Embedding dimension
    n_head = 4       # Number of heads
    dropout = 0.0    # Set to 0.0 for deterministic gradient checking
    bias = True      # Linear bias
    pos_dim = 16     # Krein coordinate dimension
    delta = 1e-4
    tau = 0.01
    init_c = 1.0

class MetricSimplexAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        B, T, C = x.size()
        
        # 1. Project and unit-normalize to force dot products into [-1, 1]
        # This maps tokens onto the Unit Sphere S^(d-1)
        q = F.normalize(self.q_proj(x), p=2, dim=-1) # (B, T, C)
        k = F.normalize(self.k_proj(x), p=2, dim=-1) # (B, T, C)
        v = self.v_proj(x)                           # (B, T, C)
        
        # 2. Compute cosine similarities (guaranteed to be in [-1, 1])
        cos_sim = torch.bmm(q, k.transpose(1, 2))    # (B, T, T)
        
        # 3. Apply the Entropic Potential mapping from your Lean code:
        # Rather than Softmax, we scale the similarities using the 
        # non-singular entropic restoring force: x / (1 - x^2)
        # We add a small delta for numerical safety near the boundary
        delta = 1e-4
        # --- Step 3: Apply the Entropic Potential mapping ---
        gated_similarities = cos_sim / (1.0 - cos_sim**2 + delta)

        # --- Step 4: Convert to a non-negative matrix via Softplus ---
        positive_similarities = F.softplus(gated_similarities)

        # --- NEW: Apply Causal Mask after Softplus by zeroing out future tokens ---
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        # Zero out future similarities (B, T, T)
        positive_similarities = positive_similarities.masked_fill(causal_mask.unsqueeze(0), 0.0)

        # --- Step 5: Normalize (row sums will only aggregate past and current tokens) ---
        row_sums = positive_similarities.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        attn_weights = positive_similarities / row_sums
        
        # 5. Compute output
        out = torch.bmm(attn_weights, v)
        return out
    
class LinearCausalMetricAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        B, T, C = x.size()
        delta = 1e-4
        
        # 1. Project and unit-normalize
        q_raw = F.normalize(self.q_proj(x), p=2, dim=-1) # (B, T, C)
        k_raw = F.normalize(self.k_proj(x), p=2, dim=-1) # (B, T, C)
        v = self.v_proj(x)                               # (B, T, C)
        
        # 2. Entropic potential mapping on the raw vectors
        # (Translates the physical potential to individual token manifolds)
        q_gated = q_raw / (1.0 - q_raw**2 + delta)
        k_gated = k_raw / (1.0 - k_raw**2 + delta)
        
        # 3. Project onto non-negative space
        q_prime = F.softplus(q_gated) # (B, T, C)
        k_prime = F.softplus(k_gated) # (B, T, C)
        
        # 4. --- LINEAR CAUSAL SUMS USING CUMSUM ---
        # Numerator state: Outer product sum of (key * value) over the time dimension
        # Shape: (B, T, C, C)
        kv_outer = torch.einsum('bti,btj->btij', k_prime, v)
        S = torch.cumsum(kv_outer, dim=1) # Cumulative sum over T (causal history)
        
        # Denominator state: Cumulative sum of keys over the time dimension
        # Shape: (B, T, C)
        Z = torch.cumsum(k_prime, dim=1)  # Causal key accumulation
        
        # 5. Contract the Query with the Causal History
        # Numerator: query @ S -> (B, T, C)
        numerator = torch.einsum('bti,btij->btj', q_prime, S)
        
        # Denominator: query . Z -> (B, T, 1)
        denominator = torch.einsum('bti,bti->bt', q_prime, Z).unsqueeze(-1)
        
        # 6. Final Simplex Output
        out = numerator / denominator.clamp(min=1e-9)
        return out
def verify_attention_causality():
    # 1. Initialize config and layer
    config = MockConfig()
    # Replace 'KreinAttention' with your actual class name if different
    attention_layer = MetricSimplexAttention(config.n_embd)
    
    # Crucial: The leak was in Step 7 (which only runs during training)
    attention_layer.train() 
    
    # 2. Create input tensor x with shape (B, T, C) and enable gradients
    B, T, C = 1, 8, config.n_embd
    x = torch.randn(B, T, C, requires_grad=True)
    
    # 3. Forward pass through attention layer
    output = attention_layer(x)
    
    # 4. Compute loss on ONLY the very first token's output (step 0)
    loss = output[0, 0].sum()
    loss.backward()
    
    # 5. Check if gradients flowed to future inputs (indices 1 to T-1)
    # Under strict causality, x.grad for indices >= 1 must be exactly zero.
    future_gradients = x.grad[0, 1:].abs().sum().item()
    
    print("--- Causality Test ---")
    if future_gradients > 0.0:
        print(f"❌ LEAK DETECTED: Gradient sum from future tokens: {future_gradients:.4e}")
    else:
        print("✅ CAUSALLY SECURE: Gradients do not flow from the future.")

# Run the test
verify_attention_causality()