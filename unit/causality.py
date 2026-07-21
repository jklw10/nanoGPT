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

def verify_attention_causality():
    # 1. Initialize config and layer
    config = MockConfig()
    # Replace 'KreinAttention' with your actual class name if different
    attention_layer = model.KreinAttention(config)
    
    # Crucial: The leak was in Step 7 (which only runs during training)
    attention_layer.train() 
    
    # 2. Create input tensor x with shape (B, T, C) and enable gradients
    B, T, C = 1, 8, config.n_embd
    x = torch.randn(B, T, C, requires_grad=True)
    
    # 3. Forward pass through attention layer
    output = attention_layer(x, causal=True)
    
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