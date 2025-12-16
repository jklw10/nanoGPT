import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GraphAttention(nn.Module):
    def __init__(self, config, k_sparse=64):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // self.n_head
        self.k_sparse = k_sparse

        # --- Projections ---
        # We can reuse projections if desired, but separate ones are clearer
        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        # --- Scorers (Shared across heads) ---
        # This is the key change: scorers now produce 1 score per token.
        self.query_scorer = nn.Linear(config.n_embd, 1)
        self.key_scorer = nn.Linear(config.n_embd, 1)

    def forward(self, x):
        B, T, C = x.size()
        k_val = min(self.k_sparse, T)
        
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        query_scores = self.query_scorer(x).squeeze(-1) # Shape: (B, T)
        
        # 2. Create and apply causal mask before selection
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        masked_scores_b = query_scores.unsqueeze(1).masked_fill(~causal_mask, float('-inf'))
        
        # 3. Select and sanitize top-k VIP query indices
        _, top_q_indices = torch.topk(masked_scores_b, k=k_val, dim=-1) # Shape: (B, T, k_val)
        query_positions = torch.arange(T, device=x.device).view(1, T, 1)
        top_q_indices = torch.min(top_q_indices, query_positions) # Sanitize

        # 4. Gather VIP queries efficiently
        # Reshape for gathering: (B, nh, T, hs) -> (B, T, nh*hs)
        q_reshaped = q.transpose(1, 2).reshape(B, T, C)
        # Gather along the sequence dimension (dim=1)
        sparse_q_flat = torch.gather(q_reshaped, 1, top_q_indices.unsqueeze(-1).expand(-1, -1, -1, C))
        # Reshape back to include head dimension
        sparse_q = sparse_q_flat.view(B, T, k_val, self.n_head, self.head_dim).permute(0, 3, 1, 2, 4)

        # 5. VIP queries attend to all keys (with causal masking)
        attn_logits_b = (sparse_q @ k.unsqueeze(2).transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        key_pos = torch.arange(T, device=x.device).view(1,1,1,1,T)
        causal_mask_attn_b = top_q_indices.unsqueeze(1).unsqueeze(-1) >= key_pos
        attn_logits_b = attn_logits_b.masked_fill(~causal_mask_attn_b, float('-inf'))
      
        key_scores = self.key_scorer(x).squeeze(-1) # Shape: (B, T)
        
        # 2. Apply causal mask and find top-k VIP key indices
        masked_scores_g = key_scores.unsqueeze(1).masked_fill(~causal_mask, float('-inf'))
        _, top_k_indices = torch.topk(masked_scores_g, k=k_val, dim=-1) # Shape: (B, T, k_val)
        top_k_indices = torch.min(top_k_indices, query_positions) # Sanitize

        # 3. *** Efficient Gather Operation ***
        # Prepare indices for gathering across heads and head_dim
        # Final index shape: (B, nh, T, k_val, hs)
        expanded_indices = top_k_indices.unsqueeze(1).unsqueeze(-1).expand(-1, self.n_head, -1, -1, self.head_dim)
        
        # Gather from k and v tensors. We gather from dim=2 (the sequence dimension)
        # Note: No need to expand the large k and v tensors anymore!
        sparse_k = torch.gather(k.unsqueeze(3).expand(-1,-1,-1,k_val,-1), 2, expanded_indices) # This still requires some expansion, let's optimize further
        
        # We can avoid expanding k and v by indexing manually.
        # But a more pytorch-native way without large expansions:
        B, nh, T, hs = q.shape
        final_indices = top_k_indices.unsqueeze(1).expand(-1, nh, -1, -1) # Shape: (B, nh, T, k_val)
        final_indices = final_indices.unsqueeze(-1).expand(-1, -1, -1, -1, hs) # Shape: (B, nh, T, k_val, hs)

        sparse_k = torch.gather(k, 2, final_indices[:,:,:,:,0]) # Simplified gather
        
        indices_expanded_g = top_k_indices.unsqueeze(1).unsqueeze(-1).expand(-1, self.n_head, -1, -1, self.head_dim)
        
        k_source_expanded = k.unsqueeze(2).expand(-1,-1,T,-1,-1) # Expand for each query
        v_source_expanded = v.unsqueeze(2).expand(-1,-1,T,-1,-1)

        sparse_k = torch.gather(k_source_expanded, 3, indices_expanded_g)
        sparse_v = torch.gather(v_source_expanded, 3, indices_expanded_g)

        # 5. Perform dot-product attention
        q_for_attn = q.unsqueeze(3) # (B, nh, T, 1, hs)
        attn_logits_g = (q_for_attn @ sparse_k.transpose(-2, -1)).squeeze(3)
        attn_logits_g = attn_logits_g * (1.0 / math.sqrt(self.head_dim))

        # 6. Final softmax and value application
        attn_probs_g = F.softmax(attn_logits_g, dim=-1)
        output = (attn_probs_g.unsqueeze(3) @ sparse_v).squeeze(3)
        output = output.transpose(1, 2).contiguous().view(x.shape)
        
        return self.c_proj(output)
        
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.n_embd))
        self.bias = nn.Parameter(torch.zeros(config.n_embd)) if config.bias else None
       
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
class GraphFormerBlock(nn.Module):
    def __init__(self, config, k_sparse=8):
        super().__init__()
        self.ln_1 = LayerNorm(config)
        self.ln_2 = LayerNorm(config)
        
        self.graph_attn = GraphAttention(config, k_sparse=k_sparse)

        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        x = x + self.graph_attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x)) # Final FFN
        
        return x