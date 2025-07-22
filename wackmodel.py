
import math
import model
import torch
import torch.nn as nn
from torch.nn import functional as F
import utils

from torch.optim import Optimizer
from typing import Iterable

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int):
    """
    Quintic Newton–Schulz orthogonalization of G.
    Works only if G.ndim>=2; otherwise returns G unchanged.
    """
    if G.ndim < 2:
        return G
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.bfloat16)
    transposed = False
    # always do (XX^T)^{-1/2}
    if X.size(-2) > X.size(-1):
        X = X.transpose(-2, -1)
        transposed = True
    X = X / (X.norm(dim=(-2,-1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.transpose(-2,-1)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.transpose(-2, -1)
    return X.to(G.dtype)

def muon_update(grad: torch.Tensor,
                buf: torch.Tensor,
                beta: float,
                ns_steps: int,
                nesterov: bool,
                ):
    """
    Muon-style momentum + NS orthogonalization of grad.
    - buf ← β·buf + (1−β)·grad
    - U = β·buf + (1−β)·grad  (if nesterov) or buf
    - quintic NS on matrix-shaped U
    - only apply magnitude-correction if grad.ndim>=2
    """
    # momentum buffer update
    buf.lerp_(grad, 1 - beta)
    # pick Nesterov vs plain
    U = grad.lerp(buf, beta) if nesterov else buf

    orig_shape = U.shape
    if U.ndim > 2:
        U_flat = U.view(U.size(0), -1)
    else:
        U_flat = U

    # orthogonalize if possible
    #U_flat = zeropower_via_newtonschulz5(U_flat, steps=ns_steps)
    U_flat = zeropower_via_newtonschulz5(U_flat, steps=ns_steps)

    # reshape back
    if U.ndim > 2:
        U = U_flat.view(orig_shape)
    else:
        U = U_flat
    #U = utils.zca_newton_schulz(U,1e-10,2,2)worse.
    # magnitude correction only for matrices or higher
    if grad.ndim >= 2:
        scale = max(1.0, grad.size(-2) / grad.size(-1)) ** 0.5
    else:
        scale = 1.0
    return U * scale

def orthograd_tangent(W: torch.Tensor, G: torch.Tensor):
    """
    Gram–Schmidt projection of G into the tangent at W on the sphere/Stiefel manifold.
    Works for vectors (ndim==1) or matrices (ndim==2).
    """
    w = W.view(-1)
    g = G.view(-1)
    w_dot_w = w.dot(w) + 1e-30
    proj    = w.dot(g) / w_dot_w
    p_flat  = g - proj * w
    return p_flat.view_as(G)

class Phi2Optimizer(Optimizer):
    """
    Full Phi2 optimizer:
      • AdamW preconditioning
      • Midpoint two-step lookahead
      • OrthoGrad projection on every grad
      • Muon momentum + quintic NS
    """

    def __init__(self,
                 params: Iterable[torch.nn.Parameter],
                 lr: float = 3e-4,
                 betas=(0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.01,
                 two_step_alpha: float = 0.5,
                 muon_beta: float = 0.95,
                 muon_nesterov: bool = True,
                 ns_steps: int = 5,
                 fused: bool = False):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        two_step_alpha=two_step_alpha,
                        muon_beta=muon_beta,
                        muon_nesterov=muon_nesterov,
                        ns_steps=ns_steps)
        super().__init__(params, defaults)

        # initialize all state entries to avoid KeyErrors
        for group in self.param_groups:
            for p in group['params']:
                st = self.state[p]
                st['step']       = 0
                st['exp_avg']    = torch.zeros_like(p)
                st['exp_avg_sq'] = torch.zeros_like(p)
                st['muon_buf']   = torch.zeros_like(p)
                st['P0']         = torch.zeros_like(p)
                st['P']          = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure):
        """
        1) closure @ W_t  → AdamW + OrthoGrad → st['P0']
        2) W_half = W_t - lr*P0
        3) closure @ W_half → OrthoGrad → combine → st['P']
        4) U = muon_update(P, muon_buf)
        5) W_{t+1} = W_t - lr*U
        """
        assert closure is not None, "Phi2Optimizer requires closure()"

        # ——— 1) first forward/backward at W_t ———
        with torch.enable_grad():
            loss = closure()

        # AdamW + OrthoGrad → P0
        for group in self.param_groups:
            β1, β2 = group['betas']
            wd      = group['weight_decay']
            lr      = group['lr']
            eps     = group['eps']

            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                st = self.state[p]
                st['step'] += 1
                t = st['step']

                # decoupled weight decay
                if wd:
                    p.mul_(1 - lr * wd)

                # Adam moments
                st['exp_avg'].mul_(β1).add_(g, alpha=1-β1)
                st['exp_avg_sq'].mul_(β2).addcmul_(g, g, value=1-β2)

                # bias-corrected direction
                bc1 = 1 - β1**t
                bc2 = 1 - β2**t
                denom = (st['exp_avg_sq'].sqrt() / math.sqrt(bc2)).add_(eps)
                base_step = (st['exp_avg'] / bc1).div_(denom)

                # project
                P0 = orthograd_tangent(p, base_step)
                st['P0'].copy_(P0)

        # ——— 2) half-step to W_half ———
        W_t = {}
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                W_t[p] = p.data.clone()
                p.data.copy_(W_t[p] - lr * self.state[p]['P0'])

        # ——— 3) second forward/backward at W_half ———
        self.zero_grad()
        with torch.enable_grad():
            loss = closure()

        # combine P = (1-α)*P0 + α*P1
        for group in self.param_groups:
            α = group['two_step_alpha']
            for p in group['params']:
                st = self.state[p]
                P  = st['P0'].clone()
                g1 = p.grad
                if g1 is not None:
                    P1 = orthograd_tangent(p, g1)
                    P  = (1-α)*st['P0'] + α*P1
                st['P'].copy_(P)

        # ——— 4) Muon + final update ———
        for group in self.param_groups:
            lr    = group['lr']
            wd    = group['weight_decay']
            mb    = group['muon_beta']
            nest  = group['muon_nesterov']
            ns    = group['ns_steps']

            for p in group['params']:
                st = self.state[p]
                U  = muon_update(st['P'], st['muon_buf'], mb, ns, nest)

                # final weight decay
                if wd:
                    p.mul_(1 - lr * wd)

                # apply step
                p.data.copy_(p.data - lr * U)

        return loss

#garbage heap file:
class PatchEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_patch = config.patch_max
        self.n_embd = config.n_embd
        
        # Learnable pooling weights
        self.pool_proj = nn.Linear(config.n_embd, 3)  # For mean/max/attention
        self.mlp = nn.Sequential(
            nn.Linear(3 * config.n_embd, config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, config.n_embd)
        )
        
    def forward(self, x, lengths):
        # x: (B, T, D), lengths: (B,)
        
        # Dynamic pooling with learned weights
        weights = torch.softmax(self.pool_proj(x), dim=-1)  # (B, T, 3)
        
        # Mask out padding
        mask = torch.arange(x.size(1), device=x.device)[None] < lengths[:, None]
        mask = mask.float().unsqueeze(-1)  # (B, T, 1)
        
        # Pooled features
        mean_pool = (x * mask * weights[..., 0]).sum(1) / (lengths[:, None] + 1e-6)
        max_pool = (x * mask * weights[..., 1]).max(1).values
        attn_pool = (x * mask * weights[..., 2]).sum(1)
        
        combined = torch.cat([mean_pool, max_pool, attn_pool], dim=-1)
        return self.mlp(combined)
class uBlock(nn.Module):

    def __init__(self, config, d):
        super().__init__()
        self.ln_1 = model.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = Softmaxless_attention(config,d)
        self.ln_2 = model.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = model.MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        #x = x + self.attn(x)
        #x = x + self.mlp(x)
        #x = mmnorm(x)
        return x

class CausalConv1d2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = ((kernel_size - 1) * dilation, 0) 
        
        self.conv1d = nn.Conv1d(in_channels,out_channels, kernel_size, **kwargs)

    def forward(self, seq):
        return self.pseudopatcher(F.pad(seq,self.padding)).contiguous()

class QrotMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_fc2   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.n_embd = config.n_embd

    def forward(self, x, y, xnorm=False, ynorm=True):
        xb,xt,xn=x.shape
        yb,yt,yn=y.shape
        x = self.c_fc(x)
        y = self.c_fc2(y)
        
        x4d = x.view(xb, xt, xn, 4) #(batch, seq_len, n_embd, 4)
        if(xnorm):
            x_norm = x4d.norm(dim=-1, keepdim=True) 
            x_normalized = x4d / x_norm             
        else:
            x_normalized = x4d
        
        y4d = y.view(yb, yt, yn, 4) # (batch, seq_len, n_embd, 4)
        if(ynorm):
            y_norm = y4d.norm(dim=-1, keepdim=True) 
            y_normalized = y4d / y_norm             
        else:
            y_normalized = y4d


        rotors = x_normalized
        rotated = y_normalized
        
        x_rotated = utils.quaternion_multiply(rotors, rotated) 
        x = x_rotated.view(xb, xt, 4 * xn)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
class Softmaxless_attention(nn.Module):

    def __init__(self, config, d):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head 
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.head_dim =  self.n_embd //  self.n_head

        self.qm = nn.Parameter(torch.ones(self.head_dim).normal_(mean=1, std=0.1))
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))


    
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        q = q * (math.log(T)*self.qm)
        # Split q and k for the diff part
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float(0))
        att = att / att.norm()
        att = self.attn_dropout(att)
        y = (att @ v)
        #y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.splits * self.head_dim) # re-assemble all head outputs side by side

        #y = mmnorm(y)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class Patcher(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ksize = 8
        self.patch_max = 10
        self.bembwidth = config.n_embd//2
        self.threshold= nn.Parameter(torch.ones(1))
        self.end_patch_token_id = config.vocab_size-1
        
        self.pseudopatcher = nn.Conv1d(self.bembwidth, self.bembwidth, self.ksize, stride=1, bias=False)
        self.padding = (self.ksize-1, 0) 
        
        self.embedder = nn.Sequential(
            nn.Linear(config.n_embd*self.patch_max, config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, config.n_embd)
        ) #eBlock(config,1)

    def loss(self, logits, patchtargets, pploss):
        logits = logits.view(
            logits.size(0), 
            logits.size(1), 
            self.patch_max, 
            self.config.vocab_size
        )   # (b, t_internal, patch_max, vocab_size)
        logits_flat = logits.contiguous().view(-1, self.config.vocab_size)  # (b * t_internal * patch_max, vocab_size)
        targets_flat = patchtargets.contiguous().view(-1)  # (b * t_internal * patch_max)
        loss = F.cross_entropy(
            logits_flat, 
            targets_flat,
            ignore_index=-1
        )
        return loss + pploss.mean()/(self.threshold/self.patch_max) #+ length_penalty
                    

    def forward(self, idx):
          #torch.compiler.cudagraph_mark_step_begin() # Add this line at the beginning of forward
        device = idx.device
        b, t = idx.size()

        device = idx.device
        b, t = idx.shape

        pos = torch.arange(0, self.patch_max, dtype=torch.long, device=device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx).permute(0, 2, 1).contiguous()
        pred_idx = self.pseudopatcher(F.pad(tok_emb[:, :self.bembwidth, :], self.padding))#.permute(0, 2, 1).contiguous()
        losses = F.mse_loss(tok_emb[:, :self.bembwidth, :-1], pred_idx[:, :, 1:], reduction='none').mean(dim=1)#.permute(0, 2, 1).contiguous()  
        
        #tok_emb = tok_emb.permute(0, 2, 1).contiguous()
        
        endtok= torch.as_tensor(self.end_patch_token_id,device=device,dtype=torch.long)
        patch_indices = torch.full((b, self.config.internal_block_size+2, self.patch_max+1), -1, dtype=torch.long, device=device)
        
        patch_embeds = torch.zeros((b, self.config.internal_block_size+2, self.patch_max, self.config.n_embd), device=device)
        accumulated_loss = torch.zeros(b, device=device)
        patch_deposit_idx = torch.zeros(b, dtype=torch.long, device=device)
        patch_length_idx = torch.zeros(b, dtype=torch.long, device=device)
        batch_indices = torch.arange(b, device=device)  # Explicit batch indices
        

        for current_token_index in range(t-1): 
            loss_value = losses[:, current_token_index]
            accumulated_loss = accumulated_loss + loss_value 
            cond1 = torch.gt(accumulated_loss,self.threshold  )
            cond2 = torch.ge(patch_length_idx,self.patch_max-1)
            mask = cond1 | cond2
            #if loss is greater, or max length is reached, start setting values into the next patch
            patch_deposit_idx = patch_deposit_idx + (mask) 
            #if patch doesn't end, increment patch length else reset patch length
            patch_length_idx = patch_length_idx + (~mask)
            patch_length_idx = patch_length_idx * (~mask)
            #if patch ends, reset patch accumulated loss
            accumulated_loss = accumulated_loss * (~mask)  

            patch_indices[batch_indices,patch_deposit_idx, patch_length_idx] = idx[batch_indices, current_token_index]
            patch_embeds[batch_indices, patch_deposit_idx, patch_length_idx, :] = tok_emb[batch_indices, :, current_token_index]
            
            patch_indices[batch_indices,patch_deposit_idx, patch_length_idx+1] = endtok
        
        patch_indices= patch_indices.narrow(-1,0,self.patch_max).contiguous()#trim the end, hacky, but eh.


        pos_emb = pos_emb.flatten(start_dim=0,end_dim=-1).unsqueeze(0).unsqueeze(0).contiguous()
        toemb = pos_emb + torch.flatten(patch_embeds, start_dim=2, end_dim=-1)
        patch_embeds = self.embedder(toemb)

        pb, pt, pe = patch_embeds.size()
        patch_embeds = patch_embeds[:,-(pt):-2,:] #(b, t_internal, n_embed)
        
        patch_targets = patch_indices[:,-(pt-1):-1,:] #(b, t_internal, patchmax)
        
        patch_embeds = patch_embeds 
        return patch_embeds, patch_targets, losses
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        block_size = self.config.block_size
        patch_max = self.patch_max
        vocab_size = self.config.vocab_size
        batch_size = idx.size(0) # Get batch size

        generated_tokens = 0
        finished_batches = torch.zeros(batch_size, dtype=torch.bool, device=idx.device) # Track finished batches
        next_indices_list = [] # List to accumulate generated indices for each batch

        # Initialize next_indices_list with current idx sequences
        for i in range(batch_size):
            next_indices_list.append(idx[i].clone()) # Clone to avoid modifying original idx

        for _ in range(max_new_tokens // patch_max):
            if generated_tokens >= max_new_tokens or torch.all(finished_batches): # Stop condition
                break

            # Prepare input for the model, handling sequence length cropping
            current_idx_batch = []
            for i in range(batch_size):
                current_seq = next_indices_list[i]
                if current_seq.size(0) > block_size:
                    current_seq = current_seq[-block_size:] # Crop if too long
                current_idx_batch.append(current_seq)
            idx_cond = torch.stack(current_idx_batch) # Stack into a batch

            logits, _ = self(idx_cond)
            logits_reshaped = logits.view(batch_size, -1, patch_max, vocab_size)
            patch_logits = logits_reshaped[:, -1, :, :]

            next_patch_tokens_batch = []
            for batch_idx in range(batch_size):
                if finished_batches[batch_idx]:
                    next_patch_tokens_batch.append(None) # Placeholder for finished batch
                    continue

                current_batch_patch_tokens = []
                for patch_pos in range(patch_max):
                    pos_logits = patch_logits[batch_idx, patch_pos, :] / temperature
                    if top_k is not None:
                        v_topk, _ = torch.topk(pos_logits, min(top_k, pos_logits.size(-1)))
                        pos_logits[pos_logits < v_topk[..., [-1]]] = -float('Inf')
                    probs = F.softmax(pos_logits, dim=-1)
                    idx_next_token = torch.multinomial(probs, num_samples=1)

                    current_batch_patch_tokens.append(idx_next_token)
                    generated_tokens += 1

                    if idx_next_token.item() == self.end_patch_token_id:
                        finished_batches[batch_idx] = True
                        break
                    if generated_tokens >= max_new_tokens:
                        break
                if not finished_batches[batch_idx]:
                    next_patch_tokens_batch.append(torch.cat(current_batch_patch_tokens))
                else:
                    next_patch_tokens_batch.append(torch.tensor([], dtype=torch.long, device=idx.device)) # Empty tensor for finished

            # Update next_indices_list with generated patch tokens
            for batch_idx in range(batch_size):
                if not finished_batches[batch_idx]:
                    patch_tokens = next_patch_tokens_batch[batch_idx]
                    next_indices_list[batch_idx] = torch.cat((next_indices_list[batch_idx], patch_tokens))


        # Stack the list of sequences back into a batch tensor for return
        #idx_result = torch.cat(next_indices_list)
        return next_indices_list
class Block(nn.Module):
    def __init__(self, config, causal= True):
        super().__init__()
        self.ln_1 = model.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, causal)
        self.ln_2 = model.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = model.MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        #x = x + self.attn(x)
        #x = x + self.mlp(x)
        #x = mmnorm(x)
        return x

class eBlock(nn.Module):
    def __init__(self, config, d):
        super().__init__()
        self.attn = CausalSelfAttention(config,d)
        self.mlp = model.MLP(config)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        x = utils.mmnorm(x)
        return x
class CausalSelfAttention(nn.Module):
    
    def __init__(self, config, causal = True):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        

        self.causal = causal
        #self.splits = 1
        # regularization
        self.diffSplits = config.diffSplits
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head // self.diffSplits
        self.n_embd = config.n_embd
        self.ssmax = config.ssmax
        self.dropout = config.dropout
        #diff
        self.head_dim =  self.n_embd //  self.n_head // self.diffSplits
        if(self.diffSplits > 1):
            #self.lambda_init = utils.lambda_init_fn(d) 
            self.q_lambdas = nn.ParameterList([nn.Parameter(torch.zeros(self.head_dim, dtype=torch.bfloat16).normal_(mean=0,std=0.1)) for _ in range(self.diffSplits)])
            self.k_lambdas = nn.ParameterList([nn.Parameter(torch.zeros(self.head_dim, dtype=torch.bfloat16).normal_(mean=0,std=0.1)) for _ in range(self.diffSplits)])
        if(self.ssmax):
            self.qm = nn.Parameter(torch.ones(self.head_dim).normal_(mean=1, std=0.1))
            self.qb = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        device = x.device
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        
        #q = mmnorm(q)
        #k = mmnorm(k)
        #v = mmnorm(v)
        #q=utils.zca_newton_schulz(q)
        #k=utils.zca_newton_schulz(k)#best
        #v=utils.zca_newton_schulz(v)

        #kernel_size = 3  # Example
        #causal_kernel = torch.ones(1, 1, kernel_size).to(x.device)  # Shape: (out_channels, in_channels, kernel_size)
        #causal_kernel[:, :, kernel_size // 2 + 1:] = 0  # Zero-out future positions

        ## Reshape Q for convolution: (B, T, C) -> (B, C, T)
        #v = v.permute(0, 2, 1)
        ## Apply convolution
        #v = self.query_conv(v)
        ## Reshape back: (B, C, T) -> (B, T, C)
        #v = v.permute(0, 2, 1)
        #q = q*(math.log(T)*self.qm)
        #q = utils.fft_trunc_squish(q)
        #k = utils.fft_trunc_squish(k)

        k = k.view(B, T, self.n_head, self.diffSplits * self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.diffSplits * self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.diffSplits * self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        #qn = q.norm(dim=-1, keepdim=True)
        #q = q /qn
        #kn = k.norm(dim=-1, keepdim=True)
        #vn = v.norm(dim=-1, keepdim=True)
        #v= v/vn
        

        # Split q and k for the diff part
        qs = q.split(self.head_dim, dim=-1)
        ks = k.split(self.head_dim, dim=-1)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash: #and self.splits >= 1:
            # efficient attention using Flash Attention CUDA kernels
            ys = []
            for i in range(self.diffSplits):#+self.qb
                if(self.ssmax):
                    ys.append(torch.nn.functional.scaled_dot_product_attention(qs[i]*(math.log(T)*self.qm), ks[i], v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal= self.causal))
                else:
                    ys.append(torch.nn.functional.scaled_dot_product_attention(qs[i], ks[i], v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=self.causal))
                        
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            
            att = math.e ^ att / (1+sum(math.e ^ att))
            #att = F.softmax(att, dim=-1)
            #att = ((self.mmnorm(att) + 1) / 2) ** 3
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = ys[0] #*kn#*vn
        if self.diffSplits > 1:
            #lambda_full = lambda_1 - lambda_2 + self.lambda_init
            lambda_full = self.lambda_init + torch.exp(torch.sum(self.q_lambdas[0] * self.k_lambdas[0], dim=-1).float()).type_as(q) # + lambda_1 - lambda_2 
            for i in range(self.diffSplits-1):
                lambda_full = lambda_full - torch.exp(torch.sum(self.q_lambdas[1+i] * self.k_lambdas[1+i], dim=-1).float()).type_as(q)
                #lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
            for i in range(self.diffSplits-1):
                y = y - lambda_full.unsqueeze(-1).unsqueeze(-1) * ys[1+i]

        #y = mmnorm(y, dim = 0)#, scale = 1) 
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.diffSplits * self.head_dim) # re-assemble all head outputs side by side

        #y = mmnorm(y) #mmnorm: with: 1.4565, without: 1.4416
        #unfixed without: step 6000: train loss 1.1178, val loss 1.4572, 12050.81ms
        #unfixed with: step 9000: train loss 1.1000, val loss 1.4497, 14662.64ms #wack
        #fixed without: step 8000: train loss 1.0563, val loss 1.4606, 18392.37ms
        #fixed with: step 10000:  val loss 1.4479, 20605.08ms

        #y = self.subln(y) # with fix:

        #y = self.subln(y) # lowest: 1.4778 (with scale and vnorm)
        #y = y * (1 - self.lambda_init) 

        
        #y = y.reshape(B, T, self.num_heads * 2 * self.head_dim)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
class GapRelu2D(nn.Module):
    def __init__(self):
        super(GapRelu2D, self).__init__()

    def forward(self, x):
        x1h, x2h = x.chunk(2, dim=2)
        x1=torch.relu(x1h)
        x2=(torch.sign(x1h)+1) * x2h
        return x1+x2
