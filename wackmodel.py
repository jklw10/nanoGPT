
import math
import model
import torch
import torch.nn as nn
from torch.nn import functional as F
import utils
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

    
    def justnorm(self, x):
        #return F.normalize(x, p=2, dim=-1)
        res = x / x.norm(p=2, dim=-1, keepdim=True)
        return res

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

    def justnorm(self, x):
        #return F.normalize(x, p=2, dim=-1)
        res = x / x.norm(p=2, dim=-1, keepdim=True)
        return res

    
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


class GapRelu2D(nn.Module):
    def __init__(self):
        super(GapRelu2D, self).__init__()

    def forward(self, x):
        x1h, x2h = x.chunk(2, dim=2)
        x1=torch.relu(x1h)
        x2=(torch.sign(x1h)+1) * x2h
        return x1+x2
