
import math
from tabnanny import check

import torch
import torch.nn as nn
from torch.nn import functional as F
import utils
import quantizer

class LayerNorm(nn.Module):
    def __init__(self, n_embd, bias, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd)) if bias else None
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class Attention(nn.Module):
    def __init__(self, n_embd, n_head, dropout, bias, **kwargs):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
      
        self.c_proj = nn.Linear(n_embd,  n_embd, bias=bias)
        
        self.attmod = nn.Linear(n_embd,  n_embd, bias=bias)
      
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head 
        self.n_embd = n_embd
        self.dropout = dropout
        self.head_dim =  self.n_embd //  self.n_head 
        
        self.qm = nn.Parameter(torch.ones(self.head_dim).normal_(mean=1, std=0.1))
        
    def forward(self, x, causal=True):
        B, T, C = x.size() 
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)

        y = torch.nn.functional.scaled_dot_product_attention(
            q * (math.log(T)*self.qm),
            k, v, 
            attn_mask=None, dropout_p=self.dropout if self.training else 0,
            is_causal= causal)

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim) 
        y = self.resid_dropout(self.c_proj(y))

        return y
class MLP(nn.Module):
    def __init__(self, n_embd, dropout, bias, **kwargs):
        super().__init__()
        up = 4
        self.c_fc    = nn.Linear(n_embd, up * n_embd, bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear( up * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.n_embd = n_embd

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout, bias, **kwargs):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias)
        self.attn = Attention(n_embd, n_head, dropout, bias)
        self.ln_2 = LayerNorm(n_embd, bias)
        self.mlp = MLP(n_embd, dropout, bias)

    def forward(self, x,  causal = True):
        x = x + self.attn(self.ln_1(x),causal)
        x = x + self.mlp(self.ln_2(x))
        return x


class LearnableFourierResampler(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_filters: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_filters = num_filters
        self.max_freq = input_dim // 2
        self.min_freq = 1.0
        initial_frequencies = torch.linspace(self.min_freq, self.max_freq, num_filters)
        initial_logits = torch.log(initial_frequencies - self.min_freq) \
            - torch.log(self.max_freq - initial_frequencies)
        self.frequency_logits = nn.Parameter(initial_logits)

        t_in = torch.linspace(0, 1, input_dim)
        t_out = torch.linspace(0, 1, output_dim)
        self.register_buffer("t_in", t_in)
        self.register_buffer("t_out", t_out)

    def _get_bases(self):
        scaled_sigmoid = torch.sigmoid(self.frequency_logits)
        freqs = self.min_freq + (self.max_freq - self.min_freq) * scaled_sigmoid
        freqs.unsqueeze_(1) 

        arg_in = 2 * math.pi * freqs * self.t_in.unsqueeze(0)
        sin_basis_in = torch.sin(arg_in)
        cos_basis_in = torch.cos(arg_in)
        
        arg_out = 2 * math.pi * freqs * self.t_out.unsqueeze(0)
        sin_basis_out = torch.sin(arg_out)
        cos_basis_out = torch.cos(arg_out)
        
        return sin_basis_in, cos_basis_in, sin_basis_out, cos_basis_out

    def forward(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert x.shape[dim] == self.input_dim, \
            f"Input tensor dim {dim} has size {x.shape[dim]}, " \
            f"but module was initialized with input_dim {self.input_dim}."
            
        x_permuted = x.transpose(dim, -1)
        original_shape = x_permuted.shape
        x_flattened = x_permuted.reshape(-1, self.input_dim)
        
        sin_basis_in, cos_basis_in, sin_basis_out, cos_basis_out = self._get_bases()
        
        c_sin = torch.einsum('ns,fs->nf', x_flattened, sin_basis_in)
        c_cos = torch.einsum('ns,fs->nf', x_flattened, cos_basis_in)
        
        sin_recon = torch.einsum('nf,fs->ns', c_sin, sin_basis_out)
        cos_recon = torch.einsum('nf,fs->ns', c_cos, cos_basis_out)
        
        x_reconstructed = sin_recon + cos_recon 
        
        output_shape = original_shape[:-1] + (self.output_dim,)
        output = x_reconstructed.view(output_shape)
        
        output = output.transpose(dim, -1).contiguous()
        
        return output

class Dreamer(nn.Module):
    def __init__(self, mem_block_size, causal, **kwargs):
        super().__init__()
        self.block = Block(**kwargs)
        self.causal = causal
        self.comp = LearnableFourierResampler(mem_block_size * 2, mem_block_size, 64)
        self.ln = LayerNorm(**kwargs)
        
    def forward(self, x):
        while x.shape[0] > 1:
            b, t, c = x.size()
            x = x.reshape(b // 2, 2, t, c) #b//2, 2, t, c
            x = x.reshape(b // 2, t * 2, c).contiguous() #b//2, 2 * t, c
            x = x.transpose(1,2)
            x = self.comp(x)
            x = x.transpose(1,2) # ?,t,c
            x = self.block(x, causal = self.causal)
            x = self.ln(x)
        return x 