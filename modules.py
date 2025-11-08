
import math
from tabnanny import check

from numpy import shape
import torch
import torch.nn as nn
from torch.nn import functional as F
import utils
import quantizer
from torch.optim.optimizer import Optimizer


class SSM(nn.Module):
    def __init__(self, n_embd, dropout, bias):
        super().__init__()

        self.q_attn =  nn.Linear(n_embd, n_embd, bias=bias)
        self.v_attn =  nn.Linear(n_embd, n_embd, bias=bias)
        
        self.c_proj = nn.Linear(n_embd,  n_embd, bias=bias)

        self.identity = nn.Parameter(torch.rand(n_embd))
        
        self.attn_dropout = nn.Dropout(dropout) #todo
        self.resid_dropout = nn.Dropout(dropout) #todo
        self.n_embd = n_embd
        self.dropout = dropout
        self.scanner = ProcrustesButterflyLinear(n_embd * 2, method="polar")
        
    def forward(self, x: torch.tensor, causal = False): #same signature to allow easy swapping.
        B, T, C = x.size() 
        q = self.q_attn(x) 
        v = self.v_attn(x)
        ow = self.scanner.compute_orthogonal_weights()
        def scan(left, right):
            z = self.scanner(left,right,ow).to(x.dtype)
            return utils.range_norm(z)
        q = utils.pscan(q, scan, self.identity)
        
        Y = q*v 
        Y = self.c_proj(Y)
        Y = self.resid_dropout(Y)
        return Y

class CombineAttention(nn.Module):
    def __init__(self, n_embd,n_head,dropout,bias):
        super().__init__()
        assert n_embd % n_head == 0
        self.q_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=bias)
      
        self.c_proj = nn.Linear(n_embd,  n_embd, bias=bias)
        
        self.attmod = nn.Linear(n_embd,  n_embd, bias=bias)
      
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head 
        self.n_embd = n_embd
        self.dropout = dropout
        self.head_dim =  self.n_embd //  self.n_head 
        
        self.qm = nn.Parameter(torch.ones(self.head_dim).normal_(mean=1, std=0.1))
        
    def _create_mask(self, T_q, T_kv, device):
        scaling_factor = T_kv / T_q
        query_indices = torch.arange(T_q, device=device)
        key_indices = torch.arange(T_kv, device=device)
        max_key_indices = ((query_indices + 1) * scaling_factor).floor() - 1
        mask = key_indices[None, :] > max_key_indices[:, None]
        return mask
    
    def forward(self, x, sx, causal = False):
        B, T, C = x.size() 
        sB, sT, sC = sx.size() 
        q = self.q_proj(sx)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(B, sT, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, sT, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        attn_mask=None
        if causal:
            if sT != T:
                attn_mask = self._create_mask(sT, T, x.device)
                causal = False

        y = torch.nn.functional.scaled_dot_product_attention(
            q * (math.log(T)*self.qm),
            k, v, 
            attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0,
            is_causal= causal)
        
        y = y.transpose(1, 2).contiguous().view(B, sT, C) 
        y = self.resid_dropout(self.c_proj(y))

        return y

class CombineBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout, bias):
        super().__init__()

        self.attn = CombineAttention(n_embd, n_head, dropout, bias)

        self.ln_1 = LayerNorm(n_embd, bias)
        self.ln_2 = LayerNorm(n_embd, bias)
        self.ln_3 = LayerNorm(n_embd, bias)
        self.mlp = MLP(n_embd, dropout, bias)

    def forward(self, x, sx, causal = False):
        a = self.attn(self.ln_1(x), self.ln_2(sx), causal)
        sx = sx + a
        sx = sx + self.mlp(self.ln_3(sx))
        return sx

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
        
        #test unit linear, unitifying :D
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

class SSMBlock(nn.Module):
    def __init__(self, n_embd, dropout, bias, **kwargs):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias)
        self.attn = SSM(n_embd, dropout, bias)
        self.ln_2 = LayerNorm(n_embd, bias)
        self.mlp = MLP(n_embd, dropout, bias)

    def forward(self, x,  causal = True):
        x = x + self.attn(self.ln_1(x),causal)
        x = x + self.mlp(self.ln_2(x))
        return x

csbr = True
class LearnableFourierResampler(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_filters: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_filters = num_filters
        self.max_freq = input_dim // 2
        self.min_freq = 1.0
        if csbr:
            initial_frequencies = self.min_freq + (self.max_freq - self.min_freq) * torch.rand(self.num_filters)
            initial_frequencies = torch.sort(initial_frequencies).values # Optional: for easier interpretation
        else:
            # compressed sensing says that random samples wouldbe better?
            initial_frequencies = torch.linspace(self.min_freq, self.max_freq, num_filters)
        initial_logits = torch.log(initial_frequencies - self.min_freq) \
            - torch.log(self.max_freq - initial_frequencies)
        self.frequency_logits = nn.Parameter(initial_logits)

        t_in = torch.linspace(0, 1, input_dim)
        t_out = torch.linspace(0, 1, output_dim)
        self.register_buffer("t_in", t_in)
        self.register_buffer("t_out", t_out)

    def forward(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert x.shape[dim] == self.input_dim, \
            f"Input tensor dim {dim} has size {x.shape[dim]}, " \
            f"but module was initialized with input_dim {self.input_dim}."
            
        x_permuted = x.transpose(dim, -1)
        original_shape = x_permuted.shape
        x_flattened = x_permuted.reshape(-1, self.input_dim)
        
        scaled_sigmoid = torch.sigmoid(self.frequency_logits)
        freqs = self.min_freq + (self.max_freq - self.min_freq) * scaled_sigmoid
        freqs.unsqueeze_(1) 

        arg_in = 2 * math.pi * freqs * self.t_in.unsqueeze(0)
        sin_basis_in = torch.sin(arg_in)
        cos_basis_in = torch.cos(arg_in)
        
        arg_out = 2 * math.pi * freqs * self.t_out.unsqueeze(0)
        sin_basis_out = torch.sin(arg_out)
        cos_basis_out = torch.cos(arg_out)
        
        c_sin = torch.einsum('ns,fs->nf', x_flattened, sin_basis_in)
        c_cos = torch.einsum('ns,fs->nf', x_flattened, cos_basis_in)
        
        sin_recon = torch.einsum('nf,fs->ns', c_sin, sin_basis_out)
        cos_recon = torch.einsum('nf,fs->ns', c_cos, cos_basis_out)
        
        x_reconstructed = sin_recon + cos_recon 
        
        output_shape = original_shape[:-1] + (self.output_dim,)
        output = x_reconstructed.view(output_shape)
        
        output = output.transpose(dim, -1).contiguous()
        
        return output


class ResampFFTGaps(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_filters: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_filters = num_filters
        self.max_freq = input_dim // 2
        self.min_freq = 1.0

        initial_gaps = (self.max_freq - self.min_freq) / num_filters
        initial_logits = torch.log(torch.exp(torch.tensor(initial_gaps)) - 1).expand(num_filters)
        self.gap_logits = nn.Parameter(initial_logits)

        t_in = torch.linspace(0, 1, input_dim)
        t_out = torch.linspace(0, 1, output_dim)
        self.register_buffer("t_in", t_in)
        self.register_buffer("t_out", t_out)
    
    def forward(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert x.shape[dim] == self.input_dim, \
            f"Input tensor dim {dim} has size {x.shape[dim]}, " \
            f"but module was initialized with input_dim {self.input_dim}."
            
        x_permuted = x.transpose(dim, -1)
        original_shape = x_permuted.shape
        x_flattened = x_permuted.reshape(-1, self.input_dim)
        
        gaps = F.softplus(self.gap_logits)
        cumulative_gaps = torch.cumsum(gaps, dim=0)
        freqs = self.min_freq + cumulative_gaps
        freqs.unsqueeze_(1) 

        arg_in = 2 * math.pi * freqs * self.t_in.unsqueeze(0)
        sin_basis_in = torch.sin(arg_in)
        cos_basis_in = torch.cos(arg_in)
        
        arg_out = 2 * math.pi * freqs * self.t_out.unsqueeze(0)
        sin_basis_out = torch.sin(arg_out)
        cos_basis_out = torch.cos(arg_out)
        
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

class OrthogonalLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = F.linear(input, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
       
        input, weight, bias = ctx.saved_tensors
        
        grad_input = grad_weight = grad_bias = None

        input_2d = input.reshape(-1, input.shape[-1])
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])

        if ctx.needs_input_grad[0]:
            grad_input = grad_output_2d @ weight
            grad_input = grad_input.reshape(grad_output.shape[:-1] + (input.shape[-1],))

        if ctx.needs_input_grad[1]:
            grad_weight = grad_output_2d.T @ input_2d

            w_flat = weight.flatten()
            grad_w_flat = grad_weight.flatten()
            
            projection_scalar = torch.dot(w_flat, grad_w_flat) / (torch.dot(w_flat, w_flat) + 1e-8)
            
            grad_parallel = projection_scalar * weight
            
            grad_weight_orthogonal = grad_weight - grad_parallel
            
            grad_weight = grad_weight_orthogonal

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output_2d.sum(dim=0)

        return grad_input, grad_weight, grad_bias

class OrthogonalLinear(nn.Linear):
    def forward(self, input):
        return OrthogonalLinearFunction.apply(input, self.weight, self.bias)
#gemini garbage:
class RiemannianLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        # To maintain orthogonality, we should ideally ensure weights start
        # on the manifold. This can be done with torch.nn.init.orthogonal_
        # outside of this function when the layer is created.
        ctx.save_for_backward(input, weight, bias)
        output = F.linear(input, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # Reshape input and grad_output to 2D matrices for matmul
        input_2d = input.reshape(-1, input.shape[-1])
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])

        if ctx.needs_input_grad[0]:
            # Standard gradient for the input
            grad_input = grad_output_2d @ weight
            grad_input = grad_input.reshape(grad_output.shape[:-1] + (input.shape[-1],))

        if ctx.needs_input_grad[1]:
            # Step 1: Compute the standard Euclidean gradient for the weights
            grad_weight_euclidean = grad_output_2d.T @ input_2d

            # Step 2: Project the Euclidean gradient onto the tangent space
            # of the Stiefel manifold. This is the Riemannian gradient.
            # Formula: grad_riemannian = grad_w - W @ sym(W.T @ grad_w)
            # This is for W @ W.T = I (orthogonal rows), which is typical for nn.Linear.
            
            A = weight @ grad_weight_euclidean.T # Note the transpose order for W.T @ grad_w
            
            # The matrix A is (out, out). sym_A is its symmetric part.
            sym_A = 0.5 * (A + A.T)
            
            # Project out the component normal to the manifold
            grad_weight_riemannian = grad_weight_euclidean - sym_A @ weight
            
            grad_weight = grad_weight_riemannian

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output_2d.sum(dim=0)

        return grad_input, grad_weight, grad_bias

class RiemannianLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        nn.init.orthogonal_(self.weight)

    def forward(self, input):
        return RiemannianLinearFunction.apply(input, self.weight, self.bias)

class RotorLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features=in_features
        self.out_features=out_features


        self.square_dim = min(in_features, out_features)
        
        self.weight = nn.Parameter(torch.empty(self.square_dim, self.square_dim))
        nn.init.orthogonal_(self.weight) # A good initialization is helpful
        
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        

    def _get_orthogonal_weight(self):
        A = self.weight - self.weight.T
        
        W_square = torch.matrix_exp(A)
        
        if self.out_features > self.in_features:
            W = F.pad(W_square, (0, 0, 0, self.out_features - self.in_features))
        else:
            W = W_square[:self.out_features, :]
            
        return W

    def forward(self, input):
        W = self._get_orthogonal_weight()
        return F.linear(input, W, self.bias)

def fast_polar_decomposition(A, polar_iter=2, inv_iter=2, power_iter=2):
    U = A.clone()
    d = U.shape[0]
    I_k = torch.eye(d, device=U.device, dtype=U.dtype)
    v = torch.randn(d, 1, device=U.device, dtype=U.dtype)
    for _ in range(polar_iter):
        
        u_p = U @ v
        s = u_p.norm() + 1e-8 
        U_scaled = U / s

        U_scaled_inv = 2 * I_k - U_scaled
            
        U_inv = U_scaled_inv / s
        
        U_inv_T = U_inv.transpose(-2, -1)
        U = 0.5 * (U + U_inv_T)
        
    return U


class ProcrustesLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, method: str = 'svd'):
        
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        if method not in ['svd', 'polar', 'qr']:
            raise ValueError("method must be one of 'svd', 'polar', or 'qr'")
        self.method = method

        self.weight_param = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.orthogonal_(self.weight_param)
    
    def compute_orthogonal_weight(self):
        
        if self.method == 'svd':
            param_f32 = self.weight_param.to(torch.float32)
            U, S, Vh = torch.linalg.svd(param_f32, full_matrices=False)
            return (U @ Vh)
        elif self.method == 'polar':
            return fast_polar_decomposition(self.weight_param.to(torch.float32))
        elif self.method == 'qr':
            param_f32 = self.weight_param.to(torch.float32)
            Q, R = torch.linalg.qr(param_f32)
            return Q
        
    def forward(self, x: torch.Tensor, weight_ortho: torch.Tensor =None) -> torch.Tensor:
        if weight_ortho is None:
            weight_ortho = self.compute_orthogonal_weight().to(self.weight_param.dtype)
        else:
            weight_ortho = weight_ortho
        
        
        return F.linear(x, weight_ortho, None)

class ProcrustesButterflyLinear(nn.Module):
    def __init__(self, in_features: int, method='qr'):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = in_features // 2
        self.n = self.out_features

        self.proj1 = ProcrustesLinear(self.n, self.n, method=method)
        self.proj2 = ProcrustesLinear(self.n, self.n, method=method)
    
    def compute_orthogonal_weights(self):
        return self.proj1.compute_orthogonal_weight(), self.proj2.compute_orthogonal_weight()
    

    def forward(self, x: torch.Tensor, y: torch.Tensor, pw) -> torch.Tensor:
        y1_proj = self.proj1(x,pw[0])
        y2_proj = self.proj2(y,pw[1])
        
        y_out = y1_proj + y2_proj

        return y_out


class LipschitzLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features

        self.params = nn.Linear(in_features, out_features, bias=False)
        
        nn.init.orthogonal_(self.params.weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mmT = torch.matmul(self.params.weight, self.params.weight.T)
        
        lipschitz_bounds_squared = torch.sum(torch.abs(mmT), dim=1)

        rescaling_factors = torch.rsqrt(lipschitz_bounds_squared + 1e-10)

        W = self.params.weight * rescaling_factors.unsqueeze(1)
            
        return F.linear(input, W, None)

class CayleyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.square_dim = max(in_features, out_features)
        
        self.weight = nn.Parameter(torch.empty(self.square_dim, self.square_dim))
        nn.init.uniform_(self.weight, -0.1, 0.1) # A simple initialization is fine

        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            nn.init.zeros_(self.bias)

    def _get_orthogonal_weight(self):
        A = self.weight - self.weight.T
        
        I = torch.eye(self.square_dim, device=A.device, dtype=A.dtype)
        
        W_square = torch.linalg.solve(I - A, I + A)
        
        W = W_square[:self.out_features, :self.in_features]
            
        return W

    def forward(self, input):
        W = self._get_orthogonal_weight()
        return F.linear(input, W, self.bias)

class LowRankCayleyLinear(nn.Module):
    def __init__(self, in_features, out_features, rank: int, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.U = nn.Parameter(torch.empty(in_features, rank))
        self.V = nn.Parameter(torch.empty(in_features, rank))
        
        nn.init.orthogonal_(self.U)
        nn.init.orthogonal_(self.V)

        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            nn.init.zeros_(self.bias)

    def _get_orthogonal_weight(self, num_iter: int = 4):
        # This entire function can now run in BFloat16
        A = self.U @ self.V.T - self.V @ self.U.T

        I = torch.eye(self.in_features, device=A.device, dtype=A.dtype)
        B = I - A

        X = I 
        for _ in range(num_iter):
            X = X @ (2*I - B @ X)

        W_square = (I + A) @ X
        
        W = W_square[:self.out_features, :self.in_features]

        return W

    def forward(self, input):
        W = self._get_orthogonal_weight()
        return F.linear(input, W, self.bias)


class LowrankWoodburyCayleyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, 
                num_iter: int = 4, bias: bool = False):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.num_iter = num_iter

        # Learnable low-rank parameters U and V
        self.U = nn.Parameter(torch.empty(in_features, rank))
        self.V = nn.Parameter(torch.empty(in_features, rank))
        
        # A robust initialization for orthogonal/unitary matrices
        nn.init.orthogonal_(self.U)
        nn.init.orthogonal_(self.V)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_flat = x.reshape(-1, self.in_features) # (B, C_in)
        
        L = torch.cat([self.U, -self.V], dim=1) # Shape: (C_in, 2k)
        R = torch.cat([self.V, self.U], dim=1)  # Shape: (C_in, 2k)

        R_T_x = (x_flat @ R).T 

        I_k = torch.eye(2 * self.rank, device=x.device, dtype=x.dtype)
        R_T_L = R.T @ L # (2k, C_in) @ (C_in, 2k) -> (2k, 2k)
        M = I_k - R_T_L

        M_inv_R_T_x = torch.linalg.solve(M.to(torch.float32), R_T_x.to(torch.float32))
        M_inv_R_T_x = M_inv_R_T_x.to(x.dtype) # Cast back to original dtype
        
        x_inv = x_flat + (L @ M_inv_R_T_x).T
        
        R_T_x_inv = (x_inv @ R).T # (2k, B)
        y_flat = x_inv + (L @ R_T_x_inv).T # (B, C_in)

        y_sliced = y_flat[..., :self.out_features]
        
        if self.bias is not None:
            y_sliced += self.bias
        
        return y_sliced.reshape(*original_shape[:-1], self.out_features)
    
def manual_inverse_newton_schulz(M, num_iter=10, power_iter=10):
    original_dtype = M.dtype
    
    d = M.shape[0]
    matrix_f32 = M.to(torch.float32)
    v = torch.randn(d, 1, device=M.device, dtype=torch.float32)
    for _ in range(power_iter):
        u = torch.matmul(matrix_f32, v)
        u = u / (u.norm() + 1e-8)
        v = torch.matmul(matrix_f32.T, u)
        v = v / (v.norm() + 1e-8)
    s = torch.matmul(u.T, torch.matmul(matrix_f32, v)).squeeze().abs()
    
    M_scaled = M / s

    I_k = torch.eye(M.shape[0], device=M.device, dtype=original_dtype)
    M_scaled_inv = I_k
    for _ in range(num_iter):
        M_scaled_inv = M_scaled_inv @ (2 * I_k - M_scaled @ M_scaled_inv)
            
    M_inv_approx = M_scaled_inv / s
    
    return M_inv_approx
class CayleyInverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M):
        M_f32 = M.to(torch.float32)
        M_inv_approx = torch.linalg.inv(M_f32)
        
        ctx.save_for_backward(M, M_inv_approx) 
        return M_inv_approx.to(M.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        M, M_inv_approx = ctx.saved_tensors
        
        M_inv_approx_T = M_inv_approx.T 
        grad_M = -M_inv_approx_T @ grad_output @ M_inv_approx_T
        
        return grad_M, None # No grad for num_iter

class FastWoodburyCayleyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, 
                 num_iter: int = 4, bias: bool = False):
        super().__init__()
        # ... (same __init__ as before)
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.num_iter = num_iter

        self.U = nn.Parameter(torch.empty(in_features, rank))
        self.V = nn.Parameter(torch.empty(in_features, rank))
        nn.init.orthogonal_(self.U)
        nn.init.orthogonal_(self.V)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        
        L = torch.cat([self.U, -self.V], dim=1)
        R = torch.cat([self.V, self.U], dim=1)
        
        R_T_L = R.T @ L
        M = torch.eye(2 * self.rank, device=x.device, dtype=x.dtype) - R_T_L

        M_inv = CayleyInverse.apply(M, self.num_iter)
        
        R_T_x = x_flat @ R
        M_inv_R_T_x = R_T_x @ M_inv.T
        L_M_inv_R_T_x = M_inv_R_T_x @ L.T
        
        y_flat = x_flat + 2 * L_M_inv_R_T_x
        # ... (same slicing and bias logic as before)
        y_sliced = y_flat[..., :self.out_features]
        if self.bias is not None:
            y_sliced += self.bias
        return y_sliced.reshape(*original_shape[:-1], self.out_features)


class AnalyticalCayleyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # Learnable low-rank parameters U and V
        self.U = nn.Parameter(torch.empty(in_features, rank))
        self.V = nn.Parameter(torch.empty(in_features, rank))
        
        # A robust initialization for orthogonal/unitary matrices
        nn.init.orthogonal_(self.U)
        nn.init.orthogonal_(self.V)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- 1. Standard Setup ---
        original_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        
        # Implicitly define the skew-symmetric matrix A = UV^T - VU^T via L and R
        L = torch.cat([self.U, -self.V], dim=1) # Shape: (C_in, 2k)
        R = torch.cat([self.V, self.U], dim=1)  # Shape: (C_in, 2k)
        
        # --- 2. Analytical Block Matrix Inversion of M = I - R^T L ---
        k = self.rank
        I_k = torch.eye(k, device=x.device, dtype=x.dtype)

        # Define the four k x k blocks of M
        V_T_U = self.V.T @ self.U
        V_T_V = self.V.T @ self.V
        U_T_U = self.U.T @ self.U
        U_T_V = self.U.T @ self.V
        
        A_block = I_k - V_T_U
        B_block = V_T_V
        C_block = -U_T_U
        D_block = I_k + U_T_V

        # Perform inversions in float32 for stability
        A_block_f32 = A_block.to(torch.float32)
        B_block_f32 = B_block.to(torch.float32)
        C_block_f32 = C_block.to(torch.float32)
        D_block_f32 = D_block.to(torch.float32)
        
        A_inv = torch.linalg.inv(A_block_f32)
        S = D_block_f32 - (C_block_f32 @ A_inv @ B_block_f32)
        S_inv = torch.linalg.inv(S)

        TopLeft_inv = A_inv + A_inv @ B_block_f32 @ S_inv @ C_block_f32 @ A_inv
        TopRight_inv = -A_inv @ B_block_f32 @ S_inv
        BottomLeft_inv = -S_inv @ C_block_f32 @ A_inv
        BottomRight_inv = S_inv

        Top_row = torch.cat([TopLeft_inv, TopRight_inv], dim=1)
        Bottom_row = torch.cat([BottomLeft_inv, BottomRight_inv], dim=1)
        M_inv = torch.cat([Top_row, Bottom_row], dim=0)
        M_inv = M_inv.to(x.dtype) # Cast back to original dtype
        
        R_T_x = x_flat @ R                 # (B, C_in) @ (C_in, 2k) -> (B, 2k)
        M_inv_R_T_x = R_T_x @ M_inv.T      # (B, 2k) @ (2k, 2k) -> (B, 2k)
        L_M_inv_R_T_x = M_inv_R_T_x @ L.T  # (B, 2k) @ (2k, C_in) -> (B, C_in)
        
        y_flat = x_flat + 2 * L_M_inv_R_T_x
        
        y_sliced = y_flat[..., :self.out_features]
        
        if self.bias is not None:
            y_sliced += self.bias
        
        return y_sliced.reshape(*original_shape[:-1], self.out_features)

class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.U = nn.Parameter(torch.randn(in_features, rank))
        self.V = nn.Parameter(torch.randn(rank, out_features))
        
    def forward(self, x):
        return (x @ self.U) @ self.V

class LowRankButterflyLinear(nn.Module):
    def __init__(self, in_features, rank):
        super().__init__()
        self.n = in_features // 2
        # Use two independent, parameter-efficient low-rank layers
        self.proj1 = LowRankLinear(self.n, self.n, rank)
        self.proj2 = LowRankLinear(self.n, self.n, rank)
        # ... bias, init etc. ...

    def forward(self, x):
        x1, x2 = x.split(self.n, dim=-1)
        y1 = self.proj1(x1) + self.proj2(x2)
        return y1

class GradientPredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, grad_output, original_input, weight):
        flat_input = original_input.flatten(1)
        flat_weight = weight.flatten(1)
        flat_grad_output = grad_output.flatten(1)

        if original_input.shape[0] > 1:
            flat_input = flat_input.mean(dim=0, keepdim=True)
            flat_grad_output = flat_grad_output.mean(dim=0, keepdim=True)

        x = torch.cat([flat_grad_output, flat_input, flat_weight], dim=-1)
        return self.net(x)

class GradientLearnerOptimizer(Optimizer):
    def __init__(self, params, defaultOptim, lr=1e-3, meta_lr=1e-4):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        
        self.meta_lr = meta_lr
        self.gradient_predictors = {}
        self.predictor_params = []
        # --- Initialization Step ---
        # Find all parameters that will need a learned gradient and build a predictor for each.
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                # We need a way to identify these special layers.
                # Let's assume we add a flag during model creation.
                if hasattr(p, '_is_learnable_gradient'):
                    # The dimensions are tricky and need to be defined carefully
                    input_dim = ... 
                    output_dim = p.numel()
                    
                    predictor = GradientPredictor(input_dim, output_dim)
                    self.gradient_predictors[id(p)] = predictor
                    self.predictor_params.append(predictor.parameters())
        
        self.defaultOptim= defaultOptim()


    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                if hasattr(p.grad, '_gradient_context'):
                    ctx = p.grad._gradient_context
                    predictor = self.gradient_predictors[id(p)]
                    predictor_optim = self.predictor_optimizers[id(p)]
                    
                    with torch.enable_grad():
                        predicted_grad = predictor(ctx['grad_output'], ctx['input'], ctx['weight'])
                        predicted_grad = predicted_grad.view_as(p)

                    with torch.enable_grad():
                        temp_p = p.detach().clone()
                        updated_p = temp_p - group['lr'] * predicted_grad

                        output_after_update = F.linear(ctx['input'], updated_p, ctx['bias'])
                        output_after_update = ctx['func'](output_after_update)
                        
                        target_output = F.linear(ctx['input'], p, ctx['bias']) - ctx['grad_output']
                        predictor_loss = F.mse_loss(output_after_update, target_output.detach())

                        predictor_optim.zero_grad()
                        predictor_loss.backward() # This computes grads for the predictor's params
                        predictor_optim.step()

                    p.add_(predicted_grad.detach(), alpha=-group['lr'])

                else:
                    p.add_(p.grad, alpha=-group['lr'])
        
        return loss
class GradientContext(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, weight, bias, func):
        # We need the inputs to the linear layer for the backward pass
        linear_input = input_tensor.clone()
        ctx.save_for_backward(linear_input, weight, bias)
        ctx.func = func
        
        output = F.linear(input_tensor, weight, bias)
        return func(output)

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, weight, bias = ctx.saved_tensors
        
        dummy_grad_w = torch.zeros_like(weight)
        dummy_grad_b = torch.zeros_like(bias) if bias is not None else None
        
        dummy_grad_w._gradient_context = {
            "input": input_tensor,
            "weight": weight,
            "bias": bias,
            "grad_output": grad_output,
            "func": ctx.func
        }
        return None, dummy_grad_w, dummy_grad_b, None   
    
class LNLinear(nn.Linear):
    def __init__(self, in_features, out_features, func):
        super().__init__(in_features, out_features, bias=True)
        self.func = func
        
    def forward(self, input):
        return GradientContext.apply(input, self.weight, self.bias, self.func)