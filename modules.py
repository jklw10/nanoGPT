
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
import utils
import quantizer

class PerWeightActivationLayer(nn.Module):
    
    def __init__(self, in_features: int, out_features: int, activation=F.relu, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
      
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.linear.weight.data.uniform_(-1, 1)
        self.linear.weight.data *= self.linear.weight.data 
        self.linear.weight.data *= self.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        weight = self.linear.weight 
        x_expanded = x.unsqueeze(-2) 
        
        weighted_inputs = x_expanded * weight
        
        activated_weighted_inputs = self.activation(weighted_inputs)
        
        # Sum over the last dimension (the 'In' dimension)
        summed_output = torch.sum(activated_weighted_inputs, dim=-1) #/ (self.in_features ** 0.5)
        
        if self.linear.bias is not None:
            summed_output += self.linear.bias
            
        return summed_output * (self.in_features ** -0.5)


class WonkyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation=F.relu, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
      
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.bias = nn.Parameter(torch.randn_like(self.linear.weight))
        
        self.bias.data.uniform_(-0.01, 0.01) 
        self.linear.weight.data.uniform_(-0.1, 0.1) 

    def forward(self, x):
        linear_term = ( self.linear(x))
        
        ac_in = self.activation(x)
        ac_weight = (self.activation(self.linear.weight) + self.bias)
        
        ac_term = utils.gaussian_bump(F.linear(ac_in, ac_weight))
        return utils.gaussian_bump(linear_term - ac_term) 
    
class FakeSinLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear.weight.data.uniform_(-0.1, 0.1) 

    def forward(self, x):
        linear_term = self.linear(x)
        
        cubic_term = F.linear(x.pow(3), self.linear.weight.pow(3)) / 6.0
        
        return (linear_term - cubic_term) 
    

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
        self.scanner = ProcrustesButterflyLinear(n_embd)
        #self.shl_lh = SinkhornLinear(n_embd)
        #self.shl_rh = SinkhornLinear(n_embd)
        self.shl_lh = ProcrustesLinear(n_embd,n_embd)
        self.shl_rh = ProcrustesLinear(n_embd,n_embd)
        with torch.no_grad():
            self.shl_lh.weight.data = (torch.eye(n_embd))
            self.shl_rh.weight.data = (torch.eye(n_embd))
    
    def get_weight(self):
        return self.scanner.compute_orthogonal_weights()

    def forward(self, x: torch.tensor, weight = None): #same signature to allow easy swapping.
        B, T, C = x.size() 
        q = self.q_attn(x) 
        v = self.v_attn(x)
        if weight is None:
            weight = self.scanner.compute_orthogonal_weights()
        #with torch.no_grad():
        #    self.shl_lh.project()
        #    self.shl_rh.project()
        def scan(left, right):
            z = self.shl_lh(left) 
            z = z + self.shl_rh(right)
            z = utils.rms(z,dim=-1)

            #z = self.scanner(left,right,weight).to(x.dtype)
            #z = utils.range_norm(z)
            return z
        q = utils.pscan2(q, scan, self.identity)
        
        Y = q*v 
        Y = self.c_proj(Y)
        Y = self.resid_dropout(Y)
        return Y, q
    
    def nexts(self, prev: torch.tensor, x: torch.tensor,  causal=False, weight = None): #same signature to allow easy swapping.
        B, T, C = x.size() 
        q = self.q_attn(x) 
        v = self.v_attn(x)
        if weight is None:
            weight = self.scanner.compute_orthogonal_weights()
        def scan(left, right):
            z = self.scanner(left,right,weight).to(x.dtype)
            return utils.range_norm(z)
        q = scan(prev,q)
        
        Y = q*v 
        Y = self.c_proj(Y)
        Y = self.resid_dropout(Y)
        return Y, q


class CombineAttention(nn.Module):
    def __init__(self, n_embd,n_head,dropout,bias):
        super().__init__()
        assert n_embd % n_head == 0
        self.q_proj = OrthogonalLinear(n_embd, n_embd, bias=bias)
        self.k_proj = OrthogonalLinear(n_embd, n_embd, bias=bias)
        self.v_proj = OrthogonalLinear(n_embd, n_embd, bias=bias)
      
        self.c_proj = OrthogonalLinear(n_embd,  n_embd, bias=bias)
        
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
        self.q_proj = OrthogonalLinear(n_embd, n_embd, bias=bias)
        self.k_proj = OrthogonalLinear(n_embd, n_embd, bias=bias)
        self.v_proj = OrthogonalLinear(n_embd, n_embd, bias=bias)
      
      
        self.c_proj = OrthogonalLinear(n_embd,  n_embd, bias=bias)
        
        self.attmod = OrthogonalLinear(n_embd,  n_embd, bias=bias)
      
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head 
        self.n_embd = n_embd
        self.dropout = dropout
        self.head_dim =  self.n_embd //  self.n_head 
        
        self.qm = nn.Parameter(torch.ones(self.head_dim).normal_(mean=1, std=0.1))
        
    def forward(self, x, causal=True):
        B, T, C = x.size() 
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
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

    def forward(self, x, w=None):
        x = x + self.attn(self.ln_1(x), w)[0]
        x = x + self.mlp(self.ln_2(x))
        return x

csbr = True

class FFTResampler(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_filters: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_filters = num_filters
        self.max_freq = input_dim // 2
        self.min_freq = 1.0

        initial_gaps = (self.max_freq - self.min_freq) / (num_filters)
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
    def __init__(self, mem_block_size, **config):
        super().__init__()
        mbs = mem_block_size
        self.block = Block(**config)
        self.comp = FFTResampler(mbs*2, mbs, mbs+1)
        self.ln = LayerNorm(**config)
        
    def forward(self, x):
        while x.shape[0] > 1:
            b, t, c = x.size()
            x = x.transpose(1,2)
            x = self.comp(x)
            x = x.transpose(1,2) # ?,t,c
            x = x.reshape(b // 2, 2, t // 2, c) #b//2, 2, t, c
            x = x.reshape(b // 2, t , c).contiguous() #b//2, 2 * t, c
            x = self.block(x, causal = False)
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

class ProcrustesLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias = None):
        
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.orthogonal_(self.weight)
    
    def compute_orthogonal_weight(self):
        return utils.fast_polar_decomposition(self.weight.to(torch.float32))
        
    def forward(self, x: torch.Tensor, weight_ortho: torch.Tensor =None) -> torch.Tensor:
        if weight_ortho is None:
            weight_ortho = self.compute_orthogonal_weight().to(self.weight.dtype)
        else:
            weight_ortho = weight_ortho
        
        return F.linear(x, weight_ortho, None)

class ProcrustesButterflyLinear(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        
        self.n = in_features

        self.proj1 = ProcrustesLinear(self.n, self.n)
        self.proj2 = ProcrustesLinear(self.n, self.n)
    
    def compute_orthogonal_weights(self):
        return self.proj1.compute_orthogonal_weight(), self.proj2.compute_orthogonal_weight()
    

    def forward(self, x: torch.Tensor, y: torch.Tensor, pw) -> torch.Tensor:
        y1_proj = self.proj1(x,pw[0])
        y2_proj = self.proj2(y,pw[1])
        
        y_out = y1_proj + y2_proj

        return y_out


class SinkhornLinear(nn.Module):
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
        if self.project_on_run:
            with torch.no_grad():
                self.project()
          
        #W = self.project(self.weight_logits, self.n_iter)
        
        return F.linear(x, self.weight)
    @torch.no_grad()
    def project(self):
        W = self.weight.clamp(min=1e-8)
        for _ in range(self.n_iter):
            W = W / W.sum(dim=1, keepdim=True)
            W = W / W.sum(dim=0, keepdim=True)
        self.weight.copy_(W)

class GdiffLinearfunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, weight_ema, bias, config):
        w_eff = utils.VotedGeometricMean(weight, weight_ema)
        ctx.save_for_backward(input, weight, weight_ema, w_eff)
        ctx.config = config
        return F.linear(input,w_eff,bias) 

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, weight_ema, w_eff = ctx.saved_tensors
        conf = ctx.config
        
        grad_input = None
        grad_weight = None
        grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)

        # --- 2. Gradient w.r.t Weight ---
        if ctx.needs_input_grad[1]:
            input_flat = input.reshape(-1, input.shape[-1])
            grad_flat = grad_output.reshape(-1, grad_output.shape[-1])
            
            raw_grad = grad_flat.t().mm(input_flat)
            
            g_norm = utils.rms(raw_grad)

            # C. Regularization
            wreg = weight
            reg = conf['lambda'] * wreg.sign() * (wreg.abs().pow(conf['reg_dim'] - 1))
            g_final = g_norm + reg
            
            # D. The "gdiff" Scheduler
            aw_diff = weight_ema - w_eff
            denom = weight_ema.abs()*0.5 + (g_final - aw_diff).abs()
            g_diff = 1.0 / (denom + 1e-9)
            
            # E. Final Gradient
            grad_weight = g_final * g_diff
            #grad_weight += torch.randn_like(grad_weight) * 1e-5 * (g_final - aw_diff).abs()
            smoothing = conf['smoothing']

            if 'grow' in conf and conf['grow'] < 1.0:
                rows = grad_weight.shape[0]
                neuron_idx = torch.arange(rows, device=grad_weight.device).unsqueeze(1)
                limit = rows * conf['grow']
                gkernel = torch.where(neuron_idx < limit, 1.0, 0.0)

                grad_weight *= gkernel
                # --- SIDE EFFECT: Update EMA ---
                smoothing *= gkernel
            
            weight_ema.add_(smoothing * (weight - weight_ema))

        # --- 3. Gradient w.r.t Bias ---
        if ctx.needs_input_grad[3] and grad_output is not None:
            # Collapse Batch and Time dimensions
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)


        return grad_input, grad_weight, None, grad_bias, None
phi = (1+math.sqrt(5))/2
def flattening_kernel(x, y, power=2.0, sensitivity=1.0):

    y = torch.as_tensor(y).clamp(min=1e-6, max=1.0)
    
    inv_width = sensitivity * (1.0 - y) / y
    
    kernel = torch.exp( - (x.abs() * inv_width).pow(power) )
    
    return kernel
class GdiffLinear(nn.Module):
    def __init__(self, in_features, out_features, 
                 smoothing=0.01, lambda_reg=0.0, reg_dim=2.0):
        super().__init__()
        self.in_features = in_features
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_normal_(self.weight)
        
        self.register_buffer('weight_ema', self.weight.clone().detach())
        
        self.weight.data.zero_()
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        self.config = {
            'smoothing': smoothing,
            'lambda': lambda_reg,
            'reg_dim': reg_dim,
            'grow': 0,
        }
        self.num = 10000.0
        
    def forward(self, x):
        with torch.no_grad():
            self.num += 1.0
            self.config['grow'] = self.num / 10000
        return GdiffLinearfunc.apply(
            x, self.weight, self.weight_ema, self.bias, self.config
        )
class GdiffEmbedding(nn.Module):
    def __init__(self, num, dim, smoothing=0.01):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num, dim))
        nn.init.normal_(self.weight)
        self.register_buffer('weight_ema', self.weight.clone().detach())
        self.smoothing = smoothing

    def forward(self, input):
        # Sparse EMA Update
        if self.training:
            with torch.no_grad():
                idx = input.unique()
                self.weight_ema[idx].lerp_(self.weight[idx], self.smoothing)

        w = F.embedding(input, self.weight)
        w_ema = F.embedding(input, self.weight_ema)

        mag = (w * w_ema).abs().sqrt()
        sign = (w + w_ema).sign()
        
        return sign * mag

class MLayer(nn.Module):
    def __init__(self, 
                 input_dim, 
                 dim_m, 
                 bias=False, 
                 matrix_squarings_exp=None):
        super(MLayer, self).__init__()
        self.input_dim = input_dim
        self.dim_m = dim_m
        self.with_bias = bias
        self.matrix_squarings_exp = matrix_squarings_exp

        self.rep_to_exp_tensor = nn.Parameter(
            torch.empty(input_dim, dim_m, dim_m)
        )

        if self.with_bias:
            self.matrix_bias = nn.Parameter(torch.empty(1, dim_m, dim_m))
        else:
            self.register_parameter('matrix_bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.rep_to_exp_tensor)
        if self.with_bias:
            nn.init.uniform_(self.matrix_bias, -0.05, 0.05)

    def forward(self, x):
        weight_flat = self.rep_to_exp_tensor.view(self.input_dim, -1)
        
        flat_output = torch.matmul(x, weight_flat)
        
        mat = flat_output.view(-1, self.dim_m, self.dim_m)

        if self.with_bias:
            mat = mat + self.matrix_bias

        if self.matrix_squarings_exp is None:
            return torch.matrix_exp(mat)

        scale_factor = 0.5 ** self.matrix_squarings_exp
        identity = torch.eye(self.dim_m, device=mat.device, dtype=mat.dtype)
        
        mat = mat * scale_factor + identity

        for _ in range(self.matrix_squarings_exp):
            mat = torch.matmul(mat, mat)
            
        return mat
class HungryLinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, config):
        ctx.save_for_backward(input, weight, bias)
        ctx.config = config
        return F.linear(input, weight, bias)
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        conf = ctx.config
        
        noise = torch.randn_like(input) / ( 1e-5 + utils.perplexity(input))
        
        input = input + noise
        #grad_output = utils.rms(grad_output)
        input_flat = input.reshape(-1, input.shape[-1])
        grad_flat = grad_output.reshape(-1, grad_output.shape[-1])
        
        grad_weight = grad_flat.t().matmul(input_flat)
        
        alpha = conf.get('lookahead_alpha', 0.1)
        
        #grad_weight = utils.rms(grad_weight)
        future_weight = weight - (grad_weight * alpha)
        
        grad_input = grad_output.matmul(future_weight)
        
        grad_bias = None
        if bias is not None:
            grad_bias = grad_flat.sum(0) if grad_output is not None else None
        #grad_input = utils.rms(grad_input)
        return grad_input, grad_weight, grad_bias, None

class HungryLinear(nn.Module):
    def __init__(self, in_features, out_features, bias = False):
        super().__init__()
        self.in_features = in_features
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        #nn.init.kaiming_normal_(self.weight)
        
        self.weight.data.zero_()
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        
        self.config = {
            'lookahead_alpha' : 1e-4,
        }
        
    def forward(self, x):
        return HungryLinearFunc.apply(
            x, self.weight, self.bias, self.config, 
        )    

class HungryGdiffLinearfunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, weight_ema, bias, config):
        w_eff = utils.VotedGeometricMean(weight, weight_ema)
        ctx.save_for_backward(input, weight, weight_ema, w_eff)
        ctx.config = config
        
        return F.linear(input,w_eff,bias) 


    @staticmethod
    def backward(ctx, grad_output):
        input, weight, weight_ema, w_eff = ctx.saved_tensors
        conf = ctx.config
        
        grad_input = None
        grad_weight = None
        grad_bias = None

        
        # --- 2. Gradient w.r.t Weight ---
        
        noise = torch.randn_like(input) / ( 1e-5 + input.sum())
        input = input + noise
        input_flat = input.reshape(-1, input.shape[-1])
        grad_flat = grad_output.reshape(-1, grad_output.shape[-1])
        
        raw_grad = grad_flat.t().mm(input_flat)
        
        adjusted_grad = raw_grad
        
        smoothing = conf['smoothing']
        # D. The "gdiff" Scheduler
        aw_diff = weight_ema - w_eff
        denom = weight_ema.abs() + (adjusted_grad - aw_diff).abs()  #joy a hyper param.
        g_diff = 1.0 / torch.sqrt(denom + 1e-9)

        adjusted_grad = adjusted_grad * g_diff
        gw2 = adjusted_grad
        adjusted_grad = utils.rms(adjusted_grad)
        # E. Final Gradient
        grad_weight = adjusted_grad 
        weight_ema.add_(smoothing * (weight - weight_ema))


        if ctx.needs_input_grad[0]:
            
            #gw2 = gfi
            #gw2 = raw_grad
            #gw2 = grad_weight
            alpha = conf.get('lookahead_alpha', 0.1)

            future_weight = weight - (gw2 * alpha)
            future_weight = utils.VotedGeometricMean(future_weight, weight_ema)
            
            #future_weight = weight_ema - (gw2 * alpha)
            grad_input = grad_output.matmul(future_weight)
        
            #grad_input = grad_output.matmul(weight)

        if ctx.needs_input_grad[3] and grad_output is not None:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)


        return grad_input, grad_weight, None, grad_bias, None
    
class HungryGdiffLinear(nn.Module):
    def __init__(self, in_features, out_features, bias = False,
                 smoothing=0.01, lambda_reg=0.0, reg_dim=2.0, lookahead= 1e-4):
        super().__init__()
        self.in_features = in_features
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_normal_(self.weight)
        
        self.register_buffer('weight_ema', self.weight.clone().detach())
        
        self.weight.data.zero_()
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        
        
        self.config = {
            'smoothing': smoothing,
            'lambda': lambda_reg,
            'reg_dim': reg_dim,
            'grow': 0,
            'lookahead_alpha': lookahead,
        }
        
    def forward(self, x):
        return HungryGdiffLinearfunc.apply(
            x, self.weight, self.weight_ema, self.bias, self.config
        ) 


class HungryVGMLinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, weight_ema, bias, config):
        w_eff = utils.VotedGeometricMean(weight, weight_ema)
        ctx.save_for_backward(input, weight, weight_ema, w_eff)
        ctx.config = config
        
        return F.linear(input,w_eff,bias) 


    @staticmethod
    def backward(ctx, grad_output):
        input, weight, weight_ema, w_eff = ctx.saved_tensors
        conf = ctx.config
        
        grad_input = None
        grad_weight = None
        grad_bias = None

        noise = torch.randn_like(input) / ( 1e-5 + input.sum())
        input = input + noise
        input_flat = input.reshape(-1, input.shape[-1])
        grad_flat = grad_output.reshape(-1, grad_output.shape[-1])
        
        raw_grad = grad_flat.t().mm(input_flat)
        
        grad_weight = raw_grad
        #grad_weight = utils.rms(grad_weight)
        
        smoothing = conf['smoothing']
        weight_ema.add_(smoothing * (weight - weight_ema))

        if ctx.needs_input_grad[0]:
            
            #gw2 = raw_grad
            gw2 = grad_weight
            alpha = conf.get('lookahead_alpha', 0.1)

            future_weight = weight - (gw2 * alpha)
            future_weight = utils.VotedGeometricMean(future_weight, weight_ema)
            
            #future_weight = weight_ema - (gw2 * alpha)
            grad_input = grad_output.matmul(future_weight)
        
            #grad_input = grad_output.matmul(weight)

        if ctx.needs_input_grad[3] and grad_output is not None:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)


        return grad_input, grad_weight, None, grad_bias, None
    
class HungryVGMLinear(nn.Module):
    def __init__(self, in_features, out_features, bias = False,
                smoothing = 0.01, lookahead= 1e-4):
        super().__init__()
        self.in_features = in_features
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_normal_(self.weight)
        
        self.register_buffer('weight_ema', self.weight.clone().detach())
        
        self.weight.data.zero_()
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        
        self.config = {
            'lookahead_alpha': lookahead,
            'smoothing': smoothing,
        }
        
    def forward(self, x):
        return HungryVGMLinearFunc.apply(
            x, self.weight, self.weight_ema, self.bias, self.config
        ) 
