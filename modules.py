
import math
from tabnanny import check

from numpy import shape
import torch
import torch.nn as nn
from torch.nn import functional as F
import utils
import quantizer
from torch.optim.optimizer import Optimizer

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
    
    def get_weight(self):
        return self.scanner.compute_orthogonal_weights()

    def forward(self, x: torch.tensor, weight = None): #same signature to allow easy swapping.
        B, T, C = x.size() 
        q = self.q_attn(x) 
        v = self.v_attn(x)
        if weight is None:
            weight = self.scanner.compute_orthogonal_weights()
        def scan(left, right):
            z = self.scanner(left,right,weight).to(x.dtype)
            return utils.range_norm(z)
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

    def forward(self, x,  causal = True):
        x = x + self.attn(self.ln_1(x),causal)
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
    def __init__(self, config):
        super().__init__()
        self.block = Block(config.n_embd, config.n_head, config.dropout, config.bias)
        self.comp = FFTResampler(config.mem_block_size, config.mem_block_size//2, 64)
        self.ln = LayerNorm(config.n_embd, config.bias)
        
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
    def __init__(self, in_features: int, out_features: int):
        
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_param = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.orthogonal_(self.weight_param)
    
    def compute_orthogonal_weight(self):
        return utils.fast_polar_decomposition(self.weight_param.to(torch.float32))
        
    def forward(self, x: torch.Tensor, weight_ortho: torch.Tensor =None) -> torch.Tensor:
        if weight_ortho is None:
            weight_ortho = self.compute_orthogonal_weight().to(self.weight_param.dtype)
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