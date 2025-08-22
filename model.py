"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

from numpy import float32
from pytorch_wavelets import DWT1DForward
import torch
import torch.nn as nn
from torch.nn import functional as F
import optim
import tritonpscan
import utils
import wackmodel
from cut_cross_entropy import linear_cross_entropy

#todo
#prediction machine
#Linear = optim.OptimizedLinear
Linear = nn.Linear
#settings :)
#known good
diffSplits      = 1
ssmax           = True
mtp             = False
#ungraded
fftmem          = True
mmnorm          = False
normless        = False
#good in owt
qnorm           = False
#good in shkspr
mix_squish      = False
mem_mix_squish  = False #or fftmem
#bad
CausalMem       = False
convemb         = False
diffembd        = False
qope            = False
side_squish     = False
#slow
spl             = False #or fftmem
qrot            = False
think           = False
repeat_block    = False
repeatCenter    = False

Qrotention      = False
symloss         = False
midchaos        = False

specl           = False
topoloss        = False
acebtl          = False
acl1            = False # unimplemented

nmix            = False
losspred        = False

auxmod          = acl1 or acebtl #or topoloss

blockin         = spl or symloss or midchaos or topoloss or specl


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.n_embd))
        self.bias = nn.Parameter(torch.zeros(config.n_embd)) if config.bias else None
       

    def forward(self, input):
        #if(tests):
        #    return input
        if(normless):
            return input
        if(mmnorm):
            return utils.soft_range_clip_norm(input, dim=-1, scale=self.weight) 
            #return torch.tanh(input)*self.weight.norm(p=2,keepdim=True)
           # return utils.acwrap(input*self.weight, torch.tanh, self.bias)
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class PrecomputedFourierResampler(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_filters: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_filters = num_filters

        frequencies = torch.linspace(1, input_dim // 2, num_filters)
        self.register_buffer("frequencies", frequencies)
        t_in = torch.linspace(0, 1, input_dim)
        t_out = torch.linspace(0, 1, output_dim)
        freqs = self.frequencies.unsqueeze(1)
        arg_in = 2 * math.pi * freqs * t_in.unsqueeze(0)
        self.register_buffer("sin_basis_in", torch.sin(arg_in))
        self.register_buffer("cos_basis_in", torch.cos(arg_in))
        arg_out = 2 * math.pi * freqs * t_out.unsqueeze(0)
        self.register_buffer("sin_basis_out", torch.sin(arg_out))
        self.register_buffer("cos_basis_out", torch.cos(arg_out))

    def forward(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
     
        if x.shape[dim] != self.input_dim:
            raise ValueError(
                f"Input tensor dimension {dim} has size {x.shape[dim]}, "
                f"but module was initialized with input_dim {self.input_dim}."
            )

        x_permuted = x.transpose(dim, -1)
        original_shape = x_permuted.shape
        x_flattened = x_permuted.reshape(-1, self.input_dim)

        # the Fourier coefficients (the compressed representation).
        c_sin = torch.einsum('ns,fs->nf', x_flattened, self.sin_basis_in)
        c_cos = torch.einsum('ns,fs->nf', x_flattened, self.cos_basis_in)

        # precomputed SYNTHESIS bases using the calculated coefficients.
        sin_recon = torch.einsum('nf,fs->ns', c_sin, self.sin_basis_out)
        cos_recon = torch.einsum('nf,fs->ns', c_cos, self.cos_basis_out)

        x_reconstructed = sin_recon + cos_recon

        output_shape = original_shape[:-1] + (self.output_dim,)
        output = x_reconstructed.view(output_shape)
        output = output.transpose(dim, -1).contiguous()

        return output

nlfft = True

class LearnableFourierResampler(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_filters: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_filters = num_filters

        # A single set of learnable frequencies defines the filter bank.
        # We initialize them based on the input dimension for stability.
        self.frequencies = nn.Parameter(torch.linspace(1, input_dim // 2, num_filters))
        
        # We need two separate time axes: one for analysis, one for synthesis.
        t_in = torch.linspace(0, 1, input_dim, device=self.frequencies.device)
        t_out = torch.linspace(0, 1, output_dim, device=self.frequencies.device)
        self.register_buffer("t_in", t_in)
        self.register_buffer("t_out", t_out)

    def _get_bases(self):
        """Helper to generate both analysis and synthesis bases."""
        freqs = self.frequencies.unsqueeze(1)
        
        # Analysis bases (for the input signal)
        arg_in = 2 * math.pi * freqs * self.t_in.unsqueeze(0)
        sin_basis_in = torch.sin(arg_in)
        cos_basis_in = torch.cos(arg_in)
        
        # Synthesis bases (for the output signal)
        arg_out = 2 * math.pi * freqs * self.t_out.unsqueeze(0)
        sin_basis_out = torch.sin(arg_out)
        cos_basis_out = torch.cos(arg_out)
        
        return sin_basis_in, cos_basis_in, sin_basis_out, cos_basis_out

    def forward(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [..., input_dim, ...].
            dim (int): The dimension to resample.
        """
        # --- Preparation ---
        assert x.shape[dim] == self.input_dim, \
            f"Input tensor dim {dim} has size {x.shape[dim]}, " \
            f"but module was initialized with input_dim {self.input_dim}."
            
        x_permuted = x.transpose(dim, -1)
        original_shape = x_permuted.shape
        x_flattened = x_permuted.reshape(-1, self.input_dim)
        
        # --- 1. ANALYSIS ---
        # Get all four basis sets.
        sin_basis_in, cos_basis_in, sin_basis_out, cos_basis_out = self._get_bases()
        
        # Project input signal onto the ANALYSIS bases to get coefficients.
        c_sin = torch.einsum('ns,fs->nf', x_flattened, sin_basis_in)
        c_cos = torch.einsum('ns,fs->nf', x_flattened, cos_basis_in)
        
        # `c_sin` and `c_cos` are the compressed representation.
        
        # --- 2. SYNTHESIS ---
        # Reconstruct a new signal by doing a weighted sum of the SYNTHESIS bases.
        sin_recon = torch.einsum('nf,fs->ns', c_sin, sin_basis_out)
        cos_recon = torch.einsum('nf,fs->ns', c_cos, cos_basis_out)
        
        # The new signal is the sum of the components.
        x_reconstructed = sin_recon + cos_recon # Shape: [N, output_dim]
        
        # --- Reshape Output ---
        # The last dimension now has size `output_dim`.
        output_shape = original_shape[:-1] + (self.output_dim,)
        output = x_reconstructed.view(output_shape)
        
        # Put the processed dimension back where it was.
        output = output.transpose(dim, -1).contiguous()
        
        return output
    
class Scanner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d = config.n_embd
        self.sampler = PrecomputedFourierResampler(config.n_embd * 2, config.n_embd, 64)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        z = torch.cat((y, x), dim=-1)
        z=self.sampler(z)
        z = utils.range_norm(z)
        return z

class QrotAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.c_attn2 =  Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)
        
        self.c_proj = Linear(config.n_embd,  config.n_embd, bias=config.bias)

        self.scanner = Scanner(config)
        self.identity = nn.Parameter(torch.rand(config.n_embd))
        

        self.attn_dropout = nn.Dropout(config.dropout) #todo
        self.resid_dropout = nn.Dropout(config.dropout) #todo
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.n_head = config.n_head 
        self.head_dim = config.n_embd//config.n_head * 2
        self.win = config.block_size // config.n_layer
     

    def forward(self, x: torch.tensor): #same signature to allow easy swapping.
        B, T, C = x.size() 
        q,   v = self.c_attn2(x).split(self.n_embd, dim=2)
        QC = q.shape[2]
        
        q = q.view(B, T, QC)
        v = v.view(B, T, C)

        q = utils.pscan(q, self.scanner, self.identity)
        
        Y = q*v 
        Y = Y.view(B, T, -1)
        Y = self.c_proj(Y)
        Y = self.resid_dropout(Y)
        return Y
    

class MemAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = Linear(config.n_embd,  config.n_embd, bias=config.bias)
        
        #attention mod modifier
        self.attmod = Linear(config.n_embd,  config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head 
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim =  self.n_embd //  self.n_head 
        
        self.qm = nn.Parameter(torch.ones(self.head_dim).normal_(mean=1, std=0.1))
        
        #self.conv = wackmodel.CausalConv1d2()
        # Memory positions are always visible
        self.mem_len = config.mem_block_size
        mask = torch.zeros((self.mem_len, self.mem_len*2),dtype=torch.bfloat16)
        # Enforce causality for non-memory positions
        for t in range(self.mem_len):
            mask[t, self.mem_len + t + 1:] = float('-inf')
        self.register_buffer("attn_mask", mask)

    def forward(self, x, causal=True):
        B, T, C = x.size() 
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)

        y = torch.nn.functional.scaled_dot_product_attention(q*(math.log(T)*self.qm),k,  v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal= causal)

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim) 
        y = self.resid_dropout(self.c_proj(y))

        return y
    
        #if mem is None:
        #    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        #    q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        #    k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        #    v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
#
        #    y = torch.nn.functional.scaled_dot_product_attention(q*(math.log(T)*self.qm),k,  v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal= causal)
#
        #    y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim) 
        #    y = self.resid_dropout(self.c_proj(y))
#
        #    return y
        #
        #q, k, v = self.c_attn(torch.cat((mem,x),dim=1)).split(self.n_embd, dim=2)
        #q = q.view(B, T+self.mem_len, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        #k = k.view(B, T+self.mem_len, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        #v = v.view(B, T+self.mem_len, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
#
        #y = torch.nn.functional.scaled_dot_product_attention(q*(math.log(T+self.mem_len)*self.qm), k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal= causal)
        #y = y.transpose(1, 2).contiguous().view(B, T+self.mem_len, self.n_head * self.head_dim) 
        #y = self.resid_dropout(self.c_proj(y))
#
        #return y#[:,self.mem_len:,:].contiguous(), y[:,:self.mem_len,:].contiguous()


class LearnableSpiral4D(nn.Module):
    def __init__(self, init_freq=1.0, init_amp=1.0):
        super().__init__()
        
        freqs = torch.ones(4) * init_freq + torch.randn(4) * 0.1
        self.frequencies = nn.Parameter(freqs)

        phases = torch.linspace(0, 2 * torch.pi, 5)[:-1] # 0, pi/2, pi, 3pi/2
        self.phase_shifts = nn.Parameter(phases)

        amps = torch.ones(4) * init_amp
        self.amplitude_scales = nn.Parameter(amps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 1:
            b, t, c = x.shape
            x = x.unsqueeze(-1)
        sin_term = torch.sin(self.frequencies * x + self.phase_shifts)
        
        amp_term = self.amplitude_scales * x
        
        output = sin_term * amp_term
        if x.shape[-1] != 1:
            output = output.view(b,t,c*4)
        return output


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        up = 4
        self.c_fc    = Linear(config.n_embd, up * config.n_embd, bias=config.bias)
        self.c_fc2    = Linear(config.n_embd, up * config.n_embd, bias=config.bias)
        if(qrot):
            self.c_proj  = Linear( 2 * config.n_embd, config.n_embd, bias=config.bias)
        else:
            self.gelu    = nn.GELU()
            self.c_proj  = Linear( up * config.n_embd, config.n_embd, bias=config.bias)
            self.c_proj2  = Linear( up * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.rx0 = nn.Parameter(torch.zeros(up * config.n_embd))
        self.n_embd = config.n_embd

    def forward(self, x):
        x = self.c_fc(x)
        #x2 = self.c_fc2(x)
        if(qrot):
            x = self.qrot(x)
        else:
            x = self.gelu(x)
        x = self.c_proj(x)
        #x2 = self.c_proj2(x)
        x = self.dropout(x)
        return x

    def qrot(self, x):
        b,t,n=x.shape
        x4d = x.view(b, t, self.n_embd, 4) # Reshape to (batch, seq_len, n_embd, 4)
        x_norm = torch.clamp_min(x4d.norm(dim=-1, keepdim=True), 1e-10) # Calculate norm along the last dimension (dim=3 or -1), keep dimensions
        x_normalized = x4d / x_norm             # Divide to normalize
        rotors, rotated = x_normalized.chunk(2, dim=2) # Split along n_embd dimension
            #xm,ym = x_norm.chunk(2, dim=2)
        x_rotated = utils.quaternion_multiply(rotors, rotated) 
            #x_rotated = x_rotated*self.gaprelu(x_norm-2)#*torch.sigmoid(xm+ym)#
        return x_rotated.view(b, t, 2 * self.n_embd)


class MemBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config)
        if(Qrotention):
            self.attn = QrotAttention(config)
        else:
            self.attn = MemAttention(config)
        self.ln_2 = LayerNorm(config)
        self.mlp = MLP(config)
        self.register_buffer("block_input",torch.zeros([config.block_size, config.n_embd]))
        if fftmem:
            self.register_buffer("mem_input",torch.zeros([config.block_size, config.n_embd]))

    def forward(self, x,  causal = True):
        self.block_input = x
        x = x + self.attn(self.ln_1(x),causal)
        x = x + self.mlp(self.ln_2(x))
        return x

    def spl(self, x, mem = None):
        if mem is not None:
            x = torch.cat([mem, x],dim=1)
        predicted_output = self(-x, causal = False)
        
        if mem is not None:
            predicted_output = predicted_output[:,mem.size(1):,:]
        sploss = F.mse_loss(
            utils.range_norm(predicted_output.flatten()),
            utils.range_norm(self.block_input.flatten())
        ) 
        return sploss
    
class Dreamer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.block = MemBlock(config)
        self.comp = LearnableFourierResampler(config.mem_block_size * 2, config.mem_block_size, 64)
        
    def forward(self, x):
        while x.shape[0] > 1:
            b, t, c = x.size()
            x = x.reshape(b // 2, 2, t, c)
            x = x.reshape(b // 2, t * 2, c).contiguous()
            #x = x.view(b//2, 2*t, c)
            x = x.transpose(1,2)
            x = self.comp(x)
            #x = utils.rfft_trunc_squish(x, target_dim= t, dim=1) # ?,t,c
            x = x.transpose(1,2)
            x = self.block(x, causal = CausalMem)

        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    internal_vocab_size_multi: int = 10
    internal_vocab_size: int = 0
    internal_block_size: int = 30
    device='cuda'
    optimizer: str = 'adam'
    mem_block_size: int = 32
    batch_size = 64
    step = 0

wn = torch.nn.utils.parametrizations.weight_norm
class wnormlearnloss(nn.Module):
    def __init__(self, dims):
        super().__init__()
        # Create one learnable log_sigma_sq parameter for each loss
        self.head = nn.Sequential(
                wn(Linear(dims, dims*2),dim=None),
                nn.GELU(),
                wn(Linear(dims*2, dims*2),dim=None),
                nn.GELU(),
                wn(Linear(dims*2, 1))
            )

    def forward(self, x):
        loss = self.head(x)
        return F.softplus(loss).mean()

class UncertaintyWeightedLoss(nn.Module):
    def __init__(self, num_losses: int):
        super().__init__()
        # Create one learnable log_sigma_sq parameter for each loss
        self.log_sigmas_sq = nn.Parameter(torch.zeros(num_losses), requires_grad=True)

    def forward(self, *losses):
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_sigmas_sq[i])
            regularization = self.log_sigmas_sq[i]
            
            total_loss += precision * loss + regularization
            
        return total_loss

    
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        config.internal_vocab_size = config.vocab_size * config.internal_vocab_size_multi
        
        self.memory_frozen = False 
        self.config = config
        self.ksize = 8
        self.patch_max = 10
        self.bembwidth = config.n_embd//2
        config.internal_block_size = config.block_size
        
        blocks = nn.ModuleList([MemBlock(config) for _ in range(config.n_layer)])
        self.denoiser = MemBlock(config) 
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            
            wpe = nn.Embedding(config.internal_block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = blocks,
            ln_f = LayerNorm(config),
        ))
        if(think):
            self.thinkthreshold = nn.Parameter(torch.ones(1))
            self.thinkstop = nn.Sequential(
                Linear(config.n_embd, config.n_embd),
                nn.GELU(),
                Linear(config.n_embd, 1)
            )
        if(qope):
            self.qoper = wackmodel.QrotMLP(config)
        if(diffembd):
            self.register_buffer('gmask', utils.gaussian_kernel(torch.ones(self.config.n_embd)))
        if(fftmem ):
            if(mem_mix_squish):
                self.register_buffer('memory', torch.zeros(1, config.internal_block_size, config.n_embd//2))
            else:
                self.register_buffer('memory', torch.zeros(1, config.mem_block_size, config.n_embd))
            if(config.dropout > 0.0):
                config.dropout = config.dropout / 4
            self.dreamer = Dreamer(config)
            self.memory_selector = MemBlock(config)
            
            self.mem_comp = LearnableFourierResampler(config.mem_block_size, config.mem_block_size//2, 32)


        self.surprise = 1
        if(convemb):
            self.convemb = wackmodel.Patcher(config)
            self.lm_head = Linear(config.n_embd, config.vocab_size * self.patch_max, bias=False)
        else:
            #it'd seem like this isn't worth optimized linear's hassle with apple's CCE loss 
            self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)
            # with weight tying when using torch.compile() some warnings get generated:
            # "UserWarning: functional_call was passed multiple values for tied weights.
            # This behavior is deprecated and will be an error in future versions"
            # not 100% sure what this is, so far seems to be harmless. TODO investigate
            #apparently bad
            self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
        


        if(auxmod):
            self.auxmlpin = 25
            self.auxmlpout = 25
            self.all_activations = None
            #self.saved_activations = []
            i=0
            for name, module in self.named_modules():
                if isinstance(module, nn.Linear):
                    # Each hook is created with its specific index (its slot)
                    def create_hook(slot_index):
                        def save_hook(module, input, output):
                            # This hook will only execute if the master tensor has been created
                            if self.all_activations is not None:
                                # Process the activation
                                if(output.shape[-1] == self.auxmlpin):
                                    squished = output #no need to squish.
                                else:
                                    squished = utils.rfft_trunc_squish(output.to(torch.float32), target_dim=self.auxmlpin)
                                self.all_activations[:, :, slot_index, :] = squished
                        return save_hook
    
                    # Register the hook for the current linear layer
                    module.register_forward_hook(create_hook(i))
                    i +=1
            self.num_slots = i
            
            self.aux_mlp = torch.nn.Sequential(
                            torch.nn.Linear(self.auxmlpin,self.auxmlpout*2),
                            torch.nn.GELU(),
                            torch.nn.Linear(self.auxmlpout*2,self.auxmlpout ),

                        )
        if topoloss:
            self.ucl = UncertaintyWeightedLoss(3)  
        #if mix_squish:
        self.c_comp = LearnableFourierResampler(config.n_embd,config.n_embd//2,config.n_embd//4)
        
        self.decomp = torch.nn.Sequential(
                            torch.nn.Linear(config.n_embd//2,config.n_embd),
                            torch.nn.GELU(),
                            torch.nn.Linear(config.n_embd,config.n_embd*2 ),
                            torch.nn.GELU(),
                            torch.nn.Linear(config.n_embd*2,config.n_embd ),

                        )
        if(losspred):
            self.lpb = MemBlock(config) 
            self.botl = torch.nn.Sequential(
                            torch.nn.Linear(config.n_embd,config.n_embd//2),
                            torch.nn.GELU(),
                            torch.nn.Linear(config.n_embd//2,config.n_embd//8 ),
                            torch.nn.GELU(),
                            torch.nn.Linear(config.n_embd//8, 1),

                        )
        #self.wnl = wnormlearnloss(config.n_embd*4)
        self.gradsink = nn.Parameter(torch.randn(config.n_embd))
        self.predict_weight = 0.01
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, module in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(module, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        self.laim = 0.1
        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def resetsink(self):
        with torch.no_grad():
            self.gradsink.data = torch.randn(self.config.n_embd, device=self.gradsink.device)

    def forward(self, idx, targets=None):
        #self.config.step += 1.0
        #torch.compiler.cudagraph_mark_step_begin()
        device = idx.device
        b, t = idx.size()
        if auxmod:
            self.all_activations = torch.zeros(
               b, t, self.num_slots, self.auxmlpin,
               device=idx.device,
               dtype=torch.float32 
            )
        #end_tok=torch.as_tensor(self.end_patch_token_id, device=device, dtype=torch.long)
        #patch_max=torch.as_tensor(self.patch_max, device=device, dtype=torch.long)
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (b, t, n_embd)
        x = tok_emb + pos_emb
        if qope:
            pos_emb = pos_emb.view(1, t, self.config.n_embd//4, 4)
            tok_emb = tok_emb.view(b, t, self.config.n_embd//4, 4)
            g1 = torch.sigmoid(pos_emb[...,0].view(1, t, self.config.n_embd//4, 1))
            g2 = torch.sigmoid(tok_emb[...,0].view(b, t, self.config.n_embd//4, 1))
            pos_emb = utils.exponential_map(pos_emb[...,1:]*g1*g2)
            x = utils.quaternion_multiply(tok_emb, pos_emb) 
            x = x.view(b,t,self.config.n_embd)
        
        x = self.transformer.drop(x)

        if mtp:
            return self.mtp_fwd(targets, x)
            
        if fftmem:
            if mem_mix_squish:
                return self.mem_mix_fwd(targets, x)
            else:
                return self.mem_fwd(targets, x)

        if blockin:
            block_inputs = []

        #if(Qrotention):
        if qnorm:
            x = utils.quatnorm(x) #doesn't work in shakespeare char by default
        #tok_emb += self.gradsink * 1e-5
        
        if mix_squish:
            fh = self.c_comp(x,dim=-1)
        for i, block in enumerate(self.transformer.h):
            #if (i==1):
            #    fs = torch.fft.rfft(x, self.config.n_embd-2)
            #    x = torch.cat([fs.real, fs.imag],dim=-1)
            if (i>1):
                if mix_squish:
                    # and i < self.config.n_layer-1):
                    sh = self.c_comp(x,dim=-1)
                    x = torch.stack([sh,fh],dim =-1).flatten(start_dim=-2,end_dim=-1)
            if blockin:
                block_inputs.append(x) 
            
            if nmix:
                # and i < self.config.n_layer-1):
                sh = self.c_comp(x,dim=-1)
                fh = torch.randn_like(sh)
                x = torch.cat([sh,fh],dim=-1)

            #xcomp = self.fcomp(x)
            #xguess = self.decomp(xcomp)
            #dcloss = F.mse_loss(x,xguess,reduction="none")#.mean(dim=2)
            #x = x + x * F.tanh(dcloss)
            x = block(x) 

        if acebtl:
            x, acl = self.acebtl(x, tok_emb)
        
        #xcomp = self.fcomp(x)
        #xguess = self.decomp(xcomp)
        #dcloss = F.mse_loss(x,xguess,reduction="none")#.mean(dim=2)
        #x = x + x * F.tanh(dcloss)#* F.sigmoid(1/dcloss)#.unsqueeze(-1)
        #noise = torch.randn_like(x)
        #nguess = self.denoiser(x+noise)
        #dnloss = F.mse_loss(nguess,noise,reduction="none").mean(dim=2)
        #x = x * F.sigmoid(1/dnloss).unsqueeze(-1)

        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = None# self.lm_head(x)
            reduction= 'none'
            if(not self.training):
                reduction = 'mean'
            
            loss = linear_cross_entropy(x.to(torch.bfloat16), self.lm_head.weight.to(torch.bfloat16), targets, impl='torch_compile', reduction=reduction)
            
            #TODO: wavelet loss

            if self.training:

                if losspred:
                    loss = (loss + F.mse_loss(self.botl(self.lpb(x)).squeeze(-1), loss.detach()))/2.0

                if auxmod:
                    if topoloss:
                        pass
                    if acebtl:
                        loss = loss+acl     
                    
                if midchaos:
                    loss = loss + torch.nn.functional.mse_loss(utils.range_norm(block_inputs[2], dim=[-1,-2]), utils.zca_newton_schulz(block_inputs[2].view(-1, self.config.n_embd)).view(b, t, self.config.n_embd))*1e-2

                if symloss:
                    for i in block_inputs:
                        loss = loss + utils.qisp_latent_dissonance_loss(i) * 0.01
                        #lcloss, lmean = utils.calculate_linear_chaos_loss(i, balance_lambda=1.0,target_chaos=self.laim)
                        #self.laim = max(lmean.detach(),self.laim)
                        #loss = loss + lcloss#utils.calculate_linear_chaos_loss(i, balance_lambda=0.1,target_chaos=0.5)
                        #loss = loss + 0.001 * torch.nn.functional.mse_loss(torch.abs(torch.fft.fft(i)),torch.softmax(torch.abs(torch.fft.fft(i)), -1))
                if spl:
                    loss = loss + self.self_prediction_loss(x) #* self.predict_weight #* 0.001
                
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def acebtl(self, x, tok_emb):
        if(self.auxmlpin != self.config.n_embd):
            xsquish = utils.rfft_trunc_squish(x,target_dim=self.auxmlpin)
        else:
            xsquish = x
                #with torch.enable_grad():
        pred = self.aux_mlp(xsquish)

        if self.training:
                    #xsquish= xsquish[:,:-1,:] # utils.rfft_trunc_squish(x[:,:-1,:],target_dim=self.auxmlpin)
                    #pred = pred[:,:-1,:]# self.aux_mlp(xsquish)
            if(self.auxmlpout != self.config.n_embd):
                auxtarget = utils.rfft_trunc_squish(tok_emb[:,1:,:],target_dim=self.auxmlpout) #too lazy to embed the targets sue me.
            else:
                auxtarget = tok_emb[:,1:,:]
            L1 = torch.nn.functional.mse_loss(pred[:,:-1,:], auxtarget)
                    #for i in range(4):
            grads = torch.autograd.grad(L1, self.aux_mlp.parameters(), create_graph=True)
                    #with torch.no_grad():
            updated_params = [
                        p - 0.1 * g if g is not None else p
                        for p, g in zip(self.aux_mlp.parameters(), grads)
                    ]
                    # Manually apply the updated weights
            h = torch.nn.functional.linear(xsquish, updated_params[0], updated_params[1])
            h = torch.nn.functional.gelu(h)
            pred = torch.nn.functional.linear(h, updated_params[2], updated_params[3])
            L1 = torch.nn.functional.mse_loss(pred[:,:-1,:], auxtarget)
                    
                    #if not self.training:
                    #    self.aux_mlp.zero_grad(set_to_none=True) #ensure no cheating probably not needed
            acl = L1 #added to loss
                #else:
        if(self.auxmlpout != self.config.n_embd):
            pf = torch.fft.rfft(pred.to(torch.float32),self.auxmlpout)
            xf = torch.fft.rfft(x.to(torch.float32),self.config.n_embd) 
            nc = pf.shape[-1]
            nt = pf.shape[-2]
            xf[:,:nt,:nc] = pf #surgery! could be mistaken for torture!
        else:
            x = pred
                
        x = torch.fft.irfft(xf,self.config.n_embd)
        return x,acl
    

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
            #{'params': self_lr, 'lr': 1.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        match self.config.optimizer:
            case "adam":
                optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
            case "sgd":
                optimizer = torch.optim.SGD(optim_groups, lr=learning_rate, **extra_args)
            case "phi2":
                optimizer = wackmodel.Phi2Optimizer(optim_groups,lr=learning_rate, **extra_args)
            
            #optimizer =torch.optim.Adadelta(optim_groups)#, **extra_args)# lr=learning_rate, **extra_args)
        print(f"using fused AdamW: {use_fused}")
                        
        return optimizer

    def self_grad_fwd(self, targets, x):
        device = x.device
        b, t, c = x.size()
        x = x.view(b, t, self.config.n_embd//4, 4)
        x = x /  torch.norm(x, p=2, dim=-1, keepdim=True)
        x = x.view(b, t, -1)
        for i, block in enumerate(self.transformer.h):
            x = block(x) 

        x = self.transformer.ln_f(x)
        loss = linear_cross_entropy(x.to(torch.bfloat16), self.lm_head.weight.to(torch.bfloat16), targets, impl='torch_compile', reduction='mean')
        loss.backward()
        blockingrad = 0
        for i, block in enumerate(self.transformer.h):
            gradguess = block(blockingrad) 

        if targets is not None:
            logits = None# self.lm_head(x)
            reduction= 'none'
            if(not self.training):
                reduction = 'mean'

            loss = linear_cross_entropy(x.to(torch.bfloat16), self.lm_head.weight.to(torch.bfloat16), targets, impl='torch_compile', reduction=reduction)
            
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss
    
    def mem_mix_fwd(self, targets, x):
        b, t, c = x.size()
        block_inputs = []

        if(self.training):
            xswish = torch.cat([x[b//2:,:,],x[:b//2,:,:]],dim=0)
            fh = self.memory.expand(b,t,self.config.n_embd//2)
            for i, block in enumerate(self.transformer.h):
                if(self.config.mix_mask[i]):
                    sh = utils.rfft_trunc_squish(xswish, target_dim= c//2,dim=-1)
                    xswish = torch.stack([fh,sh],dim =-1).flatten(start_dim=-2,end_dim=-1)
                xswish = block(xswish)
            first_mem = self.mix_mem_form(xswish,fh)
        else:
            first_mem = self.memory
        
        fh = first_mem.expand(b,t,self.config.n_embd//2)#(b,t,c//2)
        
        for i, block in enumerate(self.transformer.h):
            block_inputs.append(x) 
            x = block(x) 

            if(self.config.mix_mask[i]):
                sh = utils.rfft_trunc_squish(xswish, target_dim= c//2,dim=-1)
                x = torch.stack([sh,fh],dim =-1).flatten(start_dim=-2,end_dim=-1)
            
        updated_mem = None
        if self.training and not self.memory_frozen:
            updated_mem     = self.mix_mem_form(x,fh)
            with torch.no_grad():
                self.memory.copy_(updated_mem)

        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1) 
            
            if(self.training):
                loss = loss + self.self_prediction_loss(x) 
                loss = loss + self.memory_selector.spl(self.memory, x)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
        return logits, loss
    
    def mem_fwd(self, targets, x):
        b, t, c = x.size()
        malpha= 1.0#todo

        mit = self.memory.expand(b,self.config.mem_block_size,self.config.n_embd)
        
        if(self.training):
            xswish = torch.cat([x[b//2:,:,],x[:b//2,:,:]],dim=0)
            
            xswish = torch.cat((mit,xswish),dim=1)
            for block in self.transformer.h:
                xswish = block(xswish)
            mit = xswish[:,:self.config.mem_block_size,:]
            xswish = xswish[:,self.config.mem_block_size:,:]
            xh1 = xswish[b//2:,:,:]
            xh2 = xswish[:b//2,:,:]
            #mh1 = utils.range_norm(self.memory + self.mem_form(xh1)*malpha, dim=[-1,-2])
            #mh2 = utils.range_norm(self.memory + self.mem_form(xh2)*malpha, dim=[-1,-2])
            mh1 = (self.memory + self.mem_form(xh1)*malpha)
            mh2 = (self.memory + self.mem_form(xh2)*malpha)
            mh1 = mh1.expand(b//2, self.config.mem_block_size, self.config.n_embd)
            mh2 = mh2.expand(b//2, self.config.mem_block_size, self.config.n_embd)
            mit = torch.cat([mh1,mh2],dim =0) #disallow cheating as much as possible
        
        x = torch.cat((mit,x),dim=1)
        for i, block in enumerate(self.transformer.h):
            x = block(x)
        mit = x[:,:self.config.mem_block_size,:]
        x   = x[:,self.config.mem_block_size:,:]
            
        updated_mem = None
        if self.training and not self.memory_frozen :
            updated_mem = self.mem_form(x)
            with torch.no_grad():
                #pass
                #self.memory.add_(updated_mem)
                #self.memory = utils.range_norm(self.memory+updated_mem*malpha, dim=[-1,-2])
                self.memory = (self.memory+updated_mem*malpha)

        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1) 
            #loss = linear_cross_entropy(x.to(torch.bfloat16), self.lm_head.weight.to(torch.bfloat16), targets, impl='torch_compile')
            
            if(self.training):
                if spl:
                    loss = loss + self.self_prediction_loss(x) #*0.001
                    um = utils.range_norm(self.memory+updated_mem*malpha, dim=[-1,-2]).expand(b,self.config.mem_block_size,c)
                    loss = loss + self.memory_selector.spl(um) #*0.001
                
                
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None 
        return logits, loss
    
    def mtp_fwd(self, targets, x):
        if targets is not None:
            pathcount = 1
            if(self.training):
                pathcount = 4
            loss = 0
            firstLogit = None
            for pi in range(pathcount):
                if(pi==0):
                    for i in range(self.config.n_layer):
                        x = self.transformer.h[i](x)
                    x2 = self.transformer.ln_f(x)
                    logits = self.lm_head(x2)
                    firstLogit = logits
                else:
                    x = self.transformer.h[self.config.n_layer-1](x)
                    x2 = self.transformer.ln_f(x)
                    logits = self.lm_head(x2)
                stage_logits = logits[:, :-(pi+1), :].contiguous()
                stage_targets = targets[:, pi+1:].contiguous()

                loss = loss + F.cross_entropy(stage_logits.view(-1, stage_logits.size(-1)), stage_targets.view(-1), ignore_index=-1) 
                loss = loss / pathcount
                return firstLogit, loss
        else:
            for block in self.transformer.h:
                x = block(x)
            x = self.transformer.ln_f(x)
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
            return firstLogit, loss
    
    
    def self_prediction_loss(self,  x ):
        loss = 0
        for i, block in enumerate(self.transformer.h):
            loss = loss + block.spl(x) 
        return loss
    
    
    def mix_mem_form(self,x,fh):
        selectedMemory  = self.memory_selector(x, causal = False)#(b,t,c)
        selectedMemory  = utils.fft_trunc_csquish(selectedMemory) #(b,t,c//2)
        updated_mem     = torch.stack([selectedMemory,fh],dim =-1).flatten(start_dim=-2,end_dim=-1) #(b,t,c)
        updated_mem     = self.dreamer(updated_mem)
        return utils.fft_trunc_csquish(updated_mem)#(b,t,c//2)

    def mem_form(self, x):
        #currently assumes mem size and block size are the same
        b, t, c = x.size()
        #squishmem       = utils.rfft_trunc_squish(memory, target_dim= mt//2,dim=1).expand(b, self.config.mem_block_size//2, self.config.n_embd) #(b, m//2,c)
        
        #squishmem = self.mem_comp(memory.transpose(1,2).contiguous()).transpose(1,2)
        #squishmem = squishmem.expand(b, self.config.mem_block_size//2, self.config.n_embd)
        
        selectedMemory  = self.memory_selector(x, causal = False)[:,:self.config.mem_block_size,:] #(b, m, c)
        
        #selectedMemory  = utils.rfft_trunc_squish(selectedMemory, target_dim= mt//2,dim=1) #(b, m//2,c)
        
        #selectedMemory  = self.mem_comp(selectedMemory.transpose(1,2)).transpose(1,2)
        #mh1             = torch.cat([selectedMemory,squishmem],dim =1)       #(b, m,c)
        
        mh1             = selectedMemory
        mh1             = self.dreamer(mh1)
        return mh1#[:, :self.config.mem_block_size, :]

    
    def reinit_nonmem(self):
        # init all weights
        self.apply(self._init_nonmem_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        #for pn, p in self.named_parameters():
        #    if pn.endswith('c_proj.weight'):
        #        torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.config.n_layer))

    def _init_nonmem_weights(self, module):
        if module is self.memory or module is self.dreamer or module is self.memory_selector:
            return
        if isinstance(module, Linear):
            torch.nn.init.normal_(module.weight, mean=module.weight.mean()/2, std=(module.weight.max()-module.weight.min() + 1e-10))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=module.weight.mean()/2, std=(module.weight.max()-module.weight.min() + 1e-10))
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.normal_(module.weight, mean=module.weight.mean()/2, std=(module.weight.max()-module.weight.min() + 1e-10))

    def memory_freezeToggle(self):
        self.memory_frozen = not self.memory_frozen
        if hasattr(self, 'memory_selector') and self.memory_selector is not None:
            for param in self.memory_selector.parameters():
                param.requires_grad = not self.memory_frozen
        if hasattr(self, 'dreamer') and self.dreamer is not None:
            for param in self.dreamer.parameters():
                param.requires_grad = not self.memory_frozen
                
    def get_patch_lengths(self, logits, end_token_idx, seq_len):
        """Returns tensor of patch lengths (batch_size,) based on first end token occurrence"""
        # Find first occurrence of end token in each sequence
        end_token_positions = (logits.argmax(-1) == end_token_idx).int().argmax(dim=1)

        # Handle sequences with no end token (use full length)
        no_end_token_mask = (end_token_positions == 0)

        # Use torch.where to conditionally assign seq_len without in-place modification
        end_token_positions = torch.where(no_end_token_mask, seq_len, end_token_positions)

        return end_token_positions + 1 
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            #         record for super pos
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    