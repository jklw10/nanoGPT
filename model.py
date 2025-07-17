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

from pytorch_wavelets import DWT1DForward
import torch
import torch.nn as nn
from torch.nn import functional as F
import optim
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
fftmem          = False
mmnorm          = False
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
spl             = False or fftmem
qrot            = False
think           = False
repeat_block    = False
repeatCenter    = False

Qrotention      = True
symloss         = False


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.n_embd))
        self.bias = nn.Parameter(torch.zeros(config.n_embd)) if config.bias else None
       

    def forward(self, input):
        #if(tests):
        #    return input
        if(mmnorm):
            return torch.tanh(input*self.weight)
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class Scanner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scanner = nn.Sequential(
                nn.LayerNorm(config.n_embd * 2), #untested
                Linear(config.n_embd*2, config.n_embd*4),
                nn.GELU(),
                Linear(config.n_embd*4, config.n_embd)
            )
    def forward(self, x: torch.tensor, y: torch.tensor):
        return self.scanner( torch.cat((x,y),dim=-1))

class QrotAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        #self.c_attn3 =  Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_attn2 =  Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)
        
        #self.q_heads = Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)#doesn't scale well :/ maybe a different down projection.
        #self.v_head =  Linear(config.n_embd,  config.n_embd, bias=config.bias)
        
        self.c_proj = Linear(config.n_embd,  config.n_embd, bias=config.bias) #todo

        self.c_kattn = Linear(config.n_embd,  config.n_embd, bias=config.bias)
        self.scanner = Scanner(config)
        self.identity = nn.Parameter(torch.rand(config.n_embd))
        #nn.Sequential(
        #        Linear(config.n_embd*2, config.n_embd*2),
        #        nn.GELU(),
        #        Linear(config.n_embd*2, config.n_embd)
        #    )

        self.attn_dropout = nn.Dropout(config.dropout) #todo
        self.resid_dropout = nn.Dropout(config.dropout) #todo
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.n_head = config.n_head 
        self.head_dim = config.n_embd//config.n_head * 2
        self.win = config.block_size // config.n_layer
     

    def forward(self, x: torch.tensor, mem=None,causal=True, k=None): #same signature to allow easy swapping.
        B, T, C = x.size() 
        #q, q2,  v = self.c_attn3(x).split(self.n_embd, dim=2)
        q,   v = self.c_attn2(x).split(self.n_embd, dim=2)
        #v = self.v_head(x)
        #q = self.q_heads(x)
        #q = torch.cat((q,q2),dim=-1)
        QC = q.shape[2]
        
        #q2 = q2.view(B, T, QC//4, 4)
        #q = q.view(B, T, QC//4, 4)
        #v = v.view(B, T, C//4, 4)
        q = q.view(B, T, QC)
        v = v.view(B, T, C)

        #split, window T//n_layer, fft mix
        #q = utils.maprotgate(q)
        #q2 = utils.maprotgate(q)
        #q2 = utils.scan_quaternion_multiply_window(q, self.win)
        #q2 = self.c_kattn(q2.view(B, T, -1)).view(B, T, QC//4, 4)
        #q = utils.fft_trunc_csquish(torch.cat((q,q2),dim=-2).view(B, T, -1).to(torch.float32), C).to(x.dtype).view(B, T, C//4, 4) #win premap
        
        #q2 = utils.maprotgate(q2)#winmul
        #s = torch.sigmoid(q[..., 0].view(B, T, QC//4, 1))
        #rot = q[..., 1:]
        #q = utils.exponential_map(rot*s) 
        #HQC = QC//2
        
        #q = utils.quaternion_multiply(q, q2)#winmul

        #q = utils.fft_trunc_csquish(torch.cat((q,q2),dim=-2).view(B, T, -1).to(torch.float32), C).to(x.dtype).view(B, T, C//4, 4)#winmix
        #gates2 = utils.maprotgate(q[...,HQC:,:])
        #q = utils.parallel_quaternion_scan_log_n(q)
        #q = utils.fft_trunc_csquish(q.view(B, T, -1).to(torch.float32), C).to(x.dtype).view(B, T, C//4, 4)
        #q = utils.fft_trunc_csquish(q.view(B, T, -1).to(torch.float32), C).to(x.dtype).view(B, T, C)
        q = utils.pscan(q, self.scanner, self.identity)#.view(B, T, C//4, 4)
        #q2 = utils.pscan(q2, self.scanner, self.identity)
        
        #q = utils.fft_trunc_csquish(q.view(B, T, -1).to(torch.float32), C).to(x.dtype).view(B, T, C)
        #q = self.scanner(q,q2)
        #q= q*q2
        #q = utils.fft_trunc_csquish(torch.cat((q,q2),dim=-1).view(B, T, -1).to(torch.float32), C).to(x.dtype).view(B, T, C)
        #q = utils.quaternion_multiply(gatescan, utils.parallel_quaternion_scan_log_n(gates2))
        #q = self.q_proj(q.view(B, T, -1)).view(B, T, C//4, 4)
        #q = self.attn_dropout(q)
        #q = utils.fft_trunc_csquish(q.view(B, T, -1).to(torch.float32), C).to(x.dtype).view(B, T, C//4, 4)
        #q = utils.quaternion_multiply(q[...,:C//4,:], q[...,C//4:,:])
        #Y =  utils.quaternion_multiply(q, v)
        #Y = self.scanner(q,v) #utils.quaternion_multiply(q, v)
        
        #Y = utils.fft_trunc_csquish(torch.cat((q,q2,v),dim=-1).to(torch.float32), C).to(x.dtype)
        #q = utils.fft_trunc_csquish(torch.cat((q,q2),dim=-1).to(torch.float32), C, "low").to(x.dtype)
        #Y = utils.fft_trunc_csquish(torch.cat((q,v),dim=-1).to(torch.float32), C, "low").to(x.dtype)
        Y = q*v #utils.quaternion_multiply(q, v)
        #Y = Y / torch.norm(Y, p=2, dim=-1, keepdim=True)
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

    def forward(self, x, mem=None,causal=True):
        B, T, C = x.size() 
        
        if mem is None:
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
            q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
            k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
            v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)

            y = torch.nn.functional.scaled_dot_product_attention(q*(math.log(T)*self.qm),k,  v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal= causal)

            y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim) 
            y = self.resid_dropout(self.c_proj(y))

            return y
        
        q, k, v = self.c_attn(torch.cat((mem,x),dim=1)).split(self.n_embd, dim=2)
        q = q.view(B, T+self.mem_len, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T+self.mem_len, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T+self.mem_len, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)

        y = torch.nn.functional.scaled_dot_product_attention(q*(math.log(T)*self.qm), k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal= causal)
        y = y.transpose(1, 2).contiguous().view(B, T+self.mem_len, self.n_head * self.head_dim) 
        y = self.resid_dropout(self.c_proj(y))

        return y[:,self.mem_len:,:]



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
            x = x.unsqueeze(-1)
        sin_term = torch.sin(self.frequencies * x + self.phase_shifts)
        
        amp_term = self.amplitude_scales * x
        
        output = sin_term * amp_term
        
        return output

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        if(qrot):
            self.c_proj  = Linear( 2*config.n_embd, config.n_embd, bias=config.bias)
        else:
            self.gelu    = nn.GELU()
            self.c_proj  = Linear( 4*config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.n_embd = config.n_embd

    def forward(self, x):
        x = self.c_fc(x)
        
        if(qrot):
            b,t,n=x.shape
            x4d = x.view(b, t, self.n_embd, 4) # Reshape to (batch, seq_len, n_embd, 4)
            x_norm = torch.clamp_min(x4d.norm(dim=-1, keepdim=True), 1e-10) # Calculate norm along the last dimension (dim=3 or -1), keep dimensions
            x_normalized = x4d / x_norm             # Divide to normalize
            rotors, rotated = x_normalized.chunk(2, dim=2) # Split along n_embd dimension
            #xm,ym = x_norm.chunk(2, dim=2)
            x_rotated = utils.quaternion_multiply(rotors, rotated) 
            #x_rotated = x_rotated*self.gaprelu(x_norm-2)#*torch.sigmoid(xm+ym)#
            x = x_rotated.view(b, t, 2 * self.n_embd)
            #x = utils.fft_trunc_csquish(x) #meh
        else:
            x = self.gelu(x) # sinrelu
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    

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

    def forward(self, x, mem = None, causal = True, k =None):
        if(Qrotention):
            x = x + self.attn(self.ln_1(x),mem,causal,k)
        else:
            x = x + self.attn(self.ln_1(x),mem,causal)
        x = x + self.mlp(self.ln_2(x))
        return x

    def spl(self, block_input, x, memory = None, k = None):
        b, t, e = x.size()
        target_output = block_input #store in block?
        if memory is not None:
            predicted_output = self(-x, -self.memory.expand(b, self.config.mem_block_size, self.config.n_embd), causal = False)
        elif k is not None:
            predicted_output = self(-x, k = k, causal = False)
        else:
            predicted_output = self(-x, causal = False)
            
        sploss = F.mse_loss(
            utils.mmnorm(predicted_output.flatten()),
            utils.mmnorm(target_output.flatten())
        ) 
        return sploss
    
class Dreamer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.block = MemBlock(config)
        
    def forward(self, x):
        while x.shape[0] > 1:
            b, t, c = x.size()
            x = x.reshape(b // 2, 2, t, c)
            x = x.permute(0, 2, 1, 3).reshape(b // 2, t * 2, c)
            #x = x.view(b//2, 2*t, c)
            x = utils.fft_trunc_tsquish(x)
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
    #optim: str = 'SGD'
    optimizer: str = 'adam'
    mem_block_size: int = 8
    batch_size = 64
    step = 0

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
                self.register_buffer('memory', torch.zeros(1,config.internal_block_size,config.n_embd//2))
            else:
                self.register_buffer('memory', torch.zeros(1,config.mem_block_size,config.n_embd))
            if(config.dropout > 0.0):
                config.dropout = config.dropout / 4
            self.dreamer = Dreamer(config)
            self.memory_selector = MemBlock(config)
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
        
        self.predict_weight = 0.01
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        self.laim = 0.1
        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))


    def forward(self, idx, targets=None):
        #self.config.step += 1.0
        #torch.compiler.cudagraph_mark_step_begin()
        device = idx.device
        b, t = idx.size()
        #end_tok=torch.as_tensor(self.end_patch_token_id, device=device, dtype=torch.long)
        #patch_max=torch.as_tensor(self.patch_max, device=device, dtype=torch.long)
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        if convemb:
            x, patchtargets, pploss= self.convemb(idx)
            pb, pt, pe = x.size()
            
            pos = torch.arange(0, pt, dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos)
            x = x + pos_emb
        else:
            pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
            tok_emb = self.transformer.wte(idx)
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (b, t, n_embd)
            if qope:
                x = self.qoper(tok_emb,pos_emb.expand(b,t,self.config.n_embd),False,False)
            else:
                x = tok_emb + pos_emb
        x = self.transformer.drop(x)

        if mtp:
            return self.mtp_fwd(targets, x)
            
        if fftmem:
            if mem_mix_squish:
                return self.mem_mix_fwd(targets, x)
            else:
                return self.mem_fwd(targets, x)

        if mix_squish:
            fh = utils.fft_trunc_csquish(x)
        if spl or symloss:
            block_inputs = []
        x= x.view(b, t, self.config.n_embd//4, 4)
        ogx = None #x# utils.parallel_quaternion_scan_log_n(x)
        x = x /  torch.norm(x, p=2, dim=-1, keepdim=True)
        #ogx = x
        x = x.view(b, t, -1)
        for i, block in enumerate(self.transformer.h):
            if spl or symloss:
                block_inputs.append(x) 
            x = block(x) 

            if mix_squish:
                if self.config.mix_mask[i]:
                    sh = utils.fft_trunc_csquish(x)
                    x = torch.stack([sh,fh],dim =-1).flatten(start_dim=-2,end_dim=-1)
          
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = None# self.lm_head(x)
            reduction= 'none'
            if(not self.training):
                reduction = 'mean'
            #    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction='mean')
            #else:
            #    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction='none').view(b,t)
            loss = linear_cross_entropy(x.to(torch.bfloat16), self.lm_head.weight.to(torch.bfloat16), targets, impl='torch_compile', reduction=reduction)
            
            #loss = F.cross_entropy(logits, targets, ignore_index=-1, reduction=reduction) 
            #loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=reduction).view(b,t)
            #TODO: wavelet loss
            if(convemb):
                loss = self.convemb.loss(logits, patchtargets, pploss)
            
            if symloss and self.training:
                for i in block_inputs:
                    lcloss, lmean = utils.calculate_linear_chaos_loss(i, balance_lambda=1.0,target_chaos=self.laim)
                    self.laim = max(lmean.detach(),self.laim)
                    loss = loss + lcloss#utils.calculate_linear_chaos_loss(i, balance_lambda=0.1,target_chaos=0.5)
                    #loss = loss + 0.001 * torch.nn.functional.mse_loss(torch.abs(torch.fft.fft(i)),torch.softmax(torch.abs(torch.fft.fft(i)), -1))
            if(spl and self.training):
                loss = loss + self.self_prediction_loss(block_inputs,x) #* self.predict_weight #* 0.001
                
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
    
    def mem_mix_fwd(self, targets, x):
        b, t, e = x.size()
        block_inputs = []

        if(self.training):
            xswish = torch.cat([x[b//2:,:,],x[:b//2,:,:]],dim=0)
            fh = self.memory.expand(b,t,self.config.n_embd//2)
            for i, block in enumerate(self.transformer.h):
                if(self.config.mix_mask[i]):
                    sh = utils.fft_trunc_csquish(xswish)
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
                sh = utils.fft_trunc_csquish(x)
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
                loss = loss + self.self_prediction_loss(x,block_inputs) 
                loss = loss + self.blockspl(self.memory_selector, self.memory)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
        return logits, loss
    
    def mem_fwd(self, targets, x):
        b, t, e = x.size()
        
        block_inputs = []

        if(self.training):
            xswish = torch.cat([x[b//2:,:,],x[:b//2,:,:]],dim=0)
            for block in self.transformer.h:
                xswish = block(xswish, self.memory.expand(b,self.config.mem_block_size,self.config.n_embd))
            mh1 = self.mem_form(xswish[b//2:,:,:], self.memory).expand(b//2,self.config.mem_block_size,self.config.n_embd)
            mh2 = self.mem_form(xswish[:b//2,:,:], self.memory).expand(b//2,self.config.mem_block_size,self.config.n_embd)
            first_mem = torch.cat([mh1,mh2],dim =0) #disallow cheating as much as possible
        else:
            first_mem = self.memory.expand(b,self.config.mem_block_size,self.config.n_embd)
        
        if(mix_squish):
            fh = utils.fft_trunc_csquish(x)

        for i, block in enumerate(self.transformer.h):
            
            block_inputs.append(x) 
            x = block(x,first_mem)
            if(mix_squish):
                if(self.config.mix_mask[i]):
                    sh = utils.fft_trunc_csquish(x)
                    x = torch.stack([sh,fh],dim =-1).flatten(start_dim=-2,end_dim=-1)
            
        updated_mem = None
        if self.training and not self.memory_frozen :
            updated_mem = self.mem_form(x, first_mem)
            with torch.no_grad():
                self.memory.copy_(updated_mem)

        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1) 
            
            if(self.training):
                loss = loss + self.self_prediction_loss(block_inputs, x, self.memory) 
                loss = loss + self.blockspl(self.memory_selector, first_mem[::b//2,:,:], self.memory.expand(2,-1,-1), x)
                _=0
                
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
    
    
    def self_prediction_loss(self, block_inputs, x, memory = None, k =None):
        b, t, e = x.size()
        loss = 0
        for i, block in enumerate(self.transformer.h):
            loss = loss + block.spl( block_inputs[i], x, memory, k) 
        return loss
    
    
    def mix_mem_form(self,x,fh):
        selectedMemory  = self.memory_selector(x, causal = False)#(b,t,c)
        selectedMemory  = utils.fft_trunc_csquish(selectedMemory) #(b,t,c//2)
        updated_mem     = torch.stack([selectedMemory,fh],dim =-1).flatten(start_dim=-2,end_dim=-1) #(b,t,c)
        updated_mem     = self.dreamer(updated_mem)
        return utils.fft_trunc_csquish(updated_mem)#(b,t,c//2)

    def mem_form(self, x, memory):
        #currently assumes mem size and block size are the same
        b, t, e = x.size()
        squishmem      = utils.fft_trunc_tsquish(memory).expand(b, self.config.mem_block_size//2, self.config.n_embd) #(b, m//2,c)
        selectedMemory = self.memory_selector(x, causal = False)[:,:self.config.mem_block_size,:] #(b, m, c)
        selectedMemory = utils.fft_trunc_tsquish(selectedMemory) #(b, m//2,c)
        mh1 = torch.cat([selectedMemory,squishmem],dim =1)       #(b, m,c)
        mh1 = self.dreamer(mh1)

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
        if(self.config.optimizer == 'adam'):
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        else:
            optimizer = torch.optim.SGD(optim_groups, lr=learning_rate, **extra_args)
        
            #optimizer =torch.optim.Adadelta(optim_groups)#, **extra_args)# lr=learning_rate, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

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
    