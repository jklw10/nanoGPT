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

import torch
import torch.nn as nn
from torch.nn import functional as F
import utils
import wackmodel

#settings :)
#known good
diffSplits      = 1
ssmax           = True
#ungraded
fftmem          = True
#good in shkspr
mix_squish      = False #or fftmem
mem_mix_squish  = False #or fftmem
#bad
CausalMem       = False
convemb         = False
emamem          = False
diffembd        = False
qope            = False
side_squish     = False
#slow
spl             = False or fftmem
qrot            = False
think           = False
repeat_block    = False
repeatCenter    = False

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config, causal, memorymix):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        

        self.causal = causal
        #self.splits = 1
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head // diffSplits
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        #diff
        self.head_dim =  self.n_embd //  self.n_head // diffSplits
        if(diffSplits > 1):
            #self.lambda_init = utils.lambda_init_fn(d) 
            self.q_lambdas = nn.ParameterList([nn.Parameter(torch.zeros(self.head_dim, dtype=torch.bfloat16).normal_(mean=0,std=0.1)) for _ in range(diffSplits)])
            self.k_lambdas = nn.ParameterList([nn.Parameter(torch.zeros(self.head_dim, dtype=torch.bfloat16).normal_(mean=0,std=0.1)) for _ in range(diffSplits)])
        if(ssmax):
            self.qm = nn.Parameter(torch.ones(self.head_dim).normal_(mean=1, std=0.1))
            self.qb = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        if(fftmem and memorymix and self.causal and not mem_mix_squish):
            self.register_buffer("attn_mask", utils.create_memory_causal_mask(config.block_size,config.block_size))
            self.causal = False
        else:
            self.attn_mask=None
        #self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.bfloat16).normal_(mean=0,std=0.1))
        #self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.bfloat16).normal_(mean=0,std=0.1))
        #self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.bfloat16).normal_(mean=0,std=0.1))
        #self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.bfloat16).normal_(mean=0,std=0.1))

        #self.subln = nn.RMSNorm(config.n_embd, eps=1e-5, elementwise_affine=True)
    
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        ##self.flash = False #TO DO remember to re-enable after test
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def justnorm(self, x):
        #return F.normalize(x, p=2, dim=-1)
        res = x / x.norm(p=2, dim=-1, keepdim=True)
        return res

    
    def forward(self, x, maskless=False):
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

        k = k.view(B, T, self.n_head, diffSplits * self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, diffSplits * self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, diffSplits * self.head_dim).transpose(1, 2) # (B, nh, T, hs)
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
            #y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
            ys = []
            for i in range(diffSplits):#+self.qb
                if(ssmax):
                    if(maskless):
                        ys.append(torch.nn.functional.scaled_dot_product_attention(qs[i]*(math.log(T)*self.qm), ks[i], v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal= True))
                    else:
                        ys.append(torch.nn.functional.scaled_dot_product_attention(qs[i]*(math.log(T)*self.qm), ks[i], v, attn_mask=self.attn_mask, dropout_p=self.dropout if self.training else 0, is_causal= self.causal))
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
        if diffSplits > 1:
            #lambda_full = lambda_1 - lambda_2 + self.lambda_init
            lambda_full = self.lambda_init + torch.exp(torch.sum(self.q_lambdas[0] * self.k_lambdas[0], dim=-1).float()).type_as(q) # + lambda_1 - lambda_2 
            for i in range(diffSplits-1):
                lambda_full = lambda_full - torch.exp(torch.sum(self.q_lambdas[1+i] * self.k_lambdas[1+i], dim=-1).float()).type_as(q)
                #lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
            for i in range(diffSplits-1):
                y = y - lambda_full.unsqueeze(-1).unsqueeze(-1) * ys[1+i]

        #y = mmnorm(y, dim = 0)#, scale = 1) 
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * diffSplits * self.head_dim) # re-assemble all head outputs side by side

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



class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        if(qrot):
            self.c_proj  = nn.Linear( 2*config.n_embd, config.n_embd, bias=config.bias)
        else:
            self.gelu    = nn.GELU()
            self.c_proj  = nn.Linear( 4*config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.n_embd = config.n_embd

    def forward(self, x):
        x = self.c_fc(x)
        if(qrot):
            b,t,n=x.shape
            x4d = x.view(b, t, self.n_embd, 4) # Reshape to (batch, seq_len, n_embd, 4)
            x_norm = torch.max(x4d.norm(dim=-1, keepdim=True), torch.ones(1,device=x.device, dtype= x.dtype)*1e-10) # Calculate norm along the last dimension (dim=3 or -1), keep dimensions
            x_normalized = x4d / x_norm             # Divide to normalize
            rotors, rotated = x_normalized.chunk(2, dim=2) # Split along n_embd dimension
            #xm,ym = x_norm.chunk(2, dim=2)
            x_rotated = utils.quaternion_multiply(rotors, rotated) 
            #x_rotated = x_rotated*self.gaprelu(x_norm-2)#*torch.sigmoid(xm+ym)#
            x = x_rotated.view(b, t, 2 * self.n_embd)
            #x = utils.fft_trunc_csquish(x) //meh
        else:
            x = fast_sin_relu(x) #F.leaky_relu(x)+torch.sin(x)*F.leaky_relu(torch.sign(x))
            #x=modified_sin_gelu_leaky(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    



def fast_sin_relu(x):
     base = F.relu(x)
     sine_mod = F.relu(torch.sign(x)) * torch.sin(1.25 * x)
     return base + sine_mod


def quaternionize(x):
    b,t,n=x.shape
    x4d = x.view(b, t, n//4, 4)
    x_norm = x4d.norm(dim=-1, keepdim=True) 
    x_normalized = x4d / x_norm             
    return x_normalized


class Block(nn.Module):

    
    def justnorm(self, x):
        #return F.normalize(x, p=2, dim=-1)
        res = x / x.norm(p=2, dim=-1, keepdim=True)
        return res

    def __init__(self, config, causal, memorymix=False):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, causal,memorymix)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, maskless=False):
        x = x + self.attn(self.ln_1(x),maskless)
        x = x + self.mlp(self.ln_2(x))
        #x = x + self.attn(x)
        #x = x + self.mlp(x)
        #x = mmnorm(x)
        return x

class eBlock(nn.Module):
    def __init__(self, config, d):
        super().__init__()
        self.attn = CausalSelfAttention(config,d)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        x = utils.mmnorm(x)
        return x

class Dreamer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.block = Block(config, CausalMem)
        
        
    def forward(self, x):
        while x.shape[0] > 1:
            b, t, c = x.size()
            x = x.reshape(b // 2, 2, t, c)
            x = x.permute(0, 2, 1, 3).reshape(b // 2, t * 2, c)
            #x = x.view(b//2, 2*t, c)
            x = utils.fft_trunc_tsquish(x)
            x = self.block(x)

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
    optim: str = 'adam'


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        config.internal_vocab_size = config.vocab_size * config.internal_vocab_size_multi
        
        self.config = config
        self.ksize = 8
        self.patch_max = 10
        self.stride = 1#self.ksize//2
        self.bembwidth = config.n_embd//2
        if(convemb):
            self.threshold= nn.Parameter(torch.ones(1))
            self.end_patch_token_id = config.vocab_size
            config.vocab_size += 1
            config.internal_block_size //= self.stride
        else:
            config.internal_block_size = config.block_size
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            #swte = nn.Embedding(config.vocab_size, config.vocab_size),
            #wpe = nn.Embedding(config.block_size, config.n_embd),
            
            wpe = nn.Embedding(config.internal_block_size, config.n_embd),
            #pwpe = nn.Embedding(self.patch_max, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config, True, (i>=1 and not mem_mix_squish and fftmem)) for i in range(config.n_layer)]),
            #h = nn.ModuleList([uBlock(config,i) for i in range(config.n_layer-1)]).append(Block(config,config.n_layer-1)),
            #h = nn.ModuleList([uBlock(config,i) for i in range(config.n_layer-1)]),
            #h = Block(config), #nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        

        
        if(False): #don't remember
            self.embmix = nn.Sequential(
                nn.Linear(config.n_embd*2, config.n_embd*2),
                nn.GELU(),
                nn.Linear(config.n_embd*2, config.n_embd)
            )
        if(think):
            self.thinkthreshold = nn.Parameter(torch.ones(1))
            self.thinkstop = nn.Sequential(
                nn.Linear(config.n_embd, config.n_embd),
                nn.GELU(),
                nn.Linear(config.n_embd, 1)
            )
        if(qope):
            self.qoper = wackmodel.QrotMLP(config)
        if(diffembd):
            self.register_buffer('gmask', utils.gaussian_kernel(torch.ones(self.config.n_embd)))
        if(fftmem ):
            if(mem_mix_squish):
                self.register_buffer('memory', torch.zeros(1,config.internal_block_size,config.n_embd//2))
            else:
                self.register_buffer('memory', torch.zeros(1,config.internal_block_size,config.n_embd))

            self.dreamer = Dreamer(config)
            self.memory_selector = Block(config,CausalMem)
        if(emamem ):
            self.memory_selector = Block(config,CausalMem)
            self.memory_mixer = nn.Linear(config.n_embd, config.n_embd)
            #self.memory_predictor = nn.Linear(config.n_embd,config.n_embd)
            self.register_buffer('memory', torch.zeros(config.internal_block_size, config.n_embd))
            self.memory.data.zero_()  
        self.surprise = 1
        if(convemb):
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size * self.patch_max, bias=False)
            #self.pseudopatcher = nn.Conv1d(config.n_embd, config.n_embd, self.ksize, stride=self.stride, bias=False)#, padding='same')# padding=0, stride=7 )
            self.pseudopatcher = nn.Conv1d(self.bembwidth, self.bembwidth, self.ksize, stride=self.stride, bias=False)
            self.padding = (self.ksize-1, 0) 
            
            self.embedder = nn.Sequential(
                nn.Linear(config.n_embd*self.patch_max, config.n_embd),
                nn.GELU(),
                nn.Linear(config.n_embd, config.n_embd)
            ) #eBlock(config,1)
        else:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            # with weight tying when using torch.compile() some warnings get generated:
            # "UserWarning: functional_call was passed multiple values for tied weights.
            # This behavior is deprecated and will be an error in future versions"
            # not 100% sure what this is, so far seems to be harmless. TODO investigate
            self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
        
        self.predict_weight = 0.01

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))



    def forward(self, idx, targets=None):
        
        #torch.compiler.cudagraph_mark_step_begin() # Add this line at the beginning of forward
        device = idx.device
        b, t = idx.size()
        #end_tok=torch.as_tensor(self.end_patch_token_id, device=device, dtype=torch.long)
        #patch_max=torch.as_tensor(self.patch_max, device=device, dtype=torch.long)
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        
        if(convemb):
            patches, x, pploss = self.tok2patch(idx)

            pb, pt, pe = x.size()
            x = x[:,-(pt):-2,:] #(b, t_internal, n_embed)
            
            patchtargets = patches[:,-(pt-1):-1,:] #(b, t_internal, patchmax)
            
            pos = torch.arange(0, pt-2, dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos)
            x = x + pos_emb
        else:
            pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
            tok_emb = self.transformer.wte(idx)
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (b, t, n_embd)
            if(qope):
                x = self.qoper(tok_emb,pos_emb.expand(b,t,self.config.n_embd),False,False)
            else:
                x = tok_emb + pos_emb
        x = self.transformer.drop(x)

        if(spl):
            block_inputs = []

        if(emamem):
            x = self.memory_selector(torch.cat((self.memory.expand(b, -1, -1), x), dim=2))
            x = self.memory_mixer(x)
        i=0
        if(fftmem and self.training):
            xswish = torch.cat([x[b//2:,:,],x[:b//2,:,:]],dim=0)
            if(mem_mix_squish):
                fh = self.memory.expand(b,t,self.config.n_embd//2)
            for block in self.transformer.h:
                i+=1 
                xswish = block(xswish)
                if(mem_mix_squish):
                    if(not(i == 1 or i == self.config.n_layer)):
                        sh = utils.fft_trunc_csquish(xswish)
                        if (i) % 2 == 0:
                            xswish = torch.stack([fh,sh],dim =-1).flatten(start_dim=-2,end_dim=-1)
                        else:
                            xswish = torch.stack([sh,fh],dim =-1).flatten(start_dim=-2,end_dim=-1)
                elif i == 1:
                    xswish = torch.cat([self.memory.expand(b,t,self.config.n_embd), xswish], dim=1)
            
            if(mem_mix_squish):
                selectedMemory = self.memory_selector(xswish)#(b,t,c)
                selectedMemory = utils.fft_trunc_csquish(selectedMemory) #(b,t,c//2)
                first_mem = torch.stack([selectedMemory, fh],dim =-1).flatten(start_dim=-2,end_dim=-1) #(b,t,c)
                first_mem = self.dreamer(first_mem)
                first_mem = utils.fft_trunc_csquish(first_mem)#(b,t,c//2)
                #first_mem = first_mem.mean(dim=0).unsqueeze(0)
                fh = first_mem.expand(b,t,self.config.n_embd//2)#(b,t,c//2)
            else:
                #could be better but it works
                selectedMemory = self.memory_selector(xswish[:,:t,:])#(b,t,c)
                selectedMemory = utils.fft_trunc_tsquish(selectedMemory) #(b,t//2,c)
                squishmem = utils.fft_trunc_tsquish(self.memory).expand(b,t//2,self.config.n_embd) #(b,t//2,c)
                first_mem = torch.cat([selectedMemory,squishmem],dim =1) #(b,t,c)
                first_mem = self.dreamer(first_mem).expand(b,t,self.config.n_embd)
        
        if(fftmem and not self.training):
            if(mem_mix_squish):
                fh = self.memory.expand(b,t,self.config.n_embd//2)
            else:
                first_mem = self.memory.expand(b,t,self.config.n_embd)
        i=0
        iters=0
        if(mix_squish and not (fftmem and mem_mix_squish)):
            fh = utils.fft_trunc_csquish(x)
        for block in self.transformer.h:
            i+=1 
            if(spl):
                block_inputs.append(x) 
                
            x = block(x)
            if(not(i == 1 or i == self.config.n_layer)):
                if(mix_squish):
                    sh = utils.fft_trunc_csquish(x)
                    if (i) % 2 == 0:
                        x = torch.stack([fh,sh],dim =-1).flatten(start_dim=-2,end_dim=-1)
                    else:
                        x = torch.stack([sh,fh],dim =-1).flatten(start_dim=-2,end_dim=-1)
                if(think):
                    while True:
                        sh = utils.fft_trunc_csquish(x)
                        t = self.thinkstop(x) >= self.thinkthreshold
                        x = torch.cat([fh,sh],dim=-1) *t+~t*x
                        x = block(x) *t+~t*x
                        if(t.all()):
                            break
                        iters+=~t
                        if((iters >= 10).any()):
                            break
            
            if(not mem_mix_squish and i == 1):
                x = torch.cat([first_mem, x], dim=1)
        
        if fftmem and self.training:
            with torch.no_grad():
                if(mem_mix_squish):
                    selectedMemory = self.memory_selector(x)#(b,t,c)
                    selectedMemory = utils.fft_trunc_csquish(selectedMemory) #(b,t,c//2)
                    updated_mem = torch.stack([selectedMemory,fh],dim =-1).flatten(start_dim=-2,end_dim=-1) #(b,t,c)
                    updated_mem = self.dreamer(updated_mem)
                    updated_mem = utils.fft_trunc_csquish(updated_mem)#(b,t,c//2)
                else:
                    selectedMemory = self.memory_selector(x[:,:t,:])#(b,t,c)
                    selectedMemory = utils.fft_trunc_tsquish(selectedMemory) #(b,t//2,c)
                    squishmem = utils.fft_trunc_tsquish(first_mem).expand(b,t//2,self.config.n_embd) #(b,t,c)
                    updated_mem = torch.cat([selectedMemory,squishmem],dim =1) #(b,t,c)
                    updated_mem = self.dreamer(updated_mem)
                self.memory.copy_(updated_mem)

        if(emamem):
            with torch.no_grad():
                # Create temporary tensor for update
                new_mem = x.mean(dim=0).detach().clone()
                new_mem = utils.mmnorm(new_mem,dim = 0) #consider sleep block since it workes way too well for fftmem
                updated_mem = 0.99 * self.memory + 0.01 * new_mem
                self.memory.copy_(updated_mem)
        
        x = self.transformer.ln_f(x)
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            
            #x = x * zeropower_via_newtonschulz(x.detach()).detach() # hacks time dim
            if(not mem_mix_squish):
                logits = self.lm_head(x[:,t:,:])
            else:
                logits = self.lm_head(x)
            if(convemb):
                #logits = self.lm_head(x)  # (b, t_internal, vocab_size * patch_max)
                logits = logits.view(
                    logits.size(0), 
                    logits.size(1), 
                    self.patch_max, 
                    self.config.vocab_size
                )   # (b, t_internal, patch_max, vocab_size)
                #l = torch.ones(1,device=device,dtype=torch.long)*self.end_patch_token_id
                #pm = torch.ones(1,device=device,dtype=torch.long)*self.patch_max
                #lengths = self.get_patch_lengths(logits,l,pm)
                #length_penalty = F.mse_loss(lengths.float(), torch.full_like(lengths, self.patch_max//2)).mean()
    
                logits_flat = logits.contiguous().view(-1, self.config.vocab_size)  # (b * t_internal * patch_max, vocab_size)
                targets_flat = patchtargets.contiguous().view(-1)  # (b * t_internal * patch_max)
                loss = F.cross_entropy(
                    logits_flat, 
                    targets_flat,
                    ignore_index=-1
                )

                loss = loss + pploss.mean()/(self.threshold/self.patch_max) #+ length_penalty
                    
            else:
                
                if(fftmem and not mem_mix_squish):
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
                else:
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
                #targ_emb = self.transformer.wte(targets)
                #eloss = F.mse_loss(x, targ_emb).mean()
                if(think):
                    loss = loss + iters.float().mean() * 0.1# + eloss
                
            if(spl):
                for i in range(self.config.n_layer - 1, -1, -1):
                    if(fftmem and not mem_mix_squish):
                        
                        target_output = block_inputs[i]  # The target is the block input
                        ##tokpred
                        if(i < 1):
                            target_output = block_inputs[i]  # The target is the block input
                            predicted_output = self.transformer.h[i](-x[:,t:,:]) 
                        else:
                            target_output = block_inputs[i]  # The target is the block input
                            predicted_output = self.transformer.h[i](-x,True)  #use the negative direction of token space as the self prediction space

                        sploss = nn.functional.mse_loss(
                            predicted_output.view(-1, self.config.n_embd),
                            target_output.view(-1, self.config.n_embd)
                        ).mean()
                        if(self.training):
                            loss = loss + sploss * self.predict_weight 
                    else:
                        target_output = block_inputs[i]  # The target is the block input
                        predicted_output = self.transformer.h[i](-x)  #use the negative direction of token space as the self prediction space

                        sploss = nn.functional.mse_loss(
                            predicted_output.view(-1, self.config.n_embd),
                            target_output.view(-1, self.config.n_embd)
                        )
                        if(self.training):
                            loss = loss + sploss * self.predict_weight 

        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        #self.memory = torch.zeros_like(self.memory)

        return logits, loss

    def get_patch_lengths(self, logits, end_token_idx, seq_len):
        """Returns tensor of patch lengths (batch_size,) based on first end token occurrence"""
        # Find first occurrence of end token in each sequence
        end_token_positions = (logits.argmax(-1) == end_token_idx).int().argmax(dim=1)

        # Handle sequences with no end token (use full length)
        no_end_token_mask = (end_token_positions == 0)

        # Use torch.where to conditionally assign seq_len without in-place modification
        end_token_positions = torch.where(no_end_token_mask, seq_len, end_token_positions)

        return end_token_positions + 1 
    
    def tok2patch(self, idx, et = True):
        device = idx.device
        b, t = idx.shape

        pos = torch.arange(0, self.patch_max, dtype=torch.long, device=device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx).permute(0, 2, 1).contiguous()
        pred_idx = self.pseudopatcher(F.pad(tok_emb[:, :self.bembwidth, :], self.padding))#.permute(0, 2, 1).contiguous()
        losses = F.mse_loss(tok_emb[:, :self.bembwidth, :-1], pred_idx[:, :, 1:], reduction='none').mean(dim=1)#.permute(0, 2, 1).contiguous()  
        
        #tok_emb = tok_emb.permute(0, 2, 1).contiguous()
        
        endtok= torch.as_tensor(self.end_patch_token_id,device=device,dtype=torch.long)
        if(et):
            #end tok fill reward hacks, so make it 1 longer so i can just make the next token into end tok.
            #todo: retry fill
            patch_indices = torch.full((b, self.config.internal_block_size+2, self.patch_max+1), -1, dtype=torch.long, device=device)
        else:
            patch_indices = torch.full((b, self.config.internal_block_size+2, self.patch_max), -1, dtype=torch.long, device=device)
            #patch_indices = torch.full((b, self.config.internal_block_size+2, self.patch_max), -1, dtype=torch.long, device=device)

        patch_embeds = torch.zeros((b, self.config.internal_block_size+2, self.patch_max, self.config.n_embd), device=device)
        accumulated_loss = torch.zeros(b, device=device)
        patch_deposit_idx = torch.zeros(b, dtype=torch.long, device=device)
        patch_length_idx = torch.zeros(b, dtype=torch.long, device=device)
        batch_indices = torch.arange(b, device=device)  # Explicit batch indices
        

        for current_token_index in range(t-1): 
            #patch_indices[batch_indices,patch_deposit_idx, patch_length_idx] = idx[batch_indices, current_token_index]
            #patch_embeds[batch_indices, patch_deposit_idx, patch_length_idx, :] = tok_emb[batch_indices, :, current_token_index]
            #
            #if(et):
            #    patch_indices[batch_indices,patch_deposit_idx, patch_length_idx+1] = endtok
            #patch_embeds[batch_indices, patch_deposit_idx, patch_length_idx+1,:] = endemb#no benefit.

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
            
            if(et):
                patch_indices[batch_indices,patch_deposit_idx, patch_length_idx+1] = endtok
        
        if(et):
            patch_indices= patch_indices.narrow(-1,0,self.patch_max).contiguous()#trim the end, hacky, but eh.
        pos_emb = pos_emb.flatten(start_dim=0,end_dim=-1).unsqueeze(0).unsqueeze(0).contiguous()
        toemb = pos_emb + torch.flatten(patch_embeds, start_dim=2, end_dim=-1)
        #pb,pp,pe = toemb.shape
        #if(pp >= self.config.internal_block_size+2):
        #    toemb = toemb[:, -(self.config.internal_block_size+2):,:]
        #print(pos_emb.size())
        #print(patch_embeds.size())
        patch_embeds = self.embedder(toemb)
        #print(patch_embeds.size())

        return patch_indices, patch_embeds, losses
    
    def tok2patch2(self, idx):
        device = idx.device
        b, t = idx.shape

        pos = torch.arange(0, self.patch_max, dtype=torch.long, device=device)
        pos_emb = self.transformer.pwpe(pos)
        tok_emb = self.transformer.wte(idx).permute(0, 2, 1).contiguous()
        pred_emb = self.pseudopatcher(F.pad(tok_emb, self.padding))#.permute(0, 2, 1).contiguous()
        losses = F.mse_loss(tok_emb[:, :, :-1], pred_emb[:, :, 1:], reduction='none').mean(dim=1)#.permute(0, 2, 1).contiguous()  
        batch_indices = []
        batch_embeds = []
        for batch_index in range(b): 
            patch_indices = []
            patch_embeds = []
            accumulated_loss = 0.0
            patch_start = 0

            for current_token_index in range(t-1): 
                    if current_token_index <= patch_start:
                        continue
                    loss_value = losses[batch_index, current_token_index]
                    accumulated_loss += loss_value
                    if accumulated_loss > self.threshold or current_token_index-patch_start >= self.patch_max or current_token_index == t-2:
                        patch = idx[batch_index, patch_start:current_token_index]
                        patch_emb = tok_emb[batch_index, :, patch_start:current_token_index]

                        #patch_emb = patch_emb.permute(1, 0).contiguous()  
                        #patch_emb = patch_emb[patch_start:current_token_index,:]
                        
                        patch_emb = patch_emb.permute(1, 0).contiguous()  
                        #patch_emb = patch_emb[:,patch_start:current_token_index]
                        #patch_emb = F.pad(patch_emb, (0, 0, 0, self.patchMax-len(patch_emb)), value=0)
                        
                        pos = torch.arange(0, len(patch), dtype=torch.long, device=device)
                        pos_emb = self.transformer.pwpe(pos)
                        emb = self.embedder(pos_emb+patch_emb)
                        patch_embeds.append(emb.mean(dim=0))

                        patch = F.pad(patch, (0, self.patch_max - len(patch)), value=-1) 
                        patch_indices.append(patch)

                        patch_start = current_token_index
                        accumulated_loss = 0.0
            
            patch_embed_tensor = torch.stack(patch_embeds)
            patch_embed_tensor = patch_embed_tensor[-(self.config.internal_block_size+2):]
            #if len(patch_embed_tensor) < (self.config.internal_block_size+2):
            #    patch_embed_tensor = F.pad(patch_embed_tensor, (0, 0, 0, (self.config.internal_block_size+2)-len(patch_embed_tensor)), value=0)

            batch_embeds.append(patch_embed_tensor)

            patch_index_tensor = torch.stack(patch_indices)
            #if len(patch_index_tensor) < (self.config.internal_block_size+2):
            #    patch_index_tensor = F.pad(patch_index_tensor, (0, 0, 0, (self.config.internal_block_size+2)-len(patch_index_tensor)), value=-1)

            batch_indices.append(patch_index_tensor)

        batch_embeds=torch.stack(batch_embeds)
        batch_indices=torch.stack(batch_indices)

        return batch_indices, batch_embeds

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
        if isinstance(module, nn.Linear):
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
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        if(self.config.optim == 'adam'):
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
    #@torch.no_grad()
    #def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    #    """Generates text patch by patch in batches, stopping per batch element at end-of-patch token or patch_max."""
    #    block_size = self.config.block_size
    #    patch_max = self.patch_max
    #    vocab_size = self.config.vocab_size
    #    batch_size = idx.size(0) # Get batch size
#
    #    generated_tokens = 0
    #    finished_batches = torch.zeros(batch_size, dtype=torch.bool, device=idx.device) # Track finished batches
    #    next_indices_list = [] # List to accumulate generated indices for each batch
#
    #    # Initialize next_indices_list with current idx sequences
    #    for i in range(batch_size):
    #        next_indices_list.append(idx[i].clone()) # Clone to avoid modifying original idx
#
    #    for _ in range(max_new_tokens // patch_max):
    #        if generated_tokens >= max_new_tokens or torch.all(finished_batches): # Stop condition
    #            break
#
    #        # Prepare input for the model, handling sequence length cropping
    #        current_idx_batch = []
    #        for i in range(batch_size):
    #            current_seq = next_indices_list[i]
    #            if current_seq.size(0) > block_size:
    #                current_seq = current_seq[-block_size:] # Crop if too long
    #            current_idx_batch.append(current_seq)
    #        idx_cond = torch.stack(current_idx_batch) # Stack into a batch
#
    #        logits, _ = self(idx_cond)
    #        logits_reshaped = logits.view(batch_size, -1, patch_max, vocab_size)
    #        patch_logits = logits_reshaped[:, -1, :, :]
#
    #        next_patch_tokens_batch = []
    #        for batch_idx in range(batch_size):
    #            if finished_batches[batch_idx]:
    #                next_patch_tokens_batch.append(None) # Placeholder for finished batch
    #                continue
#
    #            current_batch_patch_tokens = []
    #            for patch_pos in range(patch_max):
    #                pos_logits = patch_logits[batch_idx, patch_pos, :] / temperature
    #                if top_k is not None:
    #                    v_topk, _ = torch.topk(pos_logits, min(top_k, pos_logits.size(-1)))
    #                    pos_logits[pos_logits < v_topk[..., [-1]]] = -float('Inf')
    #                probs = F.softmax(pos_logits, dim=-1)
    #                idx_next_token = torch.multinomial(probs, num_samples=1)
#
    #                current_batch_patch_tokens.append(idx_next_token)
    #                generated_tokens += 1
#
    #                if idx_next_token.item() == self.end_patch_token_id:
    #                    finished_batches[batch_idx] = True
    #                    break
    #                if generated_tokens >= max_new_tokens:
    #                    break
    #            if not finished_batches[batch_idx]:
    #                next_patch_tokens_batch.append(torch.cat(current_batch_patch_tokens))
    #            else:
    #                next_patch_tokens_batch.append(torch.tensor([], dtype=torch.long, device=idx.device)) # Empty tensor for finished
#
    #        # Update next_indices_list with generated patch tokens
    #        for batch_idx in range(batch_size):
    #            if not finished_batches[batch_idx]:
    #                patch_tokens = next_patch_tokens_batch[batch_idx]
    #                next_indices_list[batch_idx] = torch.cat((next_indices_list[batch_idx], patch_tokens))
#
#
    #    # Stack the list of sequences back into a batch tensor for return
    #    #idx_result = torch.cat(next_indices_list)
    #    return next_indices_list