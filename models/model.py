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

from numpy import isin
import torch
import torch.nn as nn
from torch.nn import functional as F
import parts.kattn as kattn
import parts.modules as modules
import parts.optim as optim
import parts.utils as utils
import parts.wackmodel as wackmodel
from cut_cross_entropy import linear_cross_entropy
import parts.quantizer as quantizer
#todo
#prediction machine
#Linear = optim.OptimizedLinear
#Linear = nn.Linear
Linear = modules.ZeroMeanLinear
#Linear = modules.TensionLinear
Plin = modules.ProbingLinear
#Linear = modules.HungryGdiffLinear
#Linear = modules.HungryVGMLinear
#Linear = modules.HungryLinear
Orthlin = modules.OrthogonalLinear
#settings :)
#intest
soloblock       = False
loopblock       = False
spl             = False
#good in shkspr-char
mix_squish      = False

#good in owt-char
qrot            = False
diffac          = False
rms_lnorm       = False
#good in owt
emb_norm        = True
wnHead          = False #
qksink          = False#
qkfft           = False
qkwn            = True#
pattmlp         = False#

hcmlp           = False
v_patt          = False
hcnvc           = False
hcnHead         = False
#bad
posembless      = False

mem_mix_squish  = False #or fftmem
causal_mem      = False
convemb         = False
diffembd        = False
qope            = False
side_squish     = False
#slow
think           = False
repeat_block    = False
repeat_center   = False

qrottention     = False
symloss         = False
midchaos        = False

specl           = False
topoloss        = False
acebtl          = False
acl1            = False # unimplemented

zeroinit        = False
#ungraded
fftmem          = False
qmem            = False
mmnorm          = False
normless        = False
orthqkvc        = False
orthHead        = False

plin            = False
nmix            = False
losspred        = False

dynskip         = False
k_atten         = False
p_atten         = False
sprs            = False
wnll            = False

csbr            = False

posembdrop      = False
exposemb        = False

residless       = False


residcomp       = False
mhc_lr          = 0.01
mlp_ptok        = 4096
v_ptok          = 256
#known good
ssmax           = True
mtp             = False
headChange      = hcnHead or orthHead#  wnHead reparms apparently work?
CCE             = False and not headChange

auxmod          = acl1 or acebtl #or topoloss

blockin         = spl or symloss or midchaos or topoloss or specl or dynskip


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.n_embd))
        self.bias = nn.Parameter(torch.zeros(config.n_embd)) if config.bias else None
       

    def forward(self, input):
        #if(tests):
        #    return input
        if rms_lnorm:
            return self.weight * utils.rms_norm(input)
        if(normless):
            return input
        if(mmnorm):
            return utils.soft_range_clip_norm(input, dim=-1, scale=self.weight) 
            #return torch.tanh(input)*self.weight.norm(p=2,keepdim=True)
           # return utils.acwrap(input*self.weight, torch.tanh, self.bias)
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

def get_dct_matrix(N):
    dct_m = torch.zeros(N, N)
    for k in range(N):
        for n in range(N):
            weight = math.cos((math.pi / N) * (n + 0.5) * k)
            if k == 0:
                weight *= math.sqrt(1 / N)
            else:
                weight *= math.sqrt(2 / N)
            dct_m[k, n] = weight
    return dct_m

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        
        self.q_attn = (Linear(config.n_embd, config.n_embd, bias=config.bias))
        self.k_attn = (Linear(config.n_embd, config.n_embd, bias=config.bias))
        
        
        if v_patt:
            self.v_attn = modules.PattLayer(config.n_embd,  config.n_embd, v_ptok)# bias=config.bias)
        else:
            self.v_attn = Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=config.bias)
        


        if hcnvc:
            self.v_attn = modules.HCLinear(config.n_embd, config.n_embd, 1.0, False)
            self.c_proj = modules.HCLinear(config.n_embd, config.n_embd, 1.0, False)
        

        if orthqkvc:
            self.q_attn = Orthlin(config.n_embd, config.n_embd, bias=config.bias)
            self.k_attn = Orthlin(config.n_embd, config.n_embd, bias=config.bias)
            self.v_attn = Orthlin(config.n_embd, config.n_embd, bias=config.bias)
            self.c_proj = Orthlin(config.n_embd, config.n_embd, bias=config.bias)
        
        if plin:
            self.q_attn = Plin(config.n_embd, config.n_embd, bias=config.bias)
            self.k_attn = Plin(config.n_embd, config.n_embd, bias=config.bias)
            self.v_attn = Plin(config.n_embd, config.n_embd, bias=config.bias)
            self.c_proj = Plin(config.n_embd, config.n_embd, bias=config.bias)
        
        if qkwn:
            self.q_attn = wn(self.q_attn)
            self.k_attn = wn(self.k_attn)
            #self.q_attn = modules.HCLinear(config.n_embd, config.n_embd, 1.0, True)
            #self.k_attn = modules.HCLinear(config.n_embd, config.n_embd, 1.0, True)
        
        if qksink:
            self.energy_bind1 = modules.SinkhornLinear(config.n_embd)
            self.energy_bind2 = modules.SinkhornLinear(config.n_embd)
        
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head 
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim =  self.n_embd //  self.n_head 
        
        self.qm = nn.Parameter(torch.ones(self.head_dim).normal_(mean=1, std=0.1))
    
    def apply_fft_processing(self, x, sinkhorn_layer):
        x_f = torch.fft.rfft(x.float(), n=x.shape[-1], dim=-1, norm='ortho')
        
        mag = x_f.abs() + 1e-8
        phase = x_f.angle()
        
        norm_mag= utils.rms_norm(mag,dim=-1)
        processed_mag = sinkhorn_layer(norm_mag)
        processed_complex = processed_mag * torch.exp(torch.complex(torch.zeros_like(phase),phase))

        return torch.fft.irfft(processed_complex, n=x.shape[-1], dim=-1, norm='ortho')
    def forward(self, x, causal=True):
        B, T, C = x.size() 
        
        qkx= x
        if qkfft:
            qkx = torch.fft.fft(x, dim=-1).abs()
            qkx = utils.rms_norm(qkx)

        if qksink:
            q = (self.q_attn(self.energy_bind1(qkx)))
            k = (self.k_attn(self.energy_bind2(qkx)))
        else:
            q = self.q_attn(qkx)
            k = self.k_attn(qkx)
            
        v = (self.v_attn(x))
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)

        if ssmax:
            q_scale = math.log(T)*self.qm
        else:
            q_scale = 1.0
        y = torch.nn.functional.scaled_dot_product_attention(q*q_scale,k,  v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal= causal)
            
        y = y.transpose(1, 2).contiguous().view(B, T, C) 
        y = self.resid_dropout(self.c_proj(y))

        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        up = 4
        if pattmlp:
            
            self.thing = modules.PattLayer(
                config.n_embd,
                config.n_embd,
                mlp_ptok,
            )
           
            return
        
        self.c_fc    = Linear(config.n_embd, up * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = Linear( up * config.n_embd, config.n_embd, bias=config.bias)
        if hcmlp:
            self.c_fc = modules.HCLinear(config.n_embd, up*config.n_embd, 1.0, False)
            self.c_proj = modules.HCLinear(up*config.n_embd, config.n_embd, 1.0, False)
        if(qrot):
            self.c_proj  = Linear( 2 * config.n_embd, config.n_embd, bias=config.bias)
        
        self.dropout = nn.Dropout(config.dropout)
        self.dac = utils.Puntan(config.n_embd*4)
        #self.dac = utils.Psinrelu(config.n_embd*4)

    def forward(self, x):
        if pattmlp:
            return (self.thing(x))
            #return torch.sigmoid(self.thing(x))
        if qrot:
            return self.qrot(x)
        
        x = self.c_fc(x)
        if diffac:
            x = self.dac(x)
        else:
            x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

    def qrot(self, x):
        x = self.c_fc(x)
        b, t, n = x.shape
        x4d = x.view(b, t, self.n_embd, 4) # Reshape to (batch, seq_len, n_embd, 4)
        x_norm = torch.clamp_min(x4d.norm(dim=-1, keepdim=True), 1e-10) # Calculate norm along the last dimension (dim=3 or -1), keep dimensions
        x_normalized = x4d / x_norm             # Divide to normalize
        rotors, rotated = x_normalized.chunk(2, dim=2) # Split along n_embd dimension
            #xm,ym = x_norm.chunk(2, dim=2)
        x_rotated = utils.quaternion_multiply(rotors, rotated) 
            #x_rotated = x_rotated*self.gaprelu(x_norm-2)#*torch.sigmoid(xm+ym)#
        x = x_rotated.view(b, t, 2 * self.n_embd)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
        #masked_outputs = stacked_outputs * selection_mask
        #return torch.sum(masked_outputs, dim=-1)
        
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = Attention(config)
        if(qrottention):
            self.attn = modules.SSM(config.n_embd,config.dropout,config.bias)
        elif k_atten:
            self.attn = modules.Kattention(config)
        elif p_atten:
            self.attn = modules.Pattention(config,64)
        if residcomp:
            self.shl = modules.SinkhornLinear(config.n_embd)
            self.shl2 = modules.SinkhornLinear(config.n_embd)
            with torch.no_grad():
                self.shl.weight.data = (torch.eye(config.n_embd))
                self.shl2.weight.data = (torch.eye(config.n_embd))
           
        self.ln_1 = LayerNorm(config)
        self.ln_2 = LayerNorm(config)
        self.mlp = MLP(config)
        self.register_buffer("block_input",torch.zeros([config.block_size, config.n_embd]))
        if fftmem:
            self.register_buffer("mem_input",torch.zeros([config.block_size, config.n_embd]))
        

    def forward(self, x, causal = True):
        self.block_input = x
        
        if p_atten:
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x
        

        if(qrottention):
            x = x + self.attn(self.ln_1(x))[0]
            x = x + self.mlp(self.ln_2(x))
            return x
        
        if residless:
            x = self.attn(self.ln_1(x), causal)
            x = self.mlp(self.ln_2(x))
        elif residcomp:
            #y = self.attn(self.ln_3(x), causal)
            y = self.shl(x) 
            x = y + self.attn(self.ln_1(x), causal)
            #y = self.mlp(self.ln_4(x))
            y = self.shl2(x) 
            x = y + self.mlp(self.ln_2(x))
        else:
            x = x + self.attn(self.ln_1(x), causal)
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

class Block2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = Attention(config)
        
        self.ln_1 = LayerNorm(config)
        self.ln_2 = LayerNorm(config)
        self.mlp = MLP(config)
        self.past_attn = modules.CrossAttention(config.n_embd,config.n_head,config.dropout,config.bias)
        self.past_attn2 = modules.CrossAttention(config.n_embd,config.n_head,config.dropout,config.bias)
        self.ln_3_1 = LayerNorm(config)
        self.ln_4_1 = LayerNorm(config)
        self.ln_3_2 = LayerNorm(config)
        self.ln_4_2 = LayerNorm(config)
    def forward(self, x, past=None, causal=True):
        if isinstance(x,tuple):
            past = x[1]
            x=x[0]
        B, T, D = x.shape
        
        if past is not None:
            P = past.shape[2]
            
            q_flat = x.view(B * T, 1, D)
            
            kv_flat = past.view(B * T, P, D)
            
            resid_flat = self.past_attn(self.ln_3_1(kv_flat), self.ln_3_2(q_flat), causal=False)
            
            resid = (resid_flat.view(B, T, D))
            
            x = x + resid + self.attn(self.ln_1(x), causal)
            
            past = torch.cat([past, resid.unsqueeze(2)], dim=2)
            
            P2 = past.shape[2] 
            q_flat2 = x.view(B * T, 1, D)
            kv_flat2 = past.view(B * T, P2, D)
            
            resid_flat2 = self.past_attn2(self.ln_4_1(kv_flat2), self.ln_4_2(q_flat2), causal=False)
            resid2 = (resid_flat2.view(B, T, D))
            
            x = x + resid2 + self.mlp(self.ln_2(x))
            
            past = torch.cat([past, resid2.unsqueeze(2)], dim=2)
            
            return x, past
            
        else:
            resid = x.unsqueeze(2)
            x = x + self.attn(self.ln_1(x), causal)
            
            resid2 = x.unsqueeze(2)
            x = x + self.mlp(self.ln_2(x))
            
            past = torch.cat([resid, resid2], dim=2)
            
            return x, past

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    internal_vocab_size_multi: int = 10
    internal_vocab_size: int = 0
    internal_block_size: int = 30
    device='cuda'
    optimizer: str = 'adam'
    mem_block_size: int = 64
    batch_size = 64
    step = 0

wn = torch.nn.utils.parametrizations.weight_norm

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.n_embd = config.n_embd
        self.config = config
        blocks = nn.ModuleList([Block2(config) for _ in range(config.n_layer)])
        
        if(exposemb):
            wpe = nn.Embedding(config.block_size//2, config.n_embd)
        else:
            wpe = nn.Embedding(config.block_size, config.n_embd)

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            
            wpe = wpe,
            drop = nn.Dropout(config.dropout),
            h = blocks,
            ln_f = LayerNorm(config),
        ))

        if orthHead:
            self.lm_head = Orthlin(config.n_embd, config.vocab_size, bias=False)
        else:
            self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)
            self.transformer.wte.weight = self.lm_head.weight 
        
        if wnHead:
            self.lm_head = wn(self.lm_head)
                
        if hcnHead:
            self.lm_head = modules.HCLinear(config.n_embd, config.vocab_size)
            self.transformer.wte.weight = self.lm_head.weight 

        self.wnl = utils.wnormlearnloss(config.n_embd)
        
        self.sphead = (Linear(config.n_embd, 1))
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, module in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(module, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
       
        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def forward(self, idx, targets=None, decaying = 0.0, get_emb = False):
        
        device = idx.device
        b, t = idx.size()
        
        if not exposemb:
            assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        
        tok_emb = self.transformer.wte(idx)
        pos_emb = None
        if exposemb:
            pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
            pos = pos%(self.config.block_size//2)
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (b, t, n_embd)
        else:
            pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (b, t, n_embd)
        
        if(posembdrop):
            noiseramp = torch.randn_like(pos_emb) * (1-decaying)
            demb = (pos_emb+noiseramp) * decaying
            x = tok_emb + demb 
        elif posembless:
            x = tok_emb
        else:
            x = tok_emb + pos_emb
        if get_emb:
            x.retain_grad()
            f_emb =  x
        
        x = self.transformer.drop(x)

        if emb_norm:
            x = x-x.mean()
            #x = utils.rms(x)
            #x = utils.quatnorm(x)
        
        
        
        for i, block in enumerate(self.transformer.h):
            x = block(x) 
        if isinstance(x,tuple):
            x = x[0]
        wnl = 0
        if self.training:
            if wnll:
                wnl = self.wnl(x)
        
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = None# self.lm_head(x)
            reduce = 'mean'
            if wnll:
                reduce = 'none'
            if CCE:
                loss = linear_cross_entropy(x.to(torch.bfloat16), 
                                            self.lm_head.weight.to(torch.bfloat16), 
                                            targets, 
                                            impl='torch_compile',
                                            reduction=reduce,)
            else:
                
                logits = self.lm_head(x)
                
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),reduction=reduce)
            
            
            if self.training:
                
                if spl:
                    loss = loss+ self.transformer.h[0].spl(x)
                if wnll:
                    mean_surprise = ((1.0-wnl.detach())/2.0).mean()*0.99
                    loss = loss / (1+ wnl.detach()*mean_surprise)
                    loss = (loss + wnl).mean()
                if losspred:
                    loss = (loss + F.mse_loss(self.botl(self.lpb(x)).squeeze(-1), loss.detach()))/2.0
            else:
                if wnll:
                    loss = loss.mean()
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
        if get_emb:
            return logits, loss, f_emb, x
        
        return logits, loss
    def loopfwd(self, x, targets):
        b,t,c = x.shape
        blocks=10
        blockouts=torch.zeros(b,t,c,blocks,device=x.device,dtype=x.dtype)
        block_lps=torch.zeros(b,t,blocks,device=x.device,dtype=x.dtype)
        
        for i in range(blocks):
            x = self.transformer.h[0](x) 
            
            lp = self.sphead(x).squeeze(-1) 
            blockouts[...,i] = x
            block_lps[...,i] = lp
        
        bm = block_lps.mean(dim=-2)
        onehot = quantizer.TopKHot.apply(-bm,1)
        x = (blockouts * onehot[:, None, None, :]).sum(dim=-1)
        x_flat = blockouts.permute(0, 1, 3, 2).reshape(-1, blockouts.shape[2])
        x_flat = self.transformer.ln_f(x_flat)
        t_flat = targets.unsqueeze(-1).expand(-1, -1, blocks).reshape(-1)
        
        future_ce_loss = linear_cross_entropy(
            x_flat.to(torch.bfloat16), 
            self.lm_head.weight.to(torch.bfloat16), 
            t_flat, 
            impl='torch_compile', 
            reduction='none'
        ).detach()
        loopaux = F.mse_loss(
            block_lps[..., -1], 
            future_ce_loss[..., 1:],
        )
        return x, loopaux
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        
        mhc_params = []
        decay_params = []
        nodecay_params = []

        for n, p in param_dict.items():
            # shl named modules have slower lr
            if 'shl' in n:
                mhc_params.append(p)
            elif p.dim() >= 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)
        if 'muon' in self.config.optimizer:
            other_groups = [
                {'params': nodecay_params,  'weight_decay': 0.0},
            ]
            muon_groups = [
                {'params': decay_params,    'weight_decay': weight_decay},
                {'params': mhc_params,      'weight_decay': 0.0, 'lr_scale': mhc_lr},
                #{'params': self_lr, 'lr': 1.0},
            ]
        else:
            optim_groups = [
                {'params': nodecay_params,  'weight_decay': 0.0},
                {'params': decay_params,    'weight_decay': weight_decay},
                {'params': mhc_params,      'weight_decay': 0.0, 'lr_scale': mhc_lr},
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
            case "bubbles":
                base = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
                optimizer = optim.ChaosRollbackOptimizer(self, base_optimizer=base)
            case "muon":
                optimizer = torch.optim.Muon(muon_groups, lr=learning_rate)
                optimizer2 = torch.optim.AdamW(other_groups, lr=learning_rate, betas=betas, **extra_args)
                return optimizer,optimizer2
            #optimizer =torch.optim.Adadelta(optim_groups)#, **extra_args)# lr=learning_rate, **extra_args)
        print(f"using fused AdamW: {use_fused}")
                        
        return optimizer, None

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
    