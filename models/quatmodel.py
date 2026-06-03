import math
import os
import time

from attr import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from parts.modules import HungryLinear, HungryPattLayer, PattLayer
import parts.utils as utils
torch._inductor.config.disable_cpp_codegen = True

Linear          = HungryLinear
Patt            = HungryPattLayer
#dataset        = "shakespeare_char"
dataset         = "openwebtext"
data_dir        = os.path.join('data', dataset)

device          = 'cuda' if torch.cuda.is_available() else 'cpu'
bias            = False

batch_size      = 64
block_size      = 64

pattcount       = 512

n_embd          = 256
n_head          = 8
n_layer         = 4
dropout         = 0.0
weight_decay    = 1e-1

eval_interval   = 1000
eval_iters      = 200
warmup_iters    = 100
max_iters       = 50000
lr_decay_iters  = 30000
decay_iters     = 5000

learning_rate   = 5e-4
min_lr          = 5e-5

spl             = False
soloblock       = False
sspl            = False
wnll            = True
patt            = False

# New position embedding and activation features
rope            = False
qrope           = True
qrot            = False

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
train_logging = False
exec(open('training_tools/configurator.py').read()) # overrides from command line or config file

def get_lr(it):
    
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    if it > lr_decay_iters:
        return min_lr
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
    return min_lr + coeff * (learning_rate - min_lr)

decode, vocab_size = utils.get_meta(dataset,always_byte=True)
get_batch = utils.get_get_batch(block_size,batch_size,dataset,device,always_byte=True)


# --- Helper Classes & Functions for Rotary & Quaternion Operations ---

def quaternion_multiply(q1, q2):
    """
    Computes the Hamilton product of two quaternion tensors of shape (..., 4).
    """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return torch.stack((w, x, y, z), dim=-1)


class RotaryEmbedding(nn.Module):
    """
    Standard 2D Rotary Position Embedding (RoPE)
    """
    def __init__(self, dim, theta=10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)  # (T, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, dim)
        return emb.cos(), emb.sin()


def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    # x shape: (B, nh, T, hs)
    # cos, sin shape: (T, hs)
    cos = cos.unsqueeze(0).unsqueeze(1)  # (1, 1, T, hs)
    sin = sin.unsqueeze(0).unsqueeze(1)  # (1, 1, T, hs)
    return (x * cos) + (rotate_half(x) * sin)


class QuaternionRotaryEmbedding(nn.Module):
    """
    Quaternion Rotary Position Embedding (Q-RoPE)
    Rotates 4D chunks of embeddings using unit-quaternion Hamilton products.
    """
    def __init__(self, dim, theta=10000.0):
        super().__init__()
        self.dim = dim
        # We need dim // 4 frequency bands for 4D quaternions
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 4).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)  # (T, dim/4)
        
        cos = freqs.cos()
        sin = freqs.sin()
        
        # Symmetrical axis: u = [1, 1, 1] / sqrt(3)
        u_factor = 1.0 / math.sqrt(3.0)
        
        w = cos
        x = sin * u_factor
        y = sin * u_factor
        z = sin * u_factor
        
        rotors = torch.stack((w, x, y, z), dim=-1)  # (T, dim/4, 4)
        return rotors


def apply_qrope(x, rotors):
    # x shape: (B, nh, T, hs)
    # rotors shape: (T, hs // 4, 4)
    B, nh, T, hs = x.shape
    x_4d = x.view(B, nh, T, hs // 4, 4)
    rotors = rotors.unsqueeze(0).unsqueeze(1)  # (1, 1, T, hs // 4, 4)
    
    # Left multiplication: q_out = rotors * x_4d
    out = quaternion_multiply(rotors, x_4d)
    return out.view(B, nh, T, hs)


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int    = 12
    n_head: int     = 12
    n_embd: int     = 768
    dropout: float  = 0.0
    bias: bool      = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    
    device          = 'cuda'
    optimizer: str  = 'adam'
    batch_size      = 64
    step            = 0

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.n_embd))
        self.bias = nn.Parameter(torch.zeros(config.n_embd)) if config.bias else None
       
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        
        self.q_attn = (Linear(config.n_embd, config.n_embd, bias=config.bias))
        self.k_attn = (Linear(config.n_embd, config.n_embd, bias=config.bias))
        
        self.v_attn = Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head 
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim =  self.n_embd //  self.n_head 
        
        self.qm = nn.Parameter(torch.ones(self.head_dim).normal_(mean=1, std=0.1))
        
        if rope:
            self.rope_emb = RotaryEmbedding(self.head_dim)
        elif qrope:
            assert self.head_dim % 4 == 0, "head_dim must be divisible by 4 for Q-RoPE"
            self.qrope_emb = QuaternionRotaryEmbedding(self.head_dim)
    
    def forward(self, x, causal=True):
        B, T, C = x.size() 
        
        q = self.q_attn(x)
        k = self.k_attn(x)
        v = self.v_attn(x)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)

        if rope:
            cos, sin = self.rope_emb(T, x.device)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        elif qrope:
            rotors = self.qrope_emb(T, x.device)
            q = apply_qrope(q, rotors)
            k = apply_qrope(k, rotors)

        q_scale = math.log(T)*self.qm
        y = torch.nn.functional.scaled_dot_product_attention(q*q_scale,k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal= causal)
       
        y = y.transpose(1, 2).contiguous().view(B, T, C) 
        
        y = self.resid_dropout(self.c_proj(y))

        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        up = 4
        
        self.c_fc    = Linear(config.n_embd, up * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        
        if qrot:
            self.c_proj  = Linear(2 * config.n_embd, config.n_embd, bias=config.bias)
        else:
            self.c_proj  = Linear(up * config.n_embd, config.n_embd, bias=config.bias)
        
        self.dropout = nn.Dropout(config.dropout)
        if patt:
            self.patt = Patt(config.n_embd, config.n_embd, pattcount)

    def forward(self, x):
        if qrot:
            return self.qrot(x)
        if patt:
            return self.patt(x)
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
        
    def qrot(self, x):
        x = self.c_fc(x)
        b, t, n = x.shape
        x4d = x.view(b, t, self.n_embd, 4) # Reshape to (batch, seq_len, n_embd, 4)
        x_norm = torch.clamp_min(x4d.norm(dim=-1, keepdim=True), 1e-10) # Calculate norm along the last dimension, keep dimensions
        x_normalized = x4d / x_norm             # Divide to normalize
        rotors, rotated = x_normalized.chunk(2, dim=2) # Split along n_embd dimension
        
        x_rotated = quaternion_multiply(rotors, rotated) 
        x = x_rotated.view(b, t, 2 * self.n_embd)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = Attention(config)
        
        self.ln_1 = LayerNorm(config)
        self.ln_2 = LayerNorm(config)
        self.mlp = MLP(config)
        self.register_buffer("block_input",torch.zeros([config.block_size, config.n_embd]))
        
    def forward(self, x, causal = True):
        self.block_input = x
        
        
        x = x + self.attn(self.ln_1(x), causal)
        x = x + self.mlp(self.ln_2(x))
        return x

    def spl(self, x):
        predicted_output = self(-x, causal = False)
        
        sploss = F.mse_loss(
            utils.rms_norm(predicted_output.flatten()),
            utils.rms_norm(self.block_input.flatten())
            #utils.range_norm(predicted_output.flatten()),
            #utils.range_norm(self.block_input.flatten())
        ) 
        return sploss
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.n_embd = config.n_embd
        self.config = config
        blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        
        transformer_dict = dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = blocks,
            ln_f = LayerNorm(config),
        )
        if not (rope or qrope):
            transformer_dict['wpe'] = nn.Embedding(config.block_size, config.n_embd)
            
        self.transformer = nn.ModuleDict(transformer_dict)

        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight 
        
        self.wnl = utils.wnormlearnloss(config.n_embd)
        
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, module in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(module, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
       

    def forward(self, idx, targets=None, decaying = 0.0, get_emb = False):
        
        device = idx.device
        b, t = idx.size()
        
        
        tok_emb = self.transformer.wte(idx)

        if rope or qrope:
            x = tok_emb
        else:
            pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (b, t, n_embd)
            x = tok_emb + pos_emb
        
        x = self.transformer.drop(x)
        
        if self.training:
            if wnll:
                wnl = self.wnl(x)
        x1 = x
        for i, block in enumerate(self.transformer.h):
            if soloblock:
                x = self.transformer.h[0](x) 
            else:
                x = block(x) 


        if isinstance(x,tuple):
            x = x[0]
       
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        if targets is not None:
            reduce = 'mean'
            if wnll:
                reduce = 'none'
            
                
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),reduction=reduce)
            
            
            if self.training:
                if wnll:
                    #cumsum_wnl = torch.cumsum(wnl, dim=1)
                    #indices = torch.arange(1, wnl.size(1) + 1, device=wnl.device).view(1, -1, 1)
                    #cumulative_surprise_avg = cumsum_wnl / indices
                    #wnl = cumulative_surprise_avg
                    #wnl = cumulative_surprise_avg
                    mean_surprise = ((1.0-wnl.detach())/2.0).mean()*0.99
                    loss = loss / (1+ wnl.detach()*mean_surprise)
                    loss = (loss + wnl).mean()
                    #loss = (loss * torch.exp(-wnl) + wnl).mean()
                if spl:
                    if sspl:
                        self.transformer.h[1].block_input = x1
                        loss = loss + self.transformer.h[1].spl(x)
                    else:
                        loss = loss + self.transformer.h[0].spl(x)
            else:
                if wnll:
                    loss = loss.mean()
        else:
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
        
        
        return logits, loss
    
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
    

    def configure_optimizers(self, lr, wd):
        
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        decay_params = []
        nodecay_params = []
        for n, p in param_dict.items():
            if p.dim() >= 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)
        optim_groups = [
                {'params': nodecay_params,  'weight_decay': 0.0},
                {'params': decay_params,    'weight_decay': wd},
        ]
        return torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9,0.999))

    
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=vocab_size, dropout=dropout) 
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)
optimizer = model.configure_optimizers(lr=learning_rate, wd=weight_decay)

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
torch.set_float32_matmul_precision('medium')
#dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler

model.compile()
print(f"Starting training on {device}")
print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

            
t0 = time.time()
t1 = time.time()
t2 = time.time()

def eval(max_iters, get_batch, model, optimizer, t0,t1,t2, iter, decaying):
    
    t2 = time.time()
    model.eval()
    losses = {}
    for split in ['train', 'val']:
        losses[split] = 0.0
        for k in range(eval_iters):
            X_val, Y_val = get_batch(split)
            with torch.no_grad():
                _, loss = model(X_val, Y_val, decaying= decaying)
            losses[split] += loss.detach().item()
        losses[split] /= eval_iters
    optimizer.zero_grad(set_to_none=True)
    t1 = time.time()
    print(f"step {iter:{len(str(max_iters))}d}: ",end="")
    print(f"train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}, ",end="")
    print(f"time: {(t2-t0)*1000:.0f} ms, Tval: {(t1-t2)*1000:.0f} ms")
    model.train()
    t0 = time.time()
    return t0, t1, t2

for iter in range(max_iters):
    
    lr = get_lr(iter)
    decaying = 1.0 - (iter / decay_iters)
    decaying = torch.tensor(decaying, device=device, dtype=torch.float32)
    decaying = torch.clamp(decaying, 0.0, 1.0)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    torch.compiler.cudagraph_mark_step_begin()
    X, Y = get_batch('train')
    logits, loss = model(X, Y, decaying= decaying)
    loss.backward()
    optimizer.step()
    if iter % eval_interval == 0 or iter == max_iters - 1:
        t0,t1,t2 =eval(max_iters, get_batch, model, optimizer, t0,t1,t2, iter, decaying)
    optimizer.zero_grad(set_to_none=True)