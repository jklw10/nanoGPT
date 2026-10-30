import math
import os
import time

from attr import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from parts.modules import HungryLinear, HungryPattLayer, PattLayer
import parts.utils as utils
from unit import bench
torch._inductor.config.disable_cpp_codegen = True

torch._dynamo.config.force_parameter_static_shapes = False

Linear          = nn.Linear
Patt            = HungryPattLayer
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
weight_decay    = 0.0 #1e-1

eval_interval   = 1000
eval_iters      = 200
warmup_iters    = 100
max_iters       = 50000
lr_decay_iters  = 30000
decay_iters     = 5000

learning_rate   = 1e-3
min_lr          = learning_rate / 10

spl             = False
soloblock       = False
sspl            = False
wnll            = False
patt            = False
gen             = False

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

decode, vocab_size = utils.get_meta(dataset, always_byte=True)
get_batch = utils.get_get_batch(block_size, batch_size, dataset, device, always_byte=True)

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int    = 12
    n_head: int     = 12
    n_embd: int     = 768
    dropout: float  = 0.0
    bias: bool      = False
    
    device          = 'cuda'
    optimizer: str  = 'adam'
    batch_size      = 64
    step            = 0

class LayerNorm(nn.Module):
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
        
        self.q_attn = Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_attn = Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_attn = Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head 
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim = self.n_embd // self.n_head 
        
        self.qm = nn.Parameter(torch.ones(self.head_dim).normal_(mean=1, std=0.1))
    
    def forward(self, x, causal=True):
        B, T, C = x.size() 
        
        q = self.q_attn(x)
        k = self.k_attn(x)
        v = self.v_attn(x)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q_scale = math.log(T) * self.qm
        y = torch.nn.functional.scaled_dot_product_attention(q * q_scale, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=causal)
       
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
        self.c_proj  = Linear(up * config.n_embd, config.n_embd, bias=config.bias)
        
        self.dropout = nn.Dropout(config.dropout)
        if patt:
            self.patt = Patt(config.n_embd, config.n_embd, pattcount)

    def forward(self, x):
        if patt:
            return self.patt(x)
        x = self.c_fc(x)
        x = self.gelu(x)
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
        self.register_buffer("block_input", torch.zeros([config.block_size, config.n_embd]))
        
    def forward(self, x, causal=True, fwd=True):
        if fwd:
            self.block_input = x
        x = x + self.attn(self.ln_1(x), causal)
        x = x + self.mlp(self.ln_2(x))
        return x

    def spl(self, x):
        predicted_output = self(-x, causal=False,fwd=False)
        sploss = F.mse_loss(
            utils.rms_norm(predicted_output.flatten()),
            utils.rms_norm(self.block_input.flatten())
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
        
        wpe = nn.Embedding(config.block_size, config.n_embd)

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = wpe,
            drop = nn.Dropout(config.dropout),
            h = blocks,
            ln_f = LayerNorm(config),
        ))

        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight 
        
        self.wnl = utils.wnormlearnloss(config.n_embd)
        
        self.apply(self._init_weights)
        for pn, module in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(module, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def forward(self, idx, targets=None, decaying=0.0, get_emb=False):
        device = idx.device
        b, t = idx.size()
        
        tok_emb = self.transformer.wte(idx)
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        pos_emb = self.transformer.wpe(pos)
        
        x = tok_emb + pos_emb
        x = self.transformer.drop(x)
        
        if self.training and wnll:
            wnl = self.wnl(x)
        x1 = x
        for i, block in enumerate(self.transformer.h):
            if soloblock:
                x = self.transformer.h[0](x) 
            else:
                x = block(x) 

        if isinstance(x, tuple):
            x = x[0]
       
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        if targets is not None:
            reduce = 'none' if wnll else 'mean'
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction=reduce)
            
            if self.training:
                if wnll:
                    mean_surprise = ((1.0 - wnl.detach()) / 2.0).mean() * 0.99
                    loss = loss / (1 + wnl.detach() * mean_surprise)
                    loss = (loss + wnl).mean()
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
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        
        return logits, loss
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, Linear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Embedding, nn.Conv1d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

def create_optimizer(model, opt_type, lr, wd):
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': nodecay_params, 'weight_decay': 0.0},
        {'params': decay_params, 'weight_decay': wd},
    ]
    if opt_type == 'adamw':
        return torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    elif opt_type == 'jordan':
        return CurlFreeAdamW(optim_groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")


class CurlFreeAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, gamma=0.5):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, gamma=gamma)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, (beta1, beta2), eps, wd = group['lr'], group['betas'], group['eps'], group['weight_decay']
            gamma = group['gamma']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                # 1. Decoupled weight decay
                if wd != 0:
                    p.mul_(1.0 - lr * wd)

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1
                step = state['step']
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # 2. Standard Adam moment updates
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).add_(grad.pow(2), alpha=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step

                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                # 3. Square-root preconditioner A^{1/2}
                d_scale = 1.0 / (torch.sqrt(v_hat) + eps)
                d_half = torch.sqrt(d_scale)

                # 4. Unit Gradient Direction u
                g_norm = torch.norm(grad)
                if g_norm > 1e-8:
                    u = grad / g_norm
                    
                    # 5. Orthogonal Momentum Direction v (v ⊥ u)
                    m_parallel = torch.sum(u * m_hat) * u
                    m_ortho = m_hat - m_parallel
                    m_ortho_norm = torch.norm(m_ortho)

                    if m_ortho_norm > 1e-8:
                        v = m_ortho / m_ortho_norm
                        
                        # 6. Contractive Sandwich: H(x) = x - gamma * <v, x> * v
                        x_half = d_half * m_hat
                        dot_v = torch.sum(v * x_half)
                        h_out = x_half - gamma * dot_v * v  # <-- Contractive minus sign
                        jordan_dir = d_half * h_out
                    else:
                        jordan_dir = d_scale * m_hat
                else:
                    jordan_dir = d_scale * m_hat

                # 7. Parameter update
                p.add_(jordan_dir, alpha=-lr)

        return loss

def calc_anisotropy(optimizer):
    d_vals = []
    for group in optimizer.param_groups:
        for p in group['params']:
            if p in optimizer.state and 'exp_avg_sq' in optimizer.state[p]:
                v = optimizer.state[p]['exp_avg_sq']
                d = 1.0 / (torch.sqrt(v) + 1e-8)
                d_vals.append(d.flatten())
    if not d_vals:
        return 0.0
    d_all = torch.cat(d_vals)
    return (d_all.std() / (d_all.mean() + 1e-8)).item()

if __name__ == "__main__":
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                      bias=bias, vocab_size=vocab_size, dropout=dropout) 
    gptconf = GPTConfig(**model_args)

    # 1. Instantiate 2 identical models side-by-side
    torch.manual_seed(1337)
    model_adam = GPT(gptconf).to(device)
    model_jordan = GPT(gptconf).to(device)
    
    # Ensure exact identical initial weight initialization
    model_jordan.load_state_dict(model_adam.state_dict())

    # 2. Instantiate corresponding optimizers
    optimizer_adam = create_optimizer(model_adam, 'adamw', lr=learning_rate, wd=weight_decay)
    optimizer_jordan = create_optimizer(model_jordan, 'jordan', lr=learning_rate, wd=weight_decay)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('medium')

    model_adam.compile()
    model_jordan.compile()

    print(f"Starting Side-by-Side Training Comparison on {device}")
    print(f"Model Parameters per model: {sum(p.numel() for p in model_adam.parameters()):,}")

    t0 = time.time()
    t1 = time.time()
    t2 = time.time()

    def eval(max_iters, get_batch, model_adam, model_jordan, optimizer_adam, optimizer_jordan, t0, t1, t2, iter, decaying):
        t2 = time.time()
        model_adam.eval()
        model_jordan.eval()
        losses = {'train_adam': 0.0, 'val_adam': 0.0, 'train_jordan': 0.0, 'val_jordan': 0.0}
        
        for split in ['train', 'val']:
            for k in range(eval_iters):
                X_val, Y_val = get_batch(split, block_size)
                with torch.no_grad():
                    _, loss_a = model_adam(X_val, Y_val, decaying=decaying)
                    _, loss_j = model_jordan(X_val, Y_val, decaying=decaying)
                losses[f"{split}_adam"] += loss_a.detach().item()
                losses[f"{split}_jordan"] += loss_j.detach().item()
            losses[f"{split}_adam"] /= eval_iters
            losses[f"{split}_jordan"] /= eval_iters

        optimizer_adam.zero_grad(set_to_none=True)
        optimizer_jordan.zero_grad(set_to_none=True)
        t1 = time.time()
        
        print(f"\n" + "-" * 105)
        print(f"EVALUATION [Step {iter:{len(str(max_iters))}d}]")
        print(f"AdamW  : Train Loss = {losses['train_adam']:.4f} | Val Loss = {losses['val_adam']:.4f}")
        print(f"Jordan : Train Loss = {losses['train_jordan']:.4f} | Val Loss = {losses['val_jordan']:.4f}")
        print(f"Time: {(t2-t0)*1000:.0f} ms, Tval: {(t1-t2)*1000:.0f} ms")
        print("-" * 105 + "\n")

        model_adam.train()
        model_jordan.train()
        t0 = time.time()
        return t0, t1, t2

    print("\n" + "=" * 115)
    print(f"{'Step':<6} | {'Adam Loss':<10} | {'Jrdn Loss':<10} | {'Step CosSim':<12} | {'Adam Wobble':<12} | {'Jrdn Wobble':<12} | {'Adam Aniso':<11} | {'Jrdn Aniso':<11}")
    print("=" * 115)

    prev_delta_adam = None
    prev_delta_jordan = None

    for iter in range(max_iters):
        lr = get_lr(iter)
        decaying = 1.0 - (iter / decay_iters)
        decaying = torch.tensor(decaying, device=device, dtype=torch.float32)
        decaying = torch.clamp(decaying, 0.0, 1.0)

        for pg_a, pg_j in zip(optimizer_adam.param_groups, optimizer_jordan.param_groups):
            pg_a['lr'] = lr
            pg_j['lr'] = lr

        torch.compiler.cudagraph_mark_step_begin()
        
        # Sample the EXACT SAME data batch for both models
        X, Y = get_batch('train', block_size)

        # --- 1. Forward & Backward: Model 1 (AdamW) ---
        params_adam_before = [p.detach().clone() for p in model_adam.parameters() if p.requires_grad]
        logits_a, loss_a = model_adam(X, Y, decaying=decaying)
        loss_a.backward()
        optimizer_adam.step()

        # Compute actual parameter step vector Δθ_adam
        delta_adam = torch.cat([(p_b - p.detach()).flatten() for p_b, p in zip(params_adam_before, model_adam.parameters()) if p.requires_grad])
        optimizer_adam.zero_grad(set_to_none=True)

        # --- 2. Forward & Backward: Model 2 (Jordan) ---
        params_jordan_before = [p.detach().clone() for p in model_jordan.parameters() if p.requires_grad]
        logits_j, loss_j = model_jordan(X, Y, decaying=decaying)
        loss_j.backward()
        optimizer_jordan.step()

        # Compute actual parameter step vector Δθ_jordan
        delta_jordan = torch.cat([(p_b - p.detach()).flatten() for p_b, p in zip(params_jordan_before, model_jordan.parameters()) if p.requires_grad])
        optimizer_jordan.zero_grad(set_to_none=True)

        # --- 3. Compute True Diagnostic Metrics ---
        if iter % 100 == 0 or iter < 10:
            # Step Cosine Similarity between actual AdamW step and actual Jordan step
            cossim = F.cosine_similarity(delta_adam.unsqueeze(0), delta_jordan.unsqueeze(0)).item()

            # True Adam Wobble: similarity between consecutive Adam steps
            if prev_delta_adam is not None:
                adam_wobble = F.cosine_similarity(delta_adam.unsqueeze(0), prev_delta_adam.unsqueeze(0)).item()
            else:
                adam_wobble = 1.0

            # True Jordan Wobble: similarity between consecutive Jordan steps
            if prev_delta_jordan is not None:
                jordan_wobble = F.cosine_similarity(delta_jordan.unsqueeze(0), prev_delta_jordan.unsqueeze(0)).item()
            else:
                jordan_wobble = 1.0

            adam_aniso = calc_anisotropy(optimizer_adam)
            jordan_aniso = calc_anisotropy(optimizer_jordan)

            print(f"{iter:<6d} | {loss_a.item():<10.4f} | {loss_j.item():<10.4f} | {cossim:<12.6f} | {adam_wobble:<12.4f} | {jordan_wobble:<12.4f} | {adam_aniso:<11.4f} | {jordan_aniso:<11.4f}")

        prev_delta_adam = delta_adam.clone()
        prev_delta_jordan = delta_jordan.clone()

        if iter % eval_interval == 0 or iter == max_iters - 1:
            t0, t1, t2 = eval(max_iters, get_batch, model_adam, model_jordan, optimizer_adam, optimizer_jordan, t0, t1, t2, iter, decaying)