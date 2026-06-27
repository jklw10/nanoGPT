import math
import os
import time

from attr import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from parts.modules import HungryLinear, HungryPattLayer, PattLayer
import parts.utils as utils
import numpy as np
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
patt            = True
always_byte     = True

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

# Pass a dictionary containing task mappings to the 'dataset' parameter
dataset_config = {
    'pretrain': 'openwebtext',
    'sft': 'sft',
    'cot': 'cot'
}

task_probs = {
    'pretrain': 0.40,
    'sft': 0.30,
    'cot': 0.30
}
tasks = list(task_probs.keys())
probs = list(task_probs.values())

# Call signature remains identical to your original code
decode, vocab_size = utils.get_meta('openwebtext', always_byte=always_byte)
get_batch = utils.get_get_batch(
    block_size, 
    batch_size, 
    dataset_config, # Passing the dictionary here
    device, 
    always_byte=always_byte
)

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
    
    # Active Inference Settings
    num_orders: int = 8         # 3 orders = smoothed error, velocity (LP), acceleration
    lp_alpha: float = 0.01       # EMA decay multiplier
    gamma_coupling: float = utils.phi  # Gamma coupling factor for boredom scaling


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
        self.q_attn = (Linear(config.n_embd, config.n_embd, bias=config.bias))
        self.k_attn = (Linear(config.n_embd, config.n_embd, bias=config.bias))
        self.v_attn = Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head 
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim =  self.n_embd //  self.n_head 
        self.qm = nn.Parameter(torch.ones(self.head_dim).normal_(mean=1, std=0.1))
    
    def forward(self, x, causal=True):
        B, T, C = x.size() 
        q = self.q_attn(x)
        k = self.k_attn(x)
        v = self.v_attn(x)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

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
        self.c_proj  = Linear( up * config.n_embd, config.n_embd, bias=config.bias)
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
        
        # -------------------------------------------------------------
        # Tunable-Order Active Inference Setup
        # -------------------------------------------------------------
        self.num_orders = config.num_orders
        self.lp_alpha = config.lp_alpha
        self.gamma_coupling = config.gamma_coupling
        
        # Register cascade buffer: ema_states[0] = error, ema_states[1:] = derivatives
        self.register_buffer("ema_states", torch.zeros(self.num_orders))
        
        # Generative predictor layer (hidden state to next target embedding)
        self.predictor = Linear(config.n_embd, config.n_embd, bias=config.bias)
        # -------------------------------------------------------------

        self.apply(self._init_weights)
        for pn, module in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(module, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def forward(self, idx, targets=None, decaying = 0.0, get_emb = False):
        device = idx.device
        b, t = idx.size()
        
        tok_emb = self.transformer.wte(idx)
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        pos_emb = self.transformer.wpe(pos)
        
        x = tok_emb + pos_emb
        x = self.transformer.drop(x)
        
        if self.training and wnll:
            wnl = self.wnl(x)
            wnl = wnl.flatten()
            
        x1 = x
        for i, block in enumerate(self.transformer.h):
            if soloblock:
                x = self.transformer.h[0](x) 
            else:
                x = block(x) 

        if isinstance(x, tuple):
            x = x[0]
            
        # =========================================================================
        # Active Inference & Tunable-Order Gamma Boredom
        # =========================================================================
        aux_loss = 0.0
        
        if self.training and t > 1:
            current_hidden = x[:, :-1]
            target_emb = tok_emb[:, 1:].detach()
            
            # Unit norm projection to secure latent space geometry
            norm_current = F.normalize(current_hidden, p=2, dim=-1)
            norm_target = F.normalize(target_emb, p=2, dim=-1)
            
            predicted_next = self.predictor(norm_current)
            aux_loss = F.mse_loss(predicted_next, norm_target, reduction='mean')
            
            with torch.no_grad():
                current_val = aux_loss.detach()
                next_ema_states = []
                
                # Recursive derivative EMA chain
                for k in range(self.num_orders):
                    prev_ema = self.ema_states[k]
                    deriv = prev_ema - current_val
                    
                    new_ema = prev_ema * (1.0 - self.lp_alpha) + current_val * self.lp_alpha
                    next_ema_states.append(new_ema)
                    
                    current_val = deriv
                
                self.ema_states.copy_(torch.stack(next_ema_states))
                
                # -------------------------------------------------------------
                # Dimensionless Gamma Boredom Calculation
                # -------------------------------------------------------------
                eps = 1e-8
                smoothed_error = self.ema_states[0]
                
                # Combined progress: sum positive parts of higher-order EMAs
                total_lp = torch.stack([F.relu(self.ema_states[i]) for i in range(1, self.num_orders)]).sum()
                
                # Normalize learning progress by baseline error to make it dimensionless
                relative_lp = total_lp / (smoothed_error + eps)
                relative_lp_clipped = torch.clamp(relative_lp, min=0.0, max=1.0 - eps)
                
                # Gamma coupling exponential decay
                boredom_scale = torch.pow(1.0 - relative_lp_clipped, self.gamma_coupling / 2.0)
                # -------------------------------------------------------------
            
            # Boredom-modulated stochastic foraging (noise perturbation)
            errm = torch.sqrt(boredom_scale) * 0.01
            x = x + torch.randn_like(x) * errm
        # =========================================================================

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is not None:
            reduce = 'mean' if not wnll else 'none'
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction=reduce)
            
            if self.training:
                if wnll:
                    mean_surprise = ((1.0-wnl.detach())/2.0).mean()*0.99
                    loss = loss / (1+ wnl.detach()*mean_surprise)
                    loss = (loss + wnl).mean()
                
                loss = loss + 0.1 * aux_loss
                
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
@torch.no_grad()
def generate(model, prompt_text, device, always_byte, max_new_tokens=100, temperature=0.7, top_k=50):
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    
    was_training = model.training
    model.eval()
    
    # 1. Encode prompt according to byte/token toggle
    if always_byte:
        # Convert string directly to raw UTF-8 bytes (integers 0-255)
        idx_list = list(prompt_text.encode('utf-8'))
        # Replace unexpected out-of-bounds byte codes with a standard space character (32)
        idx_list = [b if b < 256 else 32 for b in idx_list]
    else:
        # Standard GPT-2 BPE tokens
        idx_list = enc.encode_ordinary(prompt_text)
        
    idx = torch.tensor(idx_list, dtype=torch.long, device=device).unsqueeze(0)
    
    # 2. Autoregressive loop
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        
        logits = logits[:, -1, :] / temperature
        
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        
        # Stop on BPE end-of-text marker if not in byte mode
        if not always_byte and idx_next.item() == enc.eot_token:
            break
            
    # 3. Decode output according to byte/token toggle
    result_list = idx[0].tolist()
    if always_byte:
        # Safely decode byte sequences back to UTF-8 text, handling potential half-characters with 'replace'
        valid_bytes = [b for b in result_list if 0 <= b < 256]
        decoded_output = bytes(valid_bytes).decode('utf-8', errors='replace')
    else:
        decoded_output = enc.decode(result_list)
        
    if was_training:
        model.train()
        
    return decoded_output

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=vocab_size, dropout=dropout) 
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)
optimizer = model.configure_optimizers(lr=learning_rate, wd=weight_decay)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

model.compile()
print(f"Starting training on {device}")
print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

t0 = time.time()
t1 = time.time()
t2 = time.time()
def log_periodic_generations(iter, model, device, always_byte):
    prompts = {
        "PRETRAIN": "Once upon a time, in a distant galaxy, there was a",
        "SFT": "Below is an instruction that describes a task. Write a response.\n\n### Instruction:\nList three healthy foods.\n\n### Response:\n",
        "CoT": "Below is a math problem. Think step-by-step.\n\n### Question:\nIf Jerry has 12 apples and eats 4, how many apples are left?\n\n### Thought & Solution:\n"
    }
    
    print("\n" + "=" * 80)
    print(f" STEP {iter} GENERATION RUN (Byte Mode: {always_byte}) ")
    print("=" * 80)
    for label, text in prompts.items():
        output = generate(model, text, device, always_byte, max_new_tokens=80)
        print(f"\n[{label}]\n" + "-"*40 + f"\n{output.strip()}\n" + "-"*40)
    print("=" * 80 + "\n")

def eval(max_iters, get_batch, model, optimizer, t0, t1, t2, iter, decaying, tasks):
    t2 = time.time()
    model.eval()
    losses = {}
    
    for task in tasks:
        for split in ['train', 'val']:
            losses[f"{task}_{split}"] = 0.0
            for k in range(eval_iters):
                X_val, Y_val = get_batch(split, task=task)
                with torch.no_grad():
                    _, loss = model(X_val, Y_val, decaying=decaying)
                losses[f"{task}_{split}"] += loss.detach().item()
            losses[f"{task}_{split}"] /= eval_iters
            
    optimizer.zero_grad(set_to_none=True)
    t1 = time.time()
    
    print(f"\n--- Step {iter:{len(str(max_iters))}d} ---")
    for task in tasks:
        print(f"  {task:<10} | train loss: {losses[f'{task}_train']:.4f} | val loss: {losses[f'{task}_val']:.4f}")
    print(f"  time: {(t2-t0)*1000:.0f} ms, Tval: {(t1-t2)*1000:.0f} ms")
    
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
    
    # Randomly select a task according to defined probabilities
    selected_task = np.random.choice(tasks, p=probs)
    X, Y = get_batch('train', task=selected_task)
    
    logits, loss = model(X, Y, decaying=decaying)
    loss.backward()
    optimizer.step()
    
    if iter % eval_interval == 0 or iter == max_iters - 1:
        t0, t1, t2 = eval(max_iters, get_batch, model, optimizer, t0, t1, t2, iter, decaying, tasks)
    
    if iter > 0 and iter % 10000 == 0:
        log_periodic_generations(iter, model, device, always_byte)
    optimizer.zero_grad(set_to_none=True)

checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'config': gptconf,
    'always_byte': always_byte,
    'vocab_size': vocab_size
}
ckpt_path = 'model_ckpt.pt'
torch.save(checkpoint, ckpt_path)
print(f"\nTraining completed. Model checkpoint saved to '{ckpt_path}'.")

# =============================================================
# Interactive Sanity Chat Session
# =============================================================
print("\n" + "=" * 80)
print(" INTERACTIVE CHAT MODE (Type 'exit' or 'quit' to terminate) ")
print("=" * 80)

while True:
    try:
        user_input = input("\nYou: ")
    except EOFError:
        break
    if user_input.strip().lower() in ['exit', 'quit']:
        break
    
    # Structure the chat sequence as an instruction for instruction-following models
    prompt = (
        f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{user_input}\n\n"
        f"### Response:\n"
    )
    
    # Generate the response
    raw_response = generate(model, prompt, device, always_byte, max_new_tokens=150)
    
    # Isolate the Response block for clean terminal layout
    if "### Response:\n" in raw_response:
        response_body = raw_response.split("### Response:\n")[-1]
    else:
        response_body = raw_response
        
    print(f"Model: {response_body.strip()}")