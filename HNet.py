import math
import os
import time

#os.environ['CUDA_LAUNCH_BLOCKING'] = "0" 
from cut_cross_entropy import linear_cross_entropy
import requests
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import modules
import quantizer

# --- Configuration (remains the same) ---
dataset = "shakespeare_char"
data_dir = os.path.join('data', dataset)
batch_size = 64
block_size = 256

DIM_L1 = 64
DIM_L2 = 128
DIM_L3 = 256

K_GATHER_L1 = 64
K_GATHER_L2 = 16

max_iters = 30000
eval_interval = 1000
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_type = 'cuda'
skiptok = 1
warmup_iters = 200
lr_decay_iters = 15000
min_lr = 1e-5
# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# --- Data Loading (remains the same) ---
if not os.path.exists(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(os.path.join(data_dir, 'input.txt'), 'w', encoding='utf-8') as f: f.write(requests.get(url).text)
    with open(os.path.join(data_dir, 'input.txt'), 'r') as f: data = f.read()
    chars = sorted(list(set(data))); vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }; itos = { i:ch for i,ch in enumerate(chars) }
    n = len(data); train_data = data[:int(n*0.9)]; val_data = data[int(n*0.9):]
    train_ids = np.array([stoi[c] for c in train_data], dtype=np.uint16); val_ids = np.array([stoi[c] for c in val_data], dtype=np.uint16)
    train_ids.tofile(os.path.join(data_dir, 'train.bin')); val_ids.tofile(os.path.join(data_dir, 'val.bin'))
    meta = {'vocab_size': vocab_size, 'itos': itos, 'stoi': stoi}
    with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f: pickle.dump(meta, f)
meta_path = os.path.join(data_dir, 'meta.pkl')
with open(meta_path, 'rb') as f: meta = pickle.load(f)
vocab_size = meta['vocab_size']

def get_batch(split):
    data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size-skiptok, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+skiptok:i+skiptok+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda': 
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else: 
        x, y = x.to(device), y.to(device)
    return x, y

class sAttention(nn.Module):
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
        
    def forward(self, x, sx):
        B, T, C = x.size() 
        sB, sT, sC = sx.size() 
        q = self.q_proj(sx)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(B, sT, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, sT, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)

        att = torch.matmul(q, k.transpose(-2, -1)) 
        #causality isn't really necessary as it's a compressed fmt
        #and diffusion will be pretty much forced to be the real generation method.
        #y = v @ F.softmax(att,dim=-1) 
        y = F.softmax(att,dim=-1) @ v
        y = y.transpose(1, 2).contiguous().view(B, sT, C) 
        y = self.resid_dropout(self.c_proj(y))

        return y

    
class sBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout, bias):
        super().__init__()

        self.attn = sAttention(n_embd, n_head, dropout, bias)

        self.ln_1 = modules.LayerNorm(n_embd, bias)
        self.ln_2 = modules.LayerNorm(n_embd, bias)
        self.ln_3 = modules.LayerNorm(n_embd, bias)
        self.mlp = modules.MLP(n_embd, dropout, bias)

    def forward(self, x, sx):
        a = self.attn(self.ln_1(x),self.ln_2(sx))
        sx = sx + a
        sx = sx + self.mlp(self.ln_3(sx))
        return sx

class Patcher(nn.Module):
    def __init__(self, outer_dim, inner_dim, outer_seq, inner_seq):
        super().__init__()
        self.inner_seq = inner_seq
        self.outer_seq = outer_seq
        self.sdim = 10

        self.up_scan = modules.Block(outer_dim,4,0.0,False)
        self.up_proj = nn.Linear(outer_dim,inner_dim)
        
        self.output_pos_emb = nn.Linear(outer_seq, inner_dim)
        self.down_scatter = sBlock(inner_dim, 4, 0.0,False)
        self.down_proj = nn.Linear(inner_dim,outer_dim)
        self.down_scan = modules.Block(outer_dim,4,0.0,False)
        
        #self.up_gate = nn.Linear(self.sdim*outer_seq, outer_seq)
        self.up_gate = nn.Sequential(
            nn.Linear(outer_dim, outer_dim // 4),
            nn.ReLU(),
            nn.Linear(outer_dim // 4, 1)
        )

        self.down_gate = nn.Linear(self.sdim*inner_seq, outer_seq)
        
        self.down_norm = modules.LayerNorm(inner_dim, False)

        self.up_norm = modules.LayerNorm(outer_dim, False)
        self.up_norm2 = modules.LayerNorm(outer_dim, False)
        self.up_drain = nn.Parameter(torch.ones(outer_dim))
    
    def abstract_up(self, x):
        scan = self.up_scan(x)
        gate = self.up_gate(scan).squeeze(-1)
        #losses = F.mse_loss(x[:,1:,:],scan[:,:-1,:],reduction="none")
        #gate = losses.sum(dim=-1)
        #gate = F.pad(gate, (1, 0), "constant", 0)#.detach()
        gathered = quantizer.GatherByGate.apply(F.softmax(gate, dim=-1), F.softmax(scan, dim=-1), self.inner_seq)
        
        return self.up_proj(gathered), x
    def abstract_down(self, x, residual):
        sg = self.down_gate(x[...,:self.sdim].reshape(x.shape[0], self.inner_seq*self.sdim))
        sg = F.softmax(sg, dim=-1)
        scattered = quantizer.ScatterByGate.apply(sg, x, self.outer_seq)
        
        pos = torch.arange(0, self.outer_seq, dtype=torch.long, device=device) 
        phot = F.one_hot(pos, self.outer_seq).to(x.dtype)
        
        scattered = scattered + self.output_pos_emb(phot.detach())
        pds = self.down_scatter(x, scattered)
        p_down = self.down_proj(pds)+residual

        return self.down_scan(p_down)
    
    def forward(self, x, center):
        p_up, residual = self.abstract_up(x)
        passed = center(p_up)
        p_down = self.abstract_down(passed, residual)
        return p_down
        


class HNet(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedder = nn.Embedding(vocab_size, DIM_L1)
        self.pos_embedder = nn.Embedding(block_size, DIM_L1)
        self.head = nn.Linear(DIM_L1, vocab_size)
        self.n_layer = 2
        self.sLayers = nn.ModuleList([
            Patcher(DIM_L1,DIM_L2,block_size,K_GATHER_L1),
            Patcher(DIM_L2,DIM_L3,K_GATHER_L1,K_GATHER_L2),
        ])
        self.innerb = modules.Block(DIM_L3,4,0.0,False)
        self.innerb2 = modules.Block(DIM_L3,4,0.0,False)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.shape
        
        positions = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.token_embedder(idx)
        pos_emb = self.pos_embedder(positions)
        x0 = tok_emb + pos_emb 
        x = x0
        residuals = []
        for i, layer in enumerate(self.sLayers):
            
            x, res = layer.abstract_up(x)
            residuals.append(res)
        
        x = self.innerb(x)
        x = self.innerb2(x)

        for i in reversed(range(len(self.sLayers))):
            layer = self.sLayers[i]
            x = layer.abstract_down(x, residuals[i])
        
        total_cl = 0.0
        
        current_input_rep = x0[:, skiptok:, :]
        current_output_rep = x[:, :-skiptok, :]

        for i, layer in enumerate(self.sLayers):
            encoded_input, _  = layer.abstract_up(current_input_rep)
            encoded_output, _ = layer.abstract_up(current_output_rep)
            
            total_cl = total_cl + F.mse_loss(encoded_input, encoded_output)
            
            current_input_rep = encoded_input
            current_output_rep = encoded_output
        
        loss = None
        if targets is not None:
            logits = self.head(x)
            B, T_out, C = logits.shape
            targets_aligned = targets[:, :T_out]
            loss = F.cross_entropy(logits.reshape(-1, C), targets_aligned.reshape(-1))
            if self.training:
                loss = loss+total_cl#+aux
        else:
            logits = self.head(x[:, -skiptok:, :])


        return logits, loss       

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        #SOON :tm: :D
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -skiptok:, :] / temperature
            #         record for super pos
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, :, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            B, T, C = probs.shape
            probs_2d = probs.view(B * T, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs_2d, num_samples=1)
            idx_next = idx_next.view(B, T)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

# --- Training Loop ---
model = HNet(vocab_size=vocab_size)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
torch.set_float32_matmul_precision('medium')
model.compile()
print(f"Starting training on {device}")
print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
print("Using Cross-Attention DePatcher for flexible sequence reconstruction.")
gradlog = False
if gradlog:
    gradient_norms = {}

    def get_grad_hook(name):
        def hook(grad):
            gradient_norms[name] = grad.norm(2).item()
        return hook
    for name, p in model.named_parameters():
        if p.requires_grad:
            p.register_hook(get_grad_hook(name))

            
t0 = time.time()
t2 = time.time()
for iter in range(max_iters):
    lr = get_lr(iter)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if iter % eval_interval == 0 or iter == max_iters - 1:
        t2 = time.time()
        model.eval()
        losses = {}
        for split in ['train', 'val']:
            X_val, Y_val = get_batch(split)
            with torch.no_grad():
                _, loss = model(X_val, Y_val)
            losses[split] = loss.detach().item()
        optimizer.zero_grad(set_to_none=True)
        t1 = time.time()
        print(f"step {iter}: train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}, time: {(t2-t0)*1000:.0f} ms, Tval: {(t1-t2)*1000:.0f} ms ")
        model.train()
        t0 = time.time()
    X, Y = get_batch('train')
    logits, loss = model(X, Y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    
    if gradlog and iter > 300:
        sorted_grads = sorted(gradient_norms.items(), key=lambda item: item[1], reverse=True)
        for i, (name, norm) in enumerate(sorted_grads):
            print(f"{i+1}. {name:<60} | Norm: {norm:.4f}")
    #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
gen = True
if gen:
    meta_path = os.path.join('data', 'shakespeare_char/meta.pkl')

    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    def encode(s): [stoi[c] for c in s]
    def decode(ll):
        decoded = []
        for i in ll:
            if i in itos:
                decoded.append(itos[i])
            else:
                decoded.append('[UNK]')  # Or another placeholder for unknown tokens
        return ''.join(decoded)

    print("sanity check")
    X, Y = get_batch('train')
    with torch.no_grad():
        model.eval()
        outp = model.generate(X[0].unsqueeze(0), 100, temperature=1.0, top_k=5)
        model.train()
    print(decode(outp[0,-100:].detach().cpu().numpy().tolist()))
    print("Training finished.")