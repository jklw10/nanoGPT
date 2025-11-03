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
import utils

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
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_type = 'cuda'
skiptok = 1
warmup_iters = 200
lr_decay_iters = 15000
min_lr = 1e-4

idk_enabled = False
cl          = False
mem         = False
gradlog     = False
gen         = True
memcheat    = False
wnorm       = False
genloss     = False
#todo, wnorm, midreset, lr on memory, memswizzle,
#ablate hierarchy, add innerblocks, make hierarchical output generation

def get_lr(it):
    
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    if it > lr_decay_iters:
        return min_lr
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
    return min_lr + coeff * (learning_rate - min_lr)

if not os.path.exists(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(os.path.join(data_dir, 'input.txt'), 'w', encoding='utf-8') as f: 
        f.write(requests.get(url).text)
    with open(os.path.join(data_dir, 'input.txt'), 'r') as f: 
        data = f.read()
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    n = len(data)
    train_data = data[:int(n*0.9)] 
    val_data = data[int(n*0.9):]
    train_ids = np.array([stoi[c] for c in train_data], dtype=np.uint16)
    val_ids = np.array([stoi[c] for c in val_data], dtype=np.uint16)
    train_ids.tofile(os.path.join(data_dir, 'train.bin'))
    val_ids.tofile(os.path.join(data_dir, 'val.bin'))
    meta = {'vocab_size': vocab_size, 'itos': itos, 'stoi': stoi}
    with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
meta_path = os.path.join(data_dir, 'meta.pkl')
with open(meta_path, 'rb') as f: 
    meta = pickle.load(f)
vocab_size = meta['vocab_size']
if idk_enabled:
    idk_token = vocab_size
    vocab_size += 1

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

class Patcher(nn.Module):
    def __init__(self, outer_dim, inner_dim, outer_seq, inner_seq):
        super().__init__()
        self.inner_seq = inner_seq
        self.outer_seq = outer_seq
        self.sdim = 10

        self.up_scan = modules.Block(outer_dim,4,0.0,False)
        self.up_proj = nn.Linear(outer_dim,inner_dim)
        self.AE = quantizer.Autoencoder(inner_dim,inner_dim,inner_dim,128,quantizer.ThresHot)
        self.resid_up = nn.Linear(outer_dim, inner_dim)
        
        self.output_pos_emb = nn.Linear(outer_seq, inner_dim)
        
        self.down_scatter = modules.CombineBlock(inner_dim, 4, 0.0,False)
        self.down_scatter2 = modules.CombineBlock(outer_dim, 4, 0.0,False)
        self.down_proj = nn.Linear(inner_dim,outer_dim)
        self.down_scan = modules.Block(outer_dim,4,0.0,False)
        
        #self.up_gate = nn.Linear(self.sdim*outer_seq, outer_seq)
        self.up_gate = nn.Sequential(
            nn.Linear(outer_dim, outer_dim // 4),
            nn.ReLU(),
            nn.Linear(outer_dim // 4, 1)
        )

        #self.down_gate = nn.Linear(self.sdim*inner_seq, outer_seq)
        self.down_gate = nn.Sequential(
            nn.Linear(outer_dim, outer_dim // 4),
            nn.ReLU(),
            nn.Linear(outer_dim // 4, outer_seq)
        )


        self.down_norm = modules.LayerNorm(inner_dim, False)
        self.gate_norm = modules.LayerNorm(outer_seq, False)
        self.down_scan = modules.Block(outer_dim, 4, 0.0,False)

        self.up_norm = modules.LayerNorm(outer_dim, False)
        self.upgate_norm = modules.LayerNorm(outer_seq, False)
        self.up_drain = nn.Parameter(torch.ones(outer_dim))

        
        #self.query_block = modules.SSMBlock(outer_dim, 0.0, False)
        self.query_block = modules.Block(outer_dim, 4, 0.0, False)
        self.gather_block = modules.CombineBlock(outer_dim, 4, 0.0,False)
    
    def abstract_up(self, x):
        scan = self.up_scan(x, causal = False)
        ##scan = F.softmax(scan, dim=-1)
        scan = self.up_norm(scan)
        gate = self.up_gate(scan).squeeze(-1)
        ##gate = F.softmax(gate, dim=-1)
        #gate = self.up_gate(x).squeeze(-1)
        #
        ##losses = F.mse_loss(x[:,1:,:],scan[:,:-1,:],reduction="none")
        ##gate = losses.sum(dim=-1)
        ##gate = F.pad(gate, (1, 0), "constant", 0)#.detach()
        #
        #gathered = quantizer.GatherByGate.apply(gate, scan, self.inner_seq)
        #compressed sensing says that random samples should suffice?
        gathered = self.query_block(x, causal = True)
        #gathered = gathered[:,-self.inner_seq:,:]
        _,idx = torch.topk(gate,self.inner_seq)
        #idx = utils.bluenoise_indices(x.shape[0], self.outer_seq, self.inner_seq,device=x.device)
    
        eidx = idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        gathered = torch.gather(gathered,1,eidx)
        #gathered = quantizer.GatherByGate.apply(gate, gathered, self.inner_seq)
        #gathered = self.gather_block(x, gathered, causal = True)
        up = self.up_proj(gathered)
        ae_up, hot = self.AE(up)
        aux = F.mse_loss(up, ae_up)
        return up, [x,hot, idx, aux]
    
    def abstract_down(self, x, residual):
        #sg = self.down_gate(x[...,:self.sdim].reshape(x.shape[0], self.inner_seq*self.sdim))
        residual, hot, idx, upaux = residual
        #aux = F.binary_cross_entropy(self.AE.quant(x)[:,:-1,:], hot.detach()[:,1:,:])#ablate
        aux=0
        pgd = p_down = self.down_proj(x)
        pgd = self.down_scan(pgd)[:,-1,:]
        sg = self.down_gate(pgd)
        #sg = F.softmax(sg, dim=-1) 
        sg = self.gate_norm(sg)
        
        pos = torch.arange(0, self.outer_seq, dtype=torch.long, device=device) 
        phot = F.one_hot(pos, self.outer_seq).to(x.dtype).detach()
        pos_emb = self.output_pos_emb(phot).expand(x.shape[0],-1,-1)
        #query = scattered + pos_emb 
        query = pos_emb 
        #query = query + self.resid_up(residual)
        
        scattered = torch.zeros(x.shape[0], self.outer_seq, x.shape[2], dtype=x.dtype, device=x.device)
        
        eidx = idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        scattered.scatter_(1, eidx, x)
        #scattered = torch.scatter(1,x,idx)
        #scattered = quantizer.ScatterByGate.apply(gate, x, self.outer_seq)# technically causal leak
        pds = self.down_scatter(scattered, query, causal=True)#maybe detach.
        pds = self.down_proj(pds)
        #print(query.shape)
        query = self.down_proj(query)+residual#ablate
        p_down = self.down_scatter2(pds, query, causal=True)#
        #pds = self.down_scatter(x, pos_emb)# self.resid_up(residual))
        #p_down=residual+p_down
        aux = aux+upaux
        return self.down_scan(p_down), aux
    
    def forward(self, x, center):
        p_up, residual = self.abstract_up(x)
        passed, aux = center(p_up)
        p_down, aux2 = self.abstract_down(passed, residual)
        return p_down, aux+aux2
    

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
        
        if mem:
            memconf = {
                'mem_block_size': 48,
                'dropout': 0.0,
                'n_embd': DIM_L3,
                'n_head': 4,
                'bias': False,
            }
            self.mem_block_size=memconf['mem_block_size']
            self.register_buffer('memory', torch.randn(1, self.mem_block_size, DIM_L3)*1e-5)
            self.memory_frozen = False
            self.cmem= True
            self.dreamer = modules.Dreamer(causal=self.cmem, **memconf)
            self.memory_selector = modules.Block(**memconf)

    def mem_form(self, x):
        mem         = self.memory_selector(x, causal=self.cmem)[:,:self.mem_block_size,:] #(b, m, c)
        #check_nan(mem, "selector")
        return self.dreamer(mem)
    
    def mem_update(self, update, malpha=1.0):
        if self.training and not self.memory_frozen:
            with torch.no_grad():
                self.memory = (self.memory+update*malpha)

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
        if mem:
            cb,ct,cc = x.shape
            x1 = x
            x = torch.cat([self.memory.expand(cb,-1,-1), x],dim=1) #probably needs a batch swizzle to pervent cheating
            x = self.innerb(x)
            x = self.innerb2(x)
            malpha = 1.0
            if memcheat:
                mem2 = ((self.memory+self.mem_form(x)*malpha)).expand(cb,-1,-1)
            else:
                mh1 = ((self.memory+self.mem_form(x[:cb//2,...])*malpha)).expand(cb//2,-1,-1)
                mh2 = ((self.memory+self.mem_form(x[cb//2:,...])*malpha)).expand(cb//2,-1,-1)
                mem2 = torch.cat([mh2,mh1],dim=0)
            x = torch.cat([mem2, x1],dim=1) #try what memorization enhances the score
            x = self.innerb(x)
            x = self.innerb2(x)
            self.mem_update(self.mem_form(x),malpha)
            x = x[:,self.mem_block_size:,:].contiguous()
        else:
            x = self.innerb(x)
            x = self.innerb2(x)
        aux=0
        for i in reversed(range(len(self.sLayers))):
            layer = self.sLayers[i]
            #print(residuals[i][0].shape)
            x, auxl = layer.abstract_down(x, residuals[i])
            aux = aux+auxl
        
        if cl:
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
        gloss = None
        if targets is not None:
            logits = self.head(x)
            B, T_out, C = logits.shape
            targets_aligned = targets[:, :T_out]
            
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets_aligned.reshape(-1))
            gloss = 0.0
            #gloss = F.cross_entropy(logits[:,:-1,:].reshape(-1, C), targets_aligned[:,:-1,:].reshape(-1))
            if self.training:
                
                if idk_enabled:
                    loss = utils.idk_loss(loss, logits, targets, idk_token) 
                
                
                loss = loss+aux
                if cl:
                    loss = loss+total_cl
        else:
            logits = self.head(x[:, -skiptok:, :])


        return logits, loss, gloss    
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    
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
                if idk_enabled:
                    logits[:,:,idk_token] = -float('Inf')
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
    def configure_optimizers(self, lr):
        return torch.optim.AdamW(self.parameters(), lr=lr)

class FeedForward(nn.Module):
    def __init__(self, n_in, n_out, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 4 * n_out),
            nn.GELU(),
            nn.Linear(4 * n_out, n_out),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class cascadenet(nn.Module):
    def __init__(self, vocab_size, n_embd = 256, n_iter = 4, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.n_iter = n_iter

        # Embedding layers
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # Initial state projector: x_t -> h_{t,0}
        self.mlp_init = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.GELU()
        )

        # The core update MLP: (h_{t-1, k-1}, x_t) -> update_t
        # It takes the concatenation of the previous state and current input
        self.mlp_update = FeedForward(n_embd * 2, n_embd, dropout)

        # Layer norms for stability
        self.ln_in = nn.LayerNorm(n_embd)
        self.ln_out = nn.LayerNorm(n_embd)

        # Final output layer
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device

        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = self.ln_in(tok_emb + pos_emb) # (B, T, C)

        h = self.mlp_init(x) # (B, T, C)
        for _ in range(self.n_iter):
            h_prev_padded = F.pad(h, (0, 0, 1, 0))[:, :-1, :] # (B, T, C)

            mlp_input = torch.cat([h_prev_padded, x], dim=-1) # (B, T, 2*C)

            update = self.mlp_update(mlp_input) # (B, T, C)

            h = h + update

        h = self.ln_out(h)
        logits = self.lm_head(h) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B*T, C)
            targets_flat = targets.view(B*T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.train()
        return idx
    
    def configure_optimizers(self, lr):
        return torch.optim.AdamW(self.parameters(), lr=lr)
    
model = HNet(vocab_size=vocab_size)
model.to(device)
optimizer = model.configure_optimizers(lr=learning_rate)
torch.set_float32_matmul_precision('medium')
model.compile()
print(f"Starting training on {device}")
print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")


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
        glosses = {}
        for split in ['train', 'val']:
            X_val, Y_val = get_batch(split)
            with torch.no_grad():
                _, loss, gloss = model(X_val, Y_val)
            losses[split] = loss.detach().item()
            if genloss and split == 'val':
                glosses[split] = gloss.detach().item()
        optimizer.zero_grad(set_to_none=True)
        t1 = time.time()
        if genloss:
            print(f"gen loss{glosses['val']:.4f}",end="")
        print(f"step {iter}: train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}, time: {(t2-t0)*1000:.0f} ms, Tval: {(t1-t2)*1000:.0f} ms ")
        model.train()
        t0 = time.time()
    X, Y = get_batch('train')
    logits, loss, gloss = model(X, Y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    
    if gradlog and iter > 300:
        sorted_grads = sorted(gradient_norms.items(), key=lambda item: item[1], reverse=True)
        for i, (name, norm) in enumerate(sorted_grads):
            print(f"{i+1}. {name:<60} | Norm: {norm:.4f}")
    #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    if wnorm:
        with torch.no_grad():
            utils.softwnorm(model,1e-4)
    
if gen:
    meta_path = os.path.join('data', 'shakespeare_char/meta.pkl')

    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    def encode(s): 
        return [stoi[c] for c in s]
    def decode(ll):
        decoded = []
        for i in ll:
            if i in itos:
                decoded.append(itos[i])
            elif i == idk_token:
                decoded.append('[IDK]')
            else:
                decoded.append(f'[UNK{i}]')  # Or another placeholder for unknown tokens
        return ''.join(decoded)

    print("sanity check")
    X, Y = get_batch('train')
    with torch.no_grad():
        model.eval()
        outp = model.generate(X[0].unsqueeze(0), 100, temperature=1.0, top_k=5)
        model.train()
    print(decode(outp[0,-100:].detach().cpu().numpy().tolist()))
    print("Training finished.")