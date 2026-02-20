import math
import os
import time

#os.environ['CUDA_LAUNCH_BLOCKING'] = "0" 
import torch
import torch.nn as nn
import torch.nn.functional as F
import modules
import quantizer
import utils
from cut_cross_entropy import linear_cross_entropy

torch._inductor.config.disable_cpp_codegen = True
#dataset = "shakespeare_char"
dataset = "openwebtext"
data_dir = os.path.join('data', dataset)
batch_size = 64
block_size = 256

#you'd think, you really would, but you would be wrong.
DIM_L1 = 64
DIM_L2 = DIM_L1
DIM_L3 = DIM_L2

K_GATHER_L1 = block_size // 2
K_GATHER_L2 = K_GATHER_L1 // 2

mem_size = 48

dropout = 0.0
max_iters = 50000

eval_interval = 1000
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_type = 'cuda'
skiptok = 1
warmup_iters = 200
lr_decay_iters = 30000
decay_iters = 5000
min_lr = 1e-4

idk_enabled = False
cl          = False
mem         = False
gradlog     = False
gen         = True
memcheat    = False
wnorm       = False
genloss     = True
resid       = True
ael         = False
cce         = True
winit       = True
emb_wn      = False
pattlin     = False
wnll        = True
#todo, wnorm, midreset, lr on memory, memswizzle,
#ablate hierarchy, add innerblocks, make hierarchical output generation

#Linear = modules.OrthogonalLinear
Linear = nn.Linear 
#Linear = modules.HungryLinear

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

wn = torch.nn.utils.parametrizations.weight_norm




class Patcher(nn.Module):
    def __init__(self, outer_dim, inner_dim, outer_seq, inner_seq):
        super().__init__()
        self.inner_seq = inner_seq
        self.outer_seq = outer_seq
        self.sdim = 10

        self.up_scan = modules.Block(outer_dim,4,0.0,False)
        self.up_proj = Linear(outer_dim, inner_dim, bias=False)
        if pattlin:
            self.up_proj = modules.PattLayer(outer_dim, inner_dim, 4096)
        #self.up_proj = modules.Pattention(outer_dim,inner_dim,256)
        self.AE = quantizer.Autoencoder(
            inner_dim,inner_dim,inner_dim,inner_dim,
            quantizer.ThresHot
            )
        
        self.resid_up = Linear(outer_dim,inner_dim,bias= False)
        if pattlin:
            self.resid_up = modules.PattLayer(outer_dim, inner_dim, 4096)
        self.resid_drop = nn.Dropout(dropout)
        self.down_drop = nn.Dropout(dropout)
        self.output_pos_emb = nn.Embedding(outer_seq, inner_dim)
        
        self.down_block = modules.Block(inner_dim, 4, 0.0,False)
        self.down_scatter = modules.CombineBlock(inner_dim, 4, 0.0,False)
        self.down_scatter2 = modules.CombineBlock(outer_dim, 4, 0.0,False)
        self.down_proj = Linear(inner_dim,outer_dim,bias=False)
        if pattlin:
            self.down_proj = modules.PattLayer(inner_dim, outer_dim, 256)
        #self.down_proj = modules.Pattention(inner_dim,outer_dim,256)
        self.down_scan = modules.Block(outer_dim,4,0.0,False)
        
        self.up_gate = nn.Sequential(
            nn.Linear(outer_dim, outer_dim // 4,bias=False),
            nn.GELU(),
            nn.Linear(outer_dim // 4, 1)
        )

        self.down_gate = nn.Sequential(
            nn.Linear(outer_dim, outer_dim // 4,bias=False),
            nn.GELU(),
            nn.Linear(outer_dim // 4, outer_seq)
        )

        self.res_norm = modules.LayerNorm(outer_dim, False)
        self.down_norm = modules.LayerNorm(inner_dim, False)
        self.gate_norm = modules.LayerNorm(outer_seq, False)

        self.up_norm = modules.LayerNorm(outer_dim, False)
        self.upgate_norm = modules.LayerNorm(outer_seq, False)
        self.gather_norm = modules.LayerNorm(inner_dim,bias=False)
        
        
        self.query_block = modules.Block(outer_dim, 4, 0.0, False)
        self.gather_block = modules.CombineBlock(outer_dim, 4, 0.0,False)
        self.proj_block = modules.Block(inner_dim, 4, 0.0,False)
        
        #if not resid:
        #    torch.nn.init.normal_(self.down_proj.weight, std=0.02) 
            #torch.nn.init.normal_(self.down_proj.weight, std=0.02) 
    
    def abstract_up(self, x, conf=None):
        gathered = self.query_block(x)
        gate = self.up_gate(gathered).squeeze(-1)
        gate = self.gate_norm(gate)
        gathered = utils.rms(gathered)
        gathered, _ = quantizer.RLMAXgbg3.apply(
            gate, 
            gathered, 
            self.inner_seq,
            1.0,
            #False,
            self.training,
            #0.3
            )
        #noise =torch.randn_like(gathered,requires_grad=True)/(1.0+gathered.sum()) 
        #gsink = 
        #gathered= gathered+noise
        #gathered= utils.rms(gathered)
        #gathered = quantizer.HungryGatherFunc2.apply(
        #    gate, 
        #    gathered, 
        #    self.inner_seq,
        #    0.01,   # tau
        #    self.training,  # training
        #    0.0,   #magnet
        #    )
        #self.gln
        #gathered= torch.tanh(gathered)
        up = self.up_proj(gathered)
        #up= utils.rms(up)
        up = self.proj_block(up)
        hot= None
        aux= 0.0
        if ael:
            ae_up, hot = self.AE(up)
            aux = F.mse_loss(up, ae_up)
            up = ae_up
        return up, [x, hot, gate, aux]
    
    def abstract_down(self, x, residual, decaying =  None, conf=None):
        residual, hot, gate, upaux = residual
        aux=0
        x = self.down_norm(x)
        if ael:
            aux = F.binary_cross_entropy(self.AE.quant(x)[:,:-1,:], hot.detach()[:,1:,:])#ablate
        
        pos = torch.arange(0, self.outer_seq, dtype=torch.long, device=device) 
       
        scattered = quantizer.ScatterByGate.apply(gate, x, self.outer_seq)
        pos_emb = self.output_pos_emb(pos).expand(scattered.shape)
        scattered = scattered + pos_emb 
        pds = self.down_block(scattered)
        pds = self.down_proj(pds)
            
        if resid:
            query = self.res_norm(residual+pos_emb)
        else:
            query = self.down_proj(pos_emb)
            query = self.res_norm(query)
            
        xout = self.down_scatter2(pds, query, causal=True)
        if resid:
            query = self.res_norm(residual)
        else:
            query = self.down_proj(pos_emb)
            query = self.res_norm(query)
        xout = self.down_scan(xout)
        aux = aux+upaux
        return xout, aux
    
    def forward(self, x, center, conf=None):
        p_up, residual = self.abstract_up(x)
        passed, aux = center(p_up)
        p_down, aux2 = self.abstract_down(passed, residual)
        return p_down, aux+aux2
    

class HNet(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, DIM_L1)
        if emb_wn:
            torch.nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
 
            self.tok_emb = wn(self.tok_emb, dim=1)
        
        self.pos_emb = nn.Embedding(block_size, DIM_L1)
        #self.head = nn.Linear(DIM_L1, vocab_size,bias=False)
        #self.head.weight = self.tok_emb.weight
        #self.head = wn(self.head)
        self.wnll = utils.wnormlearnloss(DIM_L1)
        self.n_layer = 2
        self.h_ln = modules.LayerNorm(DIM_L1,bias=False)
        self.sLayers = nn.ModuleList([
            Patcher(DIM_L1,DIM_L2,block_size,K_GATHER_L1),
            Patcher(DIM_L2,DIM_L3,K_GATHER_L1,K_GATHER_L2),
        ])
        
        self.inner = nn.ModuleList([
            modules.Block(DIM_L3,4,0.0,False) for _ in range(4)
        ])
        
        if mem:
            self.mem_block_size=mem_size
            memconf = {
                'mem_block_size': mem_size,
                'dropout': 0.0,
                'n_embd': DIM_L3,
                'n_head': 4,
                'bias': False,
            }
            self.register_buffer('memory', torch.randn(1, self.mem_block_size, DIM_L3)*1e-5)
            self.memory_frozen = False
            self.cmem= True
            self.dreamer = modules.Dreamer(**memconf)
            self.memory_selector = modules.Block(**memconf)
        if winit:
            self.apply(self._init_weights)
   #
    def _init_weights(self, module):
        #if isinstance(module, Linear):
        #    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        #    if module.bias is not None:
        #        torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def mem_form(self, x):
        mem         = self.memory_selector(x, causal=self.cmem)[:,:self.mem_block_size,:] #(b, m, c)
        #check_nan(mem, "selector")
        return self.dreamer(mem)
    
    def mem_update(self, update, malpha=1.0):
        if self.training and not self.memory_frozen:
            with torch.no_grad():
                self.memory = (self.memory+update*malpha)
    

    def forward(self, idx, targets=None, decaying = None, conf=None):
        device = idx.device
        b, t = idx.shape
        
        positions = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(positions)
        x0 = tok_emb + pos_emb 
        x0 = utils.rms(x0)
        x = x0
        residuals = []
        for i, layer in enumerate(self.sLayers):
            
            x, res = layer.abstract_up(x)
            residuals.append(res)

        if mem:
            x = self.mem_center(x)
        else:
            for block in self.inner:
                x = block(x)
        aux=0
        for i in reversed(range(len(self.sLayers))):
            layer = self.sLayers[i]
            x, auxl = layer.abstract_down(x, residuals[i], decaying = decaying)
            aux = aux+auxl
        wnl = 0
        if self.training:
            wnl = self.wnll(x)
            aux = aux + wnl.mean()
        x = self.h_ln(x)
        if targets is None: #during inference skip the losses
            
            logits = F.linear(x[:, -skiptok:, :], self.tok_emb.weight)
            return logits
        
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
        
        
        gloss = 0.0
        
        logits = None
        red = 'mean'
        if wnll:
            red='none'
        if cce:
            T_out = x.shape[1]
            targets_aligned = targets[:, :T_out]

            loss = linear_cross_entropy(x.to(torch.bfloat16), 
                                            self.tok_emb.weight.to(torch.bfloat16), 
                                            targets_aligned, 
                                            impl='torch_compile',
                                            reduction=red)
            if genloss:
                gloss = linear_cross_entropy(x.to(torch.bfloat16)[:,-1,:], 
                                            self.tok_emb.weight.to(torch.bfloat16), 
                                            targets_aligned[:,-1], 
                                            impl='torch_compile')
        else:
            logits = F.linear(x, self.tok_emb.weight)
            
            B, T_out, C = logits.shape
            targets_aligned = targets[:, :T_out]
        
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                    targets_aligned.reshape(-1),
                                    reduction=red)
            if genloss:
                gloss = F.cross_entropy(logits[:,-1,:].reshape(-1, C), targets_aligned[:,-1].reshape(-1)).detach()
        
        if self.training:
            if wnll:
                chaos = ((1.0-wnl.detach())/2.0).mean()*0.99
                scale = chaos
                loss = loss / (1+ wnl.detach().squeeze(-1)*scale)
                
            if idk_enabled:
                loss = utils.idk_loss(loss, logits, targets, idk_token) 
            
            loss = loss+aux
            if cl:
                loss = loss+total_cl
        
        if wnll:
            loss= loss.mean()
        return logits, loss, gloss 

    def mem_center(self, x):
        cb,ct,cc = x.shape
        x1 = x
        x = torch.cat([self.memory.expand(cb,-1,-1), x],dim=1) #probably needs a batch swizzle to pervent cheating
        for block in self.inner:
            x = block(x)
        
        if not self.training:
            return x[:,self.mem_block_size:,:].contiguous()
        
        malpha = 1.0
        if memcheat:
            mem2 = ((self.memory+self.mem_form(x)*malpha)).expand(cb,-1,-1)
        else:
            mh1 = ((self.memory+self.mem_form(x[:cb//2,...])*malpha)).expand(cb//2,-1,-1)
            mh2 = ((self.memory+self.mem_form(x[cb//2:,...])*malpha)).expand(cb//2,-1,-1)
            mem2 = torch.cat([mh2,mh1],dim=0)
        x = torch.cat([mem2, x1],dim=1) #try what memorization enhances the score
        for block in self.inner:
            x = block(x)
        
        self.mem_update(self.mem_form(x),malpha)
        x = x[:,self.mem_block_size:,:].contiguous()
        return x   
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
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

class cascadenet(nn.Module):
    def __init__(self, vocab_size, n_embd = 256, n_iter = 4, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.n_iter = n_iter

        # Embedding layers
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.mlp_init = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.GELU()
        )
        self.mlp_update = nn.Sequential(
            Linear(n_embd*2, 4 * n_embd,bias=False),
            nn.GELU(),
            Linear(4 * n_embd, n_embd,bias=False),
            nn.Dropout(dropout),
        )
        # Layer norms for stability
        self.ln_in = nn.LayerNorm(n_embd)
        self.ln_out = nn.LayerNorm(n_embd)

        # Final output layer
        self.lm_head = Linear(n_embd, vocab_size,bias=False)

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
conf={}
for iter in range(max_iters):
    lr = get_lr(iter)
    conf['lr'] = 5e-5
    decaying = 1.0 - (iter / decay_iters)
    decaying = torch.tensor(decaying, device=device, dtype=torch.float32)
    decaying = torch.clamp(decaying, 0.0, 1.0)
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
                _, loss, gloss = model(X_val, Y_val, decaying= decaying)
            losses[split] = loss.detach().item()
            if genloss and split == 'val':
                glosses[split] = gloss
        optimizer.zero_grad(set_to_none=True)
        t1 = time.time()
        print(f"step {iter:{len(str(max_iters))}d}: ",end="")
        print(f"train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}, ",end="")
        if genloss:
            print(f"gen loss: {glosses['val']:.4f}, ",end="")
        print(f"time: {(t2-t0)*1000:.0f} ms, Tval: {(t1-t2)*1000:.0f} ms")
        model.train()
        t0 = time.time()
    X, Y = get_batch('train')
    logits, loss, gloss = model(X, Y, decaying= decaying,conf= conf)
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
    
    print("sanity check")
    X, Y = get_batch('train')
    with torch.no_grad():
        model.eval()
        outp = model.generate(X[0].unsqueeze(0), 100, temperature=1.0, top_k=5)
        model.train()
    print(decode(outp[0,-100:].detach().cpu().numpy().tolist()))
    print("Training finished.")