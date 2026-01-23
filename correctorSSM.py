import math
import os
import time

#os.environ['CUDA_LAUNCH_BLOCKING'] = "0" 
from cut_cross_entropy import linear_cross_entropy

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import modules
import quantizer
import utils

dataset = "openwebtext"
data_dir = os.path.join('data', dataset)
batch_size = 64
block_size = 64
n_embed = 228
n_head = 6
n_layer = 4

dropout = 0.1
max_iters = 30000
eval_interval = 1000
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_type = 'cuda'
warmup_iters = 200
lr_decay_iters = 15000
min_lr = learning_rate/10


torch._inductor.config.disable_cpp_codegen = True

idk_enabled = False
cl          = False
mem         = False
gradlog     = False
gen         = True
memcheat    = False
wnorm       = False
genloss     = True

qkhwn       = True
#todo, wnorm, midreset, lr on memory, memswizzle,
#ablate hierarchy, add innerblocks, make hierarchical output generation

wn = torch.nn.utils.parametrizations.weight_norm

def get_lr(it):
    
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    if it > lr_decay_iters:
        return min_lr
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
    return min_lr + coeff * (learning_rate - min_lr)

dataset = 'openwebtext'
always_byte=True
decode, vocab_size = utils.get_meta(dataset,always_byte)
get_batch = utils.get_get_batch(block_size,batch_size,dataset,device,always_byte=always_byte,single=True)
print(f"vocab: {vocab_size}")


class SSM(nn.Module):
    def __init__(self, n_embd, dropout, bias):
        super().__init__()
        self.lanes=4
        #n_embd=n_embd*self.lanes
        self.q_attn =  nn.Linear(n_embd, n_embd, bias=bias)
        self.v_attn =  nn.Linear(n_embd, n_embd, bias=bias)
        
        self.c_proj = nn.Linear(n_embd,  n_embd, bias=bias)

        self.identity = nn.Parameter(torch.rand(n_embd))
        
        self.attn_dropout = nn.Dropout(dropout) #todo
        self.resid_dropout = nn.Dropout(dropout) #todo
        self.n_embd = n_embd
        self.dropout = dropout
        self.up =  nn.Linear(n_embd, n_embd, bias=bias)
        self.scanner = modules.ProcrustesButterflyLinear(n_embd)
    
    def get_weight(self):
        return self.scanner.compute_orthogonal_weights()
    def get_scan(self,weight,dtype):
        def scan(left, right):
            left, right = torch.broadcast_tensors(left, right)
            
            B_curr, T_curr, C_curr = left.shape
            
            head_dim = C_curr // self.lanes

            q = left.view(B_curr, T_curr, self.lanes, head_dim)
            k = right.view(B_curr, T_curr, self.lanes, head_dim)
            
            v = self.scanner(left, right, weight)
            v = v.view(B_curr, T_curr, self.lanes, head_dim)

            z = F.scaled_dot_product_attention(q, k, v)

            denom = (1.0 + torch.arange(0, self.lanes, device=left.device)).view(1, 1, self.lanes, 1)
            
            z = utils.rms(z,dim=-1) / denom
            
            z = z.reshape(B_curr, T_curr, C_curr)
            
            return (left + z).to(dtype)
        return scan
    def forward(self, x: torch.tensor, weight = None): #same signature to allow easy swapping.
        B, T, C = x.size() 
        #q = self.q_attn(x) 
        q = self.up(x) 
        v = self.v_attn(x)
        if weight is None:
            weight = self.scanner.compute_orthogonal_weights()
        scan = self.get_scan(weight,x.dtype)
        q = utils.pscan2(q, scan, self.identity)
        
        Y = q*v 
        Y = self.c_proj(Y)
        Y = self.resid_dropout(Y)
        return Y, q
    
    def nexts(self, prev: torch.tensor, x: torch.tensor,  causal=False, weight = None): #same signature to allow easy swapping.
        B, T, C = x.size() 
        q = self.q_attn(x) 
        v = self.v_attn(x)
        if weight is None:
            weight = self.scanner.compute_orthogonal_weights()
        
        scan = self.get_scan(weight,x.dtype)
        q = scan(prev,q)
        
        Y = q*v 
        Y = self.c_proj(Y)
        Y = self.resid_dropout(Y)
        return Y, q
    
class Block(nn.Module):
    def __init__(self, n_embed, dropout, bias):
        super().__init__()
        self.ln_1 = modules.LayerNorm(n_embed, bias)
        self.attn = SSM(n_embed, dropout, bias)
        self.ln_2 = modules.LayerNorm(n_embed, bias)
        self.mlp  = modules.MLP(n_embed, dropout, bias)
    def get_weight(self):
        return self.attn.get_weight()
    def forward(self, x, prev=None, weights=None):
        if prev is not None:
             attn_output, present_key_values = self.attn.nexts(prev, self.ln_1(x),weight=weights)
        else:
             attn_output, present_key_values = self.attn(self.ln_1(x),weight=weights)
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x, present_key_values

def vicreg_loss(pred, target, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):
    # 1. Invariance (MSE)
    mse_loss = F.mse_loss(pred, target)
    
    std_pred = torch.sqrt(pred.var(dim=0) + 1e-4)
    std_targ = torch.sqrt(target.var(dim=0) + 1e-4)
    std_loss = torch.mean(F.relu(1 - std_pred)) + torch.mean(F.relu(1 - std_targ))
    
    B = pred.shape[0]
    C = pred.shape[1]
    
    pred_cent = pred - pred.mean(dim=0)
    cov_pred = (pred_cent.T @ pred_cent) / (B - 1)
    
    cov_loss = (off_diagonal(cov_pred)**2).sum() / C
    
    return sim_coeff * mse_loss + std_coeff * std_loss + cov_coeff * cov_loss

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class Dreamer(nn.Module):
    def __init__(self, mem_block_size, **config):
        super().__init__()
        mbs = mem_block_size
        self.block = modules.Block(**config)
        self.comp = modules.FFTResampler(mbs*2, mbs, mbs+1)
        self.ln = modules.LayerNorm(**config)
        
    def forward(self, x):
        while x.shape[0] > 1:
            b, t, c = x.size()
            x = x.view(b // 2, 2, t, c)
            
            x = x.transpose(1, 2)
            
            x = x.reshape(b // 2, t * 2, c)
            
            x = self.comp(x, dim=1) 
            x = self.block(x, causal = False)
            x = self.ln(x)
        return x 
    
class Model(nn.Module):
    def __init__(self, vocab_size, k=5, lambda_thought=0.1, **kwargs):
        super().__init__()
        # ... your usual model initialization (embedding, transformer blocks, lm_head)
        if qkhwn:
            wte = (nn.Embedding(vocab_size, n_embed))
        else:
            wte = nn.Embedding(vocab_size, n_embed)
        
        self.transformer = nn.ModuleDict(dict(
            wte = wte,
            wpe = nn.Embedding(block_size+1, n_embed),
            h = nn.ModuleList([Block(n_embed, dropout, False) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embed),
        ))
        
        if qkhwn:
            self.lm_head =  (modules.OrthogonalLinear(n_embed, vocab_size, bias=False))
            self.transformer.wte.weight = self.lm_head.weight 
            #self.lm_head2 =  (modules.OrthogonalLinear(n_embed, vocab_size, bias=False))
        else:
            self.lm_head =  nn.Linear(n_embed, vocab_size, bias=False)
            self.transformer.wte.weight = self.lm_head.weight 
        self.lambda_corrected = 0.0
        self.k = k 
        self.lambda_thought = lambda_thought 
        
        self.register_buffer('memory', torch.randn(1,1,n_embed))
        memconf = {
                'mem_block_size': 1,
                'dropout': 0.0,
                'n_embd': n_embed,
                'n_head': 4,
                'bias': False,
            }
        self.memorizer = Dreamer(**memconf)
        

    def get_weights(self):
        return [block.get_weight() for block in self.transformer.h]
    
    def ssmfwd(self, x, prevs = None, weights =None):
        b,t,c = x.shape
        if mem:
            xf=torch.cat([self.memory.expand(b,1,c),x],dim=1)
        else:
            xf=x
        qs = []
        for i, block in enumerate(self.transformer.h):
            
            weight = weights[i] if weights is not None else None
            prev = prevs[i] if prevs is not None else None
            
            xf, q_out = block(xf, prev, weight)
            if not mem:
                qs.append(q_out)

        if mem:
            nm = self.memorizer(xf[:,-1,:].unsqueeze(1))
            x=torch.cat([nm.expand(b,1,c),x],dim=1)
            for i, block in enumerate(self.transformer.h):

                weight = weights[i] if weights is not None else None
                prev = prevs[i] if prevs is not None else None

                x, q_out = block(x, prev, weight)
                q_out=q_out[:,1:,:]
                qs.append(q_out)
            x=x[:,1:,:]
            with torch.no_grad():
                nm2 = self.memorizer(x[:,-1,:].unsqueeze(1))
                self.memory.data = nm2
        else:
            x= xf
        return x, qs

    def forward(self, idx, Train=False, gumbel_temp=1.0, cl = 0.0):

        
        B, T = idx.size()
        T -= 2
        device = idx.device
        if not Train:
            x = self.transformer.wte(idx)
            hidden_states, _ = self.ssmfwd(x)
            logits = self.lm_head(hidden_states)
            return logits, None, None
        
        tok_emb = self.transformer.wte(idx)

        #pos_emb = self.transformer.wpe(torch.arange(T, device=device))#replace by rope

        x = tok_emb[:,:-2,:]
        Y1_gt = idx[:, 1:-1]
        ws = self.get_weights()
        
        full_x, _ = self.ssmfwd(tok_emb[:,:-1,:], weights=ws)
        
        #logits_full = self.lm_head(full_x)
        #logits_g1 = logits_full[:, :-1,:]

        #g1_pred_onehot = quantizer.TopKHot.apply(logits_g1, 3) / 3.0
        #g1_pred_onehot = torch.softmax(logits_g1, dim=-1)
        #g1_pred_onehot = F.gumbel_softmax(logits_g1, tau=gumbel_temp, hard=True, dim=-1)
        #g1_guess_emb =  g1_pred_onehot @ self.transformer.wte.weight
        g1_guess_emb = full_x[:, :-1,:]
        gen2_x, _ = self.ssmfwd(g1_guess_emb,  weights= ws)
        #logits_g2 = self.lm_head(gen2_x)

        #L_guess = utils.concordance_correlation_loss(gen2_x, tok_emb[:,2:,:].detach())
      
        #L_guess = F.cross_entropy(
        #    logits_g2.reshape(-1, logits_g2.size(-1)), 
        #    idx[:, 2:].reshape(-1), 
        #    reduction='none'
        #    ).view(Y1_gt.shape)
        bfw = self.lm_head.weight.to(torch.bfloat16)
        
        L_guess = linear_cross_entropy(
            gen2_x.to(torch.bfloat16),
            bfw, 
            idx[:, 2:], 
            impl='torch_compile', 
            reduction='none',
            ).view(Y1_gt.shape)

        #L_standard = F.cross_entropy(logits_g1.reshape(-1, logits_g1.size(-1)), Y1_gt.reshape(-1), reduction='none').view(Y1_gt.shape)
        #L_baseline = F.cross_entropy(logits_g3.reshape(-1, logits_g3.size(-1)), Y2_gt_flat, reduction='none').view(Y1_gt.shape)
        
        
        L_all = linear_cross_entropy(
            full_x.to(torch.bfloat16),
            bfw, 
            idx[:, 1:], 
            impl='torch_compile', 
            reduction='none',
            ).view(idx[:, 1:].shape)
        #L_T1 = L_all[:, :-1]
        #L_T2 = L_all[:, 1:]   
        if self.training:
            gk = utils.gaussian_kernel(L_guess,1.0,dim=-1)
            loss = L_all.mean() + torch.tanh(gk*L_guess).mean()
        else:
            loss = L_all.mean()
        L_guess = L_guess.mean().detach()
        #L_guess = None
        gloss = None#torch.zeros(1).detach()
        return None, loss, (gloss, L_guess)


    def trust_loss2(self, L_standard, L_baseline, L_guess):
        relative_improvement = L_baseline - L_guess
        
        cumulative_gain = torch.flip(torch.cumsum(torch.flip(relative_improvement, [1]), dim=1), [1])
        
        trust_gate = torch.sigmoid(cumulative_gain / 0.1)

        if self.training:
            loss = (trust_gate * L_guess) + ((1.0 - trust_gate) * L_baseline)
            loss.mean()
        else:
            loss = L_standard.mean()
        return loss, trust_gate
    
    def trust_loss(self, L_standard, L_baseline, L_guess):
        with torch.no_grad():
            Gain = L_baseline - L_guess
            T = 0.1 # A hyperparameter to tune
            trust_score = torch.sigmoid(Gain / T)
        L_masked_standard = L_standard * (1.0 - trust_score)

        if self.training:
            loss = L_masked_standard.mean() 
            loss = loss + (trust_score * L_guess).mean()
            loss = loss + L_baseline.mean()*0.1
        else:
            loss = L_standard.mean()
        gloss = trust_score.mean().detach()
        return loss, gloss
    
    def configure_optimizers(self, weight_decay=0.1, learning_rate=1e-3, betas=(0.9, 0.95)):
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
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
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

model = Model(vocab_size=vocab_size)
model.to(device)
optimizer = model.configure_optimizers(learning_rate=learning_rate)
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
    # In your training loop
    with torch.no_grad():
        max_lambda = 0.1 # Final value
        lambda_warmup_iters = 5000 # Ramp up over 5k steps
    
        # Calculate current lambda for this iteration
        current_lambda = max_lambda * min(1.0, iter / lambda_warmup_iters)
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if iter % eval_interval == 0 or iter == max_iters - 1:
        t2 = time.time()
        model.eval()
        losses = {}
        for split in ['train', 'val']:
            X_val = get_batch(split,block_size+2)
            with torch.no_grad():
                _, loss, gloss = model(X_val, True, cl = current_lambda)
            losses[split] = loss.detach().item()
            if genloss and split == 'val':
                glosses = [
                    gloss[index].item() if gloss[index] is not None else None
                    for index in range(len(gloss))
                ]
        optimizer.zero_grad(set_to_none=True)
        t1 = time.time()
        
        print(f"step {iter:{len(str(max_iters))}d}: "
              f"train loss: {losses['train']:.4f}, "
              f"val loss: {losses['val']:.4f}, ",end="")
        if genloss:
            utils.printReal("gen 1 loss: ", glosses, 0)
            utils.printReal("gen 2 loss: ", glosses, 1)
        print(f"time: {(t2-t0)*1000:.0f} ms, "
              f"Tval: {(t1-t2)*1000:.0f} ms")
        
        model.train()
        t0 = time.time()

    X = get_batch('train',block_size+2)
    logits, loss, _ = model(X, True, cl = current_lambda)

    loss.backward()
    
    if gradlog and iter > 100:
        sorted_grads = sorted(gradient_norms.items(), key=lambda item: item[1], reverse=True)
        for i, (name, norm) in enumerate(sorted_grads):
            print(f"{i+1}. {name:<60} | Norm: {norm:.4f}")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if wnorm:
        with torch.no_grad():
            utils.softwnorm(model,1e-4)
    
if gen:
    print("sanity check")
    X = get_batch('train')
    with torch.no_grad():
        model.eval()
        outp = model.generate(X[0].unsqueeze(0), 100, temperature=1.0, top_k=5)
        model.train()
    print(decode(outp[0,-100:].detach().cpu().numpy().tolist()))
    print("Training finished.")