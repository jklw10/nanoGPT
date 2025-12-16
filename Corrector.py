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
from dataclasses import dataclass

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

dropout = 0.0
max_iters = 50000
eval_interval = 1000
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_type = 'cuda'
warmup_iters = 200
lr_decay_iters = 30000
min_lr = learning_rate/10
beta2 = 0.999 

torch._inductor.config.disable_cpp_codegen = True


idk_enabled = False
cl          = False
mem         = False
gradlog     = False
gen         = False
memcheat    = False
wnorm       = False
genloss     = True

lbc         = True

qkhwn       = False
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
if dataset == "shakespeare_char":
    meta_path = os.path.join(data_dir, 'meta.pkl')
    with open(meta_path, 'rb') as f: 
        meta = pickle.load(f)
    vocab_size = meta['vocab_size']
    if idk_enabled:
        idk_token = vocab_size
        vocab_size += 1
else:
    vocab_size = 50304


def get_batch(split):
    data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
    
    # We grab block_size + 2 tokens total
    ix = torch.randint(len(data) - (block_size + 2), (batch_size,))
    
    # One single tensor fetch
    chunk = torch.stack([torch.from_numpy((data[i:i+block_size+2]).astype(np.int64)) for i in ix])
    
    if device_type == 'cuda': 
        chunk = chunk.pin_memory().to(device, non_blocking=True)
    else: 
        chunk = chunk.to(device)
        
    return chunk



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
            h = nn.ModuleList([modules.Block(n_embed, n_head, dropout, False) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embed),
        ))
        
        if qkhwn:
            self.lm_head =  (modules.OrthogonalLinear(n_embed, vocab_size, bias=False))
            self.transformer.wte.weight = self.lm_head.weight 
        else:
            self.lm_head =  nn.Linear(n_embed, vocab_size, bias=False)
            self.transformer.wte.weight = self.lm_head.weight 
        
        self.lambda_corrected = 0.0
        self.k = k 
        self.lambda_thought = lambda_thought 

    def transformer_forward(self, x, causal = True):
        

        for i, block in enumerate(self.transformer.h):
            x = block(x, causal)

        return self.transformer.ln_f(x)


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

        #TODO: replace by rope
        pos_emb = self.transformer.wpe(torch.arange(T+1, device=device))

        x = tok_emb[:,:-2,:]
        Y1_gt = idx[:, 1:-1]
        
        full_x = self.transformer_forward(tok_emb[:,:-1,:]+pos_emb)
        
        logits_full = self.lm_head(full_x)
        logits_g1 = logits_full[:, :-1,:]

        #g1_pred_onehot = quantizer.TopKHot.apply(logits_g1, 3) / 3.0
        #g1_pred_onehot = torch.softmax(logits_g1, dim=-1)
        g1_pred_onehot = F.gumbel_softmax(logits_g1, tau=gumbel_temp, hard=True, dim=-1)
        g1_guess_emb = g1_pred_onehot @ self.transformer.wte.weight
        
        gen2_x = self.transformer_forward(g1_guess_emb+pos_emb[1:,:], causal=False)
        

        bfw = self.lm_head.weight.to(torch.bfloat16)
        
        L_guess = linear_cross_entropy(
            gen2_x.to(torch.bfloat16),
            bfw, 
            idx[:, 2:], 
            impl='torch_compile', 
            reduction='none',
            ).view(Y1_gt.shape)

        
        L_all = linear_cross_entropy(
            full_x.to(torch.bfloat16),
            bfw, 
            idx[:, 1:], 
            impl='torch_compile', 
            reduction='none',
            ).view(idx[:, 1:].shape)
        if self.training:
            gk = utils.gaussian_kernel(L_guess,1.0,dim=-1) #centered on the end of sequence
            loss = L_all.mean() + (gk*L_guess).mean()
        else:
            loss = L_all.mean()
        L_guess = L_guess.mean().detach()
        gloss = None#torch.zeros(1).detach()
        return logits_g1, loss, (gloss, L_guess)

    
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

model = Model(vocab_size=vocab_size ,betas=(0.9,0.999))
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
            X_val = get_batch(split)
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

    X = get_batch('train')
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
    X, Y, _ = get_batch('train')
    with torch.no_grad():
        model.eval()
        outp = model.generate(X[0].unsqueeze(0), 100, temperature=1.0, top_k=5)
        model.train()
    print(decode(outp[0,-100:].detach().cpu().numpy().tolist()))
    print("Training finished.")