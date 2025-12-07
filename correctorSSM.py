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

class Block(nn.Module):
    def __init__(self, n_embed, dropout, bias):
        super().__init__()
        self.ln_1 = modules.LayerNorm(n_embed, bias)
        self.attn = modules.SSM(n_embed, dropout, bias)
        self.ln_2 = modules.LayerNorm(n_embed, bias)
        self.mlp = modules.MLP(n_embed, dropout, bias)
    def get_weight(self):
        return self.attn.get_weight()
    def forward(self, x, prev=None, weights=None):
        if prev is not None:
             attn_output, present_key_values = self.attn.nexts(prev, self.ln_1(x),weights)
        else:
             attn_output, present_key_values = self.attn(self.ln_1(x),weights)
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x, present_key_values
def compute_discounted_return(rewards, gamma=0.95):
    r_flipped = torch.flip(rewards, dims=[1])
    
    if gamma == 1.0:
        returns = torch.cumsum(r_flipped, dim=1)
    else:
        B, T = rewards.shape
        returns = torch.zeros_like(rewards)
        running_add = torch.zeros(B, device=rewards.device)
        
        for t in range(T):
            running_add = r_flipped[:, t] + gamma * running_add
            returns[:, t] = running_add
            
    return torch.flip(returns, dims=[1])

def CE_bkw(logits, target_indices):
    #try sigmoid at some point
    probs = F.softmax(logits, dim=-1)
    
    target_hot = torch.zeros_like(probs)
    target_hot.scatter_(-1, target_indices.unsqueeze(-1), 1.0)
    
    return probs - target_hot

class HypothesisLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g1, g2, g3, idx_data):
        
        ctx.save_for_backward(g1, g2, g3, idx_data)
        
        return F.cross_entropy(g1.reshape(-1, g1.size(-1)), idx_data[:, 1:-1].reshape(-1))
        
    @staticmethod
    def backward(ctx, grad_output):
        g1, g2, g3, idx_data = ctx.saved_tensors
        
        idx_data_flat = idx_data.view(-1, 1)
        g1_flat = g1.view(-1, 1)
        g2_flat = g2.view(-1, 1)
        g3_flat = g3.view(-1, 1)
        
        grad_g3 = CE_bkw(g3, idx_data[:,2:])

        grad_input = torch.zeros_like(g1)

        return grad_input, None, None, None

class AdaptiveDIEMLoss(nn.Module):
    def __init__(self, momentum=0.01):
        super().__init__()
        self.momentum = momentum
        
        self.register_buffer('running_mean_dist', torch.tensor(1.0))
        self.register_buffer('running_var_dist', torch.tensor(1.0))
        
    def forward(self, pred, target):
        d_actual = torch.norm(pred - target, p=2, dim=-1)
        
        if self.training:
            with torch.no_grad():
                flat_target = target.view(-1, target.size(-1))
                
                perm = torch.randperm(flat_target.size(0), device=target.device)
                shuffled_target = flat_target[perm]
                flat_pred = pred.view(-1, pred.size(-1))
                
                d_random = torch.norm(flat_pred - shuffled_target, p=2, dim=-1)
                
                batch_mean = d_random.mean()
                batch_var = d_random.var()
                
                self.running_mean_dist = (1 - self.momentum) * self.running_mean_dist + self.momentum * batch_mean
                self.running_var_dist = (1 - self.momentum) * self.running_var_dist + self.momentum * batch_var

        sigma = torch.sqrt(self.running_var_dist) + 1e-8
        mu = self.running_mean_dist
        
        loss = (d_actual - mu) / sigma
        
        return loss.mean()
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

    def get_weights(self):
        return [block.get_weight() for block in self.transformer.h]
    
    def ssmfwd(self, x, prevs = None, weights =None):
        
        qs = []
        for i, block in enumerate(self.transformer.h):
            
            weight = weights[i] if weights is not None else None
            prev = prevs[i] if prevs is not None else None
            
            x, q_out = block(x, prev, weight)
            qs.append(q_out)
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
        Y2_gt_flat = idx[:, 2:].reshape(-1)
        ws = self.get_weights()
        # PASS 1: HYPOTHESIS PASS (gen1)
        #gen1_x, gen1_qs = self.ssmfwd(x, weights=ws)
        #tok_emb_g3 = tok_emb[:, 1:-1, :]
        #gen3_x, _ = self.ssmfwd(tok_emb_g3, prevs=gen1_qs, weights=ws)
        #logits_g1 = self.lm_head(gen1_x)
        #logits_g3 = self.lm_head(gen3_x)

        full_x, full_qs = self.ssmfwd(tok_emb[:,:-1,:], weights=ws)
        
        logits_full= self.lm_head(full_x)
        logits_g1 = logits_full[:, :-1,:]
        #logits_g3 = logits_full[:, 1:, :]

        ## PASS 3: do a mtp head instead of full pass for self knowledge sample
        
        
        #T1_pred_onehot = quantizer.TopKHot.apply(logits_t1, 1)
        g1_pred_onehot = F.gumbel_softmax(logits_g1, tau=gumbel_temp, hard=True, dim=-1)
        g1_guess_emb = g1_pred_onehot @ self.transformer.wte.weight
        
        #g1_guess_emb = g1_guess_emb + self.transformer.wpe(torch.arange(1,T+1, device=device)) #this is why rope would be better in my intuition
        # PASS 2: TEST PASS (gen2)
        #gen2_x, _ = self.ssmfwd(g1_guess_emb, prevs=gen1_qs, weights= ws)
        #gen1_qs = [layer_q[:, :-1, :] for layer_q in full_qs]
        
        gen2_x, _ = self.ssmfwd(g1_guess_emb,  weights= ws)
        #logits_g2 = self.lm_head(gen2_x)

        #L_guess = utils.concordance_correlation_loss(gen2_x, tok_emb[:,2:,:].detach())
        L_guess = vicreg_loss(gen2_x, tok_emb[:,2:,:].detach())
        #L_guess = F.cross_entropy(logits_g2.reshape(-1, logits_g2.size(-1)), Y2_gt_flat, reduction='none').view(Y1_gt.shape)
        #L_standard = F.cross_entropy(logits_g1.reshape(-1, logits_g1.size(-1)), Y1_gt.reshape(-1), reduction='none').view(Y1_gt.shape)
        #L_baseline = F.cross_entropy(logits_g3.reshape(-1, logits_g3.size(-1)), Y2_gt_flat, reduction='none').view(Y1_gt.shape)
        L_all = F.cross_entropy(
            logits_full.reshape(-1, logits_full.size(-1)), 
            idx[:, 1:].reshape(-1), 
            reduction='none'
            ).view(idx[:, 1:].shape)
        L_T1 = L_all[:, :-1]
        L_T2 = L_all[:, 1:]   
        if self.training:
            loss = L_all.mean()+L_guess
        else:
            loss = L_all[:, :-1].mean()
        #loss, gloss = self.correction_loss(logits_g1, L_standard, L_baseline, g1_pred_onehot, L_corrected)
        #loss, gloss = self.trust_loss2(L_T1, L_T2, L_guess)
        #L_guess = L_guess.mean().detach()
        L_guess = None
        gloss = None#torch.zeros(1).detach()
        return logits_g1, loss, (gloss, L_guess)


    def trust_loss2(self, L_standard, L_baseline, L_guess):
        relative_improvement = L_baseline - L_guess
        
        # 2. Cumulative Future Gain (The Butterfly Effect)
        # We sum improvements from the future back to the present
        cumulative_gain = torch.flip(torch.cumsum(torch.flip(relative_improvement, [1]), dim=1), [1])
        
        # 3. The Trust Gate
        # We divide by a small temp (e.g., 0.1) to make the switch sharper.
        # gain > 0 -> gate -> 1.0 (Trust the Guess)
        # gain < 0 -> gate -> 0.0 (Trust the Dataset)
        trust_gate = torch.sigmoid(cumulative_gain / 0.1)

        if self.training:
            loss = (trust_gate * L_guess) + ((1.0 - trust_gate) * L_baseline)
        else:
            loss = L_standard.mean()
        return loss, trust_gate
    def trust_loss(self, L_standard, L_baseline, L_corrected):
        with torch.no_grad():
            Gain = L_baseline - L_corrected
            T = 0.1 # A hyperparameter to tune
            trust_score = torch.sigmoid(Gain / T)
        L_masked_standard = L_standard * (1.0 - trust_score)

        if self.training:
            loss = L_masked_standard.mean() 
            loss = loss + (trust_score * L_corrected).mean()
            loss = loss + L_baseline.mean()*0.1
        else:
            loss = L_standard.mean()
        gloss = trust_score.mean().detach()
        return loss,gloss
    
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

def printReal(definer, cont, id):
    if  cont is None or \
        id >= len(cont) or \
        cont[id] is None:
        return
    print(f"{definer}{cont[id]:.4f}, ",end="")

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
            printReal("gen 1 loss: ", glosses, 0)
            printReal("gen 2 loss: ", glosses, 1)
        print(f"time: {(t2-t0)*1000:.0f} ms, "
              f"Tval: {(t1-t2)*1000:.0f} ms")
        
        model.train()
        t0 = time.time()

    X = get_batch('train')
    logits, loss, gloss = model(X, True, cl = current_lambda)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    
    if gradlog and iter > 100:
        sorted_grads = sorted(gradient_norms.items(), key=lambda item: item[1], reverse=True)
        for i, (name, norm) in enumerate(sorted_grads):
            print(f"{i+1}. {name:<60} | Norm: {norm:.4f}")
    #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
    X = get_batch('train')
    with torch.no_grad():
        model.eval()
        outp = model.generate(X[0].unsqueeze(0), 100, temperature=1.0, top_k=5)
        model.train()
    print(decode(outp[0,-100:].detach().cpu().numpy().tolist()))
    print("Training finished.")