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
    # We now need block_size + 2 tokens for each sample
    ix = torch.randint(len(data) - (block_size + 2), (batch_size,))
    
    # X is the context, from i to i+block_size
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    
    # Y1 is the target for T+1, from i+1 to i+1+block_size
    y1 = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    # Y2 is the target for T+2, from i+2 to i+2+block_size
    y2 = torch.stack([torch.from_numpy((data[i+2:i+2+block_size]).astype(np.int64)) for i in ix])
    
    if device_type == 'cuda': 
        x, y1, y2 = x.pin_memory().to(device, non_blocking=True), y1.pin_memory().to(device, non_blocking=True), y2.pin_memory().to(device, non_blocking=True)
    else: 
        x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
        
    return x, y1, y2

@dataclass
class TransformerOutput:
    last_hidden_state: torch.Tensor
    past_key_values: tuple = None
    # You can add other outputs like attentions if needed

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embed, n_head, dropout, bias):
        super().__init__()
        assert n_embed % n_head == 0
        # key, query, value projections for all heads, but in a batch
        if qkhwn:
            self.q_attn = wn(nn.Linear(n_embed, n_embed, bias=bias))
            self.k_attn = wn(nn.Linear(n_embed, n_embed, bias=bias))
        else:
            self.q_attn = nn.Linear(n_embed, n_embed, bias=bias)
            self.k_attn = nn.Linear(n_embed, n_embed, bias=bias)

        self.v_attn = nn.Linear(n_embed, n_embed, bias=bias)

        # output projection
        self.c_proj = nn.Linear(n_embed, n_embed, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embed = n_embed
        self.dropout = dropout
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(1024, 1024))
                                     .view(1, 1, 1024, 1024))

    def forward(self, x, past_key_values=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embed)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.q_attn(x)
        k = self.k_attn(x)
        v = self.v_attn(x)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        # KV Caching logic
        if past_key_values is not None:
            # The past_key_values are of shape (B, nh, T_past, hs)
            past_key, past_value = past_key_values
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)
        
        # The T dimension for k and v might be different from q if we're using a cache
        T_kv = k.shape[-2]
        present_key_values = (k, v)


        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T_kv) -> (B, nh, T, T_kv)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1)**0.5))
        att = att.masked_fill(self.bias[:,:,:T,:T_kv] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T_kv) x (B, nh, T_kv, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side-by-side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, present_key_values

class MLP(nn.Module):
    def __init__(self, n_embed, dropout, bias):
        super().__init__()
        self.c_fc    = nn.Linear(n_embed, 4 * n_embed, bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * n_embed, n_embed, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, n_embed, n_head, dropout, bias):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embed)
        self.attn = CausalSelfAttention(n_embed, n_head, dropout, bias)
        self.ln_2 = nn.LayerNorm(n_embed)
        self.mlp = MLP(n_embed, dropout, bias)

    def forward(self, x, past_key_values=None):
        attn_output, present_key_values = self.attn(self.ln_1(x), past_key_values=past_key_values)
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x, present_key_values


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
            wpe = nn.Embedding(block_size, n_embed),
            h = nn.ModuleList([Block(n_embed, n_head, dropout, False) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embed),
        ))
        
        if qkhwn:
            self.lm_head =  (modules.OrthogonalLinear(n_embed, vocab_size, bias=False))
            #self.lm_head.parametrizations.weight[0] = self.transformer.wte.parametrizations.weight[0]
            self.transformer.wte.weight = self.lm_head.weight 
            self.lm_head2 =  (modules.OrthogonalLinear(n_embed, vocab_size, bias=False))
        else:
            self.lm_head =  nn.Linear(n_embed, vocab_size, bias=False)
            self.transformer.wte.weight = self.lm_head.weight 
        self.lambda_corrected = 0.0
        self.k = k 
        self.lambda_thought = lambda_thought 

    def transformer_forward(self, idx=None, inputs_embeds=None, past_key_values=None, use_cache=False):
        if idx is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both idx and inputs_embeds at the same time")
        if idx is None and inputs_embeds is None:
            raise ValueError("You must specify either idx or inputs_embeds")

        if idx is not None:
            B, T = idx.size()
            device = idx.device
        else:
            B, T, E = inputs_embeds.size()
            device = inputs_embeds.device

        if inputs_embeds is None:
            tok_emb = self.transformer.wte(idx)
        else:
            tok_emb = inputs_embeds

        pos_emb = self.transformer.wpe(torch.arange(T, device=device))
        x = tok_emb + pos_emb

        present_key_values = []

        for i, block in enumerate(self.transformer.h):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, present_kv = block(x, past_key_values=past_kv)
            if use_cache:
                present_key_values.append(present_kv)

        x = self.transformer.ln_f(x)

        if use_cache:
            return TransformerOutput(last_hidden_state=x, past_key_values=tuple(present_key_values))
        else:
            return TransformerOutput(last_hidden_state=x)

    def forward(self, idx, targets=None, gumbel_temp=1.0, cl = 0.0):
        if targets is None:
            # Standard inference pass (no changes needed here)
            hidden_states = self.transformer_forward(idx).last_hidden_state
            logits = self.lm_head(hidden_states)
            return logits, None, None
        
        Y1_gt, Y2_gt = targets
       
        # PASS 1: HYPOTHESIS PASS (gen1)
        transformer_output_X = self.transformer_forward(idx, use_cache=True)
        hidden_states_X = transformer_output_X.last_hidden_state
        kv_cache_X = transformer_output_X.past_key_values
        
        logits_t1 = self.lm_head(hidden_states_X)
        L_standard = F.cross_entropy(logits_t1.view(-1, logits_t1.size(-1)), Y1_gt.view(-1), reduction='none').view(Y1_gt.shape)
        
        #T1_pred_onehot = quantizer.TopKHot.apply(logits_t1, 1)
        T1_pred_onehot = F.gumbel_softmax(logits_t1, tau=gumbel_temp, hard=True, dim=-1)
        soft_embeddings_pred = T1_pred_onehot @ self.transformer.wte.weight
        
        
        # PASS 2: TEST PASS (gen2)
        hidden_states_T1_pred = self.transformer_forward(inputs_embeds=soft_embeddings_pred, past_key_values=kv_cache_X).last_hidden_state
        logits_t2_corrected = self.lm_head(hidden_states_T1_pred)
        L_corrected = F.cross_entropy(logits_t2_corrected.view(-1, logits_t2_corrected.size(-1)), Y2_gt.view(-1), reduction='none').view(Y2_gt.shape)
        
        ## PASS 3: BASELINE PASS (gen3)
        ## Note: We only pass the *new* tokens. The context is handled by the KV cache.
        #hidden_states_T1_gt = self.transformer_forward(Y1_gt, past_key_values=kv_cache_X).last_hidden_state
        #logits_t2_baseline = self.lm_head(hidden_states_T1_gt)
        #L_baseline = F.cross_entropy(logits_t2_baseline.view(-1, logits_t2_baseline.size(-1)), Y2_gt.view(-1), reduction='none').view(Y2_gt.shape)
        
        
        ## PASS 3: do a mtp head instead of full pass for self knowledge sample
        logits_t2_baseline = self.lm_head2(hidden_states_X)
        L_baseline = F.cross_entropy(logits_t2_baseline.view(-1, logits_t2_baseline.size(-1)), Y2_gt.view(-1), reduction='none').view(Y2_gt.shape)
        
        # REWARD CALCULATION
        #with torch.no_grad():
        #Gain = L_baseline - L_corrected
        #Gain = (Gain - Gain.mean().detach()) / (Gain.std().detach() + 1e-8) 
        #Gain = torch.tanh(Gain / 2.0)
        #    #Gain = torch.clamp(Gain, -5.0, 5.0) 
        ##Gain = utils.minmaxnorm(Gain)
        #log_probs_t1 = F.log_softmax(logits_t1, dim=-1)
        #log_prob_T1_pred = torch.sum(log_probs_t1 * T1_pred_onehot, dim=-1)
        #
        #L_thought = -(Gain * log_prob_T1_pred)
        
        #try2
        with torch.no_grad():
            Gain = L_baseline - L_corrected
            T = 0.1 # A hyperparameter to tune
            trust_score = torch.sigmoid(Gain / T)
        L_masked_standard = L_standard * (1.0 - trust_score)

        if self.training:
            #loss = L_standard.mean()
            #loss = loss + L_masked_standard.mean() 
            loss = L_masked_standard.mean() 
            loss = loss + (trust_score * L_corrected).mean()
            loss = loss + L_baseline.mean()*0.1
            #loss = loss + 0.1 * L_masked_standard.mean() #
        else:
            loss = L_standard.mean()
        gloss = trust_score.mean().detach()
        #gloss = None#torch.zeros(1).detach()
        return logits_t1, loss, (gloss, L_corrected.mean().detach())
    
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
        glosses = {}
        for split in ['train', 'val']:
            X_val, Y1_val, Y2_val = get_batch(split)
            with torch.no_grad():
                _, loss, gloss = model(X_val, (Y1_val, Y2_val), cl = current_lambda)
            losses[split] = loss.detach().item()
            if genloss and split == 'val':
                if gloss[0] is not None:
                    glosses['gloss'] = gloss[0].item()
                glosses['g2'] = gloss[1].item()
        optimizer.zero_grad(set_to_none=True)
        t1 = time.time()
        print(f"step {iter:{len(str(max_iters))}d}: "
              f"train loss: {losses['train']:.4f}, "
              f"val loss: {losses['val']:.4f}, ",end="")
        if genloss:
            if len(glosses) > 1:
                print(f"gen loss {glosses['gloss']:.4f}, ",end="")
                  
            print(f"g2 loss {glosses['g2']:.4f}, "
                  ,end="")
        print(f"time: {(t2-t0)*1000:.0f} ms, "
              f"Tval: {(t1-t2)*1000:.0f} ms")
        
        model.train()
        t0 = time.time()

    X, Y1, Y2 = get_batch('train')
    logits, loss, gloss = model(X, (Y1, Y2), cl = current_lambda)

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
    X, Y, _ = get_batch('train')
    with torch.no_grad():
        model.eval()
        outp = model.generate(X[0].unsqueeze(0), 100, temperature=1.0, top_k=5)
        model.train()
    print(decode(outp[0,-100:].detach().cpu().numpy().tolist()))
    print("Training finished.")