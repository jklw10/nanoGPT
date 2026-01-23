import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import modules
import utils

# --- Config ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)
torch.backends.cudnn.enabled = False
torch.backends.cuda.matmul.allow_tf32 = True 
torch._inductor.config.disable_cpp_codegen = True

# Test Settings
compiled =True
#TEST_MODES = ['Baseline', 'gdiff', 'Exact_NG']
#TEST_MODES = ['sgd','adamw','gdiff']
TEST_MODES = ['sgd_hungry_gdiff']
#TEST_MODES = ['Exact_NG']
MAX_ITERS = 10000       # Short run for comparison
EVAL_INTERVAL = 1000
BATCH_SIZE = 64       # Kept small for Exact NG calculation
BLOCK_SIZE = 64       
N_EMBED = 256         # Small dim to allow "Exact" Jacobian calc (O(N^3))
N_LAYER = 2
LR = 1e-3
NG_LAMBDA = 1e-2    # Strength of the surface minimization

dataset = 'openwebtext'
always_byte=True
decode, vocab_size = utils.get_meta(dataset,always_byte)
get_batch = utils.get_get_batch(BLOCK_SIZE,BATCH_SIZE,dataset,device,always_byte=always_byte)
print(f"vocab: {vocab_size}")
class LinearSSM(nn.Module):
    def __init__(self, n_embd, linear):
        super().__init__()
        self.n_embd = n_embd
        self.proj = linear(n_embd, 2 * n_embd)

    @staticmethod
    def binary_operator(q_i, q_j):
        g_i, v_i = q_i.chunk(2, dim=-1)
        g_j, v_j = q_j.chunk(2, dim=-1)
        
        new_gate = g_j * g_i
        new_val = (g_j * v_i) + v_j
        
        return torch.cat([new_gate, new_val], dim=-1)

    def forward(self, x):
        B, T, C = x.shape
        params = self.proj(x)
        
        gates, values = params.chunk(2, dim=-1)
        gates = torch.sigmoid(gates)
        values = torch.tanh(values)
        
        packed_input = torch.cat([gates, values], dim=-1)
        
        identity = torch.cat([
            torch.ones(1, 1, self.n_embd, device=x.device),
            torch.zeros(1, 1, self.n_embd, device=x.device)
        ], dim=-1)
        
        hidden_states_packed = utils.pscan(packed_input, self.binary_operator, identity)
        
        _, final_hidden = hidden_states_packed.chunk(2, dim=-1)
        
        return final_hidden

class TinyModel(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, N_EMBED)
        self.ssm = LinearSSM(N_EMBED,linear)
        self.head = linear(N_EMBED, vocab_size)
        
    def forward(self, idx, targets=None, return_embeddings=False):
        emb = self.token_embedding(idx)
        
        if return_embeddings:
            emb.retain_grad()
        x = self.ssm(emb)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
            
        if return_embeddings:
            return logits, loss, emb, x
        return logits, loss

# --- 4. Training Loop ---

results = {}

print(f"{'='*60}")
print(f"Comparing Surface Minimization Techniques")
print(f"Device: {device} | Embed Dim: {N_EMBED} | Lambda: {NG_LAMBDA}")
print(f"{'='*60}")


for mode in TEST_MODES:
    print(f"\n---> Training Mode: {mode}")
    
    # Reset Model
    if 'gdiff' in mode:
        if 'hungry' in mode:
            Linear=modules.HungryGdiffLinear
        else:
            Linear=modules.GdiffLinear
    elif 'hungry' in mode:
        Linear=modules.HungryLinear
    else:
        Linear=nn.Linear

    model = TinyModel(Linear).to(device)
    if compiled:
        raw_model=model
        model = torch.compile(model)
    if 'adamw' in mode:
        print("using adamw")
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR,fused=True)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=LR,fused=True)

    start_time = time.time()
    losses = []
    for iter in range(MAX_ITERS):
        optimizer.zero_grad(set_to_none=True)
        
        xb, yb = get_batch('train')
        
        if mode == 'Exact_NG':
            optimizer.zero_grad(set_to_none=True)
            logits, loss, emb, hidden = model(xb, yb, return_embeddings=True)

            reg_loss = utils.surface_minimize(emb, hidden)
            loss = loss + reg_loss*NG_LAMBDA
            loss.backward()
            optimizer.step()
            # Standard training
        else:
            optimizer.zero_grad(set_to_none=True)
            logits, loss, _, _ = model(xb, yb, return_embeddings=True)
            loss.backward()
            optimizer.step()
        
        losses.append(loss.item())
        
        if iter % EVAL_INTERVAL == 0:
            
            end_time = time.time()
            model.eval()
            with torch.no_grad():
                xv, yv = get_batch('val')
                _, val_loss = model(xv, yv)
            model.train()
            print(f"Step {iter}: train Loss {loss.item():.4f}", end="")
            print(f" val Loss {val_loss.item():.4f}", end="")
            print(f" time {end_time - start_time:.4f}")
            
            

    end_time = time.time()
    
    # Final Eval
    model.eval()
    with torch.no_grad():
        xv, yv = get_batch('val')
        _, val_loss = model(xv, yv)
    
    results[mode] = {
        'final_train': np.mean(losses[-10:]),
        'final_val': val_loss.item(),
        'time': end_time - start_time
    }
    print(f"Finished {mode}. Val Loss: {val_loss.item():.4f}. Time: {end_time - start_time:.2f}s")

# --- 5. Summary ---
print(f"\n{'='*60}")
print(f"{'Mode':<15} | {'Val Loss':<10} | {'Train Loss':<10} | {'Time (s)':<10}")
print(f"{'-'*60}")
for mode in TEST_MODES:
    r = results[mode]
    print(f"{mode:<15} | {r['final_val']:.4f}     | {r['final_train']:.4f}     | {r['time']:.2f}")
print(f"{'='*60}")

# Sanity Generation
print("\nGenerative Sanity Check (last model):")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
if compiled:
    model = raw_model
model.eval()
for _ in range(100):
    logits, _ = model(context)
    logits = logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probs, num_samples=1)
    context = torch.cat((context, idx_next), dim=1)
print(decode(context[0].tolist()))