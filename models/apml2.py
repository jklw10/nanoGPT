import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import hashlib

torch.manual_seed(42)
np.random.seed(42)

GRID_SIZE, NUM_ACTIONS, QUANT_DIM, K_ACTIVE = 8, 4, 16, 8
TOTAL_STATES = GRID_SIZE * GRID_SIZE

def position_to_grid(row, col):
    grid = torch.zeros(GRID_SIZE, GRID_SIZE)
    grid[row, col] = 1.0
    return grid.view(-1)

def step_env(row, col, action):
    if action == 0: row = max(0, row - 1)
    elif action == 1: row = min(GRID_SIZE - 1, row + 1)
    elif action == 2: col = max(0, col - 1)
    elif action == 3: col = min(GRID_SIZE - 1, col + 1)
    return row, col

# Gumbel-Nudged Top-K STE (Injected Noise for Exploration)
def gumbel_topkhotSTE(x, k=K_ACTIVE, temp=1.0, training=True):
    if training:
        u = torch.rand_like(x)
        gumbel_noise = -torch.log(-torch.log(u + 1e-10) + 1e-10)
        noisy_x = (x + gumbel_noise * 0.2) / temp
    else:
        noisy_x = x

    probs = torch.sigmoid(noisy_x)
    _, indices = torch.topk(noisy_x, k=k, dim=-1)
    hard_target = torch.zeros_like(x).scatter_(-1, indices, 1.0)
    return hard_target + probs - probs.detach()

def to_bits(z):
    return "".join(["1" if v.item() > 0.5 else "0" for v in z.squeeze()])

# ==========================================
# RESIDUAL & GROUNDED UNIFIED MODEL
# ==========================================
class ResidualGroundedModel(nn.Module):
    def __init__(self, obs_dim=64, action_dim=NUM_ACTIONS, qdim=QUANT_DIM, k=K_ACTIVE):
        super().__init__()
        self.k = k
        self.encoder = nn.Sequential(nn.Linear(obs_dim, 128), nn.LeakyReLU(), nn.Linear(128, qdim))
        self.decoder = nn.Sequential(nn.Linear(qdim, 128), nn.LeakyReLU(), nn.Linear(128, obs_dim))

        # Residual Transition Cell: Computes Delta Z given (Z_t, Action)
        self.transition_delta = nn.Sequential(
            nn.Linear(qdim + action_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, qdim)
        )

    def encode(self, x, training=True):
        logits = self.encoder(x)
        return logits, gumbel_topkhotSTE(logits, self.k, training=training)

    def decode(self, z):
        return self.decoder(z)

    def step(self, z_t, action_oh, training=True):
        inp = torch.cat([z_t, action_oh], dim=-1)
        delta_logits = self.transition_delta(inp)
        
        # RESIDUAL STREAM: Update prior logits via delta before quantization
        # This allows the model to preserve identity by default
        updated_logits = z_t + delta_logits
        z_next_quant = gumbel_topkhotSTE(updated_logits, self.k, training=training)
        
        return updated_logits, z_next_quant

# ==========================================
# EVALUATION METRICS
# ==========================================
@torch.no_grad()
def evaluate_metrics(model):
    model.eval()
    unique_codes = set()
    code_map = {}
    collisions = 0
    
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            x = position_to_grid(r, c).unsqueeze(0)
            _, z = model.encode(x, training=False)
            b_str = to_bits(z)
            if b_str in code_map: collisions += 1
            else: code_map[b_str] = (r, c)
            unique_codes.add(b_str)

    correct_transitions = 0
    total_transitions = TOTAL_STATES * NUM_ACTIONS
    
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            x_t = position_to_grid(r, c).unsqueeze(0)
            _, z_t = model.encode(x_t, training=False)
            
            for a in range(NUM_ACTIONS):
                nr, nc = step_env(r, c, a)
                x_next_true = position_to_grid(nr, nc).unsqueeze(0)
                _, z_next_true = model.encode(x_next_true, training=False)
                
                a_oh = F.one_hot(torch.tensor([a], dtype=torch.long), NUM_ACTIONS).float()
                _, z_next_pred = model.step(z_t, a_oh, training=False)
                
                if to_bits(z_next_pred) == to_bits(z_next_true):
                    correct_transitions += 1

    trans_acc = (correct_transitions / total_transitions) * 100.0
    model.train()
    return len(unique_codes), collisions, trans_acc

# ==========================================
# TRAINING LOOP WITH COMMITMENT LOSS
# ==========================================
print("--- Training Residual Model with Gumbel STE + Commitment Loss ---")
model = ResidualGroundedModel()
optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)

epochs = 5000
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
batch_size = 256

print(f"{'Epoch':>6} | {'Loss':>8} | {'LR':>8} | {'Unique States':>13} | {'Collisions':>10} | {'Transition Acc':>14}")
print("-" * 72)

for epoch in range(epochs):
    rows = np.random.randint(0, GRID_SIZE, size=batch_size)
    cols = np.random.randint(0, GRID_SIZE, size=batch_size)
    actions = np.random.randint(0, NUM_ACTIONS, size=batch_size)

    x_t = torch.stack([position_to_grid(r, c) for r, c in zip(rows, cols)])
    x_next = torch.stack([position_to_grid(*step_env(r, c, a)) for r, c, a in zip(rows, cols, actions)])
    a_oh = F.one_hot(torch.tensor(actions, dtype=torch.long), NUM_ACTIONS).float()

    _, z_t = model.encode(x_t, training=True)
    _, z_next_target = model.encode(x_next, training=True)

    updated_logits, z_next_pred = model.step(z_t, a_oh, training=True)
    rec_x_curr = model.decode(z_t)
    rec_x_next = model.decode(z_next_pred)

    # 1. Reconstruction & Dynamics Losses
    loss_rec = F.binary_cross_entropy_with_logits(rec_x_curr, x_t) + \
               F.binary_cross_entropy_with_logits(rec_x_next, x_next)
    loss_dyn = F.binary_cross_entropy_with_logits(updated_logits, z_next_target.detach())
    
    # 2. VQ Commitment Loss (drives logits close to discrete targets)
    loss_commit = F.mse_loss(updated_logits, z_next_pred.detach())
    
    # 3. Codebook Entropy Loss
    loss_div = F.mse_loss(z_next_pred.mean(dim=0), torch.full((QUANT_DIM,), 0.5))

    total_loss = 10.0 * loss_rec + 5.0 * loss_dyn + 1.0 * loss_commit + 2.0 * loss_div

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    scheduler.step()

    if (epoch + 1) % 500 == 0 or epoch == 0:
        current_lr = scheduler.get_last_lr()[0]
        unique_st, colls, t_acc = evaluate_metrics(model)
        print(f"{epoch+1:6d} | {total_loss.item():8.4f} | {current_lr:8.6f} | {unique_st:9d}/{TOTAL_STATES} | {colls:10d} | {t_acc:13.2f}%")

print("\nTraining Complete!\n")


# ==========================================
# 4. FULL STATE DIVERSITY AUDIT
# ==========================================
model.eval()

unique_codes = {}
for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        x = position_to_grid(r, c).unsqueeze(0)
        with torch.no_grad():
            _, z = model.encode(x)
        b_str = to_bits(z)
        unique_codes[b_str] = (r, c)

print("📊 [STATE DIVERSITY AUDIT]")
print(f"  Unique Bitstrings Generated: {len(unique_codes)} / {GRID_SIZE * GRID_SIZE} grid positions.")
if len(unique_codes) == GRID_SIZE * GRID_SIZE:
    print("  ✅ ZERO COLLAPSE: All 64 grid positions have 100% distinct bitstrings!\n")
else:
    print(f"  ⚠️ WARNING: {GRID_SIZE * GRID_SIZE - len(unique_codes)} positions share bitstrings!\n")

# ==========================================
# 5. VERIFICATION & PROOF SCENARIOS
# ==========================================
def compute_hash(bits): return hashlib.sha256("".join(bits).encode()).hexdigest()

actions = [1, 1, 3, 3]  # Down, Down, Right, Right
trace_tensors, trace_bits = [], []

r, c = 0, 0
x0 = position_to_grid(r, c).unsqueeze(0)
with torch.no_grad():
    _, z0 = model.encode(x0)
    z0 = z0.squeeze(0)

trace_tensors.append(z0)
trace_bits.append(to_bits(z0))

curr_z = z0.unsqueeze(0)
for a in actions:
    r, c = step_env(r, c, a)
    a_oh = F.one_hot(torch.tensor([a], dtype=torch.long), NUM_ACTIONS).float()
    with torch.no_grad():
        _, z_next = model.step(curr_z, a_oh)
        z_next = z_next.squeeze(0)
    trace_tensors.append(z_next)
    trace_bits.append(to_bits(z_next))
    curr_z = z_next.unsqueeze(0)

print("Generated topkhotSTE Bitstrings along Trajectory:")
for i, b in enumerate(trace_bits):
    print(f"  Step {i}: {b} (Active bits: {b.count('1')})")

def verify(committed_hash, bits, tensors, acts):
    print("\n🔍 [VERIFIER] Checking Proof...")
    if compute_hash(bits) != committed_hash:
        print("❌ REJECTED: Hash mismatch!")
        return False
    print("  ✓ Cryptographic Hash Commitment Valid.")

    for t in range(len(acts)):
        a_oh = F.one_hot(torch.tensor([acts[t]], dtype=torch.long), NUM_ACTIONS).float()
        with torch.no_grad():
            _, z_pred = model.step(tensors[t].unsqueeze(0), a_oh)
        if to_bits(z_pred) != bits[t+1]:
            print(f"❌ REJECTED: Transition error at step {t}->{t+1}!")
            return False
    print("  ✓ All Discrete Transitions match Model predictions exactly!")
    print("✅ PROOF ACCEPTED: Trajectory is computationally valid.\n")
    return True

print("\n--- SCENARIO A: Honest Prover ---")
honest_hash = compute_hash(trace_bits)
verify(honest_hash, trace_bits, trace_tensors, actions)

print("--- SCENARIO B: Cheating Prover (Teleport Attack Step 1 -> Step 3) ---")
cheat_tensors = [z.clone() for z in trace_tensors]
cheat_bits = list(trace_bits)
cheat_tensors[1] = cheat_tensors[3].clone()
cheat_bits[1] = trace_bits[3]

print("[Attack 1 - Original Hash Commitment]:")
verify(honest_hash, cheat_bits, cheat_tensors, actions)

print("[Attack 2 - Updated Hash Commitment]:")
verify(compute_hash(cheat_bits), cheat_bits, cheat_tensors, actions)

#exponential proofs of work via leaf proving