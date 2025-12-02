import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import modules
import quantizer

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')
# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training Config
BATCH_SIZE = 16   
AGENTS_PER_WORLD = 16
TOTAL_BATCH = BATCH_SIZE * AGENTS_PER_WORLD 
WIDTH, HEIGHT = 96, 96
VIEW_DIST = 4
VIEW_SIZE = (2 * VIEW_DIST + 1) # 9
PIXELS = VIEW_SIZE * VIEW_SIZE # 81

# Dimensions
DIM_L1 = 64   
DIM_L2 = 128  
DIM_L3 = 256  
VOCAB_SIZE = 64 

# --- Helper Modules ---

class Block(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.0, causal=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.causal = causal

    def forward(self, x, context=None):
        kv = context if context is not None else x
        x_norm = self.ln1(x)
        kv_norm = self.ln1(kv)
        attn_out, _ = self.attn(x_norm, kv_norm, kv_norm)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x

class SpatialPatcher(nn.Module):
    def __init__(self, dim_in, dim_out, size_in):
        super().__init__()
        self.size_in = size_in 
        stride = 2 if size_in > 3 else 1
        self.compressor = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=stride, padding=1),
            nn.GELU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1)
        )
        dummy = torch.zeros(1, dim_in, size_in, size_in)
        with torch.no_grad():
            out_shape = self.compressor(dummy).shape
            self.size_out = out_shape[2]
            
        self.expander = nn.Linear(dim_out, dim_in)

    def abstract_up(self, x):
        b, seq, c = x.shape
        h = w = int(np.sqrt(seq))
        img = x.transpose(1, 2).view(b, c, h, w)
        compressed = self.compressor(img) 
        flat = compressed.flatten(2).transpose(1, 2)
        return flat, x

    def abstract_down(self, x, residual):
        # Interpolate back to residual size
        # x: [B, Seq_small, Dim]
        x_p = x.transpose(1, 2) # [B, Dim, Seq]
        target_len = residual.shape[1]
        
        up = F.interpolate(x_p, size=target_len, mode='linear', align_corners=False)
        up = up.transpose(1, 2)
        
        return self.expander(up) + residual

class Quantizer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        # Project raw input (e.g. 81 pixels) to Latent Dim
        self.to_emb = nn.Linear(dim_in, dim_out)
    
    def forward(self, x):
        z_e = self.to_emb(x)
        
        # Apply the custom autograd function
        z_q = quantizer.ThresHot.apply(z_e)
        
        # Activity penalty (Sparsity regularization)
        # Encourages the representation to be sparse (zeros are cheap)
        loss = torch.mean(z_q) * 0.1 
        
        return z_q, loss

# --- The Brain ---
class HNet(nn.Module):
    def __init__(self, vocab_size): # vocab_size is kept for compatibility but effectively unused/replaced
        super().__init__()
        
        self.mem_block_size = 16
        mem_dim = DIM_L3
        
        # --- EDITED ---
        # Input: PIXELS (81) -> Latent: DIM_L1 (64)
        self.vq = Quantizer(PIXELS, DIM_L1)
        
        # Removed self.token_embedder (z_q is already the embedding)
        
        self.pos_embedder = nn.Parameter(torch.randn(1, PIXELS, DIM_L1) * 0.02)
        
        self.sLayers = nn.ModuleList([
            SpatialPatcher(DIM_L1, DIM_L2, 9), 
            SpatialPatcher(DIM_L2, DIM_L3, 5), 
        ])
        
        self.innerb = Block(DIM_L3, 4)
        self.innerb2 = Block(DIM_L3, 4)
        
        self.register_buffer('memory', torch.randn(1, self.mem_block_size, mem_dim)*0.02)
        self.memory_frozen = False
        self.cmem = True
        
        memconf = {'n_embd': DIM_L3, 'n_head': 4, 'dropout': 0.0, 'bias':False }
        self.dreamer = modules.Dreamer(mem_dim, causal=self.cmem, **memconf)
        self.memory_selector = Block(DIM_L3, 4)
        
        # --- EDITED ---
        # Predict the sparse vector (DIM_L1) instead of logits (vocab_size)
        self.head_recon = nn.Linear(DIM_L1, DIM_L1) 
        
        self.head_actor = nn.Linear(DIM_L3, 5) 
        self.head_mod = nn.Linear(DIM_L3, 9)   
        self.head_critic = nn.Linear(DIM_L3, 1)
        self.head_social = nn.Linear(DIM_L3, 1) 

    def mem_form(self, x):
        mem = self.memory_selector(x)[:, :self.mem_block_size, :]
        return self.dreamer(mem)
    
    def mem_update(self, update, malpha=0.01):
        if self.training and not self.memory_frozen:
            with torch.no_grad():
                self.memory = (self.memory+update*malpha)

    
    def mem_center(self, x):
        cb,ct,cc = x.shape
        x1 = x
        x = torch.cat([self.memory.expand(cb,-1,-1), x],dim=1)
        x = self.innerb(x)
        x = self.innerb2(x)
        
        if not self.training:
            return x[:,self.mem_block_size:,:].contiguous()
        
        malpha = 1.0
        mh1 = ((self.memory+self.mem_form(x[:cb//2,...])*malpha)).expand(cb//2,-1,-1)
        mh2 = ((self.memory+self.mem_form(x[cb//2:,...])*malpha)).expand(cb//2,-1,-1)
        mem2 = torch.cat([mh2,mh1],dim=0)
        x = torch.cat([mem2, x1],dim=1) 
        x = self.innerb(x)
        x = self.innerb2(x)
        
        self.mem_update(self.mem_form(x),malpha)
        x = x[:,self.mem_block_size:,:].contiguous()
        return x   

    def forward(self, x_continuous):
        # --- EDITED ---
        # x_continuous shape: [Batch, Seq_Len, PIXELS] or [Batch, PIXELS]
        z_q, vq_loss = self.vq(x_continuous)
        
        # Directly use the sparse latent + pos embedding
        x = z_q + self.pos_embedder
        
        residuals = []
        for layer in self.sLayers:
            residuals.append(x)
            x, _ = layer.abstract_up(x)
            
        x = self.mem_center(x)
        latent_state = x.mean(dim=1) 
        
        for i in reversed(range(len(self.sLayers))):
            x = self.sLayers[i].abstract_down(x, residuals[i])
            
        recon_logits = self.head_recon(x)
        actor_logits = self.head_actor(latent_state)
        mod_logits = self.head_mod(latent_state)
        value_pred = self.head_critic(latent_state)
        social_pred = self.head_social(latent_state)
        
        return {
            'recon_logits': recon_logits,
            'actor_logits': actor_logits,
            'mod_logits': mod_logits,
            'value_pred': value_pred,
            'social_pred': social_pred,
            'vq_loss': vq_loss,
            'target_sparse': z_q # Return the sparse vector as target
        }

class ParallelWorlds:
    def __init__(self, batch_size, w, h):
        self.bs = batch_size
        self.w = w
        self.h = h
        self.state = torch.rand(batch_size, 1, h, w, device=DEVICE)
        self.kernel = torch.ones(1, 1, 3, 3, device=DEVICE)
        self.kernel[0, 0, 1, 1] = 0.0
        
        # Pre-compute offsets
        r = torch.arange(-VIEW_DIST, VIEW_DIST+1, device=DEVICE)
        vy, vx = torch.meshgrid(r, r, indexing='ij')
        self.view_y_off = vy.reshape(1, 1, VIEW_SIZE, VIEW_SIZE)
        self.view_x_off = vx.reshape(1, 1, VIEW_SIZE, VIEW_SIZE)
        
        mr = torch.arange(-1, 2, device=DEVICE)
        my, mx = torch.meshgrid(mr, mr, indexing='ij')
        self.mod_y_off = my.reshape(1, 9)
        self.mod_x_off = mx.reshape(1, 9)

    def step(self, state):
        # Functional Step (Stateless)
        padded = F.pad(state, (1, 1, 1, 1), mode='circular')
        sum_neighbors = F.conv2d(padded, self.kernel)
        density = sum_neighbors / 8.0
        
        x_c = torch.linspace(0, 4*3.14159, self.w, device=DEVICE).view(1, 1, 1, -1)
        center = 0.3 + 0.1 * torch.sin(x_c).expand(self.bs, 1, self.h, self.w)
        
        dist = torch.abs(density - center)
        target = torch.sigmoid((0.15 - dist) * 20.0)
        
        new_state = state * 0.85 + target * 0.15
        return torch.clamp(new_state, 0.0, 1.0)

    def get_views(self, state, agents_pos):
        B, N, _ = agents_pos.shape
        cx = agents_pos[:, :, 0].view(B, N, 1, 1)
        cy = agents_pos[:, :, 1].view(B, N, 1, 1)
        
        grid_x = (cx + self.view_x_off) % self.w
        grid_y = (cy + self.view_y_off) % self.h
        
        b_idx = torch.arange(B, device=DEVICE).view(B, 1, 1, 1).expand(B, N, VIEW_SIZE, VIEW_SIZE)
        views = state[b_idx, 0, grid_y, grid_x]
        return views.view(B*N, -1) # Flatten pixel dims for VQ [Total, 81]

    def modify_worlds(self, state, agents_pos, mod_logits):
        # Differentiable Modification
        B, N, _ = agents_pos.shape
        patches = torch.sigmoid(mod_logits)
        
        cx = agents_pos[:, :, 0].view(-1, 1) 
        cy = agents_pos[:, :, 1].view(-1, 1)
        
        target_x = (cx + self.mod_x_off) % self.w
        target_y = (cy + self.mod_y_off) % self.h
        
        b_idx = torch.arange(B, device=DEVICE).repeat_interleave(N).view(-1, 1).expand(-1, 9)
        
        # Clone state to avoid in-place errors in backprop
        next_state = state.clone()
        
        current_vals = state[b_idx, 0, target_y, target_x]
        new_vals = current_vals * 0.5 + patches * 0.5
        
        next_state.index_put_(
            (b_idx, torch.zeros_like(b_idx), target_y, target_x), 
            new_vals, 
            accumulate=False
        )
        return next_state

# --- Training Loop ---

class BatchManager:
    def __init__(self):
        self.worlds = ParallelWorlds(BATCH_SIZE, WIDTH, HEIGHT)
        self.brain = HNet(VOCAB_SIZE).to(DEVICE)
        self.brain = torch.compile(self.brain, mode="reduce-overhead")
        
        self.optimizer = torch.optim.AdamW(self.brain.parameters(), lr=2e-4, fused=True)
        self.scaler = torch.amp.GradScaler('cuda')

        self.positions = torch.randint(VIEW_DIST, WIDTH-VIEW_DIST, 
                                     (BATCH_SIZE, AGENTS_PER_WORLD, 2), device=DEVICE)
        
        self.moves_tensor = torch.tensor([(0,0), (-1,0), (1,0), (0,-1), (0,1)], device=DEVICE)

    def generate_time_scape(self, start_state, agents_pos):
        # The "Scanner": Simulate future trajectory
        trajectory = []
        curr_state = start_state
        
        # We assume agents stay still during forecast (or we could model momentum)
        # This gives the "Inertial" prediction
        for _ in range(10):
            curr_state = self.worlds.step(curr_state)
            views = self.worlds.get_views(curr_state, agents_pos)
            trajectory.append(views)
            
        # Stack: [SIM_STEPS, Total, 81] -> [Total, SIM_STEPS, 81]
        return torch.stack(trajectory).transpose(0, 1)
    def train_step(self):
        views = self.worlds.get_views(self.positions)
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = self.brain(views)
            
            move_probs = F.softmax(out['actor_logits'], dim=-1)
            moves_idx = torch.multinomial(move_probs, 1).squeeze()
            
            moves = self.moves_tensor[moves_idx]
            flat_pos = self.positions.view(-1, 2) + moves
            flat_pos[:, 0] %= WIDTH
            flat_pos[:, 1] %= HEIGHT
            self.positions = flat_pos.view(BATCH_SIZE, AGENTS_PER_WORLD, 2)
            
            self.worlds.modify_worlds(self.positions, out['mod_logits'].float())
            self.worlds.step()
            
            future_views = self.worlds.get_views(self.positions)
            
            with torch.no_grad():
                future_out = self.brain(future_views) 
                
                # --- EDITED ---
                # Get the sparse representation of the future
                future_sparse, _ = self.brain.vq(future_views)
                
                # Calculate reconstruction error (BCE on the sparse vector)
                # out['recon_logits'] shape: [B, 1, DIM_L1], future_sparse shape: [B, 1, DIM_L1]
                # We want prediction error per sample
                pred_error = F.binary_cross_entropy_with_logits(
                    out['recon_logits'], 
                    future_sparse, 
                    reduction='none'
                ).mean(dim=-1).mean(dim=-1) # Mean over features and sequence
                
                r_oracle = torch.exp(-pred_error)
                
                r_chaotic = torch.tanh(future_views.std(dim=1).squeeze() * 10.0)
                r_complainer = torch.exp(-future_out['vq_loss'])

                all_agents_happiness = future_out['social_pred'].detach().squeeze()
                
                P = self.positions.float()
                diff = torch.abs(P.unsqueeze(2) - P.unsqueeze(1))
                diff = torch.min(diff, torch.tensor([WIDTH, HEIGHT], device=DEVICE) - diff)
                dist_sq = (diff ** 2).sum(dim=-1)
                mask = (dist_sq < (VIEW_DIST * 2)**2) & (dist_sq > 0.1)
                
                neighbor_happiness = all_agents_happiness.view(BATCH_SIZE, 1, AGENTS_PER_WORLD).expand(-1, AGENTS_PER_WORLD, -1)
                r_helper_raw = (neighbor_happiness * mask.float()).sum(dim=2).flatten()
                r_helper = torch.sigmoid(r_helper_raw) + 0.5

                r_base = r_oracle * r_chaotic * r_complainer
                reward = r_base * r_helper
            
            # --- LEARNING ---
            
            # --- EDITED ---
            # A. Reconstruction Loss (BCEWithLogits against the sparse target)
            # Re-calculate future_sparse with gradients if we want to drag gradients through (usually not for VQ/Sparse targets)
            # Here we treat the future sparse state as a fixed target.
            target_sparse = future_sparse.detach()
            loss_recon = F.binary_cross_entropy_with_logits(out['recon_logits'], target_sparse)
            
            val_pred = out['value_pred'].squeeze()
            loss_critic = F.mse_loss(val_pred, reward)
            
            advantage = reward - val_pred.detach()
            log_probs = torch.log(move_probs.gather(1, moves_idx.unsqueeze(1)).squeeze() + 1e-8)
            loss_actor = -(log_probs * advantage).mean()
            
            my_predicted_happiness = out['social_pred'].squeeze()
            loss_social = F.mse_loss(my_predicted_happiness, reward)
            
            total_loss = loss_recon + loss_critic + loss_actor + out['vq_loss'] + loss_social

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.brain.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return self.worlds.state, loss_recon.item()


sim = BatchManager()
TOTAL_STEPS = 5000

print(f"Starting Offline Training: {TOTAL_STEPS} steps, {BATCH_SIZE} worlds...")

t0 = time.time()
for i in range(1, TOTAL_STEPS + 1):
    # Run a step
    _, loss = sim.train_step()
    
    # Report
    if i % 1000 == 0:
        t1 = time.time()
        print(f"Batch {i}/{TOTAL_STEPS} | Recon Loss: {loss:.4f}| time: {t1-t0:.4f}")
        t0 = time.time()

print("Training Complete.")

# Optional: Visualize the result after training
fig, ax = plt.subplots(figsize=(6, 6))
img = ax.imshow(sim.worlds.state[0, 0].cpu().numpy(), cmap='magma', vmin=0, vmax=1)
scat = ax.scatter([], [], c='cyan', s=10, alpha=0.8)
ax.axis('off')

def show_result(frame):
    # Run just one step per frame to see the trained behavior
    state, _ = sim.train_step()
    img.set_data(state[0, 0].cpu().numpy())
    scat.set_offsets(sim.positions[0].cpu().numpy())
    return img, scat

print("Showing trained agents...")
ani = FuncAnimation(fig, show_result, interval=50, blit=True)
plt.show()