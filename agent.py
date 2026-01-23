import os

from scipy import optimize

import optim
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import modules
import quantizer
import utils
#torch.autograd.set_detect_anomaly(True)
#torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training Config
BATCH_SIZE = 16   
AGENTS_PER_WORLD = 16
TOTAL_BATCH = BATCH_SIZE * AGENTS_PER_WORLD 
WIDTH, HEIGHT = 96, 96
VIEW_DIST = 4
VIEW_SIZE = (2 * VIEW_DIST + 1) 
PIXELS = VIEW_SIZE * VIEW_SIZE 

DIM_L1 = 64   

class Quantizer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        # Project raw input (e.g. 81 pixels) to Latent Dim
        self.to_emb = nn.Linear(dim_in, dim_out)
        self.cb = nn.Linear(dim_out, dim_out)
        self.max_k = dim_out
    
    def forward(self, x):
        z_e = self.to_emb(x)
        
        hot = quantizer.ThresHot.apply(z_e)
        k = utils.k_from_hot(hot,self.max_k)
        z_q = self.cb(hot / k.detach())
        return z_q

class AgentBrain(nn.Module):
    def __init__(self): 
        super().__init__()
        
        self.mem_block_size = 16
        self.mem_dim = DIM_L1
        
        # --- EDITED ---
        # Input: PIXELS (81) -> Latent: DIM_L1 (64)
        self.vq = Quantizer(PIXELS, DIM_L1)
        
        #self.to_emb = nn.Linear(PIXELS, DIM_L1)
        #self.ae = quantizer.Autoencoder(PIXELS,128,64,32,quantizer.ThresHot)

        #TODO:hierarchy as futurepredictor.

        self.register_buffer('memory', torch.randn(TOTAL_BATCH, self.mem_block_size, self.mem_dim)*0.02)
        self.memory_frozen = False
        self.cmem = True
        
        #memconf = {'n_embd': DIM_L1, 'n_head': 4, 'dropout': 0.0, 'bias':False }
        #self.dreamer = modules.Dreamer(self.mem_dim, causal=self.cmem, **memconf)
        self.memory_selector = modules.Block(DIM_L1, 4 ,0.0, False)
        
        self.memory_handler = nn.ModuleList(
            [modules.Block(DIM_L1, 4 ,0.0, False) for _ in range(4)]
        )
        
        self.head_recon = nn.Linear(DIM_L1, DIM_L1) 
        
        self.head_actor = nn.Linear(DIM_L1, 5) 
        self.head_mod = nn.Linear(DIM_L1, 1)   
        self.head_social = nn.Linear(DIM_L1, 1) 

        #self.head_pred = nn.Linear(DIM_L1, PIXELS) 
        fsteps = 3
        self.heads_pred = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(DIM_L1, PIXELS),
                nn.Sigmoid()
            ) for _ in range(fsteps)]
        )
        
        self.register_buffer('memory_history', torch.zeros(TOTAL_BATCH, self.mem_block_size, self.mem_dim, fsteps))
        
        self.register_buffer('history', torch.zeros(TOTAL_BATCH, PIXELS, fsteps))
   

    def forward(self, world):

        
        preds = []
        for i, head in enumerate(self.heads_pred):
            mem = self.memory_history[...,i]
            history = self.history[...,i]
            xh = self.vq(history).unsqueeze(1)
            xh = self.mem_center(xh, memory = mem, memup=False)
            xh = xh[:,-1,:]
            preds.append(head(xh).unsqueeze(1))
        preds = torch.cat(preds,dim=1)

        futures = []
        x = self.vq(world).unsqueeze(1)
        xh = self.mem_center(x, memory = self.memory, memup=False)
        xh = xh[:,-1,:]
        for i, head in enumerate(self.heads_pred):
            futures.append(self.vq(head(xh)).unsqueeze(1))
        futures = torch.cat(futures,dim=1)
        x, newmem = self.mem_center(x, memory = self.memory, predictions=futures)
        x = x[:,-1,:]
        #recon_logits = self.head_recon(x)


        actor_logits = self.head_actor(x)
        mod_logits = self.head_mod(x)
        social_pred = self.head_social(x)
        
        return {
            'predictions': preds,
            'actor_logits': actor_logits,
            'mod_logits': mod_logits,
            'social_pred': social_pred,
            'new_memory': newmem,
        }
    
    def update_history_buffers(self, world):
        with torch.no_grad():
            new_history = torch.cat([self.history[:, :, 1:], world.detach().unsqueeze(-1)], dim=-1)
            self.history = new_history
            
            oldmem = self.memory.clone().detach()
            new_memory_history = torch.cat([self.memory_history[:, :, :, 1:], oldmem.unsqueeze(-1)], dim=-1)
            self.memory_history = new_memory_history

    def mem_update(self, update, malpha=0.01):
        if self.training and not self.memory_frozen:
            with torch.no_grad():
                oldmem = self.memory.detach().clone()
                self.memory = utils.range_norm(oldmem * (1 - malpha) + update* malpha).clone()

    def mem_center(self, x, memory =None, predictions=None, memup = True):
        #cb,ct,cc = x.shape
        if predictions is not None:
            xmix = torch.cat([memory, x, predictions],dim=1)
        else:
            xmix = torch.cat([memory, x],dim=1)

        for b in self.memory_handler:
            xmix=b(xmix)
        
        if not self.training or not memup:
            return xmix[:, self.mem_block_size:, :].contiguous()
        
        malpha = 1.0
        newmem = self.memory_selector(xmix)[:, :self.mem_block_size, :]
        mem = utils.range_norm(memory + newmem*malpha)
        if predictions is not None:
            xmix = torch.cat([mem, x, predictions],dim=1)
        else:
            xmix = torch.cat([mem, x],dim=1)

        for b in self.memory_handler:
            xmix=b(xmix)
        
        newmem = self.memory_selector(xmix)[:, :self.mem_block_size, :]
        
        xmix = xmix[:, self.mem_block_size:, :].contiguous()
        return xmix, newmem

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
        self.brain = AgentBrain().to(DEVICE)
        # Compile can sometimes delay startup, optional depending on setup
        #self.brain = torch.compile(self.brain)
        self.optimizer = optim.Grms(
            params=self.brain.parameters(),
            lr=2e-4, 
            nesterov = True, 
            momentum=0.9,
            fused=True)
        
        #self.optimizer = torch.optim.AdamW(self.brain.parameters(), lr=2e-4, fused=True)
        self.scaler = None# torch.amp.GradScaler('cuda')

        self.positions = torch.randint(VIEW_DIST, WIDTH-VIEW_DIST, 
                                     (BATCH_SIZE, AGENTS_PER_WORLD, 2), device=DEVICE)
        
        self.moves_tensor = torch.tensor([(0,0), (-1,0), (1,0), (0,-1), (0,1)], device=DEVICE)

    def train_step(self):
        torch.compiler.cudagraph_mark_step_begin()
        # 1. Observe current state
        views = self.worlds.get_views(self.worlds.state, self.positions)
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            # 2. Forward pass
            out = self.brain(views)

            move_logits = out['actor_logits'] # [B*N, 5]
            #move_logits = torch.nan_to_num(move_logits, nan=0.0, posinf=10.0, neginf=-10.0)
            move_probs = F.softmax(move_logits.float(), dim=-1)

            moves_idx = torch.multinomial(move_probs, 1).squeeze()

            moves = self.moves_tensor[moves_idx]

            flat_pos = self.positions.view(-1, 2) + moves
            flat_pos[:, 0] %= WIDTH
            flat_pos[:, 1] %= HEIGHT
            self.positions = flat_pos.view(BATCH_SIZE, AGENTS_PER_WORLD, 2)

            self.worlds.modify_worlds(self.worlds.state, self.positions, out['mod_logits'].float())

            self.worlds.state = self.worlds.step(self.worlds.state)

            next_views = self.worlds.get_views(self.worlds.state, self.positions)

            pred_pixels = out['predictions'][:, :PIXELS] 

            loss_mse_per_sample = F.mse_loss(pred_pixels, next_views.unsqueeze(1).expand(pred_pixels.shape), reduction='none').mean(dim=1).mean(dim=-1)
            #loss_mse_per_sample = loss_mse_per_sample.clamp(max=10.0)

            loss_pred = loss_mse_per_sample.mean()
            predicted_self_loss = out['social_pred'].squeeze()
            #predicted_self_loss = torch.nan_to_num(predicted_self_loss, nan=0.0, posinf=10.0, neginf=-10.0)

            target_loss = loss_mse_per_sample.detach()

            loss_social = F.mse_loss(predicted_self_loss, target_loss)

            reward = -loss_mse_per_sample.detach()

            log_probs = torch.log(move_probs.gather(1, moves_idx.unsqueeze(1)).squeeze() + 1e-8)
            loss_actor = -(log_probs * reward).mean()
            total_loss = loss_pred + loss_social + loss_actor

            # 6. Backward Pass
            total_loss.backward()
            if self.scaler is None:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            else:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                #torch.nn.utils.clip_grad_norm_(self.brain.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            if self.brain.training:
                self.brain.update_history_buffers(views.detach())
                self.brain.mem_update(out['new_memory'].detach().clone())
            self.optimizer.zero_grad(set_to_none=True)

        return self.worlds.state, loss_pred.detach()


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