import os
import urllib.request
import zipfile
import json
import glob
import random
import torch

def ensure_arc_dataset(data_dir="./arc_data"):
    if not os.path.exists(data_dir):
        print("📥 Downloading official ARC-AGI dataset...")
        url = "https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip"
        zip_path = "arc_dataset.zip"
        urllib.request.urlretrieve(url, zip_path)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(zip_path)
        print("✅ Download complete and extracted.")
    
    search_path = os.path.join(data_dir, "**", "*.json")
    task_files = glob.glob(search_path, recursive=True)
    task_files =[f for f in task_files if "data/training" in f.replace('\\', '/') or "data/evaluation" in f.replace('\\', '/')]
    
    return task_files
class BatchedARCEnv:
    def __init__(self, num_envs, agents_per_env=1, grid_size=16, dim=64, device='cuda'):
        self.B = num_envs
        self.A = agents_per_env # Locked to 1 for this architecture
        self.N = self.B * self.A 
        self.grid_size = grid_size
        self.dim = dim
        self.device = device
        
        # 16 Input Nodes (One for each 4x4 patch in the 16x16 grid)
        self.num_inputs = 16  
        
        self.phase_step = 0
        self.demo_duration = 1000 
        
        # Start fully "hungry" / maximum plasticity
        self.hunger = torch.ones((self.B, self.A), device=device)
        
        self.task_files = ensure_arc_dataset()
        self.env_tasks =[random.choice(self.task_files) for _ in range(self.B)]
        
        self.input_grid = torch.zeros((self.B, grid_size, grid_size), device=device)
        self.canvas_grid = torch.zeros((self.B, grid_size, grid_size), device=device)
        self.target_grid = torch.zeros((self.B, grid_size, grid_size), device=device)
        
        # For rendering the "cursor"
        self.last_paint_x = torch.zeros((self.B, self.A), device=device)
        self.last_paint_y = torch.zeros((self.B, self.A), device=device)
        self.last_paint_mask = torch.zeros((self.B, self.A), dtype=torch.bool, device=device)
        
        self._load_phase("train")

    def _load_phase(self, phase="train"):
        self.input_grid.zero_()
        self.canvas_grid.zero_()
        self.target_grid.zero_()
        
        for b in range(self.B):
            with open(self.env_tasks[b], 'r') as f:
                task_data = json.load(f)
            
            if phase == "train":
                example = random.choice(task_data["train"])
            else:
                example = task_data["test"][0] 
                
            inp = torch.tensor(example["input"], dtype=torch.float32, device=self.device)
            out = torch.tensor(example["output"], dtype=torch.float32, device=self.device)
            
            # Center and safely crop if the ARC task exceeds our 16x16 arena
            h, w = min(inp.shape[0], 16), min(inp.shape[1], 16)
            start_y, start_x = (self.grid_size - h) // 2, (self.grid_size - w) // 2
            self.input_grid[b, start_y:start_y+h, start_x:start_x+w] = inp[:h, :w]
            
            h_o, w_o = min(out.shape[0], 16), min(out.shape[1], 16)
            start_y_o, start_x_o = (self.grid_size - h_o) // 2, (self.grid_size - w_o) // 2
            self.target_grid[b, start_y_o:start_y_o+h_o, start_x_o:start_x_o+w_o] = out[:h_o, :w_o]
        
    def get_vision(self):
        # Slice the 16x16 grids into sixteen 4x4 spatial patches. Shape becomes: (B, 16 patches, 16 pixels)
        v_in = self.input_grid.unfold(1, 4, 4).unfold(2, 4, 4).contiguous().view(self.B, 16, 16) / 9.0
        v_canv = self.canvas_grid.unfold(1, 4, 4).unfold(2, 4, 4).contiguous().view(self.B, 16, 16) / 9.0
        
        # Target Sensor is BLINDED in Test Phase
        if self.phase_step >= self.demo_duration:
            v_targ = torch.zeros_like(v_in)
        else:
            v_targ = self.target_grid.unfold(1, 4, 4).unfold(2, 4, 4).contiguous().view(self.B, 16, 16) / 9.0
            
        v_noise = torch.randn(self.B, 16, 16, device=self.device) * self.hunger.view(self.B, 1, 1)
        
        # Concatenate the 4 channels of 16 pixels into a perfect 64-dim vector per node
        inputs = torch.cat([v_in, v_canv, v_targ, v_noise], dim=2) # Shape: (B, 16, 64)
        
        # The agent expects shape (N, num_inputs, dim)
        return inputs.view(self.N, self.num_inputs, self.dim)
        
    def step(self, agent_outputs):
        self.phase_step += 1
        
        if self.phase_step == self.demo_duration:
            self._load_phase("test")
            
        out = agent_outputs.view(self.B, self.A, -1, self.dim)
        
        # 3 Output Nodes
        pos_x_node = self.num_inputs       # Node 16
        pos_y_node = self.num_inputs + 1   # Node 17
        action_node = self.num_inputs + 2  # Node 18
        
        bx = torch.arange(self.B, device=self.device).view(-1, 1).expand(self.B, self.A)
        
        # 1. GLOBAL CURSOR MECHANICS
        x_raw = out[:, :, pos_x_node, 0]
        y_raw = out[:, :, pos_y_node, 0]
        intent = out[:, :, action_node, 0]
        color_raw = out[:, :, action_node, 1]
        
        # Map continuous outputs to 16x16 grid indices
        x_idx = ((torch.tanh(x_raw) * 0.5 + 0.5) * 15.99).long().clamp(0, 15)
        y_idx = ((torch.tanh(y_raw) * 0.5 + 0.5) * 15.99).long().clamp(0, 15)
        color_idx = ((torch.tanh(color_raw) * 0.5 + 0.5) * 9.99).round().clamp(0, 9)
        
        paint_mask = intent > 0.0
        
        # Update Canvas
        self.canvas_grid[bx[paint_mask], y_idx[paint_mask], x_idx[paint_mask]] = color_idx[paint_mask].float()
        
        # Save for visualization
        self.last_paint_x = x_idx
        self.last_paint_y = y_idx
        self.last_paint_mask = paint_mask

        # 2. PROPER ARC ERROR CALCULATION
        wrong_pixels = (self.canvas_grid != self.target_grid).float().sum(dim=(1,2))
        target_fg_pixels = (self.target_grid != 0).float().sum(dim=(1,2)).clamp(min=1.0)
        
        # Normalizing error: 1.0 = baseline. >1.0 = bad painting. 0.0 = perfect.
        true_error = wrong_pixels / target_fg_pixels 
        
        # 3. INTERNAL FEEDBACK LOOP
        if self.phase_step < self.demo_duration:
            self.hunger = self.hunger * 0.9 + true_error.unsqueeze(1) * 0.1
        else:
            self.hunger = self.hunger * 0.9 

        return true_error.unsqueeze(1).expand(self.B, self.A)