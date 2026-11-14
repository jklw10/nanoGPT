import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from datasets import load_dataset

from parts import quantizer, utils

import os
device          = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size      = 64
block_size      = 64      
n_embd          = 64
n_head          = 8
n_layer         = 4
num_nodes       = 4     
thinking_steps  = 4        
max_cycles      = 4        
gate_penalty    = 0.01

# Dynamic EBM Continuous Hyperparameters
untanh_slope    = 0.8
w_scale         = 0.8
c_moment        = 0.8
a_variance      = 0.0026

# Only run configurator if this file is executed as __main__ 
if __name__ == "__main__":
    configurator_file = 'configurator.py'
    exec(open('training_tools/configurator.py').read())

@dataclass
class SolverConfig:
    block_size: int     = globals().get('block_size', 64)
    vocab_size: int     = 256  
    n_embd: int         = globals().get('n_embd', 64)
    n_head: int         = globals().get('n_head', 8)
    n_layer: int        = globals().get('n_layer', 4)
    bias: bool          = False
    dropout: float      = 0.0
    num_nodes: int      = globals().get('num_nodes', 64)
    thinking_steps: int = globals().get('thinking_steps', 4)
    max_cycles: int     = globals().get('max_cycles', 4)
    gate_penalty: float = globals().get('gate_penalty', 0.01)
    
    use_regular_rnn: bool = False
    
    untanh_slope: float = globals().get('untanh_slope', 0.8)
    w_scale: float      = globals().get('w_scale', 0.8)
    
    UNTANH: torch.Tensor     = torch.tensor(globals().get('untanh_slope', 0.8))
    c_moment: torch.Tensor   = torch.tensor(globals().get('c_moment', 0.8))
    a_variance: torch.Tensor = torch.tensor(globals().get('a_variance', 0.0026))

# Safe UTF-8 / Custom byte decoder
def safe_decode(indices):
    if 'decode' in globals() and callable(decode):
        try:
            return decode(indices)
        except Exception:
            pass
            
    clean_indices = [max(0, min(255, int(i))) for i in indices]
    return bytes(clean_indices).decode('utf-8', errors='ignore')

def sparse_route(router, node_count, spr):
    probs = torch.sigmoid(router)
    mask = quantizer.topkhotSTE(router, int(math.sqrt(node_count)))
    sparse_probs = mask * probs
    sparse_probs = sparse_probs / (sparse_probs.sum(dim=-1, keepdim=True) + 1e-8)
    flat_probs = torch.ones_like(probs) / node_count
    ratio = spr 
    route = ((1.0-ratio) * sparse_probs) + (ratio * flat_probs)
    return route, 0.0, mask

def get_batch_prefix_causal_mask(T, prefix_lens, device):
    B = prefix_lens.size(0)
    mask = torch.zeros(B, T, T, device=device)
    clamped_prefix_lens = torch.clamp(prefix_lens, min=1, max=T)
    r = torch.arange(T, device=device)
    i_grid, j_grid = torch.meshgrid(r, r, indexing='ij')
    causal_cond = (j_grid > i_grid)
    prefix_lens_expanded = clamped_prefix_lens.view(B, 1, 1)
    past_prefix_i = (i_grid.unsqueeze(0) >= prefix_lens_expanded)
    past_prefix_j = (j_grid.unsqueeze(0) >= prefix_lens_expanded)
    final_cond = causal_cond.unsqueeze(0) & past_prefix_i & past_prefix_j
    mask = mask.masked_fill(final_cond, -float('inf'))
    return mask

class CNNInit(nn.Module):
    def __init__(self, n_embd, num_nodes, node_dim):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=n_embd, out_channels=n_embd, kernel_size=3, padding=1)
        self.proj = nn.Linear(n_embd, num_nodes * node_dim)
        
    def forward(self, x, prefix_lens):
        B, T, C = x.size()
        x_t = x.transpose(1, 2)  # (B, C, T)
        x_conv = F.gelu(self.conv(x_t))  # (B, C, T)
        
        pooled = []
        for b in range(B):
            plen = int(prefix_lens[b].item())
            plen = max(1, min(plen, T))
            item_pooled = x_conv[b, :, :plen].mean(dim=-1)
            pooled.append(item_pooled)
            
        pooled = torch.stack(pooled, dim=0)  # (B, C)
        out = self.proj(pooled)  # (B, num_nodes * node_dim)
        return out.view(B, -1, C)

class NodeParams3(nn.Module):
    def __init__(self, batch_size, num_nodes, in_dim, out_dim, device):
        super().__init__() 
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        
        base_weight = torch.randn(num_nodes, in_dim, out_dim, device=device) / math.sqrt(self.in_dim)
        self.weight = nn.Parameter(base_weight)
        self.register_buffer('error_ema', 0.01 + 0.01 * torch.randn(batch_size, num_nodes, out_dim, device=device))
        
        self.register_buffer('untanh_slope', torch.tensor(1.0, device=device))
        self.register_buffer('w_scale', torch.tensor(0.8, device=device))
        self.register_buffer('ema_speed', torch.tensor(0.2, device=device))
        self.register_buffer('rew_sens', torch.tensor(1.36, device=device))
        self.register_buffer('n_scale', torch.tensor(0.0, device=device))
        self.register_buffer('speed', torch.tensor(0.5, device=device))
        self.register_buffer('width', torch.tensor(0.01, device=device))

    def reset_state(self, batch_size, device):
        self.error_ema = 0.01 + 0.01 * torch.randn(batch_size, self.weight.shape[0], self.out_dim, device=device)

    def forward(self, x, step_counter):
        w_scale_dev = self.w_scale.to(x.device)
        untanh_slope_dev = self.untanh_slope.to(x.device)
        norm_weight = F.normalize(self.weight, p=2, dim=-2) * w_scale_dev
        raw_pred = torch.einsum('bnd,nde->bne', x, norm_weight)
        bound = math.sqrt(2.0 + math.sqrt(2.0))
        out = utils.buntanh(raw_pred, untanh_slope_dev, bound)
        return out
    
    def apply_hyperparams(self, config):
        if hasattr(config, 'UNTANH'):
            val = config.UNTANH if isinstance(config.UNTANH, torch.Tensor) else torch.tensor(config.UNTANH, device=self.device)
            self.register_buffer('untanh_slope', val.view(-1, 1, 1))
        elif hasattr(config, 'untanh_slope'):
            val = config.untanh_slope if isinstance(config.untanh_slope, torch.Tensor) else torch.tensor(config.untanh_slope, device=self.device)
            self.register_buffer('untanh_slope', val.view(-1, 1, 1))
            
        if hasattr(config, 'w_scale'):
            val = config.w_scale if isinstance(config.w_scale, torch.Tensor) else torch.tensor(config.w_scale, device=self.device)
            self.register_buffer('w_scale', val.view(-1, 1, 1))
        
        if hasattr(config, 'EMA_SPEED'):
            self.register_buffer('ema_speed', (config.EMA_SPEED if isinstance(config.EMA_SPEED, torch.Tensor) else torch.tensor(config.EMA_SPEED, device=self.device)).view(-1, 1, 1))
            self.register_buffer('rew_sens', (config.FIXED_REWARD_SENSITIVITY if isinstance(config.FIXED_REWARD_SENSITIVITY, torch.Tensor) else torch.tensor(config.FIXED_REWARD_SENSITIVITY, device=self.device)).view(-1, 1, 1, 1))
            self.register_buffer('n_scale', (config.NOISE_SCALE if isinstance(config.NOISE_SCALE, torch.Tensor) else torch.tensor(config.NOISE_SCALE, device=self.device)).view(-1, 1, 1))
            self.register_buffer('speed', (config.speed if isinstance(config.speed, torch.Tensor) else torch.tensor(config.speed, device=self.device)).view(-1, 1, 1, 1))
            self.register_buffer('width', (config.width if isinstance(config.width, torch.tensor) else torch.tensor(config.width, device=self.device)).view(-1, 1, 1, 1))

class StandardNodeNet3(nn.Module):
    def __init__(self, batch_size, num_nodes, dim, device):
        super().__init__()
        self.N = num_nodes
        self.D = dim
        self.device = device
        
        self.w1 = NodeParams3(batch_size, num_nodes, dim, dim, device)
        self.router = NodeParams3(batch_size, num_nodes, dim, num_nodes, device)
        self.wnll = utils.wnormlearnloss(dim)
        self.raw_sprs = nn.Parameter(torch.tensor(0.0, device=device))  
        
        self.register_buffer('state', 1e-5 * torch.randn(batch_size, self.N, dim, device=device))
        self.register_buffer('output', 1e-5 * torch.randn(batch_size, self.N, dim, device=device))
        self.register_buffer('error_ema', torch.ones(batch_size, self.N, device=device))
        self.register_buffer('lp_ema', torch.zeros(batch_size, self.N, device=device))
        self.register_buffer('step_counter', torch.tensor(0.0, device=device))
        self.current_input_nodes = 0
        self.lp_alpha = 0.1 
        
        self.register_buffer('c_moment', torch.tensor(0.8, device=device))
        self.register_buffer('a_variance', torch.tensor(0.0026, device=device))

    def reset_state(self, batch_size, device):
        self.state = 1e-5 * torch.randn(batch_size, self.N, self.D, device=device)
        self.output = 1e-5 * torch.randn(batch_size, self.N, self.D, device=device)
        self.error_ema = torch.ones(batch_size, self.N, device=device)
        self.lp_ema = torch.zeros(batch_size, self.N, device=device)
        self.step_counter = torch.tensor(0.0, device=device)
        self.w1.reset_state(batch_size, device)
        self.router.reset_state(batch_size, device)
    
    def apply_hyperparams(self, config):
        if hasattr(config, 'c_moment'):
            val = config.c_moment if isinstance(config.c_moment, torch.Tensor) else torch.tensor(config.c_moment, device=self.device)
            self.register_buffer('c_moment', val.view(-1, 1, 1, 1))
        if hasattr(config, 'a_variance'):
            val = config.a_variance if isinstance(config.a_variance, torch.Tensor) else torch.tensor(config.a_variance, device=self.device)
            self.register_buffer('a_variance', val.view(-1, 1, 1))
            
        self.w1.apply_hyperparams(config)

    def forward(self, env_input, state_in=None, output_in=None, error_ema_in=None, lp_ema_in=None):
        self.step_counter.add_(1.0)
        batch_size, input_dim = env_input.shape
        num_input_nodes = math.ceil(input_dim / self.D)
        
        current_state = state_in if state_in is not None else self.state
        current_output = output_in if output_in is not None else self.output
        current_error_ema = error_ema_in if error_ema_in is not None else self.error_ema
        current_lp_ema = lp_ema_in if lp_ema_in is not None else self.lp_ema
        
        prediction = self.w1(current_state, self.step_counter) 
        router = self.router(current_state, self.step_counter)
        
        curiosity_bias = F.relu(current_lp_ema).unsqueeze(1) * 5.0 
        route_logits = router + curiosity_bias
        
        sprs_dev = torch.sigmoid(self.raw_sprs)
        route, aux, mask = sparse_route(route_logits, self.N, sprs_dev)
        
        current_var = route.var(dim=[-1], keepdim=True)
        a_variance_dev = self.a_variance.to(env_input.device)
        vd = F.relu(a_variance_dev - current_var)
        
        V = current_output
        base_noise = torch.sqrt(vd) * torch.randn_like(V)
        
        boredom_scale = torch.exp(-50.0 * F.relu(current_lp_ema)).unsqueeze(-1) 
        boredom_noise = base_noise * boredom_scale
        
        target = torch.matmul(route, V) + boredom_noise
        target = utils.softsign(target)

        pad_len = num_input_nodes * self.D - input_dim
        if pad_len > 0:
            env_input_padded = F.pad(env_input, (0, pad_len))
        else:
            env_input_padded = env_input
            
        env_input_reshaped = env_input_padded.view(batch_size, num_input_nodes, self.D)
        
        if num_input_nodes < self.N:
            target = torch.cat([env_input_reshaped, target[:, num_input_nodes:, :]], dim=1)
        else:
            target = env_input_reshaped

        loss_w1 = F.mse_loss(prediction, target, reduction='none')

        with torch.no_grad():
            current_error = loss_w1.mean(dim=-1).detach()
            current_lp = current_error_ema - current_error
            next_error_ema = current_error_ema * (1.0 - self.lp_alpha) + current_error * self.lp_alpha
            next_lp_ema = current_lp_ema * (1.0 - self.lp_alpha) + current_lp * self.lp_alpha

        total_loss = (loss_w1).sum() + aux
        
        boredom_scale = torch.exp(-50.0 * torch.relu(next_lp_ema)).unsqueeze(-1) 
        errm = torch.sqrt(boredom_scale) * 0.01
        error_pass = torch.randn_like(target) * errm
        target = target + error_pass

        # Maintain raw gradient graph copies for return outputs, but store detached representations in buffers
        returned_state = target
        returned_output = prediction

        next_output = prediction.detach()
        next_state = target.detach()

        if state_in is None:
            self.error_ema = next_error_ema
            self.lp_ema = next_lp_ema
            self.output = next_output
            self.state = next_state

        return total_loss, returned_state, returned_output, next_error_ema, next_lp_ema

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias_flag):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias_flag else None
       
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class ScanCombine(nn.Module):
    def __init__(self, node_net, proj_state, proj_output, proj_error, proj_lp, num_nodes, dim):
        super().__init__()
        self.node_net = node_net
        self.proj_state = proj_state
        self.proj_output = proj_output
        self.proj_error = proj_error
        self.proj_lp = proj_lp
        self.num_nodes = num_nodes
        self.dim = dim

    def forward(self, left, right):
        B, L, C = left.shape
        left_flat = left.reshape(B * L, C)
        right_flat = right.reshape(B * L, C)
        
        # Project the single carried representation into separate multi-node spaces
        state_in = self.proj_state(left_flat).view(B * L, self.num_nodes, self.dim)
        output_in = self.proj_output(left_flat).view(B * L, self.num_nodes, self.dim)
        error_ema_in = self.proj_error(left_flat)
        lp_ema_in = self.proj_lp(left_flat)
        
        _, _, next_output, _, _ = self.node_net(
            right_flat, 
            state_in=state_in, 
            output_in=output_in,
            error_ema_in=error_ema_in,
            lp_ema_in=lp_ema_in
        )
        
        # The first node's prediction acts as the single carry value for the next tree layer
        out = next_output[:, 0, :]
        return out.view(B, L, C)
    
class GSM8KSolverEBM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        
        print("==> Initializing solver with Cognitive Node Network (StandardNodeNet3)")
        self.node_net = StandardNodeNet3(
            batch_size=batch_size,
            num_nodes=config.num_nodes,
            dim=config.n_embd,
            device=device
        )
        self.node_net.apply_hyperparams(config)
            
        # Register the projections to map single-vector representations to multi-node spaces
        self.proj_state = nn.Linear(config.n_embd, config.num_nodes * config.n_embd, bias=config.bias)
        self.proj_output = nn.Linear(config.n_embd, config.num_nodes * config.n_embd, bias=config.bias)
        self.proj_error = nn.Linear(config.n_embd, config.num_nodes, bias=config.bias)
        self.proj_lp = nn.Linear(config.n_embd, config.num_nodes, bias=config.bias)
        
        nn.init.normal_(self.proj_state.weight, std=0.02)
        nn.init.normal_(self.proj_output.weight, std=0.02)
        nn.init.normal_(self.proj_error.weight, std=0.02)
        nn.init.normal_(self.proj_lp.weight, std=0.02)
        
        self.ident = nn.Parameter(torch.zeros(config.n_embd))
        self.ln_f = LayerNorm(config.n_embd, config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight
        
    def forward(self, idx, prefix_lens, targets=None, mode='scan'):
        """
        Supports two primary execution modes:
          - 'scan': Parallel Blelloch tree scan using pscan2 (O(log T) depth).
          - 'rnn' : True sequential recurrent execution path (O(T) depth).
        """
        B, T = idx.size()
        x = self.wte(idx)
        clamped_prefix_lens = torch.clamp(prefix_lens, min=1, max=T)
        
        if mode == 'scan':
            combine_fn = ScanCombine(self.node_net, self.proj_state, self.proj_output, self.proj_error,self.proj_lp, self.config.num_nodes, self.config.n_embd)
            pred_embs = utils.pscan2(x, combine_fn, self.ident)
        elif mode == 'rnn':
            self.node_net.reset_state(B, device=idx.device)
            # Start with the identity vector as the sequence-carried representation
            current_val = self.ident.view(1, -1).expand(B, -1)  # (B, C)
            
            pred_embs = []
            for t in range(T):
                env_input = x[:, t, :]
                
                # Mirror the exact projection structure used in ScanCombine
                state_in = self.proj_state(current_val).view(B, self.config.num_nodes, self.config.n_embd)
                output_in = self.proj_output(current_val).view(B, self.config.num_nodes, self.config.n_embd)
                error_ema_in = self.proj_error(current_val)
                lp_ema_in = self.proj_lp(current_val)
                
                _, _, next_output, _, _ = self.node_net(
                    env_input,
                    state_in=state_in,
                    output_in=output_in,
                    error_ema_in=error_ema_in,
                    lp_ema_in=lp_ema_in
                )
                
                # Update current_val to the first node's output for the next sequence step
                current_val = next_output[:, 0, :]
                pred_embs.append(current_val)
                
            pred_embs = torch.stack(pred_embs, dim=1)
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        logits = self.lm_head(self.ln_f(pred_embs))
        
        loss = None
        if targets is not None:
            r = torch.arange(T, device=idx.device).unsqueeze(0)
            target_mask = (r >= clamped_prefix_lens.unsqueeze(1)).float()
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none').view(B, T)
            loss = (ce_loss * target_mask).sum() / (target_mask.sum() + 1e-8)
            
        return logits, loss

    def configure_optimizers(self, lr, wd):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': nodecay_params, 'weight_decay': 0.0},
            {'params': decay_params,   'weight_decay': wd},
        ]
        return torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.999))

class HuggingFaceGSM8KDataset:
    def __init__(self, block_size=384, workspace_size=32, device='cpu', split="train"):
        self.block_size = block_size
        self.workspace_size = workspace_size
        self.device = device
        self.vocab_size = 256
        
        hf_split = "train" if split == "train" else "test"
        print(f"Retrieving GSM8K '{hf_split}' data from Hugging Face Hub...")
        self.dataset = load_dataset("openai/gsm8k", "main", split=hf_split)
        print(f"Dataset successfully loaded. Total items: {len(self.dataset)}")
        
    def get_batch(self, batch_size=32):
        x_batch = []
        y_batch = []
        prefix_lens = []
        
        indices = np.random.randint(0, len(self.dataset), size=batch_size)
        
        for idx in indices:
            row = self.dataset[int(idx)]
            question = row['question']
            answer = row['answer']
            prompt = f"Question: {question}\nAnswer: "
            
            prompt_bytes = prompt.encode('utf-8', errors='ignore')
            workspace_bytes = b"#" * self.workspace_size
            answer_bytes = answer.encode('utf-8', errors='ignore')
            
            max_prompt_bytes = self.block_size - self.workspace_size - 64
            if len(prompt_bytes) > max_prompt_bytes:
                prompt_bytes = b"Question: ... " + prompt_bytes[-max_prompt_bytes:]
            
            full_seq_bytes = prompt_bytes + workspace_bytes + answer_bytes
            encoded = list(full_seq_bytes)
            
            if len(encoded) < self.block_size:
                encoded = encoded + [32] * (self.block_size - len(encoded))
            else:
                encoded = encoded[:self.block_size]
                
            prefix_len = len(prompt_bytes) + self.workspace_size
            target = encoded[1:] + [32]
            
            x_batch.append(torch.tensor(encoded, dtype=torch.long))
            y_batch.append(torch.tensor(target, dtype=torch.long))
            prefix_lens.append(prefix_len)
            
        x = torch.stack(x_batch).to(self.device)
        y = torch.stack(y_batch).to(self.device)
        prefix_lens_tensor = torch.tensor(prefix_lens, dtype=torch.long, device=self.device)
        
        return x, y, prefix_lens_tensor

@torch.no_grad()
def generate_math_solution(model, prompt, workspace_size=32, max_new_tokens=128, temperature=0.7, mode='rnn'):
    model.eval()
    
    prompt_bytes = prompt.encode('utf-8', errors='ignore')
    workspace_bytes = b"#" * workspace_size
    
    max_prompt_len = model.config.block_size - workspace_size - max_new_tokens
    if len(prompt_bytes) > max_prompt_len:
        prompt_bytes = b"Question: ... " + prompt_bytes[-max_prompt_len:]
        
    full_prompt_bytes = prompt_bytes + workspace_bytes
    indices = list(full_prompt_bytes)
    x = torch.tensor([indices], dtype=torch.long, device=device)
    
    prefix_len_val = len(indices)
    prefix_lens_tensor = torch.tensor([prefix_len_val], dtype=torch.long, device=device)
    
    for _ in range(max_new_tokens):
        logits, _ = model(x, prefix_lens=prefix_lens_tensor, mode=mode)
        
        next_token_logits = logits[:, -1, :] / temperature
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        x = torch.cat((x, next_token), dim=1)
        if next_token.item() == 10:  # Stop on newline (\n)
            break
            
    generated_indices = x[0, prefix_len_val:].tolist()
    return safe_decode(generated_indices)

if __name__ == "__main__":
    os.makedirs('ckpt', exist_ok=True)
    
    dataset = "openwebtext"
    decode, vocab_size = utils.get_meta(dataset, always_byte=True)
    owt_get_batch = utils.get_get_batch(block_size, batch_size, dataset, device, always_byte=True)
    
    gptconf = SolverConfig()
    gptconf.vocab_size = vocab_size
    
    model = GSM8KSolverEBM(gptconf).to(device)
    optimizer = model.configure_optimizers(lr=1e-3, wd=1e-1)
    
    gsm8k_dataset = HuggingFaceGSM8KDataset(block_size=block_size, workspace_size=32, device=device, split="train")
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    start_step = 0
    checkpoint_dir = 'ckpt'
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')] if os.path.exists(checkpoint_dir) else []
    
    if checkpoint_files:
        latest_ckpt = None
        max_step = -1
        for f in checkpoint_files:
            try:
                ckpt_path = os.path.join(checkpoint_dir, f)
                checkpoint = torch.load(ckpt_path, map_location=device)
                if 'step' in checkpoint and checkpoint['step'] > max_step:
                    max_step = checkpoint['step']
                    latest_ckpt = ckpt_path
            except Exception:
                pass
        
        if latest_ckpt:
            print(f"Found existing checkpoints. Resuming from {latest_ckpt} at step {max_step + 1}...")
            checkpoint = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_step = max_step + 1

    print("\n" + "="*50)
    print("PROFILING TIMINGS (1 step each)...")
    print("="*50)
    
    X_owt, Y_owt = owt_get_batch('train')
    prefix_lens_owt = torch.ones(X_owt.size(0), dtype=torch.long, device=device)
    
    # Profile Scan
    t_probe = time.time()
    _, _ = model(X_owt, prefix_lens_owt, Y_owt, mode='scan')
    print(f"Microphase [scan]: {(time.time() - t_probe)*1000:6.1f} ms")
    
    # Profile RNN
    t_probe = time.time()
    _, _ = model(X_owt, prefix_lens_owt, Y_owt, mode='rnn')
    print(f"Microphase [rnn] : {(time.time() - t_probe)*1000:6.1f} ms")
    print("="*50 + "\n")
    
    best_losses = {
        'pretrain': float('inf'),
        'task_finetune': float('inf')
    }

    def save_phase_checkpoint(phase_name, current_loss, step):
        if current_loss < best_losses[phase_name]:
            best_losses[phase_name] = current_loss
            checkpoint_path = f"ckpt/best_checkpoint_{phase_name}.pt"
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': current_loss,
            }, checkpoint_path)
            print(f"  [Checkpoint] Updated best checkpoint for Phase: {phase_name} (Loss: {current_loss:.4f})")

    print(f"Beginning training pipeline starting at step {start_step}...")
    t0 = time.time()
    total_steps = 35000
    
    for step in range(start_step, total_steps):
        model.train()
        
        # Determine dataset source and model phase
        if step < 30000:
            X, Y = owt_get_batch('train')
            prefix_lens = torch.ones(X.size(0), dtype=torch.long, device=device)
            phase_name = "pretrain"
            # Maintain a balance of scan-based parallel training and true RNN steps
            mode = 'rnn' if (step % 20 == 0) else 'scan'
        else:
            X, Y, prefix_lens = gsm8k_dataset.get_batch(batch_size=batch_size)
            phase_name = "task_finetune"
            mode = 'rnn' if (step % 20 == 0) else 'scan'
            
        logits, loss = model(X, prefix_lens, Y, mode=mode)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        # Log training statistics
        if step % 1000 == 0 or step == total_steps - 1:
            print(f"Step {step:5d} | Phase: {phase_name:<14s} | Mode: {mode:<5s} | Loss: {loss.item():.4f} | Batch Avg Prefix: {prefix_lens.float().mean().item():.1f} | Step Time: {(time.time() - t0)*1000:.0f}ms")
            save_phase_checkpoint(phase_name, loss.item(), step)
            t0 = time.time()
            
        # FOUL PLAY GUARD: Periodic evaluations (uses sequential RNN formulation)
        if step > 0 and step % 10000 == 0:
            sample_question = "Question: Weng earns $12 an hour for babysitting. Yesterday, she babysat for 5 hours. How much did she earn?\nAnswer: "
            solution = generate_math_solution(model, sample_question, workspace_size=32, max_new_tokens=48, mode='rnn')
            print(f"\n[Foul Play Check @ Step {step}]")
            print(f"Prompt: {sample_question.strip()}")
            print(f"RNN Decoded Output: {solution.strip()}")
            print("-" * 50 + "\n")
            t0 = time.time()
            sample_question = "Question: Weng earns $12 an hour for babysitting. Yesterday, she babysat for 5 hours. How much did she earn?\nAnswer: "
            solution = generate_math_solution(model, sample_question, workspace_size=32, max_new_tokens=48, mode='scan')
            print(f"\n[Foul Play Check @ Step {step}]")
            print(f"Prompt: {sample_question.strip()}")
            print(f"RNN Decoded Output: {solution.strip()}")
            print("-" * 50 + "\n")
            t0 = time.time()

# param searchdef
model = GSM8KSolverEBM
config = SolverConfig

config_space = {
    "untanh_slope": {
        "type": "dynamic",
        "scale": "linear",
        "low": 0.5,
        "high": 0.99,
        "default": 0.8
    },
    "w_scale": {
        "type": "dynamic",
        "scale": "linear",
        "low": 0.1,
        "high": 1.5,
        "default": 0.8
    },
    "c_moment": {
        "type": "dynamic",
        "scale": "linear",
        "low": 0.1,
        "high": 0.99,
        "default": 0.8
    },
    "a_variance": {
        "type": "dynamic",
        "scale": "log",       
        "low": 0.0001,
        "high": 0.01,
        "default": 0.0026
    },
    "dummy_scale": {
        "type": "static",
        "values": [0],
        "default": 0
    }
}