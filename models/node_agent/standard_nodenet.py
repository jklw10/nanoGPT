import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import parts.utils as utils


class CustomHebbianLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, w_scale, untanh_slope, error_ema, ema_speed, rew_sens, n_scale, step_counter, speed, width):
        norm_weight = F.normalize(weight, p=2, dim=-2) * w_scale
        raw_pred = torch.einsum('bnd,bnde->bne', x, norm_weight)
        
        bound = math.sqrt(2.0 + math.sqrt(2.0))
        out = utils.buntanh(raw_pred, untanh_slope, bound)
        
        ctx.save_for_backward(x, weight, error_ema, ema_speed, rew_sens, n_scale, step_counter, speed, width)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, error_ema, ema_speed, rew_sens, n_scale, step_counter, speed, width = ctx.saved_tensors
        
        error_signal = grad_output

        with torch.no_grad():
            new_ema = error_ema * (1.0 - ema_speed) + (error_signal * ema_speed)
            error_ema.copy_(new_ema)

            E_curr = F.softmax(error_signal, dim=-1)
            advantage = E_curr - error_ema
            plasticity = 1.0 + utils.rms_norm(advantage.unsqueeze(3) * rew_sens)

            noisy_state = utils.noised_input(x, error_signal, n_scale)
            
            lg1 = utils.rms_norm(torch.einsum('bni,bnj->bnij', error_signal, noisy_state))
            grad_weight = plasticity * lg1
            
            center = (step_counter * speed) % weight.shape[-1]
            g_mask = utils.wrapping_gaussian_kernel2(grad_weight, center, width)
            grad_weight = grad_weight * g_mask

        return None, grad_weight, None, None, None, None, None, None, None, None, None

class NodeParams(nn.Module):
    def __init__(self, batch_size, num_nodes, dim, device):
        super().__init__() 
        self.D = dim
        self.device = device
        
        base_weight = torch.randn(batch_size, num_nodes, dim, dim, device=device) / math.sqrt(self.D)
        self.weight = nn.Parameter(base_weight)
        self.register_buffer('error_ema', 0.01 + 0.01 * torch.randn(batch_size, num_nodes, dim, device=device))
        
        # Defaults
        self.untanh_slope = torch.tensor(1.0, device=device)
        self.w_scale = torch.tensor(1.0, device=device)
        self.ema_speed = torch.tensor(0.2, device=device)
        self.rew_sens = torch.tensor(1.36, device=device)
        self.n_scale = torch.tensor(1.5, device=device)
        self.speed = torch.tensor(0.5, device=device)
        self.width = torch.tensor(0.01, device=device)
     
    def forward(self, x, step_counter):
        # Apply the custom Autograd rule
        return CustomHebbianLinear.apply(
            x, self.weight, self.w_scale, self.untanh_slope,
            self.error_ema, self.ema_speed, self.rew_sens, self.n_scale,
            step_counter, self.speed, self.width
        )
        
    def apply_hyperparams(self, config):
        self.untanh_slope = config.UNTANH.view(-1, 1, 1)
        self.w_scale = config.w_scale.view(-1, 1, 1, 1)
        
        if hasattr(config, 'EMA_SPEED'):
            self.ema_speed = config.EMA_SPEED.view(-1, 1, 1)
            self.rew_sens = config.FIXED_REWARD_SENSITIVITY.view(-1, 1, 1, 1)
            self.n_scale = config.NOISE_SCALE.view(-1, 1, 1)
            self.speed = config.speed.view(-1, 1, 1, 1)
            self.width = config.width.view(-1, 1, 1, 1)

class MultiHeadTriadNodeNet(nn.Module):
    def __init__(self, batch_size, num_nodes, dim, device):
        super().__init__()
        self.N = num_nodes
        self.D = dim
        self.H = dim // 4
        self.head_dim = 4
        self.device = device
        
        self.w1 = NodeParams(batch_size, num_nodes, dim, device)
        self.w2 = NodeParams(batch_size, num_nodes, dim, device)
        self.w3 = NodeParams(batch_size, num_nodes, dim, device)
        
        self.register_buffer('state', 1e-5 * torch.randn(batch_size, self.N, dim, device=device))
        self.register_buffer('output', 1e-5 * torch.randn(batch_size, self.N, dim, device=device))
        self.register_buffer('A_ema', torch.zeros(batch_size, self.H, self.N, self.N, device=device))
        self.register_buffer('A_mask', torch.ones(batch_size, self.H, self.N, self.N, device=device))
        self.current_input_nodes = 0
        self.register_buffer('step_counter', torch.tensor(0.0, device=device))
        
        self.c_moment = 0.8
        self.a_variance = 0.0026

    def apply_hyperparams(self, config):
        if hasattr(config, 'c_moment'):
            self.c_moment = config.c_moment.view(-1, 1, 1, 1)
        if hasattr(config, 'a_variance'):
            self.a_variance = config.a_variance.view(-1, 1, 1, 1)
            
        self.w1.apply_hyperparams(config)
        self.w2.apply_hyperparams(config)
        self.w3.apply_hyperparams(config)

    def forward(self, env_input):
        self.step_counter.add_(1.0)

        batch_size, input_dim = env_input.shape
        num_input_nodes = math.ceil(input_dim / self.D)
        
        # 1. OUT-OF-PLACE Mask Updates (Fixes Inductor slicing bugs)
        if num_input_nodes != self.current_input_nodes:
            new_mask = torch.ones(batch_size, self.H, self.N, self.N, device=self.device)
            new_mask[:, :, :num_input_nodes, :] = 0.0 
            self.A_mask = new_mask # Reassigns the buffer
            
            # Clone avoids modifying the old A_ema memory saved in Autograd
            new_ema = self.A_ema.clone()
            new_ema[:, :, num_input_nodes, :num_input_nodes] += 3.0 / num_input_nodes
            self.A_ema = new_ema 
            
            self.current_input_nodes = num_input_nodes

        queries    = self.w3(self.state, self.step_counter)
        keys       = self.w2(self.state, self.step_counter) 
        prediction = self.w1(self.state, self.step_counter) 

        Q = queries.view(batch_size, self.N, self.H, self.head_dim).transpose(1, 2)
        K = keys.view(batch_size, self.N, self.H, self.head_dim).transpose(1, 2)

        raw_A = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        raw_A = utils.rms_norm(raw_A, dim=[-1, -2]) 
        
        # 2. OUT-OF-PLACE EMA Update. 
        # Detaching raw_A here intentionally breaks Backprop-Through-Time (BPTT), 
        # ensuring Q and K only learn from their proxy targets exactly as you designed.
        self.A_ema = (self.A_ema * self.c_moment + raw_A.detach() * (1.0 - self.c_moment)) * self.A_mask
        
        B_H_shape = (batch_size * self.H, self.N, self.N)
        raw_A_3D = self.A_ema.reshape(B_H_shape)
        raw_A_3D = utils.sinkhorn_knopp(raw_A_3D)
        raw_A_norm = raw_A_3D.reshape(batch_size, self.H, self.N, self.N)

        uniform_baseline = 1.0 / (self.N + 1.0)
        sparse_A = torch.where(raw_A_norm > uniform_baseline, raw_A_norm, 0.0)
        
        sparse_A_3D = sparse_A.reshape(B_H_shape)
        A = (utils.sinkhorn_knopp(sparse_A_3D).reshape(batch_size, self.H, self.N, self.N)) * self.A_mask
        
        current_var = A.var(dim=[-1, -2], keepdim=True)
        vd = F.relu(self.a_variance - current_var)
        
        V = self.output.view(batch_size, self.N, self.H, self.head_dim).transpose(1, 2)
        boredom_noise = vd * torch.randn_like(V)
        V = V + boredom_noise 
        
        target = torch.matmul(A, V) 
        target = target.transpose(1, 2).reshape(batch_size, self.N, self.D)
        target = utils.softsign(target)

        # 3. PURELY FUNCTIONAL Environment Injection
        pad_len = num_input_nodes * self.D - input_dim
        if pad_len > 0:
            env_input_padded = F.pad(env_input, (0, pad_len))
        else:
            env_input_padded = env_input
            
        env_input_reshaped = env_input_padded.view(batch_size, num_input_nodes, self.D)
        
        # Using torch.cat instead of `target[:, :] = env_input` strictly prevents inplace errors
        if num_input_nodes < self.N:
            target = torch.cat([env_input_reshaped, target[:, num_input_nodes:, :]], dim=1)
        else:
            target = env_input_reshaped

        # --- Calculate Proxy Target Losses ---
        err1_base = prediction - target.detach()
        loss_w2 = F.mse_loss(keys, err1_base.detach(), reduction='none') 

        err2 = keys.detach() - err1_base.detach()
        loss_w3 = F.mse_loss(queries, err2, reduction='none')

        err3 = queries.detach() - err2
        loss_w1 = F.mse_loss(prediction, target.detach() + err3, reduction='none')

        total_loss = (loss_w1 + loss_w2 + loss_w3).sum()
        
        # 4. OUT-OF-PLACE Buffer Update
        # Re-assigning completely bypasses the inplace Autograd protections!
        self.output = prediction.detach()
        self.state = target.detach()

        return self.output, total_loss