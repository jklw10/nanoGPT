import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import parts.utils as utils

from dataclasses import dataclass

@dataclass
class HyperParamConfig:
    UNTANH: torch.Tensor
    EMA_SPEED: torch.Tensor
    FIXED_REWARD_SENSITIVITY: torch.Tensor
    LR: torch.Tensor
    MOMENTUM: torch.Tensor
    weight_decay: torch.Tensor
    NOISE_SCALE: torch.Tensor
    c_moment: torch.Tensor
    w_scale: torch.Tensor
    sharpness: torch.Tensor
    a_variance: torch.Tensor
    speed: torch.Tensor
    width: torch.Tensor
    lateral_decay: torch.Tensor

param_space = {
    'UNTANH': {'low': 0.1, 'high': 3.0},
    'EMA_SPEED': {'low': 0.01, 'high': 0.5},
    'FIXED_REWARD_SENSITIVITY': {'low': 0.1, 'high': 2.0},
    'LR': {'low': 0.001, 'high': 0.05},
    'MOMENTUM': {'low': 0.1, 'high': 0.9},
    'weight_decay': {'low': 0.001, 'high': 0.1},
    'NOISE_SCALE': {'low': 0.1, 'high': 1.5},
    'c_moment': {'low': 0.1, 'high': 0.99},
    'w_scale': {'low': 0.1, 'high': 2.0},
    'sharpness': {'low': 0.5, 'high': 5.0},
    'a_variance': {'low': 0.0, 'high': 1.0}
}

defaults = {
    "speed": 0.5,
    "width": 0.01,
    "lateral_decay": 0.01,
    "EMA_SPEED": 0.20310950837532799,
    "FIXED_REWARD_SENSITIVITY": 1.360218046607915,
    "LR": 0.005720721430830338,
    "MOMENTUM": 0.999,
    "weight_decay": 0.021698750753204772,
    "NOISE_SCALE": 1.5,
    "c_moment": 0.8021549684435622,
    "w_scale": 0.6340345603476942,
    "sharpness": 1.0390895413424572,
    "a_variance": 0.002633467302632888
}

#referenced functions from utils.py
#def buntanh(x, slope, bound):
#    return torch.tanh(x - torch.tanh(x)*slope)*bound
#
#def noised_input(input, grad, scale=0.1):
#    state_mag = input.abs().sum(dim=[-2,-1], keepdim=True) 
#    noise1 = torch.randn_like(input) / (1.0 + state_mag)
#    noise2 = torch.randn_like(input) * grad.std() * scale
#    return input + noise1 + noise2
#torch.compile(backend='inductor', mode='max-autotune')
#def wrapping_gaussian_kernel2(activation_tensor: torch.Tensor, 
#                                     center_offset, 
#                                     sigma) -> torch.Tensor:
#    
#    features = activation_tensor.shape[-1]
#    device = activation_tensor.device
#    dtype = activation_tensor.dtype
#    
#    indices = torch.arange(features, dtype=dtype, device=device)
#    
#    center = center_offset * features
#    sigma = sigma * features
#    
#    # Calculate absolute difference
#    diff = torch.abs(indices - center)
#    diff = torch.min(diff, features - diff)
#    
#    kernel = torch.exp(-(diff).pow(2) / (2 * sigma**2))
#    return kernel
#def sinkhorn_knopp(log_alpha,iter=5,dim=[-1,-2]):
#    device = log_alpha.device
#    d1, d2 = dim
#    u = torch.zeros(log_alpha.shape[:d1] + (1,), device=device) # (batch, M, 1)
#    v = torch.zeros(log_alpha.shape[:d2] + (1, log_alpha.shape[d1]), device=device) # (batch, 1, N)
#    for _ in range(iter):
#        u = -torch.logsumexp(log_alpha + v, dim=d1, keepdim=True)
#        v = -torch.logsumexp(log_alpha + u, dim=d2, keepdim=True)
#    return torch.exp(log_alpha + u + v)

class NodeParams(nn.Module):
    def __init__(self, batch_size, num_nodes, dim, device):
        super().__init__() 
        self.D = dim
        self.device = device
        base_weight = torch.randn(1, num_nodes, dim, dim, device=device) / math.sqrt(self.D)
        self.register_buffer('weight', base_weight.expand(batch_size, -1, -1, -1).clone())
        
        self.register_buffer('g_ema', torch.randn_like(self.weight) * 0.01) 
        self.register_buffer('error_ema', 0.01 + 0.01 * torch.randn(batch_size, num_nodes, dim, device=self.device))
     
    def forward(self, x):
        raw_pred = (x.unsqueeze(2) @ self.weight).squeeze(2)
        #regular linear layer per node, + wacky activation function
        return utils.buntanh(raw_pred, self.hp_untanh, math.sqrt(2 + math.sqrt(2)))
    
    def get_gradient(self, layer_input, error_signal):
        ema_speed = self.hp_ema_speed
        rew_sens = self.hp_rew_sens
        n_scale = self.hp_noise_scale
        
        self.error_ema.mul_(1.0 - ema_speed).add_(ema_speed * error_signal)

        E_curr = F.softmax(error_signal, dim=-1)

        advantage = E_curr - self.error_ema
        plasticity = 1.0 + utils.rms_norm(advantage.unsqueeze(3) * rew_sens)  
        
        #noise divided by sum of input abs
        noisy_state = utils.noised_input(layer_input, error_signal, n_scale)
        lg1 = utils.rms_norm(-torch.einsum('bni,bnj->bnij', error_signal, noisy_state))
        return (plasticity * lg1) 
    
    def step(self, grad, step_counter):
        wd = self.hp_wd
        grad = grad - (wd * self.weight)
        
        speed = self.hp_speed 
        width = self.hp_width 
        center = (step_counter * speed) % self.D
        #per index filter, practically  only let some of the gradients through
        #depending on which index they're in
        #TODO: try randomized indices
        #sort of pseudo dropout, but predictable.
        g_mask = utils.wrapping_gaussian_kernel2(grad, center, width)
        grad = grad * g_mask
        
        momentum = self.hp_momentum 
        lr = self.hp_lr 

        self.g_ema.mul_(momentum).add_((1.0 - momentum) * grad)
        #apply ema of gradient, mostly a temporal stability experiment
        #TODO: try gdiff, maybe adamW
        self.weight.add_(lr * self.g_ema)
        
        #put weights on a sphere to prevent explosions.
        w_scale = self.hp_w_scale
        self.weight.data.copy_(utils.rms_norm(self.weight.data, dim=[-1,-2]) * w_scale )
   
    def apply_hyperparams(self, config: HyperParamConfig):
        # We read directly from the config dataclass
        self.hp_untanh = config.UNTANH.view(-1, 1, 1)
        self.hp_ema_speed = config.EMA_SPEED.view(-1, 1, 1)
        self.hp_noise_scale = config.NOISE_SCALE.view(-1, 1, 1)

        self.hp_rew_sens = config.FIXED_REWARD_SENSITIVITY.view(-1, 1, 1, 1)
        self.hp_wd = config.weight_decay.view(-1, 1, 1, 1)
        self.hp_speed = config.speed.view(-1, 1, 1, 1)
        self.hp_width = config.width.view(-1, 1, 1, 1)
        self.hp_momentum = config.MOMENTUM.view(-1, 1, 1, 1)
        self.hp_lr = config.LR.view(-1, 1, 1, 1)
        self.hp_w_scale = config.w_scale.view(-1, 1, 1, 1)

        base_weight = torch.zeros(1, self.weight.shape[1], self.D, self.D, device=self.device) / math.sqrt(self.D)
        self.weight.data.copy_(base_weight.expand_as(self.weight))
        self.weight[...,:1] = 1
        
        self.weight.data.copy_(utils.rms_norm(self.weight.data, dim=[-1,-2]) * self.hp_w_scale) 
        
        self.g_ema.normal_().mul_(0.01)
        self.error_ema.normal_().mul_(0.01).add_(0.01)

class MultiHeadTriadNodeNet(nn.Module):
    def __init__(self, batch_size, num_nodes, dim, device):
        super().__init__()
        self.N = num_nodes
        self.D = dim
        self.H = dim//4
        self.head_dim = 4
        self.batch_size = batch_size
        self.device = device
        self.w1 = NodeParams(batch_size, num_nodes, dim, device)
        self.w2 = NodeParams(batch_size, num_nodes, dim, device)
        self.w3 = NodeParams(batch_size, num_nodes, dim, device)
        
        self.register_buffer('state', 1e-5 * torch.randn(batch_size, self.N, dim, device=self.device))
        self.register_buffer('output', 1e-5 * torch.randn(batch_size, self.N, dim, device=self.device))
        self.register_buffer('step_counter', torch.tensor(0.0, device=self.device))
        
        
        self.register_buffer('A', torch.randn(batch_size, self.H, self.N, self.N, device=self.device))
        self.register_buffer('A_ema', torch.zeros(batch_size, self.H, self.N, self.N, device=self.device))
        self.A.data.copy_(F.softmax(self.A, dim=-1))
        
        self.register_buffer('A_mask', torch.ones_like(self.A, device=self.device))
        self.current_input_nodes = 0


    def forward(self, env_input):
        
        self.step_counter.add_(1.0)
        
        batch_size, input_dim = env_input.shape
        num_input_nodes = math.ceil(input_dim / self.D)
        
        if num_input_nodes >= self.N:
            raise ValueError(f"Input requires {num_input_nodes} nodes, but brain only has {self.N} nodes. Increase N or D.")
             
        if num_input_nodes != self.current_input_nodes:
            self.A_mask.fill_(1.0)
            self.A_mask[:, :, :num_input_nodes, :] = 0.0 
            self.A_ema[:, :, num_input_nodes, :num_input_nodes] += 3.0 / num_input_nodes
            self.current_input_nodes = num_input_nodes

        queries    = self.w3(self.state)
        keys       = self.w2(self.state) 
        prediction = self.w1(self.state) 

        Q = queries.view(batch_size, self.N, self.H, self.head_dim).transpose(1, 2)
        K = keys.view(batch_size, self.N, self.H, self.head_dim).transpose(1, 2)

        raw_A = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        raw_A = utils.rms_norm(raw_A, dim=[-1, -2]) 

        is_first = (self.step_counter == 1.0).float().view(1, 1, 1, 1)
        new_A_ema = raw_A * is_first + (self.A_ema * self.hp_c_moment + raw_A * (1.0 - self.hp_c_moment)) * (1.0 - is_first)
        self.A_ema.copy_(new_A_ema * self.A_mask)
        
        raw_A = self.A_ema 
        
        B_H_shape = (batch_size * self.H, self.N, self.N)
        raw_A_3D = raw_A.reshape(B_H_shape)
        raw_A_3D = utils.sinkhorn_knopp(raw_A_3D)
        raw_A = raw_A_3D.reshape(batch_size, self.H, self.N, self.N)

        uniform_baseline = 1.0 / (self.N + 1.0)
        sparse_A = torch.where(raw_A > uniform_baseline, raw_A, 0.0)
        
        sparse_A_3D = sparse_A.reshape(B_H_shape)
        self.A.copy_((utils.sinkhorn_knopp(sparse_A_3D).reshape(batch_size, self.H, self.N, self.N)) * self.A_mask)
        
        current_var = self.A.var(dim=[-1, -2], keepdim=True)
        vd = F.relu(self.hp_a_variance - current_var)
        V = (self.output).view(batch_size, self.N, self.H, self.head_dim).transpose(1, 2)
        boredom_noise = vd * torch.randn_like(V)
        V=V+boredom_noise 
        
        target = torch.matmul(self.A, V) 
        target = target.transpose(1, 2).reshape(batch_size, self.N, self.D)
        target = utils.softsign(target)

        # Pad and inject the dynamic env_input into the reserved nodes
        pad_len = num_input_nodes * self.D - input_dim
        if pad_len > 0:
            env_input_padded = F.pad(env_input, (0, pad_len))
        else:
            env_input_padded = env_input
            
        env_input_reshaped = env_input_padded.view(batch_size, num_input_nodes, self.D)
        target[:, :num_input_nodes, :] = env_input_reshaped

        err1_base = prediction - target
        err2 = keys - err1_base.detach()
        err3 = queries - err2.detach()
        err1 = prediction - (target + err3.detach())

        self.w1.step(self.w1.get_gradient(self.state, err1), self.step_counter)
        self.w2.step(self.w2.get_gradient(self.state, err2), self.step_counter)
        self.w3.step(self.w3.get_gradient(self.state, err3), self.step_counter)

        self.output.copy_(prediction)
        self.state.copy_(target)

    def apply_hyperparams(self, config: HyperParamConfig):
        # Apply the pre-calculated dataclass configuration
        self.hp_c_moment = config.c_moment.view(-1, 1, 1, 1)
        self.hp_a_variance = config.a_variance.view(-1, 1, 1, 1)
        self.hp_untanh = config.UNTANH.view(-1, 1, 1)

        self.w1.apply_hyperparams(config)
        self.w2.apply_hyperparams(config)
        self.w3.apply_hyperparams(config)
        
        self.state.normal_().mul_(1e-5)
        self.output.normal_().mul_(1e-5)
        self.step_counter.fill_(0.0)
        
        self.A.normal_()
        self.A_ema.zero_()
        self.A.data.copy_(F.softmax(self.A, dim=-1))
        
        self.A_mask.fill_(1.0)
        self.current_input_nodes = 0

class TriadNodeNet(nn.Module):
    def __init__(self, batch_size, num_nodes, dim, device):
        super().__init__()
        self.N = num_nodes
        self.D = dim
        self.batch_size = batch_size
        self.device = device
        self.w1 = NodeParams(batch_size, num_nodes, dim, device)
        self.w2 = NodeParams(batch_size, num_nodes, dim, device)
        self.w3 = NodeParams(batch_size, num_nodes, dim, device)
        
        self.register_buffer('state', 1e-5 * torch.randn(batch_size, self.N, dim, device=self.device))
        self.register_buffer('output', 1e-5 * torch.randn(batch_size, self.N, dim, device=self.device))
        self.register_buffer('step_counter', torch.tensor(0.0, device=self.device))
        
        self.register_buffer('A', torch.randn(batch_size, self.N, self.N, device=self.device))
        self.register_buffer('A_ema', torch.zeros(batch_size, self.N, self.N, device=self.device))
        self.A.data.copy_(F.softmax(self.A, dim=-1))
        
        self.register_buffer('A_mask', torch.ones_like(self.A, device=self.device))
        self.current_input_nodes = 0


    def forward(self, env_input):
        self.step_counter.add_(1.0)
        
        batch_size, input_dim = env_input.shape
        num_input_nodes = math.ceil(input_dim / self.D)
        
        if num_input_nodes >= self.N:
            raise ValueError(f"Input requires {num_input_nodes} nodes, but brain only has {self.N} nodes. Increase N or D.")
             
        if num_input_nodes != self.current_input_nodes:
            self.A_mask.fill_(1.0)
            self.A_mask[:, :num_input_nodes, :] = 0.0 
            self.A_ema[:, num_input_nodes, :num_input_nodes] += 3.0 / num_input_nodes
            self.current_input_nodes = num_input_nodes

        Q          = self.w3(self.state)
        K          = self.w2(self.state) 
        prediction = self.w1(self.state) 

        raw_A = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.D)
        raw_A = utils.rms_norm(raw_A, dim=[-1, -2]) 

        is_first = (self.step_counter == 1.0).float().view(1, 1, 1)
        new_A_ema = raw_A * is_first + (self.A_ema * self.hp_c_moment + raw_A * (1.0 - self.hp_c_moment)) * (1.0 - is_first)
        self.A_ema.copy_(new_A_ema * self.A_mask)
        
        raw_A = self.A_ema 
        
        raw_A = utils.sinkhorn_knopp(raw_A)

        uniform_baseline = 1.0 / (self.N + 1.0)
        sparse_A = torch.where(raw_A > uniform_baseline, raw_A, 0.0)
        
        self.A.copy_(utils.sinkhorn_knopp(sparse_A) * self.A_mask)
        
        current_var = self.A.var(dim=[-1, -2], keepdim=True)
        vd = F.relu(self.hp_a_variance - current_var)

        V = self.output
        boredom_noise = vd * torch.randn_like(V)
        V = V + boredom_noise 
        
        target = torch.matmul(self.A, V) 
        target = utils.softsign(target)

        # Pad and inject the dynamic env_input into the reserved nodes
        pad_len = num_input_nodes * self.D - input_dim
        if pad_len > 0:
            env_input_padded = F.pad(env_input, (0, pad_len))
        else:
            env_input_padded = env_input
            
        env_input_reshaped = env_input_padded.view(batch_size, num_input_nodes, self.D)
        target[:, :num_input_nodes, :] = env_input_reshaped

        err1_base = prediction - target
        err2 = K - err1_base.detach()
        err3 = Q - err2.detach()
        err1 = prediction - (target + err3.detach())

        self.w1.step(self.w1.get_gradient(self.state, err1), self.step_counter)
        self.w2.step(self.w2.get_gradient(self.state, err2), self.step_counter)
        self.w3.step(self.w3.get_gradient(self.state, err3), self.step_counter)

        self.output.copy_(prediction)
        self.state.copy_(target)

    def apply_hyperparams(self, config: HyperParamConfig):
        # Apply the pre-calculated dataclass configuration
        self.hp_c_moment = config.c_moment.view(-1, 1, 1)
        self.hp_a_variance = config.a_variance.view(-1, 1, 1)
        self.hp_untanh = config.UNTANH.view(-1, 1, 1)

        self.w1.apply_hyperparams(config)
        self.w2.apply_hyperparams(config)
        self.w3.apply_hyperparams(config)
        
        self.state.normal_().mul_(1e-5)
        self.output.normal_().mul_(1e-5)
        self.step_counter.fill_(0.0)
        
        self.A.normal_()
        self.A_ema.zero_()
        self.A.data.copy_(F.softmax(self.A, dim=-1))
        
        self.A_mask.fill_(1.0)
        self.current_input_nodes = 0


class NodeNet(nn.Module):
    def __init__(self, batch_size, num_nodes, dim, device):
        super().__init__()
        self.N = num_nodes
        self.D = dim
        self.batch_size = batch_size
        self.device = device
        self.w1 = NodeParams(batch_size, num_nodes, dim, device)
        
        self.register_buffer('state', 1e-5 * torch.randn(batch_size, self.N, dim, device=self.device))
        self.register_buffer('output', 1e-5 * torch.randn(batch_size, self.N, dim, device=self.device))
        self.register_buffer('step_counter', torch.tensor(0.0, device=self.device))
        
        self.register_buffer('connectome', torch.randn(batch_size, self.N, self.N, device=self.device))
        self.connectome.copy_(utils.sinkhorn_knopp(self.connectome))
        

    def forward(self, env_input):
        
        #gemini comment on naming:
        #If you view this through the lens of recurrent state-space models.
        #state -> latent_state
        #output -> hidden_state (or memory)
        #target -> observed_state

        self.step_counter.add_(1.0)
        
        batch_size, input_dim = env_input.shape

        #how many nodes dimensions are needed to represent the input
        num_input_nodes = math.ceil(input_dim / self.D)
        
        if num_input_nodes >= self.N:
            raise ValueError(f"Input requires {num_input_nodes} nodes, but brain only has {self.N} nodes. Increase N or D.")
        
        #route previous network output
        #this will be the input to the next pass
        #and is the prediction target of the previous pass
        target = torch.matmul(self.connectome, self.output) 
        #not properly tested, limit activation size.
        target = utils.softsign(target)

        #haven't gone through this padding and whether it's necessary
        pad_len = num_input_nodes * self.D - input_dim
        if pad_len > 0:
            env_input = F.pad(env_input, (0, pad_len))
        
        env_input_reshaped = env_input.view(batch_size, num_input_nodes, self.D)
        #replace the targets by environment inputs,
        #this turns some of the nodes into sort of perceptrons
        target[:, :num_input_nodes, :] = env_input_reshaped

        #based on state, which is previous rounds target,
        #try to predict what the current rounds output will be
        prediction = self.w1(self.state) 
        err1 = prediction - target

        #update weights
        self.w1.step(self.w1.get_gradient(self.state, err1), self.step_counter)

        #current rounds output is the prediction you made.
        self.output.copy_(prediction)
        #the state is the target you made.
        self.state.copy_(target)

    def apply_hyperparams(self, config: HyperParamConfig):
        self.hp_c_moment = config.c_moment.view(-1, 1, 1, 1)
        self.hp_a_variance = config.a_variance.view(-1, 1, 1, 1)
        self.hp_untanh = config.UNTANH.view(-1, 1, 1)

        self.w1.apply_hyperparams(config)
        
        self.state.normal_().mul_(1e-5)
        self.output.normal_().mul_(1e-5)
        self.step_counter.fill_(0.0)
        self.connectome.normal_()
        self.connectome.copy_(utils.sinkhorn_knopp(self.connectome))
        self.current_input_nodes = 0


class CustomHebbianLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, w_scale, untanh_slope, error_ema, ema_speed, rew_sens, n_scale, step_counter, speed, width):
        norm_weight = F.normalize(weight, p=2, dim=-2) * w_scale
        #todo, try different norms.
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

class StandardNodeParams(nn.Module):
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

class StandardMultiHeadTriadNodeNet(nn.Module):
    """autograd compatible"""
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
        
        if num_input_nodes != self.current_input_nodes:
            new_mask = torch.ones(batch_size, self.H, self.N, self.N, device=self.device)
            new_mask[:, :, :num_input_nodes, :] = 0.0 
            self.A_mask = new_mask 
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

        err1_base = prediction - target.detach()
        loss_w2 = F.mse_loss(keys, err1_base.detach(), reduction='none') 

        err2 = keys.detach() - err1_base.detach()
        loss_w3 = F.mse_loss(queries, err2, reduction='none')

        err3 = queries.detach() - err2
        loss_w1 = F.mse_loss(prediction, target.detach() + err3, reduction='none')

        total_loss = (loss_w1 + loss_w2 + loss_w3).sum()
        
        self.output = prediction.detach()
        self.state = target.detach()

        return self.output, total_loss