import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import utils

ema_thing = True
alpha = 0.9
class OptimizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, batch_size: int = None, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size # Max batch size the layer is designed for
        self.top_k_lr_selection = 5
        self.lr_update_momentum = 0.9
        self.start_perc = 0.9
        self.active_percentage = self.start_perc
        self.step = 0.0
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        
        self.register_buffer("wema", torch.zeros_like(self.weight))
        self.previous_bias_grad = None
        if bias:
            self.register_buffer("bema", torch.zeros_like(self.bias))
        
        # Stores the gradient from the *previous* backward pass practically a per weight bias, which gets added to the weight after a run with lr.
      
        self.register_buffer("previous_weight_grad", torch.zeros_like(self.weight))
        self.previous_bias_grad = None
        if bias:
            self.register_buffer("previous_bias_grad", torch.zeros_like(self.bias))
        
        # Stores the current batch of LR samples used for forward pass simulation
        if batch_size is not None:
            self.register_buffer("batch_lr_samples", torch.empty(batch_size))
        else:
            self.register_buffer("batch_lr_samples", torch.empty(0))

        self.current_lr_mean = nn.Parameter(torch.tensor(1e-5), requires_grad=False)
        self.current_lr_var = nn.Parameter(torch.tensor(1e-4), requires_grad=False) # Small initial variance

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Standard weight initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        # Initialize previous gradients and LR samples to zero/defaults
        with torch.no_grad():
            self.previous_weight_grad.zero_()
            if self.previous_bias_grad is not None:
                self.previous_bias_grad.zero_()
            self.current_lr_mean.fill_(0.5)
            self.current_lr_var.fill_(0.5)
            if(self.batch_size is not None):
                self._sample_batch_lrs() 
            if(ema_thing):
                num_active_out = math.ceil(self.out_features * self.active_percentage)
                self.wema =  self.weight.data.detach() 
                if self.bias is not None:
                    self.bema =  self.bias.data.detach() 
                    torch.zero_(self.bias.data[num_active_out:])
                torch.zero_(self.weight.data[num_active_out:,:])
            return
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input_data: (actual_batch_size, in_features)
        #if(not self.training):
        #    return nn.functional.linear(x,self.weight,self.bias)


        if(ema_thing):
            num_active_out = math.ceil(self.out_features * self.active_percentage)
            if(not self.training):
                aw = self.weight[:num_active_out, :]
                ab = None
                if self.bias is not None:
                    ab = self.bias[:num_active_out]
                active_output = nn.functional.linear(x,aw,ab)
                full_output = torch.zeros(x.shape[0],x.shape[1], self.out_features, device=x.device, dtype=x.dtype)
                full_output[:,:, :num_active_out] = active_output
                return full_output 
            # Create a mask to zero-out inactive weights
            # This is done on the fly and is very efficient on GPU
            #weight_mask = torch.zeros_like(self.weight.data, memory_format=torch.contiguous_format)
            #weight_mask[:num_active_out, :] = 1.0
            #skip ema on backward pass?
            aw = self.weight[:num_active_out, :]
            forward_weight = utils.funnyMulti(self.wema.detach()[:num_active_out, :], aw)
            simulated_weight = aw + (forward_weight - aw).detach()
            #simulated_weight = simulated_weight * weight_mask
            #simulated_weight = simulated_weight[:num_active_out, :]
            #simulated_weight = utils.funnyMulti(self.weight, self.wema)
            simulated_bias = None
            if self.bias is not None:
                ab = self.bias[:num_active_out]
                fwd_bias = utils.funnyMulti(self.bema.detach()[:num_active_out], ab)
                simulated_bias = ab + (fwd_bias - ab).detach()

            active_output = nn.functional.linear(x,simulated_weight,simulated_bias)
            full_output = torch.zeros(x.shape[0],x.shape[1], self.out_features, device=x.device, dtype=x.dtype)
            full_output[:,:, :num_active_out] = active_output
            return full_output 
        
        if(not self.training):
            return nn.functional.linear(x,self.weight,self.bias)
        if(self.batch_size is None):
            self.batch_size = x.shape[0]
            self._sample_batch_lrs()

        actual_batch_size = x.shape[0]
        if actual_batch_size != self.batch_size:
            if actual_batch_size > self.batch_size:
                raise ValueError(f"Actual batch size {actual_batch_size} exceeds layer's configured max batch size {self.batch_size}")
            
        simulated_weight = self.weight.unsqueeze(0) - \
                           self.previous_weight_grad.unsqueeze(0) * self.batch_lr_samples[:actual_batch_size].view(-1, 1, 1)
        
        output_per_sample = torch.bmm(x, simulated_weight.transpose(1, 2))
        #output_per_sample = torch.einsum('bsi,boi->bso', x, simulated_weight)
        simulated_bias = None
        if self.bias is not None:
            simulated_bias = self.bias.unsqueeze(0) - \
                             self.previous_bias_grad.unsqueeze(0) * self.batch_lr_samples[:actual_batch_size].view(-1, 1)
            output_per_sample += simulated_bias
        return output_per_sample

    def _weight_update(self) -> None:
        
        with torch.no_grad():
            
            if self.weight.grad is not None:
                if self.previous_weight_grad is not None:    
                    self.weight.data.sub_(self.previous_weight_grad * self.current_lr_mean) 

                adjusted_grad = self.weight.grad
                #adjusted_grad = utils.mmnorm(adjusted_grad)
                #adjusted_grad = utils.zca_newton_schulz(adjusted_grad, 2, 2)
                #
                self.previous_weight_grad.copy_(adjusted_grad)
                self.weight.grad.zero_()
            if self.bias is not None and self.bias.grad:
                if self.previous_bias_grad is not None:
                    self.bias.data.sub_(self.previous_bias_grad * self.current_lr_mean) 
                self.previous_bias_grad.copy_(self.bias.grad)
                self.bias.grad.zero_()
    
    def poststep(self):
        #pass
        if(ema_thing):
            num_active_out = math.ceil(self.out_features * self.active_percentage)
            self.step += 1.0
            self.active_percentage = min(self.start_perc+ (1.0-self.start_perc)*(self.step/30000.0),1.0 )
            #self.active_percentage = torch.sqrt(self.active_percentage) 
            with torch.no_grad():
                self.wema[:num_active_out,:] += alpha * (self.weight.data[:num_active_out,:] - self.wema[:num_active_out,:])
                if self.bias is not None:
                    self.bema[:num_active_out] += alpha * (self.bias.data[:num_active_out]  - self.bema[:num_active_out])

    def prestep(self, per_sample_losses: torch.Tensor):
        """
        Call this method *before* the optimizer.step().
        """
        if(ema_thing):
            #with torch.no_grad():
            #    self.wema += espeed * (self.weight.data - self.wema)
            #    if self.bias is not None:
            #        self.bema += espeed * (self.bias.data - self.bema)
            return
        
        #if(ema_thing):
        #    with torch.no_grad():
        #        self.wema += 0.1 * (self.weight.data - self.wema)
        #        if self.bias is not None:
        #            self.bema += 0.1 * (self.bias.data - self.bema)
        #    return
        self._update_lr_stats(per_sample_losses)
        self._weight_update()
        self._sample_batch_lrs()

    @torch.no_grad()
    def _sample_batch_lrs(self):
        mean_lr = self.current_lr_mean
        std_dev_lr = self.current_lr_var.sqrt()

        # Sample LRs from a normal distribution. Clamp to reasonable bounds (e.g., > 0)
        #lr_samples = torch.randn(self.batch_size, device=self.weight.device) * std_dev_lr + mean_lr
        lr_samples = torch.distributions.Beta(mean_lr, 3.0).sample([self.batch_size])
        lr_samples = torch.clamp(lr_samples, min=1e-8, max=1.0) # Ensure LRs are positive and bounded
        self.batch_lr_samples[:self.batch_size].copy_(lr_samples)

    @torch.no_grad()
    def _update_lr_stats(self, per_sample_losses: torch.Tensor):
        if per_sample_losses.ndim > 1:
            raise ValueError("per_sample_losses must be a 1D tensor (per-sample loss).")
        
        actual_batch_size = per_sample_losses.shape[0]

        lrs_used = self.batch_lr_samples
        # Find the top_k LRs that resulted in the lowest losses
        # If actual_batch_size is smaller than top_k_lr_selection, use actual_batch_size
        k_val = min(self.top_k_lr_selection, actual_batch_size)
        
        # Use topk to find the indices of the 'k_val' lowest losses
        # per_sample_losses.topk(k, largest=False) returns (values, indices)
        _, best_loss_indices = per_sample_losses.topk(k_val, largest=False)
        
        # Get the LRs corresponding to these best losses
        best_lrs = lrs_used[best_loss_indices]
        # Calculate new mean and variance from these best LRs
        new_mean = best_lrs.mean()
        new_var = best_lrs.var() if k_val > 1 else torch.tensor(0.0) # If only one sample, variance is 0
        # Smoothly update the layer's current_lr_mean and current_lr_var
        # These will be used to generate LR samples for the *next* forward pass.
        # check if sane, i'm suspicious of .mul on the data itself.
        self.current_lr_mean.data.mul_(1 - self.lr_update_momentum).add_(new_mean * self.lr_update_momentum)
        self.current_lr_var.data.mul_(1 - self.lr_update_momentum).add_(new_var * self.lr_update_momentum)
