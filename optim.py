import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class OptimizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, batch_size: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size # Max batch size the layer is designed for
        self.top_k_lr_selection = 5
        self.lr_update_momentum = 0.9

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        
        
        # Stores the gradient from the *previous* backward pass practically a per weight bias, which gets added to the weight after a run with lr.
      
        self.register_buffer("previous_weight_grad", torch.zeros_like(self.weight))
        self.previous_bias_grad = None
        if bias:
            self.register_buffer("previous_bias_grad", torch.zeros_like(self.bias))
        
        # Stores the current batch of LR samples used for forward pass simulation
        self.batch_lr_samples = nn.Parameter(torch.empty(batch_size), requires_grad=False) #i wonder what'd happen if i gave these a gradient :D

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
            self._sample_batch_lrs() 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input_data: (actual_batch_size, in_features)
        if(not self.training):
            return nn.functional.linear(x,self.weight,self.bias)
        
        actual_batch_size = x.shape[0]
        if actual_batch_size != self.batch_size:
            if actual_batch_size > self.batch_size:
                raise ValueError(f"Actual batch size {actual_batch_size} exceeds layer's configured max batch size {self.batch_size}")
            
        simulated_weight = self.weight.unsqueeze(0) - \
                           self.previous_weight_grad.unsqueeze(0) * self.batch_lr_samples[:actual_batch_size].view(-1, 1, 1)
        ##output_per_sample = torch.bmm(x, simulated_weight.transpose(1, 2))
        output_per_sample = torch.einsum('bsi,boi->bso', x, simulated_weight)
        simulated_bias = None
        if self.bias is not None:
            simulated_bias = self.bias.unsqueeze(0) - \
                             self.previous_bias_grad.unsqueeze(0) * self.batch_lr_samples[:actual_batch_size].view(-1, 1)
            output_per_sample += simulated_bias
        return output_per_sample

    def _weight_update(self) -> None:
        
        with torch.no_grad():
            
            if self.previous_weight_grad.grad is not None:
                self.weight.data.sub_(self.previous_weight_grad * self.current_lr_mean) 
                self.previous_weight_grad.copy_(self.weight.grad)
                self.weight.grad.zero_()
            if self.previous_bias_grad is not None and self.bias.grad is not None:
                self.bias.data.sub_(self.previous_bias_grad * self.current_lr_mean)
                self.previous_bias_grad.copy_(self.bias.grad)
                self.bias.grad.zero_() 
    
    
    def prestep(self, per_sample_losses: torch.Tensor):
        """
        Call this method *before* the optimizer.step().
        """
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
        if per_sample_losses.ndim != 1:
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
