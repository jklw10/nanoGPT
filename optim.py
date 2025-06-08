import torch
import torch.nn as nn
import math
import collections
import torch.optim as optim
from torch.optim.optimizer import Optimizer, required
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

        # --- Actual trainable parameters (what the optimizer updates) ---
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        # --- State for LR adaptation (non-trainable by backprop) ---
        # Stores the gradient from the *previous* backward pass
        self.previous_grad_param = nn.Parameter(torch.zeros_like(self.weight), requires_grad=False)
        self.previous_bias_grad_param = None
        if bias:
            self.previous_bias_grad_param = nn.Parameter(torch.zeros_like(self.bias), requires_grad=False)
        
        # Stores the current batch of LR samples used for forward pass simulation
        self.batch_lr_samples = nn.Parameter(torch.empty(batch_size), requires_grad=False)

        # Stores the mean and variance of the "optimal" LRs from previous steps.
        # These guide the sampling of batch_lr_samples for the *next* forward pass.
        # Initialized to default (e.g., from base LR) and a small variance.
        self.current_lr_mean = nn.Parameter(torch.tensor(0.0), requires_grad=False)
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
            self.previous_grad_param.zero_()
            if self.previous_bias_grad_param is not None:
                self.previous_bias_grad_param.zero_()
            self.batch_lr_samples.zero_() # Will be populated by optimizer
            # Initialize mean/var for LR sampling. Actual optimizer will set these.
            self.current_lr_mean.fill_(0.0)
            self.current_lr_var.fill_(1e-4) # Start with small variance

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        # input_data: (actual_batch_size, in_features)
        
        # Check if actual batch size matches expected, if not, resize batch_lr_samples
        actual_batch_size = input_data.shape[0]
        if actual_batch_size != self.batch_size:
            # This is a bit of a hack, normally batch_size is fixed.
            # But if the last batch is smaller, we need to adapt.
            if actual_batch_size > self.batch_size:
                raise ValueError(f"Actual batch size {actual_batch_size} exceeds layer's configured max batch size {self.batch_size}")
            # If smaller, resize (or just take a slice, which is handled by the optimizer)
            # The optimizer will handle providing only 'actual_batch_size' LRs
            # For simplicity, let's assume actual_batch_size == self.batch_size for now.
            # If not, the optimizer must ensure batch_lr_samples is properly sliced.
            # Here, we'll assume the optimizer provides the correct length.

        # Expand weight and previous_grad for batch multiplication
        # self.weight: (out_features, in_features) -> (1, out_features, in_features)
        # self.previous_grad_param: (out_features, in_features) -> (1, out_features, in_features)
        # self.batch_lr_samples: (actual_batch_size,) -> (actual_batch_size, 1, 1) for broadcasting
        
        # simulated_weight: (actual_batch_size, out_features, in_features)
        # This creates a slightly different weight matrix for each sample in the batch
        # based on its corresponding LR sample and the previous gradient.
        # This is the "f(input * (weight - previousGrad * batchlr))" part.
        simulated_weight = self.weight.unsqueeze(0) - \
                           self.previous_grad_param.unsqueeze(0) * self.batch_lr_samples[:actual_batch_size].view(-1, 1, 1)

        # Perform batch matrix multiplication
        # input_data: (actual_batch_size, in_features) -> (actual_batch_size, 1, in_features)
        # simulated_weight.transpose(1, 2): (actual_batch_size, in_features, out_features)
        # Result: (actual_batch_size, 1, in_features) @ (actual_batch_size, in_features, out_features)
        #       -> (actual_batch_size, 1, out_features)
        # Squeeze to get (actual_batch_size, out_features)
        output_per_sample = torch.bmm(input_data.unsqueeze(1), simulated_weight.transpose(1, 2)).squeeze(1)

        if self.bias is not None:
            # Simulate bias perturbation as well
            simulated_bias = self.bias.unsqueeze(0) - \
                             self.previous_bias_grad_param.unsqueeze(0) * self.batch_lr_samples[:actual_batch_size].view(-1, 1)
            output_per_sample += simulated_bias

        return output_per_sample

    def update_previous_grad_storage(self) -> None:
        """
        Call this method *after* the optimizer.step() has applied the actual gradients.
        It copies the current gradients (`param.grad`) to `previous_grad_param`
        for the next iteration's LR simulation.
        """
        with torch.no_grad():
            if self.weight.grad is not None:
                self.previous_grad_param.copy_(self.weight.grad)
            if self.bias is not None and self.bias.grad is not None:
                self.previous_bias_grad_param.copy_(self.bias.grad)

class AdaptiveLROptimizer(Optimizer):
    def __init__(self, params, model, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, 
                 lr_samples_per_batch: int = 64, top_k_lr_selection: int = 5,
                 lr_update_momentum: float = 0.9):
        
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 1 <= top_k_lr_selection <= lr_samples_per_batch:
            raise ValueError("Invalid top_k_lr_selection. Must be between 1 and lr_samples_per_batch.")
        if not 0.0 <= lr_update_momentum <= 1.0:
            raise ValueError("Invalid lr_update_momentum: {}".format(lr_update_momentum))

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)

        self.lr_samples_per_batch = lr_samples_per_batch
        self.top_k_lr_selection = top_k_lr_selection
        self.lr_update_momentum = lr_update_momentum

        # Find all OptimizedLinear layers in the model
        self.optimized_layers = []
        for module in model.modules():
            if isinstance(module, OptimizedLinear):
                self.optimized_layers.append(module)
                # Initialize layer's current_lr_mean/var to optimizer's base LR
                with torch.no_grad():
                    module.current_lr_mean.fill_(lr)
                    # Start with a small variance based on the base LR
                    module.current_lr_var.fill_((lr * 0.1)**2) # e.g., 10% std dev of base LR

        if not self.optimized_layers:
            print("Warning: No OptimizedLinear layers found in the model. Optimizer will behave like standard AdamW.")

    def _get_lr_for_param(self, p):
        """Helper to get the LR for a specific parameter."""
        for group in self.param_groups:
            for _p in group['params']:
                if _p is p:
                    return group['lr']
        return None # Should not happen

    @torch.no_grad()
    def _sample_batch_lrs(self, actual_batch_size: int):
        """Generates and sets batch_lr_samples for each OptimizedLinear layer."""
        for layer in self.optimized_layers:
            # Ensure the layer's batch_lr_samples parameter is on the correct device
            layer.batch_lr_samples = layer.batch_lr_samples.to(layer.weight.device)
            layer.current_lr_mean = layer.current_lr_mean.to(layer.weight.device)
            layer.current_lr_var = layer.current_lr_var.to(layer.weight.device)

            mean_lr = layer.current_lr_mean.item()
            std_dev_lr = layer.current_lr_var.sqrt().item()

            # Sample LRs from a normal distribution. Clamp to reasonable bounds (e.g., > 0)
            lr_samples = torch.randn(actual_batch_size, device=layer.weight.device) * std_dev_lr + mean_lr
            lr_samples = torch.clamp(lr_samples, min=1e-8, max=1.0) # Ensure LRs are positive and bounded

            layer.batch_lr_samples[:actual_batch_size].copy_(lr_samples)

    @torch.no_grad()
    def _update_lr_stats(self, per_sample_losses: torch.Tensor):
        """
        Updates the current_lr_mean and current_lr_var for each OptimizedLinear layer
        based on the per_sample_losses.
        """
        if per_sample_losses.ndim != 1:
            raise ValueError("per_sample_losses must be a 1D tensor (per-sample loss).")
        
        actual_batch_size = per_sample_losses.shape[0]

        for layer in self.optimized_layers:
            # Ensure LR samples are on the same device as losses for topk
            lrs_used = layer.batch_lr_samples[:actual_batch_size].to(per_sample_losses.device)

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
            layer.current_lr_mean.data.mul_(1 - self.lr_update_momentum).add_(new_mean * self.lr_update_momentum)
            layer.current_lr_var.data.mul_(1 - self.lr_update_momentum).add_(new_var * self.lr_update_momentum)

            # Optionally, update the param_group's 'lr' based on the layer's new mean
            # This makes the "base" LR for the next AdamW step more adaptive
            # A more robust solution might require a separate param_group per OptimizedLinear
            # For simplicity, we'll just update the first param_group's LR.
            # In a real scenario, you might link specific layer's LRs to specific param groups.
            # Here, we'll update the LR of the param group containing the layer's weight.
            found_group = False
            for group in self.param_groups:
                if layer.weight in group['params']:
                    group['lr'] = layer.current_lr_mean.item()
                    found_group = True
                    break
            if not found_group:
                 # If an OptimizedLinear layer's weight isn't found in any param_group,
                 # it won't benefit from this base LR update. This should ideally not happen.
                 pass


    def step(self, closure=None, per_sample_losses: torch.Tensor = None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            per_sample_losses (torch.Tensor): A 1D tensor containing the loss
                for each sample in the current batch (reduction='none').
                This is required for the LR adaptation mechanism.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if per_sample_losses is None and self.optimized_layers:
            raise ValueError(
                "per_sample_losses argument is required when using OptimizedLinear layers "
                "for dynamic LR adaptation. Please compute losses with reduction='none'."
            )
        
        # --- 1. Generate new batch_lr_samples for the upcoming forward pass ---
        # This needs to happen *before* the model's forward pass, but the optimizer's step()
        # is typically called *after* the backward pass.
        # This implies that _sample_batch_lrs needs to be called *outside* step()
        # or `step` is called *before* the forward pass (unconventional).
        # Let's adjust: We'll put `_sample_batch_lrs` here, assuming `step` is called
        # *after* loss computation but *before* the next epoch/iteration's forward pass.
        # This means the LRs sampled in `step` are for the *next* forward pass.
        # The `per_sample_losses` here correspond to the LRs *from the previous step*.

        # To correctly link: The `batch_lr_samples` inside OptimizedLinear *must*
        # be populated *before* the forward pass that computes `per_sample_losses`.
        # So, the typical flow:
        # 1. optimizer._sample_batch_lrs(actual_batch_size) # Prepares LRs for current batch
        # 2. outputs = model(inputs) # Uses prepared LRs
        # 3. per_sample_losses = loss_fn(outputs, targets, reduction='none')
        # 4. total_loss = per_sample_losses.mean()
        # 5. total_loss.backward()
        # 6. optimizer.step(per_sample_losses=per_sample_losses) # Processes losses, updates weights
        # 7. optimizer.zero_grad() # Clears grads for next batch

        # Therefore, the current `per_sample_losses` are from the `batch_lr_samples` that were
        # active *before* this `step` call.

        # --- 2. Update LR stats based on current batch's performance ---
        if self.optimized_layers:
            self._update_lr_stats(per_sample_losses)

        # --- 3. Perform standard AdamW update ---
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('AdamW does not support sparse gradients')
                    grads.append(p.grad)

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if amsgrad:
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    state_steps.append(state['step'])

                torch.functional.adamw(params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    state_steps,
                    amsgrad,
                    beta1,
                    beta2,
                    group['lr'],
                    group['weight_decay'],
                    group['eps'])

            # Increment step counter for each parameter
            for p in params_with_grad:
                self.state[p]['step'] += 1

        # --- 4. Store current gradients as previous gradients for the next simulation ---
        for layer in self.optimized_layers:
            layer.update_previous_grad_storage()

        return loss