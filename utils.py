
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

#surely a file with this name hasn't been done before.

#everything here is probably a candidate to be made into a kernel.

def scale(n_embd, step, start_perc, max):
    return math.ceil(n_embd * min(start_perc+ (1.0-start_perc)*(step/max),1.0 ))

def funnyMulti(x, y):
    return torch.sign(x) * torch.sqrt(torch.abs(x * y) + 1e-10)

def unfunnyMulti(x, y):
    return torch.sign(x) * torch.abs((x ** 2) / (y + 1e-10))

@torch.compile(backend='inductor', mode='max-autotune')
def mmnorm(x, dim = -1, scale = None):  
    if(scale is None):
        scale = 1 + 2 / x.nelement() #changed recently
    w_avg = x.mean(dim=dim, keepdim=True)
    w_range = torch.abs(x.max(dim=dim, keepdim=True)[0] - x.min(dim=dim, keepdim=True)[0] + 1e-10)
    return ((x - w_avg) / w_range) * scale

def fast_sin_leakyrelu(x):
     base = F.leaky_relu(x)
     sine_mod = F.relu(torch.sign(x)) * torch.sin(1.25 * x)
     return base + sine_mod


def oozing_floor(x):
     base = F.leaky_relu(x)
     oozing_floor = F.relu(torch.sign(x)) * torch.floor(1.25 * x)
     return base + oozing_floor

def fast_sin_gelu_leaky(x, leaky=0.01):
     base = F.gelu(x)
     sine_mod = torch.where(x >= 0, torch.sin(1.25 * x), x * leaky)
     return base + sine_mod


def fast_sin_gelu(x):
     base = F.gelu(x)
     sine_mod = torch.relu(torch.sign(x)) * torch.sin(1.25 * x)
     return base + sine_mod

def fast_sin_gelu_sinleak(x, leaky=0.01):
     base = F.gelu(x)
     lh = F.leaky_relu(torch.sign(x),negative_slope=leaky)
     sine_mod = lh * torch.sin(1.25 * x)
     return base + sine_mod

def modified_sin_gelu_leaky(x, leaky=0.01):
     sqpi = torch.sqrt(torch.ones_like(x)*2/torch.pi)
     base = 0.5 * x * (1 + F.tanh(sqpi * (x + 0.044715 * x**3)))
     sine_mod = torch.where(x >= 0, torch.sin(1.25 * x), x * leaky)
     return base + sine_mod
def create_memory_causal_mask(memory_length, incoming_length):
    # Create a mask that allows attending to memory tokens and implements
    # causal attention for the incoming tokens
    total_length = memory_length + incoming_length
    mask = torch.zeros((incoming_length, total_length), dtype=torch.bool)
    
    # Allow attending to all memory tokens
    mask[:, :memory_length] = True
    
    # For incoming tokens, implement causal masking
    for i in range(incoming_length):
        mask[i, memory_length:memory_length+i+1] = True
    
    # Convert to attention mask format (False where attention is allowed)
    return ~mask
def create_memory_causal_mask4(memory_length, incoming_length):
    total_length = memory_length + incoming_length
    mask = torch.zeros((total_length,incoming_length))
    mask[incoming_length:,:] = torch.triu(
        torch.ones((incoming_length, incoming_length), dtype=torch.float) * float('-inf'),
        diagonal=1
        )
    return mask

def create_memory_causal_mask2(memory_length, incoming_length):
    total_length = memory_length + incoming_length
    mask = torch.zeros((total_length,total_length))
    mask[incoming_length:,incoming_length:] = torch.triu(
        torch.ones((incoming_length, incoming_length), dtype=torch.float) * float('-inf'),
        diagonal=1
        )
    #mask[incoming_length:,:incoming_length] = torch.triu(
    #    torch.ones((incoming_length, incoming_length), dtype=torch.float) * float('-inf'),
    #    diagonal=1
    #    )
    return mask

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

@torch.compile(backend='inductor', mode='max-autotune')
def parallel_quaternion_scan_log_n(q_in: torch.Tensor) -> torch.Tensor:
    original_shape = q_in.shape
    if q_in.dim() == 2:
        q_in = q_in.unsqueeze(0).unsqueeze(2)

    B, T, C, _ = q_in.shape
    device = q_in.device

    next_pow_of_2 = 2**math.ceil(math.log2(T))
    
    if T == next_pow_of_2:
        x = q_in
    else:
        padding_size = next_pow_of_2 - T
        identity_padding = torch.zeros(B, padding_size, C, 4, device=device, dtype=q_in.dtype)
        identity_padding[..., 0] = 1.0
        x = torch.cat([q_in, identity_padding], dim=1)

    padded_q_original = x.clone()
    
    num_levels = int(math.log2(x.shape[1]))

    # --- 2. Up-Sweep (Reduction) Phase ---
    for d in range(num_levels):
        stride = 2**d
        source_up = x[:, :x.shape[1]-stride, :, :]
        target_up = x[:, stride:, :, :]
        res = quaternion_multiply(source_up, target_up)
        mask_indices = torch.arange(2*stride - 1, x.shape[1], 2*stride, device=device)
        x[:, mask_indices] = res[:, mask_indices - stride]

    # --- 3. Down-Sweep (Scan) Phase ---
    x[:, -1, :, :] = 0.0
    x[:, -1, :, 0] = 1.0

    for d in range(num_levels - 1, -1, -1):
        stride = 2**d
        
        mask_indices = torch.arange(stride - 1, x.shape[1] - stride, 2*stride, device=device)
        
        left_val = x[:, mask_indices]
        right_val = x[:, mask_indices + stride]
        
        res = quaternion_multiply(right_val, left_val)#it just makes sense
        
        x[:, mask_indices + stride] = res
        x[:, mask_indices] = right_val

    # --- 4. Final Conversion to Inclusive Scan ---
    inclusive_scan_padded = quaternion_multiply(x, padded_q_original)
    
    final_result = inclusive_scan_padded[:, :T, ...].reshape(original_shape)
    
    return final_result

@torch.compile(backend='inductor', mode='max-autotune')
def pscan(x_in: torch.Tensor, func: nn.Module, ident: torch.Tensor) -> torch.Tensor:
    original_shape = x_in.shape
    B, T, C = original_shape
    device = x_in.device
    dtype = x_in.dtype

    # --- 1. Padding ---
    next_pow_of_2 = 2**math.ceil(math.log2(T))
    if T == next_pow_of_2:
        x = x_in
    else:
        padding_size = next_pow_of_2 - T
        identity_padding = ident.expand(B, padding_size, C)
        x = torch.cat([x_in, identity_padding], dim=1)

    a = x.clone()
    num_items = a.shape[1]
    num_levels = int(math.log2(num_items))

    # --- 2. Up-Sweep (Reduction Phase) ---
    for d in range(num_levels):
        stride = 2**d
        indices = torch.arange(2*stride - 1, num_items, 2*stride, device=device)

        # Use index_select (functional) instead of direct indexing for clarity
        left_operands = torch.index_select(a, 1, indices - stride)
        right_operands = torch.index_select(a, 1, indices)
        
        # Use index_put (functional) instead of in-place assignment
        # This creates a new 'a' tensor, which torch.compile can handle.
        # We need to provide indices for the batch dimension too.
        batch_indices = torch.arange(B, device=device).unsqueeze(1)
        # The indices tuple tells index_put which elements to update: (batch_idx, sequence_idx)
        a = torch.index_put(a, (batch_indices, indices), func(left_operands, right_operands))

    # --- 3. Down-Sweep (Scan Phase) ---
    # **CRITICAL CHANGE**: Replace slice assignment with torch.cat
    # This is much friendlier to the compiler.
    a = torch.cat([a[:, :-1, :], ident.expand(B, 1, C)], dim=1)

    for d in range(num_levels - 1, -1, -1):
        stride = 2**d
        indices = torch.arange(2*stride - 1, num_items, 2*stride, device=device)
        batch_indices = torch.arange(B, device=device).unsqueeze(1)

        # Select the values we need *before* any modification
        left_val = torch.index_select(a, 1, indices - stride)
        right_val = torch.index_select(a, 1, indices)

        # **CRITICAL CHANGE**: Perform the swap and update using two index_put calls
        # 1. Put the right_val into the left_pos
        a = torch.index_put(a, (batch_indices, indices - stride), right_val)
        # 2. Put the result of func(right_val, left_val) into the right_pos
        a = torch.index_put(a, (batch_indices, indices), func(right_val, left_val))

    inclusive_scan_padded = func(a, x)
    final_result = inclusive_scan_padded[:, :T, :].reshape(original_shape)
    
    return final_result

@torch.compile(backend='inductor', mode='max-autotune')
def pscan2(x_in: torch.Tensor, func: nn.Module, ident: torch.tensor) -> torch.Tensor:
    original_shape = x_in.shape
    B, T, C = original_shape
    device = x_in.device
    dtype = x_in.dtype

    next_pow_of_2 = 2**math.ceil(math.log2(T))
    
    if T == next_pow_of_2:
        x = x_in
    else:
        padding_size = next_pow_of_2 - T
        identity_padding = ident.expand(B, padding_size, C)
        x = torch.cat([x_in, identity_padding], dim=1)

    a = x.clone()
    num_items = a.shape[1]
    num_levels = int(math.log2(num_items))

    for d in range(num_levels):
        stride = 2**d
        indices = torch.arange(2*stride - 1, num_items, 2*stride, device=device)
        
        left_operands = a[:, indices - stride, :]
        right_operands = a[:, indices, :]
        
        a[:, indices, :] = func(left_operands, right_operands)


    a[:, -1, :] = ident.expand(B, 1, C).clone()

    # At each level d (from top down), we propagate values
    for d in range(num_levels - 1, -1, -1):
        stride = 2**d
        indices = torch.arange(2*stride - 1, num_items, 2*stride, device=device)
        
        # Select the left sum (temp) and the right element (which will become the new left sum)
        temp = a[:, indices - stride, :]
        
        # Update the left element with the value from the right
        a[:, indices - stride, :] = a[:, indices, :]
        
        # Update the right element by combining the stored temp value with the new left value
        a[:, indices, :] = func(a[:, indices, :], temp)

    # The result `a` is now an EXCLUSIVE scan.
    # To make it INCLUSIVE, we combine it with the original input sequence.
    inclusive_scan_padded = func(a, x)
    
    # --- 4. Unpad and Reshape ---
    final_result = inclusive_scan_padded[:, :T, :].reshape(original_shape)
    
    return final_result

@torch.compile(backend='inductor', mode='max-autotune')
def quaternion_inverse(q: torch.Tensor) -> torch.Tensor:
    q_conj = q * torch.tensor([1., -1., -1., -1.], device=q.device, dtype=q.dtype)
    q_norm_sq = torch.sum(q * q, dim=-1, keepdim=True)
    return q_conj / (q_norm_sq + 1e-8)

@torch.compile(backend='inductor', mode='max-autotune')
def lerp(q1, q2, t):
    return (1.0 - t) * q1 + t * q2

@torch.compile(backend='inductor', mode='max-autotune')
def qlerp(q, qident, t):
    "lerp, but a little safer for quaternions"
    dot = torch.sum(qident * q, dim=-1, keepdim=True)
    q = q * ((dot > 0.0).float()*2.0 - 1.0 )
    return lerp(q, qident, t)

@torch.compile(backend='inductor', mode='max-autotune')
def qmlerp(q, qident, s):
    """lerp for sequential quaternion rotations, using previous as a tie breaker, a pseudo momentum, actual momentum is also an interesting idea """
    B, T, C, Q = q.size()
    q_prev = torch.cat([
        qident.expand(B, 1, C, Q), 
        q[:, :-1]
    ], dim=1)
    dot_product = torch.sum(q_prev * q, dim=-1, keepdim=True)
    sign = torch.where(dot_product < 0.0, -1.0, 1.0)
    q=q*sign
    return  (1.0 - s) * qident + s * q

@torch.compile(backend='inductor', mode='max-autotune')
def maprotgate(q: torch.Tensor) -> torch.Tensor:
    B, T, C, Q = q.size() 
    rot = q[..., 1:]
    s = torch.sigmoid(q[..., 0].view(B, T, C, 1))

    return exponential_map(rot*s) 

@torch.compile(backend='inductor', mode='max-autotune')
def exponential_map(rot: torch.Tensor, epsilon=1e-8) -> torch.Tensor:
    
    angle = torch.norm(rot, p=2, dim=-1, keepdim=True)
    
    axis = rot / (angle + epsilon)
    
    half_angle = angle / 2
    q_w = torch.cos(half_angle)
    
    q_xyz = axis * torch.sin(half_angle)
    
    return torch.cat([q_w, q_xyz], dim=-1)

@torch.compile(backend='inductor', mode='max-autotune')
def slerp(q1, q2, t, epsilon=1e-7):
    dot = torch.sum(q1 * q2, dim=-1, keepdim=True)

    q2_corr = q2 * ((dot > 0.0).float() * 2.0 - 1.0)

    dot = torch.abs(dot)
    angle = torch.acos(torch.clamp(dot, -1.0, 1.0))
    
    sin_angle = torch.sin(angle)
    
    c1 = torch.sin((1.0 - t) * angle) / (sin_angle + epsilon)
    c2 = torch.sin(t * angle) / (sin_angle + epsilon)
    
    q_slerp = c1 * q1 + c2 * q2_corr
    
    q_lerp = (1.0 - t) * q1 + t * q2_corr
    
    is_close = dot > 0.9995 
    return torch.where(is_close, q_lerp, q_slerp)

@torch.compile(backend='inductor', mode='max-autotune')
def scan_quaternion_multiply_window(
    q: torch.Tensor, 
    window_size: int
) -> torch.Tensor:
    B, T, H, _ = q.shape
    if not (isinstance(window_size, int) and window_size > 0):
        raise ValueError("window_size must be a positive integer.")
    
    if window_size == 1:
        return q.clone()

    cum_prod = parallel_quaternion_scan_log_n(q.clone()) 

    if window_size >= T:
        return cum_prod

    identity_q = torch.tensor([1., 0., 0., 0.], device=q.device, dtype=q.dtype)
    identity_q = identity_q.view(1, 1, 1, 4).expand(B, 1, H, 4)
    
    p_t_minus_k = torch.cat(
        (identity_q, cum_prod[:, :T - window_size, :, :]), 
        dim=1 # Concatenate along the time dimension
    )
    
    inv_p_t_minus_k = quaternion_inverse(p_t_minus_k)
    
    p_t = cum_prod[:, window_size - 1:, :, :]
    
    sliding_results = quaternion_multiply(inv_p_t_minus_k, p_t)
    
    initial_results = cum_prod[:, :window_size - 1, :, :]
    
    final_result = torch.cat((initial_results, sliding_results), dim=1)
    
    return final_result

def naive_scan_quaternion_multiply_window(
    q: torch.Tensor, 
    window_size: int
) -> torch.Tensor:
    """
    Ground truth to check against if touching the window scan
    """
    B, T, H, _ = q.shape
    outputs = []
    for t in range(T):
        start_idx = max(0, t - window_size + 1)
        window_q = q[:, start_idx : t + 1, :, :]
        current_product = window_q[:, 0, :, :]
        for i in range(1, window_q.shape[1]):
            next_q = window_q[:, i, :, :]
            current_product = quaternion_multiply(current_product, next_q)
        outputs.append(current_product)
    return torch.stack(outputs, dim=1)

@torch.compile(backend='inductor', mode='max-autotune')
def parallel_quaternion_scan(q: torch.Tensor) -> torch.Tensor:
    T = q.shape[1]
    cumulative_prod = torch.empty_like(q)
    if T == 0:
        return cumulative_prod
        
    cumulative_prod[:, 0, :, :] = q[:, 0, :, :]

    for i in range(1, T):
        cumulative_prod[:, i, :, :] = quaternion_multiply(cumulative_prod[:, i-1, :, :], q[:, i, :, :])
    
    return cumulative_prod

@torch.compile(backend='inductor', mode='max-autotune')
def qcumprod(q: torch.Tensor) -> torch.Tensor:
    T = q.shape[1]
    cumulative_prod = torch.empty_like(q[:, 0, :, :].view(1,1,1,4))
    if T == 0:
        return cumulative_prod
        
    cumulative_prod= q[:, 0, :, :]

    for i in range(1, T):
        cumulative_prod = quaternion_multiply(cumulative_prod, q[:, i, :, :])
    
    return cumulative_prod

@torch.compile(backend='inductor', mode='max-autotune')
def qumgate(q: torch.Tensor) -> torch.Tensor:
    T = q.shape[1]
    scan = parallel_quaternion_scan_log_n(q)
    output = torch.empty_like(q)

    #the nth token would be multiplied with all the previous tokens,
    #those multiply results are now the gates for the next cumulative product, up to the nth token.
    #and so for each token i need cumulative product up to that token, not quite  O(N) is that o log n ?
    #and this would still be causal because even though for N token, the gate for N-X is non causal, that gate only knows up to Nth token.


    for x in range( T):
        gates = torch.empty_like(q[:, :x, :, :])
        for y in range(x):
            gates[:, y, :, :] = maprotgate(quaternion_multiply(scan[:, y, :, :] ,scan[:, x, :, :]))
        gates = quaternion_multiply(gates, q[:, :x, :, :])
        output[:, x, :, :] = qcumprod(gates[:, :x, :, :])

    
    return output


@torch.compile(backend='inductor', mode='max-autotune')
def quaternion_multiply(q, p):
    # q and p are tensors of shape [..., 4]
    # Ensure they are contiguous, which is good practice anyway
    q = q.contiguous()
    p = p.contiguous()

    # Extract components
    w1, x1, y1, z1 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    w2, x2, y2, z2 = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    
    # Calculate the product components directly
    # This is the "Hamilton product"
    w_new = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x_new = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y_new = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z_new = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w_new, x_new, y_new, z_new], dim=-1)

@torch.compile(backend='inductor', mode='max-autotune')
def zca_newton_schulz(G, epsilon=1e-5, steps=5, power_iters=10):
    if G.ndim < 2:
        return G
    G = G - G.mean(dim=0, keepdim=True)
    n, d = G.shape
    device, dtype = G.device, G.dtype
    
    # Compute covariance matrix with regularization
    Cov = (1/n) * torch.matmul(G.T, G) + epsilon * torch.eye(d, device=device, dtype=dtype)
    
    # Estimate spectral norm using power iteration
    v = torch.randn(d, 1, device=device, dtype=dtype)
    for _ in range(power_iters):
        u = torch.matmul(Cov, v)
        u = u / u.norm()
        v = torch.matmul(Cov.T, u)
        v = v / v.norm()
    s = torch.matmul(u.T, torch.matmul(Cov, v))#.squeeze()
    
    # Scale Cov to ensure convergence
    B = Cov / s
    Y = torch.eye(d, device=device, dtype=dtype)
    
    # Newton-Schulz iterations for inverse square root of B
    for _ in range(steps):
        Y = 0.5 * torch.matmul(Y, 3 * torch.eye(d, device=device, dtype=dtype) - torch.matmul(Y, torch.matmul(B, Y)))
    
    # Rescale to get Cov^{-1/2}
    W = Y / (s ** 0.5)
    
    return torch.matmul(G, W)


@torch.compile(backend='inductor', mode='max-autotune')
def GradOrth(param, adjusted_grad):
    w = param.view(-1)
    g = adjusted_grad.view(-1)
    w_norm_sq = torch.dot(w, w) + 1e-30
    proj = torch.dot(w, g) / w_norm_sq
    g_orth = g - proj * w
    g_norm = g.norm(2)
    g_orth_norm = g_orth.norm(2) + 1e-30
    g_orth_scaled = g_orth * (g_norm / g_orth_norm)
    adjusted_grad = (g_orth_scaled.view_as(adjusted_grad))
    return adjusted_grad

@torch.compile(backend='inductor', mode='max-autotune')
def fftnorm(grad,center=0.5,sigma=0.5, dim=None):
    if(grad.ndim>=2):
        adjusted_grad = torch.fft.fft2(grad,dim=dim) 
        #bump width centered on 0 (phase), width of sigma 0.1
        gk = gaussian_kernel(adjusted_grad.real, center, sigma, dim = dim)
        adjusted_grad = torch.complex(gk*(adjusted_grad.real),gk*(adjusted_grad.imag))
        return torch.fft.ifft2(adjusted_grad,dim=dim).real
    else:
        adjusted_grad = torch.fft.fft(grad,dim=dim) 
        #bump width centered on 0 (phase), width of sigma 0.1
        gk = gaussian_kernel(adjusted_grad.real, center, sigma)
        #print(gk)
        adjusted_grad = torch.complex(gk*(adjusted_grad.real),gk*(adjusted_grad.imag))
        return torch.fft.ifft(adjusted_grad).real

@torch.compile(backend='inductor', mode='max-autotune')
def fft_bmean_csquish_add(x, x2):
    b,t,c = x.size()
    halfft2 = torch.fft.fft(x2,dim=-1)
    halfft = torch.fft.fft(x,dim=-1)
    xc = halfft.mean(dim=0).unsqueeze(0)
    xc = torch.complex(xc.real[:,:,c//4:-c//4],xc.imag[:,:,c//4:-c//4])
    if((x2 != 0).any()):
        xc += halfft2
        xc /= 2
    
    return torch.fft.ifft(xc,dim=-1).real


#def fft_bmean_csquish_add(x, x2):
#    b,t,c = x.size()
#    halfft2 = torch.fft.fft(x2,dim=-1)
#    halfft = torch.fft.fft(x,dim=-1)
#    xc = halfft
#    if((x2 != 0).any()):
#        xc += halfft2
#        xc /= 2
#    xc = torch.complex(xc.real[:,:,c//4:-c//4],xc.imag[:,:,c//4:-c//4])
#    
#    return torch.fft.ifft(xc,dim=-1).real

def fft_bmean_tsquish_add(x, x2):
    b,t,c = x.size()
    halfft2 = torch.fft.fft(x2,dim=1)
    halfft = torch.fft.fft(x,dim=1)
    xc = halfft.mean(dim=0).unsqueeze(0)
    xc = torch.complex(xc.real[:,t//4:-t//4,:],xc.imag[:,t//4:-t//4,:])
    if((x2 != 0).any()):
        xc += halfft2
        xc /= 2
    
    return torch.fft.ifft(xc,dim=1).real

@torch.compile(backend='inductor', mode='max-autotune')
def calculate_linear_chaos_loss(logits: torch.tensor, balance_lambda= 0.1 , target_chaos= 0.1 ) -> float:
    logits = logits.float()
    vocab_size = logits.size(-1)

    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities + 1e-9

    # 2. Calculate Shannon Entropy: H(P) = -Σ p * log(p)
    entropy = -torch.sum(probabilities * torch.log(probabilities), dim=-1)

    # 3. Calculate Perplexity: exp(H(P))
    perplexity = torch.exp(entropy)

    # 4. Normalize Perplexity to get LinearChaos [0, 1]
    # This gives us our linear metric.
    linear_chaosness = perplexity / vocab_size

    # 5. Calculate the balance loss
    # We want the average chaosness of the batch to be near our target.
    # Using MSE for a smooth, differentiable loss.
    # .mean() averages the chaosness across all token predictions in the batch.
    l_balance = (linear_chaosness.mean() - target_chaos)**2
    
    return l_balance * balance_lambda, linear_chaosness.mean()

@torch.compile(backend='inductor', mode='max-autotune')
def fft_trunc_tsquish(x):
    b,t,c = x.size()
    halfft = torch.fft.fft(x,dim=1)
    xc = torch.complex(halfft.real[:,t//4:-t//4,:],halfft.imag[:,t//4:-t//4,:])
    return torch.fft.ifft(xc,dim=1).real

@torch.compile(backend='inductor', mode='max-autotune')
def rfft_trunc_squish(x, target_dim=None, dim=-1, band = "low"):
    c = x.size(dim)

    if target_dim is None:
        target_dim = c // 2
    
    ff_x = torch.fft.rfft(x, dim=dim)
    
    match band:
        case "low":
            start = 0
            end = target_dim
        case "mid":
            center = c//2
            start = center - target_dim//2
            end = center + target_dim//2
        case "high":
            start = -target_dim
            end = -1
        case _:
            pass
    
    slicer = [slice(None)] * x.ndim 
    slicer[dim] = slice(start, end)
    
    ff_x_trunc = ff_x[tuple(slicer)]

    o = torch.fft.irfft(ff_x_trunc, dim=dim).real

    return o


@torch.compile(backend='inductor', mode='max-autotune')
def fft_trunc_csquish(x, target_dim = None, band = "low"):
    b,t,c = x.size()
    halfft = torch.fft.rfft(x,dim=-1)
    if(target_dim is None):
        target_dim = c//2
    match band:
        case "low":
            start = 0
            end = target_dim
        case "mid":
            center = c//2
            start = center - target_dim//2
            end = center + target_dim//2
        case "high":
            start = -target_dim
            end = -1
        case _:
            pass

    xc = torch.complex(halfft.real[:,:,start:end],halfft.imag[:,:,start:end])
    o = torch.fft.irfft(xc,target_dim,dim=-1).real
    return o 
    
def gaussian_kernel(grad, center_offset: float, sigma = 3.0, dim=None) -> torch.Tensor:
    size = grad.size(dim)
    device= grad.device
    dtype= grad.dtype
    
    # Calculate total elements and validate input
    num_elements = torch.prod(torch.tensor(size, dtype=dtype, device=device))
    # Generate position indices
    indices = torch.arange(num_elements, dtype=dtype, device=device)
    
    # Calculate Gaussian parameters
    center = center_offset * (num_elements - 1)
    #max_distance = max(center, (num_elements - 1) - center)
    sigma = sigma * (num_elements - 1)
    
    # Compute Gaussian values and reshape
    kernel = torch.exp(-(indices - center).pow(2) / (2 * sigma**2))
    return kernel.view(size)



def mmdiem( a :torch.Tensor,b :torch.Tensor):
    numel = a.numel()# torch.sqrt(torch.ones(1,device=a.device,dtype=a.dtype)* a.numel())
    afl = a.flatten()
    bfl = b.flatten()
    arange = afl.max() - afl.min() + 1e-10
    brange = bfl.max() - bfl.min() + 1e-10
    anorm = (afl - afl.mean()) / (arange)
    bnorm = (bfl - bfl.mean()) / (brange)
    variance = torch.sqrt(a.var()*b.var()+1e-10)
    return torch.dot(anorm, bnorm) /  numel / variance # + (torch.abs(arange-brange)) + torch.abs(a.var() - b.var())

def fast_sin_relu(x):
     base = F.relu(x)
     sine_mod = F.relu(torch.sign(x)) * torch.sin(1.25 * x)
     return base + sine_mod


def quaternionize(x):
    b,t,n=x.shape
    x4d = x.view(b, t, n//4, 4)
    x_norm = x4d.norm(dim=-1, keepdim=True) 
    x_normalized = x4d / x_norm             
    return x_normalized