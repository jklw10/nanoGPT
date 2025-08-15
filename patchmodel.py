"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
import model
import utils
import wackmodel
from cut_cross_entropy import linear_cross_entropy

Linear = nn.Linear
#settings :)
#known good
ssmax           = True
#good in owt
qnorm           = False
#good in shkspr
#slow
spl             = False 
qrot            = False
think           = False

class LayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.n_embd))
        self.bias = nn.Parameter(torch.zeros(config.n_embd)) if config.bias else None 
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class Scanner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d = config.n_embd
        self.sampler = model.LearnableFourierResampler(config.n_embd * 2, config.n_embd, 64)
        

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        z = torch.cat((y, x), dim=-1)
        z=self.sampler(z)
        
        z = utils.mmnorm(z)
        return z

class SSMBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.c_attn2 =  Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)
        
        self.c_proj = Linear(config.n_embd,  config.n_embd, bias=config.bias) 
        self.scanner = Scanner(config)
        self.identity = nn.Parameter(torch.rand(config.n_embd))
        
        self.attn_dropout = nn.Dropout(config.dropout) #todo
        self.resid_dropout = nn.Dropout(config.dropout) #todo
        self.n_embd = config.n_embd
     
    def forward(self, x: torch.tensor): 
        B, T, C = x.size() 
        q,   v = self.c_attn2(x).split(self.n_embd, dim=2)
        
        
        q = q.view(B, T, C)
        v = v.view(B, T, C)

        q = utils.pscan(q, self.scanner, self.identity)
        Y = q*v 
        Y = Y.view(B, T, -1)
        Y = self.c_proj(Y)
        Y = self.resid_dropout(Y)
        return Y
    

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = Linear(config.n_embd,  config.n_embd, bias=config.bias)
        
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head 
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim =  self.n_embd //  self.n_head 
        
        self.qm = nn.Parameter(torch.ones(self.head_dim).normal_(mean=1, std=0.1))
        

    def forward(self, x, causal=True):
        B, T, C = x.size() 
        
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        y = torch.nn.functional.scaled_dot_product_attention(q*(math.log(T)*self.qm),k,  v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal= causal)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim) 
        y = self.resid_dropout(self.c_proj(y))

        return y
        


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = Linear( 4*config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.rx0 = nn.Parameter(torch.zeros(4*config.n_embd))
        self.n_embd = config.n_embd

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config)
        self.attn = Attention(config)
        self.ln_2 = LayerNorm(config)
        self.mlp = MLP(config)
        self.register_buffer("block_input",torch.zeros(config.n_embd))

    def forward(self, x):
        self.block_input = x
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def spl(self, x):
        predicted_output = self(-x, causal = False)
        sploss = F.mse_loss(
            utils.mmnorm(predicted_output.flatten()),
            utils.mmnorm(self.block_input.flatten())
        ) 
        return sploss
    
class Patcher(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_max = 20
        self.patch_embd = 64
        self.tok_emb = self.transformer.wte
        self.threshold= nn.Parameter(torch.ones(1))
        self.end_patch_token_id = config.vocab_size-1
        patcherconfig = config
        patcherconfig.n_embd = self.patch_embd
        self.pseudopatcher = SSMBlock(patcherconfig)
        
        self.embedder = SSMBlock(config)

    def loss(self, logits, patchtargets, pploss):
        logits = logits.view(
            logits.size(0), 
            logits.size(1), 
            self.patch_max, 
            self.config.vocab_size
        )   # (b, t_internal, patch_max, vocab_size)
        logits_flat = logits.contiguous().view(-1, self.config.vocab_size)  # (b * t_internal * patch_max, vocab_size)
        targets_flat = patchtargets.contiguous().view(-1)  # (b * t_internal * patch_max)
        loss = F.cross_entropy(
            logits_flat, 
            targets_flat,
            ignore_index=-1
        )
        return loss + pploss.mean()/(self.threshold/self.patch_max) #+ length_penalty
                    

    def forward(self, idx):
          #torch.compiler.cudagraph_mark_step_begin() # Add this line at the beginning of forward
        device = idx.device
        b, t = idx.size()

        device = idx.device
        b, t = idx.shape

        pos = torch.arange(0, self.patch_max, dtype=torch.long, device=device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.tok_emb(idx)
        pred_idx = self.pseudopatcher(tok_emb[:, :, :self.patch_embd])
        losses = F.mse_loss(tok_emb[:, :-1, :self.patch_embd], pred_idx[:, 1:, :], reduction='none').mean(dim=1)
        
        
        endtok = torch.as_tensor(self.end_patch_token_id,device=device,dtype=torch.long)
        patch_indices = torch.full((b, self.config.internal_block_size+2, self.patch_max+1), -1, dtype=torch.long, device=device)
        
        patch_embeds = torch.zeros((b, self.config.internal_block_size+2, self.patch_max, self.config.n_embd), device=device)
        accumulated_loss = torch.zeros(b, device=device)
        patch_deposit_idx = torch.zeros(b, dtype=torch.long, device=device)
        patch_length_idx = torch.zeros(b, dtype=torch.long, device=device)
        batch_indices = torch.arange(b, device=device)  # Explicit batch indices
        

        for current_token_index in range(t-1): 
            loss_value = losses[:, current_token_index]
            accumulated_loss = accumulated_loss + loss_value 
            cond1 = torch.gt(accumulated_loss,self.threshold  )
            cond2 = torch.ge(patch_length_idx,self.patch_max-1)
            mask = cond1 | cond2
            #if loss is greater, or max length is reached, start setting values into the next patch
            patch_deposit_idx = patch_deposit_idx + (mask) 
            #if patch doesn't end, increment patch length else reset patch length
            patch_length_idx = patch_length_idx + (~mask)
            patch_length_idx = patch_length_idx * (~mask)
            #if patch ends, reset patch accumulated loss
            accumulated_loss = accumulated_loss * (~mask)  

            patch_indices[batch_indices,patch_deposit_idx, patch_length_idx] = idx[batch_indices, current_token_index]
            patch_embeds[batch_indices, patch_deposit_idx, patch_length_idx, :] = tok_emb[batch_indices, :, current_token_index]
            
            patch_indices[batch_indices,patch_deposit_idx, patch_length_idx+1] = endtok
        
        patch_indices= patch_indices.narrow(-1,0,self.patch_max).contiguous()#trim the end, hacky, but eh.

        
        pos_emb = pos_emb.flatten(start_dim=0,end_dim=-1).unsqueeze(0).unsqueeze(0).contiguous()
        toemb = pos_emb + torch.flatten(patch_embeds, start_dim=2, end_dim=-1)
        patch_embeds = self.embedder(toemb)

        pb, pt, pe = patch_embeds.size()
        patch_embeds = patch_embeds[:,-(pt):-2,:] #(b, t_internal, n_embed)
        
        patch_targets = patch_indices[:,-(pt-1):-1,:] #(b, t_internal, patchmax)
        
        patch_embeds = patch_embeds 
        return patch_embeds, patch_targets, losses
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        block_size = self.config.block_size
        patch_max = self.patch_max
        vocab_size = self.config.vocab_size
        batch_size = idx.size(0) # Get batch size

        generated_tokens = 0
        finished_batches = torch.zeros(batch_size, dtype=torch.bool, device=idx.device) # Track finished batches
        next_indices_list = [] # List to accumulate generated indices for each batch

        # Initialize next_indices_list with current idx sequences
        for i in range(batch_size):
            next_indices_list.append(idx[i].clone()) # Clone to avoid modifying original idx

        for _ in range(max_new_tokens // patch_max):
            if generated_tokens >= max_new_tokens or torch.all(finished_batches): # Stop condition
                break

            # Prepare input for the model, handling sequence length cropping
            current_idx_batch = []
            for i in range(batch_size):
                current_seq = next_indices_list[i]
                if current_seq.size(0) > block_size:
                    current_seq = current_seq[-block_size:] # Crop if too long
                current_idx_batch.append(current_seq)
            idx_cond = torch.stack(current_idx_batch) # Stack into a batch

            logits, _ = self(idx_cond)
            logits_reshaped = logits.view(batch_size, -1, patch_max, vocab_size)
            patch_logits = logits_reshaped[:, -1, :, :]

            next_patch_tokens_batch = []
            for batch_idx in range(batch_size):
                if finished_batches[batch_idx]:
                    next_patch_tokens_batch.append(None) # Placeholder for finished batch
                    continue

                current_batch_patch_tokens = []
                for patch_pos in range(patch_max):
                    pos_logits = patch_logits[batch_idx, patch_pos, :] / temperature
                    if top_k is not None:
                        v_topk, _ = torch.topk(pos_logits, min(top_k, pos_logits.size(-1)))
                        pos_logits[pos_logits < v_topk[..., [-1]]] = -float('Inf')
                    probs = F.softmax(pos_logits, dim=-1)
                    idx_next_token = torch.multinomial(probs, num_samples=1)

                    current_batch_patch_tokens.append(idx_next_token)
                    generated_tokens += 1

                    if idx_next_token.item() == self.end_patch_token_id:
                        finished_batches[batch_idx] = True
                        break
                    if generated_tokens >= max_new_tokens:
                        break
                if not finished_batches[batch_idx]:
                    next_patch_tokens_batch.append(torch.cat(current_batch_patch_tokens))
                else:
                    next_patch_tokens_batch.append(torch.tensor([], dtype=torch.long, device=idx.device)) # Empty tensor for finished

            # Update next_indices_list with generated patch tokens
            for batch_idx in range(batch_size):
                if not finished_batches[batch_idx]:
                    patch_tokens = next_patch_tokens_batch[batch_idx]
                    next_indices_list[batch_idx] = torch.cat((next_indices_list[batch_idx], patch_tokens))


        # Stack the list of sequences back into a batch tensor for return
        #idx_result = torch.cat(next_indices_list)
        return next_indices_list
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    internal_vocab_size_multi: int = 10
    internal_vocab_size: int = 0
    internal_block_size: int = 30
    device='cuda'
    optimizer: str = 'adam'
    mem_block_size: int = 8
    batch_size = 64
    step = 0

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        config.internal_vocab_size = config.vocab_size * config.internal_vocab_size_multi
        
        self.memory_frozen = False 
        self.config = config
        self.ksize = 8
        self.patch_max = 10
        self.bembwidth = config.n_embd//2
        config.internal_block_size = config.block_size // (self.patch_max//2)
        
        blocks = nn.ModuleList([MemBlock(config) for _ in range(config.n_layer)])
        self.denoiser = MemBlock(config) 
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            
            wpe = nn.Embedding(config.internal_block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = blocks,
            ln_f = LayerNorm(config),
        ))
        if(think):
            self.thinkthreshold = nn.Parameter(torch.ones(1))
            self.thinkstop = nn.Sequential(
                Linear(config.n_embd, config.n_embd),
                nn.GELU(),
                Linear(config.n_embd, 1)
            )
        
        self.convemb = wackmodel.Patcher(config)
        self.lm_head = Linear(config.n_embd, config.vocab_size * self.patch_max, bias=False)
        

        
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, module in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(module, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        self.laim = 0.1
        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        x, patchtargets, pploss= self.convemb(idx)
        pb, pt, pe = x.size()
        
        pos = torch.arange(0, pt, dtype=torch.long, device=device)
        pos_emb = self.transformer.wpe(pos)
        x = x + pos_emb
        
        x = self.transformer.drop(x)

        if qnorm:
            x = quatnorm(x) 
        for i, block in enumerate(self.transformer.h):
            x = block(x) 

        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = None# self.lm_head(x)
            
            loss = self.convemb.loss(logits, patchtargets, pploss)

            if self.training:
                pass #auxlosses later
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
            #{'params': self_lr, 'lr': 1.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        match self.config.optimizer:
            case "adam":
                optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
            case "sgd":
                optimizer = torch.optim.SGD(optim_groups, lr=learning_rate, **extra_args)
            case "phi2":
                optimizer = wackmodel.Phi2Optimizer(optim_groups,lr=learning_rate, **extra_args)
            
            #optimizer =torch.optim.Adadelta(optim_groups)#, **extra_args)# lr=learning_rate, **extra_args)
        print(f"using fused AdamW: {use_fused}")
                        
        return optimizer

    def get_patch_lengths(self, logits, end_token_idx, seq_len):
        """Returns tensor of patch lengths (batch_size,) based on first end token occurrence"""
        # Find first occurrence of end token in each sequence
        end_token_positions = (logits.argmax(-1) == end_token_idx).int().argmax(dim=1)

        # Handle sequences with no end token (use full length)
        no_end_token_mask = (end_token_positions == 0)

        # Use torch.where to conditionally assign seq_len without in-place modification
        end_token_positions = torch.where(no_end_token_mask, seq_len, end_token_positions)

        return end_token_positions + 1 
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    
