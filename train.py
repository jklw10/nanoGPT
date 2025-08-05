"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import gc
import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import torch._dynamo
import torch._inductor.config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from model import MLP, GPTConfig, GPT
import optim
import utils

from torch.utils.data import Dataset, DataLoader

torch._dynamo.optimize()
#torch._inductor.config.force_disable_caches = True
torch._inductor.config.disable_cpp_codegen = True
#torch._inductor.config.triton = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
train_logging = False
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------



#torch._dynamo.reset() #in case of cache corruption throw it off a bridge.

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

data_dir = os.path.join('data', dataset)
def get_batch(split, sequential = False):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    #print(f"datalen {len(data)}")
    if(sequential):
        max_start_index = len(data) - block_size - batch_size
        start_ix = torch.randint(max_start_index, (1,)).item()
        ix = range(start_ix, start_ix + batch_size)
    else:
        ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    #outp = model.generate(X, 100, temperature=0.01, top_k=200)
    
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    
    return x, y

meta_path = os.path.join('data', 'shakespeare_char/meta.pkl')
    
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
def encode(s): [stoi[c] for c in s]
def decode(ll):
    decoded = []
    for i in ll:
        if i in itos:
            decoded.append(itos[i])
        else:
            decoded.append('[UNK]')  # Or another placeholder for unknown tokens
    return ''.join(decoded)

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

force_gc = False
#switches
best                = False

mnorm_enabled       = False
#wack
dfw_enabled         = False
decor_enabled       = False
#rough
wnorm_enabled       = False
gdiff_enabled       = False
grokfast            = False
sign_enabled        = False

#??
gradorth            = True
wwhite_enabled      = False

gsphere             = False
ndiff_enabled       = False
gchaos              = False #or best
ggauss              = False #or best

decay_wa            = False
#known good
gfft                = False #or best
gnorm               = False #or best
gzca_enabled        = False #or best
swnorm_enabled      = False or best


muon                = False
#for fftmem owt: 1e-5
#for fftmem skspr: 5e-5
#swna                = 1.0e-5
#swwa                = 1.0e-5
zcastep             = 2 #2, 5
szcapow             = 2 #2, 10

ghook = gdiff_enabled or ndiff_enabled or gnorm or dfw_enabled or grokfast or gfft or gzca_enabled or sign_enabled or gsphere or ggauss or gchaos or gradorth or muon

decay = False
decaying = 1.0

eigenInit = False

lrfinder = False




if(ghook):
    gemas = None
    wemas = None
    if(gdiff_enabled or dfw_enabled or muon):
        if init_from!='resume':
            wemas = [p.clone().detach()for p in model.parameters()]

    if (grokfast):
        gemas = [torch.zeros_like(p) for p in model.parameters()]
    
    for i, p in enumerate(model.parameters()):
        if p.requires_grad:
            wema = None if wemas is None else wemas[i]
            gema = 0.0 if gemas is None else gemas[i]
            
            p.register_hook(lambda grad, p=p, gema=gema, wema=wema: custom_gradient_adjustment(grad, p, gema=gema, wema=wema))


@torch.compile(backend='inductor', mode='max-autotune')
def custom_gradient_adjustment(grad, param, wema = None, gema = None):
    if(grad is None):
        return
    input_size = grad.size(0)
    adjusted_grad = torch.where(torch.isnan(grad), torch.zeros_like(grad), grad)
    
    
    if(gchaos):
        adjusted_grad += torch.rand_like(adjusted_grad)*1e-3
    if(gnorm): 
        
        grad_avg = adjusted_grad.nanmean()
        grad_range = adjusted_grad.max() - adjusted_grad.min() + 1e-10
        adjusted_grad = ((adjusted_grad - grad_avg) / grad_range)  * (2 - 2 / input_size) 
    if(gchaos):
        adjusted_grad += torch.rand_like(adjusted_grad)*1e-3
    d2 = not (adjusted_grad.ndim < 2)
    if(gsphere and d2): 
        adjusted_grad = adjusted_grad / adjusted_grad.norm() # * (2 - 2 / input_size) 
    if(gfft):
        adjusted_grad = utils.rfft_trunc_squish(adjusted_grad)
    if(sign_enabled):
        adjusted_grad = torch.sign(torch.round(adjusted_grad)) 
    if(gzca_enabled and d2):
        adjusted_grad = utils.zca_newton_schulz(adjusted_grad, zcastep, szcapow)
    if(gradorth and d2):
        adjusted_grad = utils.orthogonalize_gradients(adjusted_grad,param)
    if(ndiff_enabled):
        w_range = torch.abs(p.max() - p.min() + 1e-10)
        w_avg = param.nanmean()
        norm = ((param - w_avg) / w_range) * (1 + 2 / input_size) 
        normgrad = norm - param
        adjusted_grad *= torch.abs(adjusted_grad - normgrad)

        #weight_ema += 0.01 * (adjusted_grad - weight_ema)
    
    
    if(gdiff_enabled or dfw_enabled):
        wema += 0.01 * (adjusted_grad - wema)

    #if(gdiff_enabled ):
    #    awdiff = weight_ema - adjusted_grad 
    #    gn = adjusted_grad.flatten().norm()
    #    eman = weight_ema.flatten().norm()
    #    awdot = torch.dot(weight_ema.flatten()/eman, adjusted_grad.flatten()/gn)
    #    #npar = torch.abs(param)
    #    npar = torch.abs(eman-gn)# 1.0-awdot + # torch.square(torch.abs(awdiff)) 
    #    gdiff = 1.0 / torch.abs(npar) 
    #    adjusted_grad *= gdiff 

    if(gdiff_enabled and d2):
        awdiff = wema - param 

        npar = torch.abs(wema)
        npar += torch.abs(adjusted_grad - awdiff) 
        gdiff = 1.0 / torch.abs(npar)
        adjusted_grad *= gdiff 
    
    if(grokfast):
        gema += 0.9999 * (adjusted_grad - gema)
        adjusted_grad += gema * 2
        adjusted_grad /= 3
    if(ggauss):
        gema += 0.1

        gk = utils.gaussian_kernel(adjusted_grad,torch.sin(gema),3)
        adjusted_grad = gk * adjusted_grad
    return adjusted_grad 



# initialize a GradScaler. If enabled=False scaler is a no-op
#scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
scaler = torch.amp.GradScaler(device_type, enabled=(dtype == 'float16'))


# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    #print(torch._dynamo.list_backends())
    unoptimized_model = model
    model = torch.compile(model, backend="inductor") # requires PyTorch 2.0 backend="inductor" is fast, backend="cudagraphs" is debuggable
    print("compiled")


# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    torch.compiler.cudagraph_mark_step_begin()
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            
            #torch.compiler.cudagraph_mark_step_begin()
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.detach().mean()
    model.train()

    del X
    del Y
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)
    return out

lr_sin = False

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    if(lr_sin):
        return min_lr + (math.cos(decay_ratio*math.pi*5)*learning_rate -min_lr)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
#X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0


scaler.is_enabled = False
scaler = None
if(force_gc):
    gc.collect()
    torch.cuda.empty_cache()

def model_step(iter_num, tl, best_val_loss):
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if iter_num % eval_interval == 0 and master_process: 
        if(iter_num > 0):
            
            if(force_gc):
                gc.collect()
                torch.cuda.empty_cache()
            
            t1 = time.time()
            losses = estimate_loss()
            t2 = time.time()
            saved = False
            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu*100, # convert to percentage
                })
            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val'].detach()
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                saved = True
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
            
            print(f"train loss {losses['train']:.4f}, val loss {losses['val']:.4f} : step {iter_num:{len(str(max_iters))}d} : saved {saved} , total time {(t1-tl)*1000:.2f}ms, validation time {(t2-t1)*1000:.2f}ms")
            
        
        if(force_gc):
            torch.cuda.empty_cache()
    if iter_num == 0 and eval_only:
        return 0, False
    
    
    X, Y = get_batch('train',True)
    def closure():
        torch.compiler.cudagraph_mark_step_begin()
        logits, loss = model(X, Y)
        loss = loss / gradient_accumulation_steps 
        loss = loss.mean(dim=0)
        #loss.mean().backward()
        return loss
    l1 = closure()
    l1.mean().backward()
    #loss.backward()
    
    if(force_gc):
        torch.cuda.empty_cache()  
    
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    for mod in model.modules():
        if isinstance(mod, optim.OptimizedLinear):
            pass
            #mod.prestep(loss)

    # step the optimizer and scaler if training in fp16
    optimizer.step()
    
    #optimizer.zero_grad(set_to_none=True)
    #l2 = closure()
    #rdiff = (l2 - l1.detach() )
    #loss = l2 * (rdiff-rdiff.mean())
    #loss.mean().backward()
    #optimizer.step()

    for mod in model.modules():
        if isinstance(mod, optim.OptimizedLinear):
            mod.poststep()

    optimizer.zero_grad(set_to_none=True)

    with torch.no_grad():
        
        if(decay_wa):
            lrgpu = torch.tensor(lr,device='cuda') 
        if(lrfinder):
            lr = 1
        if(swnorm_enabled):
            if(decay_wa):
                swna = lrgpu/20.0
            else:
                swna = 1e-5
            utils.softwnorm(model, swna)
        if(wwhite_enabled):
            if(decay_wa):
                swwa = lrgpu/20.0
            else:
                swwa = 1e-5
            utils.wwhite(model, swwa)
        if(dfw_enabled):
            utils.wfunny(model, wemas)  
        
        model.resetsink()

    if iter_num > max_iters:
        return 0, False, best_val_loss
    return iter_num + 1, True, best_val_loss
run = True
tl = time.time()
while run:
    if iter_num % eval_interval == 3:
        tl= time.time()
    iter_num, run, best_val_loss = model_step(iter_num, tl, best_val_loss)
torch.save(checkpoint, os.path.join(out_dir, 'fckpt.pt'))

if ddp:
    destroy_process_group()
