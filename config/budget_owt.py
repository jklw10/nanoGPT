
out_dir = 'out-shakespeare-char'
eval_interval = 1000 # keep frequent because we'll overfit
eval_iters = 100
log_interval = 1000 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'budget-owt'
wandb_run_name = 'mini-gpt'

dataset = 'openwebtext'

gradient_accumulation_steps = 1
batch_size = 64    

block_size = 512  

n_layer = 4
n_head = 6 
n_embd = 228  
dropout = 0.0 

learning_rate = 1e-3
max_iters = 100000
lr_decay_iters = 30000 
min_lr = 1e-4 
warmup_iters = 100

beta2 = 0.99 
grad_clip=0.0    
#warmup_iters = 100 

