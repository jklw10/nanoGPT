
out_dir = 'out-b-owt'
eval_interval = 1000 
eval_iters = 50
log_interval = 1000 

always_save_checkpoint = False

wandb_log = False
wandb_project = 'budget-owt'
wandb_run_name = 'mini-gpt'

dataset = 'openwebtext'

gradient_accumulation_steps = 1
batch_size = 64    

block_size = 64  

n_layer = 4
n_head = 6 
n_embd = 228  
dropout = 0.0 

weight_decay = 0.1

learning_rate = 1e-3
max_iters = 50000
lr_decay_iters = 30000 
min_lr = 1e-4 
warmup_iters = 100

beta2 = 0.999
grad_clip=0.0    

