# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-ts-char'
eval_interval = 500 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'ts-char'
wandb_run_name = 'llama2-mod'

dataset = 'tinystories'
gradient_accumulation_steps = 16
batch_size = 16
block_size = 512 # context of up to 256 previous characters

# chincilla optimal-ish llama2
n_layer = 8
n_head = 16
n_kv_head = 8
n_embd = 1024
hidden_dim = n_embd * 8
dropout = 0.1

learning_rate = 1e-4 # with baby networks can afford to go a bit higher
max_iters = 50000
lr_decay_iters = 50000 # make equal to max_iters usually
min_lr = 1e-5 # learning_rate / 10 usually
beta2 = 0.95 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
