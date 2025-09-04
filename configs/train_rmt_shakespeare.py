# Train a miniature Shakespeare model using the RMT variant

# Output and data
out_dir = 'gpt2_shakespeare_rmt'
dataset = 'shakespeare'

# Batching
gradient_accumulation_steps = 16
batch_size = 4
context_size = 256  # context length

# Optimizer + schedule
warmup_pct = 0.4
learning_rate = 2e-3
num_iters = 500
warmup_iters = 100
lr_decay_iters = 500
min_lr = 1e-4  # learning_rate / 10 usually
beta2 = 0.99

# Model size (token embedding dims still used to form RMT input)
n_layer = 6
n_head = 8  # RMT heads
n_embd = 384
dropout = 0.2
bias = False

# RMT-specific knobs
use_rmt = True
Dk = 256
Dv = 64
rmt_dropout = 0.0

# RoPE (applies in Dv space)
rope_base = 10000.0
rope_scale = 1.0

# Eval/logging
save_interval = 1000
eval_interval = 250
eval_iters = 200
log_interval = 10

