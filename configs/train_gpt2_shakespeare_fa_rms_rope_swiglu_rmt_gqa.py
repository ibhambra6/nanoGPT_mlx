# Train a miniature Shakespeare model using the full-featured RMT variant with Grouped Query Attention (GQA)
# Features: Flash Attention, RMSNorm, RoPE, SwiGLU activation, RMT architecture, and Grouped Query Attention

# Output and data
out_dir = 'trained_models'
dataset = 'shakespeare'

# Batching
gradient_accumulation_steps = 16
batch_size = 4
context_size = 256  # context length

# Optimizer + schedule
warmup_pct = 0.4
learning_rate = 2e-3  # with baby networks can afford to go a bit higher
num_iters = 1000
warmup_iters = 100
lr_decay_iters = 500
min_lr = 1e-4  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

# Model size (token embedding dims used to form RMT input)
n_layer = 6
n_head = 8      # Number of query heads
n_kv_head = 4   # Number of key/value heads (GQA: n_head > n_kv_head)
n_embd = 384
dropout = 0.2
bias = False  # False is better with RMSNorm

# RMT-specific parameters
use_rmt = True
Dk = 256  # Matrix dimension for RMT residual stream
Dv = 64   # Head dimension in RMT space
rmt_dropout = 0.1  # Dropout specifically for RMT components

# RoPE parameters (applies in Dv space for RMT attention)
rope_base = 10000.0  # Base frequency for RoPE
rope_scale = 1.0     # Scale factor for RoPE

# Eval/logging
save_interval = 1000
eval_interval = 250  # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10  # don't print too often

# GQA-specific notes:
# - n_head=8, n_kv_head=4 means each K/V head is shared by 2 Q heads
# - This reduces memory usage for K/V caches while maintaining model capacity
# - For more aggressive memory savings, try n_kv_head=2 or n_kv_head=1 (MQA)
