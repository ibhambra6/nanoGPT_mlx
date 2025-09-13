# Train a miniature Shakespeare model using the full-featured RMT variant with GQA + Stabilization
# Features: Flash Attention, RMSNorm, RoPE, SwiGLU activation, RMT architecture, GQA, and Stabilization

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
bias = False  # False for bias-free linears (better with RMSNorm)

# RMT-specific parameters
use_rmt = True
Dk = 256  # Matrix dimension for RMT residual stream
Dv = 64   # Head dimension in RMT space
rmt_dropout = 0.1  # Dropout specifically for RMT components

# RoPE parameters (applies in Dv space for RMT attention)
rope_base = 10000.0  # Base frequency for RoPE
rope_scale = 1.0     # Scale factor for RoPE

# Stabilization features
parallel_residual = True  # Run attention and MLP in parallel for better gradient flow
residual_alpha_learnable = True  # Learnable residual scaling
residual_alpha_init = None  # Will be set to 1/sqrt(2*n_layer) = ~0.204
qk_norm = True  # QK normalization to stabilize attention
qk_norm_scale_learnable = True  # Learnable scale for QK norm
attention_softcap = 20.0  # Softcap for attention logits to improve stability
talking_heads = True  # Talking heads mechanism for head interaction
weight_tying = True  # Tie input/output embeddings
logit_scale = 1.0  # Final logit scaling factor
z_loss_weight = 0.001  # Z-loss weight for reducing overconfidence (small value)

# Eval/logging
save_interval = 1000
eval_interval = 250  # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10  # don't print too often

# Model description:
# This is the most advanced variant with all stabilization features:
# - Parallel residual: Better optimization depth-wise
# - Residual alpha: Depth-based scaling for small-data stability
# - QK normalization: Prevents extreme attention logits
# - Softcap: Improves stability at longer sequences
# - Talking heads: Encourages head specialization and interaction
# - Weight tying: Reduces parameters and improves regularization
# - Z-loss: Reduces overconfidence for better calibration
# - Bias-free: Cleaner optimization with RMSNorm
