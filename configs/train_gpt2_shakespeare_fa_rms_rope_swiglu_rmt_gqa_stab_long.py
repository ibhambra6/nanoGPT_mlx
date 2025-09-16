# Train a miniature Shakespeare model using the long-context RMT variant with GQA + Stabilization
# Features: Flash Attention, RMSNorm, RoPE, SwiGLU activation, RMT architecture, GQA, Stabilization, and Long-Context

# Output and data
out_dir = 'trained_models'
dataset = 'shakespeare'

# Batching
gradient_accumulation_steps = 16
batch_size = 4
context_size = 512  # Increased context length for long-context testing
max_context_size = 2048  # Maximum context during inference

# Optimizer + schedule
warmup_pct = 0.4
learning_rate = 1.5e-3  # Slightly lower for stability with long context
num_iters = 1200
warmup_iters = 120
lr_decay_iters = 600
min_lr = 1e-4
beta2 = 0.99

# Model size
n_layer = 6
n_head = 8      # Number of query heads
n_kv_head = 2   # More aggressive GQA for memory efficiency (4:1 ratio)
n_embd = 384
dropout = 0.2
bias = False  # Bias-free for better optimization

# RMT-specific parameters
use_rmt = True
Dk = 256
Dv = 64
rmt_dropout = 0.1

# RoPE parameters with long-context scaling
rope_base = 10000.0
rope_scale = 2.0  # Scale factor for extrapolation
rope_scale_mode = 'ntk'  # NTK-aware scaling for better extrapolation
rope_partial_factor = 0.75  # Apply RoPE to 75% of dimensions to reduce over-rotation

# Stabilization features
parallel_residual = True
residual_alpha_learnable = True
residual_alpha_init = None  # Auto-set based on depth
qk_norm = True
qk_norm_scale_learnable = True
attention_softcap = 30.0  # Higher softcap for longer sequences
talking_heads = True
weight_tying = True
logit_scale = 1.0
z_loss_weight = 0.0005  # Small z-loss

# Long-context features
attention_sink_tokens = 4  # Keep first 4 tokens always visible
local_window_size = 256  # Sliding window size for lower layers
local_window_layers = [0, 1, 2]  # Apply sliding window to bottom 3 layers
rmt_refresh_interval = 3  # Refresh RMT state every 3 blocks
rmt_compress_factor = 0.8  # Light compression for memory efficiency
kv_cache_dtype = 'float16'  # Use float16 for KV cache
use_paged_kv = False  # Disabled for simplicity

# Eval/logging
save_interval = 1200
eval_interval = 300
eval_iters = 200
log_interval = 10

# Long-context model description:
# This variant is optimized for handling longer sequences efficiently:
# - NTK RoPE scaling: Better extrapolation to longer contexts
# - Partial RoPE: Reduces over-rotation artifacts in small models
# - Attention sinks: Stabilizes very long context decoding
# - Local windows: Lower layers use sliding windows, top layers keep full attention
# - Aggressive GQA: 4:1 ratio for maximum memory efficiency
# - RMT refresh: Periodic re-anchoring for stability
# - RMT compression: Bounds memory growth during long sequences
# - KV efficiency: float16 cache for memory optimization