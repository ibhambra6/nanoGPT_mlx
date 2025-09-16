# Train a miniature Shakespeare model using the ultimate MoE variant with all features
# Features: Flash Attention, RMSNorm, RoPE, SwiGLU, RMT, GQA, Stabilization, Long-Context, and MoE

# Output and data
out_dir = 'trained_models'
dataset = 'shakespeare'

# Batching
gradient_accumulation_steps = 16
batch_size = 4
context_size = 512  # Increased context length
max_context_size = 2048

# Optimizer + schedule
warmup_pct = 0.4
learning_rate = 1.2e-3  # Slightly lower for MoE stability
num_iters = 1500  # More iterations for MoE convergence
warmup_iters = 150
lr_decay_iters = 750
min_lr = 1e-4
beta2 = 0.99

# Model size
n_layer = 8  # Deeper for MoE benefits
n_head = 8
n_kv_head = 2  # Aggressive GQA (4:1 ratio)
n_embd = 384
dropout = 0.15  # Slightly higher dropout for MoE
bias = False

# RMT-specific parameters
use_rmt = True
Dk = 256
Dv = 64
rmt_dropout = 0.1

# RoPE parameters with long-context scaling
rope_base = 10000.0
rope_scale = 2.0
rope_scale_mode = 'ntk'
rope_partial_factor = 0.75

# Stabilization features
parallel_residual = True
residual_alpha_learnable = True
residual_alpha_init = None
qk_norm = True
qk_norm_scale_learnable = True
attention_softcap = 30.0
talking_heads = True
weight_tying = True
logit_scale = 1.0
z_loss_weight = 0.0005

# Long-context features
attention_sink_tokens = 4
local_window_size = 256
local_window_layers = [0, 1, 2]  # Bottom layers use sliding window
rmt_refresh_interval = 4  # More frequent refresh for MoE
rmt_compress_factor = 0.8
kv_cache_dtype = 'float16'
use_paged_kv = False

# MoE features
moe_layers = [4, 6]  # Late-layer MoE placement (layers 4 and 6 out of 8)
moe_n_experts = 4  # Small number of experts
moe_expert_dropout = 0.15  # Strong expert dropout
moe_load_balance_weight = 0.02  # Slightly higher load balance weight
per_head_scaling = True  # Learnable per-head scaling
drop_path_rate = 0.05  # Tiny stochastic depth for gentle regularization

# Advanced sampling
enable_advanced_sampling = True
speculative_decode = False  # Disabled by default

# Eval/logging
save_interval = 1500
eval_interval = 375
eval_iters = 200
log_interval = 10

# MoE model description:
# This is the ultimate variant combining all advanced features:
# - Tiny MoE: Only 4 experts with top-2 routing in late layers
# - Sparse placement: MoE only in layers 4 and 6 to minimize overfitting
# - Strong regularization: Expert dropout + load balancing + DropPath
# - Shared expert: Always-active expert for stability
# - Per-head scaling: Helps with expert specialization
# - Advanced sampling: Multiple strategies for better generation quality
# - Speculative decode interface: Ready for acceleration (optional)
# 
# Why for Shakespeare:
# - Tests whether minimal MoE improves rare token/style modeling
# - Late-layer placement focuses on high-level patterns
# - Strong regularization prevents overfitting on small data
# - Advanced sampling improves perceived quality at inference