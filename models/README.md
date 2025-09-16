# Models Directory

This directory contains the core model implementations for nanoGPT_mlx, which provides MLX-based implementations of GPT-style transformer models.

## Available Models

### 1. `base_model.py` - Standard GPT Implementation

The standard implementation of a GPT (Generative Pre-trained Transformer) model using traditional attention computation.

#### Components:

- **`LayerNorm`**: Custom layer normalization implementation
  - Supports optional bias parameter
  - Implements standard layer normalization with learnable scale and bias parameters
  - Uses MLX operations for efficient computation

- **`CausalSelfAttention`**: Standard multi-head causal self-attention
  - **Attention Mechanism**: Manual implementation using matrix multiplications
  - Computes attention as: `softmax(QK^T / √d_k)V`
  - Includes causal masking to prevent attention to future tokens
  - Supports attention and residual dropout
  - Compatible with KV caching for efficient generation

- **`MLP`**: Feed-forward network with GELU activation
  - Two linear transformations with GELU activation in between
  - Expansion factor of 4x (hidden dim = 4 * embedding dim)
  - Includes dropout for regularization

- **`Block`**: Single transformer block
  - Pre-layer normalization architecture
  - Residual connections around attention and MLP
  - Structure: `x + attention(layernorm(x))` and `x + mlp(layernorm(x))`

- **`GPTConfig`**: Configuration dataclass for model hyperparameters
  - `block_size`: Maximum sequence length (default: 1024)
  - `vocab_size`: Vocabulary size (default: 50304, padded for efficiency)
  - `n_layer`: Number of transformer blocks (default: 12)
  - `n_head`: Number of attention heads (default: 12)
  - `n_embd`: Embedding dimensionality (default: 768)
  - `dropout`: Dropout probability (default: 0.0)
  - `bias`: Whether to use bias in linear layers and layer norms (default: True)

- **`GPT`**: Main model class
  - Token and position embeddings
  - Stack of transformer blocks
  - Final layer normalization and output projection
  - Includes `generate()` method for autoregressive text generation
  - Supports top-k sampling during generation

#### Key Features:
- **Manual Attention**: Uses explicit matrix operations for attention computation
- **KV Caching**: Supports key-value caching for efficient generation
- **Causal Masking**: Proper causal attention for autoregressive generation
- **Generation**: Built-in text generation with temperature and top-k sampling

---

### 2. `base_model_fa.py` - Flash Attention Implementation

An optimized version of the GPT model that uses MLX's flash attention implementation for improved memory efficiency and performance.

#### Components:

All components are identical to `base_model.py` except for the attention mechanism:

- **`CausalSelfAttention`**: Flash attention-enabled multi-head causal self-attention
  - **Flash Attention**: Uses `mlx.core.fast.scaled_dot_product_attention`
  - More memory-efficient than standard attention, especially for long sequences
  - Automatically handles scaling (1/√d_k) and softmax computation
  - Maintains the same interface as the standard implementation
  - Supports both regular forward passes and KV caching

#### Key Differences from `base_model.py`:

1. **Import**: Adds `import mlx.core.fast as fast`
2. **Attention Computation**: Replaces manual attention with:
   ```python
   y = fast.scaled_dot_product_attention(
       query, key, value,
       scale=1.0 / (key.shape[-1] ** 0.5),
       mask=causal_mask
   )
   ```
3. **Mask Format**: Converts causal mask to additive format (-inf for masked positions)
4. **Memory Efficiency**: Significantly reduced memory usage for long sequences

#### Flash Attention Benefits:
- **Memory Efficiency**: O(N) memory complexity instead of O(N²) for attention computation
- **Speed**: Optimized kernel implementation for faster computation
- **Scalability**: Better performance with longer sequences
- **Drop-in Replacement**: Same API as standard attention

---

### 3. `base_model_fa_rms_rope.py` - Flash Attention + RMSNorm + RoPE Implementation

The most advanced implementation that combines Flash Attention with RMSNorm and Rotary Positional Encoding (RoPE) for enhanced performance and better positional understanding.

#### Components:

All components share similarities with the flash attention model, but with key architectural improvements:

- **`CausalSelfAttention`**: Flash attention with RoPE integration
  - **Flash Attention**: Uses `mlx.core.fast.scaled_dot_product_attention` for memory efficiency
  - **RoPE Integration**: Incorporates Rotary Positional Encoding directly into attention computation
  - Applies rotary embeddings to query and key vectors before attention calculation
  - Supports configurable RoPE parameters (`rope_base`, `rope_scale`)
  - Maintains compatibility with KV caching for efficient generation

- **`Block`**: Transformer block with RMSNorm
  - **RMSNorm**: Uses `nn.RMSNorm` instead of LayerNorm for improved training stability
  - RMS (Root Mean Square) normalization provides better gradient flow
  - More computationally efficient than LayerNorm (no mean subtraction)
  - Structure remains: `x + attention(rmsnorm(x))` and `x + mlp(rmsnorm(x))`

- **`GPTConfig`**: Extended configuration with RoPE parameters
  - All standard GPT parameters (block_size, vocab_size, n_layer, etc.)
  - `rope_base`: Base frequency for RoPE computation (default: 10000.0)
  - `rope_scale`: Scale factor for RoPE frequencies (default: 1.0)
  - `bias`: Recommended to set to `False` for optimal performance with RMSNorm

- **`GPT`**: Main model class with positional encoding changes
  - **No Positional Embeddings**: Eliminates learned positional embeddings (`wpe`)
  - **RoPE Handling**: Positional information encoded directly in attention mechanism
  - Uses RMSNorm for final layer normalization before output projection
  - Maintains identical generation interface with improved sequence extrapolation

#### Key Features:
- **RoPE (Rotary Positional Encoding)**: 
  - Better extrapolation to longer sequences than learned embeddings
  - Relative positional encoding that preserves rotational properties
  - No sequence length limitations imposed by positional embeddings
- **RMSNorm**: 
  - Improved training stability compared to LayerNorm
  - Faster computation (no mean calculation)
  - Better gradient flow properties
- **Flash Attention**: All memory and computational benefits from the FA model
- **Enhanced Extrapolation**: Superior performance on sequences longer than training length

#### RoPE Benefits:
- **Length Extrapolation**: Can handle sequences longer than training data
- **Relative Positions**: Encodes relative rather than absolute positions
- **Rotational Invariance**: Maintains consistent relative position encoding
- **No Length Limits**: No hard sequence length constraints from embeddings

#### RMSNorm Benefits:
- **Training Stability**: More stable gradients during training
- **Computational Efficiency**: ~15% faster than LayerNorm
- **Memory Efficiency**: Reduced memory footprint
- **Modern Architecture**: Used in state-of-the-art models like LLaMA

---

## Model Architecture

All three models implement the GPT architecture with the following structure:

```
Input Tokens → Token Embedding + Position Embedding → Dropout
    ↓
Transformer Block 1:
    LayerNorm → CausalSelfAttention → Residual Connection
    LayerNorm → MLP → Residual Connection
    ↓
Transformer Block 2:
    ...
    ↓
Transformer Block N:
    ...
    ↓
Final LayerNorm → Output Projection → Logits
```

## Usage Examples

### Standard Model (`base_model.py`):
```python
from models.base_model import GPT, GPTConfig

# Create configuration
config = GPTConfig(
    block_size=1024,
    vocab_size=50304,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.1,
    bias=True
)

# Initialize model
model = GPT(config)

# Forward pass
logits = model(input_tokens)  # Shape: (batch_size, seq_len, vocab_size)

# Generate text
generated = model.generate(
    input_tokens, 
    max_new_tokens=100, 
    temperature=1.0, 
    top_k=50
)
```

### 4. `base_model_fa_rms_rope_swiglu_rmt.py` - RMT (Residual Matrix Transformer) + Flash Attention + RMSNorm + RoPE + SwiGLU

Implements a Residual Matrix Transformer (RMT) where the residual state per token is a matrix `X ∈ ℝ^{B×T×Dk×Dv}` instead of a vector. Heads read vectors from `X` with read keys (contract over `Dk`), attend with standard SDPA in `Dv` space, then write back via write keys (contract heads into `Dk`). Uses RoPE over the `Dv` dimension and a shared SwiGLU MLP for updates. Designed for long-context stability and controllable capacity via `Dk`.

Key components:
- Residual matrix state: `X ∈ (B,T,Dk,Dv)` built from token embeddings via an input adapter
- RMTSelfAttention: `Linear(Dk→H)` readers produce per-head vectors in `Dv` → Flash Attention → `Linear(H→Dk)` write-back
- RMTMLP: shared SwiGLU MLP in `Dv` with `Linear(Dk→H)` read and `Linear(H→Dk)` write
- MatrixRMSNorm: RMSNorm along `Dv` per `(B,T,Dk,·)` slice
- Output head: retrieves per-head representations and projects to logits

Config additions (in `GPTConfig`):
- `use_rmt: bool` – enable RMT path (defaults True in this file)
- `Dk: int` – residual key dimension (matrix rows)
- `Dv: int` – residual value dimension (per-head vector length)
- `rmt_dropout: float` – dropout used in RMT writes/MLP
- Reuses `rope_base`, `rope_scale` for RoPE over `Dv`

Usage example:
```python
from models.base_model_fa_rms_rope_swiglu_rmt import GPT, GPTConfig

config = GPTConfig(
    vocab_size=50304,
    block_size=512,
    n_layer=8,
    n_head=8,
    n_embd=768,
    dropout=0.0,
    bias=False,
    # RMT knobs
    use_rmt=True,
    Dk=256,
    Dv=64,
    rmt_dropout=0.0,
)

model = GPT(config)
logits = model(input_tokens)         # (B,T,V)
generated = model.generate(
    input_tokens, max_new_tokens=128, temperature=0.8, top_k=50
)
```

Factory loading:
- The project’s model factory auto-selects this variant when `use_rmt=True` is present in the saved config (see `models/__init__.py`). Scripts like `sample.py` and `scripts/eval_*` that use the factory will load it automatically.

When to use:
- Need improved long-context handling and a controllable capacity knob (`Dk`)
- Prefer matrix residual updates and SwiGLU MLPs
- Training new models with modern stack: Flash Attention + RoPE + RMSNorm + SwiGLU

### 5. `base_model_fa_rms_rope_swiglu_rmt_gqa.py` - RMT + Flash Attention + RMSNorm + RoPE + SwiGLU + GQA

Extends the RMT variant with Grouped Query Attention (GQA) in both the vector and matrix attention paths. Queries use `n_head` while keys/values use `n_kv_head` (supports GQA and MQA when `n_kv_head=1`). Uses RoPE over head dimensions, Flash Attention kernels, and maintains KV caches with reduced memory via grouped KV heads.

Key components and differences:
- Grouped Query Attention: separate `wq`, `wk`, `wv` where `n_head` can be a multiple of `n_kv_head`.
- GQA in RMT: `read_q` heads can exceed `read_k`/`read_v` heads; MLX SDPA handles grouping.
- RoPE over Q/K with cache offset support for autoregressive generation.
- Additive causal masks compatible with MLX `fast.scaled_dot_product_attention`.
- Config adds `n_kv_head` and validates `n_head % n_kv_head == 0`.

When to use:
- You want RMT benefits plus lower KV cache memory and faster inference through GQA/MQA.
- Ablate quality vs. memory by sweeping `n_kv_head ∈ {n_head, n_head/2, 1}`.

### 6. `base_model_fa_rms_rope_swiglu_rmt_gqa_stab.py` - GQA Variant with Stabilization Features

Builds on the GQA model with small‑data‑friendly stabilization and expressivity improvements while keeping parameter count nearly unchanged.

Added features:
- Parallel residual: attention and MLP computed in parallel from a shared RMSNorm and summed into the residual.
- Residual alpha: learnable or fixed scaling per branch; default init `~1/√(2L)` from depth.
- QK normalization: per‑head RMSNorm for Q/K with optional learnable scale to tame attention logits.
- Attention softcap: optional tanh softcap on attention logits to avoid extreme values at long context.
- Talking‑heads: lightweight linear mixing across heads pre/post attention to encourage head interaction.
- Bias‑free linears: `bias=False` by default for cleaner optimization with RMSNorm.
- Weight tying and logit scale: tie input/output embeddings and optionally scale final logits; optional z‑loss helper.

When to use:
- Training on small corpora (e.g., Shakespeare) where stability and calibration matter more than added params.
- Ablate each toggle (qk_norm, softcap, talking_heads, parallel_residual) to measure impact.

### 7. `base_model_fa_rms_rope_swiglu_rmt_gqa_stab_long.py` - Stabilized + Long‑Context Features

Extends the stabilized GQA model with long‑context readiness and memory‑efficiency options.

Added features (in addition to Model 6):
- LongContextRoPE: NTK/YaRN/linear RoPE scaling modes plus partial‑RoPE (apply RoPE to a fraction of head dims).
- Attention sinks: first N sink tokens always visible to stabilize decoding over long sequences.
- Local window attention: optional sliding‑window attention in selected layers; other layers keep full attention.
- KV efficiency: KV cache dtype control (`float32`/`float16`/`int8`) and optional paged‑KV flag; cache offset aware.
- RMT refresh + compression: periodic refresh via lightweight low‑rank or pass‑through compressor to bound state growth.

Config highlights:
- `rope_scale_mode={'linear','ntk','yarn'}`, `rope_partial_factor∈(0,1]`
- `attention_sink_tokens`, `local_window_size`, `local_window_layers`
- `kv_cache_dtype`, `use_paged_kv`
- `rmt_refresh_interval`, `rmt_compress_factor`

When to use:
- Evaluating throughput/memory vs. quality for longer contexts on Apple Silicon.
- Side‑by‑side with Model 5/6 to quantify benefits of long‑context scaling without changing core params.

### 8. `base_model_fa_rms_rope_swiglu_rmt_gqa_stab_long_moe.py` - Stabilized + Long‑Context + MoE‑lite

Extends the stabilized long‑context model with a tiny Mixture‑of‑Experts (MoE) FFN in selected layers. Designed for small‑data settings: few experts, top‑2 routing, a shared expert for stability, and strong expert dropout. Keeps core attention stack unchanged (FA + RoPE + RMSNorm + GQA) and retains long‑context features (sinks, local windows, KV controls, RMT refresh/compression).

Added features (in addition to Model 7):
- MoE FFN (top‑2): Replace standard SwiGLU FFN with a routed expert MLP in a subset of layers.
- Few experts + shared expert: Small `num_experts` (e.g., 4–8) and an always‑available shared expert to stabilize routing.
- Capacity factor: Token‑to‑expert capacity bound with overflow handling to keep compute predictable.
- Router regularization: Auxiliary load‑balancing loss, optional router noise, and optional router z‑loss for calibration.
- Sparse placement: Apply MoE only to late or every‑k layers to avoid overfitting on tiny corpora.

Config highlights:
- `ffn_type={'swiglu','moe_top2'}` – choose FFN per model or per layer.
- `moe_layers`: list of layer indices (or stride rule) to enable MoE.
- `num_experts`, `top_k=2`, `capacity_factor`, `expert_dropout`, `router_noise`.
- `aux_loss_weight` (load‑balancing), `router_z_loss_weight`, `shared_expert=True/False`.
- Works alongside: `rope_scale_mode`, `rope_partial_factor`, `attention_sink_tokens`, `local_window_size`, `kv_cache_dtype`, `rmt_refresh_interval`, `rmt_compress_factor`.

When to use:
- You want to probe expert specialization benefits at tiny scale without large parameter growth.
- You need a controlled MoE baseline with strong regularization on small datasets (e.g., Shakespeare).
- You want to compare dense vs. sparse FFNs while holding attention and long‑context features constant.
- You plan to scale to more data later but want MoE interfaces and metrics in place now.

### Flash Attention Model (`base_model_fa.py`):
```python
from models.base_model_fa import GPT, GPTConfig

# Usage is identical to base_model.py
config = GPTConfig(...)
model = GPT(config)
logits = model(input_tokens)
generated = model.generate(input_tokens, max_new_tokens=100)
```

### Flash Attention + RMSNorm + RoPE Model (`base_model_fa_rms_rope.py`):
```python
from models.base_model_fa_rms_rope import GPT, GPTConfig

# Create configuration with RoPE parameters
config = GPTConfig(
    block_size=1024,
    vocab_size=50304,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.1,
    bias=False,  # Recommended for RMSNorm
    rope_base=10000.0,  # RoPE base frequency
    rope_scale=1.0      # RoPE scale factor
)

# Initialize model (no positional embeddings needed)
model = GPT(config)

# Forward pass and generation work identically
logits = model(input_tokens)
generated = model.generate(
    input_tokens, 
    max_new_tokens=100, 
    temperature=1.0, 
    top_k=50
)
```

## When to Use Which Model

### Use `base_model.py` when:
- You need a reference implementation
- Debugging attention computations
- Working with shorter sequences (< 512 tokens)
- Maximum compatibility and stability

### Use `base_model_fa.py` when:
- Training or inference with long sequences (> 512 tokens)
- Memory is a constraint
- You want optimal performance
- Using modern hardware that benefits from optimized kernels

### Use `base_model_fa_rms_rope.py` when:
- You need the best performance and modern architecture
- Working with very long sequences or need length extrapolation
- Training new models from scratch (recommended for new projects)
- You want maximum training stability and efficiency
- Need relative positional encoding benefits
- Working with sequences that may exceed training length

### Use `base_model_fa_rms_rope_swiglu_rmt.py` when:
- You want Residual Matrix Transformer behavior (matrix residuals per token)
- You want to tune capacity primarily via `Dk` and keep `Dv` modest (e.g., 64)
- You prefer SwiGLU MLPs and RoPE over `Dv` with Flash Attention
- You aim for robust long-context training with a modern stack

### Use `base_model_fa_rms_rope_swiglu_rmt_gqa.py` when:
- You want RMT with GQA/MQA to reduce KV cache memory and speed up inference
- You plan to sweep `n_kv_head` to study memory/quality tradeoffs
- You want a modern baseline (FA + RMSNorm + RoPE + SwiGLU) with grouped KV heads
- You need a strong, efficient default for small-data long-context experiments

### Use `base_model_fa_rms_rope_swiglu_rmt_gqa_stab.py` when:
- You train on tiny corpora (e.g., Shakespeare) and want extra stability without adding params
- You want parallel residuals, QK normalization, and attention softcap/talking-heads toggles
- You prefer bias-free linears and embedding–logit weight tying with optional logit scaling
- You want a calibrated, robust everyday baseline for ablations

### Use `base_model_fa_rms_rope_swiglu_rmt_gqa_stab_long.py` when:
- You evaluate longer contexts and need NTK/YaRN/partial‑RoPE scaling options
- You want attention sinks and optional local window layers for throughput
- You need KV cache dtype control (`float32`/`float16`/`int8`) or paged‑KV
- You want periodic RMT refresh/compression to bound memory growth

### Use `base_model_fa_rms_rope_swiglu_rmt_gqa_stab_long_moe.py` when:
- You want a tiny, regularized MoE FFN to test expert routing benefits
- You prefer sparse FFNs in late or every‑k layers while keeping attention identical
- You need auxiliary load‑balancing and router calibration losses for stability on small data
- You want MoE hooks ready for future scaling without changing the attention stack

## Configuration Notes

- **Vocabulary Size**: Default 50304 is GPT-2's vocabulary (50257) padded to the nearest multiple of 64 for computational efficiency
- **Embedding Dimension**: Must be divisible by the number of heads
- **Block Size**: Maximum sequence length the model can handle
- **Bias**: Setting to `False` can improve performance slightly at the cost of some compatibility with GPT-2 checkpoints. Recommended for RMSNorm models.
- **RoPE Parameters** (for `base_model_fa_rms_rope.py`):
  - `rope_base`: Base frequency for RoPE computation (default: 10000.0) - higher values = slower frequency decay
  - `rope_scale`: Scale factor for RoPE frequencies (default: 1.0) - can be used for fine-tuning extrapolation behavior

- **RMT Parameters** (for `base_model_fa_rms_rope_swiglu_rmt.py`):
  - `use_rmt`: Turns on Residual Matrix Transformer path
  - `Dk`: Matrix rows (capacity knob). Start around 128–512
  - `Dv`: Per-head vector size. Start around 64
  - `n_head`: Number of heads that read/write `Dv` vectors
  - `rmt_dropout`: Dropout applied in RMT attention writes and MLP

## Dependencies

- **MLX**: Core framework for Apple Silicon optimization
- **MLX Neural Networks**: High-level neural network components (includes RMSNorm and RoPE)
- **MLX Fast**: Optimized kernels (for flash attention models: `base_model_fa.py` and `base_model_fa_rms_rope.py`)

All three models are designed to be drop-in replacements for each other, with the flash attention versions providing performance benefits for longer sequences, and the RMSNorm + RoPE version offering the most modern architecture with enhanced training stability and length extrapolation capabilities.

The RMT variant extends this with a matrix residual stream, allowing capacity scaling through `Dk` while keeping `Dv` compact, and integrates neatly with Flash Attention, RoPE, RMSNorm, and SwiGLU for strong performance and stability on Apple Silicon via MLX.
