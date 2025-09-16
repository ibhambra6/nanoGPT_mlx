import mlx.core as mx
import mlx.nn as nn
import mlx.core.fast as fast
import math
import random

from dataclasses import dataclass
from .components import topk as components_topk


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample for regularization."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def __call__(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + mx.random.uniform(shape, dtype=x.dtype)
        random_tensor = mx.floor(random_tensor)  # binarize
        output = x * random_tensor / keep_prob
        return output


class MoERouter(nn.Module):
    """Top-k router for MoE with load balancing."""
    def __init__(self, n_embd: int, n_experts: int, top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(n_embd, n_experts, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def __call__(self, x):
        # x: (batch, seq_len, n_embd)
        B, T, D = x.shape
        
        # Compute router logits
        router_logits = self.gate(x)  # (B, T, n_experts)
        
        # Apply dropout to router for expert dropout effect
        router_logits = self.dropout(router_logits)
        
        # Get top-k experts
        top_k_logits, top_k_indices = components_topk(router_logits, self.top_k)
        
        # Compute routing weights (softmax over top-k)
        top_k_weights = mx.softmax(top_k_logits, axis=-1)
        
        # For load balancing loss computation
        router_probs = mx.softmax(router_logits, axis=-1)
        
        return top_k_weights, top_k_indices, router_probs


class ExpertMLP(nn.Module):
    """Single expert MLP with SwiGLU activation."""
    def __init__(self, n_embd: int, hidden_factor: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(n_embd * hidden_factor)
        
        self.gate = nn.Linear(n_embd, hidden_dim, bias=False)
        self.up = nn.Linear(n_embd, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def __call__(self, x):
        gate_out = self.gate(x)
        up_out = self.up(x)
        hidden = gate_out * mx.sigmoid(gate_out) * up_out  # SwiGLU
        return self.dropout(self.down(hidden))


class TinyMoE(nn.Module):
    """Tiny MoE FFN with shared expert and load balancing."""
    def __init__(self, config, n_experts: int = 4, top_k: int = 2, expert_dropout: float = 0.1):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.n_embd = config.n_embd
        
        # Router
        self.router = MoERouter(config.n_embd, n_experts, top_k, expert_dropout)
        
        # Experts
        self.experts = [ExpertMLP(config.n_embd, dropout=config.dropout) for _ in range(n_experts)]
        
        # Shared expert for stability
        self.shared_expert = ExpertMLP(config.n_embd, hidden_factor=2.0, dropout=config.dropout)
        
        # Load balancing loss weight
        self.load_balance_loss_weight = getattr(config, 'moe_load_balance_weight', 0.01)
        
    def compute_load_balance_loss(self, router_probs, top_k_indices):
        """Compute auxiliary load balancing loss."""
        if self.load_balance_loss_weight <= 0:
            return 0.0
            
        # Compute token assignment frequencies
        batch_size, seq_len, _ = router_probs.shape
        total_tokens = batch_size * seq_len
        
        # Average probability of each expert being selected
        mean_prob = mx.mean(router_probs, axis=(0, 1))  # (n_experts,)
        
        # Actual selection frequency
        selected = mx.zeros((self.n_experts,))
        for i in range(self.n_experts):
            mask = (top_k_indices == i).astype(mx.float32)
            selected = selected.at[i].set(mx.sum(mask) / (total_tokens * self.top_k))
        
        # Load balance loss: encourage uniform distribution
        load_loss = mx.sum(mean_prob * selected) * self.n_experts
        return self.load_balance_loss_weight * load_loss
        
    def __call__(self, x):
        # x: (batch, seq_len, n_embd)
        B, T, D = x.shape
        
        # Get routing decisions
        top_k_weights, top_k_indices, router_probs = self.router(x)
        
        # Compute load balance loss
        load_balance_loss = self.compute_load_balance_loss(router_probs, top_k_indices)
        
        # Route to experts
        expert_outputs = []
        for i in range(self.top_k):
            expert_idx = top_k_indices[..., i]  # (B, T)
            weights = top_k_weights[..., i:i+1]  # (B, T, 1)
            
            # Create one-hot encoding for expert selection
            expert_output = mx.zeros_like(x)
            for j in range(self.n_experts):
                mask = (expert_idx == j).astype(mx.float32)[..., None]  # (B, T, 1)
                expert_result = self.experts[j](x)
                expert_output = expert_output + mask * expert_result
            
            expert_outputs.append(weights * expert_output)
        
        # Combine expert outputs
        moe_output = mx.sum(mx.stack(expert_outputs, axis=0), axis=0)
        
        # Add shared expert (always active)
        shared_output = self.shared_expert(x)
        
        # Combine MoE and shared expert
        final_output = moe_output + 0.5 * shared_output  # Shared expert with reduced weight
        
        return final_output, load_balance_loss


class LongContextRoPE(nn.Module):
    """Enhanced RoPE with NTK/YaRN scaling and partial application."""
    
    def __init__(self, dims: int, base: float = 10000.0, scale: float = 1.0, 
                 scale_mode: str = 'linear', partial_factor: float = 1.0, 
                 max_seq_len: int = 2048):
        super().__init__()
        self.dims = dims
        self.base = base
        self.scale = scale
        self.scale_mode = scale_mode
        self.partial_factor = partial_factor
        self.max_seq_len = max_seq_len
        
        # Determine how many dimensions to apply RoPE to
        self.rope_dims = int(dims * partial_factor)
        self.rope_dims = self.rope_dims - (self.rope_dims % 2)  # Ensure even
        
        # Pre-compute frequency adjustment
        if scale_mode == 'ntk':
            # NTK-aware scaling
            alpha = scale
            self.adjusted_base = base * (alpha ** (dims / (dims - 2)))
        elif scale_mode == 'yarn':
            # YaRN scaling
            alpha = scale
            beta_fast = 32
            beta_slow = 1
            mscale = math.sqrt(1 + math.log(alpha) / math.log(base))
            self.adjusted_base = base * alpha
            self.mscale = mscale
        else:
            # Linear scaling
            self.adjusted_base = base
        
        # Create frequency tensor for RoPE dimensions only
        freqs = mx.arange(0, self.rope_dims, 2, dtype=mx.float32)
        freqs = 1.0 / (self.adjusted_base ** (freqs / self.rope_dims))
        self.freqs = freqs
    
    def __call__(self, x, offset: int = 0):
        *leading, seq_len, D = x.shape
        
        if self.rope_dims == 0:
            return x
        
        # Extract the portion to apply RoPE to
        x_rope = x[..., :self.rope_dims]
        x_pass = x[..., self.rope_dims:] if self.rope_dims < D else None
        
        # Create position tensor
        positions = mx.arange(offset, offset + seq_len, dtype=mx.float32)
        
        # Scale positions if needed
        if self.scale_mode == 'linear' and self.scale != 1.0:
            positions = positions / self.scale
        elif self.scale_mode == 'yarn' and hasattr(self, 'mscale'):
            positions = positions * self.mscale
        
        # Compute angles
        angles = positions[:, None] * self.freqs[None, :]  # (seq_len, rope_dims//2)
        
        # Create cos and sin
        cos_vals = mx.cos(angles)
        sin_vals = mx.sin(angles)
        
        # Repeat for all leading dimensions
        for _ in leading:
            cos_vals = mx.expand_dims(cos_vals, 0)
            sin_vals = mx.expand_dims(sin_vals, 0)
        
        # Apply rotary embedding
        x_rope_even = x_rope[..., 0::2]  # Even indices
        x_rope_odd = x_rope[..., 1::2]   # Odd indices
        
        x_rope_rot_even = x_rope_even * cos_vals - x_rope_odd * sin_vals
        x_rope_rot_odd = x_rope_even * sin_vals + x_rope_odd * cos_vals
        
        # Interleave back
        x_rope_rot = mx.stack([x_rope_rot_even, x_rope_rot_odd], axis=-1)
        x_rope_rot = x_rope_rot.reshape(*leading, seq_len, self.rope_dims)
        
        # Concatenate with non-RoPE portion
        if x_pass is not None:
            return mx.concatenate([x_rope_rot, x_pass], axis=-1)
        else:
            return x_rope_rot


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        """
        Initializes the Causal Self-Attention layer with Grouped Query Attention (GQA) support.

        Args:
            config (GPTConfig): An instance of the configuration class
                specifying the hyperparameters for the Causal Self-Attention layer.

        Attributes:
            wq (nn.Linear): Linear layer for query projections.
            wk (nn.Linear): Linear layer for key projections.
            wv (nn.Linear): Linear layer for value projections.
            c_proj (nn.Linear): Linear layer for output projection.
            attn_dropout (nn.Dropout): Dropout layer for attention weights.
            resid_dropout (nn.Dropout): Dropout layer for residual connections.
            n_head (int): Number of query heads.
            n_kv_head (int): Number of key/value heads.
            n_embd (int): Dimensionality of the embedding.
            dropout (float): Dropout probability.
            head_dim (int): Dimension per head.

        Notes:
            - Supports Grouped Query Attention where n_head >= n_kv_head
            - When n_head == n_kv_head, this is standard Multi-Head Attention
            - When n_kv_head == 1, this is Multi-Query Attention
            - MLX's scaled_dot_product_attention handles the grouping internally
        """
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim = config.n_embd // config.n_head
        
        # Separate projections for GQA support
        # Query uses n_head heads, Key/Value use n_kv_head heads
        self.wq = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=config.bias)
        self.wk = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=config.bias)
        self.wv = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=config.bias)

        # output projection
        self.c_proj = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Enhanced RoPE with long-context features
        rope_base = getattr(config, 'rope_base', 10000.0)
        rope_scale = getattr(config, 'rope_scale', 1.0)
        rope_scale_mode = getattr(config, 'rope_scale_mode', 'linear')
        rope_partial_factor = getattr(config, 'rope_partial_factor', 1.0)
        
        self.rope = LongContextRoPE(
            dims=self.head_dim,
            base=rope_base,
            scale=rope_scale,
            scale_mode=rope_scale_mode,
            partial_factor=rope_partial_factor,
            max_seq_len=config.block_size
        )
        
        # QK Normalization for stability
        self.qk_norm = getattr(config, 'qk_norm', False)
        if self.qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
            if getattr(config, 'qk_norm_scale_learnable', False):
                self.qk_scale = mx.ones((1,))
            else:
                self.qk_scale = None
        
        # Talking heads mechanism
        self.talking_heads = getattr(config, 'talking_heads', False)
        if self.talking_heads:
            self.talking_heads_pre = nn.Linear(self.n_head, self.n_head, bias=False)
            self.talking_heads_post = nn.Linear(self.n_head, self.n_head, bias=False)
        
        # Attention softcap
        self.attention_softcap = getattr(config, 'attention_softcap', None)
        
        # Per-head scaling for MoE specialization
        self.per_head_scaling = getattr(config, 'per_head_scaling', True)
        if self.per_head_scaling:
            self.head_scales = mx.ones((self.n_head,))
        
        # Long-context features
        self.attention_sink_tokens = getattr(config, 'attention_sink_tokens', 4)
        self.local_window_size = getattr(config, 'local_window_size', None)
        self.kv_cache_dtype = getattr(config, 'kv_cache_dtype', 'float32')
        self.use_paged_kv = getattr(config, 'use_paged_kv', False)

    def __call__(self, x, mask, cache=None):
        B, T, C = x.shape # batch size, sequence length, embedding dimensionality (n_embd)

        # Project to query, key, value with GQA support
        query = self.wq(x)  # (B, T, n_head * head_dim)
        key = self.wk(x)    # (B, T, n_kv_head * head_dim)
        value = self.wv(x)  # (B, T, n_kv_head * head_dim)
        
        # Reshape and transpose for attention
        # Query: (B, n_head, T, head_dim)
        query = query.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        # Key/Value: (B, n_kv_head, T, head_dim)
        key = key.reshape(B, T, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
        value = value.reshape(B, T, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)

        # Apply RoPE to query and key
        # RoPE expects input in shape (batch, sequence, heads, head_dim), so we need to transpose
        query = query.transpose(0, 2, 1, 3)  # (B, T, n_head, head_dim)
        key = key.transpose(0, 2, 1, 3)      # (B, T, n_kv_head, head_dim)
        
        if cache is not None:
            # For cached generation, we need to handle the offset
            offset = cache[0].shape[2] if cache[0] is not None else 0
            query = self.rope(query, offset=offset)
            key = self.rope(key, offset=offset)
        else:
            query = self.rope(query)
            key = self.rope(key)
        
        # Transpose back to (B, n_head/n_kv_head, T, head_dim) for attention computation
        query = query.transpose(0, 2, 1, 3)  # (B, n_head, T, head_dim)
        key = key.transpose(0, 2, 1, 3)      # (B, n_kv_head, T, head_dim)
        
        # Apply QK normalization if enabled
        if self.qk_norm:
            # Normalize over head dimension
            query = self.q_norm(query)
            key = self.k_norm(key)
            if self.qk_scale is not None:
                query = query * self.qk_scale
                key = key * self.qk_scale

        # Apply KV cache efficiency features
        if cache is not None:
            key_cache, value_cache = cache
            
            # Convert cache to appropriate dtype
            if self.kv_cache_dtype == 'float16':
                key_cache = key_cache.astype(mx.float16)
                value_cache = value_cache.astype(mx.float16)
                key = key.astype(mx.float16)
                value = value.astype(mx.float16)
            elif self.kv_cache_dtype == 'int8':
                # Simple int8 quantization (for demonstration)
                # In practice, you'd want proper quantization/dequantization
                key_cache = (key_cache * 127).astype(mx.int8).astype(mx.float32) / 127
                value_cache = (value_cache * 127).astype(mx.int8).astype(mx.float32) / 127
            
            key = mx.concatenate([key_cache, key], axis=2)
            value = mx.concatenate([value_cache, value], axis=2)
            
            # Convert back to float32 for computation if needed
            if self.kv_cache_dtype != 'float32':
                key = key.astype(mx.float32)
                value = value.astype(mx.float32)

        # Enhanced attention mask with sink tokens and local windows
        current_seq_len = query.shape[2]  # T dimension
        if cache is not None:
            full_seq_len = key.shape[2]  # Full sequence length including cached tokens
        else:
            full_seq_len = current_seq_len
        
        # Create base causal mask
        causal_mask = mx.tril(mx.ones([current_seq_len, full_seq_len]))
        
        # Add attention sink tokens (first N tokens always visible)
        if self.attention_sink_tokens > 0 and full_seq_len > self.attention_sink_tokens:
            # Allow attention to first sink_tokens positions
            causal_mask[:, :self.attention_sink_tokens] = 1.0
        
        # Apply local window attention if specified
        if self.local_window_size is not None and self.local_window_size < full_seq_len:
            # Create sliding window mask
            window_mask = mx.zeros([current_seq_len, full_seq_len])
            for i in range(current_seq_len):
                # Current position in full sequence
                pos = full_seq_len - current_seq_len + i if cache is not None else i
                
                # Window boundaries
                start = max(0, pos - self.local_window_size + 1)
                end = pos + 1
                
                # Sink tokens are always visible
                if self.attention_sink_tokens > 0:
                    window_mask[i, :self.attention_sink_tokens] = 1.0
                
                # Local window
                window_mask[i, start:end] = 1.0
            
            # Combine with causal mask
            causal_mask = mx.minimum(causal_mask, window_mask)
        
        # Convert to additive mask (0 for allowed, -inf for masked)
        causal_mask = mx.where(causal_mask == 0, float('-inf'), 0.0)
        
        # Talking heads pre-attention mixing
        if self.talking_heads:
            # Mix attention weights across heads before attention
            # For GQA, we need to handle the different number of K/V heads
            # We'll apply talking heads to query heads only for simplicity
            query_for_talking = query.transpose(0, 2, 1, 3)  # (B, T, n_head, head_dim)
            B_q, T_q, n_head_q, head_dim_q = query_for_talking.shape
            query_mixed = query_for_talking.reshape(B_q * T_q * head_dim_q, n_head_q)
            query_mixed = self.talking_heads_pre(query_mixed)
            query = query_mixed.reshape(B_q, T_q, n_head_q, head_dim_q).transpose(0, 2, 1, 3)
        
        # Compute attention scores manually to apply softcap
        scale = 1.0 / (self.head_dim ** 0.5)
        
        if self.attention_softcap is not None:
            # Manual attention computation with softcap
            # For GQA, we need to broadcast K/V across query heads
            if self.n_head != self.n_kv_head:
                # Repeat K/V to match Q heads for softcap computation
                group_size = self.n_head // self.n_kv_head
                key_expanded = mx.repeat(key, group_size, axis=1)  # (B, n_head, T, head_dim)
                value_expanded = mx.repeat(value, group_size, axis=1)  # (B, n_head, T, head_dim)
            else:
                key_expanded = key
                value_expanded = value
            
            # Compute scores
            scores = mx.matmul(query, key_expanded.transpose(0, 1, 3, 2)) * scale  # (B, n_head, T, T)
            
            # Apply softcap
            scores = mx.tanh(scores / self.attention_softcap) * self.attention_softcap
            
            # Add causal mask
            scores = scores + causal_mask
            
            # Apply softmax
            attn_weights = mx.softmax(scores, axis=-1)
            attn_weights = self.attn_dropout(attn_weights)
            
            # Apply to values
            y = mx.matmul(attn_weights, value_expanded)  # (B, n_head, T, head_dim)
        else:
            # Use MLX flash attention with GQA support
            y = fast.scaled_dot_product_attention(
                query, 
                key, 
                value,
                scale=scale,
                mask=causal_mask
            )  # Output: (B, n_head, T, head_dim)
        
        # Talking heads post-attention mixing
        if self.talking_heads:
            y_for_talking = y.transpose(0, 2, 1, 3)  # (B, T, n_head, head_dim)
            B_y, T_y, n_head_y, head_dim_y = y_for_talking.shape
            y_mixed = y_for_talking.reshape(B_y * T_y * head_dim_y, n_head_y)
            y_mixed = self.talking_heads_post(y_mixed)
            y = y_mixed.reshape(B_y, T_y, n_head_y, head_dim_y).transpose(0, 2, 1, 3)
        
        # Apply per-head scaling before reshaping
        if self.per_head_scaling:
            # y: (B, n_head, T, head_dim)
            y = y * self.head_scales[None, :, None, None]
        
        # Re-assemble all head outputs side by side
        y = y.transpose(0, 2, 1, 3).reshape(B, T, self.n_head * self.head_dim)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, (key, value)

    @staticmethod
    def create_additive_causal_mask(N: int, dtype: mx.Dtype = mx.float32):
        return mx.tril(mx.ones([N, N])).reshape(1, 1, N, N).astype(dtype)


class RMTInputAdapter(nn.Module):
    """Build initial residual matrix X from token embeddings E (B,T,C).

    Projects embeddings to per-head data vectors and writes into a matrix via
    a learned mapping from heads to Dk.
    """
    def __init__(self, n_embd: int, Dk: int, Dv: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.n_head = n_head
        self.Dk = Dk
        self.Dv = Dv

        self.in_data = nn.Linear(n_embd, n_head * Dv, bias=False)
        self.in_write = nn.Linear(n_head, Dk, bias=False)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, E):  # E: (B,T,C)
        B, T, _ = E.shape
        H, Dk, Dv = self.n_head, self.Dk, self.Dv

        data = self.in_data(E).reshape(B, T, H, Dv)  # (B,T,H,Dv)
        data = self.dropout(data)
        data_H_last = data.transpose(0, 1, 3, 2)     # (B,T,Dv,H)
        X = self.in_write(data_H_last)               # (B,T,Dv,Dk)
        X = X.transpose(0, 1, 3, 2)                  # (B,T,Dk,Dv)
        return X


class MatrixRMSNorm(nn.Module):
    """RMSNorm over the Dv axis for matrix residuals X: (B,T,Dk,Dv)."""
    def __init__(self, Dv: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((Dv,))

    def __call__(self, X):
        rms = mx.sqrt(mx.mean(X * X, axis=-1, keepdims=True) + self.eps)  # (B,T,Dk,1)
        Y = X / rms
        return Y * self.weight


class RMTSelfAttention(nn.Module):
    """RMT attention: read from matrix residual, SDPA in Dv space, write back to matrix.
    
    Supports Grouped Query Attention (GQA) where n_head can be different from n_kv_head.
    """
    def __init__(self, Dk: int, Dv: int, n_head: int, n_kv_head: int = None, dropout: float = 0.0,
                 rope_base: float = 10000.0, rope_scale: float = 1.0):
        super().__init__()
        self.Dk, self.Dv = Dk, Dv
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        
        # Validate GQA configuration
        assert self.n_head % self.n_kv_head == 0, f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})"

        # Separate projections for GQA support
        self.read_q = nn.Linear(Dk, self.n_head, bias=False)
        self.read_k = nn.Linear(Dk, self.n_kv_head, bias=False)
        self.read_v = nn.Linear(Dk, self.n_kv_head, bias=False)

        self.write_o = nn.Linear(self.n_head, Dk, bias=False)

        self.rope = nn.RoPE(dims=Dv, traditional=False, base=rope_base, scale=rope_scale)

        self.attn_dropout = nn.Dropout(dropout)
        self.write_dropout = nn.Dropout(dropout)

    def _retrieve(self, X, reader: nn.Linear, n_heads: int):
        # X: (B,T,Dk,Dv) -> (B,n_heads,T,Dv)
        B, T, Dk, Dv = X.shape
        Y = X.transpose(0, 1, 3, 2)            # (B,T,Dv,Dk)
        Y = Y.reshape(B * T * Dv, Dk)          # (B*T*Dv, Dk)
        Y = reader(Y)                          # (B*T*Dv, n_heads)
        Y = Y.reshape(B, T, Dv, n_heads).transpose(0, 3, 1, 2)  # (B,n_heads,T,Dv)
        return Y

    def __call__(self, X, causal_mask, cache=None):
        # Use GQA: Query uses n_head, Key/Value use n_kv_head
        Q = self._retrieve(X, self.read_q, self.n_head)      # (B, n_head, T, Dv)
        K = self._retrieve(X, self.read_k, self.n_kv_head)   # (B, n_kv_head, T, Dv)
        V = self._retrieve(X, self.read_v, self.n_kv_head)   # (B, n_kv_head, T, Dv)

        # Apply RoPE in (B,T,H,Dv) view
        Q = Q.transpose(0, 2, 1, 3)  # (B,T,n_head,Dv)
        K = K.transpose(0, 2, 1, 3)  # (B,T,n_kv_head,Dv)
        Q = self.rope(Q)
        K = self.rope(K)
        Q = Q.transpose(0, 2, 1, 3)  # (B,n_head,T,Dv)
        K = K.transpose(0, 2, 1, 3)  # (B,n_kv_head,T,Dv)

        if cache is not None:
            Kc, Vc = cache
            if Kc is not None:
                K = mx.concatenate([Kc, K], axis=2)
            if Vc is not None:
                V = mx.concatenate([Vc, V], axis=2)

        # Build additive causal mask compatible with MLX fast attention
        cur_T = Q.shape[2]
        full_T = K.shape[2]
        causal = mx.tril(mx.ones([cur_T, full_T]))
        causal = mx.where(causal == 0, float('-inf'), 0.0)

        scale = 1.0 / (self.Dv ** 0.5)
        # MLX handles GQA automatically when Q has more heads than K/V
        Y = fast.scaled_dot_product_attention(Q, K, V, scale=scale, mask=causal)  # (B,n_head,T,Dv)
        Y = self.attn_dropout(Y)

        # Write back to matrix
        Y_t = Y.transpose(0, 2, 3, 1)     # (B,T,Dv,n_head)
        delta = self.write_o(Y_t)         # (B,T,Dv,Dk)
        delta = delta.transpose(0, 1, 3, 2)  # (B,T,Dk,Dv)
        delta = self.write_dropout(delta)

        new_cache = (K, V)
        return delta, new_cache


class RMTMLP(nn.Module):
    """RMT MLP: retrieve per-head vectors, shared SwiGLU MLP on Dv, write back."""
    def __init__(self, Dk: int, Dv: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.Dk, self.Dv, self.n_head = Dk, Dv, n_head

        self.read_f = nn.Linear(Dk, n_head, bias=False)   # Dk -> n_head
        self.write_f = nn.Linear(n_head, Dk, bias=False)  # n_head -> Dk

        hidden = 4 * Dv
        self.fc_gate = nn.Linear(Dv, hidden, bias=True)
        self.fc_val  = nn.Linear(Dv, hidden, bias=True)
        self.fc_out  = nn.Linear(hidden, Dv,  bias=True)

        self.dropout = nn.Dropout(dropout)

    def _retrieve(self, X):
        B, T, Dk, Dv = X.shape
        Y = X.transpose(0, 1, 3, 2)            # (B,T,Dv,Dk)
        Y = Y.reshape(B * T * Dv, Dk)          # (B*T*Dv, Dk)
        Y = self.read_f(Y)                     # (B*T*Dv, n_head)
        Y = Y.reshape(B, T, Dv, self.n_head).transpose(0, 3, 1, 2)  # (B,n_head,T,Dv)
        return Y

    def __call__(self, X):
        Z = self._retrieve(X)   # (B,n_head,T,Dv)
        B, n_head, T, Dv = Z.shape
        Z2 = Z.reshape(B * n_head * T, Dv)
        a = self.fc_gate(Z2)
        b = self.fc_val(Z2)
        y = a * mx.sigmoid(a) * b
        y = self.fc_out(y)
        y = self.dropout(y)
        Y = y.reshape(B, n_head, T, Dv)

        Yt = Y.transpose(0, 2, 3, 1)          # (B,T,Dv,n_head)
        delta = self.write_f(Yt)               # (B,T,Dv,Dk)
        delta = delta.transpose(0, 1, 3, 2)    # (B,T,Dk,Dv)
        return delta


class RMTCompressor(nn.Module):
    """Light compressor for RMT state to bound memory growth."""
    def __init__(self, Dk: int, Dv: int, compress_factor: float = 0.5):
        super().__init__()
        self.Dk, self.Dv = Dk, Dv
        self.compress_factor = compress_factor
        
        if compress_factor < 1.0:
            # Low-rank compression
            compressed_Dk = int(Dk * compress_factor)
            self.compress_proj = nn.Linear(Dk, compressed_Dk, bias=False)
            self.decompress_proj = nn.Linear(compressed_Dk, Dk, bias=False)
        else:
            self.compress_proj = None
            self.decompress_proj = None
    
    def __call__(self, X):
        if self.compress_proj is None:
            return X
        
        B, T, Dk, Dv = X.shape
        # Reshape for compression
        X_flat = X.transpose(0, 1, 3, 2).reshape(B * T * Dv, Dk)
        
        # Compress
        X_compressed = self.compress_proj(X_flat)
        
        # Decompress
        X_decompressed = self.decompress_proj(X_compressed)
        
        # Reshape back
        X_out = X_decompressed.reshape(B, T, Dv, Dk).transpose(0, 1, 3, 2)
        return X_out


class RMTBlock(nn.Module):
    def __init__(self, Dk, Dv, n_head, n_kv_head=None, dropout=0.0, rope_base=10000.0, rope_scale=1.0, 
                 compress_factor=1.0):
        super().__init__()
        self.ln1 = MatrixRMSNorm(Dv)
        self.attn = RMTSelfAttention(Dk, Dv, n_head, n_kv_head, dropout, rope_base, rope_scale)
        self.ln2 = MatrixRMSNorm(Dv)
        self.mlp = RMTMLP(Dk, Dv, n_head, dropout)
        
        # Optional compression
        self.compressor = RMTCompressor(Dk, Dv, compress_factor) if compress_factor < 1.0 else None

    def __call__(self, X, causal_mask, cache=None, apply_compression=False):
        A, new_cache = self.attn(self.ln1(X), causal_mask, cache=cache)
        X = X + A
        X = X + self.mlp(self.ln2(X))
        
        # Apply compression if requested
        if apply_compression and self.compressor is not None:
            X = self.compressor(X)
        
        return X, new_cache


class RMTOutputHead(nn.Module):
    def __init__(self, Dk: int, Dv: int, n_head: int, vocab_size: int):
        super().__init__()
        self.read_u = nn.Linear(Dk, n_head, bias=False)
        self.proj = nn.Linear(Dv, vocab_size, bias=False)
        self.n_head = n_head
        self.Dk, self.Dv = Dk, Dv

    def __call__(self, X):  # X: (B,T,Dk,Dv)
        B, T, Dk, Dv = X.shape
        Y = X.transpose(0, 1, 3, 2).reshape(B * T * Dv, Dk)  # (B*T*Dv, Dk)
        Y = self.read_u(Y).reshape(B, T, Dv, self.n_head)
        Y = mx.mean(Y, axis=-1)                               # (B,T,Dv)
        logits = self.proj(Y)                                 # (B,T,V)
        return logits


class MLP(nn.Module):

    def __init__(self, config):
        """
        Initializes the Multi-Layer Perceptron (MLP) layer with SwiGLU activation.

        Args:
            config (GPTConfig): An instance of the configuration class
                specifying the hyperparameters for the MLP layer.

        Attributes:
            linear1 (nn.Linear): First linear projection for SwiGLU gate.
            linear2 (nn.Linear): Second linear projection for SwiGLU gate.
            c_proj (nn.Linear): Linear layer for output projection.
            dropout (nn.Dropout): Dropout layer for regularization.

        Notes:
            - This implementation uses SwiGLU activation instead of GELU.
            - SwiGLU uses two linear projections with gated activation: a * sigmoid(a) * b
            - The hidden dimension is typically 8/3 * n_embd for SwiGLU to maintain
              similar parameter count to standard MLP with 4 * n_embd.
        """
        super().__init__()
        # For SwiGLU, we typically use 8/3 * n_embd as hidden dim to maintain similar param count
        # But for simplicity and compatibility, we'll use 4 * n_embd like the original
        hidden_dim = 4 * config.n_embd
        
        self.linear1 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)  # Gate projection
        self.linear2 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)  # Value projection
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, x):
        # SwiGLU activation: a * sigmoid(a) * b
        a = self.linear1(x)      # First projection (gate)
        b = self.linear2(x)      # Second projection (value)
        y = a * mx.sigmoid(a) * b  # SwiGLU activation
        y = self.c_proj(y)
        y = self.dropout(y)
        return y


class Block(nn.Module):

    def __init__(self, config, layer_idx=0):
        """
        Initializes a stabilized block with MoE, parallel residuals and learnable scaling.

        Args:
            config (GPTConfig): Configuration with stabilization features.
            layer_idx (int): Layer index for local window and MoE decisions.

        Features:
            - MoE FFN: Mixture of Experts in specified layers
            - Parallel residual: Attention and MLP computed in parallel
            - Residual alpha: Learnable scaling for each residual branch
            - DropPath: Stochastic depth for regularization
            - Pre-normalization with shared or separate norms
            - Local window attention in specified layers
        """
        super().__init__()
        self.parallel_residual = getattr(config, 'parallel_residual', False)
        self.layer_idx = layer_idx
        
        # Check if this layer should use local window attention
        local_window_layers = getattr(config, 'local_window_layers', None)
        use_local_window = (local_window_layers is not None and 
                           layer_idx in local_window_layers)
        
        # Check if this layer should use MoE
        moe_layers = getattr(config, 'moe_layers', None)
        use_moe = (moe_layers is not None and layer_idx in moe_layers)
        
        # Create a modified config for this layer
        layer_config = config
        if use_local_window:
            # This layer will use local window attention
            pass  # CausalSelfAttention will handle this based on config
        else:
            # Disable local window for this layer by setting it to None
            layer_config = type(config)(**config.__dict__)
            layer_config.local_window_size = None
        
        if self.parallel_residual:
            # Shared normalization for parallel residual
            self.ln = nn.RMSNorm(config.n_embd, eps=1e-5)
        else:
            # Separate normalizations for sequential residual
            self.ln_1 = nn.RMSNorm(config.n_embd, eps=1e-5)
            self.ln_2 = nn.RMSNorm(config.n_embd, eps=1e-5)
        
        self.attn = CausalSelfAttention(layer_config)
        
        # MoE or regular MLP
        if use_moe:
            n_experts = getattr(config, 'moe_n_experts', 4)
            expert_dropout = getattr(config, 'moe_expert_dropout', 0.1)
            self.mlp = TinyMoE(config, n_experts=n_experts, expert_dropout=expert_dropout)
            self.is_moe_layer = True
        else:
        self.mlp = MLP(config)
            self.is_moe_layer = False
        
        # DropPath for stochastic depth
        drop_path_rate = getattr(config, 'drop_path_rate', 0.0)
        # Scale drop path rate by layer depth
        layer_drop_rate = drop_path_rate * layer_idx / max(1, config.n_layer - 1)
        self.drop_path = DropPath(layer_drop_rate) if layer_drop_rate > 0 else None
        
        # Residual scaling parameters
        if getattr(config, 'residual_alpha_learnable', False):
            alpha_init = getattr(config, 'residual_alpha_init', 1.0)
            self.attn_alpha = mx.full((1,), alpha_init)
            self.mlp_alpha = mx.full((1,), alpha_init)
        else:
            self.attn_alpha = getattr(config, 'residual_alpha_init', 1.0)
            self.mlp_alpha = getattr(config, 'residual_alpha_init', 1.0)

    def __call__(self, x, mask, cache=None):
        # Initialize MoE load balance loss
        moe_loss = 0.0
        
        if self.parallel_residual:
            # Parallel residual: compute attention and MLP in parallel
            normed_x = self.ln(x)
            att, cache = self.attn(normed_x, mask, cache)
            
            # Handle MoE vs regular MLP
            if self.is_moe_layer:
                mlp_out, moe_loss = self.mlp(normed_x)
            else:
            mlp_out = self.mlp(normed_x)
            
            # Apply DropPath if configured
            if self.drop_path is not None:
                att = self.drop_path(att)
                mlp_out = self.drop_path(mlp_out)
            
            # Scale and combine residuals
            if isinstance(self.attn_alpha, mx.array):
                x = x + self.attn_alpha * att + self.mlp_alpha * mlp_out
            else:
                x = x + self.attn_alpha * att + self.mlp_alpha * mlp_out
        else:
            # Sequential residual: traditional transformer block
            att, cache = self.attn(self.ln_1(x), mask, cache)
            
            # Apply DropPath to attention
            if self.drop_path is not None:
                att = self.drop_path(att)
            
            if isinstance(self.attn_alpha, mx.array):
                x = x + self.attn_alpha * att
            else:
                x = x + self.attn_alpha * att
            
            # Handle MoE vs regular MLP
            if self.is_moe_layer:
                mlp_out, moe_loss = self.mlp(self.ln_2(x))
            else:
            mlp_out = self.mlp(self.ln_2(x))
            
            # Apply DropPath to MLP
            if self.drop_path is not None:
                mlp_out = self.drop_path(mlp_out)
            
            if isinstance(self.mlp_alpha, mx.array):
                x = x + self.mlp_alpha * mlp_out
            else:
                x = x + self.mlp_alpha * mlp_out
        
        return x, cache, moe_loss


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12  # Number of query heads (for backward compatibility)
    n_kv_head: int = None  # Number of key/value heads (if None, defaults to n_head for MHA)
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False # False for bias-free linears (better with RMSNorm and cleaner optimization)
    
    # RoPE parameters
    rope_base: float = 10000.0  # Base for RoPE frequency computation
    rope_scale: float = 1.0     # Scale factor for RoPE
    
    # RMT parameters
    use_rmt: bool = True
    Dk: int = 256
    Dv: int = 64
    rmt_dropout: float = 0.0
    
    # Stabilization features
    parallel_residual: bool = True  # Run attention and MLP in parallel
    residual_alpha_learnable: bool = True  # Learnable residual scaling
    residual_alpha_init: float = None  # Initial residual alpha (if None, uses 1/sqrt(2*n_layer))
    qk_norm: bool = True  # QK normalization to stabilize attention
    qk_norm_scale_learnable: bool = True  # Learnable scale for QK norm
    attention_softcap: float = 20.0  # Softcap for attention logits (None to disable)
    talking_heads: bool = True  # Talking heads mechanism
    weight_tying: bool = True  # Tie input/output embeddings
    logit_scale: float = 1.0  # Final logit scaling factor
    z_loss_weight: float = 0.0  # Z-loss weight for reducing overconfidence
    
    # Long-context features
    rope_scale_mode: str = 'linear'  # 'linear', 'ntk', 'yarn' for context extrapolation
    rope_partial_factor: float = 1.0  # Fraction of head dims to apply RoPE (1.0 = full RoPE)
    attention_sink_tokens: int = 4  # Number of sink tokens always visible
    local_window_size: int = None  # Sliding window size (None = full attention)
    local_window_layers: list = None  # Layers to apply sliding window (None = none)
    rmt_refresh_interval: int = None  # Refresh RMT state every N blocks (None = no refresh)
    rmt_compress_factor: float = 1.0  # Compression factor for RMT state (1.0 = no compression)
    kv_cache_dtype: str = 'float32'  # 'float32', 'float16', 'int8' for KV cache
    use_paged_kv: bool = False  # Use paged KV cache for memory efficiency
    
    # MoE features
    moe_layers: list = None  # Layers to use MoE (e.g., [3, 5] for layers 3 and 5)
    moe_n_experts: int = 4  # Number of experts per MoE layer
    moe_expert_dropout: float = 0.1  # Expert dropout rate for routing
    moe_load_balance_weight: float = 0.01  # Weight for load balancing loss
    per_head_scaling: bool = True  # Learnable per-head scaling
    drop_path_rate: float = 0.0  # Stochastic depth rate
    
    # Advanced sampling
    enable_advanced_sampling: bool = True  # Enable advanced sampling methods
    speculative_decode: bool = False  # Enable speculative decoding interface
    
    def __post_init__(self):
        # Set n_kv_head to n_head if not specified (standard MHA)
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        
        # Set default residual alpha based on depth
        if self.residual_alpha_init is None:
            self.residual_alpha_init = 1.0 / (2 * self.n_layer) ** 0.5
        
        # Validate GQA configuration
        assert self.n_head % self.n_kv_head == 0, f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})"
        assert self.n_embd % self.n_head == 0, f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"


class GPT(nn.Module):
    def __init__(self, config):
        """
        Initializes a GPT (Generative Pre-trained Transformer) model with SwiGLU activation.

        Args:
            config (GPTConfig): An instance of the configuration class
                specifying the hyperparameters for the GPT model.

        Attributes:
            config (GPTConfig): Configuration instance containing model hyperparameters.
            wte (nn.Embedding): Token embedding layer for input tokens.
            transformer (List[Block]): List of transformer blocks with RoPE, RMSNorm, and SwiGLU.
            out_proj (nn.Linear): Linear layer for output projection.
            
        Note:
            This model uses RoPE (Rotary Positional Encoding) instead of learned 
            positional embeddings, providing better extrapolation to longer sequences.
            It also uses SwiGLU activation in the MLP layers for improved performance.
        """
        super().__init__()

        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        
        # Weight tying and logit scaling
        self.weight_tying = getattr(config, 'weight_tying', False)
        self.logit_scale = getattr(config, 'logit_scale', 1.0)
        self.z_loss_weight = getattr(config, 'z_loss_weight', 0.0)

        # Long-context features
        self.rmt_refresh_interval = getattr(config, 'rmt_refresh_interval', None)
        self.rmt_compress_factor = getattr(config, 'rmt_compress_factor', 1.0)

        if config.use_rmt:
            self.rmt_in = RMTInputAdapter(
                n_embd=config.n_embd, Dk=config.Dk, Dv=config.Dv,
                n_head=config.n_head, dropout=config.rmt_dropout
            )
            self.blocks = [
                RMTBlock(
                    Dk=config.Dk, Dv=config.Dv, n_head=config.n_head, n_kv_head=config.n_kv_head,
                    dropout=config.rmt_dropout,
                    rope_base=config.rope_base, rope_scale=config.rope_scale,
                    compress_factor=self.rmt_compress_factor,
                ) for _ in range(config.n_layer)
            ]
            self.ln_f = MatrixRMSNorm(config.Dv)
            self.rmt_out = RMTOutputHead(config.Dk, config.Dv, config.n_head, config.vocab_size)
        else:
            # Fallback vector transformer path with layer-specific configs
            self.transformer = [Block(config, layer_idx=i) for i in range(config.n_layer)]
            self.ln_f = nn.RMSNorm(config.n_embd, eps=1e-5)
            
            # Output projection with optional weight tying
            if self.weight_tying:
                # Weight tying: output projection shares weights with input embedding
                self.out_proj = None  # Will use wte.weight.T in forward pass
            else:
                self.out_proj = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def _sample_next_token(self, x, temperature):
        logits = mx.expand_dims(x[:, -1], axis=0) @ self.wte.weight.T
        y = logits[:, -1, :]
        y = mx.random.categorical(y * (1 / temperature))
        return y

    def generate(self, idx: mx.array, max_new_tokens=256, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        # Use the provided initial sequence context (idx) - don't override it!
        # idx should already be properly shaped as (batch_size, sequence_length)
        
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            # idx_cond = idx if idx[0].shape[1] <= self.config.block_size else idx[:, -self.config.block_size:]
            idx_cond = idx if idx.shape[1] <= self.config.block_size else idx[:, -self.config.block_size:]

            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)

            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature

            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = components_topk(logits, min(top_k, logits.shape[-1]))
                
                # Get the k-th largest value (the cutoff threshold)
                if v.size > 0:
                    cutoff = v[0, -1]  # Last value in top-k (smallest of the top-k)
                    # Set all values below cutoff to -inf
                    logits = mx.where(logits < cutoff, float('-inf'), logits)

            # apply softmax to convert logits to (normalized) probabilities
            probs = mx.softmax(logits)

            # Sample from the distribution
            idx_next = mx.random.categorical(probs, 1)

            # Append sampled index to the running sequence
            idx = mx.concatenate([idx, mx.expand_dims(idx_next, axis=0)], axis=1)

        return idx

    def _forward_transformer(
        self, x: mx.array, mask=None, cache=None, build_cache=False
    ):
        # Only token embeddings, no positional embeddings (RoPE handles positions)
        if self.config.use_rmt:
            tok = self.wte(x)
            X = self.rmt_in(self.drop(tok))  # (B,T,Dk,Dv)
            kv_cache = []
            if cache is not None:
                for i in range(len(cache)):
                    # Apply refresh mechanism if specified
                    apply_compression = (self.rmt_refresh_interval is not None and 
                                       i % self.rmt_refresh_interval == 0 and 
                                       i > 0)
                    X, cache[i] = self.blocks[i](X, mask, cache=cache[i], 
                                               apply_compression=apply_compression)
            else:
                for i, block in enumerate(self.blocks):
                    # Apply compression periodically during training too
                    apply_compression = (self.rmt_refresh_interval is not None and 
                                       i % self.rmt_refresh_interval == 0 and 
                                       i > 0)
                    X, curr_cache = block(X, mask, apply_compression=apply_compression)
                    if build_cache:
                        kv_cache.append(curr_cache)
            X = self.ln_f(X)
            return X, kv_cache if build_cache else cache
        else:
            tok_emb = self.wte(x)
            x = self.drop(tok_emb)
            kv_cache = []
            total_moe_loss = 0.0
            
            if cache is not None:
                for i in range(len(cache)):
                    x, cache[i], moe_loss = self.transformer[i](x, mask=None, cache=cache[i])
                    total_moe_loss += moe_loss
            else:
                for block in self.transformer:
                    x, curr_cache, moe_loss = block(x, mask=mask)
                    total_moe_loss += moe_loss
                    if build_cache:
                        kv_cache.append(curr_cache)
            x = self.ln_f(x)
            
            # Store MoE loss for training
            self._last_moe_loss = total_moe_loss
            
            return x, kv_cache if build_cache else cache

    def __call__(self, x):
        b, t = x.shape
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        mask = CausalSelfAttention.create_additive_causal_mask(x.shape[1])

        if self.config.use_rmt:
            X, _ = self._forward_transformer(x, mask=mask)
            logits = self.rmt_out(X)
        else:
            x, _ = self._forward_transformer(x, mask=mask)
            if self.weight_tying:
                # Use tied weights: x @ embedding.T
                logits = mx.matmul(x, self.wte.weight.T)
            else:
                logits = self.out_proj(x)
        
        # Apply logit scaling
        if self.logit_scale != 1.0:
            logits = logits * self.logit_scale
        
        return logits

    def compute_z_loss(self, logits):
        """
        Compute z-loss to reduce overconfidence.
        
        Z-loss encourages the model to have lower confidence by penalizing
        large logit magnitudes. This helps with calibration and reduces
        overconfident predictions.
        
        Args:
            logits: Model logits of shape (batch, sequence, vocab)
            
        Returns:
            z_loss: Scalar loss value
        """
        if self.z_loss_weight <= 0:
            return 0.0
        
        # Compute log(sum(exp(logits))) for each position
        log_sum_exp = mx.logsumexp(logits, axis=-1)  # (batch, sequence)
        
        # Z-loss is the squared magnitude of log_sum_exp
        z_loss = mx.mean(log_sum_exp ** 2)
        
        return self.z_loss_weight * z_loss
        
    def get_moe_loss(self):
        """Get the last computed MoE load balancing loss."""
        return getattr(self, '_last_moe_loss', 0.0)

    def advanced_generate(self, idx: mx.array, max_new_tokens=256, temperature=1.0, 
                         top_k=None, top_p=0.9, typical_p=None, repetition_penalty=1.0,
                         mirostat_mode=0, mirostat_tau=5.0, mirostat_eta=0.1):
        """
        Advanced generation with multiple sampling strategies.
        
        Args:
            idx: Input token sequence
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling cutoff
            top_p: Nucleus sampling probability
            typical_p: Typical sampling probability
            repetition_penalty: Penalty for repeating tokens
            mirostat_mode: 0=disabled, 1=Mirostat v1, 2=Mirostat v2
            mirostat_tau: Target cross-entropy (for Mirostat)
            mirostat_eta: Learning rate (for Mirostat)
        """
        if not getattr(self.config, 'enable_advanced_sampling', False):
            # Fallback to simple generation
            return self.generate(idx, max_new_tokens, temperature, top_k)
        
        # Mirostat state
        mirostat_s = mirostat_tau * 2.0 if mirostat_mode > 0 else 0.0
        
        for _ in range(max_new_tokens):
            # Context cropping
            idx_cond = idx if idx.shape[1] <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward pass
            logits = self(idx_cond)[:, -1, :] / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                # Penalize tokens that have appeared before
                for token in set(idx[0].tolist()):
                    if logits[0, token] > 0:
                        logits[0, token] /= repetition_penalty
                    else:
                        logits[0, token] *= repetition_penalty
            
            # Mirostat sampling
            if mirostat_mode == 2:
                # Mirostat v2
                sorted_logits, sorted_indices = mx.sort(logits, axis=-1, descending=True)
                probs = mx.softmax(sorted_logits, axis=-1)
                
                # Find cutoff based on surprise
                cumulative_probs = mx.cumsum(probs, axis=-1)
                surprisal = -mx.log2(probs)
                
                # Dynamic cutoff based on target entropy
                cutoff_idx = mx.argmax((cumulative_probs > mirostat_s).astype(mx.float32), axis=-1)
                cutoff_idx = mx.maximum(cutoff_idx, 1)  # At least keep top token
                
                # Apply cutoff
                mask = mx.arange(logits.shape[-1]) <= cutoff_idx[:, None]
                logits = mx.where(mask, logits, float('-inf'))
                
                # Update mirostat_s
                probs = mx.softmax(logits, axis=-1)
                entropy = -mx.sum(probs * mx.log2(probs + 1e-10), axis=-1)
                mirostat_s = mirostat_s + mirostat_eta * (mirostat_tau - entropy.item())
                
            elif mirostat_mode == 1:
                # Mirostat v1 (simpler version)
                sorted_logits, sorted_indices = mx.sort(logits, axis=-1, descending=True)
                probs = mx.softmax(sorted_logits, axis=-1)
                
                cutoff = int(mirostat_s)
                cutoff = max(1, min(cutoff, logits.shape[-1]))
                
                mask = mx.arange(logits.shape[-1]) < cutoff
                logits = mx.where(mask, logits, float('-inf'))
            
            # Typical sampling
            elif typical_p is not None:
                probs = mx.softmax(logits, axis=-1)
                # Compute entropy and surprisal
                entropy = -mx.sum(probs * mx.log(probs + 1e-10), axis=-1, keepdims=True)
                surprisal = -mx.log(probs)
                
                # Keep tokens close to expected surprisal
                diff = mx.abs(surprisal - entropy)
                indices = mx.argsort(diff, axis=-1)
                
                sorted_probs = mx.take_along_axis(probs, indices, axis=-1)
                cumulative = mx.cumsum(sorted_probs, axis=-1)
                
                cutoff = mx.argmax((cumulative >= typical_p).astype(mx.float32), axis=-1)
                cutoff = mx.maximum(cutoff, 1)
                
                mask = mx.arange(logits.shape[-1]) <= cutoff[:, None]
                reordered_mask = mx.take_along_axis(mask, mx.argsort(indices, axis=-1), axis=-1)
                logits = mx.where(reordered_mask, logits, float('-inf'))
            
            # Top-p (nucleus) sampling
            elif top_p < 1.0:
                sorted_logits, sorted_indices = mx.sort(logits, axis=-1, descending=True)
                sorted_probs = mx.softmax(sorted_logits, axis=-1)
                cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
                
                # Find cutoff
                mask = cumulative_probs <= top_p
                # Always keep at least the top token
                mask = mask.at[:, 0].set(True)
                
                # Apply mask
                cutoff_logits = mx.where(mask, sorted_logits, float('-inf'))
                # Reorder back to original positions
                logits = mx.take_along_axis(cutoff_logits, mx.argsort(sorted_indices, axis=-1), axis=-1)
            
            # Top-k sampling
            elif top_k is not None:
                v, _ = components_topk(logits, min(top_k, logits.shape[-1]))
                if v.size > 0:
                    cutoff = v[0, -1]
                    logits = mx.where(logits < cutoff, float('-inf'), logits)
            
            # Sample
            probs = mx.softmax(logits, axis=-1)
            idx_next = mx.random.categorical(probs, 1)
            idx = mx.concatenate([idx, mx.expand_dims(idx_next, axis=0)], axis=1)
        
        return idx

    def speculative_decode_interface(self, draft_model=None, draft_tokens=4):
        """
        Interface for speculative decoding (placeholder).
        
        Args:
            draft_model: Smaller model for draft generation
            draft_tokens: Number of draft tokens to generate
        
        Note: This is a placeholder interface for future implementation.
        """
        if not getattr(self.config, 'speculative_decode', False):
            return None
        
        # Placeholder for speculative decoding implementation
        # In practice, this would:
        # 1. Use draft_model to quickly generate candidate tokens
        # 2. Verify candidates with the main model in parallel
        # 3. Accept/reject based on probability ratios
        # 4. Fallback to regular sampling for rejected tokens
        
        return {
            'enabled': self.config.speculative_decode,
            'draft_model': draft_model,
            'draft_tokens': draft_tokens,
            'status': 'interface_ready'
        }
        

 
    
