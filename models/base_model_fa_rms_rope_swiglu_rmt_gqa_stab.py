import mlx.core as mx
import mlx.nn as nn
import mlx.core.fast as fast

from dataclasses import dataclass
from .components import topk as components_topk


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
        
        # RoPE (Rotary Positional Encoding)
        rope_base = getattr(config, 'rope_base', 10000.0)
        rope_scale = getattr(config, 'rope_scale', 1.0)
        self.rope = nn.RoPE(
            dims=self.head_dim,
            traditional=False,  # Use efficient implementation
            base=rope_base,
            scale=rope_scale
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

        if cache is not None:
            key_cache, value_cache = cache
            key = mx.concatenate([key_cache, key], axis=2)
            value = mx.concatenate([value_cache, value], axis=2)

        # Flash attention implementation using MLX's scaled_dot_product_attention with GQA
        # MLX automatically handles grouped query attention when n_head > n_kv_head
        # Convert mask to the format expected by scaled_dot_product_attention
        # MLX expects causal mask as additive mask (0 for allowed, -inf for masked)
        current_seq_len = query.shape[2]  # T dimension
        if cache is not None:
            # When using cache, we need to handle the full sequence length
            full_seq_len = key.shape[2]  # Full sequence length including cached tokens
            causal_mask = mx.tril(mx.ones([current_seq_len, full_seq_len]))
        else:
            causal_mask = mx.tril(mx.ones([current_seq_len, current_seq_len]))
        
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


class RMTBlock(nn.Module):
    def __init__(self, Dk, Dv, n_head, n_kv_head=None, dropout=0.0, rope_base=10000.0, rope_scale=1.0):
        super().__init__()
        self.ln1 = MatrixRMSNorm(Dv)
        self.attn = RMTSelfAttention(Dk, Dv, n_head, n_kv_head, dropout, rope_base, rope_scale)
        self.ln2 = MatrixRMSNorm(Dv)
        self.mlp = RMTMLP(Dk, Dv, n_head, dropout)

    def __call__(self, X, causal_mask, cache=None):
        A, new_cache = self.attn(self.ln1(X), causal_mask, cache=cache)
        X = X + A
        X = X + self.mlp(self.ln2(X))
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

    def __init__(self, config):
        """
        Initializes a stabilized block with parallel residuals and learnable scaling.

        Args:
            config (GPTConfig): Configuration with stabilization features.

        Features:
            - Parallel residual: Attention and MLP computed in parallel
            - Residual alpha: Learnable scaling for each residual branch
            - Pre-normalization with shared or separate norms
        """
        super().__init__()
        self.parallel_residual = getattr(config, 'parallel_residual', False)
        
        if self.parallel_residual:
            # Shared normalization for parallel residual
            self.ln = nn.RMSNorm(config.n_embd, eps=1e-5)
        else:
            # Separate normalizations for sequential residual
            self.ln_1 = nn.RMSNorm(config.n_embd, eps=1e-5)
            self.ln_2 = nn.RMSNorm(config.n_embd, eps=1e-5)
        
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        
        # Residual scaling parameters
        if getattr(config, 'residual_alpha_learnable', False):
            alpha_init = getattr(config, 'residual_alpha_init', 1.0)
            self.attn_alpha = mx.full((1,), alpha_init)
            self.mlp_alpha = mx.full((1,), alpha_init)
        else:
            self.attn_alpha = getattr(config, 'residual_alpha_init', 1.0)
            self.mlp_alpha = getattr(config, 'residual_alpha_init', 1.0)

    def __call__(self, x, mask, cache=None):
        if self.parallel_residual:
            # Parallel residual: compute attention and MLP in parallel
            normed_x = self.ln(x)
            att, cache = self.attn(normed_x, mask, cache)
            mlp_out = self.mlp(normed_x)
            
            # Scale and combine residuals
            if isinstance(self.attn_alpha, mx.array):
                x = x + self.attn_alpha * att + self.mlp_alpha * mlp_out
            else:
                x = x + self.attn_alpha * att + self.mlp_alpha * mlp_out
        else:
            # Sequential residual: traditional transformer block
            att, cache = self.attn(self.ln_1(x), mask, cache)
            if isinstance(self.attn_alpha, mx.array):
                x = x + self.attn_alpha * att
            else:
                x = x + self.attn_alpha * att
            
            mlp_out = self.mlp(self.ln_2(x))
            if isinstance(self.mlp_alpha, mx.array):
                x = x + self.mlp_alpha * mlp_out
            else:
                x = x + self.mlp_alpha * mlp_out
        
        return x, cache


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
                ) for _ in range(config.n_layer)
            ]
            self.ln_f = MatrixRMSNorm(config.Dv)
            self.rmt_out = RMTOutputHead(config.Dk, config.Dv, config.n_head, config.vocab_size)
        else:
            # Fallback vector transformer path (original)
            self.transformer = [Block(config) for _ in range(config.n_layer)]
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
                    X, cache[i] = self.blocks[i](X, mask, cache=cache[i])
            else:
                for block in self.blocks:
                    X, curr_cache = block(X, mask)
                    if build_cache:
                        kv_cache.append(curr_cache)
            X = self.ln_f(X)
            return X, kv_cache if build_cache else cache
        else:
            tok_emb = self.wte(x)
            x = self.drop(tok_emb)
            kv_cache = []
            if cache is not None:
                for i in range(len(cache)):
                    x, cache[i] = self.transformer[i](x, mask=None, cache=cache[i])
            else:
                for block in self.transformer:
                    x, curr_cache = block(x, mask=mask)
                    if build_cache:
                        kv_cache.append(curr_cache)
            x = self.ln_f(x)
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
        

 
    
