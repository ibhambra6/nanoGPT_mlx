import mlx.core as mx
import mlx.nn as nn
import mlx.core.fast as fast

from dataclasses import dataclass
from .components import topk as components_topk


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        """
        Initializes the Causal Self-Attention layer.

        Args:
            config (GPTConfig): An instance of the configuration class
                specifying the hyperparameters for the Causal Self-Attention layer.

        Attributes:
            c_attn (nn.Linear): Linear layer for key, query, and value projections.
            c_proj (nn.Linear): Linear layer for output projection.
            attn_dropout (nn.Dropout): Dropout layer for attention weights.
            resid_dropout (nn.Dropout): Dropout layer for residual connections.
            n_head (int): Number of attention heads.
            n_embd (int): Dimensionality of the embedding.
            dropout (float): Dropout probability.

        Notes:
            - The configuration class should contain the necessary hyperparameters for
              configuring the Causal Self-Attention layer.
            - The `c_attn` layer combines key, query, and value projections in a batch.
            - The `c_proj` layer handles the output projection.
            - The `attn_dropout` and `resid_dropout` layers apply dropout regularization.
            - The `n_head`, `n_embd`, and `dropout` attributes store hyperparameter values.
        """
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # RoPE (Rotary Positional Encoding)
        self.head_dim = config.n_embd // config.n_head
        rope_base = getattr(config, 'rope_base', 10000.0)
        rope_scale = getattr(config, 'rope_scale', 1.0)
        self.rope = nn.RoPE(
            dims=self.head_dim,
            traditional=False,  # Use efficient implementation
            base=rope_base,
            scale=rope_scale
        )

    def __call__(self, x, mask, cache=None):
        B, T, C = x.shape # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query, key, value = mx.split(self.c_attn(x), 3, axis=2)
        key = key.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3) # (B, nh, T, hs)
        query = query.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3) # (B, nh, T, hs)
        value = value.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3) # (B, nh, T, hs)

        # Apply RoPE to query and key
        # RoPE expects input in shape (batch, sequence, heads, head_dim), so we need to transpose
        query = query.transpose(0, 2, 1, 3)  # (B, T, nh, hs)
        key = key.transpose(0, 2, 1, 3)      # (B, T, nh, hs)
        
        if cache is not None:
            # For cached generation, we need to handle the offset
            offset = cache[0].shape[2] if cache[0] is not None else 0
            query = self.rope(query, offset=offset)
            key = self.rope(key, offset=offset)
        else:
            query = self.rope(query)
            key = self.rope(key)
        
        # Transpose back to (B, nh, T, hs) for attention computation
        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)

        if cache is not None:
            key_cache, value_cache = cache
            key = mx.concatenate([key_cache, key], axis=2)
            value = mx.concatenate([value_cache, value], axis=2)

        # Flash attention implementation using MLX's scaled_dot_product_attention
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
        
        # Use MLX flash attention
        # MLX scaled_dot_product_attention expects scale parameter and mask parameter
        scale = 1.0 / (key.shape[-1] ** 0.5)  # 1/sqrt(head_dim)
        y = fast.scaled_dot_product_attention(
            query, 
            key, 
            value,
            scale=scale,
            mask=causal_mask
        )
        
        # Re-assemble all head outputs side by side
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, (key, value)

    @staticmethod
    def create_additive_causal_mask(N: int, dtype: mx.Dtype = mx.float32):
        return mx.tril(mx.ones([N, N])).reshape(1, 1, N, N).astype(dtype)


class MLP(nn.Module):

    def __init__(self, config):
        """
        Initializes the Multi-Layer Perceptron (MLP) layer.

        Args:
            config (GPTConfig): An instance of the configuration class
                specifying the hyperparameters for the MLP layer.

        Attributes:
            c_fc (nn.Linear): Linear layer for fully connected transformations.
            gelu (nn.GELU): GELU activation function.
            c_proj (nn.Linear): Linear layer for output projection.
            dropout (nn.Dropout): Dropout layer for regularization.

        Notes:
            - Ensure that the `config` parameter is an instance of `MLPConfig`.
            - The configuration class should contain the necessary hyperparameters for
              configuring the MLP layer.
            - The `c_fc` layer performs a fully connected transformation.
            - The `gelu` layer applies the GELU activation function.
            - The `c_proj` layer handles the output projection.
            - The `dropout` layer applies dropout regularization.
        """
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        """
        Initializes a block in a transformer architecture.

        Args:
            config (BlockConfig): An instance of the configuration class
                specifying the hyperparameters for the block.

        Attributes:
            ln_1 (RMSNorm): RMS normalization for the first sub-block.
            attn (CausalSelfAttention): Causal Self-Attention sub-block.
            ln_2 (RMSNorm): RMS normalization for the second sub-block.
            mlp (MLP): Multi-Layer Perceptron sub-block.

        Notes:
            - Ensure that the `config` parameter is an instance of `BlockConfig`.
            - The configuration class should contain the necessary hyperparameters for
              configuring the block.
            - The `ln_1` layer performs RMS normalization for the first sub-block.
            - The `attn` layer represents the Causal Self-Attention sub-block.
            - The `ln_2` layer performs RMS normalization for the second sub-block.
            - The `mlp` layer represents the Multi-Layer Perceptron sub-block.
        """
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embd, eps=1e-5)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.RMSNorm(config.n_embd, eps=1e-5)
        self.mlp = MLP(config)

    def __call__(self, x, mask, cache=None):
        att, cache = self.attn(self.ln_1(x), mask, cache)
        x = x + att
        x = x + self.mlp(self.ln_2(x))
        return x, cache


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears, like GPT-2. False: a bit better and faster (RMSNorm doesn't use bias)
    # RoPE parameters
    rope_base: float = 10000.0  # Base for RoPE frequency computation
    rope_scale: float = 1.0     # Scale factor for RoPE


class GPT(nn.Module):
    def __init__(self, config):
        """
        Initializes a GPT (Generative Pre-trained Transformer) model.

        Args:
            config (GPTConfig): An instance of the configuration class
                specifying the hyperparameters for the GPT model.

        Attributes:
            config (GPTConfig): Configuration instance containing model hyperparameters.
            wte (nn.Embedding): Token embedding layer for input tokens.
            transformer (List[Block]): List of transformer blocks with RoPE and RMSNorm.
            out_proj (nn.Linear): Linear layer for output projection.
            
        Note:
            This model uses RoPE (Rotary Positional Encoding) instead of learned 
            positional embeddings, providing better extrapolation to longer sequences.
        """
        super().__init__()

        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # No positional embeddings needed with RoPE
        self.drop = nn.Dropout(config.dropout)
        self.transformer = [Block(config) for _ in range(config.n_layer)]
        self.ln_f = nn.RMSNorm(config.n_embd, eps=1e-5)
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

        x, _ = self._forward_transformer(x, mask=mask)
        return self.out_proj(x)
        

 
    
