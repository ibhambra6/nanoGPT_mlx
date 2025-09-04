import mlx.core as mx
import mlx.nn as nn


class LayerNorm(nn.Module):
    """Lightweight LayerNorm with optional bias.

    Mirrors nn.LayerNorm behavior but allows toggling bias independently.
    """

    def __init__(self, dims: int, eps: float = 1e-5, affine: bool = True, bias: bool = False):
        super().__init__()
        self.eps = eps
        self.dims = dims
        self.use_bias = bias

        if affine:
            self.weight = mx.ones((dims,))
            if bias:
                self.bias = mx.zeros((dims,))
        else:
            self.weight = None
            self.bias = None

    def _extra_repr(self):
        return f"{self.dims}, eps={self.eps}, affine={'weight' in self}, bias={self.use_bias}"

    def __call__(self, x):
        means = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x = (x - means) * mx.rsqrt(var + self.eps)

        if self.weight is not None:
            x = self.weight * x

        if self.use_bias and hasattr(self, "bias"):
            x = x + self.bias

        return x


def topk(values: mx.array, k: int):
    """Simple top-k for MLX tensors returning (values, indices) with batch support.

    Accepts [..., V] shaped input and returns top-k along the last axis.
    """
    # Flatten batch dims, keep last dim as feature dim
    last_dim = values.shape[-1]
    flat = values.reshape((-1, last_dim))

    # argsort descending
    idx = mx.argsort(flat)
    idx = mx.take(idx, mx.arange(idx.size - 1, -1, -1))
    idx = idx.reshape(flat.shape)

    # take first k indices per row
    k = min(k, last_dim)
    select = mx.arange(k)
    topk_idx = mx.take_along_axis(idx, mx.tile(select.reshape(1, -1), (idx.shape[0], 1)), axis=1)
    topk_vals = mx.take_along_axis(flat, topk_idx, axis=1)

    # Restore batch dims
    batch_shape = values.shape[:-1]
    topk_vals = topk_vals.reshape(batch_shape + (k,))
    topk_idx = topk_idx.reshape(batch_shape + (k,))
    return topk_vals, topk_idx

