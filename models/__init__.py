"""Model factory utilities for nanoGPT_mlx.

Provides helpers to instantiate the correct GPT variant based on a saved config
and load weights from disk.
"""

import os
import json
from typing import Tuple, Dict, Any

import mlx.core as mx
from mlx.utils import tree_unflatten


def select_arch_from_config(config: Dict[str, Any]) -> str:
    """Pick model architecture based on config fields.

    - If RMT + GQA + Stabilization + Long-context -> fa_rms_rope_swiglu_rmt_gqa_stab_long
    - If RMT + GQA + Stabilization -> fa_rms_rope_swiglu_rmt_gqa_stab
    - If RMT + GQA fields are present -> fa_rms_rope_swiglu_rmt_gqa
    - If RMT fields are present -> fa_rms_rope_swiglu_rmt
    - If RoPE fields are present -> fa_rms_rope
    - Else default to flash-attention variant
    """
    if config.get('use_rmt'):
        # Check if GQA is configured (n_kv_head different from n_head)
        n_head = config.get('n_head')
        n_kv_head = config.get('n_kv_head')
        if n_kv_head is not None and n_kv_head != n_head:
            # Check for stabilization features
            stabilization_features = [
                'parallel_residual', 'residual_alpha_learnable', 'qk_norm',
                'attention_softcap', 'talking_heads', 'weight_tying'
            ]
            # Check for long-context features
            long_context_features = [
                'rope_scale_mode', 'rope_partial_factor', 'attention_sink_tokens',
                'local_window_size', 'rmt_refresh_interval', 'rmt_compress_factor'
            ]
            # Check for MoE features
            moe_features = [
                'moe_layers', 'moe_n_experts', 'per_head_scaling', 'drop_path_rate'
            ]
            
            has_stabilization = any(config.get(feature) for feature in stabilization_features)
            has_long_context = any(config.get(feature) is not None for feature in long_context_features)
            has_moe = any(config.get(feature) is not None for feature in moe_features)
            
            if has_stabilization and has_long_context and has_moe:
                return 'fa_rms_rope_swiglu_rmt_gqa_stab_long_moe'
            elif has_stabilization and has_long_context:
                return 'fa_rms_rope_swiglu_rmt_gqa_stab_long'
            elif has_stabilization:
                return 'fa_rms_rope_swiglu_rmt_gqa_stab'
            return 'fa_rms_rope_swiglu_rmt_gqa'
        return 'fa_rms_rope_swiglu_rmt'
    if 'rope_base' in config or 'rope_scale' in config:
        return 'fa_rms_rope'
    return 'fa'


def make_model_from_config(config: Dict[str, Any]):
    """Instantiate a GPT model matching the config.

    Returns the model and the resolved arch string.
    """
    arch = select_arch_from_config(config)
    if arch == 'fa_rms_rope_swiglu_rmt_gqa_stab_long_moe':
        from .base_model_fa_rms_rope_swiglu_rmt_gqa_stab_long_moe import GPT, GPTConfig
    elif arch == 'fa_rms_rope_swiglu_rmt_gqa_stab_long':
        from .base_model_fa_rms_rope_swiglu_rmt_gqa_stab_long import GPT, GPTConfig
    elif arch == 'fa_rms_rope_swiglu_rmt_gqa_stab':
        from .base_model_fa_rms_rope_swiglu_rmt_gqa_stab import GPT, GPTConfig
    elif arch == 'fa_rms_rope_swiglu_rmt_gqa':
        from .base_model_fa_rms_rope_swiglu_rmt_gqa import GPT, GPTConfig
    elif arch == 'fa_rms_rope_swiglu_rmt':
        from .base_model_fa_rms_rope_swiglu_rmt import GPT, GPTConfig
    elif arch == 'fa_rms_rope':
        from .base_model_fa_rms_rope import GPT, GPTConfig
    elif arch == 'fa':
        from .base_model_fa import GPT, GPTConfig
    else:
        from .base_model import GPT, GPTConfig

    gpt_config = GPTConfig(**config)
    model = GPT(gpt_config)
    return model, arch


def load_model_from_files(model_path: str, config_path: str):
    """Load a trained model by auto-selecting the correct variant.

    Returns (model, arch).
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    model, arch = make_model_from_config(config)

    weights = mx.load(model_path)
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    return model, arch
