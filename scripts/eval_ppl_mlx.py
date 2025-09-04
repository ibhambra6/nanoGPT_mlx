#!/usr/bin/env python3
"""
Perplexity evaluation script for MLX models.
Based on the plan in docs/plan.md - Module 1: Eval & Benchmark Harness.

Usage:
    python scripts/eval_ppl_mlx.py --model_path out/model.npz --config_path out/model.json
    python scripts/eval_ppl_mlx.py --model_dir out/  # Auto-detects model files
"""

import os
import math
import json
import time
import argparse
from typing import Tuple, Optional, Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import tiktoken

# Add parent directory to path to import model
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import load_model_from_files


def load_model(model_path: str, config_path: str) -> Any:
    """Load a model from checkpoint files.
    
    Args:
        model_path: Path to model weights (.npz file)
        config_path: Path to config file (.json file)
    
    Returns:
        Loaded model instance
    """
    model, arch = load_model_from_files(model_path, config_path)
    return model


class EvalLoader:
    """Data loader for evaluation sets."""
    
    def __init__(self, data_path: str, tokenizer_name: str = "gpt2"):
        """Initialize evaluation data loader.
        
        Args:
            data_path: Path to evaluation data (.bin or .txt file)
            tokenizer_name: Name of tokenizer to use
        """
        self.tokenizer_name = tokenizer_name
        self.enc = tiktoken.encoding_for_model(tokenizer_name)
        
        if data_path.endswith('.bin'):
            # Load pre-tokenized binary data
            self.tokens = np.memmap(data_path, dtype=np.uint16, mode='r')
        elif data_path.endswith('.txt'):
            # Load and tokenize text data
            with open(data_path, 'r', encoding='utf-8') as f:
                text = f.read()
            self.tokens = np.array(self.enc.encode_ordinary(text), dtype=np.uint16)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        print(f"Loaded evaluation data with {len(self.tokens):,} tokens")
    
    def iter(self, ctx: int, batch_size: int = 1) -> mx.array:
        """Iterate over evaluation data in chunks.
        
        Args:
            ctx: Context length for evaluation
            batch_size: Batch size (usually 1 for evaluation)
            
        Yields:
            Batched input sequences of shape [batch_size, ctx]
        """
        num_batches = (len(self.tokens) - 1) // ctx
        
        for i in range(0, num_batches * ctx, ctx):
            # Extract sequence of length ctx
            if i + ctx >= len(self.tokens):
                break
                
            batch_data = []
            for b in range(batch_size):
                start_idx = i + b * ctx
                if start_idx + ctx < len(self.tokens):
                    seq = self.tokens[start_idx:start_idx + ctx]
                    batch_data.append(seq)
            
            if len(batch_data) == batch_size:
                yield mx.array(np.stack(batch_data), dtype=mx.int64)


def cross_entropy(logits: mx.array, targets: mx.array) -> mx.array:
    """Compute cross-entropy loss.
    
    Args:
        logits: Model output logits [B, T, V]
        targets: Target token IDs [B, T]
        
    Returns:
        Cross-entropy loss scalar
    """
    # Reshape for cross-entropy computation
    logits_flat = logits.reshape(-1, logits.shape[-1])  # [B*T, V]
    targets_flat = targets.reshape(-1)  # [B*T]
    
    # Compute log probabilities
    log_probs = nn.log_softmax(logits_flat, axis=-1)
    
    # Gather log probabilities for target tokens
    target_indices = mx.expand_dims(targets_flat, -1)
    gathered_log_probs = mx.take_along_axis(log_probs, target_indices, axis=-1)
    gathered_log_probs = mx.squeeze(gathered_log_probs, axis=-1)
    
    # Return negative log likelihood
    return -mx.mean(gathered_log_probs)


def eval_ppl(model: Any, loader: EvalLoader, ctx: int, max_batches: Optional[int] = None) -> float:
    """Evaluate perplexity on a dataset.
    
    Args:
        model: GPT model to evaluate
        loader: Data loader for evaluation set
        ctx: Context length for evaluation
        max_batches: Maximum number of batches to evaluate (None for all)
        
    Returns:
        Perplexity score
    """
    model.eval()  # Set model to evaluation mode
    
    total_nll = 0.0
    total_tokens = 0
    num_batches = 0
    
    print(f"Evaluating perplexity with context length {ctx}...")
    
    for batch_idx, x in enumerate(loader.iter(ctx=ctx, batch_size=1)):
        if max_batches is not None and batch_idx >= max_batches:
            break
            
        # Forward pass
        logits = model(x)  # [B, T, V]
        
        # Compute loss on all tokens except the first (no previous context)
        logits_pred = logits[:, :-1, :]  # [B, T-1, V]
        targets = x[:, 1:]  # [B, T-1]
        
        # Compute cross-entropy loss
        loss = cross_entropy(logits_pred, targets)
        
        # Accumulate statistics
        num_tokens = targets.size
        total_nll += float(loss) * num_tokens
        total_tokens += num_tokens
        num_batches += 1
        
        if (batch_idx + 1) % 10 == 0:
            current_ppl = math.exp(total_nll / total_tokens)
            print(f"  Batch {batch_idx + 1}: current perplexity = {current_ppl:.3f}")
    
    # Compute final perplexity
    if total_tokens == 0:
        return float('inf')
    
    avg_nll = total_nll / total_tokens
    perplexity = math.exp(avg_nll)
    
    print(f"Evaluated {num_batches} batches with {total_tokens:,} total tokens")
    return perplexity


 


def auto_detect_model_files(model_dir: str) -> Tuple[str, str]:
    """Auto-detect model and config files in a directory.
    
    Args:
        model_dir: Directory containing model files
        
    Returns:
        Tuple of (model_path, config_path)
    """
    # Look for .npz and .json files
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.npz')]
    config_files = [f for f in os.listdir(model_dir) if f.endswith('.json')]
    
    if not model_files:
        raise FileNotFoundError(f"No .npz model files found in {model_dir}")
    if not config_files:
        raise FileNotFoundError(f"No .json config files found in {model_dir}")
    
    # Use the first found files, or try to match names
    model_path = os.path.join(model_dir, model_files[0])
    config_path = os.path.join(model_dir, config_files[0])
    
    # Try to find matching names
    for model_file in model_files:
        base_name = os.path.splitext(model_file)[0]
        matching_config = base_name + '.json'
        if matching_config in config_files:
            model_path = os.path.join(model_dir, model_file)
            config_path = os.path.join(model_dir, matching_config)
            break
    
    return model_path, config_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate perplexity of MLX models")
    
    # Model loading arguments
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--model_dir', type=str,
                            help='Directory containing model files (auto-detects .npz and .json)')
    model_group.add_argument('--model_paths', nargs=2, metavar=('MODEL_PATH', 'CONFIG_PATH'),
                            help='Explicit paths to model.npz and config.json files')
    
    # Evaluation arguments
    parser.add_argument('--eval_data', type=str, 
                       default='eval/eval_text/shakespeare_eval.bin',
                       help='Path to evaluation data (.bin or .txt)')
    parser.add_argument('--ctx', type=int, default=512,
                       help='Context length for evaluation')
    parser.add_argument('--max_batches', type=int, default=None,
                       help='Maximum number of batches to evaluate (None for all)')
    parser.add_argument('--tokenizer', type=str, default='gpt2',
                       help='Tokenizer to use')
    
    # Output arguments
    parser.add_argument('--output_file', type=str, default=None,
                       help='File to save results (JSON format)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Determine model and config paths
    if args.model_dir:
        model_path, config_path = auto_detect_model_files(args.model_dir)
        print(f"Auto-detected model: {model_path}")
        print(f"Auto-detected config: {config_path}")
    else:
        model_path, config_path = args.model_paths
    
    # Validate files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.exists(args.eval_data):
        raise FileNotFoundError(f"Evaluation data not found: {args.eval_data}")
    
    # Load model
    print("Loading model...")
    model, arch = load_model_from_files(model_path, config_path)
    from mlx.utils import tree_flatten
    nparams = sum(x.size for k, x in tree_flatten(model.parameters()))
    print(f"Loaded {arch} model with {nparams / 1e6:.2f}M parameters")
    
    # Load evaluation data
    print("Loading evaluation data...")
    loader = EvalLoader(args.eval_data, args.tokenizer)
    
    # Run evaluation
    start_time = time.time()
    perplexity = eval_ppl(model, loader, args.ctx, args.max_batches)
    eval_time = time.time() - start_time
    
    # Print results
    print(f"\n{'='*50}")
    print("PERPLEXITY EVALUATION RESULTS")
    print("=" * 50)
    print(f"Model: {model_path}")
    print(f"Evaluation data: {args.eval_data}")
    print(f"Context length: {args.ctx}")
    print(f"Perplexity: {perplexity:.3f}")
    print(f"Evaluation time: {eval_time:.2f} seconds")
    
    # Save results if requested
    if args.output_file:
        results = {
            'model_path': model_path,
            'config_path': config_path,
            'eval_data': args.eval_data,
            'context_length': args.ctx,
            'perplexity': perplexity,
            'evaluation_time': eval_time,
            'timestamp': time.time(),
            'max_batches': args.max_batches
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
