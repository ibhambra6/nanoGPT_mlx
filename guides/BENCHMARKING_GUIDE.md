# nanoGPT_mlx Benchmarking Guide

This guide provides detailed information about the comprehensive benchmarking system implemented for nanoGPT_mlx, based on the plan in `docs/plan.md` modules 0 and 1.

## Overview

The benchmarking system allows you to:
- Compare different model architectures and configurations
- Track performance improvements over time
- Evaluate models on multiple metrics (perplexity, instruction following, speed)
- Maintain a historical record of all experiments

## System Components

### 1. Evaluation Datasets

- **Shakespeare Evaluation Text** (`eval/eval_text/shakespeare_eval.bin`): 34,279 tokens of held-out Shakespeare text for perplexity evaluation
- **Instruction Dataset** (`eval/eval_instruct.jsonl`): 20 Shakespeare-style instruction-following tasks

### 2. Evaluation Scripts

- **`scripts/eval_ppl_mlx.py`**: Perplexity evaluation on held-out text
- **`scripts/eval_instruct_mlx.py`**: Instruction following quality assessment
- **`scripts/benchmark_report.py`**: Comprehensive benchmarking and metrics logging
- **`scripts/run_eval.py`**: Convenient wrapper for all evaluation tasks

### 3. Results Storage

- **`reports/bench.csv`**: Main benchmark results database
- Individual evaluation result JSON files (optional)

## Quick Start

1. **Setup** (run once):
   ```bash
   python eval/create_eval_sets.py
   ```

2. **Train a model**:
   ```bash
   python train.py configs/train_gpt2_shakespeare.py
   ```

3. **Benchmark the model**:
   ```bash
   python scripts/run_eval.py benchmark --model_dir gpt2_shakespeare_pretrain --model_name baseline
   ```

4. **View results**:
   ```bash
   python scripts/run_eval.py compare
   ```

## Metrics Explained

### Perplexity
- **What it measures**: How well the model predicts the next token
- **Range**: 1+ (lower is better)
- **Good values**: <50 for Shakespeare models
- **Context**: Evaluated on held-out Shakespeare text

### Instruction Quality
- **What it measures**: How well the model follows instructions
- **Range**: 0-1 (higher is better)
- **Evaluation method**: Heuristic-based scoring of generated responses
- **Context**: Shakespeare-style instruction tasks

### Inference Speed
- **What it measures**: Generation speed in tokens per second
- **Range**: Depends on hardware (higher is better)
- **Context**: Measured during text generation with temperature=1.0

## Advanced Usage

### Individual Evaluations

```bash
# Perplexity only
python scripts/eval_ppl_mlx.py --model_dir out/ --ctx 1024

# Instruction following only
python scripts/eval_instruct_mlx.py --model_dir out/ --max_examples 20

# Speed measurement only  
python scripts/benchmark_report.py --model_dir out/ --model_name test --speed_only
```

### Custom Parameters

```bash
# Perplexity with custom context length
python scripts/run_eval.py ppl --model_dir out/ --ctx 1024

# Instruction evaluation with custom generation settings
python scripts/run_eval.py instruct --model_dir out/ --temperature 0.7 --max_new_tokens 200

# Benchmark with notes
python scripts/run_eval.py benchmark --model_dir out/ --model_name experiment_1 --notes "Testing new architecture"
```

### Working with Results

The benchmark results are stored in `reports/bench.csv` and can be analyzed with any tool that reads CSV:

```python
import pandas as pd

# Load results
df = pd.read_csv('reports/bench.csv')

# Show best models by perplexity
print(df.nsmallest(5, 'ppl_eval')[['model_name', 'ppl_eval', 'instruct_quality']])

# Compare model sizes vs performance
import matplotlib.pyplot as plt
plt.scatter(df['params_m'], df['ppl_eval'])
plt.xlabel('Parameters (millions)')
plt.ylabel('Perplexity')
plt.show()
```

## Benchmark Protocol

Following the plan in `docs/plan.md`, the recommended benchmarking protocol is:

1. **Baseline**: Train and benchmark a stock nanoGPT model
2. **Incremental Changes**: Apply one architectural change at a time
3. **Re-benchmark**: Evaluate each change against the baseline
4. **Track Progress**: Use the CSV to track improvements over time

Example workflow:
```bash
# Baseline
python train.py configs/train_gpt2_shakespeare.py
python scripts/run_eval.py benchmark --model_dir gpt2_shakespeare_pretrain --model_name baseline

# Add RMSNorm + RoPE
# ... make changes to model.py ...
python train.py configs/train_gpt2_shakespeare.py  # retrain
python scripts/run_eval.py benchmark --model_dir gpt2_shakespeare_pretrain --model_name rmsnorm_rope

# Add SwiGLU
# ... make more changes ...
python scripts/run_eval.py benchmark --model_dir gpt2_shakespeare_pretrain --model_name rmsnorm_rope_swiglu

# Compare results
python scripts/run_eval.py compare
```

## Troubleshooting

### Common Issues

1. **"Model file not found"**: Ensure your model directory contains both `.npz` and `.json` files
2. **"Evaluation data not found"**: Run `python eval/create_eval_sets.py` first
3. **Import errors**: Make sure you're running from the project root directory

### Validation

Run the test suite to validate your setup:
```bash
python scripts/test_eval_system.py
```

## Extending the System

### Adding New Metrics

To add new evaluation metrics:

1. Add the metric to `benchmark_report.py` in the `run_full_benchmark` method
2. Update the CSV headers in `_init_benchmark_csv`
3. Implement the evaluation logic

### Adding New Datasets

To add new evaluation datasets:

1. Create the dataset in the `eval/` directory
2. Update `create_eval_sets.py` to include it
3. Modify the evaluation scripts to use the new data

### Custom Evaluation Logic

For domain-specific evaluation:

1. Copy `eval_instruct_mlx.py` as a template
2. Implement your custom evaluation logic
3. Integrate with `benchmark_report.py` or use as standalone

## Performance Expectations

On an M2 Mac with 8GB RAM, expect:
- **Perplexity evaluation**: ~30 seconds for 50 batches
- **Instruction evaluation**: ~2-3 minutes for 20 examples  
- **Speed measurement**: ~10 seconds
- **Full benchmark**: ~5 minutes total

Evaluation times scale with model size and context length.
