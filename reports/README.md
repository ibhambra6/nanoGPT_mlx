# Benchmark Reports

This directory contains benchmark results and evaluation reports for nanoGPT_mlx models.

## Files

- `bench.csv` - Main benchmark results CSV file with metrics for all evaluated models
- Individual evaluation result JSON files (if saved during evaluation)

## Benchmark CSV Format

The `bench.csv` file contains the following columns:

| Column | Description |
|--------|-------------|
| `timestamp` | ISO timestamp of when the benchmark was run |
| `commit` | Git commit hash when the model was evaluated |
| `model_name` | Identifier for the model (e.g., "baseline", "with_rope") |
| `params_m` | Model size in millions of parameters |
| `ctx_train` | Context length used during training |
| `ctx_eval` | Context length used for evaluation |
| `ppl_eval` | Perplexity score on evaluation set (lower is better) |
| `instruct_quality` | Instruction following quality score 0-1 (higher is better) |
| `tok_per_sec` | Inference speed in tokens per second |
| `eval_time_sec` | Time taken for perplexity evaluation |
| `model_path` | Path to the model files |
| `notes` | Additional notes about the benchmark run |

## Running Benchmarks

See the main README.md for instructions on how to run benchmarks and evaluations.

## Analyzing Results

You can analyze benchmark results by:

1. Opening `bench.csv` in a spreadsheet application
2. Using the `compare` command in the evaluation scripts
3. Writing custom analysis scripts that read the CSV data

Example Python code to read benchmark history:

```python
import pandas as pd

# Load benchmark results
df = pd.read_csv('reports/bench.csv')

# Show latest results
print(df.tail())

# Compare models by perplexity
print(df.groupby('model_name')['ppl_eval'].min().sort_values())

# Plot performance over time
import matplotlib.pyplot as plt
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.plot(x='timestamp', y='ppl_eval', kind='line')
plt.show()
```
