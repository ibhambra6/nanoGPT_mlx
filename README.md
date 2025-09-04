# `nanoGPT_mlx`

A port of Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) in Apple's new machine learning framework, [MLX](https://github.com/ml-explore/mlx).

Train OpenAI's GPT-2 models or custom GPT-style models from scratch, all on your Mac's GPU!

Still under active development, but currently the file `train.py` closely resembles the nanoGPT codebase.

## project purpose

This repository exists to systematically compare architectural improvements to GPT-style models by training multiple variants and benchmarking them side‑by‑side. The focus is on modifying the model architecture (e.g., Flash Attention, RMSNorm, RoPE, SwiGLU, Residual Matrix Transformer, etc.) and observing the downstream effects on perplexity, instruction following, and inference speed on the same evaluation set (Shakespeare).

The codebase includes a small, opinionated evaluation/benchmark harness to make A/B comparisons fast and repeatable on a single machine. Configs under `configs/` make it easy to toggle architectures and hyperparameters for consistent experiments.

## hardware & framework

All testing and training in this project has been performed locally on an Apple MacBook Air with an M2 chip and 8 GB of unified memory, using Apple’s MLX framework. The implementation choices, default configs, and the provided evaluation loops are tuned for this environment (e.g., moderate context lengths, modest batch sizes, flash-attention kernels). Results will scale with stronger hardware, but the repo is optimized for lightweight, reproducible experiments on Mac laptops.

## long-term aim

The end goal is to assemble a compact, modern GPT architecture—borrowing ingredients from the latest OpenAI models (e.g., attention optimizations, improved normalization, rotary/relative positions, modern MLP activations)—but at a much smaller scale suitable for local training. The intent is not to match the capabilities or scale of proprietary frontier models, but to replicate the architectural “shape” and training ergonomics in a small, open, and fast-to-iterate package that runs entirely on a laptop with limited iterations.

## install

Create a conda environment using the provided
   environment configuration file.

```bash
conda env create -f environment.yaml
```

Activate conda environment.
```bash
conda activate apple_mlx
```

Dependencies:
- [mlx](https://ml-explore.github.io/mlx/build/html/index.html)
- [numpy](https://numpy.org/install/)
-  `datasets` for huggingface datasets (if you want to download + preprocess OpenWebText)
-  `tiktoken` for OpenAI's fast BPE code
-  `tensorboardX` for optional logging
-  `tqdm` for progress bars

## quick start
To train a character-level GPT, prepare shakespeare dataset similar to nanoGPT. This will create a `train.bin` and `val.bin` in that data directory.
```bash
python data/shakespeare/prepare.py
```

Now, let's train a "baby GPT" model on your MAC GPU:
```bash
python train.py configs/train_gpt2_shakespeare.py
```

On my Macbook M3 Pro, I am observing `~0.37 iterations/second` when training a `~45M parameter` GPT-2 model, at `batch_size=64` (i.e., `local_batch_size=4` and `gradient_accumulation=16`).

![repro124m](assets/baby_gpt2_shakespeare_pretrain_loss.png)

So once the training finishes we can sample from the best model by pointing the sampling script at this directory:
```bash
python sample.py --out_dir=gpt2_small_shakespeare
```

## openwebtext
To train a GPT-2 model on OpenWebText similar to nanoGPT, first prepare the dataset:
```bash
python data/openwebtext/prepare.py
```

Then, train a 124M GPT-2 model on your MAC GPU:
```bash
python train.py configs/train_gpt2_owt.py
```

## benchmarking & evaluation

This project includes a comprehensive benchmarking system to evaluate and compare different models you train. The system evaluates models on perplexity, instruction following, and inference speed.

### Quick Start

```bash
# Create evaluation datasets (run once)
python eval/create_eval_sets.py

# Run full benchmark on a trained model
python scripts/run_eval.py benchmark --model_dir out/ --model_name baseline

# View benchmark history
python scripts/run_eval.py compare
```

### Evaluation Components

The benchmarking system includes three main evaluation components:

1. **Perplexity Evaluation** - Measures how well the model predicts held-out text
2. **Instruction Following** - Tests ability to follow Shakespeare-style instructions  
3. **Inference Speed** - Measures tokens generated per second

### Individual Evaluations

Run individual evaluations with these commands:

```bash
# Perplexity evaluation only
python scripts/run_eval.py ppl --model_dir out/ --ctx 512

# Instruction following evaluation only  
python scripts/run_eval.py instruct --model_dir out/ --max_examples 20

# Full benchmark suite
python scripts/run_eval.py benchmark --model_dir out/ --model_name my_model
```

### Benchmark Results

All benchmark results are logged to `reports/bench.csv` with the following metrics:

- **Perplexity** - Lower is better (measures prediction accuracy)
- **Instruction Quality** - 0-1 scale, higher is better (measures instruction following)
- **Tokens/sec** - Higher is better (measures inference speed)
- **Model size** - Parameters in millions
- **Context lengths** - Training and evaluation context sizes

### Example Workflow

1. Train a baseline model:
   ```bash
   python train.py configs/train_gpt2_shakespeare.py
   ```

2. Benchmark the baseline:
   ```bash
   python scripts/run_eval.py benchmark --model_dir gpt2_shakespeare_pretrain --model_name baseline
   ```

3. Make improvements (e.g., add RoPE, change architecture)

4. Train the improved model and benchmark again:
   ```bash
   python scripts/run_eval.py benchmark --model_dir gpt2_shakespeare_improved --model_name with_rope
   ```

5. Compare results:
   ```bash
   python scripts/run_eval.py compare
   ```

### Evaluation Datasets

The system uses two evaluation datasets:

- **Shakespeare Evaluation Text** (`eval/eval_text/`) - 34k tokens of held-out Shakespeare text for perplexity measurement
- **Instruction Dataset** (`eval/eval_instruct.jsonl`) - 20 Shakespeare-style instruction-following tasks

### Advanced Usage

For more control, use the individual evaluation scripts directly:

```bash
# Detailed perplexity evaluation with custom parameters
python scripts/eval_ppl_mlx.py --model_dir out/ --ctx 1024 --output_file results.json

# Instruction evaluation with custom generation settings
python scripts/eval_instruct_mlx.py --model_dir out/ --temperature 0.7 --max_new_tokens 200

# Custom benchmark with specific settings
python scripts/benchmark_report.py --model_dir out/ --model_name experiment_1 --notes "Testing new optimizer"
```

See individual script help for all options:
```bash
python scripts/eval_ppl_mlx.py --help
python scripts/eval_instruct_mlx.py --help  
python scripts/benchmark_report.py --help
```

## todos
- [ ] disable weight decay on non-decay params in optimizer
- [ ] add bfloat16 training support
- [x] integrate evaluation harness (comprehensive benchmarking system)
- [ ] add checkpoint conversion for loading pre-trained HF models 
- [x] add saveing and loading pre-trained MLX models 
- [ ] enable finetuning models from pre-trained checkpoints
- [x] enable inference with pre-trained models

## acknowledgements
Thank you [Andrej Karpthy](https://github.com/karpathy) for creating the nanoGPT codebase. It's been awesome for quick prototyping!
