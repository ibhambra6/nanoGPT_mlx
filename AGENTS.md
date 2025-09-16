# AGENTS.md — Coding Agent Guidelines

This repository implements a family of GPT-style models using Apple’s MLX, with small-data training (primarily Shakespeare) and modern architectural variants (Flash Attention, RMSNorm, RoPE, SwiGLU, RMT, GQA, stabilization, long‑context features, and an MoE‑lite option). This document defines how coding agents should work within this repo: what to change, how to run and test, and what guardrails to follow.

## Project Overview

- Goal: Keep a simple GPT training/eval loop while incrementally aligning the architecture with state‑of‑the‑art features, optimized for Apple Silicon via MLX. Focus on small‑data stability and clean ablations across variants.
- Key directories:
  - `models/`: Model variants. Each variant exposes `GPTConfig` and `GPT`, and is documented in `models/README.md`.
  - `configs/`: Training presets (Shakespeare, OWT, and advanced variants such as GQA, stabilization, long‑context, and MoE‑lite).
  - `scripts/`: Evaluation, benchmarking, and utilities (`eval_ppl_mlx.py`, `eval_instruct_mlx.py`, `benchmark_report.py`, `run_eval.py`).
  - `eval/`: Evaluation datasets and helpers (`create_eval_sets.py`, held‑out Shakespeare eval).
  - `reports/`: Benchmark CSV (`reports/bench.csv`) and related artifacts.
  - `train.py`: Main training loop (uses `configurator.py` to apply config file and CLI overrides).
- Model selection factory: `models/__init__.py` inspects a saved config and instantiates the correct variant automatically. When adding new variants, update the factory and `models/README.md`.
- Outputs: Training saves weights (`.npz`) and config (`.json`) in the run’s `out_dir`.

## Build and Test Commands

Environment setup
- Conda (recommended):
  - `conda env create -f environment.yaml`
  - `conda activate apple_mlx`
- Python venv (alternative):
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r <(awk '/^- pip:/{flag=1;next}/^[^-]/{flag=0}flag{print substr($0,9)}' environment.yaml)`
  - If the above is inconvenient, install essentials: `pip install mlx tiktoken tensorboard datasets pandas`

Prepare data (Shakespeare)
- Shakespeare data is already checked in under `data/shakespeare/` (including `train.bin` and `val.bin`). If missing:
  - `python data/shakespeare/prepare.py`
- Prepare evaluation sets (held‑out):
  - `python eval/create_eval_sets.py`

Train (examples)
- Baseline Shakespeare:
  - `python train.py configs/train_gpt2_shakespeare.py out_dir=trained_models/shakespeare_baseline`
- RMT + FA + RMSNorm + RoPE + SwiGLU + GQA:
  - `python train.py configs/train_gpt2_shakespeare_fa_rms_rope_swiglu_rmt_gqa.py out_dir=trained_models/gqa`
- + Stabilization:
  - `python train.py configs/train_gpt2_shakespeare_fa_rms_rope_swiglu_rmt_gqa_stab.py out_dir=trained_models/gqa_stab`
- + Long‑context:
  - `python train.py configs/train_gpt2_shakespeare_fa_rms_rope_swiglu_rmt_gqa_stab_long.py out_dir=trained_models/gqa_stab_long`
- + MoE‑lite:
  - `python train.py configs/train_gpt2_shakespeare_fa_rms_rope_swiglu_rmt_gqa_stab_long_moe.py out_dir=trained_models/gqa_stab_long_moe`

Sampling
- From latest run directory:
  - `python sample.py init_from=resume out_dir=trained_models/gqa_stab_long start="To be or not to be" max_new_tokens=128 temperature=0.8 top_k=50`

Perplexity evaluation
- Direct:
  - `python scripts/eval_ppl_mlx.py --model_dir trained_models/gqa_stab_long --ctx 512 --eval_data eval/eval_text/shakespeare_eval.bin`
- Or via wrapper:
  - `python scripts/run_eval.py ppl --model_dir trained_models/gqa_stab_long --ctx 512 --quick --quick_batches 50`

Benchmark suite and history
- Full benchmark (logs to `reports/bench.csv`):
  - `python scripts/benchmark_report.py --run_full_benchmark --model_dir trained_models/gqa_stab_long --model_name gqa_stab_long`
- Quick (PPL only):
  - `python scripts/run_eval.py benchmark --model_dir trained_models/gqa_stab_long --model_name gqa_stab_long --quick --quick_batches 50`
- Show history:
  - `python scripts/run_eval.py compare`

Notes
- `train.py` uses `configurator.py` to apply a config file and `--key=value` overrides. Treat config files as trusted code (see Security).
- Training logs TensorBoard to `<out_dir>/tboard_log`.

## Code Style Guidelines

General
- Follow PEP 8 style and clear, concise docstrings. Prefer explicit names over one‑letter variables, especially for tensors: use `(B, T, C)`, `(B, H, T, D)` naming consistently.
- Keep changes surgical and feature‑scoped. Avoid broad refactors unless necessary.
- Update documentation when adding features: `models/README.md` and (if applicable) config files under `configs/`.

Model architecture patterns
- Keep `GPTConfig` as a `dataclass` and expose new features behind clearly named flags with safe defaults. Validate shapes and divisibility (e.g., `n_head % n_kv_head == 0`).
- Maintain the factory contract in `models/__init__.py` (update `select_arch_from_config` and imports in `make_model_from_config`).
- Use MLX fast attention where possible: `mlx.core.fast.scaled_dot_product_attention` with additive masks (0 allowed, `-inf` masked). Keep RoPE applied to Q/K only.
- For long‑context features, construct masks additively and keep KV cache handling explicit. Respect cache offsets for positional encodings.
- For RMT components, maintain consistent tensor layouts and matrix read/write paths. Keep `MatrixRMSNorm` semantics unchanged.

Performance and clarity
- Favor bias‑free linears with RMSNorm in modern variants unless compatibility requires otherwise.
- Keep default sampling simple; advanced sampling should be optional and behind flags.
- Do not introduce heavyweight dependencies. Align with packages in `environment.yaml`.

Documentation and examples
- When adding a new variant, include: (1) README section, (2) a config file in `configs/`, and (3) a one‑line “When to use” addition in the README’s selection guide.

## Testing Instructions

Sanity and environment checks
- Run repository evaluation tests:
  - `python scripts/test_eval_system.py`
  - If datasets are missing, create them: `python eval/create_eval_sets.py`

Quick functional checks for new changes
- Minimal forward pass: Add a tiny config (e.g., 2 layers, 2 heads, 64 dim) and verify a forward call and `generate()` run without errors.
- Factory selection: Save a config JSON with the new flags and ensure `models/__init__.py` selects the intended variant.
- Perplexity smoke test: `python scripts/eval_ppl_mlx.py --model_dir <run_dir> --ctx 256 --max_batches 5`

Benchmarking
- Use `scripts/benchmark_report.py` or `scripts/run_eval.py benchmark` to log results into `reports/bench.csv`. Do not edit the CSV manually—use the script’s management options to delete or view rows.

Reproducibility
- Prefer fixed seeds in sampling and keep training/eval commands recorded in commits or notes.

## Security Considerations

Config execution
- `train.py` loads configs using `exec` via `configurator.py`. Only pass trusted, local config files. Do not execute untrusted configs or arbitrary code from external sources.

File I/O boundaries
- Write only to designated output directories (`out_dir`, `reports/`). Avoid writing outside the repository unless explicitly required.
- Model loading uses `.npz` weights and `.json` configs (no pickle). Keep paths explicit and validated.

Subprocess usage
- Benchmark scripts call `git rev-parse` and run internal Python scripts. Avoid adding subprocess calls that fetch network resources or execute untrusted commands.

Networking and data
- The project should run offline. Do not add code that downloads remote data or models at runtime.
- Treat any future credentials or tokens as secrets—never commit them. No secrets are expected for normal operation.

Dependencies
- Keep dependencies aligned with `environment.yaml`. Avoid heavy or unsafe packages. Stick to MLX, tiktoken, tensorboard, and small utilities already present.

Privacy
- Do not log or store sensitive data. Example prompts and evaluation sets are public Shakespeare material.

---

If you add a new model variant:
1) Add a config under `configs/` with clear comments.
2) Update `models/__init__.py` factory rules and imports.
3) Document the variant in `models/README.md` and add a “When to use” bullet.
4) Provide minimal commands in this file’s Build and Test sections if usage differs.
5) Log benchmark results with `scripts/benchmark_report.py` to keep `reports/bench.csv` up to date.

