#!/usr/bin/env python3
"""
Convenience script to run evaluations and benchmarks.
This is a wrapper around the individual evaluation scripts.

Usage:
    python scripts/run_eval.py --help
    python scripts/run_eval.py ppl --model_dir out/
    python scripts/run_eval.py instruct --model_dir out/
    python scripts/run_eval.py benchmark --model_dir out/ --model_name baseline
    python scripts/run_eval.py compare  # Show benchmark history
"""

import os
import sys
import subprocess
import argparse
from typing import List, Optional

def run_command(cmd: List[str], description: str = "") -> int:
    """Run a subprocess command and return exit code."""
    if description:
        print(f"Running: {description}")
    
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"Error running command: {e}")
        return 1

def run_perplexity_eval(args):
    """Run perplexity evaluation."""
    cmd = ["python", "scripts/eval_ppl_mlx.py"]
    
    if args.model_dir:
        cmd.extend(["--model_dir", args.model_dir])
    elif args.model_paths:
        cmd.extend(["--model_paths"] + args.model_paths)
    
    if args.ctx:
        cmd.extend(["--ctx", str(args.ctx)])
    
    if args.eval_data:
        cmd.extend(["--eval_data", args.eval_data])
    
    if args.output_file:
        cmd.extend(["--output_file", args.output_file])
    
    return run_command(cmd, "Perplexity evaluation")

def run_instruction_eval(args):
    """Run instruction following evaluation."""
    cmd = ["python", "scripts/eval_instruct_mlx.py"]
    
    if args.model_dir:
        cmd.extend(["--model_dir", args.model_dir])
    elif args.model_paths:
        cmd.extend(["--model_paths"] + args.model_paths)
    
    if args.max_examples:
        cmd.extend(["--max_examples", str(args.max_examples)])
    
    if args.eval_data:
        cmd.extend(["--eval_data", args.eval_data])
    
    if args.output_file:
        cmd.extend(["--output_file", args.output_file])
    
    # Generation parameters
    if args.temperature:
        cmd.extend(["--temperature", str(args.temperature)])
    
    if args.max_new_tokens:
        cmd.extend(["--max_new_tokens", str(args.max_new_tokens)])
    
    return run_command(cmd, "Instruction following evaluation")

def run_benchmark(args):
    """Run full benchmark suite."""
    cmd = ["python", "scripts/benchmark_report.py"]
    
    if args.model_dir:
        cmd.extend(["--model_dir", args.model_dir])
    elif args.model_paths:
        cmd.extend(["--model_paths"] + args.model_paths)
    
    cmd.extend(["--model_name", args.model_name])
    
    if args.commit_hash:
        cmd.extend(["--commit_hash", args.commit_hash])
    
    if args.notes:
        cmd.extend(["--notes", args.notes])
    
    if args.ctx:
        cmd.extend(["--ctx_eval", str(args.ctx)])
    
    if args.run_full:
        cmd.append("--run_full_benchmark")
    elif args.ppl_only:
        cmd.append("--ppl_only")
    elif args.instruct_only:
        cmd.append("--instruct_only")
    elif args.speed_only:
        cmd.append("--speed_only")
    else:
        cmd.append("--run_full_benchmark")  # Default to full benchmark
    
    return run_command(cmd, "Benchmark suite")

def show_benchmark_history(args):
    """Show benchmark history."""
    cmd = ["python", "scripts/benchmark_report.py", "--show_history"]
    return run_command(cmd, "Benchmark history")

def main():
    parser = argparse.ArgumentParser(description="Run evaluations and benchmarks")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Common arguments for model loading
    def add_model_args(p):
        model_group = p.add_mutually_exclusive_group(required=True)
        model_group.add_argument('--model_dir', type=str,
                                help='Directory containing model files')
        model_group.add_argument('--model_paths', nargs=2,
                                help='Model and config file paths')
    
    # Perplexity evaluation command
    ppl_parser = subparsers.add_parser('ppl', help='Run perplexity evaluation')
    add_model_args(ppl_parser)
    ppl_parser.add_argument('--ctx', type=int, default=512,
                           help='Context length for evaluation')
    ppl_parser.add_argument('--eval_data', type=str,
                           help='Path to evaluation data')
    ppl_parser.add_argument('--output_file', type=str,
                           help='Output file for results')
    
    # Instruction evaluation command
    instruct_parser = subparsers.add_parser('instruct', help='Run instruction evaluation')
    add_model_args(instruct_parser)
    instruct_parser.add_argument('--max_examples', type=int, default=20,
                                help='Maximum examples to evaluate')
    instruct_parser.add_argument('--eval_data', type=str,
                                help='Path to instruction evaluation data')
    instruct_parser.add_argument('--output_file', type=str,
                                help='Output file for results')
    instruct_parser.add_argument('--temperature', type=float, default=0.8,
                                help='Generation temperature')
    instruct_parser.add_argument('--max_new_tokens', type=int, default=256,
                                help='Maximum tokens to generate')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmark suite')
    add_model_args(benchmark_parser)
    benchmark_parser.add_argument('--model_name', type=str, required=True,
                                 help='Name for this model')
    benchmark_parser.add_argument('--commit_hash', type=str,
                                 help='Git commit hash')
    benchmark_parser.add_argument('--notes', type=str, default='',
                                 help='Notes about this run')
    benchmark_parser.add_argument('--ctx', type=int, default=512,
                                 help='Context length for evaluation')
    
    # Benchmark type selection
    bench_group = benchmark_parser.add_mutually_exclusive_group()
    bench_group.add_argument('--run_full', action='store_true',
                            help='Run full benchmark (default)')
    bench_group.add_argument('--ppl_only', action='store_true',
                            help='Run only perplexity')
    bench_group.add_argument('--instruct_only', action='store_true',
                            help='Run only instruction evaluation')
    bench_group.add_argument('--speed_only', action='store_true',
                            help='Run only speed measurement')
    
    # Compare/history command
    compare_parser = subparsers.add_parser('compare', help='Show benchmark history')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    os.chdir(project_dir)
    
    # Run the appropriate command
    if args.command == 'ppl':
        return run_perplexity_eval(args)
    elif args.command == 'instruct':
        return run_instruction_eval(args)
    elif args.command == 'benchmark':
        return run_benchmark(args)
    elif args.command == 'compare':
        return show_benchmark_history(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
