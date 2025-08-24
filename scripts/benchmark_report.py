#!/usr/bin/env python3
"""
Benchmark reporting and metrics logging script.
Coordinates evaluation runs and logs results to reports/bench.csv.

Usage:
    python scripts/benchmark_report.py --model_dir out/ --commit_hash abc123 --model_name baseline
    python scripts/benchmark_report.py --run_full_benchmark --model_dir out/
"""

import os
import csv
import json
import time
import subprocess
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import mlx.core as mx
from mlx.utils import tree_flatten

# Add parent directory to path to import model
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model import GPT, GPTConfig
from scripts.eval_ppl_mlx import load_model, auto_detect_model_files


class BenchmarkReporter:
    """Manages benchmark reporting and metrics logging."""
    
    def __init__(self, reports_dir: str = "reports"):
        """Initialize benchmark reporter.
        
        Args:
            reports_dir: Directory to store benchmark reports
        """
        self.reports_dir = reports_dir
        self.bench_csv_path = os.path.join(reports_dir, "bench.csv")
        
        # Ensure reports directory exists
        os.makedirs(reports_dir, exist_ok=True)
        
        # Initialize CSV file with headers if it doesn't exist
        self._init_benchmark_csv()
    
    def _init_benchmark_csv(self):
        """Initialize the benchmark CSV file with proper headers."""
        if not os.path.exists(self.bench_csv_path):
            headers = [
                'timestamp',
                'commit',
                'model_name',
                'params_m',
                'ctx_train',
                'ctx_eval',
                'ppl_eval',
                'instruct_quality',
                'tok_per_sec',
                'eval_time_sec',
                'model_path',
                'notes'
            ]
            
            with open(self.bench_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            
            print(f"Initialized benchmark CSV: {self.bench_csv_path}")
    
    def get_git_commit_hash(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--short', 'HEAD'],
                capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"
    
    def get_model_info(self, model_path: str, config_path: str) -> Tuple[float, Dict[str, Any]]:
        """Get model parameter count and configuration info.
        
        Args:
            model_path: Path to model weights
            config_path: Path to model config
            
        Returns:
            Tuple of (param_count_millions, config_dict)
        """
        # Load model to count parameters
        model = load_model(model_path, config_path)
        
        # Count parameters
        nparams = sum(x.size for k, x in tree_flatten(model.parameters()))
        params_millions = nparams / 1e6
        
        # Load config
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        return params_millions, config_dict
    
    def run_perplexity_evaluation(self, model_path: str, config_path: str,
                                 eval_data: str = "eval/eval_text/shakespeare_eval.bin",
                                 ctx_eval: int = 512) -> Tuple[float, float]:
        """Run perplexity evaluation.
        
        Args:
            model_path: Path to model weights
            config_path: Path to model config
            eval_data: Path to evaluation data
            ctx_eval: Context length for evaluation
            
        Returns:
            Tuple of (perplexity, evaluation_time)
        """
        print(f"Running perplexity evaluation (ctx={ctx_eval})...")
        
        # Import here to avoid circular imports
        from scripts.eval_ppl_mlx import eval_ppl, EvalLoader
        
        # Load model and data
        model = load_model(model_path, config_path)
        loader = EvalLoader(eval_data)
        
        # Run evaluation
        start_time = time.time()
        perplexity = eval_ppl(model, loader, ctx_eval, max_batches=50)  # Limit for benchmarking
        eval_time = time.time() - start_time
        
        return perplexity, eval_time
    
    def run_instruction_evaluation(self, model_path: str, config_path: str,
                                  eval_data: str = "eval/eval_instruct.jsonl",
                                  max_examples: int = 10) -> float:
        """Run instruction following evaluation.
        
        Args:
            model_path: Path to model weights
            config_path: Path to model config  
            eval_data: Path to instruction evaluation data
            max_examples: Maximum examples to evaluate (for speed)
            
        Returns:
            Average instruction following quality score
        """
        print(f"Running instruction evaluation ({max_examples} examples)...")
        
        # Import here to avoid circular imports
        from scripts.eval_instruct_mlx import evaluate_instructions
        
        # Run evaluation with limited examples for benchmarking
        summary = evaluate_instructions(
            model_path, config_path, eval_data,
            max_examples=max_examples,
            generation_params={'max_new_tokens': 128, 'temperature': 0.8}  # Faster generation
        )
        
        return summary['overall_quality']['mean']
    
    def measure_inference_speed(self, model_path: str, config_path: str,
                               ctx_length: int = 512, num_tokens: int = 100) -> float:
        """Measure inference speed in tokens per second.
        
        Args:
            model_path: Path to model weights
            config_path: Path to model config
            ctx_length: Context length for speed test
            num_tokens: Number of tokens to generate for measurement
            
        Returns:
            Tokens per second
        """
        print(f"Measuring inference speed ({num_tokens} tokens)...")
        
        # Load model
        model = load_model(model_path, config_path)
        
        # Prepare input (random tokens)
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt2")
        
        # Use a simple prompt
        prompt = "To be or not to be, that is the question:"
        prompt_tokens = enc.encode(prompt)
        
        # Pad or trim to desired context length
        if len(prompt_tokens) > ctx_length:
            prompt_tokens = prompt_tokens[:ctx_length]
        else:
            # Pad with random tokens if needed
            while len(prompt_tokens) < min(ctx_length, 100):  # Don't make it too long
                prompt_tokens.append(enc.encode(" the")[0])
        
        input_array = mx.array([prompt_tokens], dtype=mx.int64)
        
        # Warm-up run
        model.eval()
        _ = model.generate(input_array, max_new_tokens=10, temperature=1.0)
        
        # Measure generation speed
        start_time = time.time()
        generated = model.generate(input_array, max_new_tokens=num_tokens, temperature=1.0)
        end_time = time.time()
        
        # Calculate tokens per second
        generation_time = end_time - start_time
        tokens_per_sec = num_tokens / generation_time if generation_time > 0 else 0
        
        return tokens_per_sec
    
    def run_full_benchmark(self, model_path: str, config_path: str,
                          model_name: str, commit_hash: Optional[str] = None,
                          notes: str = "") -> Dict[str, Any]:
        """Run a full benchmark suite on a model.
        
        Args:
            model_path: Path to model weights
            config_path: Path to model config
            model_name: Name/identifier for this model
            commit_hash: Git commit hash (auto-detected if None)
            notes: Additional notes about this benchmark run
            
        Returns:
            Dictionary with all benchmark results
        """
        print(f"Running full benchmark for model: {model_name}")
        print(f"Model path: {model_path}")
        
        # Get model info
        params_millions, config_dict = self.get_model_info(model_path, config_path)
        ctx_train = config_dict.get('block_size', 'unknown')
        
        # Get commit hash
        if commit_hash is None:
            commit_hash = self.get_git_commit_hash()
        
        # Default evaluation context
        ctx_eval = min(512, ctx_train if isinstance(ctx_train, int) else 512)
        
        benchmark_results = {
            'timestamp': datetime.now().isoformat(),
            'commit': commit_hash,
            'model_name': model_name,
            'params_m': params_millions,
            'ctx_train': ctx_train,
            'ctx_eval': ctx_eval,
            'model_path': model_path,
            'notes': notes
        }
        
        try:
            # Run perplexity evaluation
            ppl, eval_time = self.run_perplexity_evaluation(
                model_path, config_path, ctx_eval=ctx_eval
            )
            benchmark_results['ppl_eval'] = ppl
            benchmark_results['eval_time_sec'] = eval_time
            
        except Exception as e:
            print(f"Perplexity evaluation failed: {e}")
            benchmark_results['ppl_eval'] = None
            benchmark_results['eval_time_sec'] = None
        
        try:
            # Run instruction evaluation  
            instruct_quality = self.run_instruction_evaluation(model_path, config_path)
            benchmark_results['instruct_quality'] = instruct_quality
            
        except Exception as e:
            print(f"Instruction evaluation failed: {e}")
            benchmark_results['instruct_quality'] = None
        
        try:
            # Measure inference speed
            tok_per_sec = self.measure_inference_speed(model_path, config_path)
            benchmark_results['tok_per_sec'] = tok_per_sec
            
        except Exception as e:
            print(f"Speed measurement failed: {e}")
            benchmark_results['tok_per_sec'] = None
        
        # Log results to CSV
        self.log_benchmark_result(benchmark_results)
        
        return benchmark_results
    
    def log_benchmark_result(self, results: Dict[str, Any]):
        """Log benchmark results to CSV file.
        
        Args:
            results: Dictionary with benchmark results
        """
        # Prepare row data in correct order
        row_data = [
            results.get('timestamp', ''),
            results.get('commit', ''),
            results.get('model_name', ''),
            f"{results.get('params_m', 0):.2f}" if results.get('params_m') else '',
            results.get('ctx_train', ''),
            results.get('ctx_eval', ''),
            f"{results.get('ppl_eval', 0):.3f}" if results.get('ppl_eval') else '',
            f"{results.get('instruct_quality', 0):.3f}" if results.get('instruct_quality') else '',
            f"{results.get('tok_per_sec', 0):.1f}" if results.get('tok_per_sec') else '',
            f"{results.get('eval_time_sec', 0):.1f}" if results.get('eval_time_sec') else '',
            results.get('model_path', ''),
            results.get('notes', '')
        ]
        
        # Append to CSV
        with open(self.bench_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)
        
        print(f"Results logged to: {self.bench_csv_path}")
    
    def load_benchmark_history(self) -> List[Dict[str, Any]]:
        """Load historical benchmark results from CSV.
        
        Returns:
            List of benchmark result dictionaries
        """
        results = []
        
        if not os.path.exists(self.bench_csv_path):
            return results
        
        with open(self.bench_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                for field in ['params_m', 'ppl_eval', 'instruct_quality', 'tok_per_sec', 'eval_time_sec']:
                    if row.get(field) and row[field] != '':
                        try:
                            row[field] = float(row[field])
                        except ValueError:
                            row[field] = None
                
                results.append(row)
        
        return results
    
    def print_benchmark_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of benchmark results.
        
        Args:
            results: Benchmark results dictionary
        """
        print(f"\n{'='*60}")
        print(f"BENCHMARK RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Model: {results['model_name']}")
        print(f"Commit: {results['commit']}")
        print(f"Parameters: {results.get('params_m', 0):.2f}M")
        print(f"Training context: {results['ctx_train']}")
        print(f"Evaluation context: {results['ctx_eval']}")
        print(f"")
        print(f"Perplexity: {results.get('ppl_eval', 'Failed')}")
        print(f"Instruction quality: {results.get('instruct_quality', 'Failed')}")  
        print(f"Inference speed: {results.get('tok_per_sec', 'Failed')} tokens/sec")
        print(f"Evaluation time: {results.get('eval_time_sec', 'Failed')} seconds")
        
        if results.get('notes'):
            print(f"Notes: {results['notes']}")
        
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark and report model performance")
    
    # Model loading arguments
    model_group = parser.add_mutually_exclusive_group(required=False)
    model_group.add_argument('--model_dir', type=str,
                            help='Directory containing model files (auto-detects .npz and .json)')
    model_group.add_argument('--model_paths', nargs=2, metavar=('MODEL_PATH', 'CONFIG_PATH'),
                            help='Explicit paths to model.npz and config.json files')
    
    # Benchmark arguments
    parser.add_argument('--model_name', type=str, required=False,
                       help='Name/identifier for this model (e.g., "baseline", "with_rope")')
    parser.add_argument('--commit_hash', type=str, default=None,
                       help='Git commit hash (auto-detected if not provided)')
    parser.add_argument('--notes', type=str, default='',
                       help='Additional notes about this benchmark run')
    
    # Action arguments
    parser.add_argument('--run_full_benchmark', action='store_true',
                       help='Run full benchmark suite (perplexity + instruction + speed)')
    parser.add_argument('--ppl_only', action='store_true',
                       help='Run only perplexity evaluation')
    parser.add_argument('--instruct_only', action='store_true',
                       help='Run only instruction evaluation')
    parser.add_argument('--speed_only', action='store_true',
                       help='Run only speed measurement')
    
    # Evaluation parameters
    parser.add_argument('--ctx_eval', type=int, default=512,
                       help='Context length for evaluation')
    parser.add_argument('--max_instruct_examples', type=int, default=10,
                       help='Maximum instruction examples for benchmark')
    
    # Output arguments
    parser.add_argument('--reports_dir', type=str, default='reports',
                       help='Directory for benchmark reports')
    parser.add_argument('--show_history', action='store_true',
                       help='Show historical benchmark results')
    
    args = parser.parse_args()
    
    # Initialize reporter
    reporter = BenchmarkReporter(args.reports_dir)
    
    # Show history if requested
    if args.show_history:
        history = reporter.load_benchmark_history()
        if history:
            print(f"Found {len(history)} historical benchmark results:")
            for i, result in enumerate(history[-10:]):  # Show last 10
                print(f"{i+1:2d}. {result['model_name']:15} "
                     f"ppl={result.get('ppl_eval', 'N/A'):>6} "
                     f"instruct={result.get('instruct_quality', 'N/A'):>5} "
                     f"({result['timestamp'][:10]})")
        else:
            print("No historical benchmark results found.")
        return
    
    # For non-history commands, model paths and name are required
    if not args.model_dir and not args.model_paths:
        parser.error("Model path is required unless using --show_history")
    
    if not args.model_name:
        parser.error("--model_name is required unless using --show_history")
    
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
    
    # Run requested benchmarks
    if args.run_full_benchmark or not any([args.ppl_only, args.instruct_only, args.speed_only]):
        # Run full benchmark
        results = reporter.run_full_benchmark(
            model_path, config_path, args.model_name,
            commit_hash=args.commit_hash, notes=args.notes
        )
        reporter.print_benchmark_summary(results)
        
    else:
        # Run individual benchmarks
        results = {
            'model_name': args.model_name,
            'commit': args.commit_hash or reporter.get_git_commit_hash(),
            'model_path': model_path,
            'notes': args.notes
        }
        
        # Get model info
        params_millions, config_dict = reporter.get_model_info(model_path, config_path)
        results.update({
            'params_m': params_millions,
            'ctx_train': config_dict.get('block_size', 'unknown'),
            'ctx_eval': args.ctx_eval
        })
        
        if args.ppl_only:
            ppl, eval_time = reporter.run_perplexity_evaluation(
                model_path, config_path, ctx_eval=args.ctx_eval
            )
            results.update({'ppl_eval': ppl, 'eval_time_sec': eval_time})
        
        if args.instruct_only:
            instruct_quality = reporter.run_instruction_evaluation(
                model_path, config_path, max_examples=args.max_instruct_examples
            )
            results['instruct_quality'] = instruct_quality
        
        if args.speed_only:
            tok_per_sec = reporter.measure_inference_speed(model_path, config_path)
            results['tok_per_sec'] = tok_per_sec
        
        # Log and print results
        reporter.log_benchmark_result(results)
        reporter.print_benchmark_summary(results)


if __name__ == "__main__":
    main()
