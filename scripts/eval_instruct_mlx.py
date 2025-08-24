#!/usr/bin/env python3
"""
Instruction following evaluation script for MLX models.
Evaluates how well models follow Shakespeare-style instructions.

Usage:
    python scripts/eval_instruct_mlx.py --model_dir out/ --eval_data eval/eval_instruct.jsonl
"""

import os
import json
import time
import argparse
from typing import Dict, List, Tuple, Any, Optional

import mlx.core as mx
import tiktoken
from mlx.utils import tree_unflatten

# Add parent directory to path to import model
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model import GPT, GPTConfig
from scripts.eval_ppl_mlx import load_model, auto_detect_model_files


class InstructionEvaluator:
    """Evaluates instruction following capability of models."""
    
    def __init__(self, model: GPT, tokenizer_name: str = "gpt2"):
        """Initialize instruction evaluator.
        
        Args:
            model: GPT model to evaluate
            tokenizer_name: Name of tokenizer to use
        """
        self.model = model
        self.enc = tiktoken.encoding_for_model(tokenizer_name)
        
    def format_prompt(self, instruction: str, input_text: str = "") -> str:
        """Format instruction and input into a prompt.
        
        Args:
            instruction: The instruction to follow
            input_text: Optional input text
            
        Returns:
            Formatted prompt string
        """
        if input_text.strip():
            prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse:"
        else:
            prompt = f"Instruction: {instruction}\nResponse:"
        
        return prompt
    
    def generate_response(self, prompt: str, max_new_tokens: int = 256, 
                         temperature: float = 0.8, top_k: Optional[int] = 50) -> str:
        """Generate a response to a prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            
        Returns:
            Generated response text
        """
        # Encode prompt
        prompt_tokens = self.enc.encode(prompt)
        prompt_array = mx.array([prompt_tokens], dtype=mx.int64)
        
        # Generate response
        self.model.eval()
        with mx.no_grad():
            generated = self.model.generate(
                prompt_array, 
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k
            )
        
        # Decode full sequence and extract response
        full_text = self.enc.decode(generated[0].tolist())
        
        # Extract just the generated part (after the prompt)
        response = full_text[len(prompt):]
        
        return response.strip()
    
    def evaluate_response_quality(self, instruction: str, input_text: str, 
                                response: str, expected_style: str, 
                                category: str) -> Dict[str, Any]:
        """Evaluate the quality of a generated response.
        
        This is a simple heuristic-based evaluation. In practice, you might
        want to use more sophisticated methods like:
        - Human evaluation
        - LLM-as-a-judge evaluation
        - BLEU/ROUGE scores against reference responses
        - Style classifiers
        
        Args:
            instruction: Original instruction
            input_text: Input text (if any)
            response: Generated response
            expected_style: Expected style/format
            category: Category of the instruction
            
        Returns:
            Dictionary with evaluation metrics
        """
        scores = {}
        
        # Basic response quality checks
        scores['response_length'] = len(response.split())
        scores['has_content'] = len(response.strip()) > 0
        scores['non_repetitive'] = self._check_non_repetitive(response)
        
        # Style-specific checks
        if expected_style == "shakespearean":
            scores['archaic_language'] = self._check_archaic_language(response)
            scores['poetic_style'] = self._check_poetic_elements(response)
        elif expected_style == "iambic_pentameter":
            scores['meter_attempt'] = self._check_meter_attempt(response)
        elif expected_style == "rhyming_couplet":
            scores['rhyme_attempt'] = self._check_rhyme_attempt(response)
        elif expected_style == "dialogue":
            scores['dialogue_format'] = self._check_dialogue_format(response)
        elif expected_style == "sonnet":
            scores['sonnet_structure'] = self._check_sonnet_structure(response)
        
        # Category-specific checks
        if category == "poetry":
            scores['line_breaks'] = self._check_line_breaks(response)
        elif category == "explanation":
            scores['explanatory_content'] = self._check_explanatory_content(response)
        
        # Overall quality score (simple heuristic)
        quality_indicators = [
            scores['has_content'],
            scores['non_repetitive'],
            scores['response_length'] > 5,  # At least 5 words
            scores['response_length'] < 200,  # Not too verbose
        ]
        
        # Add style-specific indicators
        style_indicators = [v for k, v in scores.items() 
                          if k not in ['response_length', 'has_content', 'non_repetitive']]
        
        if style_indicators:
            # Weight style indicators
            style_score = sum(style_indicators) / len(style_indicators)
            quality_indicators.append(style_score > 0.5)
        
        scores['overall_quality'] = sum(quality_indicators) / len(quality_indicators)
        
        return scores
    
    def _check_non_repetitive(self, text: str) -> bool:
        """Check if text is not overly repetitive."""
        words = text.lower().split()
        if len(words) < 3:
            return True
        
        # Check for immediate repetition
        for i in range(len(words) - 2):
            if words[i] == words[i+1] == words[i+2]:
                return False
        
        # Check for high repetition rate
        unique_words = len(set(words))
        repetition_rate = unique_words / len(words) if words else 0
        return repetition_rate > 0.6
    
    def _check_archaic_language(self, text: str) -> bool:
        """Check for Shakespearean/archaic language patterns."""
        archaic_words = [
            'thou', 'thee', 'thy', 'thine', 'ye', 'hath', 'doth', 'shall',
            'art', 'ere', 'oft', 'nay', 'aye', 'whence', 'whither', 'wherefore',
            'prithee', 'methinks', 'forsooth', 'marry', 'good morrow'
        ]
        
        text_lower = text.lower()
        archaic_count = sum(1 for word in archaic_words if word in text_lower)
        return archaic_count > 0
    
    def _check_poetic_elements(self, text: str) -> bool:
        """Check for poetic elements like metaphors, imagery."""
        poetic_words = [
            'like', 'as', 'golden', 'silver', 'crimson', 'azure', 'gentle',
            'fair', 'beauty', 'love', 'heart', 'soul', 'sweet', 'soft',
            'bright', 'dark', 'light', 'shadow', 'dream', 'rose', 'star'
        ]
        
        text_lower = text.lower()
        poetic_count = sum(1 for word in poetic_words if word in text_lower)
        return poetic_count >= 2
    
    def _check_meter_attempt(self, text: str) -> bool:
        """Check if text attempts rhythmic meter."""
        # Simple heuristic: look for lines with 8-12 syllables
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not lines:
            return False
        
        # Rough syllable count (vowel groups)
        import re
        
        for line in lines:
            vowel_groups = len(re.findall(r'[aeiouAEIOU]+', line))
            if 8 <= vowel_groups <= 12:
                return True
        
        return False
    
    def _check_rhyme_attempt(self, text: str) -> bool:
        """Check if text attempts to rhyme."""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if len(lines) < 2:
            return False
        
        # Simple rhyme check: look for similar endings
        for i in range(len(lines) - 1):
            line1_end = lines[i].split()[-1].lower() if lines[i].split() else ""
            line2_end = lines[i+1].split()[-1].lower() if lines[i+1].split() else ""
            
            if len(line1_end) > 2 and len(line2_end) > 2:
                if line1_end[-2:] == line2_end[-2:] or line1_end[-3:] == line2_end[-3:]:
                    return True
        
        return False
    
    def _check_dialogue_format(self, text: str) -> bool:
        """Check if text is formatted as dialogue."""
        dialogue_markers = [':', '"', "'", '—', '-']
        speaker_indicators = ['said', 'spoke', 'replied', 'asked', 'answered']
        
        has_markers = any(marker in text for marker in dialogue_markers)
        has_indicators = any(indicator in text.lower() for indicator in speaker_indicators)
        
        return has_markers or has_indicators
    
    def _check_sonnet_structure(self, text: str) -> bool:
        """Check if text attempts sonnet structure."""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return 10 <= len(lines) <= 18  # Roughly sonnet length
    
    def _check_line_breaks(self, text: str) -> bool:
        """Check if text has appropriate line breaks for poetry."""
        return '\n' in text and len(text.split('\n')) >= 2
    
    def _check_explanatory_content(self, text: str) -> bool:
        """Check if text provides explanatory content."""
        explanatory_words = [
            'is', 'are', 'means', 'refers', 'describes', 'involves',
            'consists', 'includes', 'definition', 'explanation'
        ]
        
        text_lower = text.lower()
        return any(word in text_lower for word in explanatory_words)


def load_instruction_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load instruction evaluation dataset.
    
    Args:
        dataset_path: Path to JSONL file with instructions
        
    Returns:
        List of instruction examples
    """
    instructions = []
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                instructions.append(json.loads(line.strip()))
    
    return instructions


def evaluate_instructions(model_path: str, config_path: str, 
                        dataset_path: str, output_file: Optional[str] = None,
                        max_examples: Optional[int] = None,
                        generation_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Evaluate instruction following on a dataset.
    
    Args:
        model_path: Path to model weights
        config_path: Path to model config  
        dataset_path: Path to instruction dataset
        output_file: Optional file to save detailed results
        max_examples: Maximum number of examples to evaluate
        generation_params: Parameters for text generation
        
    Returns:
        Summary evaluation results
    """
    # Default generation parameters
    if generation_params is None:
        generation_params = {
            'max_new_tokens': 256,
            'temperature': 0.8,
            'top_k': 50
        }
    
    # Load model and dataset
    print("Loading model...")
    model = load_model(model_path, config_path)
    
    print("Loading instruction dataset...")
    instructions = load_instruction_dataset(dataset_path)
    
    if max_examples:
        instructions = instructions[:max_examples]
    
    print(f"Evaluating {len(instructions)} instructions...")
    
    # Initialize evaluator
    evaluator = InstructionEvaluator(model)
    
    # Evaluate each instruction
    results = []
    category_scores = {}
    
    for i, example in enumerate(instructions):
        print(f"\nEvaluating {i+1}/{len(instructions)}: {example['category']}")
        
        # Generate response
        prompt = evaluator.format_prompt(example['instruction'], example.get('input', ''))
        
        start_time = time.time()
        response = evaluator.generate_response(prompt, **generation_params)
        generation_time = time.time() - start_time
        
        # Evaluate response
        scores = evaluator.evaluate_response_quality(
            example['instruction'],
            example.get('input', ''),
            response,
            example['expected_style'],
            example['category']
        )
        
        # Store result
        result = {
            'example_id': i,
            'instruction': example['instruction'],
            'input': example.get('input', ''),
            'expected_style': example['expected_style'],
            'category': example['category'],
            'prompt': prompt,
            'response': response,
            'scores': scores,
            'generation_time': generation_time
        }
        
        results.append(result)
        
        # Accumulate category scores
        category = example['category']
        if category not in category_scores:
            category_scores[category] = []
        category_scores[category].append(scores['overall_quality'])
        
        # Print brief result
        print(f"  Quality: {scores['overall_quality']:.2f}, Length: {scores['response_length']} words")
        if len(response) > 100:
            print(f"  Response: {response[:100]}...")
        else:
            print(f"  Response: {response}")
    
    # Compute summary statistics
    overall_scores = [r['scores']['overall_quality'] for r in results]
    summary = {
        'total_examples': len(instructions),
        'overall_quality': {
            'mean': sum(overall_scores) / len(overall_scores),
            'min': min(overall_scores),
            'max': max(overall_scores)
        },
        'category_scores': {
            cat: {
                'mean': sum(scores) / len(scores),
                'count': len(scores)
            }
            for cat, scores in category_scores.items()
        },
        'generation_params': generation_params,
        'model_path': model_path,
        'config_path': config_path,
        'dataset_path': dataset_path,
        'timestamp': time.time()
    }
    
    # Save detailed results if requested
    if output_file:
        detailed_results = {
            'summary': summary,
            'results': results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed results saved to: {output_file}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate instruction following of MLX models")
    
    # Model loading arguments
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--model_dir', type=str,
                            help='Directory containing model files (auto-detects .npz and .json)')
    model_group.add_argument('--model_paths', nargs=2, metavar=('MODEL_PATH', 'CONFIG_PATH'),
                            help='Explicit paths to model.npz and config.json files')
    
    # Evaluation arguments
    parser.add_argument('--eval_data', type=str, 
                       default='eval/eval_instruct.jsonl',
                       help='Path to instruction evaluation data (.jsonl)')
    parser.add_argument('--max_examples', type=int, default=None,
                       help='Maximum number of examples to evaluate')
    
    # Generation arguments
    parser.add_argument('--max_new_tokens', type=int, default=256,
                       help='Maximum tokens to generate per response')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-k sampling parameter')
    
    # Output arguments
    parser.add_argument('--output_file', type=str, default=None,
                       help='File to save detailed results (JSON format)')
    
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
    
    # Set up generation parameters
    generation_params = {
        'max_new_tokens': args.max_new_tokens,
        'temperature': args.temperature,
        'top_k': args.top_k
    }
    
    # Run evaluation
    print("Starting instruction evaluation...")
    summary = evaluate_instructions(
        model_path, config_path, args.eval_data,
        output_file=args.output_file,
        max_examples=args.max_examples,
        generation_params=generation_params
    )
    
    # Print summary results
    print(f"\n{'='*60}")
    print(f"INSTRUCTION FOLLOWING EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Dataset: {args.eval_data}")
    print(f"Examples evaluated: {summary['total_examples']}")
    print(f"Overall quality: {summary['overall_quality']['mean']:.3f}")
    print(f"Quality range: {summary['overall_quality']['min']:.3f} - {summary['overall_quality']['max']:.3f}")
    
    print(f"\nCategory breakdown:")
    for category, stats in summary['category_scores'].items():
        print(f"  {category:15}: {stats['mean']:.3f} ({stats['count']} examples)")


if __name__ == "__main__":
    main()
