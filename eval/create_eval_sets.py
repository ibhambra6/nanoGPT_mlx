#!/usr/bin/env python3
"""
Create evaluation datasets from the Shakespeare corpus.
This script creates held-out evaluation sets for perplexity and instruction evaluation.
"""

import os
import json
import random
import tiktoken
import numpy as np
from typing import List, Dict, Any

def create_shakespeare_eval_text():
    """Create a held-out evaluation text dataset from Shakespeare."""
    # Read the original Shakespeare data
    shakespeare_path = os.path.join('data', 'shakespeare', 'input.txt')
    
    if not os.path.exists(shakespeare_path):
        print("Shakespeare dataset not found. Please run: python data/shakespeare/prepare.py")
        return
        
    with open(shakespeare_path, 'r', encoding='utf-8') as f:
        data = f.read()
    
    # Use the last 10% as evaluation (different from val set which is 90-100%)
    # We'll use 80-90% as our eval set to avoid overlap
    n = len(data)
    eval_start = int(n * 0.8)
    eval_end = int(n * 0.9)
    eval_text = data[eval_start:eval_end]
    
    # Save eval text
    eval_text_path = os.path.join('eval', 'eval_text', 'shakespeare_eval.txt')
    os.makedirs(os.path.dirname(eval_text_path), exist_ok=True)
    
    with open(eval_text_path, 'w', encoding='utf-8') as f:
        f.write(eval_text)
    
    # Tokenize and save as binary for faster loading
    enc = tiktoken.encoding_for_model("gpt2")
    eval_ids = enc.encode_ordinary(eval_text)
    eval_ids = np.array(eval_ids, dtype=np.uint16)
    eval_ids.tofile(os.path.join('eval', 'eval_text', 'shakespeare_eval.bin'))
    
    print(f"Created evaluation text with {len(eval_ids):,} tokens")
    return eval_ids

def create_instruction_eval_set():
    """Create a simple instruction evaluation set with Shakespeare-style prompts."""
    
    # Simple instruction-following tasks in Shakespeare style
    instructions = [
        {
            "instruction": "Write a soliloquy about love in the style of Shakespeare.",
            "input": "",
            "expected_style": "shakespearean",
            "category": "creative_writing"
        },
        {
            "instruction": "Complete this line in iambic pentameter: 'To be or not to be'",
            "input": "To be or not to be",
            "expected_style": "iambic_pentameter", 
            "category": "poetry"
        },
        {
            "instruction": "Explain what a soliloquy is using Elizabethan language.",
            "input": "",
            "expected_style": "explanatory",
            "category": "explanation"
        },
        {
            "instruction": "Write a dialogue between two characters discussing honor.",
            "input": "",
            "expected_style": "dialogue",
            "category": "dialogue"
        },
        {
            "instruction": "Describe a castle using flowery, poetic language.",
            "input": "",
            "expected_style": "descriptive",
            "category": "description"
        },
        {
            "instruction": "Continue this famous quote: 'All the world's a stage'",
            "input": "All the world's a stage",
            "expected_style": "continuation",
            "category": "completion"
        },
        {
            "instruction": "Write a brief tragic monologue about betrayal.",
            "input": "",
            "expected_style": "tragic",
            "category": "monologue"
        },
        {
            "instruction": "Compose a rhyming couplet about time passing.",
            "input": "",
            "expected_style": "rhyming_couplet",
            "category": "poetry"
        },
        {
            "instruction": "Rewrite this in Shakespearean language: 'I am very sad today'",
            "input": "I am very sad today",
            "expected_style": "translation",
            "category": "translation"
        },
        {
            "instruction": "Write a short speech about courage suitable for a king.",
            "input": "",
            "expected_style": "royal_speech",
            "category": "speech"
        },
        {
            "instruction": "Create a metaphor comparing life to a journey.",
            "input": "",
            "expected_style": "metaphorical",
            "category": "metaphor"
        },
        {
            "instruction": "Write the opening lines of a play set in a medieval court.",
            "input": "",
            "expected_style": "dramatic_opening",
            "category": "drama"
        },
        {
            "instruction": "Compose a sonnet about the changing seasons.",
            "input": "",
            "expected_style": "sonnet",
            "category": "poetry"
        },
        {
            "instruction": "Write a character's dying words that are both noble and sad.",
            "input": "",
            "expected_style": "dying_words",
            "category": "dramatic"
        },
        {
            "instruction": "Create a conversation between a wise fool and a king.",
            "input": "",
            "expected_style": "wise_fool",
            "category": "dialogue"
        },
        {
            "instruction": "Write a proclamation announcing a royal wedding.",
            "input": "",
            "expected_style": "proclamation",
            "category": "formal"
        },
        {
            "instruction": "Describe a fierce battle using vivid imagery.",
            "input": "",
            "expected_style": "battle_description",
            "category": "action"
        },
        {
            "instruction": "Write a love letter in the style of Romeo to Juliet.",
            "input": "",
            "expected_style": "love_letter",
            "category": "romantic"
        },
        {
            "instruction": "Create a curse or spell spoken by a witch.",
            "input": "",
            "expected_style": "mystical",
            "category": "supernatural"
        },
        {
            "instruction": "Write the final lines of a tragedy where justice is served.",
            "input": "",
            "expected_style": "tragic_resolution",
            "category": "resolution"
        }
    ]
    
    # Save as JSONL
    eval_instruct_path = os.path.join('eval', 'eval_instruct.jsonl')
    
    with open(eval_instruct_path, 'w', encoding='utf-8') as f:
        for item in instructions:
            f.write(json.dumps(item) + '\n')
    
    print(f"Created instruction evaluation set with {len(instructions)} examples")
    return instructions

def create_eval_datasets():
    """Create both evaluation datasets."""
    print("Creating evaluation datasets...")
    
    # Create evaluation text dataset
    eval_text_tokens = create_shakespeare_eval_text()
    
    # Create instruction evaluation dataset  
    instructions = create_instruction_eval_set()
    
    # Create metadata file
    metadata = {
        "eval_text": {
            "path": "eval/eval_text/shakespeare_eval.txt",
            "binary_path": "eval/eval_text/shakespeare_eval.bin", 
            "num_tokens": int(len(eval_text_tokens)) if eval_text_tokens is not None else 0,
            "description": "Held-out Shakespeare text for perplexity evaluation"
        },
        "eval_instruct": {
            "path": "eval/eval_instruct.jsonl",
            "num_examples": len(instructions),
            "description": "Shakespeare-style instruction following evaluation set"
        },
        "created_by": "create_eval_sets.py",
        "tokenizer": "tiktoken gpt2"
    }
    
    with open(os.path.join('eval', 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\nEvaluation datasets created successfully!")
    print(f"- Eval text: {metadata['eval_text']['num_tokens']:,} tokens")
    print(f"- Instruction set: {metadata['eval_instruct']['num_examples']} examples")

if __name__ == "__main__":
    create_eval_datasets()
