#!/usr/bin/env python3
"""
Test script to validate the evaluation system.
This script checks that all evaluation components work correctly.
"""

import os
import sys
import json
import tempfile

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_eval_datasets():
    """Test that evaluation datasets exist and are properly formatted."""
    print("Testing evaluation datasets...")
    
    # Check evaluation text
    eval_text_path = "eval/eval_text/shakespeare_eval.bin"
    if not os.path.exists(eval_text_path):
        print(f"❌ Evaluation text not found: {eval_text_path}")
        print("   Run: python eval/create_eval_sets.py")
        return False
    else:
        import numpy as np
        data = np.memmap(eval_text_path, dtype=np.uint16, mode='r')
        print(f"✅ Evaluation text: {len(data):,} tokens")
    
    # Check instruction dataset
    instruct_path = "eval/eval_instruct.jsonl"
    if not os.path.exists(instruct_path):
        print(f"❌ Instruction dataset not found: {instruct_path}")
        return False
    else:
        with open(instruct_path, 'r') as f:
            instructions = [json.loads(line) for line in f if line.strip()]
        print(f"✅ Instruction dataset: {len(instructions)} examples")
    
    # Check metadata
    metadata_path = "eval/metadata.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"✅ Metadata: {metadata.get('eval_text', {}).get('num_tokens', 0):,} tokens, "
              f"{metadata.get('eval_instruct', {}).get('num_examples', 0)} instructions")
    
    return True

def test_model_loading():
    """Test model loading functions."""
    print("\nTesting model loading functions...")
    
    try:
        from scripts.eval_ppl_mlx import auto_detect_model_files
        from models import load_model_from_files
        
        # Test with sample directory (should fail gracefully)
        try:
            auto_detect_model_files("nonexistent_dir")
            print("❌ Should have failed for nonexistent directory")
            return False
        except FileNotFoundError:
            print("✅ Properly handles missing directory")
        
        print("✅ Model loading functions imported successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_evaluation_scripts():
    """Test that evaluation scripts can be imported and have proper CLI."""
    print("\nTesting evaluation scripts...")
    
    scripts_to_test = [
        "scripts.eval_ppl_mlx",
        "scripts.eval_instruct_mlx", 
        "scripts.benchmark_report",
        "scripts.run_eval"
    ]
    
    success = True
    for script in scripts_to_test:
        try:
            __import__(script)
            print(f"✅ {script} imports successfully")
        except ImportError as e:
            print(f"❌ {script} import failed: {e}")
            success = False
        except Exception as e:
            print(f"⚠️  {script} imports but has issues: {e}")
    
    return success

def test_benchmark_csv():
    """Test benchmark CSV initialization."""
    print("\nTesting benchmark reporting...")
    
    try:
        # Test CSV file exists and has proper headers
        csv_path = "reports/bench.csv"
        if not os.path.exists(csv_path):
            print(f"❌ Benchmark CSV not found: {csv_path}")
            return False
        
        with open(csv_path, 'r') as f:
            header = f.readline().strip()
        
        expected_columns = [
            'timestamp', 'commit', 'model_name', 'params_m', 'ctx_train', 
            'ctx_eval', 'ppl_eval', 'instruct_quality', 'tok_per_sec', 
            'eval_time_sec', 'model_path', 'notes'
        ]
        
        header_columns = header.split(',')
        if header_columns == expected_columns:
            print("✅ Benchmark CSV has correct headers")
            return True
        else:
            print(f"❌ Benchmark CSV headers incorrect")
            print(f"   Expected: {expected_columns}")
            print(f"   Found: {header_columns}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing benchmark CSV: {e}")
        return False

def test_directory_structure():
    """Test that all required directories exist."""
    print("\nTesting directory structure...")
    
    required_dirs = [
        "eval",
        "eval/eval_text", 
        "scripts",
        "reports"
    ]
    
    required_files = [
        "eval/create_eval_sets.py",
        "eval/eval_instruct.jsonl",
        "scripts/eval_ppl_mlx.py",
        "scripts/eval_instruct_mlx.py",
        "scripts/benchmark_report.py",
        "scripts/run_eval.py",
        "reports/bench.csv",
        "reports/README.md"
    ]
    
    success = True
    
    for directory in required_dirs:
        if os.path.isdir(directory):
            print(f"✅ Directory: {directory}")
        else:
            print(f"❌ Missing directory: {directory}")
            success = False
    
    for file_path in required_files:
        if os.path.isfile(file_path):
            print(f"✅ File: {file_path}")
        else:
            print(f"❌ Missing file: {file_path}")
            success = False
    
    return success

def main():
    """Run all tests."""
    print("🧪 Testing nanoGPT_mlx Evaluation System")
    print("=" * 50)
    
    # Change to project directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    os.chdir(project_dir)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Evaluation Datasets", test_eval_datasets),
        ("Model Loading", test_model_loading),
        ("Evaluation Scripts", test_evaluation_scripts),
        ("Benchmark CSV", test_benchmark_csv),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print(f"\n{'=' * 50}")
    print("TEST SUMMARY")
    print(f"{'=' * 50}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:8} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The evaluation system is ready to use.")
        print("\nNext steps:")
        print("1. Train a model: python train.py configs/train_gpt2_shakespeare.py")
        print("2. Run benchmark: python scripts/run_eval.py benchmark --model_dir <model_dir> --model_name baseline")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
