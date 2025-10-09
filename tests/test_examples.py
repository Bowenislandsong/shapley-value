#!/usr/bin/env python3
"""
Test Runner for Shapley Value Examples

This script runs all examples and reports on their success/failure status.
"""

import subprocess
import sys
import time
from pathlib import Path


def run_example(example_name):
    """Run a single example and return success status and output"""
    try:
        print(f"\n{'='*60}")
        print(f"TESTING: {example_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, "-m", f"examples.{example_name}"],
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        end_time = time.time()
        
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ SUCCESS ({duration:.2f}s)")
            return True, result.stdout, duration
        else:
            print(f"❌ FAILED ({duration:.2f}s)")
            print("STDERR:", result.stderr)
            return False, result.stderr, duration
            
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT (>120s)")
        return False, "Timeout expired", 120
    except Exception as e:
        print(f"💥 ERROR: {e}")
        return False, str(e), 0


def main():
    """Run all examples and provide summary"""
    print("Shapley Value Calculator - Example Test Suite")
    print("=" * 60)
    
    examples = [
        "example_basic_coalition",
        "example_function_evaluation", 
        "example_business_case",
        "example_ml_features",
        "example_parallel_processing"
    ]
    
    results = {}
    total_time = 0
    
    for example in examples:
        success, output, duration = run_example(example)
        results[example] = {
            'success': success,
            'output': output,
            'duration': duration
        }
        total_time += duration
        
        # Brief pause between tests
        time.sleep(0.5)
    
    # Summary Report
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    successful = sum(1 for r in results.values() if r['success'])
    total_tests = len(results)
    
    print(f"Results: {successful}/{total_tests} examples passed")
    print(f"Total execution time: {total_time:.2f}s")
    print()
    
    for example, result in results.items():
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        duration = result['duration']
        print(f"{example:30s} {status:8s} ({duration:6.2f}s)")
    
    if successful == total_tests:
        print(f"\n🎉 All examples working correctly!")
        return 0
    else:
        print(f"\n⚠️  {total_tests - successful} example(s) failed")
        print("\nFailed examples details:")
        for example, result in results.items():
            if not result['success']:
                print(f"\n{example}:")
                print(result['output'][:500] + "..." if len(result['output']) > 500 else result['output'])
        return 1


if __name__ == "__main__":
    sys.exit(main())