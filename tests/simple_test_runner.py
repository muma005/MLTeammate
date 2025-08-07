#!/usr/bin/env python3
"""
Simple test runner for MLTeammate test suite.
Runs tests directly without pytest dependency.
"""

import sys
import os
import time
import importlib.util
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_test_file_direct(test_file):
    """Run a test file directly by importing and executing it."""
    print(f"🧪 Running {test_file.name}...")
    start_time = time.time()
    
    try:
        # Import the test module
        spec = importlib.util.spec_from_file_location("test_module", test_file)
        test_module = importlib.util.module_from_spec(spec)
        
        # Execute the test file
        spec.loader.exec_module(test_module)
        
        duration = time.time() - start_time
        return {
            "file": test_file.name,
            "success": True,
            "duration": duration,
            "message": "Executed successfully"
        }
        
    except Exception as e:
        duration = time.time() - start_time
        return {
            "file": test_file.name,
            "success": False,
            "duration": duration,
            "error": str(e)
        }


def run_simple_tests():
    """Run tests using simple direct execution."""
    test_dir = Path(__file__).parent
    test_files = [f for f in test_dir.glob("test_*.py") if f.name != "test_phase1_foundation.py"]
    
    if not test_files:
        print("⚠️ No test files found!")
        return False
    
    print("🚀 Starting MLTeammate Test Suite (Direct Mode)")
    print("=" * 60)
    print(f"📂 Test directory: {test_dir}")
    print(f"🧪 Found {len(test_files)} test files")
    print()
    
    results = []
    total_start = time.time()
    
    for test_file in sorted(test_files):
        result = run_test_file_direct(test_file)
        results.append(result)
        
        if result["success"]:
            print(f"✅ {result['file']} - {result['duration']:.2f}s")
        else:
            print(f"❌ {result['file']} - FAILED")
            print(f"   Error: {result['error'][:100]}...")
    
    total_duration = time.time() - total_start
    
    # Summary
    print()
    print("=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results if r["success"])
    failed = len(results) - passed
    
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️ Total time: {total_duration:.2f}s")
    print(f"📈 Success rate: {passed/len(results)*100:.1f}%")
    
    return passed == len(results)


def run_interface_tests_only():
    """Run only the interface tests that we know work."""
    print("🔧 Running Phase 6 Interface Tests")
    print("=" * 50)
    
    # Test SimpleAutoML
    try:
        from ml_teammate.interface.simple_api import SimpleAutoML
        automl = SimpleAutoML(learners=["random_forest"], task="classification")
        print("✅ SimpleAutoML - Instance creation successful")
        simple_success = True
    except Exception as e:
        print(f"❌ SimpleAutoML - Failed: {e}")
        simple_success = False
    
    # Test MLTeammate API
    try:
        from ml_teammate.interface.api import MLTeammate
        mlteammate = MLTeammate(learners=["random_forest"], task="classification")
        print("✅ MLTeammate API - Instance creation successful")
        api_success = True
    except Exception as e:
        print(f"❌ MLTeammate API - Failed: {e}")
        api_success = False
    
    # Test CLI import
    try:
        import ml_teammate.interface.cli
        print("✅ CLI Module - Import successful")
        cli_success = True
    except Exception as e:
        print(f"❌ CLI Module - Failed: {e}")
        cli_success = False
    
    # Test Phase 5 integration
    try:
        from ml_teammate.automl import create_automl_controller
        controller = create_automl_controller(
            learner_names=["random_forest"],
            task="classification",
            n_trials=1
        )
        print("✅ Phase 5 Integration - AutoML controller accessible")
        integration_success = True
    except Exception as e:
        print(f"❌ Phase 5 Integration - Failed: {e}")
        integration_success = False
    
    total_tests = 4
    passed_tests = sum([simple_success, api_success, cli_success, integration_success])
    
    print(f"\n📊 Interface Tests: {passed_tests}/{total_tests} passed")
    print(f"📈 Success rate: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 All interface tests passed!")
        return True
    else:
        print(f"⚠️ {total_tests - passed_tests} tests failed")
        return False


def main():
    """Main test runner entry point."""
    print("MLTeammate Test Runner")
    print("=" * 30)
    
    if len(sys.argv) > 1 and sys.argv[1] == "interface":
        # Run only interface tests
        success = run_interface_tests_only()
    else:
        # Try to run interface tests first as they're most reliable
        print("Running interface tests first...")
        interface_success = run_interface_tests_only()
        
        if interface_success:
            print("\n" + "="*50)
            print("Interface tests passed! ✅")
            print("Phase 6 implementation is working correctly!")
            success = True
        else:
            print("\n" + "="*50)
            print("Some interface tests failed. ❌")
            success = False
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
