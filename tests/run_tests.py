# tests/run_tests.py
"""
Test runner for MLTeammate test suite.

This script runs all tests and provides comprehensive reporting.
"""

import sys
import os
import subprocess
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_test_file(test_file):
    """Run a single test file and return results."""
    print(f"🧪 Running {test_file}...")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_file), "-v"],
            capture_output=True,
            text=True,
            cwd=project_root
        )
        
        duration = time.time() - start_time
        
        return {
            "file": test_file.name,
            "success": result.returncode == 0,
            "duration": duration,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except Exception as e:
        return {
            "file": test_file.name,
            "success": False,
            "duration": time.time() - start_time,
            "error": str(e)
        }


def run_all_tests():
    """Run all test files in the tests directory."""
    test_dir = Path(__file__).parent
    test_files = list(test_dir.glob("test_*.py"))
    
    if not test_files:
        print("⚠️ No test files found!")
        return False
    
    print("🚀 Starting MLTeammate Test Suite")
    print("=" * 50)
    print(f"📂 Test directory: {test_dir}")
    print(f"🧪 Found {len(test_files)} test files")
    print()
    
    results = []
    total_start = time.time()
    
    for test_file in sorted(test_files):
        result = run_test_file(test_file)
        results.append(result)
        
        if result["success"]:
            print(f"✅ {result['file']} - {result['duration']:.2f}s")
        else:
            print(f"❌ {result['file']} - FAILED")
            if "error" in result:
                print(f"   Error: {result['error']}")
    
    total_duration = time.time() - total_start
    
    # Summary
    print()
    print("=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for r in results if r["success"])
    failed = len(results) - passed
    
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️ Total time: {total_duration:.2f}s")
    print(f"📈 Success rate: {passed/len(results)*100:.1f}%")
    
    # Failed test details
    if failed > 0:
        print()
        print("❌ FAILED TESTS:")
        for result in results:
            if not result["success"]:
                print(f"  - {result['file']}")
                if result.get("stderr"):
                    print(f"    Error: {result['stderr'][:200]}...")
    
    return passed == len(results)


def run_specific_test(test_name):
    """Run a specific test file."""
    test_dir = Path(__file__).parent
    test_file = test_dir / f"test_{test_name}.py"
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    result = run_test_file(test_file)
    
    if result["success"]:
        print(f"✅ {result['file']} passed in {result['duration']:.2f}s")
        return True
    else:
        print(f"❌ {result['file']} failed")
        if result.get("stderr"):
            print(f"Error: {result['stderr']}")
        return False


def run_interface_tests():
    """Run only interface-related tests."""
    print("🔧 Running Interface Tests")
    print("=" * 40)
    
    interface_tests = [
        "test_simple_api.py",
        "test_automl_controller.py"
    ]
    
    test_dir = Path(__file__).parent
    results = []
    
    for test_name in interface_tests:
        test_file = test_dir / test_name
        if test_file.exists():
            result = run_test_file(test_file)
            results.append(result)
            
            if result["success"]:
                print(f"✅ {result['file']} - {result['duration']:.2f}s")
            else:
                print(f"❌ {result['file']} - FAILED")
    
    passed = sum(1 for r in results if r["success"])
    total = len(results)
    
    print(f"\nInterface Tests: {passed}/{total} passed")
    return passed == total


def main():
    """Main test runner entry point."""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "all":
            # Run all tests
            success = run_all_tests()
        elif command == "interface":
            # Run interface tests
            success = run_interface_tests()
        else:
            # Run specific test
            success = run_specific_test(command)
        
        sys.exit(0 if success else 1)
    else:
        # Default: run all tests
        success = run_all_tests()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
