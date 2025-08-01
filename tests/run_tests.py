#!/usr/bin/env python3
"""
Comprehensive test runner for MLTeammate.

This script runs all tests and generates comprehensive reports including:
- Unit test results
- Integration test results
- Performance benchmarks
- Error handling tests
- Coverage reports
- Test summaries
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from typing import Dict, List, Any


class TestRunner:
    """Comprehensive test runner for MLTeammate."""
    
    def __init__(self):
        """Initialize the test runner."""
        self.project_root = Path(__file__).parent.parent
        self.tests_dir = self.project_root / "tests"
        self.results = {
            "unit_tests": {},
            "integration_tests": {},
            "performance_tests": {},
            "error_handling_tests": {},
            "coverage": {},
            "summary": {}
        }
        self.start_time = None
    
    def run_command(self, command: List[str], capture_output: bool = True) -> Dict[str, Any]:
        """Run a command and return results."""
        print(f"Running: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                capture_output=capture_output,
                text=True,
                cwd=self.project_root
            )
            
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except Exception as e:
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": str(e)
            }
    
    def run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests."""
        print("\n" + "="*60)
        print("RUNNING UNIT TESTS")
        print("="*60)
        
        test_files = [
            "test_automl_controller.py",
            "test_search_components.py", 
            "test_learner_registry.py",
            "test_simple_api.py",
            "test_callbacks.py"
        ]
        
        results = {}
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        
        for test_file in test_files:
            test_path = self.tests_dir / test_file
            if test_path.exists():
                print(f"\nRunning {test_file}...")
                result = self.run_command([
                    sys.executable, "-m", "pytest", str(test_path), "-v"
                ])
                
                # Parse test results
                if result["success"]:
                    # Count passed tests
                    lines = result["stdout"].split('\n')
                    for line in lines:
                        if "passed" in line and "failed" not in line:
                            passed_tests += 1
                        elif "failed" in line:
                            failed_tests += 1
                        total_tests += 1
                
                results[test_file] = result
            else:
                print(f"Warning: {test_file} not found")
        
        return {
            "results": results,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success": failed_tests == 0
        }
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests."""
        print("\n" + "="*60)
        print("RUNNING INTEGRATION TESTS")
        print("="*60)
        
        test_file = "test_tutorials_integration.py"
        test_path = self.tests_dir / test_file
        
        if test_path.exists():
            print(f"Running {test_file}...")
            result = self.run_command([
                sys.executable, "-m", "pytest", str(test_path), "-v"
            ])
            
            # Parse integration test results
            total_tests = 0
            passed_tests = 0
            failed_tests = 0
            
            if result["success"]:
                lines = result["stdout"].split('\n')
                for line in lines:
                    if "passed" in line and "failed" not in line:
                        passed_tests += 1
                    elif "failed" in line:
                        failed_tests += 1
                    total_tests += 1
            
            return {
                "result": result,
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success": failed_tests == 0
            }
        else:
            print(f"Warning: {test_file} not found")
            return {"success": False, "error": "Test file not found"}
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests."""
        print("\n" + "="*60)
        print("RUNNING PERFORMANCE TESTS")
        print("="*60)
        
        test_file = "test_performance_benchmarks.py"
        test_path = self.tests_dir / test_file
        
        if test_path.exists():
            print(f"Running {test_file}...")
            result = self.run_command([
                sys.executable, "-m", "pytest", str(test_path), "-v", "--tb=short"
            ])
            
            return {
                "result": result,
                "success": result["success"]
            }
        else:
            print(f"Warning: {test_file} not found")
            return {"success": False, "error": "Test file not found"}
    
    def run_error_handling_tests(self) -> Dict[str, Any]:
        """Run error handling tests."""
        print("\n" + "="*60)
        print("RUNNING ERROR HANDLING TESTS")
        print("="*60)
        
        test_file = "test_error_handling.py"
        test_path = self.tests_dir / test_file
        
        if test_path.exists():
            print(f"Running {test_file}...")
            result = self.run_command([
                sys.executable, "-m", "pytest", str(test_path), "-v"
            ])
            
            # Parse error handling test results
            total_tests = 0
            passed_tests = 0
            failed_tests = 0
            
            if result["success"]:
                lines = result["stdout"].split('\n')
                for line in lines:
                    if "passed" in line and "failed" not in line:
                        passed_tests += 1
                    elif "failed" in line:
                        failed_tests += 1
                    total_tests += 1
            
            return {
                "result": result,
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success": failed_tests == 0
            }
        else:
            print(f"Warning: {test_file} not found")
            return {"success": False, "error": "Test file not found"}
    
    def run_coverage_tests(self) -> Dict[str, Any]:
        """Run tests with coverage reporting."""
        print("\n" + "="*60)
        print("RUNNING COVERAGE TESTS")
        print("="*60)
        
        # Check if coverage is installed
        coverage_result = self.run_command([
            sys.executable, "-c", "import coverage; print('coverage available')"
        ])
        
        if not coverage_result["success"]:
            print("coverage not installed, installing...")
            install_result = self.run_command([
                sys.executable, "-m", "pip", "install", "coverage"
            ])
            if not install_result["success"]:
                return {"success": False, "error": "Failed to install coverage"}
        
        # Run coverage
        coverage_result = self.run_command([
            sys.executable, "-m", "coverage", "run", "--source=ml_teammate", 
            "-m", "pytest", str(self.tests_dir)
        ])
        
        if coverage_result["success"]:
            # Generate coverage report
            report_result = self.run_command([
                sys.executable, "-m", "coverage", "report"
            ])
            
            # Generate HTML report
            html_result = self.run_command([
                sys.executable, "-m", "coverage", "html"
            ])
            
            return {
                "coverage_run": coverage_result,
                "coverage_report": report_result,
                "coverage_html": html_result,
                "success": coverage_result["success"]
            }
        else:
            return {"success": False, "error": "Coverage run failed"}
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and generate comprehensive report."""
        self.start_time = time.time()
        
        print("MLTeammate Comprehensive Test Suite")
        print("="*60)
        print(f"Project root: {self.project_root}")
        print(f"Tests directory: {self.tests_dir}")
        print(f"Python executable: {sys.executable}")
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run all test suites
        self.results["unit_tests"] = self.run_unit_tests()
        self.results["integration_tests"] = self.run_integration_tests()
        self.results["performance_tests"] = self.run_performance_tests()
        self.results["error_handling_tests"] = self.run_error_handling_tests()
        self.results["coverage"] = self.run_coverage_tests()
        
        # Generate summary
        self.results["summary"] = self.generate_summary()
        
        # Save results
        self.save_results()
        
        # Print final summary
        self.print_summary()
        
        return self.results
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate test summary."""
        end_time = time.time()
        duration = end_time - self.start_time
        
        # Calculate totals
        total_tests = 0
        total_passed = 0
        total_failed = 0
        
        # Unit tests
        unit_tests = self.results["unit_tests"]
        total_tests += unit_tests.get("total_tests", 0)
        total_passed += unit_tests.get("passed_tests", 0)
        total_failed += unit_tests.get("failed_tests", 0)
        
        # Integration tests
        integration_tests = self.results["integration_tests"]
        total_tests += integration_tests.get("total_tests", 0)
        total_passed += integration_tests.get("passed_tests", 0)
        total_failed += integration_tests.get("failed_tests", 0)
        
        # Error handling tests
        error_tests = self.results["error_handling_tests"]
        total_tests += error_tests.get("total_tests", 0)
        total_passed += error_tests.get("passed_tests", 0)
        total_failed += error_tests.get("failed_tests", 0)
        
        # Calculate success rates
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        return {
            "duration_seconds": duration,
            "duration_formatted": f"{duration:.2f}s",
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "success_rate_percent": success_rate,
            "all_tests_passed": total_failed == 0,
            "unit_tests_passed": unit_tests.get("success", False),
            "integration_tests_passed": integration_tests.get("success", False),
            "performance_tests_passed": self.results["performance_tests"].get("success", False),
            "error_handling_tests_passed": error_tests.get("success", False),
            "coverage_passed": self.results["coverage"].get("success", False)
        }
    
    def save_results(self):
        """Save test results to file."""
        results_file = self.project_root / "test_results.json"
        
        # Convert to serializable format
        serializable_results = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                serializable_results[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, dict) and "stdout" in subvalue:
                        # Truncate stdout/stderr for JSON serialization
                        serializable_results[key][subkey] = {
                            k: v[:1000] + "..." if isinstance(v, str) and len(v) > 1000 else v
                            for k, v in subvalue.items()
                        }
                    else:
                        serializable_results[key][subkey] = subvalue
            else:
                serializable_results[key] = value
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nTest results saved to: {results_file}")
    
    def print_summary(self):
        """Print test summary."""
        summary = self.results["summary"]
        
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Duration: {summary['duration_formatted']}")
        print(f"Total tests: {summary['total_tests']}")
        print(f"Passed: {summary['total_passed']}")
        print(f"Failed: {summary['total_failed']}")
        print(f"Success rate: {summary['success_rate_percent']:.1f}%")
        print()
        
        print("Test Suite Results:")
        print(f"  Unit Tests: {'✅ PASSED' if summary['unit_tests_passed'] else '❌ FAILED'}")
        print(f"  Integration Tests: {'✅ PASSED' if summary['integration_tests_passed'] else '❌ FAILED'}")
        print(f"  Performance Tests: {'✅ PASSED' if summary['performance_tests_passed'] else '❌ FAILED'}")
        print(f"  Error Handling Tests: {'✅ PASSED' if summary['error_handling_tests_passed'] else '❌ FAILED'}")
        print(f"  Coverage Tests: {'✅ PASSED' if summary['coverage_passed'] else '❌ FAILED'}")
        print()
        
        if summary['all_tests_passed']:
            print("🎉 ALL TESTS PASSED! 🎉")
        else:
            print("⚠️  SOME TESTS FAILED! ⚠️")
        
        print("="*60)


def main():
    """Main function to run all tests."""
    runner = TestRunner()
    results = runner.run_all_tests()
    
    # Exit with appropriate code
    if results["summary"]["all_tests_passed"]:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main() 