"""
MLTeammate Experiments Module

Phase 5: Experiment management and benchmarking system.

This module provides:
- MLflow integration for experiment tracking
- Comprehensive benchmarking tools
- Configuration management
- Performance analysis and reporting

Components:
- mlflow_helper.py: Enhanced MLflow integration with nested runs
- benchmark_runner.py: Automated benchmarking across datasets and algorithms
- config.yaml: Default configuration for experiments and AutoML
"""

# Import main components
from .mlflow_helper import MLflowHelper
from .benchmark_runner import BenchmarkRunner, run_quick_benchmark

# Version and metadata
__version__ = "0.5.0"
__phase__ = "Phase 5: Experiments"

# Public API
__all__ = [
    "MLflowHelper",
    "BenchmarkRunner", 
    "run_quick_benchmark"
]
