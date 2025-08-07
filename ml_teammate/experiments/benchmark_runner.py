"""
Phase 5: Benchmark Runner for MLTeammate

Comprehensive benchmarking system for evaluating AutoML performance
across different datasets, algorithms, and configurations.

This module provides:
- Automated benchmarking workflows
- Cross-dataset comparison
- Statistical significance testing
- Performance visualization
- Report generation
"""

import os
import json
import time
import warnings
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression, load_digits, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# Import our frozen components
from ml_teammate.utils import get_logger, time_context, MetricCalculator
from ..automl.controller import AutoMLController, create_automl_controller
from ..learners.registry import get_learner_registry


class BenchmarkRunner:
    """
    Runs comprehensive benchmarks for MLTeammate AutoML system.
    
    Evaluates performance across multiple datasets, learners, and search algorithms
    with statistical analysis and reporting.
    """
    
    def __init__(self, output_dir: str = "./benchmark_results", 
                 random_state: int = 42, log_level: str = "INFO"):
        """
        Initialize benchmark runner.
        
        Args:
            output_dir: Directory to save benchmark results
            random_state: Random seed for reproducibility
            log_level: Logging level
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        self.logger = get_logger("BenchmarkRunner", log_level)
        
        # Initialize components
        self.metric_calculator = MetricCalculator()
        
        # Results storage
        self.results = {
            "benchmark_info": {},
            "dataset_results": {},
            "summary_stats": {},
            "performance_matrix": {}
        }
        
        self.logger.info(f"BenchmarkRunner initialized, output: {output_dir}")
    
    def run_classification_benchmark(self, n_trials: int = 10, 
                                   searchers: List[str] = None,
                                   learners: List[str] = None,
                                   datasets: List[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive classification benchmark.
        
        Args:
            n_trials: Number of optimization trials per experiment
            searchers: List of search algorithms to test
            learners: List of learners to include
            datasets: List of datasets to test on
            
        Returns:
            Comprehensive benchmark results
        """
        if searchers is None:
            searchers = ["random", "optuna"]
        
        if learners is None:
            registry = get_learner_registry()
            available_learners = registry.list_learners("classification")
            learners = list(available_learners.keys())[:4]  # Use first 4 learners
        
        if datasets is None:
            datasets = ["synthetic_easy", "synthetic_hard", "wine", "digits_small"]
        
        self.logger.info(f"Starting classification benchmark: {len(datasets)} datasets, "
                        f"{len(searchers)} searchers, {len(learners)} learners")
        
        return self._run_benchmark("classification", n_trials, searchers, learners, datasets)
    
    def run_regression_benchmark(self, n_trials: int = 10,
                               searchers: List[str] = None,
                               learners: List[str] = None,
                               datasets: List[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive regression benchmark.
        
        Args:
            n_trials: Number of optimization trials per experiment
            searchers: List of search algorithms to test
            learners: List of learners to include
            datasets: List of datasets to test on
            
        Returns:
            Comprehensive benchmark results
        """
        if searchers is None:
            searchers = ["random", "optuna"]
        
        if learners is None:
            registry = get_learner_registry()
            available_learners = registry.list_learners("regression")
            learners = list(available_learners.keys())[:4]  # Use first 4 learners
        
        if datasets is None:
            datasets = ["synthetic_linear", "synthetic_nonlinear"]
        
        self.logger.info(f"Starting regression benchmark: {len(datasets)} datasets, "
                        f"{len(searchers)} searchers, {len(learners)} learners")
        
        return self._run_benchmark("regression", n_trials, searchers, learners, datasets)
    
    def _run_benchmark(self, task: str, n_trials: int, searchers: List[str],
                      learners: List[str], datasets: List[str]) -> Dict[str, Any]:
        """Run benchmark for specified configuration."""
        benchmark_start = time.time()
        
        # Initialize results structure
        self.results["benchmark_info"] = {
            "task": task,
            "n_trials": n_trials,
            "searchers": searchers,
            "learners": learners,
            "datasets": datasets,
            "start_time": benchmark_start,
            "random_state": self.random_state
        }
        
        # Run experiments for each combination
        for dataset_name in datasets:
            self.logger.info(f"Testing dataset: {dataset_name}")
            
            # Load dataset
            X, y = self._load_dataset(dataset_name, task)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state
            )
            
            dataset_results = {
                "dataset_info": {
                    "name": dataset_name,
                    "n_samples": X.shape[0],
                    "n_features": X.shape[1],
                    "task": task
                },
                "searcher_results": {}
            }
            
            # Test each searcher
            for searcher_name in searchers:
                self.logger.info(f"  Testing searcher: {searcher_name}")
                
                try:
                    with time_context(f"Benchmark_{dataset_name}_{searcher_name}") as timer:
                        # Create and run AutoML controller
                        controller = create_automl_controller(
                            learner_names=learners,
                            task=task,
                            searcher_type=searcher_name,
                            n_trials=n_trials,
                            random_state=self.random_state,
                            log_level="WARNING"  # Reduce noise during benchmarks
                        )
                        
                        # Fit and evaluate
                        controller.fit(X_train, y_train, X_test, y_test)
                        
                        # Get results
                        results = controller.get_results()
                        test_score = controller.score(X_test, y_test)
                        
                        searcher_result = {
                            "best_score": results["best_score"],
                            "test_score": test_score,
                            "best_learner": results["best_config"]["learner_name"],
                            "best_config": results["best_config"],
                            "n_trials": len(results["trials"]),
                            "experiment_time": timer.get_elapsed(),
                            "search_history": results["search_history"]
                        }
                        
                        dataset_results["searcher_results"][searcher_name] = searcher_result
                        
                        self.logger.info(f"    {searcher_name}: test_score = {test_score:.4f}, "
                                       f"time = {timer.get_elapsed():.1f}s")
                
                except Exception as e:
                    self.logger.error(f"    {searcher_name} failed: {str(e)}")
                    dataset_results["searcher_results"][searcher_name] = {
                        "error": str(e),
                        "status": "failed"
                    }
            
            self.results["dataset_results"][dataset_name] = dataset_results
        
        # Compute summary statistics
        self._compute_summary_stats()
        
        # Save results
        total_time = time.time() - benchmark_start
        self.results["benchmark_info"]["total_time"] = total_time
        self._save_results()
        
        self.logger.info(f"Benchmark completed in {total_time:.1f}s")
        
        return self.results.copy()
    
    def _load_dataset(self, dataset_name: str, task: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load a benchmark dataset."""
        if dataset_name == "synthetic_easy" and task == "classification":
            return make_classification(
                n_samples=1000, n_features=10, n_informative=8, n_redundant=2,
                n_clusters_per_class=1, random_state=self.random_state
            )
        
        elif dataset_name == "synthetic_hard" and task == "classification":
            return make_classification(
                n_samples=1000, n_features=20, n_informative=10, n_redundant=5,
                n_clusters_per_class=2, class_sep=0.5, random_state=self.random_state
            )
        
        elif dataset_name == "wine" and task == "classification":
            data = load_wine()
            return data.data, data.target
        
        elif dataset_name == "digits_small" and task == "classification":
            data = load_digits()
            # Use subset for faster benchmarking
            indices = np.random.RandomState(self.random_state).choice(
                len(data.data), size=500, replace=False
            )
            return data.data[indices], data.target[indices]
        
        elif dataset_name == "synthetic_linear" and task == "regression":
            X, y = make_classification(
                n_samples=1000, n_features=10, n_informative=8,
                random_state=self.random_state
            )
            # Convert to regression by making target continuous
            y = y.astype(float) + np.random.RandomState(self.random_state).normal(0, 0.1, len(y))
            return X, y
        
        elif dataset_name == "synthetic_nonlinear" and task == "regression":
            X = np.random.RandomState(self.random_state).uniform(-1, 1, (1000, 5))
            y = (X[:, 0] ** 2 + np.sin(X[:, 1] * 3) + X[:, 2] * X[:, 3] + 
                 0.1 * np.random.RandomState(self.random_state).normal(0, 1, 1000))
            return X, y
        
        else:
            raise ValueError(f"Unknown dataset: {dataset_name} for task {task}")
    
    def _compute_summary_stats(self):
        """Compute summary statistics across all benchmark results."""
        searcher_stats = {}
        
        for dataset_name, dataset_results in self.results["dataset_results"].items():
            for searcher_name, searcher_result in dataset_results["searcher_results"].items():
                if "error" in searcher_result:
                    continue  # Skip failed experiments
                
                if searcher_name not in searcher_stats:
                    searcher_stats[searcher_name] = {
                        "test_scores": [],
                        "best_scores": [],
                        "experiment_times": [],
                        "n_experiments": 0
                    }
                
                stats = searcher_stats[searcher_name]
                stats["test_scores"].append(searcher_result["test_score"])
                stats["best_scores"].append(searcher_result["best_score"])
                stats["experiment_times"].append(searcher_result["experiment_time"])
                stats["n_experiments"] += 1
        
        # Compute aggregated statistics
        for searcher_name, stats in searcher_stats.items():
            if stats["n_experiments"] > 0:
                searcher_stats[searcher_name].update({
                    "mean_test_score": np.mean(stats["test_scores"]),
                    "std_test_score": np.std(stats["test_scores"]),
                    "mean_best_score": np.mean(stats["best_scores"]),
                    "std_best_score": np.std(stats["best_scores"]),
                    "mean_time": np.mean(stats["experiment_times"]),
                    "std_time": np.std(stats["experiment_times"])
                })
        
        self.results["summary_stats"] = searcher_stats
    
    def _save_results(self):
        """Save benchmark results to files."""
        # Save main results as JSON
        results_file = self.output_dir / "benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save summary as CSV
        if self.results["summary_stats"]:
            summary_df = pd.DataFrame(self.results["summary_stats"]).T
            summary_file = self.output_dir / "benchmark_summary.csv"
            summary_df.to_csv(summary_file)
        
        self.logger.info(f"Results saved to {self.output_dir}")
    
    def generate_report(self) -> str:
        """
        Generate a human-readable benchmark report.
        
        Returns:
            String containing formatted report
        """
        if not self.results["summary_stats"]:
            return "No benchmark results available. Run a benchmark first."
        
        report = []
        report.append("=" * 60)
        report.append("MLTeammate AutoML Benchmark Report")
        report.append("=" * 60)
        
        # Benchmark info
        info = self.results["benchmark_info"]
        report.append(f"Task: {info['task']}")
        report.append(f"Datasets: {len(info['datasets'])}")
        report.append(f"Searchers: {', '.join(info['searchers'])}")
        report.append(f"Learners: {', '.join(info['learners'])}")
        report.append(f"Trials per experiment: {info['n_trials']}")
        report.append(f"Total time: {info.get('total_time', 0):.1f}s")
        report.append("")
        
        # Summary statistics
        report.append("Summary Statistics:")
        report.append("-" * 30)
        
        for searcher, stats in self.results["summary_stats"].items():
            if stats["n_experiments"] > 0:
                report.append(f"{searcher}:")
                report.append(f"  Mean test score: {stats['mean_test_score']:.4f} ± {stats['std_test_score']:.4f}")
                report.append(f"  Mean time: {stats['mean_time']:.1f}s ± {stats['std_time']:.1f}s")
                report.append(f"  Experiments: {stats['n_experiments']}")
                report.append("")
        
        return "\n".join(report)


def run_quick_benchmark(task: str = "classification", n_trials: int = 5) -> Dict[str, Any]:
    """
    Run a quick benchmark for testing purposes.
    
    Args:
        task: Task type ("classification" or "regression")
        n_trials: Number of trials per experiment
        
    Returns:
        Benchmark results
    """
    runner = BenchmarkRunner(output_dir=f"./quick_benchmark_{task}")
    
    if task == "classification":
        return runner.run_classification_benchmark(
            n_trials=n_trials,
            searchers=["random"],
            datasets=["synthetic_easy"]
        )
    else:
        return runner.run_regression_benchmark(
            n_trials=n_trials,
            searchers=["random"],
            datasets=["synthetic_linear"]
        )


if __name__ == "__main__":
    # Example usage
    runner = BenchmarkRunner()
    results = runner.run_classification_benchmark(n_trials=5)
    print(runner.generate_report())
