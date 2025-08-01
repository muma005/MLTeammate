"""
Performance benchmarks for MLTeammate.

This test suite measures performance characteristics including:
- Execution time for different dataset sizes
- Memory usage patterns
- Scalability with number of trials
- Search algorithm performance comparison
- Cross-validation performance impact
"""

import numpy as np
import pytest
import time
import psutil
import os
from unittest.mock import patch, MagicMock
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from ml_teammate.interface.simple_api import SimpleAutoML
from ml_teammate.search.optuna_search import OptunaSearcher
from ml_teammate.search.flaml_search import FLAMLSearcher
from ml_teammate.automl.controller import AutoMLController
from ml_teammate.learners.registry import create_learners_dict, create_config_space


class PerformanceBenchmark:
    """Base class for performance benchmarks."""
    
    def setup_method(self):
        """Set up benchmark fixtures."""
        self.process = psutil.Process(os.getpid())
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
    
    def measure_execution_time(self, func, *args, **kwargs):
        """Measure execution time of a function."""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return end_time - start_time, result
    
    def measure_memory_usage(self, func, *args, **kwargs):
        """Measure memory usage of a function."""
        # Force garbage collection
        import gc
        gc.collect()
        
        # Measure baseline memory
        baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Measure peak memory
        peak_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Force garbage collection again
        gc.collect()
        
        # Measure final memory
        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'baseline_memory': baseline_memory,
            'peak_memory': peak_memory,
            'final_memory': final_memory,
            'memory_increase': peak_memory - baseline_memory,
            'memory_retained': final_memory - baseline_memory,
            'result': result
        }


class TestDatasetSizePerformance(PerformanceBenchmark):
    """Test performance with different dataset sizes."""
    
    def test_small_dataset_performance(self):
        """Test performance with small dataset (100 samples)."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        automl = SimpleAutoML(
            learners=["random_forest", "logistic_regression"],
            task="classification",
            n_trials=5,
            cv=3
        )
        
        # Measure execution time
        execution_time, _ = self.measure_execution_time(automl.fit, X_train, y_train)
        
        # Measure memory usage
        memory_info = self.measure_memory_usage(automl.fit, X_train, y_train)
        
        # Assertions for small dataset
        assert execution_time < 30  # Should complete in under 30 seconds
        assert memory_info['memory_increase'] < 100  # Should use less than 100MB additional memory
        
        print(f"Small dataset performance:")
        print(f"  Execution time: {execution_time:.2f} seconds")
        print(f"  Memory increase: {memory_info['memory_increase']:.2f} MB")
    
    def test_medium_dataset_performance(self):
        """Test performance with medium dataset (1000 samples)."""
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        automl = SimpleAutoML(
            learners=["random_forest", "logistic_regression"],
            task="classification",
            n_trials=10,
            cv=5
        )
        
        # Measure execution time
        execution_time, _ = self.measure_execution_time(automl.fit, X_train, y_train)
        
        # Measure memory usage
        memory_info = self.measure_memory_usage(automl.fit, X_train, y_train)
        
        # Assertions for medium dataset
        assert execution_time < 120  # Should complete in under 2 minutes
        assert memory_info['memory_increase'] < 200  # Should use less than 200MB additional memory
        
        print(f"Medium dataset performance:")
        print(f"  Execution time: {execution_time:.2f} seconds")
        print(f"  Memory increase: {memory_info['memory_increase']:.2f} MB")
    
    def test_large_dataset_performance(self):
        """Test performance with large dataset (10000 samples)."""
        X, y = make_classification(n_samples=10000, n_features=50, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        automl = SimpleAutoML(
            learners=["random_forest", "logistic_regression"],
            task="classification",
            n_trials=15,
            cv=5
        )
        
        # Measure execution time
        execution_time, _ = self.measure_execution_time(automl.fit, X_train, y_train)
        
        # Measure memory usage
        memory_info = self.measure_memory_usage(automl.fit, X_train, y_train)
        
        # Assertions for large dataset
        assert execution_time < 600  # Should complete in under 10 minutes
        assert memory_info['memory_increase'] < 500  # Should use less than 500MB additional memory
        
        print(f"Large dataset performance:")
        print(f"  Execution time: {execution_time:.2f} seconds")
        print(f"  Memory increase: {memory_info['memory_increase']:.2f} MB")
    
    def test_dataset_size_scalability(self):
        """Test scalability with increasing dataset sizes."""
        sizes = [100, 500, 1000, 2000]
        execution_times = []
        memory_usage = []
        
        for size in sizes:
            X, y = make_classification(n_samples=size, n_features=10, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            automl = SimpleAutoML(
                learners=["random_forest"],
                task="classification",
                n_trials=5,
                cv=3
            )
            
            # Measure performance
            execution_time, _ = self.measure_execution_time(automl.fit, X_train, y_train)
            memory_info = self.measure_memory_usage(automl.fit, X_train, y_train)
            
            execution_times.append(execution_time)
            memory_usage.append(memory_info['memory_increase'])
        
        # Check that performance scales reasonably
        # Execution time should not grow faster than O(n log n)
        for i in range(1, len(execution_times)):
            ratio = execution_times[i] / execution_times[i-1]
            size_ratio = sizes[i] / sizes[i-1]
            # Allow some overhead but not exponential growth
            assert ratio < size_ratio * 2
        
        print(f"Dataset size scalability:")
        for i, size in enumerate(sizes):
            print(f"  Size {size}: {execution_times[i]:.2f}s, {memory_usage[i]:.2f}MB")


class TestTrialCountPerformance(PerformanceBenchmark):
    """Test performance with different numbers of trials."""
    
    def test_few_trials_performance(self):
        """Test performance with few trials (5)."""
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        automl = SimpleAutoML(
            learners=["random_forest", "logistic_regression"],
            task="classification",
            n_trials=5,
            cv=3
        )
        
        # Measure execution time
        execution_time, _ = self.measure_execution_time(automl.fit, X_train, y_train)
        
        # Measure memory usage
        memory_info = self.measure_memory_usage(automl.fit, X_train, y_train)
        
        # Assertions for few trials
        assert execution_time < 60  # Should complete in under 1 minute
        assert memory_info['memory_increase'] < 150  # Should use less than 150MB additional memory
        
        print(f"Few trials performance:")
        print(f"  Execution time: {execution_time:.2f} seconds")
        print(f"  Memory increase: {memory_info['memory_increase']:.2f} MB")
    
    def test_many_trials_performance(self):
        """Test performance with many trials (50)."""
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        automl = SimpleAutoML(
            learners=["random_forest", "logistic_regression"],
            task="classification",
            n_trials=50,
            cv=3
        )
        
        # Measure execution time
        execution_time, _ = self.measure_execution_time(automl.fit, X_train, y_train)
        
        # Measure memory usage
        memory_info = self.measure_memory_usage(automl.fit, X_train, y_train)
        
        # Assertions for many trials
        assert execution_time < 300  # Should complete in under 5 minutes
        assert memory_info['memory_increase'] < 300  # Should use less than 300MB additional memory
        
        print(f"Many trials performance:")
        print(f"  Execution time: {execution_time:.2f} seconds")
        print(f"  Memory increase: {memory_info['memory_increase']:.2f} MB")
    
    def test_trial_count_scalability(self):
        """Test scalability with increasing trial counts."""
        trial_counts = [5, 10, 20, 30]
        execution_times = []
        memory_usage = []
        
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        for n_trials in trial_counts:
            automl = SimpleAutoML(
                learners=["random_forest"],
                task="classification",
                n_trials=n_trials,
                cv=3
            )
            
            # Measure performance
            execution_time, _ = self.measure_execution_time(automl.fit, X_train, y_train)
            memory_info = self.measure_memory_usage(automl.fit, X_train, y_train)
            
            execution_times.append(execution_time)
            memory_usage.append(memory_info['memory_increase'])
        
        # Check that performance scales linearly with trial count
        for i in range(1, len(execution_times)):
            time_ratio = execution_times[i] / execution_times[i-1]
            trial_ratio = trial_counts[i] / trial_counts[i-1]
            # Allow some overhead but should be roughly linear
            assert 0.5 < time_ratio / trial_ratio < 2.0
        
        print(f"Trial count scalability:")
        for i, n_trials in enumerate(trial_counts):
            print(f"  {n_trials} trials: {execution_times[i]:.2f}s, {memory_usage[i]:.2f}MB")


class TestSearchAlgorithmPerformance(PerformanceBenchmark):
    """Test performance of different search algorithms."""
    
    def test_optuna_performance(self):
        """Test Optuna searcher performance."""
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create learners and config space
        learners = create_learners_dict(["random_forest", "logistic_regression"])
        config_space = create_config_space(["random_forest", "logistic_regression"])
        
        # Create Optuna searcher
        searcher = OptunaSearcher(config_space)
        
        # Create controller
        controller = AutoMLController(
            learners=learners,
            searcher=searcher,
            config_space=config_space,
            task="classification",
            n_trials=10,
            cv=3
        )
        
        # Measure performance
        execution_time, _ = self.measure_execution_time(controller.fit, X_train, y_train)
        memory_info = self.measure_memory_usage(controller.fit, X_train, y_train)
        
        print(f"Optuna performance:")
        print(f"  Execution time: {execution_time:.2f} seconds")
        print(f"  Memory increase: {memory_info['memory_increase']:.2f} MB")
    
    def test_flaml_performance(self):
        """Test FLAML searcher performance."""
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create learners and config space
        learners = create_learners_dict(["random_forest", "logistic_regression"])
        config_space = create_config_space(["random_forest", "logistic_regression"])
        
        # Create FLAML searcher
        searcher = FLAMLSearcher(config_space, time_budget=30)
        
        # Create controller
        controller = AutoMLController(
            learners=learners,
            searcher=searcher,
            config_space=config_space,
            task="classification",
            n_trials=10,
            cv=3
        )
        
        # Measure performance
        execution_time, _ = self.measure_execution_time(controller.fit, X_train, y_train)
        memory_info = self.measure_memory_usage(controller.fit, X_train, y_train)
        
        print(f"FLAML performance:")
        print(f"  Execution time: {execution_time:.2f} seconds")
        print(f"  Memory increase: {memory_info['memory_increase']:.2f} MB")
    
    def test_search_algorithm_comparison(self):
        """Compare performance of different search algorithms."""
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create learners and config space
        learners = create_learners_dict(["random_forest", "logistic_regression"])
        config_space = create_config_space(["random_forest", "logistic_regression"])
        
        results = {}
        
        # Test Optuna
        searcher = OptunaSearcher(config_space)
        controller = AutoMLController(
            learners=learners,
            searcher=searcher,
            config_space=config_space,
            task="classification",
            n_trials=10,
            cv=3
        )
        
        execution_time, _ = self.measure_execution_time(controller.fit, X_train, y_train)
        memory_info = self.measure_memory_usage(controller.fit, X_train, y_train)
        
        results['optuna'] = {
            'execution_time': execution_time,
            'memory_increase': memory_info['memory_increase'],
            'best_score': controller.best_score
        }
        
        # Test FLAML
        searcher = FLAMLSearcher(config_space, time_budget=30)
        controller = AutoMLController(
            learners=learners,
            searcher=searcher,
            config_space=config_space,
            task="classification",
            n_trials=10,
            cv=3
        )
        
        execution_time, _ = self.measure_execution_time(controller.fit, X_train, y_train)
        memory_info = self.measure_memory_usage(controller.fit, X_train, y_train)
        
        results['flaml'] = {
            'execution_time': execution_time,
            'memory_increase': memory_info['memory_increase'],
            'best_score': controller.best_score
        }
        
        print(f"Search algorithm comparison:")
        for algorithm, metrics in results.items():
            print(f"  {algorithm.capitalize()}:")
            print(f"    Execution time: {metrics['execution_time']:.2f} seconds")
            print(f"    Memory increase: {metrics['memory_increase']:.2f} MB")
            print(f"    Best score: {metrics['best_score']:.4f}")


class TestCrossValidationPerformance(PerformanceBenchmark):
    """Test performance impact of cross-validation."""
    
    def test_no_cv_performance(self):
        """Test performance without cross-validation."""
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        automl = SimpleAutoML(
            learners=["random_forest", "logistic_regression"],
            task="classification",
            n_trials=10,
            cv=None
        )
        
        # Measure performance
        execution_time, _ = self.measure_execution_time(automl.fit, X_train, y_train)
        memory_info = self.measure_memory_usage(automl.fit, X_train, y_train)
        
        print(f"No CV performance:")
        print(f"  Execution time: {execution_time:.2f} seconds")
        print(f"  Memory increase: {memory_info['memory_increase']:.2f} MB")
    
    def test_cv_performance(self):
        """Test performance with cross-validation."""
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        automl = SimpleAutoML(
            learners=["random_forest", "logistic_regression"],
            task="classification",
            n_trials=10,
            cv=5
        )
        
        # Measure performance
        execution_time, _ = self.measure_execution_time(automl.fit, X_train, y_train)
        memory_info = self.measure_memory_usage(automl.fit, X_train, y_train)
        
        print(f"CV performance:")
        print(f"  Execution time: {execution_time:.2f} seconds")
        print(f"  Memory increase: {memory_info['memory_increase']:.2f} MB")
    
    def test_cv_fold_scalability(self):
        """Test scalability with different CV fold counts."""
        cv_folds = [None, 3, 5, 10]
        execution_times = []
        memory_usage = []
        
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        for cv in cv_folds:
            automl = SimpleAutoML(
                learners=["random_forest"],
                task="classification",
                n_trials=5,
                cv=cv
            )
            
            # Measure performance
            execution_time, _ = self.measure_execution_time(automl.fit, X_train, y_train)
            memory_info = self.measure_memory_usage(automl.fit, X_train, y_train)
            
            execution_times.append(execution_time)
            memory_usage.append(memory_info['memory_increase'])
        
        print(f"CV fold scalability:")
        for i, cv in enumerate(cv_folds):
            cv_name = "No CV" if cv is None else f"{cv}-fold CV"
            print(f"  {cv_name}: {execution_times[i]:.2f}s, {memory_usage[i]:.2f}MB")


class TestMemoryEfficiency(PerformanceBenchmark):
    """Test memory efficiency characteristics."""
    
    def test_memory_cleanup(self):
        """Test that memory is properly cleaned up after experiments."""
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Measure baseline memory
        baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Run multiple experiments
        for i in range(3):
            automl = SimpleAutoML(
                learners=["random_forest", "logistic_regression"],
                task="classification",
                n_trials=5,
                cv=3
            )
            
            automl.fit(X_train, y_train)
            
            # Force garbage collection
            import gc
            gc.collect()
        
        # Measure final memory
        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Memory should not grow significantly after cleanup
        memory_growth = final_memory - baseline_memory
        assert memory_growth < 100  # Should not retain more than 100MB
        
        print(f"Memory cleanup test:")
        print(f"  Baseline memory: {baseline_memory:.2f} MB")
        print(f"  Final memory: {final_memory:.2f} MB")
        print(f"  Memory growth: {memory_growth:.2f} MB")
    
    def test_large_model_memory(self):
        """Test memory usage with large models."""
        X, y = make_classification(n_samples=2000, n_features=100, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        automl = SimpleAutoML(
            learners=["random_forest"],  # Random Forest can be memory-intensive
            task="classification",
            n_trials=10,
            cv=5
        )
        
        # Measure memory usage
        memory_info = self.measure_memory_usage(automl.fit, X_train, y_train)
        
        # Large models should still be reasonable
        assert memory_info['memory_increase'] < 500  # Should use less than 500MB additional memory
        
        print(f"Large model memory test:")
        print(f"  Memory increase: {memory_info['memory_increase']:.2f} MB")
        print(f"  Peak memory: {memory_info['peak_memory']:.2f} MB")
        print(f"  Memory retained: {memory_info['memory_retained']:.2f} MB") 