"""
Error handling tests for MLTeammate.

This test suite verifies that the system handles errors gracefully including:
- Invalid input data
- Missing dependencies
- Configuration errors
- Resource constraints
- Edge cases
- Clear error messages
"""

import numpy as np
import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from sklearn.datasets import make_classification, make_regression

from ml_teammate.interface.simple_api import SimpleAutoML
from ml_teammate.automl.controller import AutoMLController
from ml_teammate.search.optuna_search import OptunaSearcher
from ml_teammate.learners.registry import get_learner, get_config_space
from ml_teammate.automl.callbacks import LoggerCallback, ProgressCallback


class TestDataValidationErrors:
    """Test handling of invalid data inputs."""
    
    def test_empty_data(self):
        """Test handling of empty datasets."""
        automl = SimpleAutoML(
            learners=["random_forest"],
            task="classification",
            n_trials=5
        )
        
        # Test with empty X
        with pytest.raises(ValueError, match="Empty dataset"):
            automl.fit(np.array([]), np.array([1, 2, 3]))
        
        # Test with empty y
        with pytest.raises(ValueError, match="Empty dataset"):
            automl.fit(np.array([[1, 2], [3, 4]]), np.array([]))
        
        # Test with both empty
        with pytest.raises(ValueError, match="Empty dataset"):
            automl.fit(np.array([]), np.array([]))
    
    def test_single_sample_data(self):
        """Test handling of single sample datasets."""
        automl = SimpleAutoML(
            learners=["random_forest"],
            task="classification",
            n_trials=5
        )
        
        # Test with single sample
        with pytest.raises(ValueError, match="Insufficient samples"):
            automl.fit(np.array([[1, 2, 3]]), np.array([0]))
    
    def test_mismatched_dimensions(self):
        """Test handling of mismatched X and y dimensions."""
        automl = SimpleAutoML(
            learners=["random_forest"],
            task="classification",
            n_trials=5
        )
        
        # Test with more samples in y than X
        with pytest.raises(ValueError, match="Mismatched dimensions"):
            automl.fit(np.array([[1, 2], [3, 4]]), np.array([0, 1, 2]))
        
        # Test with more samples in X than y
        with pytest.raises(ValueError, match="Mismatched dimensions"):
            automl.fit(np.array([[1, 2], [3, 4], [5, 6]]), np.array([0, 1]))
    
    def test_invalid_data_types(self):
        """Test handling of invalid data types."""
        automl = SimpleAutoML(
            learners=["random_forest"],
            task="classification",
            n_trials=5
        )
        
        # Test with non-numeric data
        with pytest.raises(ValueError, match="Invalid data type"):
            automl.fit([["a", "b"], ["c", "d"]], [0, 1])
        
        # Test with None values
        with pytest.raises(ValueError, match="Invalid data type"):
            automl.fit(None, [0, 1])
        
        # Test with string instead of array
        with pytest.raises(ValueError, match="Invalid data type"):
            automl.fit("invalid", [0, 1])
    
    def test_nan_values(self):
        """Test handling of NaN values in data."""
        automl = SimpleAutoML(
            learners=["random_forest"],
            task="classification",
            n_trials=5
        )
        
        # Test with NaN in X
        X_with_nan = np.array([[1, 2], [3, np.nan], [5, 6]])
        y = np.array([0, 1, 0])
        
        with pytest.raises(ValueError, match="NaN values detected"):
            automl.fit(X_with_nan, y)
        
        # Test with NaN in y
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y_with_nan = np.array([0, np.nan, 0])
        
        with pytest.raises(ValueError, match="NaN values detected"):
            automl.fit(X, y_with_nan)
    
    def test_infinite_values(self):
        """Test handling of infinite values in data."""
        automl = SimpleAutoML(
            learners=["random_forest"],
            task="classification",
            n_trials=5
        )
        
        # Test with inf in X
        X_with_inf = np.array([[1, 2], [3, np.inf], [5, 6]])
        y = np.array([0, 1, 0])
        
        with pytest.raises(ValueError, match="Infinite values detected"):
            automl.fit(X_with_inf, y)


class TestConfigurationErrors:
    """Test handling of configuration errors."""
    
    def test_invalid_task_type(self):
        """Test handling of invalid task types."""
        with pytest.raises(ValueError, match="Task must be 'classification' or 'regression'"):
            SimpleAutoML(
                learners=["random_forest"],
                task="invalid_task",
                n_trials=5
            )
    
    def test_invalid_n_trials(self):
        """Test handling of invalid n_trials values."""
        # Test with zero trials
        with pytest.raises(ValueError, match="n_trials must be positive"):
            SimpleAutoML(
                learners=["random_forest"],
                task="classification",
                n_trials=0
            )
        
        # Test with negative trials
        with pytest.raises(ValueError, match="n_trials must be positive"):
            SimpleAutoML(
                learners=["random_forest"],
                task="classification",
                n_trials=-5
            )
        
        # Test with non-integer trials
        with pytest.raises(ValueError, match="n_trials must be an integer"):
            SimpleAutoML(
                learners=["random_forest"],
                task="classification",
                n_trials=5.5
            )
    
    def test_invalid_cv_folds(self):
        """Test handling of invalid CV fold values."""
        # Test with zero CV folds
        with pytest.raises(ValueError, match="cv must be positive"):
            SimpleAutoML(
                learners=["random_forest"],
                task="classification",
                n_trials=5,
                cv=0
            )
        
        # Test with negative CV folds
        with pytest.raises(ValueError, match="cv must be positive"):
            SimpleAutoML(
                learners=["random_forest"],
                task="classification",
                n_trials=5,
                cv=-3
            )
        
        # Test with CV folds larger than dataset
        X, y = make_classification(n_samples=10, n_features=5, random_state=42)
        
        with pytest.raises(ValueError, match="cv folds cannot exceed dataset size"):
            automl = SimpleAutoML(
                learners=["random_forest"],
                task="classification",
                n_trials=5,
                cv=15
            )
            automl.fit(X, y)
    
    def test_empty_learners_list(self):
        """Test handling of empty learners list."""
        with pytest.raises(ValueError, match="At least one learner must be specified"):
            SimpleAutoML(
                learners=[],
                task="classification",
                n_trials=5
            )
    
    def test_invalid_learner_names(self):
        """Test handling of invalid learner names."""
        with pytest.raises(ValueError, match="Unknown learner"):
            SimpleAutoML(
                learners=["invalid_learner"],
                task="classification",
                n_trials=5
            )
    
    def test_mixed_task_learners(self):
        """Test handling of learners that don't match the task."""
        # Test classification learners for regression task
        with pytest.raises(ValueError, match="Learner.*not compatible with.*regression"):
            SimpleAutoML(
                learners=["random_forest"],  # Classification learner
                task="regression",
                n_trials=5
            )
        
        # Test regression learners for classification task
        with pytest.raises(ValueError, match="Learner.*not compatible with.*classification"):
            SimpleAutoML(
                learners=["linear_regression"],  # Regression learner
                task="classification",
                n_trials=5
            )


class TestDependencyErrors:
    """Test handling of missing dependencies."""
    
    def test_missing_sklearn(self):
        """Test handling of missing sklearn dependency."""
        with patch('ml_teammate.learners.registry.sklearn', None):
            with pytest.raises(ImportError, match="scikit-learn is required"):
                get_learner("random_forest")
    
    def test_missing_optuna(self):
        """Test handling of missing optuna dependency."""
        with patch('ml_teammate.search.optuna_search.optuna', None):
            with pytest.raises(ImportError, match="optuna is required"):
                OptunaSearcher({"test": {}})
    
    def test_missing_flaml(self):
        """Test handling of missing flaml dependency."""
        with patch('ml_teammate.search.flaml_search.flaml', None):
            with pytest.raises(ImportError, match="flaml is required"):
                from ml_teammate.search.flaml_search import FLAMLSearcher
                FLAMLSearcher({"test": {}})
    
    def test_missing_mlflow(self):
        """Test handling of missing mlflow dependency."""
        with patch('ml_teammate.experiments.mlflow_helper.mlflow', None):
            with pytest.raises(ImportError, match="mlflow is required"):
                from ml_teammate.experiments.mlflow_helper import MLflowHelper
                MLflowHelper()
    
    def test_missing_xgboost(self):
        """Test handling of missing xgboost dependency."""
        with patch('ml_teammate.learners.xgboost_learner.xgb', None):
            with pytest.raises(ImportError, match="xgboost is required"):
                get_learner("xgboost")
    
    def test_missing_lightgbm(self):
        """Test handling of missing lightgbm dependency."""
        with patch('ml_teammate.learners.lightgbm_learner.lgb', None):
            with pytest.raises(ImportError, match="lightgbm is required"):
                get_learner("lightgbm")


class TestResourceErrors:
    """Test handling of resource constraint errors."""
    
    def test_insufficient_memory(self):
        """Test handling of insufficient memory."""
        # Create a very large dataset that might cause memory issues
        X, y = make_classification(n_samples=100000, n_features=1000, random_state=42)
        
        # Mock memory check to simulate low memory
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.available = 100 * 1024 * 1024  # 100MB available
            
            automl = SimpleAutoML(
                learners=["random_forest"],
                task="classification",
                n_trials=5
            )
            
            with pytest.raises(MemoryError, match="Insufficient memory"):
                automl.fit(X, y)
    
    def test_disk_space_error(self):
        """Test handling of insufficient disk space for artifacts."""
        # Mock disk space check
        with patch('shutil.disk_usage') as mock_disk:
            mock_disk.return_value.free = 1024  # 1KB free space
            
            automl = SimpleAutoML(
                learners=["random_forest"],
                task="classification",
                n_trials=5
            )
            
            X, y = make_classification(n_samples=100, n_features=10, random_state=42)
            
            # Should handle gracefully without crashing
            automl.fit(X, y)
    
    def test_file_permission_error(self):
        """Test handling of file permission errors."""
        # Create a read-only directory
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chmod(temp_dir, 0o444)  # Read-only
            
            automl = SimpleAutoML(
                learners=["random_forest"],
                task="classification",
                n_trials=5
            )
            
            X, y = make_classification(n_samples=100, n_features=10, random_state=42)
            
            # Should handle gracefully without crashing
            automl.fit(X, y)


class TestModelTrainingErrors:
    """Test handling of model training errors."""
    
    def test_model_failure_during_training(self):
        """Test handling of model failures during training."""
        # Create a learner that always fails
        def failing_learner_factory(config):
            def failing_model():
                raise RuntimeError("Model training failed")
            return failing_model()
        
        # Mock the registry to return failing learner
        with patch('ml_teammate.learners.registry.get_learner') as mock_get_learner:
            mock_get_learner.return_value = failing_learner_factory
            
            automl = SimpleAutoML(
                learners=["random_forest"],
                task="classification",
                n_trials=5
            )
            
            X, y = make_classification(n_samples=100, n_features=10, random_state=42)
            
            # Should handle gracefully and continue with other trials
            with pytest.raises(RuntimeError, match="All trials failed"):
                automl.fit(X, y)
    
    def test_invalid_hyperparameters(self):
        """Test handling of invalid hyperparameters."""
        # Create a learner that fails with certain hyperparameters
        def sensitive_learner_factory(config):
            if config.get('max_depth', 0) > 100:
                raise ValueError("max_depth too large")
            return Mock()
        
        # Mock the registry to return sensitive learner
        with patch('ml_teammate.learners.registry.get_learner') as mock_get_learner:
            mock_get_learner.return_value = sensitive_learner_factory
            
            automl = SimpleAutoML(
                learners=["random_forest"],
                task="classification",
                n_trials=5
            )
            
            X, y = make_classification(n_samples=100, n_features=10, random_state=42)
            
            # Should handle gracefully
            automl.fit(X, y)
    
    def test_convergence_failure(self):
        """Test handling of model convergence failures."""
        # Create a learner that fails to converge
        def non_converging_learner_factory(config):
            model = Mock()
            model.fit.side_effect = RuntimeError("Failed to converge")
            return model
        
        # Mock the registry to return non-converging learner
        with patch('ml_teammate.learners.registry.get_learner') as mock_get_learner:
            mock_get_learner.return_value = non_converging_learner_factory
            
            automl = SimpleAutoML(
                learners=["random_forest"],
                task="classification",
                n_trials=5
            )
            
            X, y = make_classification(n_samples=100, n_features=10, random_state=42)
            
            # Should handle gracefully
            with pytest.raises(RuntimeError, match="All trials failed"):
                automl.fit(X, y)


class TestCallbackErrors:
    """Test handling of callback errors."""
    
    def test_callback_exception(self):
        """Test handling of callback exceptions."""
        # Create a callback that raises an exception
        class FailingCallback:
            def on_experiment_start(self, config):
                raise RuntimeError("Callback failed")
        
        automl = SimpleAutoML(
            learners=["random_forest"],
            task="classification",
            n_trials=5,
            callbacks=[FailingCallback()]
        )
        
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        # Should handle gracefully and continue
        automl.fit(X, y)
    
    def test_mlflow_connection_error(self):
        """Test handling of MLflow connection errors."""
        # Mock MLflow to raise connection error
        with patch('ml_teammate.experiments.mlflow_helper.mlflow') as mock_mlflow:
            mock_mlflow.set_experiment.side_effect = Exception("Connection failed")
            
            automl = SimpleAutoML(
                learners=["random_forest"],
                task="classification",
                n_trials=5
            )
            
            X, y = make_classification(n_samples=100, n_features=10, random_state=42)
            
            # Should handle gracefully and continue without MLflow
            automl.fit(X, y)


class TestEdgeCases:
    """Test handling of edge cases."""
    
    def test_single_class_data(self):
        """Test handling of single class data."""
        # Create data with only one class
        X = np.random.randn(100, 10)
        y = np.zeros(100)  # All samples belong to class 0
        
        automl = SimpleAutoML(
            learners=["random_forest"],
            task="classification",
            n_trials=5
        )
        
        # Should handle gracefully
        automl.fit(X, y)
    
    def test_constant_features(self):
        """Test handling of constant features."""
        # Create data with constant features
        X = np.ones((100, 10))  # All features are constant
        y = np.random.randint(0, 2, 100)
        
        automl = SimpleAutoML(
            learners=["random_forest"],
            task="classification",
            n_trials=5
        )
        
        # Should handle gracefully
        automl.fit(X, y)
    
    def test_extremely_imbalanced_data(self):
        """Test handling of extremely imbalanced data."""
        # Create highly imbalanced data
        X = np.random.randn(1000, 10)
        y = np.zeros(1000)
        y[:10] = 1  # Only 10 positive samples out of 1000
        
        automl = SimpleAutoML(
            learners=["random_forest"],
            task="classification",
            n_trials=5
        )
        
        # Should handle gracefully
        automl.fit(X, y)
    
    def test_very_high_dimensional_data(self):
        """Test handling of very high dimensional data."""
        # Create high dimensional data
        X = np.random.randn(100, 10000)  # 10000 features
        y = np.random.randint(0, 2, 100)
        
        automl = SimpleAutoML(
            learners=["random_forest"],
            task="classification",
            n_trials=5
        )
        
        # Should handle gracefully (though may be slow)
        automl.fit(X, y)
    
    def test_very_large_numbers(self):
        """Test handling of very large numbers in data."""
        # Create data with very large numbers
        X = np.array([[1e10, 1e15], [1e12, 1e14], [1e11, 1e13]])
        y = np.array([0, 1, 0])
        
        automl = SimpleAutoML(
            learners=["random_forest"],
            task="classification",
            n_trials=5
        )
        
        # Should handle gracefully
        automl.fit(X, y)


class TestErrorMessageQuality:
    """Test quality of error messages."""
    
    def test_clear_error_messages(self):
        """Test that error messages are clear and helpful."""
        # Test empty data error message
        try:
            automl = SimpleAutoML(learners=["random_forest"], task="classification", n_trials=5)
            automl.fit(np.array([]), np.array([]))
        except ValueError as e:
            assert "Empty dataset" in str(e)
            assert "Please provide non-empty data" in str(e)
        
        # Test invalid task error message
        try:
            SimpleAutoML(learners=["random_forest"], task="invalid", n_trials=5)
        except ValueError as e:
            assert "Task must be 'classification' or 'regression'" in str(e)
        
        # Test invalid learner error message
        try:
            SimpleAutoML(learners=["invalid_learner"], task="classification", n_trials=5)
        except ValueError as e:
            assert "Unknown learner" in str(e)
            assert "invalid_learner" in str(e)
    
    def test_suggestions_in_error_messages(self):
        """Test that error messages include helpful suggestions."""
        # Test missing dependency error message
        with patch('ml_teammate.learners.registry.sklearn', None):
            try:
                get_learner("random_forest")
            except ImportError as e:
                assert "scikit-learn is required" in str(e)
                assert "pip install scikit-learn" in str(e)
        
        # Test invalid configuration error message
        try:
            SimpleAutoML(learners=["random_forest"], task="classification", n_trials=0)
        except ValueError as e:
            assert "n_trials must be positive" in str(e)
            assert "Use a value greater than 0" in str(e)


class TestRecoveryMechanisms:
    """Test recovery mechanisms and graceful degradation."""
    
    def test_partial_failure_recovery(self):
        """Test recovery from partial failures."""
        # Create some learners that fail and some that succeed
        def mixed_learner_factory(learner_name):
            if learner_name == "failing_learner":
                def failing_factory(config):
                    raise RuntimeError("This learner always fails")
                return failing_factory
            else:
                def working_factory(config):
                    return Mock()
                return working_factory
        
        # Mock the registry to return mixed learners
        with patch('ml_teammate.learners.registry.get_learner') as mock_get_learner:
            mock_get_learner.side_effect = mixed_learner_factory
            
            automl = SimpleAutoML(
                learners=["random_forest", "failing_learner"],
                task="classification",
                n_trials=5
            )
            
            X, y = make_classification(n_samples=100, n_features=10, random_state=42)
            
            # Should recover and continue with working learners
            automl.fit(X, y)
    
    def test_graceful_degradation(self):
        """Test graceful degradation when features are unavailable."""
        # Test without MLflow
        with patch('ml_teammate.experiments.mlflow_helper.mlflow', None):
            automl = SimpleAutoML(
                learners=["random_forest"],
                task="classification",
                n_trials=5
            )
            
            X, y = make_classification(n_samples=100, n_features=10, random_state=42)
            
            # Should work without MLflow
            automl.fit(X, y)
        
        # Test without advanced search features
        with patch('ml_teammate.search.optuna_search.optuna', None):
            automl = SimpleAutoML(
                learners=["random_forest"],
                task="classification",
                n_trials=5
            )
            
            X, y = make_classification(n_samples=100, n_features=10, random_state=42)
            
            # Should fall back to basic search
            automl.fit(X, y) 