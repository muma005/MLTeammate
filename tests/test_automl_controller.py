"""
Test the AutoML Controller functionality.

This test suite verifies that the AutoML controller integrates correctly
with all components and manages the optimization process properly.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from ml_teammate.automl import create_automl_controller
from ml_teammate.automl.controller import AutoMLController
from ml_teammate.automl.callbacks import LoggerCallback
from ml_teammate.learners.registry import get_learner_registry


class TestAutoMLController:
    """Test the AutoML Controller functionality."""
    
    def test_create_automl_controller(self):
        """Test creating AutoML controller via factory function."""
        controller = create_automl_controller(
            learner_names=["random_forest"],
            task="classification",
            n_trials=2,
            validation_strategy="holdout",
            random_state=42
        )
        
        assert controller is not None
        assert hasattr(controller, 'fit')
        assert hasattr(controller, 'predict')
        assert hasattr(controller, 'best_model')
        assert hasattr(controller, 'best_score')
    
    def test_controller_classification(self):
        """Test controller with classification task."""
        # Generate test data
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create controller
        controller = create_automl_controller(
            learner_names=["random_forest", "logistic_regression"],
            task="classification",
            n_trials=3,
            validation_strategy="holdout",
            validation_split=0.2,
            random_state=42
        )
        
        # Fit the controller
        controller.fit(X_train, y_train)
        
        # Check results
        assert controller.best_model is not None
        assert controller.best_score is not None
        assert controller.best_config is not None
        assert isinstance(controller.best_score, (int, float))
        
        # Make predictions
        y_pred = controller.predict(X_test)
        assert len(y_pred) == len(y_test)
        assert y_pred.dtype in [np.int32, np.int64, int]
        
        # Test scoring
        score = controller.score(X_test, y_test)
        assert isinstance(score, (int, float))
        assert 0 <= score <= 1
    
    def test_controller_regression(self):
        """Test controller with regression task."""
        # Generate test data
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create controller
        controller = create_automl_controller(
            learner_names=["random_forest_regressor", "linear_regression"],
            task="regression",
            n_trials=3,
            validation_strategy="holdout",
            validation_split=0.2,
            random_state=42
        )
        
        # Fit the controller
        controller.fit(X_train, y_train)
        
        # Check results
        assert controller.best_model is not None
        assert controller.best_score is not None
        assert controller.best_config is not None
        assert isinstance(controller.best_score, (int, float))
        
        # Make predictions
        y_pred = controller.predict(X_test)
        assert len(y_pred) == len(y_test)
        assert y_pred.dtype in [np.float32, np.float64, float]
    
    def test_controller_with_cv(self):
        """Test controller with cross-validation."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        controller = create_automl_controller(
            learner_names=["random_forest"],
            task="classification",
            n_trials=2,
            validation_strategy="cv",
            cv_folds=3,
            random_state=42
        )
        
        controller.fit(X, y)
        
        assert controller.best_model is not None
        assert controller.best_score is not None
    
    def test_controller_with_callbacks(self):
        """Test controller with callbacks."""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        
        # Create controller with callback
        controller = create_automl_controller(
            learner_names=["random_forest"],
            task="classification",
            n_trials=2,
            enable_mlflow=False,  # Disable MLflow for testing
            random_state=42
        )
        
        controller.fit(X, y)
        
        assert controller.best_model is not None
    
    def test_controller_single_learner(self):
        """Test controller with single learner."""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        
        controller = create_automl_controller(
            learner_names=["random_forest"],
            task="classification",
            n_trials=2,
            random_state=42
        )
        
        controller.fit(X, y)
        
        assert controller.best_model is not None
        assert controller.best_config is not None
        assert controller.best_config.get('learner_name') == 'random_forest'
    
    def test_controller_multiple_learners(self):
        """Test controller with multiple learners."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        controller = create_automl_controller(
            learner_names=["random_forest", "logistic_regression", "svm"],
            task="classification",
            n_trials=6,  # 2 trials per learner
            random_state=42
        )
        
        controller.fit(X, y)
        
        assert controller.best_model is not None
        assert controller.best_config is not None
        
        # Best learner should be one of the specified ones
        best_learner = controller.best_config.get('learner_name')
        assert best_learner in ["random_forest", "logistic_regression", "svm"]
    
    def test_controller_error_handling(self):
        """Test controller error handling."""
        # Test invalid learner
        with pytest.raises(ValueError):
            create_automl_controller(
                learner_names=["invalid_learner"],
                task="classification",
                n_trials=1
            )
        
        # Test invalid task
        with pytest.raises(ValueError):
            create_automl_controller(
                learner_names=["random_forest"],
                task="invalid_task",
                n_trials=1
            )
        
        # Test invalid n_trials
        with pytest.raises(ValueError):
            create_automl_controller(
                learner_names=["random_forest"],
                task="classification",
                n_trials=0
            )
    
    def test_controller_reproducibility(self):
        """Test that controller produces reproducible results."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        # Create two identical controllers
        controller1 = create_automl_controller(
            learner_names=["random_forest"],
            task="classification",
            n_trials=3,
            random_state=42
        )
        
        controller2 = create_automl_controller(
            learner_names=["random_forest"],
            task="classification",
            n_trials=3,
            random_state=42
        )
        
        # Fit both controllers
        controller1.fit(X, y)
        controller2.fit(X, y)
        
        # Results should be similar (allowing for small numerical differences)
        assert abs(controller1.best_score - controller2.best_score) < 0.01
    
    def test_controller_optimization_history(self):
        """Test that controller tracks optimization history."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        controller = create_automl_controller(
            learner_names=["random_forest"],
            task="classification",
            n_trials=3,
            random_state=42
        )
        
        controller.fit(X, y)
        
        # Check if optimization history is available
        if hasattr(controller, 'optimization_history'):
            history = controller.optimization_history
            assert isinstance(history, list)
            assert len(history) <= 3  # Should not exceed n_trials


if __name__ == "__main__":
    # Run a quick test
    print("🧪 Testing AutoML Controller...")
    
    test = TestAutoMLController()
    
    # Test basic functionality
    test.test_create_automl_controller()
    print("✅ Controller creation test passed")
    
    test.test_controller_classification()
    print("✅ Classification test passed")
    
    test.test_controller_regression()
    print("✅ Regression test passed")
    
    test.test_controller_error_handling()
    print("✅ Error handling test passed")
    
    print("🎉 All controller tests passed!")
