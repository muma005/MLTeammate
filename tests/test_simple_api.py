"""
Test the simplified API functionality.

This test suite verifies that the new simplified API works correctly
and that users can use MLTeammate without writing custom code.
"""

import sys
from pathlib import Path

# Ensure we're using the correct MLTeammate-1 directory
current_dir = Path(__file__).parent.parent
sys.path = [p for p in sys.path if 'MLTeammate' not in p]  # Remove any MLTeammate paths
sys.path.insert(0, str(current_dir))  # Add our current directory first

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from ml_teammate.interface.simple_api import (
    SimpleAutoML,
    quick_classification,
    quick_regression,
    get_available_learners_by_task,
    get_learner_info
)
from ml_teammate.learners.registry import get_learner_registry


class TestSimpleAPI:
    """Test the simplified API functionality."""
    
    def test_list_available_learners(self):
        """Test that we can list available learners."""
        learners = get_available_learners_by_task()
        
        assert "classification" in learners
        assert "regression" in learners
        assert "all" in learners
        
        # Check that we have some learners
        assert len(learners["all"]) > 0
        assert len(learners["classification"]) > 0
        assert len(learners["regression"]) > 0
        
        # Check that common learners are available
        assert "random_forest" in learners["classification"]
        assert "logistic_regression" in learners["classification"]
        assert "xgboost" in learners["classification"]
        assert "random_forest_regressor" in learners["regression"]
        assert "linear_regression" in learners["regression"]
    
    def test_get_learner_info(self):
        """Test getting information about learners."""
        # Test classification learner
        rf_info = get_learner_info("random_forest")
        assert "error" not in rf_info
        assert rf_info["name"] == "random_forest"
        assert rf_info["is_classification"] is True
        assert rf_info["is_regression"] is False
        assert len(rf_info["parameters"]) > 0
        
        # Test regression learner
        lr_info = get_learner_info("linear_regression")
        assert "error" not in lr_info
        assert lr_info["name"] == "linear_regression"
        assert lr_info["is_classification"] is False
        assert lr_info["is_regression"] is True
        
        # Test invalid learner
        invalid_info = get_learner_info("invalid_learner")
        assert "error" in invalid_info
    
    def test_simple_automl_classification(self):
        """Test SimpleAutoML with classification task."""
        # Generate test data
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create SimpleAutoML instance
        automl = SimpleAutoML(
            learners=["random_forest", "logistic_regression"],
            task="classification",
            n_trials=2,
            cv=2
        )
        
        # Fit the model
        automl.fit(X_train, y_train)
        
        # Check that we have results
        assert automl.best_score is not None
        assert automl.best_model is not None
        assert automl.best_config is not None
        
        # Make predictions
        y_pred = automl.predict(X_test)
        assert len(y_pred) == len(y_test)
        
        # Check score
        score = automl.score(X_test, y_test)
        assert isinstance(score, (int, float))
        assert 0 <= score <= 1  # Accuracy should be between 0 and 1
    
    def test_simple_automl_regression(self):
        """Test SimpleAutoML with regression task."""
        # Generate test data
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create SimpleAutoML instance
        automl = SimpleAutoML(
            learners=["random_forest_regressor", "linear_regression"],
            task="regression",
            n_trials=2,
            cv=2
        )
        
        # Fit the model
        automl.fit(X_train, y_train)
        
        # Check that we have results
        assert automl.best_score is not None
        assert automl.best_model is not None
        assert automl.best_config is not None
        
        # Make predictions
        y_pred = automl.predict(X_test)
        assert len(y_pred) == len(y_test)
    
    def test_single_learner(self):
        """Test SimpleAutoML with a single learner."""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Use single learner as string
        automl = SimpleAutoML(
            learners="random_forest",
            task="classification",
            n_trials=2,
            cv=2
        )
        
        automl.fit(X_train, y_train)
        assert automl.best_score is not None
        assert automl.best_model is not None
    
    def test_quick_classification(self):
        """Test the quick_classification function."""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Use quick classification
        automl = quick_classification(
            X_train, y_train,
            learners=["random_forest"],
            n_trials=2,
            cv=2
        )
        
        assert automl.best_score is not None
        assert automl.best_model is not None
        
        # Test predictions
        y_pred = automl.predict(X_test)
        assert len(y_pred) == len(y_test)
    
    def test_quick_regression(self):
        """Test the quick_regression function."""
        X, y = make_regression(n_samples=50, n_features=5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Use quick regression
        automl = quick_regression(
            X_train, y_train,
            learners=["random_forest_regressor"],
            n_trials=2,
            cv=2
        )
        
        assert automl.best_score is not None
        assert automl.best_model is not None
        
        # Test predictions
        y_pred = automl.predict(X_test)
        assert len(y_pred) == len(y_test)
    
    def test_results_summary(self):
        """Test getting results summary."""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        automl = SimpleAutoML(
            learners=["random_forest"],
            task="classification",
            n_trials=2,
            cv=2
        )
        
        automl.fit(X_train, y_train)
        
        summary = automl.get_results_summary()
        assert summary is not None
        assert "task" in summary
        assert "learners_used" in summary
        assert "n_trials" in summary
        assert "best_score" in summary
        assert summary["task"] == "classification"
        assert "random_forest" in summary["learners_used"]
    
    def test_invalid_learner(self):
        """Test that invalid learners raise appropriate errors."""
        with pytest.raises(ValueError):
            SimpleAutoML(learners=["invalid_learner"])
    
    def test_default_learners(self):
        """Test that default learners are used when none specified."""
        # Get registry for validation
        registry = get_learner_registry()
        
        # Classification defaults
        automl_clf = SimpleAutoML(task="classification")
        assert len(automl_clf.learner_names) > 0
        # Validate learners exist in registry
        for learner_name in automl_clf.learner_names:
            assert registry.has_learner(learner_name), f"Learner {learner_name} not found in registry"
        
        # Regression defaults
        automl_reg = SimpleAutoML(task="regression")
        assert len(automl_reg.learner_names) > 0
        # Validate learners exist in registry
        for learner_name in automl_reg.learner_names:
            assert registry.has_learner(learner_name), f"Learner {learner_name} not found in registry"


if __name__ == "__main__":
    # Run a quick test
    print("🧪 Testing Simple API...")
    
    test = TestSimpleAPI()
    
    # Test basic functionality
    test.test_list_available_learners()
    print("✅ Available learners test passed")
    
    test.test_get_learner_info()
    print("✅ Learner info test passed")
    
    test.test_simple_automl_classification()
    print("✅ Classification test passed")
    
    test.test_simple_automl_regression()
    print("✅ Regression test passed")
    
    print("🎉 All tests passed!") 