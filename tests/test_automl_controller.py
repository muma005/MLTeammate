"""
Unit tests for AutoMLController.

This test suite verifies the core AutoML functionality including:
- Controller initialization and configuration
- Learner management
- Search algorithm integration
- Cross-validation
- Trial execution
- Results management
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ml_teammate.automl.controller import AutoMLController
from ml_teammate.search.optuna_search import OptunaSearcher
from ml_teammate.automl.callbacks import LoggerCallback, ProgressCallback


class TestAutoMLController:
    """Test the AutoMLController class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create test data
        self.X_clf, self.y_clf = make_classification(
            n_samples=100, n_features=10, random_state=42
        )
        self.X_reg, self.y_reg = make_regression(
            n_samples=100, n_features=10, random_state=42
        )
        
        # Create mock learners
        self.mock_learners = {
            "random_forest": Mock(return_value=RandomForestClassifier(random_state=42)),
            "random_forest_regressor": Mock(return_value=RandomForestRegressor(random_state=42))
        }
        
        # Create config spaces
        self.config_space = {
            "random_forest": {
                "n_estimators": {"type": "int", "bounds": [10, 100]},
                "max_depth": {"type": "int", "bounds": [3, 10]}
            },
            "random_forest_regressor": {
                "n_estimators": {"type": "int", "bounds": [10, 100]},
                "max_depth": {"type": "int", "bounds": [3, 10]}
            }
        }
    
    def test_controller_initialization(self):
        """Test controller initialization with valid parameters."""
        searcher = OptunaSearcher(self.config_space)
        callbacks = [LoggerCallback(), ProgressCallback(total_trials=5)]
        
        controller = AutoMLController(
            learners=self.mock_learners,
            searcher=searcher,
            config_space=self.config_space,
            task="classification",
            n_trials=5,
            cv=3,
            callbacks=callbacks
        )
        
        assert controller.learners == self.mock_learners
        assert controller.searcher == searcher
        assert controller.config_space == self.config_space
        assert controller.task == "classification"
        assert controller.n_trials == 5
        assert controller.cv == 3
        assert len(controller.callbacks) == 2
    
    def test_controller_initialization_defaults(self):
        """Test controller initialization with default parameters."""
        searcher = OptunaSearcher(self.config_space)
        
        controller = AutoMLController(
            learners=self.mock_learners,
            searcher=searcher,
            config_space=self.config_space,
            task="classification"
        )
        
        assert controller.n_trials == 10  # Default
        assert controller.cv is None  # Default
        assert controller.callbacks == []  # Default
    
    def test_controller_invalid_task(self):
        """Test controller initialization with invalid task."""
        searcher = OptunaSearcher(self.config_space)
        
        with pytest.raises(ValueError, match="Task must be 'classification' or 'regression'"):
            AutoMLController(
                learners=self.mock_learners,
                searcher=searcher,
                config_space=self.config_space,
                task="invalid_task"
            )
    
    def test_controller_invalid_n_trials(self):
        """Test controller initialization with invalid n_trials."""
        searcher = OptunaSearcher(self.config_space)
        
        with pytest.raises(ValueError, match="n_trials must be positive"):
            AutoMLController(
                learners=self.mock_learners,
                searcher=searcher,
                config_space=self.config_space,
                task="classification",
                n_trials=0
            )
    
    def test_controller_invalid_cv(self):
        """Test controller initialization with invalid cv."""
        searcher = OptunaSearcher(self.config_space)
        
        with pytest.raises(ValueError, match="cv must be positive"):
            AutoMLController(
                learners=self.mock_learners,
                searcher=searcher,
                config_space=self.config_space,
                task="classification",
                cv=0
            )
    
    def test_fit_classification(self):
        """Test fitting the controller for classification task."""
        searcher = OptunaSearcher(self.config_space)
        callbacks = [LoggerCallback(), ProgressCallback(total_trials=3)]
        
        controller = AutoMLController(
            learners=self.mock_learners,
            searcher=searcher,
            config_space=self.config_space,
            task="classification",
            n_trials=3,
            cv=2,
            callbacks=callbacks
        )
        
        # Mock the searcher to return predictable results
        with patch.object(searcher, 'suggest') as mock_suggest:
            mock_suggest.return_value = {
                "learner_name": "random_forest",
                "n_estimators": 50,
                "max_depth": 5
            }
            
            controller.fit(self.X_clf, self.y_clf)
        
        assert controller.best_score is not None
        assert controller.best_model is not None
        assert controller.best_config is not None
    
    def test_fit_regression(self):
        """Test fitting the controller for regression task."""
        searcher = OptunaSearcher(self.config_space)
        callbacks = [LoggerCallback(), ProgressCallback(total_trials=3)]
        
        controller = AutoMLController(
            learners=self.mock_learners,
            searcher=searcher,
            config_space=self.config_space,
            task="regression",
            n_trials=3,
            cv=2,
            callbacks=callbacks
        )
        
        # Mock the searcher to return predictable results
        with patch.object(searcher, 'suggest') as mock_suggest:
            mock_suggest.return_value = {
                "learner_name": "random_forest_regressor",
                "n_estimators": 50,
                "max_depth": 5
            }
            
            controller.fit(self.X_reg, self.y_reg)
        
        assert controller.best_score is not None
        assert controller.best_model is not None
        assert controller.best_config is not None
    
    def test_fit_without_cv(self):
        """Test fitting without cross-validation."""
        searcher = OptunaSearcher(self.config_space)
        
        controller = AutoMLController(
            learners=self.mock_learners,
            searcher=searcher,
            config_space=self.config_space,
            task="classification",
            n_trials=3,
            cv=None,
            callbacks=[]
        )
        
        # Mock the searcher
        with patch.object(searcher, 'suggest') as mock_suggest:
            mock_suggest.return_value = {
                "learner_name": "random_forest",
                "n_estimators": 50,
                "max_depth": 5
            }
            
            controller.fit(self.X_clf, self.y_clf)
        
        assert controller.best_score is not None
    
    def test_predict(self):
        """Test making predictions."""
        searcher = OptunaSearcher(self.config_space)
        
        controller = AutoMLController(
            learners=self.mock_learners,
            searcher=searcher,
            config_space=self.config_space,
            task="classification",
            n_trials=3,
            cv=2
        )
        
        # Mock the searcher and fit
        with patch.object(searcher, 'suggest') as mock_suggest:
            mock_suggest.return_value = {
                "learner_name": "random_forest",
                "n_estimators": 50,
                "max_depth": 5
            }
            
            controller.fit(self.X_clf, self.y_clf)
        
        # Test predictions
        X_test = self.X_clf[:10]
        predictions = controller.predict(X_test)
        
        assert len(predictions) == 10
        assert all(isinstance(pred, (int, np.integer)) for pred in predictions)
    
    def test_score(self):
        """Test scoring the model."""
        searcher = OptunaSearcher(self.config_space)
        
        controller = AutoMLController(
            learners=self.mock_learners,
            searcher=searcher,
            config_space=self.config_space,
            task="classification",
            n_trials=3,
            cv=2
        )
        
        # Mock the searcher and fit
        with patch.object(searcher, 'suggest') as mock_suggest:
            mock_suggest.return_value = {
                "learner_name": "random_forest",
                "n_estimators": 50,
                "max_depth": 5
            }
            
            controller.fit(self.X_clf, self.y_clf)
        
        # Test scoring
        X_test, y_test = self.X_clf[:20], self.y_clf[:20]
        score = controller.score(X_test, y_test)
        
        assert isinstance(score, (int, float))
        assert 0 <= score <= 1  # Accuracy should be between 0 and 1
    
    def test_predict_without_fit(self):
        """Test that predict raises error if model not fitted."""
        searcher = OptunaSearcher(self.config_space)
        
        controller = AutoMLController(
            learners=self.mock_learners,
            searcher=searcher,
            config_space=self.config_space,
            task="classification",
            n_trials=3,
            cv=2
        )
        
        with pytest.raises(ValueError, match="Model not fitted"):
            controller.predict(self.X_clf[:10])
    
    def test_score_without_fit(self):
        """Test that score raises error if model not fitted."""
        searcher = OptunaSearcher(self.config_space)
        
        controller = AutoMLController(
            learners=self.mock_learners,
            searcher=searcher,
            config_space=self.config_space,
            task="classification",
            n_trials=3,
            cv=2
        )
        
        with pytest.raises(ValueError, match="Model not fitted"):
            controller.score(self.X_clf[:10], self.y_clf[:10])
    
    def test_callback_integration(self):
        """Test that callbacks are called during fitting."""
        searcher = OptunaSearcher(self.config_space)
        
        # Create mock callbacks
        mock_callback = Mock()
        callbacks = [mock_callback]
        
        controller = AutoMLController(
            learners=self.mock_learners,
            searcher=searcher,
            config_space=self.config_space,
            task="classification",
            n_trials=3,
            cv=2,
            callbacks=callbacks
        )
        
        # Mock the searcher
        with patch.object(searcher, 'suggest') as mock_suggest:
            mock_suggest.return_value = {
                "learner_name": "random_forest",
                "n_estimators": 50,
                "max_depth": 5
            }
            
            controller.fit(self.X_clf, self.y_clf)
        
        # Check that callback methods were called
        assert mock_callback.on_experiment_start.called
        assert mock_callback.on_experiment_end.called
    
    def test_trial_failure_handling(self):
        """Test handling of trial failures."""
        searcher = OptunaSearcher(self.config_space)
        
        # Create a learner that raises an exception
        failing_learners = {
            "failing_learner": Mock(side_effect=Exception("Learner failed"))
        }
        
        controller = AutoMLController(
            learners=failing_learners,
            searcher=searcher,
            config_space=self.config_space,
            task="classification",
            n_trials=3,
            cv=2
        )
        
        # Mock the searcher
        with patch.object(searcher, 'suggest') as mock_suggest:
            mock_suggest.return_value = {
                "learner_name": "failing_learner",
                "n_estimators": 50,
                "max_depth": 5
            }
            
            # Should not raise exception, should handle gracefully
            controller.fit(self.X_clf, self.y_clf)
        
        # Should still have some results even with failures
        assert controller.best_score is not None
    
    def test_empty_data_handling(self):
        """Test handling of empty datasets."""
        searcher = OptunaSearcher(self.config_space)
        
        controller = AutoMLController(
            learners=self.mock_learners,
            searcher=searcher,
            config_space=self.config_space,
            task="classification",
            n_trials=3,
            cv=2
        )
        
        # Test with empty data
        with pytest.raises(ValueError, match="Empty dataset"):
            controller.fit(np.array([]), np.array([]))
    
    def test_single_sample_handling(self):
        """Test handling of single sample datasets."""
        searcher = OptunaSearcher(self.config_space)
        
        controller = AutoMLController(
            learners=self.mock_learners,
            searcher=searcher,
            config_space=self.config_space,
            task="classification",
            n_trials=3,
            cv=2
        )
        
        # Test with single sample
        X_single = self.X_clf[:1]
        y_single = self.y_clf[:1]
        
        with pytest.raises(ValueError, match="Insufficient samples"):
            controller.fit(X_single, y_single)
    
    def test_property_access(self):
        """Test accessing controller properties."""
        searcher = OptunaSearcher(self.config_space)
        
        controller = AutoMLController(
            learners=self.mock_learners,
            searcher=searcher,
            config_space=self.config_space,
            task="classification",
            n_trials=3,
            cv=2
        )
        
        # Test properties before fitting
        assert controller.best_score is None
        assert controller.best_model is None
        assert controller.best_config is None
        
        # Mock the searcher and fit
        with patch.object(searcher, 'suggest') as mock_suggest:
            mock_suggest.return_value = {
                "learner_name": "random_forest",
                "n_estimators": 50,
                "max_depth": 5
            }
            
            controller.fit(self.X_clf, self.y_clf)
        
        # Test properties after fitting
        assert controller.best_score is not None
        assert controller.best_model is not None
        assert controller.best_config is not None 