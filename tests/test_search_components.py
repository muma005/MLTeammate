"""
Unit tests for search components.

This test suite verifies the search functionality including:
- OptunaSearcher functionality
- FLAMLSearcher functionality  
- Early Convergence Indicators (ECI)
- Configuration space validation
- Search algorithm integration
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from ml_teammate.search.optuna_search import OptunaSearcher
from ml_teammate.search.flaml_search import FLAMLSearcher, FLAMLTimeBudgetSearcher
from ml_teammate.search.eci import EarlyConvergenceIndicator, AdaptiveECI, MultiObjectiveECI
from ml_teammate.search import get_searcher, get_eci, list_available_searchers, list_available_eci_types


class TestOptunaSearcher:
    """Test the OptunaSearcher class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config_space = {
            "random_forest": {
                "n_estimators": {"type": "int", "bounds": [10, 100]},
                "max_depth": {"type": "int", "bounds": [3, 10]},
                "criterion": {"type": "categorical", "choices": ["gini", "entropy"]}
            },
            "logistic_regression": {
                "C": {"type": "float", "bounds": [0.1, 10.0]},
                "max_iter": {"type": "int", "bounds": [100, 1000]}
            }
        }
    
    def test_optuna_searcher_initialization(self):
        """Test OptunaSearcher initialization."""
        searcher = OptunaSearcher(self.config_space)
        
        assert searcher.config_spaces == self.config_space
        assert searcher.study is not None
        assert searcher.trial_history == []
    
    def test_optuna_searcher_with_custom_study(self):
        """Test OptunaSearcher with custom study."""
        import optuna
        
        study = optuna.create_study(direction="maximize")
        searcher = OptunaSearcher(self.config_space, study=study)
        
        assert searcher.study == study
    
    def test_suggest_parameters(self):
        """Test parameter suggestion functionality."""
        searcher = OptunaSearcher(self.config_space)
        
        # Test suggesting parameters for random_forest
        params = searcher.suggest("trial_1", "random_forest")
        
        assert "learner_name" in params
        assert params["learner_name"] == "random_forest"
        assert "n_estimators" in params
        assert "max_depth" in params
        assert "criterion" in params
        
        # Check parameter bounds
        assert 10 <= params["n_estimators"] <= 100
        assert 3 <= params["max_depth"] <= 10
        assert params["criterion"] in ["gini", "entropy"]
    
    def test_suggest_parameters_different_learner(self):
        """Test parameter suggestion for different learners."""
        searcher = OptunaSearcher(self.config_space)
        
        # Test suggesting parameters for logistic_regression
        params = searcher.suggest("trial_2", "logistic_regression")
        
        assert "learner_name" in params
        assert params["learner_name"] == "logistic_regression"
        assert "C" in params
        assert "max_iter" in params
        
        # Check parameter bounds
        assert 0.1 <= params["C"] <= 10.0
        assert 100 <= params["max_iter"] <= 1000
    
    def test_report_results(self):
        """Test reporting trial results."""
        searcher = OptunaSearcher(self.config_space)
        
        # Suggest parameters
        params = searcher.suggest("trial_1", "random_forest")
        
        # Report results
        searcher.report("trial_1", 0.85)
        
        # Check that trial was recorded
        assert len(searcher.trial_history) == 1
        assert searcher.trial_history[0]["trial_id"] == "trial_1"
        assert searcher.trial_history[0]["score"] == 0.85
        assert searcher.trial_history[0]["params"] == params
    
    def test_get_best_results(self):
        """Test getting best results."""
        searcher = OptunaSearcher(self.config_space)
        
        # Run multiple trials
        for i in range(3):
            params = searcher.suggest(f"trial_{i}", "random_forest")
            score = 0.7 + i * 0.1  # Increasing scores
            searcher.report(f"trial_{i}", score)
        
        # Get best results
        best = searcher.get_best()
        
        assert "score" in best
        assert "params" in best
        assert best["score"] == 0.9  # Highest score
        assert best["params"]["learner_name"] == "random_forest"
    
    def test_invalid_learner_name(self):
        """Test handling of invalid learner names."""
        searcher = OptunaSearcher(self.config_space)
        
        with pytest.raises(ValueError, match="Unknown learner"):
            searcher.suggest("trial_1", "invalid_learner")
    
    def test_empty_config_space(self):
        """Test handling of empty configuration space."""
        with pytest.raises(ValueError, match="Empty configuration space"):
            OptunaSearcher({})


class TestFLAMLSearcher:
    """Test the FLAMLSearcher class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config_space = {
            "random_forest": {
                "n_estimators": {"type": "int", "bounds": [10, 100]},
                "max_depth": {"type": "int", "bounds": [3, 10]}
            }
        }
    
    def test_flaml_searcher_initialization(self):
        """Test FLAMLSearcher initialization."""
        searcher = FLAMLSearcher(self.config_space, time_budget=60)
        
        assert searcher.config_spaces == self.config_space
        assert searcher.time_budget == 60
        assert searcher.metric == "accuracy"
        assert searcher.mode == "max"
    
    def test_flaml_searcher_custom_settings(self):
        """Test FLAMLSearcher with custom settings."""
        searcher = FLAMLSearcher(
            self.config_space,
            time_budget=30,
            metric="r2",
            mode="max",
            estimator_list=["lgbm", "rf"]
        )
        
        assert searcher.time_budget == 30
        assert searcher.metric == "r2"
        assert searcher.mode == "max"
        assert searcher.estimator_list == ["lgbm", "rf"]
    
    def test_convert_config_space(self):
        """Test configuration space conversion."""
        searcher = FLAMLSearcher(self.config_space)
        
        converted = searcher._convert_config_space("random_forest")
        
        assert "n_estimators" in converted
        assert "max_depth" in converted
        assert isinstance(converted["n_estimators"], tuple)
        assert isinstance(converted["max_depth"], tuple)
    
    def test_flaml_fit(self):
        """Test FLAML fitting functionality."""
        searcher = FLAMLSearcher(self.config_space, time_budget=5)
        
        # Generate test data
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        
        # Mock FLAML AutoML
        with patch('ml_teammate.search.flaml_search.AutoML') as mock_automl:
            mock_automl_instance = Mock()
            mock_automl.return_value = mock_automl_instance
            
            searcher.fit(X, y, task="classification")
            
            # Check that FLAML was called
            mock_automl_instance.fit.assert_called_once()
    
    def test_get_best_results(self):
        """Test getting best results from FLAML."""
        searcher = FLAMLSearcher(self.config_space)
        
        # Mock FLAML results
        searcher.automl = Mock()
        searcher.automl.best_config = {"n_estimators": 50, "max_depth": 5}
        searcher.automl.best_loss = 0.1
        searcher.automl.best_estimator = RandomForestClassifier()
        
        best = searcher.get_best()
        
        assert "score" in best
        assert "params" in best
        assert "model" in best


class TestEarlyConvergenceIndicator:
    """Test the EarlyConvergenceIndicator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.eci = EarlyConvergenceIndicator(
            window_size=5,
            min_trials=3,
            improvement_threshold=0.001,
            patience=3
        )
    
    def test_eci_initialization(self):
        """Test ECI initialization."""
        assert self.eci.window_size == 5
        assert self.eci.min_trials == 3
        assert self.eci.improvement_threshold == 0.001
        assert self.eci.patience == 3
        assert self.eci.trials == []
    
    def test_add_trial(self):
        """Test adding trials to ECI."""
        self.eci.add_trial("trial_1", 0.8)
        self.eci.add_trial("trial_2", 0.85)
        
        assert len(self.eci.trials) == 2
        assert self.eci.trials[0]["trial_id"] == "trial_1"
        assert self.eci.trials[0]["score"] == 0.8
        assert self.eci.trials[1]["trial_id"] == "trial_2"
        assert self.eci.trials[1]["score"] == 0.85
    
    def test_moving_average_convergence(self):
        """Test moving average convergence detection."""
        # Add trials with improving scores
        for i in range(5):
            self.eci.add_trial(f"trial_{i}", 0.8 + i * 0.01)
        
        # Should not converge (improving scores)
        assert not self.eci.should_stop()
        
        # Add trials with plateauing scores
        for i in range(5):
            self.eci.add_trial(f"trial_{i+5}", 0.85)
        
        # Should converge (no improvement)
        assert self.eci.should_stop()
    
    def test_improvement_rate_convergence(self):
        """Test improvement rate convergence detection."""
        eci = EarlyConvergenceIndicator(convergence_method="improvement_rate")
        
        # Add trials with decreasing improvement
        scores = [0.8, 0.82, 0.83, 0.835, 0.836, 0.8365, 0.8365, 0.8365]
        for i, score in enumerate(scores):
            eci.add_trial(f"trial_{i}", score)
        
        # Should converge (very small improvements)
        assert eci.should_stop()
    
    def test_confidence_interval_convergence(self):
        """Test confidence interval convergence detection."""
        eci = EarlyConvergenceIndicator(convergence_method="confidence_interval")
        
        # Add trials with stable scores
        for i in range(10):
            eci.add_trial(f"trial_{i}", 0.85 + np.random.normal(0, 0.01))
        
        # Should converge (stable scores)
        assert eci.should_stop()
    
    def test_plateau_convergence(self):
        """Test plateau convergence detection."""
        eci = EarlyConvergenceIndicator(convergence_method="plateau")
        
        # Add trials with plateau
        scores = [0.8, 0.82, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85]
        for i, score in enumerate(scores):
            eci.add_trial(f"trial_{i}", score)
        
        # Should converge (plateau detected)
        assert eci.should_stop()
    
    def test_get_convergence_info(self):
        """Test getting convergence information."""
        # Add some trials
        for i in range(5):
            self.eci.add_trial(f"trial_{i}", 0.8 + i * 0.01)
        
        info = self.eci.get_convergence_info()
        
        assert "n_trials" in info
        assert "converged" in info
        assert "method" in info
        assert info["n_trials"] == 5
    
    def test_reset(self):
        """Test resetting ECI state."""
        # Add some trials
        for i in range(3):
            self.eci.add_trial(f"trial_{i}", 0.8 + i * 0.01)
        
        assert len(self.eci.trials) == 3
        
        # Reset
        self.eci.reset()
        
        assert len(self.eci.trials) == 0


class TestAdaptiveECI:
    """Test the AdaptiveECI class."""
    
    def test_adaptive_eci_initialization(self):
        """Test AdaptiveECI initialization."""
        eci = AdaptiveECI()
        
        assert hasattr(eci, 'window_size')
        assert hasattr(eci, 'improvement_threshold')
        assert hasattr(eci, 'patience')
    
    def test_adaptive_parameter_adjustment(self):
        """Test adaptive parameter adjustment."""
        eci = AdaptiveECI()
        
        # Add trials to trigger adaptation
        for i in range(10):
            eci.add_trial(f"trial_{i}", 0.8 + i * 0.01)
        
        # Check that parameters were adjusted
        assert eci.window_size > 0
        assert eci.improvement_threshold > 0


class TestMultiObjectiveECI:
    """Test the MultiObjectiveECI class."""
    
    def test_multi_objective_eci_initialization(self):
        """Test MultiObjectiveECI initialization."""
        eci = MultiObjectiveECI(objectives=["accuracy", "speed"])
        
        assert eci.objectives == ["accuracy", "speed"]
        assert len(eci.trials) == 0
    
    def test_multi_objective_trial_addition(self):
        """Test adding multi-objective trials."""
        eci = MultiObjectiveECI(objectives=["accuracy", "speed"])
        
        # Add trial with multiple objectives
        eci.add_trial("trial_1", {"accuracy": 0.85, "speed": 0.9})
        
        assert len(eci.trials) == 1
        assert eci.trials[0]["trial_id"] == "trial_1"
        assert eci.trials[0]["score"]["accuracy"] == 0.85
        assert eci.trials[0]["score"]["speed"] == 0.9


class TestSearchFactoryFunctions:
    """Test the search factory functions."""
    
    def test_get_searcher(self):
        """Test get_searcher factory function."""
        # Test Optuna searcher
        searcher = get_searcher("optuna", config_spaces={"test": {}})
        assert isinstance(searcher, OptunaSearcher)
        
        # Test FLAML searcher
        searcher = get_searcher("flaml", config_spaces={"test": {}})
        assert isinstance(searcher, FLAMLSearcher)
        
        # Test invalid searcher
        with pytest.raises(ValueError, match="Unknown searcher type"):
            get_searcher("invalid", config_spaces={"test": {}})
    
    def test_get_eci(self):
        """Test get_eci factory function."""
        # Test standard ECI
        eci = get_eci("standard")
        assert isinstance(eci, EarlyConvergenceIndicator)
        
        # Test adaptive ECI
        eci = get_eci("adaptive")
        assert isinstance(eci, AdaptiveECI)
        
        # Test multi-objective ECI
        eci = get_eci("multi_objective", objectives=["acc", "speed"])
        assert isinstance(eci, MultiObjectiveECI)
        
        # Test invalid ECI
        with pytest.raises(ValueError, match="Unknown ECI type"):
            get_eci("invalid")
    
    def test_list_available_searchers(self):
        """Test list_available_searchers function."""
        searchers = list_available_searchers()
        
        assert "optuna" in searchers
        assert "flaml" in searchers
        assert "flaml_time_budget" in searchers
        assert "flaml_resource_aware" in searchers
        
        # Check structure
        for name, info in searchers.items():
            assert "description" in info
            assert "features" in info
            assert "dependencies" in info
    
    def test_list_available_eci_types(self):
        """Test list_available_eci_types function."""
        eci_types = list_available_eci_types()
        
        assert "standard" in eci_types
        assert "adaptive" in eci_types
        assert "multi_objective" in eci_types
        
        # Check structure
        for name, info in eci_types.items():
            assert "description" in info
            assert "methods" in info
            assert "features" in info 