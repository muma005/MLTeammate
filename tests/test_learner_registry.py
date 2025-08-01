"""
Unit tests for learner registry system.

This test suite verifies the learner registry functionality including:
- SklearnWrapper functionality
- LearnerRegistry management
- Pre-built custom learners
- Configuration space generation
- Factory functions
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR

from ml_teammate.learners.registry import (
    SklearnWrapper,
    LearnerRegistry,
    get_learner_registry,
    get_learner,
    get_config_space,
    get_all_learners,
    get_classification_learners,
    get_regression_learners,
    create_learners_dict,
    create_config_space
)


class TestSklearnWrapper:
    """Test the SklearnWrapper class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.X, self.y = make_classification(n_samples=100, n_features=10, random_state=42)
        self.X_reg, self.y_reg = make_regression(n_samples=100, n_features=10, random_state=42)
    
    def test_sklearn_wrapper_initialization(self):
        """Test SklearnWrapper initialization."""
        wrapper = SklearnWrapper(RandomForestClassifier, {"n_estimators": 100})
        
        assert wrapper.model_class == RandomForestClassifier
        assert wrapper.config == {"n_estimators": 100}
        assert wrapper.model is None
    
    def test_sklearn_wrapper_initialization_no_config(self):
        """Test SklearnWrapper initialization without config."""
        wrapper = SklearnWrapper(RandomForestClassifier)
        
        assert wrapper.model_class == RandomForestClassifier
        assert wrapper.config == {}
        assert wrapper.model is None
    
    def test_create_model(self):
        """Test model creation."""
        wrapper = SklearnWrapper(RandomForestClassifier, {"n_estimators": 50, "random_state": 42})
        
        model = wrapper._create_model()
        
        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 50
        assert model.random_state == 42
    
    def test_fit(self):
        """Test fitting the wrapper."""
        wrapper = SklearnWrapper(RandomForestClassifier, {"n_estimators": 10, "random_state": 42})
        
        result = wrapper.fit(self.X, self.y)
        
        assert result == wrapper
        assert wrapper.model is not None
        assert isinstance(wrapper.model, RandomForestClassifier)
    
    def test_predict(self):
        """Test making predictions."""
        wrapper = SklearnWrapper(RandomForestClassifier, {"n_estimators": 10, "random_state": 42})
        wrapper.fit(self.X, self.y)
        
        predictions = wrapper.predict(self.X[:10])
        
        assert len(predictions) == 10
        assert all(isinstance(pred, (int, np.integer)) for pred in predictions)
    
    def test_predict_proba(self):
        """Test making probability predictions."""
        wrapper = SklearnWrapper(RandomForestClassifier, {"n_estimators": 10, "random_state": 42})
        wrapper.fit(self.X, self.y)
        
        probas = wrapper.predict_proba(self.X[:10])
        
        assert probas.shape == (10, 2)  # Binary classification
        assert np.allclose(probas.sum(axis=1), 1.0)  # Probabilities sum to 1
    
    def test_predict_proba_not_supported(self):
        """Test predict_proba with model that doesn't support it."""
        wrapper = SklearnWrapper(LinearRegression)
        wrapper.fit(self.X_reg, self.y_reg)
        
        with pytest.raises(AttributeError):
            wrapper.predict_proba(self.X_reg[:10])
    
    def test_get_params(self):
        """Test getting parameters."""
        config = {"n_estimators": 50, "random_state": 42}
        wrapper = SklearnWrapper(RandomForestClassifier, config)
        
        params = wrapper.get_params()
        
        assert params == config
    
    def test_set_params(self):
        """Test setting parameters."""
        wrapper = SklearnWrapper(RandomForestClassifier, {"n_estimators": 10})
        
        result = wrapper.set_params(n_estimators=100, random_state=42)
        
        assert result == wrapper
        assert wrapper.config["n_estimators"] == 100
        assert wrapper.config["random_state"] == 42
    
    def test_predict_without_fit(self):
        """Test that predict raises error if model not fitted."""
        wrapper = SklearnWrapper(RandomForestClassifier)
        
        with pytest.raises(ValueError, match="Model not fitted"):
            wrapper.predict(self.X[:10])
    
    def test_predict_proba_without_fit(self):
        """Test that predict_proba raises error if model not fitted."""
        wrapper = SklearnWrapper(RandomForestClassifier)
        
        with pytest.raises(ValueError, match="Model not fitted"):
            wrapper.predict_proba(self.X[:10])


class TestLearnerRegistry:
    """Test the LearnerRegistry class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.registry = LearnerRegistry()
    
    def test_registry_initialization(self):
        """Test LearnerRegistry initialization."""
        assert hasattr(self.registry, '_learners')
        assert hasattr(self.registry, '_config_spaces')
        assert hasattr(self.registry, '_dependencies')
        
        # Check that some learners are registered
        assert len(self.registry._learners) > 0
        assert len(self.registry._config_spaces) > 0
    
    def test_register_learner(self):
        """Test registering a custom learner."""
        def custom_factory(config):
            return RandomForestClassifier(**config)
        
        config_space = {
            "n_estimators": {"type": "int", "bounds": [10, 100]},
            "max_depth": {"type": "int", "bounds": [3, 10]}
        }
        
        self.registry._register_learner("custom_rf", custom_factory, config_space)
        
        assert "custom_rf" in self.registry._learners
        assert "custom_rf" in self.registry._config_spaces
        assert self.registry._learners["custom_rf"] == custom_factory
        assert self.registry._config_spaces["custom_rf"] == config_space
    
    def test_get_learner(self):
        """Test getting a learner factory."""
        # Test existing learner
        factory = self.registry.get_learner("random_forest")
        assert callable(factory)
        
        # Test non-existent learner
        with pytest.raises(ValueError, match="Unknown learner"):
            self.registry.get_learner("non_existent")
    
    def test_get_config_space(self):
        """Test getting a configuration space."""
        # Test existing learner
        config_space = self.registry.get_config_space("random_forest")
        assert isinstance(config_space, dict)
        assert len(config_space) > 0
        
        # Test non-existent learner
        with pytest.raises(ValueError, match="Unknown learner"):
            self.registry.get_config_space("non_existent")
    
    def test_get_all_learners(self):
        """Test getting all learner names."""
        learners = self.registry.get_all_learners()
        
        assert isinstance(learners, list)
        assert len(learners) > 0
        assert "random_forest" in learners
        assert "logistic_regression" in learners
    
    def test_get_classification_learners(self):
        """Test getting classification learners."""
        learners = self.registry.get_classification_learners()
        
        assert isinstance(learners, list)
        assert len(learners) > 0
        assert "random_forest" in learners
        assert "logistic_regression" in learners
        assert "svm" in learners
    
    def test_get_regression_learners(self):
        """Test getting regression learners."""
        learners = self.registry.get_regression_learners()
        
        assert isinstance(learners, list)
        assert len(learners) > 0
        assert "random_forest_regressor" in learners
        assert "linear_regression" in learners
        assert "ridge" in learners
    
    def test_check_dependencies(self):
        """Test dependency checking."""
        # Test sklearn dependency (should be available)
        missing = self.registry._check_dependencies("random_forest")
        assert len(missing) == 0
        
        # Test non-existent learner
        missing = self.registry._check_dependencies("non_existent")
        assert len(missing) == 0  # No dependencies for non-existent learner
    
    def test_create_learners_dict(self):
        """Test creating learners dictionary."""
        learner_names = ["random_forest", "logistic_regression"]
        learners_dict = self.registry.create_learners_dict(learner_names)
        
        assert isinstance(learners_dict, dict)
        assert len(learners_dict) == 2
        assert "random_forest" in learners_dict
        assert "logistic_regression" in learners_dict
        assert all(callable(factory) for factory in learners_dict.values())
    
    def test_create_config_space(self):
        """Test creating configuration space."""
        learner_names = ["random_forest", "logistic_regression"]
        config_space = self.registry.create_config_space(learner_names)
        
        assert isinstance(config_space, dict)
        assert len(config_space) == 2
        assert "random_forest" in config_space
        assert "logistic_regression" in config_space
    
    def test_create_learners_dict_invalid_learner(self):
        """Test creating learners dict with invalid learner."""
        with pytest.raises(ValueError, match="Unknown learner"):
            self.registry.create_learners_dict(["random_forest", "invalid_learner"])
    
    def test_create_config_space_invalid_learner(self):
        """Test creating config space with invalid learner."""
        with pytest.raises(ValueError, match="Unknown learner"):
            self.registry.create_config_space(["random_forest", "invalid_learner"])
    
    def test_pre_built_custom_learners(self):
        """Test that pre-built custom learners are available."""
        # Check that custom learners are registered
        assert "custom_rf" in self.registry._learners
        assert "custom_lr" in self.registry._learners
        assert "ensemble" in self.registry._learners
        
        # Test getting custom learners
        custom_rf_factory = self.registry.get_learner("custom_rf")
        custom_lr_factory = self.registry.get_learner("custom_lr")
        ensemble_factory = self.registry.get_learner("ensemble")
        
        assert callable(custom_rf_factory)
        assert callable(custom_lr_factory)
        assert callable(ensemble_factory)
    
    def test_create_ensemble_learner(self):
        """Test the _create_ensemble_learner method."""
        config = {
            "rf_n_estimators": 50,
            "rf_max_depth": 5,
            "lr_C": 1.0,
            "lr_max_iter": 1000
        }
        
        ensemble = self.registry._create_ensemble_learner(config)
        
        assert ensemble is not None
        # Should be a VotingClassifier
        from sklearn.ensemble import VotingClassifier
        assert isinstance(ensemble, VotingClassifier)


class TestRegistryFactoryFunctions:
    """Test the registry factory functions."""
    
    def test_get_learner_registry(self):
        """Test get_learner_registry function."""
        registry = get_learner_registry()
        
        assert isinstance(registry, LearnerRegistry)
        
        # Should return the same instance (singleton)
        registry2 = get_learner_registry()
        assert registry is registry2
    
    def test_get_learner(self):
        """Test get_learner function."""
        # Test existing learner
        factory = get_learner("random_forest")
        assert callable(factory)
        
        # Test non-existent learner
        with pytest.raises(ValueError, match="Unknown learner"):
            get_learner("non_existent")
    
    def test_get_config_space(self):
        """Test get_config_space function."""
        # Test existing learner
        config_space = get_config_space("random_forest")
        assert isinstance(config_space, dict)
        assert len(config_space) > 0
        
        # Test non-existent learner
        with pytest.raises(ValueError, match="Unknown learner"):
            get_config_space("non_existent")
    
    def test_get_all_learners(self):
        """Test get_all_learners function."""
        learners = get_all_learners()
        
        assert isinstance(learners, list)
        assert len(learners) > 0
        assert "random_forest" in learners
        assert "logistic_regression" in learners
    
    def test_get_classification_learners(self):
        """Test get_classification_learners function."""
        learners = get_classification_learners()
        
        assert isinstance(learners, list)
        assert len(learners) > 0
        assert "random_forest" in learners
        assert "logistic_regression" in learners
    
    def test_get_regression_learners(self):
        """Test get_regression_learners function."""
        learners = get_regression_learners()
        
        assert isinstance(learners, list)
        assert len(learners) > 0
        assert "random_forest_regressor" in learners
        assert "linear_regression" in learners
    
    def test_create_learners_dict(self):
        """Test create_learners_dict function."""
        learner_names = ["random_forest", "logistic_regression"]
        learners_dict = create_learners_dict(learner_names)
        
        assert isinstance(learners_dict, dict)
        assert len(learners_dict) == 2
        assert "random_forest" in learners_dict
        assert "logistic_regression" in learners_dict
        assert all(callable(factory) for factory in learners_dict.values())
    
    def test_create_config_space(self):
        """Test create_config_space function."""
        learner_names = ["random_forest", "logistic_regression"]
        config_space = create_config_space(learner_names)
        
        assert isinstance(config_space, dict)
        assert len(config_space) == 2
        assert "random_forest" in config_space
        assert "logistic_regression" in config_space


class TestLearnerIntegration:
    """Test integration between learners and registry."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.X, self.y = make_classification(n_samples=100, n_features=10, random_state=42)
        self.X_reg, self.y_reg = make_regression(n_samples=100, n_features=10, random_state=42)
    
    def test_random_forest_integration(self):
        """Test Random Forest learner integration."""
        factory = get_learner("random_forest")
        config_space = get_config_space("random_forest")
        
        # Test factory function
        model = factory({"n_estimators": 10, "random_state": 42})
        assert isinstance(model, RandomForestClassifier)
        
        # Test model fitting
        model.fit(self.X, self.y)
        predictions = model.predict(self.X[:10])
        assert len(predictions) == 10
        
        # Test config space
        assert "n_estimators" in config_space
        assert "max_depth" in config_space
    
    def test_logistic_regression_integration(self):
        """Test Logistic Regression learner integration."""
        factory = get_learner("logistic_regression")
        config_space = get_config_space("logistic_regression")
        
        # Test factory function
        model = factory({"C": 1.0, "random_state": 42})
        assert isinstance(model, LogisticRegression)
        
        # Test model fitting
        model.fit(self.X, self.y)
        predictions = model.predict(self.X[:10])
        assert len(predictions) == 10
        
        # Test config space
        assert "C" in config_space
        assert "max_iter" in config_space
    
    def test_custom_learners_integration(self):
        """Test custom learners integration."""
        # Test custom_rf
        factory = get_learner("custom_rf")
        config_space = get_config_space("custom_rf")
        
        model = factory({"n_estimators": 50, "max_depth": 5})
        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 50
        assert model.max_depth == 5
        assert model.random_state == 42  # Default from custom learner
        
        # Test custom_lr
        factory = get_learner("custom_lr")
        config_space = get_config_space("custom_lr")
        
        model = factory({"C": 2.0, "max_iter": 500})
        assert isinstance(model, LogisticRegression)
        assert model.C == 2.0
        assert model.max_iter == 500
        assert model.random_state == 42  # Default from custom learner
    
    def test_ensemble_integration(self):
        """Test ensemble learner integration."""
        factory = get_learner("ensemble")
        config_space = get_config_space("ensemble")
        
        model = factory({
            "rf_n_estimators": 50,
            "rf_max_depth": 5,
            "lr_C": 1.0,
            "lr_max_iter": 1000
        })
        
        # Should be a VotingClassifier
        from sklearn.ensemble import VotingClassifier
        assert isinstance(model, VotingClassifier)
        
        # Test model fitting
        model.fit(self.X, self.y)
        predictions = model.predict(self.X[:10])
        assert len(predictions) == 10
        
        # Test config space
        assert "rf_n_estimators" in config_space
        assert "rf_max_depth" in config_space
        assert "lr_C" in config_space
        assert "lr_max_iter" in config_space 