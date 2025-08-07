"""
Learner Registry System for MLTeammate

Provides comprehensive learner management and registration capabilities
with automatic sklearn wrapping, configuration spaces, and dependency handling
using our frozen Phase 1 utilities and Phase 2 data processing.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Callable, List, Union, Type
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor
)
from sklearn.linear_model import (
    LogisticRegression, LinearRegression,
    Ridge, Lasso, ElasticNet
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# Import our frozen Phase 1 utilities
from ml_teammate.utils import (
    get_logger, validate_data_arrays, ValidationError,
    Timer, time_context
)


class SklearnWrapper(BaseEstimator):
    """
    Universal wrapper for sklearn models with robust error handling and validation.
    
    Provides consistent interface, configuration management, and validation
    using our frozen Phase 1 utilities.
    """
    
    def __init__(self, model_class: Type[BaseEstimator], config: Optional[Dict[str, Any]] = None, 
                 task: str = "classification", name: str = None, **kwargs):
        """
        Initialize sklearn wrapper.
        
        Args:
            model_class: The sklearn model class
            config: Dictionary of hyperparameters  
            task: Task type ("classification" or "regression")
            name: Optional name for logging
            **kwargs: Additional keyword arguments
        """
        self.model_class = model_class
        self.task = task
        self.name = name or model_class.__name__
        self.config = (config or {}).copy()
        self.config.update(kwargs)
        
        # Initialize logger
        self.logger = get_logger(f"SklearnWrapper_{self.name}")
        
        # Model state
        self.model: Optional[BaseEstimator] = None
        self.is_fitted = False
        self.fit_time: Optional[float] = None
        self.predict_time: Optional[float] = None
        
        # Validate task
        if self.task not in ["classification", "regression"]:
            raise ValidationError(f"Invalid task: {self.task}. Must be 'classification' or 'regression'")
        
        self.logger.info(f"SklearnWrapper initialized for {self.name} ({self.task})")
    
    def _create_model(self) -> BaseEstimator:
        """Create sklearn model with current configuration."""
        try:
            with time_context(f"SklearnWrapper.create_model_{self.name}") as timer:
                # Filter out None values and invalid parameters
                clean_config = {k: v for k, v in self.config.items() if v is not None}
                
                # Create model
                model = self.model_class(**clean_config)
                
                # Validate task compatibility
                if self.task == "classification" and not hasattr(model, 'predict_proba'):
                    # Some classifiers don't have predict_proba, that's ok
                    pass
                elif self.task == "regression" and isinstance(model, ClassifierMixin):
                    raise ValidationError(f"{self.name} is a classifier but task is regression")
                elif self.task == "classification" and isinstance(model, RegressorMixin):
                    raise ValidationError(f"{self.name} is a regressor but task is classification")
                
                self.logger.debug(f"Model created in {timer.get_elapsed():.3f}s with config: {clean_config}")
                return model
                
        except Exception as e:
            error_msg = f"Failed to create {self.name} with config {self.config}: {str(e)}"
            self.logger.error(error_msg)
            raise ValidationError(error_msg)
    
    def fit(self, X, y) -> 'SklearnWrapper':
        """
        Fit the model to training data.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            self: Fitted wrapper
        """
        with time_context(f"SklearnWrapper.fit_{self.name}") as timer:
            # Validate inputs
            X = np.asarray(X)
            y = np.asarray(y)
            validate_data_arrays(X, y, self.task)
            
            # Create model if needed
            if self.model is None:
                self.model = self._create_model()
            
            # Fit model
            try:
                self.model.fit(X, y)
                self.is_fitted = True
                self.fit_time = timer.get_elapsed()
                
                self.logger.info(f"Model fitted in {self.fit_time:.3f}s on {X.shape[0]} samples")
                
            except Exception as e:
                error_msg = f"Fitting failed for {self.name}: {str(e)}"
                self.logger.error(error_msg)
                raise ValidationError(error_msg)
        
        return self
    
    def predict(self, X) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.is_fitted:
            raise ValidationError("Model not fitted. Call fit() first.")
        
        with time_context(f"SklearnWrapper.predict_{self.name}") as timer:
            X = np.asarray(X)
            
            try:
                predictions = self.model.predict(X)
                self.predict_time = timer.get_elapsed()
                
                self.logger.debug(f"Predictions made in {self.predict_time:.3f}s for {X.shape[0]} samples")
                return predictions
                
            except Exception as e:
                error_msg = f"Prediction failed for {self.name}: {str(e)}"
                self.logger.error(error_msg)
                raise ValidationError(error_msg)
    
    def predict_proba(self, X) -> np.ndarray:
        """
        Predict class probabilities (classification only).
        
        Args:
            X: Feature matrix
            
        Returns:
            np.ndarray: Class probabilities
        """
        if not self.is_fitted:
            raise ValidationError("Model not fitted. Call fit() first.")
        
        if self.task != "classification":
            raise ValidationError("predict_proba only available for classification tasks")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValidationError(f"{self.name} does not support predict_proba")
        
        with time_context(f"SklearnWrapper.predict_proba_{self.name}") as timer:
            X = np.asarray(X)
            
            try:
                probabilities = self.model.predict_proba(X)
                self.logger.debug(f"Probabilities computed in {timer.get_elapsed():.3f}s")
                return probabilities
                
            except Exception as e:
                error_msg = f"Probability prediction failed for {self.name}: {str(e)}"
                self.logger.error(error_msg)
                raise ValidationError(error_msg)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for sklearn compatibility."""
        params = {
            'model_class': self.model_class,
            'config': self.config,
            'task': self.task,
            'name': self.name
        }
        return params
    
    def set_params(self, **params) -> 'SklearnWrapper':
        """Set parameters for sklearn compatibility."""
        if 'config' in params:
            self.config.update(params['config'])
        else:
            # Filter out wrapper-specific params
            model_params = {k: v for k, v in params.items() 
                          if k not in ['model_class', 'task', 'name']}
            self.config.update(model_params)
        
        # Reset model to force recreation with new params
        self.model = None
        self.is_fitted = False
        
        return self
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            "name": self.name,
            "model_class": self.model_class.__name__,
            "task": self.task,
            "config": self.config.copy(),
            "is_fitted": self.is_fitted,
            "fit_time": self.fit_time,
            "predict_time": self.predict_time,
            "has_predict_proba": hasattr(self.model, 'predict_proba') if self.model else None
        }


class LearnerRegistry:
    """
    Central registry for all available learners in MLTeammate.
    
    Provides learner registration, configuration management, dependency tracking,
    and string-based learner selection with our frozen utilities.
    """
    
    def __init__(self, log_level: str = "INFO"):
        """
        Initialize learner registry.
        
        Args:
            log_level: Logging level
        """
        self.logger = get_logger("LearnerRegistry", log_level)
        
        # Registry storage
        self._learners: Dict[str, Dict[str, Any]] = {}
        self._config_spaces: Dict[str, Dict[str, Any]] = {}
        self._dependencies: Dict[str, List[str]] = {}
        self._task_mappings: Dict[str, List[str]] = {
            "classification": [],
            "regression": []
        }
        
        # Initialize with default learners
        self._initialize_default_learners()
        
        self.logger.info(f"LearnerRegistry initialized with {len(self._learners)} learners")
    
    def _initialize_default_learners(self):
        """Initialize registry with default sklearn learners."""
        
        # =======================================================================
        # CLASSIFICATION LEARNERS
        # =======================================================================
        
        # Random Forest Classifier
        self.register_learner(
            name="random_forest",
            model_class=RandomForestClassifier,
            task="classification",
            config_space={
                "n_estimators": {"type": "int", "bounds": [50, 300], "default": 100},
                "max_depth": {"type": "int", "bounds": [3, 20], "default": None},
                "min_samples_split": {"type": "int", "bounds": [2, 10], "default": 2},
                "min_samples_leaf": {"type": "int", "bounds": [1, 10], "default": 1},
                "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None], "default": "sqrt"},
                "criterion": {"type": "categorical", "choices": ["gini", "entropy"], "default": "gini"},
                "random_state": {"type": "fixed", "value": 42}
            },
            dependencies=["sklearn"],
            description="Random Forest classifier with ensemble of decision trees"
        )
        
        # Logistic Regression
        self.register_learner(
            name="logistic_regression", 
            model_class=LogisticRegression,
            task="classification",
            config_space={
                "C": {"type": "float", "bounds": [0.001, 100.0], "default": 1.0, "log_scale": True},
                "penalty": {"type": "categorical", "choices": ["l1", "l2", "elasticnet", None], "default": "l2"},
                "solver": {"type": "categorical", "choices": ["liblinear", "saga", "lbfgs", "newton-cg"], "default": "lbfgs"},
                "max_iter": {"type": "int", "bounds": [100, 2000], "default": 1000},
                "class_weight": {"type": "categorical", "choices": [None, "balanced"], "default": None},
                "random_state": {"type": "fixed", "value": 42}
            },
            dependencies=["sklearn"],
            description="Logistic regression for linear classification"
        )
        
        # Support Vector Classifier
        self.register_learner(
            name="svm",
            model_class=SVC,
            task="classification", 
            config_space={
                "C": {"type": "float", "bounds": [0.001, 100.0], "default": 1.0, "log_scale": True},
                "kernel": {"type": "categorical", "choices": ["rbf", "linear", "poly", "sigmoid"], "default": "rbf"},
                "gamma": {"type": "categorical", "choices": ["scale", "auto"], "default": "scale"},
                "degree": {"type": "int", "bounds": [2, 5], "default": 3},
                "class_weight": {"type": "categorical", "choices": [None, "balanced"], "default": None},
                "probability": {"type": "fixed", "value": True},  # Enable predict_proba
                "random_state": {"type": "fixed", "value": 42}
            },
            dependencies=["sklearn"],
            description="Support Vector Machine with various kernels"
        )
        
        # Gradient Boosting Classifier
        self.register_learner(
            name="gradient_boosting",
            model_class=GradientBoostingClassifier,
            task="classification",
            config_space={
                "n_estimators": {"type": "int", "bounds": [50, 300], "default": 100},
                "learning_rate": {"type": "float", "bounds": [0.01, 0.3], "default": 0.1},
                "max_depth": {"type": "int", "bounds": [3, 10], "default": 3},
                "min_samples_split": {"type": "int", "bounds": [2, 10], "default": 2},
                "min_samples_leaf": {"type": "int", "bounds": [1, 10], "default": 1},
                "subsample": {"type": "float", "bounds": [0.6, 1.0], "default": 1.0},
                "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None], "default": None},
                "random_state": {"type": "fixed", "value": 42}
            },
            dependencies=["sklearn"],
            description="Gradient boosting classifier with sequential tree building"
        )
        
        # =======================================================================
        # REGRESSION LEARNERS  
        # =======================================================================
        
        # Linear Regression
        self.register_learner(
            name="linear_regression",
            model_class=LinearRegression,
            task="regression",
            config_space={
                "fit_intercept": {"type": "categorical", "choices": [True, False], "default": True}
            },
            dependencies=["sklearn"],
            description="Ordinary least squares linear regression"
        )
        
        # Ridge Regression
        self.register_learner(
            name="ridge",
            model_class=Ridge,
            task="regression",
            config_space={
                "alpha": {"type": "float", "bounds": [0.001, 100.0], "default": 1.0, "log_scale": True},
                "fit_intercept": {"type": "categorical", "choices": [True, False], "default": True},
                "solver": {"type": "categorical", "choices": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"], "default": "auto"},
                "random_state": {"type": "fixed", "value": 42}
            },
            dependencies=["sklearn"],
            description="Ridge regression with L2 regularization"
        )
        
        # Random Forest Regressor
        self.register_learner(
            name="random_forest_regressor",
            model_class=RandomForestRegressor,
            task="regression",
            config_space={
                "n_estimators": {"type": "int", "bounds": [50, 300], "default": 100},
                "max_depth": {"type": "int", "bounds": [3, 20], "default": None},
                "min_samples_split": {"type": "int", "bounds": [2, 10], "default": 2},
                "min_samples_leaf": {"type": "int", "bounds": [1, 10], "default": 1},
                "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None], "default": "sqrt"},
                "criterion": {"type": "categorical", "choices": ["squared_error", "absolute_error"], "default": "squared_error"},
                "random_state": {"type": "fixed", "value": 42}
            },
            dependencies=["sklearn"],
            description="Random Forest regressor with ensemble of decision trees"
        )
        
        # Gradient Boosting Regressor
        self.register_learner(
            name="gradient_boosting_regressor",
            model_class=GradientBoostingRegressor,
            task="regression",
            config_space={
                "n_estimators": {"type": "int", "bounds": [50, 300], "default": 100},
                "learning_rate": {"type": "float", "bounds": [0.01, 0.3], "default": 0.1},
                "max_depth": {"type": "int", "bounds": [3, 10], "default": 3},
                "min_samples_split": {"type": "int", "bounds": [2, 10], "default": 2},
                "min_samples_leaf": {"type": "int", "bounds": [1, 10], "default": 1},
                "subsample": {"type": "float", "bounds": [0.6, 1.0], "default": 1.0},
                "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None], "default": None},
                "random_state": {"type": "fixed", "value": 42}
            },
            dependencies=["sklearn"],
            description="Gradient boosting regressor with sequential tree building"
        )
    
    def register_learner(self, name: str, model_class: Type[BaseEstimator], task: str,
                        config_space: Dict[str, Any], dependencies: List[str] = None,
                        description: str = None) -> None:
        """
        Register a new learner in the registry.
        
        Args:
            name: Unique name for the learner
            model_class: Sklearn model class
            task: Task type ("classification" or "regression")
            config_space: Configuration space definition
            dependencies: List of required packages
            description: Optional description
        """
        if name in self._learners:
            self.logger.warning(f"Overwriting existing learner: {name}")
        
        # Validate task
        if task not in ["classification", "regression"]:
            raise ValidationError(f"Invalid task: {task}")
        
        # Store learner information
        self._learners[name] = {
            "model_class": model_class,
            "task": task,
            "description": description or f"{model_class.__name__} for {task}"
        }
        
        self._config_spaces[name] = config_space
        self._dependencies[name] = dependencies or []
        
        # Add to task mapping
        if name not in self._task_mappings[task]:
            self._task_mappings[task].append(name)
        
        self.logger.info(f"Registered learner: {name} ({task})")
    
    def create_learner(self, name: str, config: Optional[Dict[str, Any]] = None) -> SklearnWrapper:
        """
        Create a learner instance by name.
        
        Args:
            name: Learner name
            config: Configuration dictionary
            
        Returns:
            SklearnWrapper: Configured learner instance
        """
        if name not in self._learners:
            available = list(self._learners.keys())
            raise ValidationError(f"Unknown learner: {name}. Available: {available}")
        
        learner_info = self._learners[name]
        
        # Check dependencies
        self._check_dependencies(name)
        
        # Use provided config or get default config
        if config is None:
            config = self.get_default_config(name)
        
        # Create wrapper
        wrapper = SklearnWrapper(
            model_class=learner_info["model_class"],
            config=config,
            task=learner_info["task"],
            name=name
        )
        
        self.logger.debug(f"Created learner: {name} with config: {config}")
        return wrapper
    
    def get_default_config(self, name: str) -> Dict[str, Any]:
        """
        Get default configuration for a learner.
        
        Args:
            name: Learner name
            
        Returns:
            dict: Default configuration
        """
        if name not in self._config_spaces:
            raise ValidationError(f"No config space for learner: {name}")
        
        config_space = self._config_spaces[name]
        default_config = {}
        
        for param_name, param_spec in config_space.items():
            if "default" in param_spec:
                default_config[param_name] = param_spec["default"]
            elif param_spec["type"] == "fixed":
                default_config[param_name] = param_spec["value"]
            elif param_spec["type"] == "categorical":
                default_config[param_name] = param_spec["choices"][0]
            elif param_spec["type"] == "int":
                bounds = param_spec["bounds"]
                default_config[param_name] = (bounds[0] + bounds[1]) // 2
            elif param_spec["type"] == "float":
                bounds = param_spec["bounds"]
                if param_spec.get("log_scale", False):
                    import math
                    log_mid = (math.log(bounds[0]) + math.log(bounds[1])) / 2
                    default_config[param_name] = math.exp(log_mid)
                else:
                    default_config[param_name] = (bounds[0] + bounds[1]) / 2
        
        return default_config
    
    def _check_dependencies(self, name: str) -> None:
        """Check if all dependencies for a learner are available."""
        if name not in self._dependencies:
            return
        
        for dependency in self._dependencies[name]:
            try:
                __import__(dependency)
            except ImportError:
                self.logger.warning(f"Missing dependency '{dependency}' for learner '{name}'")
                # Don't raise error - sklearn should always be available
    
    def get_learners_by_task(self, task: str) -> List[str]:
        """
        Get all learner names for a specific task.
        
        Args:
            task: Task type
            
        Returns:
            list: List of learner names
        """
        if task not in self._task_mappings:
            raise ValidationError(f"Invalid task: {task}")
        
        return self._task_mappings[task].copy()
    
    def list_learners(self, task: Optional[str] = None) -> Dict[str, Any]:
        """
        List all available learners.
        
        Args:
            task: Optional task filter
            
        Returns:
            dict: Learner information
        """
        if task is None:
            learners = list(self._learners.keys())
        else:
            learners = self.get_learners_by_task(task)
        
        result = {}
        for name in learners:
            learner_info = self._learners[name]
            result[name] = {
                "task": learner_info["task"],
                "description": learner_info["description"],
                "model_class": learner_info["model_class"].__name__
            }
        
        return result
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive registry summary.
        
        Returns:
            dict: Registry statistics and information
        """
        return {
            "total_learners": len(self._learners),
            "classification_learners": len(self._task_mappings["classification"]),
            "regression_learners": len(self._task_mappings["regression"]),
            "learners_by_task": self._task_mappings.copy(),
            "all_learners": list(self._learners.keys()),
            "dependencies": set().union(*self._dependencies.values()) if self._dependencies else set()
        }


# Global registry instance
_global_registry: Optional[LearnerRegistry] = None


def get_learner_registry() -> LearnerRegistry:
    """
    Get the global learner registry instance.
    
    Returns:
        LearnerRegistry: Global registry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = LearnerRegistry()
    return _global_registry


def create_learner(name: str, config: Optional[Dict[str, Any]] = None) -> SklearnWrapper:
    """
    Convenience function to create a learner from the global registry.
    
    Args:
        name: Learner name
        config: Configuration dictionary
        
    Returns:
        SklearnWrapper: Configured learner instance
    """
    registry = get_learner_registry()
    return registry.create_learner(name, config)


def list_available_learners(task: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to list available learners.
    
    Args:
        task: Optional task filter
        
    Returns:
        dict: Available learners
    """
    registry = get_learner_registry()
    return registry.list_learners(task)


def get_learner_config_space(name: str) -> Dict[str, Any]:
    """
    Convenience function to get learner configuration space.
    
    Args:
        name: Learner name
        
    Returns:
        dict: Configuration space
    """
    registry = get_learner_registry()
    if name not in registry._config_spaces:
        raise ValidationError(f"No config space for learner: {name}")
    return registry._config_spaces[name]
