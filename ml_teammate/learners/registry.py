"""
Learner Registry System for MLTeammate

This module provides a comprehensive registry of pre-built learners that automatically:
1. Wrap sklearn models with proper compatibility
2. Generate sensible configuration spaces
3. Handle dependencies gracefully
4. Provide string-based learner selection

Users can simply specify learner names without writing any custom code.
"""

import numpy as np
from typing import Dict, Any, Optional, Callable, List, Union
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# Import existing learners
from .xgboost_learner import XGBoostLearner
from .lightgbm_learner import LightGBMLearner


class SklearnWrapper(BaseEstimator):
    """
    Universal wrapper for sklearn models that provides:
    - Consistent interface across all sklearn models
    - Configuration dictionary support
    - Proper sklearn compatibility
    - Error handling and validation
    """
    
    def __init__(self, model_class, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize the wrapper with a sklearn model class and configuration.
        
        Args:
            model_class: The sklearn model class (e.g., RandomForestClassifier)
            config: Dictionary of hyperparameters
            **kwargs: Additional keyword arguments
        """
        self.model_class = model_class
        self.config = (config or {}).copy()
        self.config.update(kwargs)
        self.model = None
        
        # Initialize model if config is provided
        if self.config:
            self._create_model()
    
    def _create_model(self):
        """Create the sklearn model with current configuration."""
        try:
            self.model = self.model_class(**self.config)
        except Exception as e:
            raise ValueError(f"Failed to create {self.model_class.__name__} with config {self.config}: {e}")
    
    def fit(self, X, y):
        """Fit the model to the data."""
        if self.model is None:
            self._create_model()
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities (if available)."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise AttributeError(f"{self.model_class.__name__} does not support predict_proba")
    
    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        return self.config.copy()
    
    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        self.config.update(params)
        return self


class LearnerRegistry:
    """
    Central registry for all available learners in MLTeammate.
    
    Provides:
    - Automatic sklearn model wrapping
    - Pre-built configuration spaces
    - Dependency management
    - String-based learner selection
    """
    
    def __init__(self):
        self._learners = {}
        self._config_spaces = {}
        self._dependencies = {}
        self._initialize_registry()
    
    def _initialize_registry(self):
        """Initialize the registry with all available learners."""
        
        # ========================================================================
        # CLASSIFICATION LEARNERS
        # ========================================================================
        
        # Random Forest
        self._register_learner(
            "random_forest",
            lambda config: SklearnWrapper(RandomForestClassifier, config),
            {
                "n_estimators": {"type": "int", "bounds": [50, 300]},
                "max_depth": {"type": "int", "bounds": [3, 15]},
                "min_samples_split": {"type": "int", "bounds": [2, 10]},
                "min_samples_leaf": {"type": "int", "bounds": [1, 5]},
                "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None]},
                "criterion": {"type": "categorical", "choices": ["gini", "entropy"]}
            },
            dependencies=["sklearn"]
        )
        
        # Logistic Regression
        self._register_learner(
            "logistic_regression",
            lambda config: SklearnWrapper(LogisticRegression, config),
            {
                "C": {"type": "float", "bounds": [0.1, 10.0]},
                "penalty": {"type": "categorical", "choices": ["l1", "l2", "elasticnet", None]},
                "solver": {"type": "categorical", "choices": ["liblinear", "saga", "lbfgs", "newton-cg"]},
                "max_iter": {"type": "int", "bounds": [100, 1000]},
                "class_weight": {"type": "categorical", "choices": [None, "balanced"]}
            },
            dependencies=["sklearn"]
        )
        
        # Support Vector Machine
        self._register_learner(
            "svm",
            lambda config: SklearnWrapper(SVC, config),
            {
                "C": {"type": "float", "bounds": [0.1, 10.0]},
                "kernel": {"type": "categorical", "choices": ["rbf", "linear", "poly", "sigmoid"]},
                "gamma": {"type": "categorical", "choices": ["scale", "auto"]},
                "degree": {"type": "int", "bounds": [2, 5]},
                "class_weight": {"type": "categorical", "choices": [None, "balanced"]}
            },
            dependencies=["sklearn"]
        )
        
        # Gradient Boosting
        self._register_learner(
            "gradient_boosting",
            lambda config: SklearnWrapper(GradientBoostingClassifier, config),
            {
                "n_estimators": {"type": "int", "bounds": [50, 300]},
                "learning_rate": {"type": "float", "bounds": [0.01, 0.3]},
                "max_depth": {"type": "int", "bounds": [3, 10]},
                "min_samples_split": {"type": "int", "bounds": [2, 10]},
                "subsample": {"type": "float", "bounds": [0.6, 1.0]},
                "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None]}
            },
            dependencies=["sklearn"]
        )
        
        # K-Nearest Neighbors
        self._register_learner(
            "knn",
            lambda config: SklearnWrapper(KNeighborsClassifier, config),
            {
                "n_neighbors": {"type": "int", "bounds": [3, 15]},
                "weights": {"type": "categorical", "choices": ["uniform", "distance"]},
                "algorithm": {"type": "categorical", "choices": ["auto", "ball_tree", "kd_tree", "brute"]},
                "leaf_size": {"type": "int", "bounds": [10, 50]},
                "p": {"type": "int", "bounds": [1, 3]}
            },
            dependencies=["sklearn"]
        )
        
        # Decision Tree
        self._register_learner(
            "decision_tree",
            lambda config: SklearnWrapper(DecisionTreeClassifier, config),
            {
                "max_depth": {"type": "int", "bounds": [3, 15]},
                "min_samples_split": {"type": "int", "bounds": [2, 10]},
                "min_samples_leaf": {"type": "int", "bounds": [1, 5]},
                "criterion": {"type": "categorical", "choices": ["gini", "entropy"]},
                "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None]}
            },
            dependencies=["sklearn"]
        )
        
        # Naive Bayes
        self._register_learner(
            "naive_bayes",
            lambda config: SklearnWrapper(GaussianNB, config),
            {
                "var_smoothing": {"type": "float", "bounds": [1e-9, 1e-6]}
            },
            dependencies=["sklearn"]
        )
        
        # Linear Discriminant Analysis
        self._register_learner(
            "lda",
            lambda config: SklearnWrapper(LinearDiscriminantAnalysis, config),
            {
                "solver": {"type": "categorical", "choices": ["svd", "lsqr", "eigen"]},
                "shrinkage": {"type": "categorical", "choices": [None, "auto"]}
            },
            dependencies=["sklearn"]
        )
        
        # Extra Trees
        self._register_learner(
            "extra_trees",
            lambda config: SklearnWrapper(ExtraTreesClassifier, config),
            {
                "n_estimators": {"type": "int", "bounds": [50, 300]},
                "max_depth": {"type": "int", "bounds": [3, 15]},
                "min_samples_split": {"type": "int", "bounds": [2, 10]},
                "min_samples_leaf": {"type": "int", "bounds": [1, 5]},
                "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None]},
                "criterion": {"type": "categorical", "choices": ["gini", "entropy"]}
            },
            dependencies=["sklearn"]
        )
        
        # ========================================================================
        # PRE-BUILT CUSTOM LEARNERS (No def functions needed!)
        # ========================================================================
        
        # Custom Random Forest with specific defaults
        self._register_learner(
            "custom_rf",
            lambda config: SklearnWrapper(RandomForestClassifier, {
                "n_estimators": config.get('n_estimators', 100),
                "max_depth": config.get('max_depth', 10),
                "min_samples_split": config.get('min_samples_split', 2),
                "random_state": 42
            }),
            {
                "n_estimators": {"type": "int", "bounds": [50, 300]},
                "max_depth": {"type": "int", "bounds": [3, 15]},
                "min_samples_split": {"type": "int", "bounds": [2, 20]}
            },
            dependencies=["sklearn"]
        )
        
        # Custom Logistic Regression with specific defaults
        self._register_learner(
            "custom_lr",
            lambda config: SklearnWrapper(LogisticRegression, {
                "C": config.get('C', 1.0),
                "max_iter": config.get('max_iter', 1000),
                "random_state": 42
            }),
            {
                "C": {"type": "float", "bounds": [0.1, 10.0]},
                "max_iter": {"type": "int", "bounds": [100, 2000]}
            },
            dependencies=["sklearn"]
        )
        
        # Ensemble learner (Random Forest + Logistic Regression)
        self._register_learner(
            "ensemble",
            lambda config: self._create_ensemble_learner(config),
            {
                "rf_n_estimators": {"type": "int", "bounds": [50, 200]},
                "rf_max_depth": {"type": "int", "bounds": [3, 15]},
                "lr_C": {"type": "float", "bounds": [0.1, 10.0]},
                "lr_max_iter": {"type": "int", "bounds": [100, 1000]}
            },
            dependencies=["sklearn"]
        )
        
        # ========================================================================
        # REGRESSION LEARNERS
        # ========================================================================
        
        # Linear Regression
        self._register_learner(
            "linear_regression",
            lambda config: SklearnWrapper(LinearRegression, config),
            {
                "fit_intercept": {"type": "categorical", "choices": [True, False]},
                "normalize": {"type": "categorical", "choices": [True, False]}
            },
            dependencies=["sklearn"]
        )
        
        # Ridge Regression
        self._register_learner(
            "ridge",
            lambda config: SklearnWrapper(Ridge, config),
            {
                "alpha": {"type": "float", "bounds": [0.1, 10.0]},
                "fit_intercept": {"type": "categorical", "choices": [True, False]},
                "normalize": {"type": "categorical", "choices": [True, False]},
                "solver": {"type": "categorical", "choices": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]}
            },
            dependencies=["sklearn"]
        )
        
        # Lasso Regression
        self._register_learner(
            "lasso",
            lambda config: SklearnWrapper(Lasso, config),
            {
                "alpha": {"type": "float", "bounds": [0.1, 10.0]},
                "fit_intercept": {"type": "categorical", "choices": [True, False]},
                "normalize": {"type": "categorical", "choices": [True, False]},
                "max_iter": {"type": "int", "bounds": [100, 2000]}
            },
            dependencies=["sklearn"]
        )
        
        # Random Forest Regressor
        self._register_learner(
            "random_forest_regressor",
            lambda config: SklearnWrapper(RandomForestRegressor, config),
            {
                "n_estimators": {"type": "int", "bounds": [50, 300]},
                "max_depth": {"type": "int", "bounds": [3, 15]},
                "min_samples_split": {"type": "int", "bounds": [2, 10]},
                "min_samples_leaf": {"type": "int", "bounds": [1, 5]},
                "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None]},
                "criterion": {"type": "categorical", "choices": ["mse", "mae"]}
            },
            dependencies=["sklearn"]
        )
        
        # Gradient Boosting Regressor
        self._register_learner(
            "gradient_boosting_regressor",
            lambda config: SklearnWrapper(GradientBoostingRegressor, config),
            {
                "n_estimators": {"type": "int", "bounds": [50, 300]},
                "learning_rate": {"type": "float", "bounds": [0.01, 0.3]},
                "max_depth": {"type": "int", "bounds": [3, 10]},
                "min_samples_split": {"type": "int", "bounds": [2, 10]},
                "subsample": {"type": "float", "bounds": [0.6, 1.0]},
                "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None]},
                "criterion": {"type": "categorical", "choices": ["mse", "mae"]}
            },
            dependencies=["sklearn"]
        )
        
        # Support Vector Regressor
        self._register_learner(
            "svr",
            lambda config: SklearnWrapper(SVR, config),
            {
                "C": {"type": "float", "bounds": [0.1, 10.0]},
                "kernel": {"type": "categorical", "choices": ["rbf", "linear", "poly", "sigmoid"]},
                "gamma": {"type": "categorical", "choices": ["scale", "auto"]},
                "degree": {"type": "int", "bounds": [2, 5]},
                "epsilon": {"type": "float", "bounds": [0.01, 0.1]}
            },
            dependencies=["sklearn"]
        )
        
        # K-Nearest Neighbors Regressor
        self._register_learner(
            "knn_regressor",
            lambda config: SklearnWrapper(KNeighborsRegressor, config),
            {
                "n_neighbors": {"type": "int", "bounds": [3, 15]},
                "weights": {"type": "categorical", "choices": ["uniform", "distance"]},
                "algorithm": {"type": "categorical", "choices": ["auto", "ball_tree", "kd_tree", "brute"]},
                "leaf_size": {"type": "int", "bounds": [10, 50]},
                "p": {"type": "int", "bounds": [1, 3]}
            },
            dependencies=["sklearn"]
        )
        
        # ========================================================================
        # EXISTING LEARNERS (XGBoost, LightGBM)
        # ========================================================================
        
        # XGBoost (existing)
        self._register_learner(
            "xgboost",
            lambda config: XGBoostLearner(config),
            {
                "max_depth": {"type": "int", "bounds": [3, 8]},
                "learning_rate": {"type": "float", "bounds": [0.01, 0.3]},
                "n_estimators": {"type": "int", "bounds": [50, 300]},
                "subsample": {"type": "float", "bounds": [0.6, 1.0]},
                "colsample_bytree": {"type": "float", "bounds": [0.6, 1.0]},
                "reg_alpha": {"type": "float", "bounds": [0, 1.0]},
                "reg_lambda": {"type": "float", "bounds": [0, 1.0]}
            },
            dependencies=["xgboost"]
        )
        
        # LightGBM (existing)
        self._register_learner(
            "lightgbm",
            lambda config: LightGBMLearner(config),
            {
                "max_depth": {"type": "int", "bounds": [3, 8]},
                "learning_rate": {"type": "float", "bounds": [0.01, 0.3]},
                "n_estimators": {"type": "int", "bounds": [50, 300]},
                "subsample": {"type": "float", "bounds": [0.6, 1.0]},
                "colsample_bytree": {"type": "float", "bounds": [0.6, 1.0]},
                "reg_alpha": {"type": "float", "bounds": [0, 1.0]},
                "reg_lambda": {"type": "float", "bounds": [0, 1.0]}
            },
            dependencies=["lightgbm"]
        )
    
    def _register_learner(self, name: str, factory: Callable, config_space: Dict[str, Any], 
                         dependencies: List[str] = None):
        """Register a learner with its factory function and configuration space."""
        self._learners[name] = factory
        self._config_spaces[name] = config_space
        self._dependencies[name] = dependencies or []
    
    def get_learner(self, name: str) -> Callable:
        """Get a learner factory function by name."""
        if name not in self._learners:
            available = ", ".join(sorted(self._learners.keys()))
            raise ValueError(f"Unknown learner '{name}'. Available learners: {available}")
        
        # Check dependencies
        missing_deps = self._check_dependencies(name)
        if missing_deps:
            print(f"⚠️ Warning: Missing dependencies for '{name}': {missing_deps}")
            print(f"   The learner may not work correctly without these packages.")
        
        return self._learners[name]
    
    def get_config_space(self, name: str) -> Dict[str, Any]:
        """Get the configuration space for a learner."""
        if name not in self._config_spaces:
            available = ", ".join(sorted(self._config_spaces.keys()))
            raise ValueError(f"Unknown learner '{name}'. Available learners: {available}")
        
        return self._config_spaces[name].copy()
    
    def get_all_learners(self) -> List[str]:
        """Get a list of all available learner names."""
        return sorted(self._learners.keys())
    
    def get_classification_learners(self) -> List[str]:
        """Get a list of classification learners."""
        classification_learners = [
            "random_forest", "logistic_regression", "svm", "gradient_boosting",
            "knn", "decision_tree", "naive_bayes", "lda", "extra_trees",
            "xgboost", "lightgbm"
        ]
        return [name for name in classification_learners if name in self._learners]
    
    def get_regression_learners(self) -> List[str]:
        """Get a list of regression learners."""
        regression_learners = [
            "linear_regression", "ridge", "lasso", "random_forest_regressor",
            "gradient_boosting_regressor", "svr", "knn_regressor"
        ]
        return [name for name in regression_learners if name in self._learners]
    
    def _check_dependencies(self, learner_name: str) -> List[str]:
        """Check if all dependencies for a learner are available."""
        if learner_name not in self._dependencies:
            return []
        
        missing = []
        for dep in self._dependencies[learner_name]:
            if dep == "sklearn":
                try:
                    import sklearn
                except ImportError:
                    missing.append(dep)
            elif dep == "xgboost":
                try:
                    import xgboost
                except ImportError:
                    missing.append(dep)
            elif dep == "lightgbm":
                try:
                    import lightgbm
                except ImportError:
                    missing.append(dep)
        
        return missing
    
    def create_learners_dict(self, learner_names: List[str]) -> Dict[str, Callable]:
        """Create a learners dictionary from a list of learner names."""
        learners = {}
        for name in learner_names:
            learners[name] = self.get_learner(name)
        return learners
    
    def create_config_space(self, learner_names: List[str]) -> Dict[str, Any]:
        """Create a configuration space from a list of learner names."""
        config_space = {}
        for name in learner_names:
            config_space[name] = self.get_config_space(name)
        return config_space
    
    def _create_ensemble_learner(self, config):
        """Create an ensemble learner (Random Forest + Logistic Regression)."""
        from sklearn.ensemble import VotingClassifier
        
        rf = RandomForestClassifier(
            n_estimators=config.get('rf_n_estimators', 100),
            max_depth=config.get('rf_max_depth', 10),
            random_state=42
        )
        lr = LogisticRegression(
            C=config.get('lr_C', 1.0),
            max_iter=config.get('lr_max_iter', 1000),
            random_state=42
        )
        
        return VotingClassifier(
            estimators=[('rf', rf), ('lr', lr)],
            voting='soft'
        )


# Global registry instance
_learner_registry = None

def get_learner_registry() -> LearnerRegistry:
    """Get the global learner registry instance."""
    global _learner_registry
    if _learner_registry is None:
        _learner_registry = LearnerRegistry()
    return _learner_registry

def get_learner(name: str) -> Callable:
    """Get a learner factory function by name."""
    return get_learner_registry().get_learner(name)

def get_config_space(name: str) -> Dict[str, Any]:
    """Get the configuration space for a learner."""
    return get_learner_registry().get_config_space(name)

def get_all_learners() -> List[str]:
    """Get a list of all available learner names."""
    return get_learner_registry().get_all_learners()

def get_classification_learners() -> List[str]:
    """Get a list of classification learners."""
    return get_learner_registry().get_classification_learners()

def get_regression_learners() -> List[str]:
    """Get a list of regression learners."""
    return get_learner_registry().get_regression_learners()

def create_learners_dict(learner_names: List[str]) -> Dict[str, Callable]:
    """Create a learners dictionary from a list of learner names."""
    return get_learner_registry().create_learners_dict(learner_names)

def create_config_space(learner_names: List[str]) -> Dict[str, Any]:
    """Create a configuration space from a list of learner names."""
    return get_learner_registry().create_config_space(learner_names) 