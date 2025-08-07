"""
High-level AutoML convenience functions for MLTeammate.

This module provides simple, one-function AutoML capabilities
for users who want quick results without detailed configuration.

Functions:
- quick_automl(): Run complete AutoML with minimal configuration
- automl_classify(): Classification-specific AutoML
- automl_regress(): Regression-specific AutoML
"""

import numpy as np
from typing import Union, List, Optional, Dict, Any, Tuple

from .controller import AutoMLController, create_automl_controller
from .callbacks import create_default_callbacks

# Import frozen components for validation
from ml_teammate.learners.registry import get_learner_registry
from ml_teammate.utils import ensure_numpy_array, ValidationError


def quick_automl(X, y, task: str = "auto", n_trials: int = 20, 
                 searcher: str = "random", test_size: float = 0.2,
                 random_state: int = 42, verbose: bool = True) -> Dict[str, Any]:
    """
    Quick AutoML with minimal configuration.
    
    Args:
        X: Training features
        y: Training targets
        task: Task type ("auto", "classification", "regression")
        n_trials: Number of optimization trials
        searcher: Search algorithm ("random", "optuna", "flaml")
        test_size: Fraction of data for testing
        random_state: Random seed
        verbose: Whether to show progress
        
    Returns:
        Dictionary with fitted controller, predictions, and results
    """
    # Auto-detect task if needed
    if task == "auto":
        task = _detect_task(y)
    
    # Get appropriate learners for task
    learner_names = _get_default_learners(task)
    
    # Split data
    X_train, X_test, y_train, y_test = _train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Create controller
    callbacks = create_default_callbacks(
        log_level="INFO" if verbose else "WARNING",
        use_mlflow=False  # Keep simple for quick runs
    )
    
    controller = create_automl_controller(
        learner_names=learner_names,
        task=task,
        searcher_type=searcher,
        n_trials=n_trials,
        random_state=random_state,
        callbacks=callbacks
    )
    
    # Fit controller
    controller.fit(X_train, y_train, X_test, y_test)
    
    # Make predictions
    y_pred = controller.predict(X_test)
    test_score = controller.score(X_test, y_test)
    
    # Return comprehensive results
    return {
        "controller": controller,
        "predictions": y_pred,
        "test_score": test_score,
        "best_model": controller.get_best_model(),
        "results": controller.get_results(),
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }


def automl_classify(X, y, learners: Optional[List[str]] = None,
                   n_trials: int = 30, cv_folds: int = 5,
                   searcher: str = "optuna", callbacks: Optional[List] = None,
                   **kwargs) -> 'AutoMLController':
    """
    Classification-specific AutoML with cross-validation.
    
    Args:
        X: Training features
        y: Training targets (class labels)
        learners: List of learner names (None for defaults)
        n_trials: Number of optimization trials
        cv_folds: Cross-validation folds
        searcher: Search algorithm
        callbacks: Custom callbacks (None for defaults)
        **kwargs: Additional controller arguments
        
    Returns:
        Fitted AutoMLController
    """
    # Get default classification learners if not specified
    if learners is None:
        learners = _get_default_learners("classification")
    
    # Create default callbacks if not provided
    if callbacks is None:
        callbacks = create_default_callbacks()
    
    # Create and fit controller
    controller = create_automl_controller(
        learner_names=learners,
        task="classification",
        searcher_type=searcher,
        n_trials=n_trials,
        cv_folds=cv_folds,
        callbacks=callbacks,
        **kwargs
    )
    
    controller.fit(X, y)
    return controller


def automl_regress(X, y, learners: Optional[List[str]] = None,
                  n_trials: int = 30, cv_folds: int = 5,
                  searcher: str = "optuna", callbacks: Optional[List] = None,
                  **kwargs) -> 'AutoMLController':
    """
    Regression-specific AutoML with cross-validation.
    
    Args:
        X: Training features
        y: Training targets (continuous values)
        learners: List of learner names (None for defaults)
        n_trials: Number of optimization trials
        cv_folds: Cross-validation folds
        searcher: Search algorithm
        callbacks: Custom callbacks (None for defaults)
        **kwargs: Additional controller arguments
        
    Returns:
        Fitted AutoMLController
    """
    # Get default regression learners if not specified
    if learners is None:
        learners = _get_default_learners("regression")
    
    # Create default callbacks if not provided
    if callbacks is None:
        callbacks = create_default_callbacks()
    
    # Create and fit controller
    controller = create_automl_controller(
        learner_names=learners,
        task="regression",
        searcher_type=searcher,
        n_trials=n_trials,
        cv_folds=cv_folds,
        callbacks=callbacks,
        **kwargs
    )
    
    controller.fit(X, y)
    return controller


def compare_searchers(X, y, task: str = "auto", learners: Optional[List[str]] = None,
                     n_trials: int = 20, searchers: Optional[List[str]] = None,
                     random_state: int = 42) -> Dict[str, Any]:
    """
    Compare different search algorithms on the same dataset.
    
    Args:
        X: Training features
        y: Training targets
        task: Task type ("auto", "classification", "regression")
        learners: List of learner names
        n_trials: Number of trials per searcher
        searchers: List of searcher names to compare
        random_state: Random seed
        
    Returns:
        Dictionary with results for each searcher
    """
    # Auto-detect task if needed
    if task == "auto":
        task = _detect_task(y)
    
    # Get default learners if not specified
    if learners is None:
        learners = _get_default_learners(task)
    
    # Get default searchers if not specified
    if searchers is None:
        searchers = ["random", "optuna", "flaml"]
    
    # Split data for consistent comparison
    X_train, X_test, y_train, y_test = _train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    results = {}
    
    for searcher in searchers:
        print(f"\n🔍 Running {searcher} search...")
        
        # Create controller with minimal callbacks for comparison
        callbacks = [create_default_callbacks()[0]]  # Just logger
        
        controller = create_automl_controller(
            learner_names=learners,
            task=task,
            searcher_type=searcher,
            n_trials=n_trials,
            random_state=random_state,
            callbacks=callbacks
        )
        
        # Fit and evaluate
        controller.fit(X_train, y_train, X_test, y_test)
        test_score = controller.score(X_test, y_test)
        
        results[searcher] = {
            "controller": controller,
            "test_score": test_score,
            "best_score": controller.get_results()["best_score"],
            "search_history": controller.get_search_history(),
            "experiment_info": controller.get_results()["experiment_info"]
        }
        
        print(f"✅ {searcher}: test_score = {test_score:.4f}")
    
    return results


# Helper functions

def _detect_task(y) -> str:
    """Automatically detect task type from target variable."""
    y = ensure_numpy_array(y)
    
    # Check if targets are continuous or discrete
    if np.issubdtype(y.dtype, np.integer):
        unique_values = np.unique(y)
        n_unique = len(unique_values)
        
        # Heuristic: if less than 20 unique integer values, likely classification
        if n_unique < 20:
            return "classification"
        else:
            return "regression"
    
    elif np.issubdtype(y.dtype, np.floating):
        # Check if all values are effectively integers
        if np.allclose(y, np.round(y)):
            unique_values = np.unique(np.round(y))
            if len(unique_values) < 20:
                return "classification"
        
        return "regression"
    
    else:
        # String or categorical data
        return "classification"


def _get_default_learners(task: str) -> List[str]:
    """Get default learners for a task."""
    registry = get_learner_registry()
    available = registry.list_learners(task)
    
    if task == "classification":
        # Prefer diverse, robust classifiers
        preferred = ["random_forest", "xgboost", "lightgbm", "logistic_regression"]
    else:
        # Prefer diverse, robust regressors
        preferred = ["random_forest", "xgboost", "lightgbm", "linear_regression"]
    
    # Return preferred learners that are available
    selected = []
    for learner in preferred:
        if learner in available:
            selected.append(learner)
    
    # If no preferred learners available, use all available
    if not selected:
        selected = list(available.keys())[:4]  # Limit to 4 for reasonable runtime
    
    return selected


def _train_test_split(X, y, test_size: float = 0.2, random_state: int = 42):
    """Simple train/test split implementation."""
    from sklearn.model_selection import train_test_split
    
    X = ensure_numpy_array(X)
    y = ensure_numpy_array(y)
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
