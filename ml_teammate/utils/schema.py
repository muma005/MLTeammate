"""
Schema Validation for MLTeammate

Provides type checking and validation for configurations and data structures
used throughout the MLTeammate framework.
"""

import numpy as np
from typing import Any, Dict, List, Union, Optional, Tuple
from numbers import Number


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_config_space(config_space: Dict[str, Any]) -> bool:
    """
    Validate a configuration space dictionary.
    
    Args:
        config_space: Configuration space to validate
        
    Returns:
        bool: True if valid
        
    Raises:
        ValidationError: If configuration space is invalid
    """
    if not isinstance(config_space, dict):
        raise ValidationError(f"Config space must be dict, got {type(config_space)}")
    
    if not config_space:
        raise ValidationError("Config space cannot be empty")
    
    for param_name, param_spec in config_space.items():
        if not isinstance(param_name, str):
            raise ValidationError(f"Parameter name must be string, got {type(param_name)}")
        
        if not isinstance(param_spec, dict):
            raise ValidationError(f"Parameter spec for '{param_name}' must be dict, got {type(param_spec)}")
        
        if "type" not in param_spec:
            raise ValidationError(f"Parameter '{param_name}' missing 'type' field")
        
        param_type = param_spec["type"]
        
        if param_type == "int":
            _validate_int_param(param_name, param_spec)
        elif param_type == "float":
            _validate_float_param(param_name, param_spec)
        elif param_type == "categorical":
            _validate_categorical_param(param_name, param_spec)
        else:
            raise ValidationError(f"Parameter '{param_name}' has invalid type: {param_type}")
    
    return True


def _validate_int_param(param_name: str, param_spec: Dict[str, Any]):
    """Validate integer parameter specification."""
    if "bounds" not in param_spec:
        raise ValidationError(f"Integer parameter '{param_name}' missing 'bounds' field")
    
    bounds = param_spec["bounds"]
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ValidationError(f"Parameter '{param_name}' bounds must be [low, high]")
    
    low, high = bounds
    if not isinstance(low, int) or not isinstance(high, int):
        raise ValidationError(f"Parameter '{param_name}' bounds must be integers")
    
    if low >= high:
        raise ValidationError(f"Parameter '{param_name}' low bound must be < high bound")


def _validate_float_param(param_name: str, param_spec: Dict[str, Any]):
    """Validate float parameter specification."""
    if "bounds" not in param_spec:
        raise ValidationError(f"Float parameter '{param_name}' missing 'bounds' field")
    
    bounds = param_spec["bounds"]
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ValidationError(f"Parameter '{param_name}' bounds must be [low, high]")
    
    low, high = bounds
    if not isinstance(low, Number) or not isinstance(high, Number):
        raise ValidationError(f"Parameter '{param_name}' bounds must be numbers")
    
    if low >= high:
        raise ValidationError(f"Parameter '{param_name}' low bound must be < high bound")


def _validate_categorical_param(param_name: str, param_spec: Dict[str, Any]):
    """Validate categorical parameter specification."""
    if "choices" not in param_spec:
        raise ValidationError(f"Categorical parameter '{param_name}' missing 'choices' field")
    
    choices = param_spec["choices"]
    if not isinstance(choices, (list, tuple)):
        raise ValidationError(f"Parameter '{param_name}' choices must be list or tuple")
    
    if len(choices) == 0:
        raise ValidationError(f"Parameter '{param_name}' choices cannot be empty")
    
    # Check for duplicates
    if len(set(choices)) != len(choices):
        raise ValidationError(f"Parameter '{param_name}' choices contain duplicates")


def validate_learner_config(config: Dict[str, Any], learner_name: str) -> bool:
    """
    Validate a learner configuration.
    
    Args:
        config: Configuration dictionary
        learner_name: Name of the learner
        
    Returns:
        bool: True if valid
        
    Raises:
        ValidationError: If configuration is invalid
    """
    if not isinstance(config, dict):
        raise ValidationError(f"Learner config for '{learner_name}' must be dict, got {type(config)}")
    
    # Check for None values (often cause issues)
    for key, value in config.items():
        if value is None:
            raise ValidationError(f"Learner config for '{learner_name}' has None value for '{key}'")
    
    return True


def validate_data_arrays(X, y, task: str = "classification") -> bool:
    """
    Validate input data arrays.
    
    Args:
        X: Feature matrix
        y: Target vector
        task: Task type ("classification" or "regression")
        
    Returns:
        bool: True if valid
        
    Raises:
        ValidationError: If data is invalid
    """
    # Check basic types
    if not hasattr(X, '__len__'):
        raise ValidationError("X must be array-like")
    
    if not hasattr(y, '__len__'):
        raise ValidationError("y must be array-like")
    
    # Convert to numpy arrays for validation
    try:
        X = np.asarray(X)
        y = np.asarray(y)
    except Exception as e:
        raise ValidationError(f"Cannot convert data to arrays: {e}")
    
    # Check dimensions
    if X.ndim != 2:
        raise ValidationError(f"X must be 2D array, got shape {X.shape}")
    
    if y.ndim != 1:
        raise ValidationError(f"y must be 1D array, got shape {y.shape}")
    
    # Check lengths match
    if len(X) != len(y):
        raise ValidationError(f"X and y length mismatch: {len(X)} != {len(y)}")
    
    # Check for empty data
    if len(X) == 0:
        raise ValidationError("Empty dataset provided")
    
    if X.shape[1] == 0:
        raise ValidationError("No features in dataset")
    
    # Check for NaN/inf values
    if np.any(np.isnan(X)):
        raise ValidationError("X contains NaN values")
    
    if np.any(np.isinf(X)):
        raise ValidationError("X contains infinite values")
    
    if np.any(np.isnan(y)):
        raise ValidationError("y contains NaN values")
    
    if np.any(np.isinf(y)):
        raise ValidationError("y contains infinite values")
    
    # Task-specific validation
    if task == "classification":
        # Check if y contains valid class labels
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            raise ValidationError("Classification requires at least 2 classes")
        
        # Check for reasonable number of classes
        if len(unique_classes) > len(y) // 2:
            raise ValidationError("Too many classes relative to sample size")
    
    elif task == "regression":
        # Check that y is numeric
        if not np.issubdtype(y.dtype, np.number):
            raise ValidationError("Regression target must be numeric")
    
    else:
        raise ValidationError(f"Invalid task: {task}")
    
    return True


def validate_trial_config(trial_config: Dict[str, Any]) -> bool:
    """
    Validate trial configuration from hyperparameter optimization.
    
    Args:
        trial_config: Trial configuration dictionary
        
    Returns:
        bool: True if valid
        
    Raises:
        ValidationError: If configuration is invalid
    """
    if not isinstance(trial_config, dict):
        raise ValidationError(f"Trial config must be dict, got {type(trial_config)}")
    
    if "learner_name" not in trial_config:
        raise ValidationError("Trial config missing 'learner_name'")
    
    learner_name = trial_config["learner_name"]
    if not isinstance(learner_name, str):
        raise ValidationError(f"learner_name must be string, got {type(learner_name)}")
    
    if not learner_name.strip():
        raise ValidationError("learner_name cannot be empty")
    
    return True


def validate_learners_dict(learners: Dict[str, Any]) -> bool:
    """
    Validate learners dictionary.
    
    Args:
        learners: Dictionary of learner name -> learner factory
        
    Returns:
        bool: True if valid
        
    Raises:
        ValidationError: If learners dict is invalid
    """
    if not isinstance(learners, dict):
        raise ValidationError(f"Learners must be dict, got {type(learners)}")
    
    if not learners:
        raise ValidationError("Learners dictionary cannot be empty")
    
    for name, factory in learners.items():
        if not isinstance(name, str):
            raise ValidationError(f"Learner name must be string, got {type(name)}")
        
        if not name.strip():
            raise ValidationError("Learner name cannot be empty")
        
        if not callable(factory):
            raise ValidationError(f"Learner '{name}' factory must be callable")
    
    return True


def safe_convert_numeric(value: Any, target_type: type) -> Any:
    """
    Safely convert a value to numeric type.
    
    Args:
        value: Value to convert
        target_type: Target type (int, float)
        
    Returns:
        Converted value
        
    Raises:
        ValidationError: If conversion fails
    """
    try:
        if target_type == int:
            return int(float(value))  # Handle string numbers
        elif target_type == float:
            return float(value)
        else:
            raise ValidationError(f"Unsupported target type: {target_type}")
    except (ValueError, TypeError) as e:
        raise ValidationError(f"Cannot convert {value} to {target_type.__name__}: {e}")


def validate_cv_folds(cv: Optional[int], n_samples: int) -> bool:
    """
    Validate cross-validation fold parameter.
    
    Args:
        cv: Number of CV folds (None means no CV)
        n_samples: Number of samples in dataset
        
    Returns:
        bool: True if valid
        
    Raises:
        ValidationError: If CV parameter is invalid
    """
    if cv is None:
        return True
    
    if not isinstance(cv, int):
        raise ValidationError(f"CV folds must be integer, got {type(cv)}")
    
    if cv < 2:
        raise ValidationError("CV folds must be >= 2")
    
    if cv > n_samples:
        raise ValidationError(f"CV folds ({cv}) cannot exceed number of samples ({n_samples})")
    
    if cv > n_samples // 2:
        raise ValidationError(f"CV folds ({cv}) too large for dataset size ({n_samples})")
    
    return True


def ensure_numpy_array(data) -> np.ndarray:
    """
    Convert data to numpy array with validation.
    
    Args:
        data: Input data (array-like)
        
    Returns:
        np.ndarray: Validated numpy array
        
    Raises:
        ValidationError: If data cannot be converted
    """
    try:
        if isinstance(data, np.ndarray):
            return data
        
        # Convert to numpy array
        arr = np.asarray(data)
        
        # Validate result
        if arr.size == 0:
            raise ValidationError("Empty array provided")
        
        if arr.ndim == 0:
            raise ValidationError("Scalar values not supported, need array-like data")
        
        return arr
        
    except Exception as e:
        raise ValidationError(f"Could not convert data to numpy array: {e}")
