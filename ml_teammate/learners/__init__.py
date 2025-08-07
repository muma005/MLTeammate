"""
MLTeammate Learners Module

Provides access to the learner registry system and individual learner implementations.
This module serves as the main entry point for all learner-related functionality.
"""

# Import existing learners for backward compatibility
from .xgboost_learner import XGBoostLearner, get_xgboost_learner
from .lightgbm_learner import LightGBMLearner, get_lightgbm_learner

# Import the new registry system
from .registry import (
    get_learner_registry,
    create_learner,
    list_available_learners,
    get_learner_config_space,
    SklearnWrapper,
    LearnerRegistry
)


def get_learner(name: str):
    """
    Get a learner factory function by name.
    
    This function provides backward compatibility while also supporting
    the new registry system.
    
    Args:
        name: Name of the learner (e.g., "xgboost", "random_forest", "logistic_regression")
    
    Returns:
        Factory function or learner instance
    
    Raises:
        ValueError: If the learner is not found
    """
    try:
        # Try the new registry system first
        return create_learner(name)
    except Exception:
        # Fall back to legacy system for backward compatibility
        if name == "xgboost":
            return get_xgboost_learner
        elif name == "lightgbm":
            return get_lightgbm_learner
        else:
            registry = get_learner_registry()
            available = list(registry._learners.keys())
            raise ValueError(f"Unknown learner: {name}. Available learners: {', '.join(available)}")


def get_all_learners():
    """Get all available learner names."""
    registry = get_learner_registry()
    legacy_learners = ["xgboost", "lightgbm"]
    registry_learners = list(registry._learners.keys())
    return legacy_learners + registry_learners


def get_classification_learners():
    """Get classification learner names."""
    registry = get_learner_registry()
    legacy_clf = ["xgboost", "lightgbm"]  # These support classification
    registry_clf = registry.get_learners_by_task("classification")
    return legacy_clf + registry_clf


def get_regression_learners():
    """Get regression learner names.""" 
    registry = get_learner_registry()
    legacy_reg = ["xgboost", "lightgbm"]  # These support regression
    registry_reg = registry.get_learners_by_task("regression")
    return legacy_reg + registry_reg


def get_config_space(name: str):
    """Get configuration space for a learner."""
    try:
        return get_learner_config_space(name)
    except Exception:
        # Return empty config for legacy learners
        return {}


def create_learners_dict(learner_names):
    """Create a dictionary of learner factory functions."""
    learners = {}
    for name in learner_names:
        learners[name] = get_learner(name)
    return learners


def create_config_space(learner_names):
    """Create configuration spaces for multiple learners."""
    config_spaces = {}
    for name in learner_names:
        config_spaces[name] = get_config_space(name)
    return config_spaces


# Export all functions for backward compatibility
__all__ = [
    # Legacy exports for backward compatibility
    'XGBoostLearner', 
    'get_learner', 
    'get_xgboost_learner',
    'LightGBMLearner',
    'get_lightgbm_learner',
    
    # Registry functions with backward compatible names
    'get_all_learners',
    'get_classification_learners', 
    'get_regression_learners',
    'get_config_space',
    'create_learners_dict',
    'create_config_space',
    
    # New registry system exports
    'get_learner_registry',
    'create_learner',
    'list_available_learners',
    'get_learner_config_space',
    'SklearnWrapper',
    'LearnerRegistry'
]
