# ml_teammate/learners/__init__.py

# Import existing learners
from .xgboost_learner import XGBoostLearner, get_xgboost_learner
from .lightgbm_learner import LightGBMLearner, get_lightgbm_learner

# Import the new registry system
from .registry import (
    get_learner_registry,
    get_learner,
    get_config_space,
    get_all_learners,
    get_classification_learners,
    get_regression_learners,
    create_learners_dict,
    create_config_space,
    SklearnWrapper,
    LearnerRegistry
)

def get_learner(name):
    """
    Get a learner factory function by name.
    
    This function provides backward compatibility while also supporting
    the new registry system.
    
    Args:
        name: Name of the learner (e.g., "xgboost", "random_forest", "logistic_regression")
    
    Returns:
        Factory function for the learner
    
    Raises:
        ValueError: If the learner is not found
    """
    try:
        # Try the new registry system first
        return get_learner(name)
    except ValueError:
        # Fall back to legacy system for backward compatibility
        if name == "xgboost":
            return get_xgboost_learner
        elif name == "lightgbm":
            return get_lightgbm_learner
        else:
            available = ", ".join(get_all_learners())
            raise ValueError(f"Unknown learner: {name}. Available learners: {available}")

# Export all the new registry functions
__all__ = [
    # Legacy exports for backward compatibility
    'XGBoostLearner', 
    'get_learner', 
    'get_xgboost_learner',
    'LightGBMLearner',
    'get_lightgbm_learner',
    
    # New registry system exports
    'get_learner_registry',
    'get_config_space',
    'get_all_learners',
    'get_classification_learners',
    'get_regression_learners',
    'create_learners_dict',
    'create_config_space',
    'SklearnWrapper',
    'LearnerRegistry'
]  