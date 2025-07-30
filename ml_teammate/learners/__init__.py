#learmers.init__.py
# ml_teammate/learners/__init__.py
from .xgboost_learner import XGBoostLearner, get_xgboost_learner  # NEW: Explicit imports

def get_learner(name):
    """
    Changed to return class directly (Phase 2 improvement)
    while maintaining backward compatibility with factory pattern.
    """
    if name == "xgboost":
        return get_xgboost_learner  # Preserves existing API contract
    
    raise ValueError(f"Unknown learner: {name}")

# NEW: Explicit exports (Principle 4: Module Boundaries)
__all__ = ['XGBoostLearner', 'get_learner', 'get_xgboost_learner']  