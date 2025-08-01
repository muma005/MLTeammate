# ml_teammate/search/__init__.py

# Import existing search components
from .optuna_search import OptunaSearcher
from .config_space import lightgbm_config, xgboost_config

# Import new search components
from .flaml_search import (
    FLAMLSearcher,
    FLAMLTimeBudgetSearcher,
    FLAMLResourceAwareSearcher
)

from .eci import (
    EarlyConvergenceIndicator,
    AdaptiveECI,
    MultiObjectiveECI
)

# Export all search components
__all__ = [
    # Core searchers
    'OptunaSearcher',
    'FLAMLSearcher',
    'FLAMLTimeBudgetSearcher',
    'FLAMLResourceAwareSearcher',
    
    # Early convergence indicators
    'EarlyConvergenceIndicator',
    'AdaptiveECI',
    'MultiObjectiveECI',
    
    # Configuration spaces
    'lightgbm_config',
    'xgboost_config'
]


def get_searcher(searcher_type: str, **kwargs):
    """
    Factory function to create searchers by type.
    
    Args:
        searcher_type: Type of searcher ("optuna", "flaml", "flaml_time", "flaml_resource")
        **kwargs: Arguments for the searcher
        
    Returns:
        Configured searcher instance
        
    Raises:
        ValueError: If searcher type is not supported
    """
    if searcher_type == "optuna":
        return OptunaSearcher(**kwargs)
    elif searcher_type == "flaml":
        return FLAMLSearcher(**kwargs)
    elif searcher_type == "flaml_time":
        return FLAMLTimeBudgetSearcher(**kwargs)
    elif searcher_type == "flaml_resource":
        return FLAMLResourceAwareSearcher(**kwargs)
    else:
        available = ["optuna", "flaml", "flaml_time", "flaml_resource"]
        raise ValueError(f"Unknown searcher type '{searcher_type}'. Available: {available}")


def get_eci(eci_type: str = "standard", **kwargs):
    """
    Factory function to create Early Convergence Indicators by type.
    
    Args:
        eci_type: Type of ECI ("standard", "adaptive", "multi_objective")
        **kwargs: Arguments for the ECI
        
    Returns:
        Configured ECI instance
        
    Raises:
        ValueError: If ECI type is not supported
    """
    if eci_type == "standard":
        return EarlyConvergenceIndicator(**kwargs)
    elif eci_type == "adaptive":
        return AdaptiveECI(**kwargs)
    elif eci_type == "multi_objective":
        objectives = kwargs.pop("objectives", ["accuracy"])
        return MultiObjectiveECI(objectives=objectives, **kwargs)
    else:
        available = ["standard", "adaptive", "multi_objective"]
        raise ValueError(f"Unknown ECI type '{eci_type}'. Available: {available}")


def list_available_searchers():
    """
    List all available searcher types.
    
    Returns:
        Dictionary with searcher information
    """
    return {
        "optuna": {
            "description": "Optuna-based hyperparameter optimization",
            "features": ["TPE sampler", "Random sampler", "Multi-objective", "Pruning"],
            "dependencies": ["optuna"]
        },
        "flaml": {
            "description": "FLAML-based hyperparameter optimization",
            "features": ["Time budget", "Resource management", "Early stopping"],
            "dependencies": ["flaml"]
        },
        "flaml_time": {
            "description": "FLAML with time budget focus",
            "features": ["Time-bounded optimization", "Fast convergence"],
            "dependencies": ["flaml"]
        },
        "flaml_resource": {
            "description": "FLAML with resource awareness",
            "features": ["Memory budget", "Computational constraints"],
            "dependencies": ["flaml"]
        }
    }


def list_available_eci_types():
    """
    List all available ECI types.
    
    Returns:
        Dictionary with ECI information
    """
    return {
        "standard": {
            "description": "Standard early convergence detection",
            "methods": ["moving_average", "improvement_rate", "confidence_interval", "plateau_detection"],
            "features": ["Statistical analysis", "Multiple convergence methods"]
        },
        "adaptive": {
            "description": "Adaptive convergence detection",
            "methods": ["All standard methods", "Parameter adaptation"],
            "features": ["Self-tuning parameters", "Performance-based adaptation"]
        },
        "multi_objective": {
            "description": "Multi-objective convergence detection",
            "methods": ["Composite scoring", "Objective-wise analysis"],
            "features": ["Multiple objectives", "Weighted convergence"]
        }
    }
