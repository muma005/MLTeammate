"""
MLTeammate Search Module

Phase 4: Clean, modern hyperparameter optimization system.

Provides:
- Multiple search algorithms (Optuna, FLAML, Random)
- Consistent interfaces across all searchers
- Integration with frozen Phase 3 learner registry
- Comprehensive result tracking and analysis
- Factory pattern for easy searcher creation
"""

from typing import List

# Import base classes
from .base import (
    BaseSearcher, SearchResult, 
    validate_config_against_space, create_searcher
)

# Import specific searchers
from .random_search import RandomSearcher, create_random_searcher

# Conditional imports for optional dependencies
try:
    from .optuna_search import OptunaSearcher, create_optuna_searcher
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    OptunaSearcher = None
    create_optuna_searcher = None

try:
    from .flaml_search import FLAMLSearcher, create_flaml_searcher
    FLAML_AVAILABLE = True
except ImportError:
    FLAML_AVAILABLE = False
    FLAMLSearcher = None
    create_flaml_searcher = None

# Availability checks
def check_optuna_available() -> bool:
    """Check if Optuna is available."""
    return OPTUNA_AVAILABLE

def check_flaml_available() -> bool:
    """Check if FLAML is available."""
    return FLAML_AVAILABLE

def get_available_searchers() -> List[str]:
    """Get list of available searcher types."""
    searchers = ["random"]
    
    if OPTUNA_AVAILABLE:
        searchers.append("optuna")
    
    if FLAML_AVAILABLE:
        searchers.append("flaml")
    
    return searchers

# Convenience functions
def create_searcher_by_name(searcher_type: str, learner_names, task: str = "classification", **kwargs):
    """
    Create searcher by name with availability checking.
    
    Args:
        searcher_type: Type of searcher ("optuna", "flaml", "random")
        learner_names: List of learner names
        task: Task type
        **kwargs: Additional arguments
        
    Returns:
        BaseSearcher: Configured searcher instance
        
    Raises:
        ValueError: If searcher type is not available
    """
    searcher_type = searcher_type.lower()
    
    if searcher_type == "random":
        return RandomSearcher(learner_names, task, **kwargs)
    
    elif searcher_type == "optuna":
        if not OPTUNA_AVAILABLE:
            raise ValueError("Optuna is not available. Install with: pip install optuna")
        return OptunaSearcher(learner_names, task, **kwargs)
    
    elif searcher_type == "flaml":
        if not FLAML_AVAILABLE:
            raise ValueError("FLAML is not available. Install with: pip install flaml")
        return FLAMLSearcher(learner_names, task, **kwargs)
    
    else:
        available = get_available_searchers()
        raise ValueError(f"Unknown searcher type: {searcher_type}. Available: {available}")

# Export everything that's available
__all__ = [
    # Base classes (always available)
    "BaseSearcher",
    "SearchResult", 
    "validate_config_against_space",
    "create_searcher",
    
    # Random search (always available)
    "RandomSearcher",
    "create_random_searcher",
    
    # Utility functions
    "check_optuna_available",
    "check_flaml_available", 
    "get_available_searchers",
    "create_searcher_by_name"
]

# Add conditional exports
if OPTUNA_AVAILABLE:
    __all__.extend(["OptunaSearcher", "create_optuna_searcher"])

if FLAML_AVAILABLE:
    __all__.extend(["FLAMLSearcher", "create_flaml_searcher"])
