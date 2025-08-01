# ml_teammate/interface/__init__.py

# Import the simplified API for easy access
from .simple_api import (
    SimpleAutoML,
    quick_classification,
    quick_regression,
    list_available_learners,
    get_learner_info
)

# Import existing API
from .api import MLTeammate

# Export all interfaces
__all__ = [
    # Simple API (recommended for most users)
    'SimpleAutoML',
    'quick_classification',
    'quick_regression',
    'list_available_learners',
    'get_learner_info',
    
    # Advanced API
    'MLTeammate'
]
