# 02_with_cross_validation.py

# tutorials/02_with_cross_validation.py

"""
02_with_cross_validation.py
---------------------------
Show how to run MLTeammate with k-fold cross‑validation (cv > 1).
"""

# tutorials/02_with_cross_validation.py
from ml_teammate.automl.controller import AutoMLController
from ml_teammate.search.optuna_search import OptunaSearcher
from ml_teammate.learners.xgboost_learner import XGBoostLearner  # Direct import
from sklearn.datasets import make_classification

# Manually define config space matching your frozen OptunaSearcher format

XGBOOST_CONFIG = {
    "n_estimators": {"type": "int", "bounds": [50, 200]},
    "max_depth": {"type": "int", "bounds": [3, 10]},
    "learning_rate": {"type": "float", "bounds": [0.01, 0.3]}
}

# Data generation
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Initialize controller with current frozen components
controller = AutoMLController(
    learners={"xgboost": XGBoostLearner},  # Using class directly
    searcher=OptunaSearcher({"xgboost": XGBOOST_CONFIG}),
    config_space={"xgboost": XGBOOST_CONFIG},
    task="classification",
    n_trials=10,
    cv=5  # Using existing CV parameter
)

# Execute
controller.fit(X, y)
print(f"Best model score: {controller.best_score:.4f}")