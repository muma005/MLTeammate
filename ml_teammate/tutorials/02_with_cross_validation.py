# 02_with_cross_validation.py

# tutorials/02_with_cross_validation.py

"""
02_with_cross_validation.py
---------------------------
Show how to run MLTeammate with k-fold cross‑validation (cv > 1).
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from ml_teammate.automl.controller import AutoMLController
from ml_teammate.search.optuna_search import OptunaSearcher
from ml_teammate.learners.xgboost_learner import get_xgboost_learner

# 1) Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2) Define the per-learner config space (MUST match OptunaSearcher schema)
config_spaces = {
    "xgboost": {
        "max_depth": {"type": "int", "bounds": [3, 8]},
        "learning_rate": {"type": "float", "bounds": [0.01, 0.3]},
        "n_estimators": {"type": "int", "bounds": [50, 300]},
    }
}

# 3) Registry of learners (name -> factory/callable that accepts config and returns a model)
learners = {
    "xgboost": get_xgboost_learner
}

# 4) Instantiate the searcher
searcher = OptunaSearcher(config_spaces=config_spaces)

# 5) Build the controller with CV enabled (e.g., 5-fold)
automl = AutoMLController(
    learners=learners,
    searcher=searcher,
    config_space=config_spaces,
    task="classification",
    n_trials=5,
    cv=5,                 # ✅ turn on cross-validation
    callbacks=None,
    mlflow_helper=None
)

# 6) Fit, predict, score
automl.fit(X_train, y_train)
test_score = automl.score(X_test, y_test)

print("\n==================== RESULTS ====================")
print(f"Best (CV) training score found: {automl.best_score:.4f}")
print(f"Test score: {test_score:.4f}")
