# 02_with_cross_validation.py

# tutorials/02_with_cross_validation.py

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from ml_teammate.automl.controller import AutoMLController
from ml_teammate.search.optuna_search import OptunaSearcher
from ml_teammate.learners.xgboost_learner import get_xgboost_learner
from ml_teammate.search.config_space import xgboost_config
# Prepare dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define learner and config space
learners = {
    "xgboost": get_xgboost_learner
}
config_space = {
    "xgboost": xgboost_config
}

# Set up Optuna Searcher
searcher = OptunaSearcher(config_spaces=config_space)

# Initialize AutoML Controller
automl = AutoMLController(
    learners=learners,
    searcher=searcher,
    config_space=config_space,
    task="classification",
    n_trials=10,
    cv=3  # ✅ Using cross-validation
)

# Run search
automl.fit(X_train, y_train)

# Evaluate
score = automl.score(X_test, y_test)
print(f"Final accuracy: {1.0 - score:.4f}")
