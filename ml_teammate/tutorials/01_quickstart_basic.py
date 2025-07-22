import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from ml_teammate.automl.controller import AutoMLController
from ml_teammate.search.optuna_search import OptunaSearcher
from ml_teammate.learners import get_learner
from ml_teammate.search.config_space import xgboost_config

# 1. Create synthetic classification dataset
X, y = make_classification(
    n_samples=500,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Define config space
config_space = {"xgboost": xgboost_config}

# 3. Set up the controller
controller = AutoMLController(
    learners={"xgboost": get_learner("xgboost")},
    searcher=OptunaSearcher(config_space),
    config_space=config_space,
    task="classification",
    n_trials=5,
    cv=None
)

# 4. Fit the model
controller.fit(X_train, y_train)

# 5. Predict on test set
preds = controller.predict(X_test)

# 6. Score
acc = accuracy_score(y_test, preds)
print(f"\n✅ Test Accuracy: {acc:.4f}")
