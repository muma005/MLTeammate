from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import xgboost as xgb

from ml_teammate.automl.controller import AutoMLController
from ml_teammate.search.optuna_search import OptunaSearcher

# --- Load Data ---
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Define Config Space ---
config_space = {
    "xgboost": {
        "max_depth": {"type": "int", "bounds": [2, 4]},
        "learning_rate": {"type": "float", "bounds": [0.01, 0.1]},
        "n_estimators": {"type": "int", "bounds": [10, 50]},
    }
}

# --- Define Learner ---
def xgb_wrapper(config):
    return xgb.XGBClassifier(**config, use_label_encoder=False, eval_metric="mlogloss")

learners = {"xgboost": xgb_wrapper}

# --- Create Searcher ---
searcher = OptunaSearcher(config_space)

# --- Create AutoML Controller with CV ---
automl = AutoMLController(
    learners=learners,
    searcher=searcher,
    config_space=config_space,
    task="classification",
    n_trials=2,
    cv=3  # ✅ Cross-validation enabled
)

# --- Train ---
automl.fit(X_train, y_train)

# --- Predict and Evaluate ---
preds = automl.predict(X_test)

from sklearn.metrics import accuracy_score
print("✅ Test Accuracy:", accuracy_score(y_test, preds))
