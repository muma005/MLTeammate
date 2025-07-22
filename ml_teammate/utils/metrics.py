# ml_teammate/utils/metrics.py

from sklearn.metrics import accuracy_score, mean_squared_error

def evaluate(y_true, y_pred, task: str = "classification") -> float:
    if task == "classification":
        return accuracy_score(y_true, y_pred)  # ✅ maximize accuracy
    elif task == "regression":
        return -mean_squared_error(y_true, y_pred)  # ✅ maximize (i.e. minimize error)
    else:
        raise ValueError(f"Unknown task: {task}")
