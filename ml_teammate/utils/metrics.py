# ml_teammate/utils/metrics.py

from sklearn.metrics import accuracy_score, mean_squared_error

def evaluate(y_true, y_pred, task: str = "classification") -> float:
    if task == "classification":
        # Lower is better for optimization
        return 1.0 - accuracy_score(y_true, y_pred)
    elif task == "regression":
        return mean_squared_error(y_true, y_pred)
    else:
        raise ValueError(f"Unknown task: {task}")

