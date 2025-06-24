# ml_teammate/utils/metrics.py

from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

def evaluate(y_true, y_pred, task="classification"):
    if task == "classification":
        return accuracy_score(y_true, y_pred)
    elif task == "f1":
        return f1_score(y_true, y_pred, average="macro")
    elif task == "regression":
        return mean_squared_error(y_true, y_pred)
    else:
        raise ValueError(f"Unknown task type: {task}")
