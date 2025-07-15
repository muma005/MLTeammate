
# examples/classification.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from ml_teammate.interface.api import MLTeammate

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

automl = MLTeammate(task="classification", n_trials=5, enable_mlflow=False)
automl.fit(X_train, y_train)
preds = automl.predict(X_test)
print("Final predictions:", preds)

