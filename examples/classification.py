
# examples/classification.py

from sklearn.datasets import load_iris
from ml_teammate.interface.api import MLTeammate

X, y = load_iris(return_X_y=True)

automl = MLTeammate(task="classification", n_trials=5)
automl.fit(X, y)

preds = automl.predict(X)
print("Predictions:", preds)
