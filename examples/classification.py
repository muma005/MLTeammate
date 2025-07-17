
# examples/classification.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from ml_teammate.interface.api import MLTeammate
from ml_teammate.data.preprocessing import preprocess_data
from ml_teammate.data.resampler import oversample_smote


X, y = load_iris(return_X_y=True)

# Optional preprocessing
X_processed = preprocess_data(X)

# Optional resampling
X_resampled, y_resampled = oversample_smote(X_processed, y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)


automl = MLTeammate(task="classification", n_trials=5, enable_mlflow=False)
automl.fit(X_train, y_train)
preds = automl.predict(X_test)
print("Final predictions:", preds)

