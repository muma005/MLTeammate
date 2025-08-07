"""
MLTeammate Tutorial 1: Quickstart Basic

This tutorial demonstrates the simplest way to use MLTeammate for AutoML.
Uses the MLTeammate API interface that's compatible with frozen phases.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Use the MLTeammate API interface instead of SimpleAutoML
from ml_teammate.interface.api import MLTeammate

print("🚀 MLTeammate Tutorial 1: Quickstart Basic")
print("=" * 50)

# 1. Create synthetic classification dataset
print("📊 Creating synthetic dataset...")
X, y = make_classification(
    n_samples=500,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"   Training data: {X_train.shape}")
print(f"   Test data: {X_test.shape}")

# 2. Create AutoML instance using MLTeammate API
print("\n🤖 Setting up AutoML...")
automl = MLTeammate(
    learners=["random_forest", "logistic_regression"],  # Use available learners
    task="classification",
    n_trials=5
)

# 3. Fit the model
print("\n🏋️ Training AutoML models...")
automl.fit(X_train, y_train)

# 4. Get results
print(f"\n📈 Best model: {automl.get_best_model()}")
print(f"📈 Best score available from controller")

# 5. Predict on test set
print("\n🔮 Making predictions...")
preds = automl.predict(X_test)

# 6. Calculate test accuracy
test_accuracy = accuracy_score(y_test, preds)
print(f"\n✅ Test Accuracy: {test_accuracy:.4f}")

# 7. Show summary
print("\n📋 Results Summary:")
summary = automl.summary()
for key, value in summary.items():
    print(f"   {key}: {value}")

print("\n🎉 Tutorial 1 completed successfully!")
print("💡 Next: Try tutorial 02 for cross-validation examples")
