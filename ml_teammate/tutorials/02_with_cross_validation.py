"""
MLTeammate Tutorial 2: Cross-Validation

This tutorial demonstrates how to use MLTeammate with k-fold cross-validation.
Uses the MLTeammate API interface with cv_folds parameter.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# Use the MLTeammate API interface
from ml_teammate.interface.api import MLTeammate

print("🚀 MLTeammate Tutorial 2: Cross-Validation")
print("=" * 50)

# 1. Create a larger synthetic classification dataset
print("📊 Creating synthetic dataset...")
X, y = make_classification(
    n_samples=1000, 
    n_features=20, 
    n_informative=15,
    n_redundant=3,
    random_state=42
)

print(f"   Dataset shape: {X.shape}")
print(f"   Classes: {len(np.unique(y))}")

# 2. Create AutoML instance with cross-validation
print("\n🤖 Setting up AutoML with 5-fold CV...")
automl = MLTeammate(
    learners=["random_forest", "logistic_regression"],
    task="classification",
    searcher_type="random",
    n_trials=8,
    cv_folds=5,  # Enable 5-fold cross-validation
    random_state=42
)

# 3. Fit the model with cross-validation
print("\n🏋️ Training AutoML models with CV...")
automl.fit(X, y)

# 4. Get results
print(f"\n📈 Cross-validation completed!")
print(f"📈 Best CV Score: {automl.best_score:.4f}")
print(f"🏆 Best learner configuration found")

# 5. Show summary
print("\n📋 Results Summary:")
summary = automl.summary()
for key, value in summary.items():
    print(f"   {key}: {value}")

print(f"\n🔄 Cross-validation benefits:")
print(f"   ✓ More robust performance estimates")
print(f"   ✓ Better generalization assessment") 
print(f"   ✓ Reduced overfitting risk")
print(f"   ✓ Uses all data for training and validation")

print("\n🎉 Tutorial 2 completed successfully!")
print("💡 Cross-validation gives more reliable performance estimates!")
print("💡 Next: Try tutorial 03 for MLflow integration examples")