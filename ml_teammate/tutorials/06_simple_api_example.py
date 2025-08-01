# tutorials/06_simple_api_example.py
"""
06_simple_api_example.py
------------------------
Demonstrate the pandas-style MLTeammate API that requires ZERO function definitions.

This tutorial shows how users can run AutoML experiments by simply calling methods,
just like using pandas - no custom classes, functions, or configuration needed.

Perfect for Jupyter notebook users and beginners!
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# Import the pandas-style API
from ml_teammate.interface import SimpleAutoML, quick_classification, quick_regression

# ============================================================================
# STEP 1: Explore Available Learners (No Functions!)
# ============================================================================

print("🔍 STEP 1: Explore Available Learners")
print("=" * 50)

# Just create an instance and call the method - no function needed!
automl = SimpleAutoML()
automl.explore_learners()  # Auto-executes and prints results

# ============================================================================
# STEP 2: Simple Classification (No Functions!)
# ============================================================================

print("\n🚀 STEP 2: Simple Classification")
print("=" * 50)

# Generate sample data
print("📊 Creating sample classification dataset...")
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"📈 Dataset shape: {X.shape}")
print(f"🎯 Classes: {np.unique(y)}")

# Method 1: SimpleAutoML with auto-execution
print("\n🔬 Method 1: SimpleAutoML with auto-execution")
automl = SimpleAutoML(
    learners=["random_forest", "logistic_regression", "xgboost"],
    task="classification",
    n_trials=10,
    cv=3
)
automl.quick_classify(X_train, y_train)  # Auto-executes and prints results!

# Test the model
y_pred = automl.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Test Accuracy: {test_accuracy:.4f}")

# ============================================================================
# STEP 3: Quick Classification Function (No Functions!)
# ============================================================================

print("\n🚀 STEP 3: Quick Classification Function")
print("=" * 50)

# Method 2: One-liner function (even simpler!)
print("\n🔬 Method 2: One-liner function")
automl = quick_classification(X_train, y_train)  # Auto-executes and prints results!

# Test the model
y_pred = automl.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Test Accuracy: {test_accuracy:.4f}")

# ============================================================================
# STEP 4: Regression (No Functions!)
# ============================================================================

print("\n📈 STEP 4: Regression")
print("=" * 50)

# Generate regression data
print("📊 Creating sample regression dataset...")
X_reg, y_reg = make_regression(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    random_state=42
)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

print(f"📈 Dataset shape: {X_reg.shape}")

# Method 1: SimpleAutoML with auto-execution
print("\n🔬 Method 1: SimpleAutoML with auto-execution")
automl_reg = SimpleAutoML(
    learners=["random_forest_regressor", "linear_regression", "ridge"],
    task="regression",
    n_trials=10,
    cv=3
)
automl_reg.quick_regress(X_train_reg, y_train_reg)  # Auto-executes and prints results!

# Method 2: One-liner function
print("\n🔬 Method 2: One-liner function")
automl_reg = quick_regression(X_train_reg, y_train_reg)  # Auto-executes and prints results!

# Test the model
y_pred_reg = automl_reg.predict(X_test_reg)
test_r2 = r2_score(y_test_reg, y_pred_reg)
print(f"🎯 Test R² Score: {test_r2:.4f}")

# ============================================================================
# STEP 5: Single Learner (No Functions!)
# ============================================================================

print("\n🎯 STEP 5: Single Learner")
print("=" * 50)

# Generate data
X, y = make_classification(n_samples=800, n_features=15, n_informative=8, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Dataset shape: {X.shape}")

# Use just one learner
print("\n🔬 Using Single Learner (XGBoost):")
automl = SimpleAutoML(learners="xgboost", n_trials=5)
automl.quick_classify(X_train, y_train)  # Auto-executes and prints results!

# Test the model
y_pred = automl.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Test Accuracy: {test_accuracy:.4f}")

# ============================================================================
# STEP 6: MLflow Integration (No Functions!)
# ============================================================================

print("\n📊 STEP 6: MLflow Integration")
print("=" * 50)

# Generate data
X, y = make_classification(n_samples=600, n_features=12, n_informative=6, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Dataset shape: {X.shape}")

# Method 1: SimpleAutoML with MLflow
print("\n🔬 Method 1: SimpleAutoML with MLflow")
automl = SimpleAutoML(
    learners=["random_forest", "logistic_regression"],
    task="classification",
    n_trials=5,
    cv=3,
    use_mlflow=True,
    experiment_name="tutorial_example"
)
automl.quick_classify(X_train, y_train)  # Auto-executes and prints results!

# Method 2: Method chaining (like pandas!)
print("\n🔬 Method 2: Method chaining (like pandas!)")
automl = SimpleAutoML()
automl.with_mlflow(experiment_name="chaining_example").quick_classify(X_train, y_train)

# Test the model
y_pred = automl.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Test Accuracy: {test_accuracy:.4f}")

# ============================================================================
# STEP 7: Smart Defaults (No Functions!)
# ============================================================================

print("\n🧠 STEP 7: Smart Defaults")
print("=" * 50)

# Generate data
X, y = make_classification(n_samples=500, n_features=10, n_informative=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Dataset shape: {X.shape}")

# Let SimpleAutoML auto-configure everything!
print("\n🔬 Auto-configured experiment:")
automl = SimpleAutoML()  # No parameters needed!
automl.quick_classify(X_train, y_train)  # Auto-detects task, configures trials, CV, etc.

# Test the model
y_pred = automl.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Test Accuracy: {test_accuracy:.4f}")

# ============================================================================
# STEP 8: Results Summary (No Functions!)
# ============================================================================

print("\n📋 STEP 8: Results Summary")
print("=" * 50)

# Generate data
X, y = make_classification(n_samples=400, n_features=8, n_informative=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Dataset shape: {X.shape}")

# Run experiment
automl = SimpleAutoML(learners=["random_forest", "logistic_regression"], n_trials=3)
automl.quick_classify(X_train, y_train)

# Get comprehensive results summary
print("\n📋 Results Summary:")
summary = automl.get_results_summary()
for key, value in summary.items():
    print(f"   {key}: {value}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 60)
print("🎉 SIMPLE API TUTORIAL COMPLETED!")
print("=" * 60)
print("✅ You've used MLTeammate with ZERO function definitions!")
print("✅ Just like pandas: import → call methods → get results!")
print("✅ Auto-execution, smart defaults, and method chaining!")
print("\n💡 Key Takeaways:")
print("   • No function definitions needed")
print("   • Auto-execution and printing")
print("   • Method chaining like pandas")
print("   • Smart defaults and auto-detection")
print("   • One-liner functions for ultimate simplicity")
print("\n🚀 Ready to use MLTeammate like pandas!") 