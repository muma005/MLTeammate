# tutorials/08_pandas_style_example.py
"""
08_pandas_style_example.py
---------------------------
Demonstrate MLTeammate's pandas-style interface.

This tutorial shows how to use MLTeammate with ZERO function definitions,
just like using pandas where you call methods directly.

Perfect for users who want the simplest possible experience!
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# Import the simplified API - that's all you need!
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
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Dataset shape: {X.shape}")
print(f"🎯 Classes: {np.unique(y)}")

# Method 1: SimpleAutoML with auto-execution
print("\n🔬 Method 1: SimpleAutoML with auto-execution")
automl = SimpleAutoML()
automl.quick_classify(X_train, y_train)  # Auto-executes and prints results

# Method 2: One-liner function (even simpler!)
print("\n🔬 Method 2: One-liner function")
automl = quick_classification(X_train, y_train)  # Auto-executes and prints results

# Test the model
y_pred = automl.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Test Accuracy: {test_accuracy:.4f}")

# ============================================================================
# STEP 3: Regression (No Functions!)
# ============================================================================

print("\n📈 STEP 3: Regression")
print("=" * 50)

# Generate regression data
X_reg, y_reg = make_regression(n_samples=1000, n_features=20, n_informative=10, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

print(f"📊 Dataset shape: {X_reg.shape}")

# Method 1: SimpleAutoML with auto-execution
print("\n🔬 Method 1: SimpleAutoML with auto-execution")
automl_reg = SimpleAutoML()
automl_reg.quick_regress(X_train_reg, y_train_reg)  # Auto-executes and prints results

# Method 2: One-liner function
print("\n🔬 Method 2: One-liner function")
automl_reg = quick_regression(X_train_reg, y_train_reg)  # Auto-executes and prints results

# Test the model
y_pred_reg = automl_reg.predict(X_test_reg)
test_r2 = r2_score(y_test_reg, y_pred_reg)
print(f"🎯 Test R² Score: {test_r2:.4f}")

# ============================================================================
# STEP 4: Method Chaining (No Functions!)
# ============================================================================

print("\n🔗 STEP 4: Method Chaining")
print("=" * 50)

# Generate data
X, y = make_classification(n_samples=500, n_features=15, n_informative=8, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Dataset shape: {X.shape}")

# Chain multiple methods together - just like pandas!
print("\n🔬 Method Chaining Example:")
automl = SimpleAutoML()
automl.with_mlflow(experiment_name="chaining_example").quick_classify(X_train, y_train)

# Another chain example
print("\n🔬 Another Chain Example:")
automl = SimpleAutoML()
automl.with_flaml(time_budget=30).with_mlflow().quick_classify(X_train, y_train)

# ============================================================================
# STEP 5: Smart Defaults (No Functions!)
# ============================================================================

print("\n🧠 STEP 5: Smart Defaults")
print("=" * 50)

# Generate different sized datasets
datasets = [
    ("Small", make_classification(n_samples=100, n_features=10, random_state=42)),
    ("Medium", make_classification(n_samples=1000, n_features=20, random_state=42)),
    ("Large", make_classification(n_samples=5000, n_features=30, random_state=42))
]

for name, (X, y) in datasets:
    print(f"\n📊 {name} Dataset ({X.shape[0]} samples, {X.shape[1]} features):")
    
    # Let SimpleAutoML auto-configure everything!
    automl = SimpleAutoML()  # No parameters needed!
    automl.quick_classify(X, y)  # Auto-detects task, configures trials, CV, etc.

# ============================================================================
# STEP 6: Advanced Features (No Functions!)
# ============================================================================

print("\n⚡ STEP 6: Advanced Features")
print("=" * 50)

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Dataset shape: {X.shape}")

# Advanced configuration with method chaining
print("\n🔬 Advanced Configuration:")
automl = SimpleAutoML()
automl.with_mlflow(experiment_name="advanced_example").with_flaml(time_budget=45).quick_classify(X_train, y_train)

# Get results summary
print("\n📋 Results Summary:")
summary = automl.get_results_summary()
for key, value in summary.items():
    print(f"   {key}: {value}")

# ============================================================================
# STEP 7: Custom Learners (No Functions!)
# ============================================================================

print("\n🎯 STEP 7: Custom Learners")
print("=" * 50)

# Generate data
X, y = make_classification(n_samples=800, n_features=15, n_informative=8, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Dataset shape: {X.shape}")

# Use specific learners
print("\n🔬 Using Specific Learners:")
automl = SimpleAutoML(learners=["random_forest", "logistic_regression"])
automl.quick_classify(X_train, y_train)

# Use single learner
print("\n🔬 Using Single Learner:")
automl = SimpleAutoML(learners="xgboost")
automl.quick_classify(X_train, y_train)

# ============================================================================
# STEP 8: Auto-Detection (No Functions!)
# ============================================================================

print("\n🔍 STEP 8: Auto-Detection")
print("=" * 50)

# Generate both classification and regression data
X_clf, y_clf = make_classification(n_samples=500, n_features=10, random_state=42)
X_reg, y_reg = make_regression(n_samples=500, n_features=10, random_state=42)

print("🔬 Auto-detecting task type:")

# Classification data (few unique values)
print(f"\n📊 Classification data: {X_clf.shape}, unique targets: {len(np.unique(y_clf))}")
automl = SimpleAutoML()  # No task specified!
automl.quick_classify(X_clf, y_clf)  # Auto-detects classification

# Regression data (many unique values)
print(f"\n📊 Regression data: {X_reg.shape}, unique targets: {len(np.unique(y_reg))}")
automl = SimpleAutoML()  # No task specified!
automl.quick_regress(X_reg, y_reg)  # Auto-detects regression

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 60)
print("🎉 PANDAS-STYLE TUTORIAL COMPLETED!")
print("=" * 60)
print("✅ You've used MLTeammate with ZERO function definitions!")
print("✅ Just like pandas: import → call methods → get results!")
print("✅ Auto-detection, smart defaults, and method chaining!")
print("\n💡 Key Takeaways:")
print("   • No function definitions needed")
print("   • Auto-execution and printing")
print("   • Method chaining like pandas")
print("   • Smart defaults and auto-detection")
print("   • One-liner functions for ultimate simplicity")
print("\n🚀 Ready to use MLTeammate like pandas!") 