# tutorials/04_add_custom_learner.py
"""
04_add_custom_learner.py
------------------------
Demonstrate how to add custom learners to MLTeammate with ZERO function definitions.

This tutorial shows:
1. How to use pre-built custom learners (no def functions needed!)
2. How to create custom learners using backend methods
3. How to use custom learners with the pandas-style API
4. How the backend handles all the complexity

Perfect for users who want to extend MLTeammate with their own models!
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Import the pandas-style API
from ml_teammate.interface import SimpleAutoML

# ============================================================================
# STEP 1: Explore Available Learners (No Functions!)
# ============================================================================

print("🔍 STEP 1: Explore Available Learners")
print("=" * 50)

# Just call the method - no function needed!
automl = SimpleAutoML()
automl.explore_learners()  # Auto-executes and prints results

# ============================================================================
# STEP 2: Use Pre-Built Custom Learners (No Functions!)
# ============================================================================

print("\n🎯 STEP 2: Use Pre-Built Custom Learners")
print("=" * 50)

# Generate sample data
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

print(f"📊 Dataset shape: {X.shape}")
print(f"🎯 Classes: {np.unique(y)}")

# Method 1: Use pre-built custom learners (no def functions needed!)
print("\n🔬 Method 1: Pre-built custom learners")
automl = SimpleAutoML(
    learners=["custom_rf", "custom_lr", "random_forest"],  # Mix of custom and built-in
    task="classification",
    n_trials=10,
    cv=3
)
automl.quick_classify(X_train, y_train)  # Auto-executes and prints results!

# Test the model
y_pred = automl.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Test Accuracy: {test_accuracy:.4f}")

# Method 2: Use only pre-built custom learners
print("\n🔬 Method 2: Only pre-built custom learners")
automl = SimpleAutoML(learners=["custom_rf", "custom_lr"], n_trials=5)
automl.quick_classify(X_train, y_train)  # Auto-executes and prints results!

# Test the model
y_pred = automl.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Test Accuracy: {test_accuracy:.4f}")

# ============================================================================
# STEP 3: Use Pre-Built Ensemble Learner (No Functions!)
# ============================================================================

print("\n⚡ STEP 3: Use Pre-Built Ensemble Learner")
print("=" * 50)

# Generate data
X, y = make_classification(n_samples=800, n_features=15, n_informative=8, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Dataset shape: {X.shape}")

# Use the pre-built ensemble learner
print("\n🔬 Using the pre-built ensemble learner:")
automl = SimpleAutoML(learners="ensemble", n_trials=5)
automl.quick_classify(X_train, y_train)  # Auto-executes and prints results!

# Test the model
y_pred = automl.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Test Accuracy: {test_accuracy:.4f}")

# ============================================================================
# STEP 4: Create Custom Learners with Backend Methods (No Functions!)
# ============================================================================

print("\n🔧 STEP 4: Create Custom Learners with Backend Methods")
print("=" * 50)

# Generate data
X, y = make_classification(n_samples=600, n_features=12, n_informative=6, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Dataset shape: {X.shape}")

# Method 1: Add custom learner using backend method
print("\n🔬 Method 1: Add custom learner using backend method")
automl = SimpleAutoML()

# Add a custom Random Forest with specific parameters (no def function!)
automl.add_custom_learner(
    name="my_rf",
    model_class=RandomForestClassifier,
    config_space={
        "n_estimators": {"type": "int", "bounds": [100, 500]},
        "max_depth": {"type": "int", "bounds": [5, 20]},
        "min_samples_split": {"type": "int", "bounds": [2, 15]}
    },
    random_state=42,
    criterion="gini"
)

# Add a custom Logistic Regression
automl.add_custom_learner(
    name="my_lr",
    model_class=LogisticRegression,
    config_space={
        "C": {"type": "float", "bounds": [0.01, 100.0]},
        "max_iter": {"type": "int", "bounds": [500, 3000]}
    },
    random_state=42,
    solver="liblinear"
)

# Now use the custom learners
automl.quick_classify(X_train, y_train)  # Auto-executes and prints results!

# Test the model
y_pred = automl.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Test Accuracy: {test_accuracy:.4f}")

# ============================================================================
# STEP 5: Create Ensemble Learners with Backend Methods (No Functions!)
# ============================================================================

print("\n🔗 STEP 5: Create Ensemble Learners with Backend Methods")
print("=" * 50)

# Generate data
X, y = make_classification(n_samples=500, n_features=10, n_informative=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Dataset shape: {X.shape}")

# Method 1: Add ensemble learner using backend method
print("\n🔬 Method 1: Add ensemble learner using backend method")
automl = SimpleAutoML()

# Add an ensemble learner (no def function!)
automl.add_ensemble_learner(
    name="my_ensemble",
    learners=["random_forest", "logistic_regression"],
    config_space={
        "random_forest_n_estimators": {"type": "int", "bounds": [50, 200]},
        "random_forest_max_depth": {"type": "int", "bounds": [3, 15]},
        "logistic_regression_C": {"type": "float", "bounds": [0.1, 10.0]},
        "logistic_regression_max_iter": {"type": "int", "bounds": [100, 1000]}
    }
)

# Use the custom ensemble learner
automl.quick_classify(X_train, y_train)  # Auto-executes and prints results!

# Test the model
y_pred = automl.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Test Accuracy: {test_accuracy:.4f}")

# ============================================================================
# STEP 6: Method Chaining with Custom Learners (No Functions!)
# ============================================================================

print("\n🔗 STEP 6: Method Chaining with Custom Learners")
print("=" * 50)

# Generate data
X, y = make_classification(n_samples=400, n_features=8, n_informative=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Dataset shape: {X.shape}")

# Chain multiple methods with custom learners
print("\n🔬 Method chaining with custom learners:")
automl = SimpleAutoML()

# Add custom learner and configure everything in one chain
automl.add_custom_learner(
    name="chained_rf",
    model_class=RandomForestClassifier,
    config_space={
        "n_estimators": {"type": "int", "bounds": [50, 150]},
        "max_depth": {"type": "int", "bounds": [3, 10]}
    },
    random_state=42
).with_mlflow(experiment_name="custom_learners").with_flaml(time_budget=20).quick_classify(X_train, y_train)

# Test the model
y_pred = automl.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Test Accuracy: {test_accuracy:.4f}")

# ============================================================================
# STEP 7: Compare Pre-Built vs Custom Learners (No Functions!)
# ============================================================================

print("\n📊 STEP 7: Compare Pre-Built vs Custom Learners")
print("=" * 50)

# Generate data
X, y = make_classification(n_samples=350, n_features=6, n_informative=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Dataset shape: {X.shape}")

# Compare different learner combinations
learner_combinations = [
    ("Built-in only", ["random_forest", "logistic_regression"]),
    ("Pre-built custom", ["custom_rf", "custom_lr"]),
    ("Pre-built ensemble", ["ensemble"]),
    ("Mixed", ["random_forest", "custom_rf", "ensemble"])
]

results = {}

for name, learners in learner_combinations:
    print(f"\n🔬 Testing {name}:")
    
    try:
        automl = SimpleAutoML(learners=learners, n_trials=3)
        automl.quick_classify(X_train, y_train)
        
        y_pred = automl.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            "accuracy": accuracy,
            "best_score": automl.best_score,
            "best_learner": automl.best_config.get('learner_name', 'unknown')
        }
        
        print(f"   ✅ Accuracy: {accuracy:.4f}")
        print(f"   📈 Best CV Score: {automl.best_score:.4f}")
        print(f"   🏆 Best Learner: {automl.best_config.get('learner_name', 'unknown')}")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        results[name] = {"error": str(e)}

# Summary
print("\n📋 Comparison Summary:")
for name, result in results.items():
    if "error" not in result:
        print(f"   {name}: {result['accuracy']:.4f} accuracy, best: {result['best_learner']}")
    else:
        print(f"   {name}: Failed - {result['error']}")

# ============================================================================
# STEP 8: Smart Defaults with Custom Learners (No Functions!)
# ============================================================================

print("\n🧠 STEP 8: Smart Defaults with Custom Learners")
print("=" * 50)

# Generate data
X, y = make_classification(n_samples=300, n_features=5, n_informative=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Dataset shape: {X.shape}")

# Let SimpleAutoML auto-configure with custom learners
print("\n🔬 Auto-configured experiment with custom learners:")
automl = SimpleAutoML(learners=["custom_rf", "custom_lr"])  # Just specify learners!
automl.quick_classify(X_train, y_train)  # Auto-detects task, configures trials, CV, etc.

# Test the model
y_pred = automl.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Test Accuracy: {test_accuracy:.4f}")

# ============================================================================
# STEP 9: Results Summary with Custom Learners (No Functions!)
# ============================================================================

print("\n📋 STEP 9: Results Summary with Custom Learners")
print("=" * 50)

# Generate data
X, y = make_classification(n_samples=250, n_features=4, n_informative=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Dataset shape: {X.shape}")

# Run experiment with custom learners
automl = SimpleAutoML(learners=["custom_rf", "custom_lr", "ensemble"], n_trials=2)
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
print("🎉 CUSTOM LEARNER TUTORIAL COMPLETED!")
print("=" * 60)
print("✅ You've created and used custom learners with ZERO function definitions!")
print("✅ Pre-built custom learners available out of the box!")
print("✅ Backend methods for easy custom learner creation!")
print("✅ Pandas-style interface works seamlessly with custom learners!")
print("\n💡 Key Takeaways:")
print("   • No def functions needed anywhere!")
print("   • Pre-built custom learners ready to use")
print("   • Backend methods for easy customization")
print("   • Method chaining and auto-execution work perfectly")
print("   • Easy to extend MLTeammate with your own models")
print("\n🚀 Ready to create custom learners like a pro!")
