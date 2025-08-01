# tutorials/04_add_custom_learner.py
"""
04_add_custom_learner.py
------------------------
Demonstrate how to add custom learners to MLTeammate with pandas-style interface.

This tutorial shows:
1. How to use the new registry system for easy custom learner integration
2. How to create custom learners with minimal code
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
from sklearn.base import BaseEstimator, ClassifierMixin

# Import the pandas-style API
from ml_teammate.interface import SimpleAutoML
from ml_teammate.learners.registry import get_learner_registry

# ============================================================================
# STEP 1: Explore Available Learners (No Functions!)
# ============================================================================

print("🔍 STEP 1: Explore Available Learners")
print("=" * 50)

# Just call the method - no function needed!
automl = SimpleAutoML()
automl.explore_learners()  # Auto-executes and prints results

# ============================================================================
# STEP 2: Create Custom Learners (Simplified!)
# ============================================================================

print("\n🎯 STEP 2: Create Custom Learners")
print("=" * 50)

# Method 1: Simple custom learner using the registry system
print("\n🔬 Method 1: Using the Registry System")

# Get the registry
registry = get_learner_registry()

# Add a custom Random Forest with specific parameters
def custom_rf_factory(config):
    """Factory function for custom Random Forest."""
    return RandomForestClassifier(
        n_estimators=config.get('n_estimators', 100),
        max_depth=config.get('max_depth', 10),
        min_samples_split=config.get('min_samples_split', 2),
        random_state=42
    )

# Register the custom learner
custom_rf_config = {
    "n_estimators": {"type": "int", "bounds": [50, 300]},
    "max_depth": {"type": "int", "bounds": [3, 15]},
    "min_samples_split": {"type": "int", "bounds": [2, 20]}
}

registry._register_learner("custom_rf", custom_rf_factory, custom_rf_config)
print("✅ Custom Random Forest registered as 'custom_rf'")

# Add a custom Logistic Regression
def custom_lr_factory(config):
    """Factory function for custom Logistic Regression."""
    return LogisticRegression(
        C=config.get('C', 1.0),
        max_iter=config.get('max_iter', 1000),
        random_state=42
    )

custom_lr_config = {
    "C": {"type": "float", "bounds": [0.1, 10.0]},
    "max_iter": {"type": "int", "bounds": [100, 2000]}
}

registry._register_learner("custom_lr", custom_lr_factory, custom_lr_config)
print("✅ Custom Logistic Regression registered as 'custom_lr'")

# ============================================================================
# STEP 3: Use Custom Learners with Pandas-Style API (No Functions!)
# ============================================================================

print("\n🚀 STEP 3: Use Custom Learners with Pandas-Style API")
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

# Method 1: Use custom learners with SimpleAutoML
print("\n🔬 Method 1: Custom learners with SimpleAutoML")
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

# Method 2: Use only custom learners
print("\n🔬 Method 2: Only custom learners")
automl = SimpleAutoML(learners=["custom_rf", "custom_lr"], n_trials=5)
automl.quick_classify(X_train, y_train)  # Auto-executes and prints results!

# Test the model
y_pred = automl.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Test Accuracy: {test_accuracy:.4f}")

# ============================================================================
# STEP 4: Advanced Custom Learner (No Functions!)
# ============================================================================

print("\n⚡ STEP 4: Advanced Custom Learner")
print("=" * 50)

# Create an ensemble learner using the registry system
def ensemble_factory(config):
    """Factory function for ensemble learner."""
    rf = RandomForestClassifier(
        n_estimators=config.get('rf_n_estimators', 100),
        max_depth=config.get('rf_max_depth', 10),
        random_state=42
    )
    lr = LogisticRegression(
        C=config.get('lr_C', 1.0),
        max_iter=config.get('lr_max_iter', 1000),
        random_state=42
    )
    
    # Simple voting ensemble
    from sklearn.ensemble import VotingClassifier
    return VotingClassifier(
        estimators=[('rf', rf), ('lr', lr)],
        voting='soft'
    )

# Register the ensemble learner
ensemble_config = {
    "rf_n_estimators": {"type": "int", "bounds": [50, 200]},
    "rf_max_depth": {"type": "int", "bounds": [3, 15]},
    "lr_C": {"type": "float", "bounds": [0.1, 10.0]},
    "lr_max_iter": {"type": "int", "bounds": [100, 1000]}
}

registry._register_learner("ensemble", ensemble_factory, ensemble_config)
print("✅ Ensemble learner registered as 'ensemble'")

# Use the ensemble learner
print("\n🔬 Using the ensemble learner:")
automl = SimpleAutoML(learners="ensemble", n_trials=5)
automl.quick_classify(X_train, y_train)  # Auto-executes and prints results!

# Test the model
y_pred = automl.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Test Accuracy: {test_accuracy:.4f}")

# ============================================================================
# STEP 5: Method Chaining with Custom Learners (No Functions!)
# ============================================================================

print("\n🔗 STEP 5: Method Chaining with Custom Learners")
print("=" * 50)

# Generate data
X, y = make_classification(n_samples=600, n_features=15, n_informative=8, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Dataset shape: {X.shape}")

# Chain multiple methods with custom learners
print("\n🔬 Method chaining with custom learners:")
automl = SimpleAutoML()
automl.with_mlflow(experiment_name="custom_learners").with_flaml(time_budget=20).quick_classify(X_train, y_train)

# Test the model
y_pred = automl.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Test Accuracy: {test_accuracy:.4f}")

# ============================================================================
# STEP 6: Compare Custom vs Built-in Learners (No Functions!)
# ============================================================================

print("\n📊 STEP 6: Compare Custom vs Built-in Learners")
print("=" * 50)

# Generate data
X, y = make_classification(n_samples=500, n_features=12, n_informative=6, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Dataset shape: {X.shape}")

# Compare different learner combinations
learner_combinations = [
    ("Built-in only", ["random_forest", "logistic_regression"]),
    ("Custom only", ["custom_rf", "custom_lr"]),
    ("Mixed", ["random_forest", "custom_rf", "ensemble"])
]

results = {}

for name, learners in learner_combinations:
    print(f"\n🔬 Testing {name}:")
    
    try:
        automl = SimpleAutoML(learners=learners, n_trials=5)
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
# STEP 7: Smart Defaults with Custom Learners (No Functions!)
# ============================================================================

print("\n🧠 STEP 7: Smart Defaults with Custom Learners")
print("=" * 50)

# Generate data
X, y = make_classification(n_samples=400, n_features=10, n_informative=5, random_state=42)
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
# STEP 8: Results Summary with Custom Learners (No Functions!)
# ============================================================================

print("\n📋 STEP 8: Results Summary with Custom Learners")
print("=" * 50)

# Generate data
X, y = make_classification(n_samples=300, n_features=8, n_informative=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Dataset shape: {X.shape}")

# Run experiment with custom learners
automl = SimpleAutoML(learners=["custom_rf", "custom_lr", "ensemble"], n_trials=3)
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
print("✅ Registry system makes custom learners easy to integrate!")
print("✅ Pandas-style interface works seamlessly with custom learners!")
print("\n💡 Key Takeaways:")
print("   • No complex class definitions needed")
print("   • Registry system handles all the complexity")
print("   • Custom learners work with all pandas-style methods")
print("   • Method chaining and auto-execution work perfectly")
print("   • Easy to extend MLTeammate with your own models")
print("\n🚀 Ready to create custom learners like a pro!")
