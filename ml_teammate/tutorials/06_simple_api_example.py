# tutorials/06_simple_api_example.py
"""
06_simple_api_example.py
------------------------
Demonstrate simple and user-friendly MLTeammate patterns for beginners.

This tutorial shows the easiest ways to use MLTeammate with minimal configuration,
smart defaults, and one-liner approaches - perfect for quick experiments!

Perfect for Jupyter notebook users and beginners who want simplicity!
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# Import the simple MLTeammate API
from ml_teammate.interface.api import MLTeammate

# ============================================================================
# STEP 1: Simple Default Configuration
# ============================================================================

print("🔍 STEP 1: Simple Default Configuration")
print("=" * 50)

# Show available learners (hardcoded for simplicity)
available_learners = {
    "classification": ["random_forest", "logistic_regression", "svm", "gradient_boosting"],
    "regression": ["random_forest_regressor", "linear_regression", "ridge", "gradient_boosting_regressor"]
}

print("📊 Available Learners:")
for task, learners in available_learners.items():
    print(f"   {task.capitalize()}:")
    for learner in learners:
        print(f"      • {learner}")

print("\n💡 Pro Tip: You can use any combination of these learners!")

# ============================================================================
# STEP 2: One-Liner Classification (Minimal Setup!)
# ============================================================================

print("\n🚀 STEP 2: One-Liner Classification (Minimal Setup!)")
print("=" * 50)

# Generate sample data
print("📊 Creating sample classification dataset...")
X, y = make_classification(
    n_samples=600,
    n_features=15,
    n_informative=8,
    n_redundant=3,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"📈 Dataset shape: {X.shape}")
print(f"🎯 Classes: {np.unique(y)}")

# Method 1: Minimal configuration (almost one-liner!)
print("\n🔬 Method 1: Minimal configuration")
automl = MLTeammate(
    learners=["random_forest", "logistic_regression"],  # Use available learners
    task="classification"
)  # Uses smart defaults!
automl.fit(X_train, y_train)

# Test the model
y_pred = automl.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Best CV Score: {automl.best_score:.4f}")
print(f"🎯 Test Accuracy: {test_accuracy:.4f}")
print(f"🏆 Best Learner: {automl.best_config.get('learner_name', 'unknown')}")

# ============================================================================
# STEP 3: Quick Multi-Learner Setup
# ============================================================================

print("\n🚀 STEP 3: Quick Multi-Learner Setup")
print("=" * 50)

# Method 2: Quick multi-learner setup (still simple!)
print("\n🔬 Method 2: Quick multi-learner comparison")
automl = MLTeammate(
    learners=["random_forest", "logistic_regression", "gradient_boosting"], 
    task="classification"
)
automl.fit(X_train, y_train)

# Test the model
y_pred = automl.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Best CV Score: {automl.best_score:.4f}")
print(f"🎯 Test Accuracy: {test_accuracy:.4f}")
print(f"🏆 Best Learner: {automl.best_config.get('learner_name', 'unknown')}")

# ============================================================================
# STEP 4: Simple Regression (One-Liner Style!)
# ============================================================================

print("\n📈 STEP 4: Simple Regression (One-Liner Style!)")
print("=" * 50)

# Generate regression data
print("📊 Creating sample regression dataset...")
X_reg, y_reg = make_regression(
    n_samples=600,
    n_features=12,
    n_informative=6,
    random_state=42
)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

print(f"📈 Dataset shape: {X_reg.shape}")

# Method 1: Default regression setup
print("\n🔬 Method 1: Default regression")
automl_reg = MLTeammate(
    learners=["random_forest_regressor", "linear_regression"],  # Use available learners
    task="regression"
)  # Auto-selects regression learners!
automl_reg.fit(X_train_reg, y_train_reg)

# Test the model
y_pred_reg = automl_reg.predict(X_test_reg)
test_r2 = r2_score(y_test_reg, y_pred_reg)
print(f"✅ Best CV Score: {automl_reg.best_score:.4f}")
print(f"🎯 Test R² Score: {test_r2:.4f}")
print(f"🏆 Best Learner: {automl_reg.best_config.get('learner_name', 'unknown')}")

# Method 2: Multi-learner regression
print("\n🔬 Method 2: Multi-learner regression")
automl_reg = MLTeammate(
    learners=["random_forest_regressor", "linear_regression", "ridge"], 
    task="regression"
)
automl_reg.fit(X_train_reg, y_train_reg)

y_pred_reg = automl_reg.predict(X_test_reg)
test_r2 = r2_score(y_test_reg, y_pred_reg)
print(f"✅ Best CV Score: {automl_reg.best_score:.4f}")
print(f"🎯 Test R² Score: {test_r2:.4f}")
print(f"🏆 Best Learner: {automl_reg.best_config.get('learner_name', 'unknown')}")

# ============================================================================
# STEP 5: Single Learner (Quick Focus!)
# ============================================================================

print("\n🎯 STEP 5: Single Learner (Quick Focus!)")
print("=" * 50)

# Generate data
X, y = make_classification(n_samples=500, n_features=10, n_informative=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Dataset shape: {X.shape}")

# Focus on just one learner
print("\n🔬 Focus on Random Forest:")
automl = MLTeammate(learners=["random_forest"], task="classification")
automl.fit(X_train, y_train)

# Test the model
y_pred = automl.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Best CV Score: {automl.best_score:.4f}")
print(f"🎯 Test Accuracy: {test_accuracy:.4f}")
print(f"🏆 Best Learner: {automl.best_config.get('learner_name', 'unknown')}")

# ============================================================================
# STEP 6: Quick MLflow Integration
# ============================================================================

print("\n📊 STEP 6: Quick MLflow Integration")
print("=" * 50)

# Generate data
X, y = make_classification(n_samples=400, n_features=8, n_informative=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Dataset shape: {X.shape}")

# Simple MLflow integration
print("\n🔬 MLflow tracking (one parameter!):")
automl = MLTeammate(
    learners=["random_forest", "logistic_regression"],
    task="classification",
    enable_mlflow=True  # Just add this one parameter!
)
automl.fit(X_train, y_train)

# Test the model
y_pred = automl.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Best CV Score: {automl.best_score:.4f}")
print(f"🎯 Test Accuracy: {test_accuracy:.4f}")
print(f"🏆 Best Learner: {automl.best_config.get('learner_name', 'unknown')}")
print(f"📊 MLflow tracking enabled - all trials logged!")

# ============================================================================
# STEP 7: Smart Defaults in Action
# ============================================================================

print("\n🧠 STEP 7: Smart Defaults in Action")
print("=" * 50)

# Generate data
X, y = make_classification(n_samples=300, n_features=6, n_informative=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Dataset shape: {X.shape}")

# Let MLTeammate auto-configure everything!
print("\n🔬 Auto-configured experiment (truly minimal!):")
automl = MLTeammate(
    learners=["random_forest", "logistic_regression"],  # Use available learners
    task="classification"
)  # No complex parameters needed - smart defaults!
automl.fit(X_train, y_train)

# Test the model
y_pred = automl.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Best CV Score: {automl.best_score:.4f}")
print(f"🎯 Test Accuracy: {test_accuracy:.4f}")
print(f"🏆 Best Learner: {automl.best_config.get('learner_name', 'unknown')}")

# Show what was auto-configured
print(f"\n🔧 Auto-configured settings:")
print(f"   • Task: Classification task specified")
print(f"   • Learners: Available classification learners")
print(f"   • Trials: Default number of trials")
print(f"   • CV: Default cross-validation")

# ============================================================================
# STEP 8: Quick Results Summary
# ============================================================================

print("\n📋 STEP 8: Quick Results Summary")
print("=" * 50)

# Generate data
X, y = make_classification(n_samples=250, n_features=5, n_informative=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Dataset shape: {X.shape}")

# Run simple experiment
automl = MLTeammate(learners=["random_forest", "logistic_regression"], task="classification")
automl.fit(X_train, y_train)

# Get comprehensive results summary
print("\n📋 Quick Results Summary:")
summary = automl.summary()
for key, value in summary.items():
    print(f"   {key}: {value}")

# Test accuracy
y_pred = automl.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Final Test Accuracy: {test_accuracy:.4f}")

# ============================================================================
# STEP 9: Comparison of Simple Approaches
# ============================================================================

print("\n⚖️ STEP 9: Comparison of Simple Approaches")
print("=" * 50)

# Generate data for comparison
X, y = make_classification(n_samples=400, n_features=8, n_informative=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Dataset shape: {X.shape}")

# Test different simple approaches
approaches = [
    ("Minimal setup", {"learners": ["random_forest"], "task": "classification"}),
    ("Task + learner", {"learners": ["logistic_regression"], "task": "classification"}),
    ("Multi-learner", {"learners": ["random_forest", "logistic_regression"], "task": "classification"}),
    ("With gradient boost", {"learners": ["random_forest", "gradient_boosting"], "task": "classification"}),
    ("With MLflow", {"learners": ["random_forest"], "task": "classification", "enable_mlflow": True})
]

results = {}

for name, config in approaches:
    print(f"\n🔬 Testing {name}:")
    
    try:
        automl = MLTeammate(**config)
        automl.fit(X_train, y_train)
        
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

# Summary comparison
print(f"\n📊 Simple Approaches Comparison:")
for name, result in results.items():
    if "error" not in result:
        print(f"   {name}: {result['accuracy']:.4f} accuracy, best: {result['best_learner']}")
    else:
        print(f"   {name}: Failed - {result['error']}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 60)
print("🎉 SIMPLE API TUTORIAL COMPLETED!")
print("=" * 60)
print("✅ You've used MLTeammate with minimal configuration!")
print("✅ Smart defaults make it easy - just call fit()!")
print("✅ One-liner approaches for quick experiments!")
print("\n💡 Key Takeaways:")
print("   • Minimal parameters needed - smart defaults work!")
print("   • Task auto-detection for maximum simplicity")
print("   • Single parameter enables major features (MLflow)")
print("   • Works great for quick prototyping and experiments")
print("   • Progressive complexity - start simple, add features as needed")
print("\n🚀 Ready to use MLTeammate with minimal setup!")

if __name__ == "__main__":
    pass 