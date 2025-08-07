"""
MLTeammate Tutorial 3: MLflow Integration

This tutorial demonstrates how to use MLTeammate with MLflow experiment tracking.
Uses the MLTeammate API interface with MLflow integration enabled.
"""

import os
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Use the MLTeammate API interface
from ml_teammate.interface.api import MLTeammate

def create_feature_importance_plot(model, feature_names=None, save_path="./feature_importance.png"):
    """Create and save feature importance plot."""
    try:
        import matplotlib.pyplot as plt
        
        # Get feature importance from the wrapped model
        if hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
            importances = model.model.feature_importances_
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            print("⚠️ Model doesn't have feature_importances_ attribute")
            return None
        
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        indices = np.argsort(importances)[::-1]
        plt.title("Feature Importance")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Feature importance plot saved to: {save_path}")
        return save_path
        
    except ImportError:
        print("⚠️ matplotlib not available. Skipping feature importance plot.")
        return None

def main():
    print("🚀 MLTeammate Tutorial 3: MLflow Integration")
    print("=" * 50)
    
    # Create synthetic dataset
    print("🔬 Creating synthetic classification dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"📊 Dataset shape: {X.shape}")
    print(f"🎯 Classes: {np.unique(y)}")
    
    # Create AutoML instance with MLflow enabled
    print("\n🤖 Setting up AutoML with MLflow tracking...")
    automl = MLTeammate(
        learners=["random_forest", "gradient_boosting"],  # Use available learners
        task="classification",
        searcher_type="random",
        n_trials=8,
        cv_folds=3,
        enable_mlflow=True,  # Enable MLflow tracking
        random_state=42
    )
    
    print("🚀 Starting AutoML experiment with MLflow tracking...")
    print("📋 This will log experiment data to MLflow")
    
    # Fit the model
    print("\n🏋️ Training AutoML models...")
    automl.fit(X_train, y_train)
    
    # Make predictions
    print("\n🔮 Making predictions...")
    y_pred = automl.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    # Create and save feature importance plot
    print("\n📊 Creating feature importance plot...")
    if automl.best_model:
        plot_path = create_feature_importance_plot(
            automl.best_model,
            feature_names=[f"feature_{i}" for i in range(X.shape[1])]
        )
    
    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"\n🎉 Experiment completed!")
    print(f"📈 Best CV Score: {automl.best_score:.4f}")
    print(f"🎯 Test Accuracy: {test_accuracy:.4f}")
    print(f"🏆 Best Configuration: {automl.best_config}")
    
    # Show summary
    print("\n� Results Summary:")
    summary = automl.summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    # MLflow information
    print(f"\n📊 MLflow Integration:")
    print(f"   ✓ Experiment tracking enabled")
    print(f"   ✓ Trial parameters and metrics logged")
    print(f"   ✓ Best model performance recorded")
    print(f"   ✓ Cross-validation results tracked")
    
    print(f"\n� MLflow Benefits:")
    print(f"   • Automatic experiment tracking")
    print(f"   • Parameter and metric logging")
    print(f"   • Model performance comparison")
    print(f"   • Reproducible experiments")
    
    print(f"\n🎯 To view results in MLflow UI:")
    print(f"   Run: mlflow ui")
    print(f"   Then visit: http://localhost:5000")

if __name__ == "__main__":
    main()