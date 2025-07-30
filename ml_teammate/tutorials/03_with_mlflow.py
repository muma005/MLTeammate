# tutorials/03_with_mlflow.py
"""
03_with_mlflow.py
-----------------
Demonstrate MLTeammate with enhanced MLflow experiment tracking and artifact management.
Shows professional-grade experiment tracking with nested runs for trial-level granularity.
"""

import os
import mlflow
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from ml_teammate.automl.controller import AutoMLController
from ml_teammate.search.optuna_search import OptunaSearcher
from ml_teammate.learners.xgboost_learner import XGBoostLearner
from ml_teammate.automl.callbacks import LoggerCallback, ProgressCallback, ArtifactCallback
from ml_teammate.experiments.mlflow_helper import MLflowHelper

def create_feature_importance_plot(model, feature_names=None, save_path="./feature_importance.png"):
    """Create and save feature importance plot."""
    try:
        import matplotlib.pyplot as plt
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
            importances = model.model.feature_importances_
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
    # Set up MLflow tracking
    os.makedirs("./mlruns", exist_ok=True)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("mlteammate_tutorial")
    
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
    
    # Define configuration space
    config_space = {
        "xgboost": {
            "n_estimators": {"type": "int", "bounds": [50, 200]},
            "max_depth": {"type": "int", "bounds": [3, 10]},
            "learning_rate": {"type": "float", "bounds": [0.01, 0.3]},
            "subsample": {"type": "float", "bounds": [0.6, 1.0]},
            "colsample_bytree": {"type": "float", "bounds": [0.6, 1.0]}
        }
    }
    
    # Create callbacks with enhanced MLflow support
    callbacks = [
        LoggerCallback(
            use_mlflow=True,
            log_level="INFO",
            experiment_name="mlteammate_tutorial"
        ),
        ProgressCallback(total_trials=10, patience=3),
        ArtifactCallback(
            save_best_model=True,
            save_configs=True,
            output_dir="./mlteammate_artifacts"
        )
    ]
    
    # Initialize MLflow helper (optional - callbacks handle MLflow internally)
    mlflow_helper = MLflowHelper(
        experiment_name="mlteammate_tutorial",
        tracking_uri="file:./mlruns"
    )
    
    print("🚀 Starting AutoML experiment with enhanced MLflow tracking...")
    print("📋 This will create nested runs for each trial with detailed tracking")
    
    # Create and run AutoML controller
    controller = AutoMLController(
        learners={"xgboost": XGBoostLearner},
        searcher=OptunaSearcher(config_space),
        config_space=config_space,
        task="classification",
        n_trials=10,
        cv=3,
        callbacks=callbacks,
        mlflow_helper=mlflow_helper  # Optional - callbacks handle MLflow
    )
    
    # Fit the model
    controller.fit(X_train, y_train)
    
    # Make predictions
    y_pred = controller.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    # Create and save feature importance plot
    if controller.best_model:
        plot_path = create_feature_importance_plot(
            controller.best_model,
            feature_names=[f"feature_{i}" for i in range(X.shape[1])]
        )
        
        # Log the plot as an experiment artifact
        if plot_path and hasattr(controller, 'callbacks'):
            for callback in controller.callbacks:
                if hasattr(callback, 'mlflow') and callback.mlflow:
                    try:
                        callback.mlflow.log_experiment_artifact(plot_path, "feature_importance.png")
                        print(f"📊 Feature importance plot logged to MLflow")
                    except Exception as e:
                        print(f"⚠️ Failed to log plot to MLflow: {e}")
    
    # Log classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Save the best model
    if hasattr(controller, 'best_model') and controller.best_model:
        try:
            import joblib
            model_path = "./best_model.pkl"
            joblib.dump(controller.best_model, model_path)
            
            # Log model as experiment artifact
            for callback in controller.callbacks:
                if hasattr(callback, 'mlflow') and callback.mlflow:
                    try:
                        callback.mlflow.log_experiment_artifact(model_path, "best_model.pkl")
                        print(f"💾 Best model saved and logged to MLflow")
                    except Exception as e:
                        print(f"⚠️ Failed to log model to MLflow: {e}")
            
            # Clean up
            os.remove(model_path)
        except ImportError:
            print("⚠️ joblib not available. Skipping model save.")
    
    print(f"\n🎉 Experiment completed!")
    print(f"📈 Best CV Score: {controller.best_score:.4f}")
    print(f"🎯 Test Accuracy: {test_accuracy:.4f}")
    print(f"📊 Best Configuration: {controller.searcher.get_best()}")
    print(f"📁 MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"🔗 View results: mlflow ui --backend-store-uri {mlflow.get_tracking_uri()}")
    
    # Show MLflow experiment structure
    print(f"\n📋 MLflow Experiment Structure:")
    print(f"   Experiment: mlteammate_tutorial")
    print(f"   ├── Parent Run: Contains experiment summary and best results")
    print(f"   ├── Trial 1: Individual trial with parameters and metrics")
    print(f"   ├── Trial 2: Individual trial with parameters and metrics")
    print(f"   └── ... (one nested run per trial)")
    print(f"\n💡 Benefits of this structure:")
    print(f"   • Each trial has its own run with detailed parameters")
    print(f"   • Easy to compare trials within the experiment")
    print(f"   • Professional-grade experiment tracking")
    print(f"   • Rich metadata and artifact management")

if __name__ == "__main__":
    main()