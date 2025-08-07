#!/usr/bin/env python3
"""
MLTeammate Command Line Interface

Provides a comprehensive CLI for running AutoML experiments from the command line.
Features include:
- Simple one-command AutoML runs
- Advanced configuration options
- Learner discovery and validation
- Results export and visualization
- Integration with MLflow
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml_teammate.interface.simple_api import SimpleAutoML, get_available_learners_by_task
from ml_teammate.interface.api import MLTeammate


def create_parser():
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="MLTeammate AutoML Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick classification with auto-detection
  mlteammate run data.csv target_column

  # Specify learners and trials
  mlteammate run data.csv target_column --learners xgboost lightgbm --trials 20

  # Regression with MLflow tracking
  mlteammate run data.csv price --task regression --mlflow --experiment house_prices

  # List available learners
  mlteammate list-learners

  # Validate a dataset
  mlteammate validate data.csv target_column
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run AutoML experiment')
    run_parser.add_argument('data_file', help='Path to CSV data file')
    run_parser.add_argument('target_column', help='Name of target column')
    run_parser.add_argument('--task', choices=['classification', 'regression', 'auto'], 
                           default='auto', help='Type of ML task (default: auto-detect)')
    run_parser.add_argument('--learners', nargs='+', 
                           help='List of learners to use (default: auto-select)')
    run_parser.add_argument('--trials', type=int, 
                           help='Number of optimization trials (default: auto-configure)')
    run_parser.add_argument('--cv', type=int, 
                           help='Number of CV folds (default: auto-configure)')
    run_parser.add_argument('--output', '-o', default='./mlteammate_results',
                           help='Output directory for results (default: ./mlteammate_results)')
    run_parser.add_argument('--mlflow', action='store_true',
                           help='Enable MLflow experiment tracking')
    run_parser.add_argument('--experiment', help='MLflow experiment name')
    run_parser.add_argument('--searcher', choices=['random', 'flaml'],
                           default='random', help='Search strategy (default: random)')
    run_parser.add_argument('--time-budget', type=int,
                           help='Time budget for FLAML searcher (seconds)')
    run_parser.add_argument('--quiet', '-q', action='store_true',
                           help='Suppress progress output')
    run_parser.add_argument('--verbose', '-v', action='store_true',
                           help='Enable verbose logging')
    
    # List learners command
    list_parser = subparsers.add_parser('list-learners', help='List available learners')
    list_parser.add_argument('--task', choices=['classification', 'regression', 'all'],
                            default='all', help='Filter by task type')
    list_parser.add_argument('--format', choices=['table', 'json', 'list'],
                            default='table', help='Output format')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate dataset')
    validate_parser.add_argument('data_file', help='Path to CSV data file')
    validate_parser.add_argument('target_column', help='Name of target column')
    validate_parser.add_argument('--checks', nargs='+', 
                                choices=['missing', 'duplicates', 'types', 'balance', 'all'],
                                default=['all'], help='Validation checks to perform')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Get information about MLTeammate')
    info_parser.add_argument('--version', action='store_true', help='Show version info')
    info_parser.add_argument('--config', action='store_true', help='Show configuration')
    
    return parser


def load_data(data_file: str, target_column: str):
    """Load and validate data from CSV file."""
    try:
        # Load data
        data = pd.read_csv(data_file)
        
        # Check if target column exists
        if target_column not in data.columns:
            available_cols = ", ".join(data.columns.tolist())
            raise ValueError(f"Target column '{target_column}' not found. Available columns: {available_cols}")
        
        # Split features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        return X, y, data
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)


def detect_task(y):
    """Auto-detect task type from target variable."""
    unique_values = y.nunique()
    total_values = len(y)
    
    # If few unique values relative to sample size, likely classification
    if unique_values <= min(10, total_values * 0.1):
        return "classification"
    else:
        return "regression"


def run_automl(args):
    """Run AutoML experiment based on CLI arguments."""
    print("🚀 MLTeammate AutoML Experiment")
    print("=" * 50)
    
    # Load data
    print(f"📂 Loading data from: {args.data_file}")
    X, y, data = load_data(args.data_file, args.target_column)
    print(f"✅ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Auto-detect task if needed
    task = args.task
    if task == 'auto':
        task = detect_task(y)
        print(f"🔍 Auto-detected task: {task}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir.absolute()}")
    
    # Configure AutoML
    automl_kwargs = {
        'task': task,
        'use_mlflow': args.mlflow,
        'save_artifacts': True,
        'output_dir': str(output_dir),
        'log_level': 'DEBUG' if args.verbose else 'WARNING' if args.quiet else 'INFO'
    }
    
    if args.learners:
        automl_kwargs['learners'] = args.learners
        print(f"🧠 Using learners: {args.learners}")
    
    if args.trials:
        automl_kwargs['n_trials'] = args.trials
        print(f"⚙️ Optimization trials: {args.trials}")
    
    if args.cv:
        automl_kwargs['cv'] = args.cv
        print(f"🔄 Cross-validation folds: {args.cv}")
    
    if args.experiment:
        automl_kwargs['experiment_name'] = args.experiment
        print(f"📊 MLflow experiment: {args.experiment}")
    
    # Create and run AutoML
    try:
        print("\n🏃 Starting AutoML experiment...")
        
        if args.searcher == 'flaml':
            automl = SimpleAutoML(**automl_kwargs).with_flaml(
                time_budget=args.time_budget or 60
            )
        else:
            automl = SimpleAutoML(**automl_kwargs)
        
        # Fit the model
        automl.fit(X, y)
        
        # Get results
        results = automl.get_results_summary()
        
        # Print results
        print("\n🎉 Experiment completed successfully!")
        print("=" * 50)
        print(f"🏆 Best Score: {results['best_score']:.4f}")
        print(f"🧠 Best Model: {results['best_model']}")
        print(f"⚙️ Best Config: {json.dumps(results['best_config'], indent=2)}")
        
        # Save results
        results_file = output_dir / 'results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"💾 Results saved to: {results_file}")
        
        # Save model
        import joblib
        model_file = output_dir / 'best_model.pkl'
        joblib.dump(automl.best_model, model_file)
        print(f"🤖 Model saved to: {model_file}")
        
        if args.mlflow:
            print(f"📊 Results logged to MLflow experiment: {results['experiment_name']}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


def list_learners(args):
    """List available learners."""
    learners = get_available_learners_by_task()
    
    if args.format == 'json':
        print(json.dumps(learners, indent=2))
        return
    
    if args.format == 'list':
        if args.task == 'all':
            for learner in learners['all']:
                print(learner)
        else:
            for learner in learners[args.task]:
                print(learner)
        return
    
    # Table format
    print("🧠 Available MLTeammate Learners")
    print("=" * 50)
    
    if args.task in ['classification', 'all']:
        print("\n📊 Classification Learners:")
        for i, learner in enumerate(learners['classification'], 1):
            print(f"   {i:2d}. {learner}")
    
    if args.task in ['regression', 'all']:
        print("\n📈 Regression Learners:")
        for i, learner in enumerate(learners['regression'], 1):
            print(f"   {i:2d}. {learner}")
    
    if args.task == 'all':
        print(f"\n📋 Total Available Learners: {len(learners['all'])}")
        print("💡 Use any learner name with the --learners option")


def validate_data(args):
    """Validate dataset for AutoML readiness."""
    print("🔍 MLTeammate Data Validation")
    print("=" * 40)
    
    # Load data
    X, y, data = load_data(args.data_file, args.target_column)
    
    # Perform validation checks
    checks = args.checks
    if 'all' in checks:
        checks = ['missing', 'duplicates', 'types', 'balance']
    
    issues_found = 0
    
    print(f"\n📊 Dataset Overview:")
    print(f"   Samples: {len(data)}")
    print(f"   Features: {len(X.columns)}")
    print(f"   Target: {args.target_column}")
    
    if 'missing' in checks:
        print(f"\n🔍 Missing Values Check:")
        missing = data.isnull().sum()
        if missing.sum() == 0:
            print("   ✅ No missing values found")
        else:
            print("   ⚠️ Missing values detected:")
            for col, count in missing[missing > 0].items():
                percentage = (count / len(data)) * 100
                print(f"      {col}: {count} ({percentage:.1f}%)")
                if percentage > 20:
                    issues_found += 1
    
    if 'duplicates' in checks:
        print(f"\n🔍 Duplicate Rows Check:")
        duplicates = data.duplicated().sum()
        if duplicates == 0:
            print("   ✅ No duplicate rows found")
        else:
            percentage = (duplicates / len(data)) * 100
            print(f"   ⚠️ {duplicates} duplicate rows ({percentage:.1f}%)")
            if percentage > 5:
                issues_found += 1
    
    if 'types' in checks:
        print(f"\n🔍 Data Types Check:")
        print("   📊 Feature types:")
        for dtype, count in X.dtypes.value_counts().items():
            print(f"      {dtype}: {count} columns")
        
        print(f"   🎯 Target type: {y.dtype}")
        
        # Check for potential issues
        if X.select_dtypes(include=['object']).shape[1] > 0:
            print("   ⚠️ Categorical features detected - ensure proper encoding")
    
    if 'balance' in checks:
        print(f"\n🔍 Target Balance Check:")
        if detect_task(y) == 'classification':
            value_counts = y.value_counts()
            print("   📊 Class distribution:")
            for cls, count in value_counts.items():
                percentage = (count / len(y)) * 100
                print(f"      {cls}: {count} ({percentage:.1f}%)")
            
            # Check for severe imbalance
            min_class_pct = (value_counts.min() / len(y)) * 100
            if min_class_pct < 5:
                print("   ⚠️ Severe class imbalance detected")
                issues_found += 1
        else:
            print(f"   📈 Target statistics:")
            print(f"      Mean: {y.mean():.3f}")
            print(f"      Std:  {y.std():.3f}")
            print(f"      Min:  {y.min():.3f}")
            print(f"      Max:  {y.max():.3f}")
    
    # Summary
    print(f"\n🎯 Validation Summary:")
    if issues_found == 0:
        print("   ✅ Dataset looks good for AutoML!")
    else:
        print(f"   ⚠️ {issues_found} potential issues found")
        print("   💡 Consider data preprocessing before running AutoML")
    
    return issues_found == 0


def show_info(args):
    """Show MLTeammate information."""
    if args.version:
        from ml_teammate.interface import __version__
        print(f"MLTeammate version {__version__}")
        return
    
    if args.config:
        print("🔧 MLTeammate Configuration")
        print("=" * 30)
        print(f"Python version: {sys.version.split()[0]}")
        print(f"Working directory: {os.getcwd()}")
        print(f"MLTeammate path: {project_root}")
        
        # Check for optional dependencies
        try:
            import mlflow
            print(f"MLflow: {mlflow.__version__} ✅")
        except ImportError:
            print("MLflow: Not installed ❌")
        
        try:
            import optuna
            print(f"Optuna: {optuna.__version__} ✅")
        except ImportError:
            print("Optuna: Not installed ❌")
        
        try:
            import flaml
            print(f"FLAML: {flaml.__version__} ✅")
        except ImportError:
            print("FLAML: Not installed ❌")
        
        return
    
    # Default info
    print("🤖 MLTeammate AutoML Framework")
    print("=" * 35)
    print("A user-friendly AutoML framework for rapid machine learning experiments.")
    print("\nCommands:")
    print("  run          - Run AutoML experiment")
    print("  list-learners - List available learners")
    print("  validate     - Validate dataset")
    print("  info         - Show information")
    print("\nFor detailed help: mlteammate <command> --help")


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    try:
        if args.command == 'run':
            success = run_automl(args)
            sys.exit(0 if success else 1)
        elif args.command == 'list-learners':
            list_learners(args)
        elif args.command == 'validate':
            success = validate_data(args)
            sys.exit(0 if success else 1)
        elif args.command == 'info':
            show_info(args)
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\n⏹️ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
