# 📦 Installation Guide

This guide covers all the ways to install and set up MLTeammate for different use cases.

---

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.8 or higher
- **pip**: Latest version recommended
- **Git**: For cloning the repository

### Option 1: Clone and Install (Recommended for Development)

```bash
# Clone the repository
git clone https://github.com/yourusername/ml_teammate.git
cd ml_teammate

# Install in editable mode for development
pip install -e .

# Install additional dependencies for full functionality
pip install -r requirements.txt
```

### Option 2: Direct Installation

```bash
# Install directly from GitHub
pip install git+https://github.com/yourusername/ml_teammate.git
```

---

## 🔧 Detailed Installation

### Step 1: Environment Setup

We recommend using a virtual environment:

```bash
# Create virtual environment
python -m venv mlteammate_env

# Activate on Windows
mlteammate_env\Scripts\activate

# Activate on macOS/Linux
source mlteammate_env/bin/activate
```

### Step 2: Install Dependencies

MLTeammate has different dependency levels:

#### Core Dependencies (Required)
```bash
pip install numpy scikit-learn optuna
```

#### Full Installation (Recommended)
```bash
pip install numpy scikit-learn optuna xgboost lightgbm mlflow matplotlib joblib
```

#### Development Dependencies
```bash
pip install pytest black flake8 mypy
```

### Step 3: Verify Installation

```python
# Test basic import
from ml_teammate.automl.controller import AutoMLController
print("✅ MLTeammate installed successfully!")
```

---

## 📋 Requirements by Feature

| Feature | Required Packages | Optional Packages |
|---------|------------------|-------------------|
| **Core AutoML** | `numpy`, `scikit-learn`, `optuna` | - |
| **XGBoost Support** | `xgboost` | - |
| **LightGBM Support** | `lightgbm` | - |
| **MLflow Tracking** | `mlflow` | - |
| **Visualization** | `matplotlib` | `seaborn` |
| **Model Persistence** | `joblib` | `pickle` |
| **Development** | `pytest` | `black`, `flake8`, `mypy` |

---

## 🐳 Docker Installation

### Option 1: Use Pre-built Image

```bash
# Pull the image
docker pull yourusername/mlteammate:latest

# Run with Jupyter
docker run -p 8888:8888 yourusername/mlteammate:latest
```

### Option 2: Build from Dockerfile

```bash
# Build the image
docker build -t mlteammate .

# Run the container
docker run -it mlteammate python -c "import ml_teammate; print('Ready!')"
```

---

## 🔍 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# If you get import errors, try:
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

#### 2. XGBoost Installation Issues
```bash
# On Windows, you might need:
pip install xgboost --no-cache-dir

# On macOS with M1/M2:
conda install -c conda-forge xgboost
```

#### 3. LightGBM Installation Issues
```bash
# On Windows:
pip install lightgbm --no-cache-dir

# On Linux/macOS:
pip install lightgbm
```

#### 4. MLflow Issues
```bash
# If MLflow fails to install:
pip install mlflow[extras]
```

### Platform-Specific Notes

#### Windows
- Use Anaconda/Miniconda for easier dependency management
- Install Visual Studio Build Tools for some packages

#### macOS
- Use Homebrew for system dependencies
- M1/M2 Macs: Use conda-forge for better compatibility

#### Linux
- Install system dependencies: `sudo apt-get install python3-dev`
- Use virtual environments to avoid permission issues

---

## 🧪 Testing Your Installation

Run the test suite to verify everything works:

```bash
# Run basic tests
python -m pytest tests/ -v

# Run tutorials
python ml_teammate/tutorials/01_quickstart_basic.py
```

Expected output:
```
✅ MLTeammate installed successfully!
🎉 Quickstart tutorial completed!
```

---

## 📚 Next Steps

After installation:

1. **Read the Quickstart Guide**: `docs/getting_started.md`
2. **Try the Tutorials**: `ml_teammate/tutorials/`
3. **Explore Examples**: `examples/`
4. **Check Documentation**: `docs/`

---

## 🤝 Getting Help

If you encounter issues:

1. **Check the FAQ**: Common questions and solutions
2. **Search Issues**: Look for similar problems on GitHub
3. **Create an Issue**: Provide detailed error messages and system info
4. **Join Discussions**: Ask questions in GitHub Discussions

---

## 🔄 Updating MLTeammate

```bash
# Update from GitHub
git pull origin main
pip install -e . --upgrade

# Or update via pip
pip install --upgrade git+https://github.com/yourusername/ml_teammate.git
```

---

## 📄 License

MLTeammate is released under the MIT License. See `LICENSE` file for details. 