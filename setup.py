from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "MLTeammate - An AutoML framework for machine learning automation"

setup(
    name="ml-teammate",  # This will be the pip install name
    version="0.1.0",
    author="muma005",
    author_email="your.email@example.com",  # Replace with your email
    description="An AutoML framework for machine learning automation",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/muma005/MLTeammate",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",  # Update if different
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "mlflow>=1.20.0",
        "optuna>=3.0.0",
        "flaml>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
        ],
        "full": [
            "xgboost>=1.5.0",
            "lightgbm>=3.2.0",
            "h2o>=3.36.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mlteammate=ml_teammate.interface.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
