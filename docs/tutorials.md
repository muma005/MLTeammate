## 03 - MLflow Integration
### Key Requirements
- Uses `MLflowHelper` (v1 frozen interface)
- Compatible with `AutoMLController` v2.1+
### Usage Patterns
```python
# Minimal integration
mlflow_helper = MLflowHelper()
controller = AutoMLController(..., mlflow_helper=mlflow_helper)