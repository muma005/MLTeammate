"""
Phase 1 Foundation Tests

Tests for the core utilities foundation of MLTeammate.
These tests ensure all utilities work independently and provide
the solid foundation for subsequent phases.
"""

import pytest
import numpy as np
from unittest.mock import patch
import sys
from io import StringIO

# Import Phase 1 utilities
from ml_teammate.utils import (
    # Logging
    MLTeammateLogger, get_logger, set_global_log_level,
    debug, info, warning, error, critical,
    
    # Metrics  
    evaluate, classification_metrics, regression_metrics,
    get_detailed_report, calculate_score_improvement, validate_predictions,
    
    # Schema validation
    ValidationError, validate_config_space, validate_learner_config,
    validate_data_arrays, validate_trial_config, validate_learners_dict,
    validate_cv_folds, safe_convert_numeric,
    
    # Timing
    Timer, time_context, time_function, ExperimentTimer,
    format_duration, estimate_remaining_time
)


class TestLogger:
    """Test logger functionality."""
    
    def test_logger_creation(self):
        """Test logger can be created."""
        logger = MLTeammateLogger("test_logger")
        assert logger.logger.name == "test_logger"
    
    def test_logger_levels(self):
        """Test logger level setting."""
        logger = MLTeammateLogger("test_logger", "DEBUG")
        logger.set_level("INFO")
        logger.set_level("WARNING")
        logger.set_level("ERROR")
        logger.set_level("CRITICAL")
    
    def test_logger_invalid_level(self):
        """Test logger with invalid level."""
        logger = MLTeammateLogger("test_logger")
        with pytest.raises(ValueError):
            logger.set_level("INVALID")
    
    def test_global_logger(self):
        """Test global logger functions."""
        # Should not raise exceptions
        debug("Test debug")
        info("Test info")
        warning("Test warning")
        error("Test error")
        critical("Test critical")
    
    def test_logger_output(self):
        """Test logger actually outputs messages."""
        # Capture stdout
        captured_output = StringIO()
        
        with patch('sys.stdout', captured_output):
            logger = MLTeammateLogger("test_output")
            logger.info("Test message")
        
        output = captured_output.getvalue()
        assert "Test message" in output
        assert "INFO" in output


class TestMetrics:
    """Test metrics functionality."""
    
    def test_classification_metrics(self):
        """Test classification metrics."""
        y_true = [0, 1, 0, 1, 0, 1]
        y_pred = [0, 1, 1, 1, 0, 0]
        
        # Test main evaluate function
        score = evaluate(y_true, y_pred, "classification")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    def test_regression_metrics(self):
        """Test regression metrics."""
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.1, 2.1, 2.9, 4.1, 4.9]
        
        # Test main evaluate function (returns negative MSE)
        score = evaluate(y_true, y_pred, "regression")
        assert isinstance(score, float)
        assert score <= 0.0  # Negative MSE
    
    def test_detailed_classification_metrics(self):
        """Test detailed classification metrics."""
        y_true = [0, 1, 0, 1, 0, 1]
        y_pred = [0, 1, 1, 1, 0, 0]
        
        metrics = classification_metrics(y_true, y_pred)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        
        # All metrics should be between 0 and 1
        for value in metrics.values():
            if isinstance(value, float):
                assert 0.0 <= value <= 1.0
    
    def test_detailed_regression_metrics(self):
        """Test detailed regression metrics."""
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.1, 2.1, 2.9, 4.1, 4.9]
        
        metrics = regression_metrics(y_true, y_pred)
        
        assert "mse" in metrics
        assert "rmse" in metrics  
        assert "mae" in metrics
        assert "r2" in metrics
        assert "mean_residual" in metrics
        assert "std_residual" in metrics
        
        # MSE and RMSE should be positive
        assert metrics["mse"] >= 0
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0
    
    def test_metrics_validation(self):
        """Test metrics input validation."""
        y_true = [0, 1, 0]
        y_pred_wrong_length = [0, 1]
        
        with pytest.raises(ValueError):
            evaluate(y_true, y_pred_wrong_length, "classification")
        
        with pytest.raises(ValueError):
            evaluate([], [], "classification")
        
        with pytest.raises(ValueError):
            evaluate(y_true, [0, 1, 0], "invalid_task")
    
    def test_detailed_reports(self):
        """Test detailed report generation."""
        y_true = [0, 1, 0, 1, 0, 1]
        y_pred = [0, 1, 1, 1, 0, 0]
        
        # Classification report
        report = get_detailed_report(y_true, y_pred, "classification")
        assert isinstance(report, str)
        assert len(report) > 0
        
        # Regression report
        y_true_reg = [1.0, 2.0, 3.0]
        y_pred_reg = [1.1, 2.1, 2.9]
        report = get_detailed_report(y_true_reg, y_pred_reg, "regression")
        assert isinstance(report, str)
        assert "MSE" in report
    
    def test_score_improvement(self):
        """Test score improvement calculation."""
        improvement = calculate_score_improvement(0.8, 0.9)
        
        assert "absolute_improvement" in improvement
        assert "relative_improvement_percent" in improvement
        assert "is_improvement" in improvement
        
        assert improvement["absolute_improvement"] == 0.1
        assert improvement["is_improvement"] == True


class TestSchemaValidation:
    """Test schema validation functionality."""
    
    def test_config_space_validation(self):
        """Test configuration space validation."""
        # Valid config space
        valid_config = {
            "param1": {"type": "int", "bounds": [1, 10]},
            "param2": {"type": "float", "bounds": [0.1, 1.0]},
            "param3": {"type": "categorical", "choices": ["a", "b", "c"]}
        }
        
        assert validate_config_space(valid_config) == True
    
    def test_invalid_config_space(self):
        """Test invalid configuration spaces."""
        # Empty config
        with pytest.raises(ValidationError):
            validate_config_space({})
        
        # Missing type
        with pytest.raises(ValidationError):
            validate_config_space({"param1": {"bounds": [1, 10]}})
        
        # Invalid type
        with pytest.raises(ValidationError):
            validate_config_space({"param1": {"type": "invalid"}})
        
        # Invalid bounds
        with pytest.raises(ValidationError):
            validate_config_space({"param1": {"type": "int", "bounds": [10, 1]}})
    
    def test_data_validation(self):
        """Test data array validation."""
        # Valid data
        X = [[1, 2], [3, 4], [5, 6]]
        y = [0, 1, 0]
        
        assert validate_data_arrays(X, y, "classification") == True
        assert validate_data_arrays(X, [1.0, 2.0, 3.0], "regression") == True
    
    def test_invalid_data(self):
        """Test invalid data arrays."""
        X = [[1, 2], [3, 4]]
        y_wrong_length = [0]
        y_with_nan = [0, float('nan')]
        
        # Length mismatch
        with pytest.raises(ValidationError):
            validate_data_arrays(X, y_wrong_length, "classification")
        
        # NaN values
        with pytest.raises(ValidationError):
            validate_data_arrays(X, y_with_nan, "classification")
        
        # Empty data
        with pytest.raises(ValidationError):
            validate_data_arrays([], [], "classification")
    
    def test_learner_config_validation(self):
        """Test learner configuration validation."""
        valid_config = {"param1": 1, "param2": 0.5}
        assert validate_learner_config(valid_config, "test_learner") == True
        
        # Config with None values
        invalid_config = {"param1": 1, "param2": None}
        with pytest.raises(ValidationError):
            validate_learner_config(invalid_config, "test_learner")
    
    def test_trial_config_validation(self):
        """Test trial configuration validation."""
        valid_trial = {"learner_name": "random_forest", "param1": 1}
        assert validate_trial_config(valid_trial) == True
        
        # Missing learner_name
        with pytest.raises(ValidationError):
            validate_trial_config({"param1": 1})
    
    def test_cv_validation(self):
        """Test cross-validation parameter validation."""
        assert validate_cv_folds(5, 100) == True
        assert validate_cv_folds(None, 100) == True
        
        # Too many folds
        with pytest.raises(ValidationError):
            validate_cv_folds(150, 100)
        
        # Too few folds
        with pytest.raises(ValidationError):
            validate_cv_folds(1, 100)
    
    def test_safe_numeric_conversion(self):
        """Test safe numeric conversion."""
        assert safe_convert_numeric("5", int) == 5
        assert safe_convert_numeric("5.5", float) == 5.5
        assert safe_convert_numeric(5.7, int) == 5
        
        with pytest.raises(ValidationError):
            safe_convert_numeric("not_a_number", int)


class TestTimer:
    """Test timer functionality."""
    
    def test_timer_basic(self):
        """Test basic timer functionality."""
        timer = Timer("test_timer")
        
        assert not timer.is_running
        assert timer.get_elapsed() == 0.0
        
        timer.start()
        assert timer.is_running
        
        # Let some time pass
        import time
        time.sleep(0.01)
        
        elapsed = timer.stop()
        assert elapsed > 0
        assert not timer.is_running
        assert timer.elapsed_time == elapsed
    
    def test_timer_context(self):
        """Test timer context manager."""
        with time_context("test_operation") as timer:
            import time
            time.sleep(0.01)
        
        assert timer.elapsed_time > 0
        assert not timer.is_running
    
    def test_timer_function(self):
        """Test function timing."""
        def dummy_function(x):
            import time
            time.sleep(0.01)
            return x * 2
        
        result, elapsed = time_function(dummy_function, 5)
        assert result == 10
        assert elapsed > 0
    
    def test_experiment_timer(self):
        """Test experiment timer."""
        exp_timer = ExperimentTimer("test_experiment")
        
        exp_timer.start_experiment()
        
        exp_timer.start_phase("phase1")
        import time
        time.sleep(0.01)
        phase1_time = exp_timer.end_phase("phase1")
        
        exp_timer.start_phase("phase2")
        time.sleep(0.01)
        phase2_time = exp_timer.end_phase("phase2")
        
        total_time = exp_timer.end_experiment()
        
        assert phase1_time > 0
        assert phase2_time > 0
        assert total_time > 0
        assert total_time >= phase1_time + phase2_time
    
    def test_format_duration(self):
        """Test duration formatting."""
        assert "ms" in format_duration(0.001)
        assert "s" in format_duration(1.5)
        assert "m" in format_duration(65)
        assert "h" in format_duration(3665)
    
    def test_time_estimation(self):
        """Test remaining time estimation."""
        remaining = estimate_remaining_time(5, 10, 50.0)
        assert remaining == 50.0  # 5 iterations remaining at 10s each
        
        remaining = estimate_remaining_time(0, 10, 50.0)
        assert remaining is None
        
        remaining = estimate_remaining_time(10, 10, 50.0) 
        assert remaining == 0.0


def test_package_imports():
    """Test that all utilities can be imported."""
    # This test ensures the __init__.py exports work correctly
    from ml_teammate.utils import evaluate, get_logger, Timer, ValidationError
    
    # Test basic functionality
    logger = get_logger()
    assert logger is not None
    
    timer = Timer()
    assert timer is not None
    
    # Test metrics work
    score = evaluate([0, 1, 0], [0, 1, 1], "classification")
    assert isinstance(score, float)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
