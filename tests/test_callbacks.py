# tests/test_callbacks.py
"""
Comprehensive tests for MLTeammate callback system.
Tests all callback types and their integration with the AutoML pipeline.
"""

import pytest
import tempfile
import os
import json
import time
from unittest.mock import Mock, patch

from ml_teammate.automl.callbacks import (
    BaseCallback, 
    LoggerCallback, 
    ProgressCallback, 
    ArtifactCallback,
    create_callbacks
)


class TestBaseCallback:
    """Test the base callback class."""
    
    def test_base_callback_initialization(self):
        """Test that base callback can be instantiated."""
        callback = BaseCallback()
        assert callback is not None
    
    def test_base_callback_methods_exist(self):
        """Test that all required methods exist."""
        callback = BaseCallback()
        
        # Check that all methods exist
        assert hasattr(callback, 'on_experiment_start')
        assert hasattr(callback, 'on_trial_start')
        assert hasattr(callback, 'on_trial_end')
        assert hasattr(callback, 'on_experiment_end')
    
    def test_base_callback_methods_callable(self):
        """Test that all methods are callable without errors."""
        callback = BaseCallback()
        
        # These should not raise any exceptions
        callback.on_experiment_start({"test": "config"})
        callback.on_trial_start("trial1", {"param": 1})
        callback.on_trial_end("trial1", {"param": 1}, 0.8, True)
        callback.on_experiment_end(0.9, {"param": 2})


class TestLoggerCallback:
    """Test the LoggerCallback class."""
    
    def test_logger_callback_initialization(self):
        """Test LoggerCallback initialization."""
        callback = LoggerCallback()
        assert callback.use_mlflow == False
        assert callback.log_level == "INFO"
        assert callback.log_file is None
        assert callback.experiment_name == "mlteammate_experiment"
    
    def test_logger_callback_with_mlflow(self):
        """Test LoggerCallback with MLflow enabled."""
        with patch('ml_teammate.experiments.mlflow_helper.MLflowHelper'):
            callback = LoggerCallback(use_mlflow=True)
            assert callback.use_mlflow == True
            assert callback.mlflow is not None
    
    def test_logger_callback_with_file(self):
        """Test LoggerCallback with file logging."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            log_file = f.name
        
        try:
            callback = LoggerCallback(log_file=log_file)
            callback._log("Test message", "INFO")
            
            # Check that message was written to file
            with open(log_file, 'r') as f:
                content = f.read()
                assert "Test message" in content
        finally:
            os.unlink(log_file)
    
    def test_logger_callback_level_filtering(self):
        """Test that log level filtering works correctly."""
        callback = LoggerCallback(log_level="WARNING")
        
        # INFO message should not be logged
        with patch('builtins.print') as mock_print:
            callback._log("Info message", "INFO")
            mock_print.assert_not_called()
        
        # WARNING message should be logged
        with patch('builtins.print') as mock_print:
            callback._log("Warning message", "WARNING")
            mock_print.assert_called_once()
    
    def test_logger_callback_experiment_lifecycle(self):
        """Test LoggerCallback through full experiment lifecycle."""
        callback = LoggerCallback()
        
        # Test experiment start
        with patch.object(callback, '_log') as mock_log:
            callback.on_experiment_start({"n_trials": 10})
            mock_log.assert_called()
        
        # Test trial start
        with patch.object(callback, '_log') as mock_log:
            callback.on_trial_start("trial1", {"param": 1})
            mock_log.assert_called()
        
        # Test trial end
        with patch.object(callback, '_log') as mock_log:
            callback.on_trial_end("trial1", {"param": 1}, 0.8, True)
            mock_log.assert_called()
        
        # Test experiment end
        with patch.object(callback, '_log') as mock_log:
            callback.on_experiment_end(0.9, {"param": 2})
            mock_log.assert_called()
    
    def test_logger_callback_mlflow_integration(self):
        """Test LoggerCallback with MLflow integration."""
        with patch('ml_teammate.experiments.mlflow_helper.MLflowHelper') as mock_mlflow_class:
            mock_mlflow = Mock()
            mock_mlflow_class.return_value = mock_mlflow
            
            callback = LoggerCallback(use_mlflow=True)
            
            # Test experiment start with MLflow
            callback.on_experiment_start({"n_trials": 10})
            mock_mlflow.start_experiment.assert_called_once()
            
            # Test trial start with MLflow
            callback.on_trial_start("trial1", {"param": 1})
            mock_mlflow.start_trial.assert_called_once()
            
            # Test trial end with MLflow
            callback.on_trial_end("trial1", {"param": 1}, 0.8, True)
            mock_mlflow.log_trial_metrics.assert_called_once()
            mock_mlflow.end_trial.assert_called_once()
            
            # Test experiment end with MLflow
            callback.on_experiment_end(0.9, {"param": 2})
            mock_mlflow.log_experiment_summary.assert_called_once()
            mock_mlflow.end_experiment.assert_called_once()


class TestProgressCallback:
    """Test the ProgressCallback class."""
    
    def test_progress_callback_initialization(self):
        """Test ProgressCallback initialization."""
        callback = ProgressCallback(total_trials=10)
        assert callback.total_trials == 10
        assert callback.patience == 5
        assert callback.completed_trials == 0
        assert callback.best_score is None
    
    def test_progress_callback_experiment_start(self):
        """Test ProgressCallback experiment start."""
        callback = ProgressCallback(total_trials=10)
        
        with patch('builtins.print') as mock_print:
            callback.on_experiment_start({})
            mock_print.assert_called_with("🎯 Progress tracking enabled for 10 trials")
    
    def test_progress_callback_trial_end(self):
        """Test ProgressCallback trial end."""
        callback = ProgressCallback(total_trials=5)
        callback.start_time = time.time() - 10  # 10 seconds ago
        
        # Test first trial
        with patch('builtins.print') as mock_print:
            callback.on_trial_end("trial1", {"param": 1}, 0.8, True)
            mock_print.assert_called()
            assert callback.completed_trials == 1
            assert callback.best_score == 0.8
        
        # Test second trial (not best)
        with patch('builtins.print') as mock_print:
            callback.on_trial_end("trial2", {"param": 2}, 0.7, False)
            mock_print.assert_called()
            assert callback.completed_trials == 2
            assert callback.best_score == 0.8  # Should remain the same
    
    def test_progress_callback_eta_calculation(self):
        """Test ETA calculation."""
        callback = ProgressCallback(total_trials=10)
        callback.start_time = time.time() - 10
        callback.completed_trials = 5
        
        eta = callback._calculate_eta()
        assert eta != "Unknown"
        assert "s" in eta or "m" in eta or "h" in eta
    
    def test_progress_callback_early_stopping_suggestion(self):
        """Test early stopping suggestion."""
        callback = ProgressCallback(total_trials=10, patience=2)
        callback.start_time = time.time() - 10
        callback.completed_trials = 5
        callback.trials_since_improvement = 2
        
        with patch('builtins.print') as mock_print:
            callback.on_trial_end("trial1", {"param": 1}, 0.7, False)
            # Should suggest early stopping
            mock_print.assert_called()
            assert "early stopping" in str(mock_print.call_args_list[-1])


class TestArtifactCallback:
    """Test the ArtifactCallback class."""
    
    def test_artifact_callback_initialization(self):
        """Test ArtifactCallback initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            callback = ArtifactCallback(output_dir=temp_dir)
            assert callback.save_best_model == True
            assert callback.save_configs == True
            assert callback.output_dir == temp_dir
            assert os.path.exists(temp_dir)
    
    def test_artifact_callback_trial_end(self):
        """Test ArtifactCallback trial end."""
        with tempfile.TemporaryDirectory() as temp_dir:
            callback = ArtifactCallback(output_dir=temp_dir)
            
            # Test trial end
            callback.on_trial_end("trial1", {"param": 1}, 0.8, True)
            assert len(callback.all_configs) == 1
            
            trial_info = callback.all_configs[0]
            assert trial_info["trial_id"] == "trial1"
            assert trial_info["config"] == {"param": 1}
            assert trial_info["score"] == 0.8
            assert trial_info["is_best"] == True
    
    def test_artifact_callback_save_configs(self):
        """Test saving trial configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            callback = ArtifactCallback(output_dir=temp_dir)
            
            # Add some trial data
            callback.on_trial_end("trial1", {"param": 1}, 0.8, True)
            callback.on_trial_end("trial2", {"param": 2}, 0.9, True)
            
            # Save configs
            callback._save_configs()
            
            # Check that file was created
            config_file = os.path.join(temp_dir, "trial_configs.json")
            assert os.path.exists(config_file)
            
            # Check content
            with open(config_file, 'r') as f:
                data = json.load(f)
                assert len(data) == 2
                assert data[0]["trial_id"] == "trial1"
                assert data[1]["trial_id"] == "trial2"
    
    def test_artifact_callback_save_model(self):
        """Test model saving functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            callback = ArtifactCallback(output_dir=temp_dir)
            
            # Mock model
            mock_model = Mock()
            
            with patch('joblib.dump') as mock_dump:
                callback.save_model(mock_model, "test_model.pkl")
                mock_dump.assert_called_once()
    
    def test_artifact_callback_experiment_end(self):
        """Test ArtifactCallback experiment end."""
        with tempfile.TemporaryDirectory() as temp_dir:
            callback = ArtifactCallback(output_dir=temp_dir)
            
            with patch('builtins.print') as mock_print:
                callback.on_experiment_end(0.9, {"param": 2})
                mock_print.assert_called_with(f"📁 Artifacts saved to: {temp_dir}")


class TestCreateCallbacks:
    """Test the create_callbacks factory function."""
    
    def test_create_callbacks_default(self):
        """Test create_callbacks with default parameters."""
        callbacks = create_callbacks()
        assert len(callbacks) == 0  # No total_trials provided
    
    def test_create_callbacks_with_logging(self):
        """Test create_callbacks with logging enabled."""
        callbacks = create_callbacks(logging=True)
        assert len(callbacks) == 1
        assert isinstance(callbacks[0], LoggerCallback)
    
    def test_create_callbacks_with_progress(self):
        """Test create_callbacks with progress enabled."""
        callbacks = create_callbacks(progress=True, total_trials=10)
        assert len(callbacks) == 1
        assert isinstance(callbacks[0], ProgressCallback)
    
    def test_create_callbacks_with_artifacts(self):
        """Test create_callbacks with artifacts enabled."""
        callbacks = create_callbacks(artifacts=True)
        assert len(callbacks) == 1
        assert isinstance(callbacks[0], ArtifactCallback)
    
    def test_create_callbacks_all_enabled(self):
        """Test create_callbacks with all callbacks enabled."""
        callbacks = create_callbacks(
            logging=True, 
            progress=True, 
            artifacts=True,
            total_trials=10
        )
        assert len(callbacks) == 3
        assert isinstance(callbacks[0], LoggerCallback)
        assert isinstance(callbacks[1], ProgressCallback)
        assert isinstance(callbacks[2], ArtifactCallback)
    
    def test_create_callbacks_with_kwargs(self):
        """Test create_callbacks with custom parameters."""
        callbacks = create_callbacks(
            logging=True,
            progress=True,
            total_trials=10,
            logging_kwargs={"log_level": "DEBUG"},
            progress_kwargs={"patience": 3}
        )
        
        assert callbacks[0].log_level == "DEBUG"
        assert callbacks[1].patience == 3


class TestCallbackIntegration:
    """Test callback integration with AutoML pipeline."""
    
    def test_callback_lifecycle(self):
        """Test complete callback lifecycle."""
        callbacks = [
            LoggerCallback(),
            ProgressCallback(total_trials=3),
            ArtifactCallback()
        ]
        
        # Test experiment start
        for callback in callbacks:
            callback.on_experiment_start({"n_trials": 3})
        
        # Test trial lifecycle
        for i in range(3):
            trial_id = f"trial_{i}"
            config = {"param": i}
            score = 0.7 + i * 0.1
            is_best = (i == 2)  # Last trial is best
            
            for callback in callbacks:
                callback.on_trial_start(trial_id, config)
                callback.on_trial_end(trial_id, config, score, is_best)
        
        # Test experiment end
        for callback in callbacks:
            callback.on_experiment_end(0.9, {"param": 2})
    
    def test_mlflow_integration_lifecycle(self):
        """Test MLflow integration through complete lifecycle."""
        with patch('ml_teammate.experiments.mlflow_helper.MLflowHelper') as mock_mlflow_class:
            mock_mlflow = Mock()
            mock_mlflow_class.return_value = mock_mlflow
            
            callbacks = [LoggerCallback(use_mlflow=True)]
            
            # Test experiment start
            callbacks[0].on_experiment_start({"n_trials": 3})
            mock_mlflow.start_experiment.assert_called_once()
            
            # Test trial lifecycle
            for i in range(3):
                trial_id = f"trial_{i}"
                config = {"param": i}
                score = 0.7 + i * 0.1
                is_best = (i == 2)
                
                callbacks[0].on_trial_start(trial_id, config)
                callbacks[0].on_trial_end(trial_id, config, score, is_best)
            
            # Test experiment end
            callbacks[0].on_experiment_end(0.9, {"param": 2})
            
            # Verify MLflow calls
            assert mock_mlflow.start_trial.call_count == 3
            assert mock_mlflow.log_trial_metrics.call_count == 3
            assert mock_mlflow.end_trial.call_count == 3
            mock_mlflow.log_experiment_summary.assert_called_once()
            mock_mlflow.end_experiment.assert_called_once()


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])
