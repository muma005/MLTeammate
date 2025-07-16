from ml_teammate.automl.callbacks import LoggerCallback

def test_logger_callback_runs():
    callback = LoggerCallback()
    trial_id = "test-trial"
    config = {"max_depth": 3}
    score = 0.9
    is_best = True

    try:
        callback.on_trial_end(trial_id, config, score, is_best)
        print("✅ LoggerCallback test passed.")
    except Exception as e:
        print(f"❌ LoggerCallback test failed: {e}")

if __name__ == "__main__":
    test_logger_callback_runs()
