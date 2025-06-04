
class AutoML:
    def __init__(self):
        from ml_teammate.automl.controller import Controller
        self.controller = Controller()
        self.best_learner = None

    def fit(self, X, y, time_budget=60):
        config = self.controller.select_next_config(X, y)
        self.best_learner = self.controller.train(config, X, y)

    def predict(self, X):
        return self.best_learner.predict(X)

    def log(self):
        print("Logging results...")
