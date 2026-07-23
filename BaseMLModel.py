class BaseMLModel:
    def __init__(self):
        pass

    def fit(self, train_features, train_target):
        self.train_features = train_features
        self.train_target = train_target

    def predict(self, new_features):
        raise NotImplementedError("The 'predict' method must be implemented in the subclass.")