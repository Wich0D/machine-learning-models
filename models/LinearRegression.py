import numpy as np

from BaseMLModel import BaseMLModel


class LinearRegression(BaseMLModel):
    """A simple linear regression model with a bias term."""

    def fit(self, train_features, train_target):
        # Add a column of ones to represent the intercept term.
        X = np.concatenate((np.ones((train_features.shape[0], 1)), train_features), axis=1)
        y = train_target

        # Compute the optimal weights using the normal equation.
        w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

        # Store the coefficients: the first one is the bias term, the rest are feature weights.
        self.w = w[1:]
        self.w0 = w[0]

    def predict(self, test_features):
        # Use the learned weights to make predictions on new data.
        return test_features.dot(self.w) + self.w0
