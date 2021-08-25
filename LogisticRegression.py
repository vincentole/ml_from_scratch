import numpy as np
from sklearn.datasets import make_blobs

class LogisticRegression():
    """2 Class Logistic Regression"""
    def __init__(self, X, learning_rate = 0.1, n_iters = 10000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.m, self.n = X.shape
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # init parameters
        self.weights = np.zeros((self.n,1))
        self.bias = 0

        for i in range(self.n_iters + 1):
            y_predict = self._sigmoid(np.dot(X, self.weights) + self.bias)

            # using cross entropy
            cost = (-1/self.m) * np.sum(y * np.log(y_predict) + (1 - y) * np.log(1 - y_predict))
            dw = (1/self.m) * np.dot(X.T, (y_predict - y))
            db = (1/self.m) * np.sum(y_predict - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if i % 1000 == 0:
                print(f"Cost after interation {i}: {cost}")

        return self.weights, self.bias

    def predict(self, X):
        y_predict = self._sigmoid(np.dot(X, self.weights) + self.bias)
        y_predict_cls = y_predict > 0.5
        return y_predict_cls

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


if __name__ == '__main__':
    np.random.seed(1)
    X, y = make_blobs(n_samples = 1000, centers = 2)
    y = y[:, np.newaxis]

    logreg = LogisticRegression(X)
    w, b = logreg.fit(X, y)
    y_predict = logreg.predict(X)

    print(f'Accuracy: {np.sum(y==y_predict)/X.shape[0]}')



