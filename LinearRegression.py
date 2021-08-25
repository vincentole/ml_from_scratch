import numpy as np


class LinearRegression:
    def __init__(self, X, learning_rate=0.01, n_interations=10000, gradient_descent=False):
        self.lr = learning_rate
        self.theta = None
        self.n_iter = n_interations
        self.m, self.n = X.shape
        self.gd = gradient_descent

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.theta = np.zeros((self.n + 1, 1))

        if self.gd:
            for i in range(self.n_iter + 1):
                cost = (1 / (2 * self.m)) * np.dot((np.dot(X, self.theta) - y).T, (np.dot(X, self.theta) - y))
                self.theta -= (self.lr / self.m) * np.dot(X.T, (np.dot(X, self.theta) - y))

                if i % 1000 == 0:
                    print(f'The current cost at iteration {i}: {cost}')
            return self.theta
        else:
            self.theta = np.linalg.inv(np.dot(X.T, X)).dot(np.dot(X.T, y))
            return self.theta

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return np.dot(X, self.theta)


if __name__ == '__main__':
    X = np.random.rand(500, 1)
    y = 3 * X + np.random.randn(500, 1) * 0.1
    model = LinearRegression(X, gradient_descent=True)
    theta = model.fit(X, y)
    predict = model.predict(X)

    print(f'theta is: {theta} and the error is: {sum(predict - y)}')
