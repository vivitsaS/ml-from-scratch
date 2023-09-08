import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, learning_rate=0.001, n_iters=100):
        self.weights = None
        self.bias = None
        self.lr = learning_rate
        self.n_iters = n_iters

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent for weight update
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias

            # Calculate gradients
            dJ_dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            dJ_db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.lr * dJ_dw
            self.bias -= self.lr * dJ_db

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted

            

if __name__ == "__main__":
    # Imports
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def r2_score(y_true, y_pred):
        # Calculate the residual sum of squares (RSS)
        rss = np.sum((y_true - y_pred) ** 2)

        # Calculate the total sum of squares (TSS)
        tss = np.sum((y_true - np.mean(y_true)) ** 2)

        # Calculate R-squared (R2) score
        r2 = 1 - (rss / tss)
        return r2


    X, y = datasets.make_regression(
        n_samples=100, n_features=1, noise=20, random_state=4
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )
    # # Perform Min-Max normalization on input features (X) before training
    # min_max_scaler = MinMaxScaler()
    # X_train_normalized = min_max_scaler.fit_transform(X_train)
    # X_test_normalized = min_max_scaler.transform(X_test)

    # regressor = LinearRegression(learning_rate=0.06, n_iters=1200)
    # regressor.fit(X_train_normalized, y_train)
    # predictions = regressor.predict(X_test_normalized)

    regressor = LinearRegression(learning_rate=0.06, n_iters=100)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    print("MSE:", mse)

    accu = r2_score(y_test, predictions)
    print("Accuracy:", accu)

    y_pred_line = regressor.predict(X)
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(8, 6))
    m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
    plt.savefig('linearRegression/plot.png')
    
