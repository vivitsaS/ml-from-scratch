import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, learning_rate=0.001, n_iters=100):
        self.weights = None
        self.bias = None
        self.lr = learning_rate
        self.n_iters = n_iters
        
    def sigmoid(self,y):
        # s(y) = 1/(1+(e^-y)
        return 1/(1+np.exp(-y))

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent for weight update
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(y_predicted)
            # Calculate gradients
            dJ_dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            dJ_db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.lr * dJ_dw
            self.bias -= self.lr * dJ_db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

            

if __name__ == "__main__":
   
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    regressor = LogisticRegression(learning_rate=0.0001, n_iters=1000)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)
    #print(predictions)
    #print(X_test)
    print("LR classification accuracy:", accuracy(y_test, predictions))
