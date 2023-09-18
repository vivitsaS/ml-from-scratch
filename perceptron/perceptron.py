"""
perceptron is similar to regression models, except it does not use
gradient descent and it updates based on true difference between
true and predicted vectors. It's a binary classifier, so the vectors have
binary values
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class Perceptron():
    def __init__(self, learning_rate = 0.01, n_iters = 1000):
        self.weights = None
        self.bias = None
        self.lr = learning_rate
        self.n_iters = n_iters
        # defining activation function here because we can use this 
        # class to create mlp and hyperparamters should be tweakable
        self.activation_function = self.unit_step_function
    def fit(self,X,y):
        n_samples, n_features = X.shape

        # initializing parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # y_true is just activation function applied to the entire y array
        y_true = np.array([1 if i>0 else 0 for i in y])
        
        # each iteration will tweak the paramters w, b
        # optimization is done on the entire X, n_iters times 
        for n_iter in range(self.n_iters):
            # each iteration goes through one sample in X
            for idx, x_i in enumerate(X):
                linear_ouptut = np.dot(x_i, self.weights)+self.bias
                y_predicted = self.activation_function(linear_ouptut)
                difference = y_true[idx]-y_predicted

                # update w,b on one sample in X
                update = self.lr*difference

                self.weights += update*x_i
                self.bias += update

    def unit_step_function(self,linear_output):
        # returns 
        return np.where(linear_output>=0, 1, 0)
    
    def predict(self,X):
        linear_ouptut = np.dot(X, self.weights)+self.bias
        y_predicted = self.activation_function(linear_ouptut)
        return y_predicted
                

if __name__ == "__main__":
    # Imports
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    X, y = datasets.make_blobs(
        n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    p = Perceptron(learning_rate=0.01, n_iters=1000)
    p.fit(X_train, y_train)
    predictions = p.predict(X_test)

    print("Perceptron classification accuracy", accuracy(y_test, predictions))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

    x0_1 = np.amin(X_train[:, 0])
    x0_2 = np.amax(X_train[:, 0])

    x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
    x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

    ymin = np.amin(X_train[:, 1])
    ymax = np.amax(X_train[:, 1])
    ax.set_ylim([ymin - 3, ymax + 3])

    plt.savefig("perceptron/plot.png")

