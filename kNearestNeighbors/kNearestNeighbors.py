import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


def euclidean_distance(x1,x2):
   return  np.sqrt(np.sum((x1-x2)**2))

from collections import Counter
class KNN:
    def __init__ (self, k=3):
        self.k = k

    def fit(self,X,y):
        self.X_train = X
        self.y_train = y

    # for the entire set of X_test
    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    # each sample of X_test
    def _predict(self,x):
        # compute euclidean distances between x,y in test and all x_n,y_n in train
        # trians have labels. tests do not.
        distances = [euclidean_distance(x,x_train) for x_train in self.X_train]

        # get k nearest neighbouring samples, labels
        # sort distances array, read first k values

        # used to perform an indirect sort along the given axis using the algorithm
        #specified by the keyword. It returns an array of indices of the same
        #shape as arr that would sort the array. It means indices of value arranged
        #in ascending order.
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # majority vote, most common label
        most_common = Counter(k_nearest_labels).most_common(1)

        return most_common[0][0]



if __name__ == "__main__":
    iris = datasets.load_iris()
    X,y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 1234)

    plt.figure()
    plt.scatter(X[:,0], X[:,1], c = y, cmap = cmap, edgecolor = 'k', s = 20)
    plt.savefig("kNearestNeighbors/test.png")

    # predictions
    #from knn import KNN
    knn = KNN(k=3)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    # get accuracy
    acc = np.sum(predictions==y_test)/len(y_test)
    print(acc)