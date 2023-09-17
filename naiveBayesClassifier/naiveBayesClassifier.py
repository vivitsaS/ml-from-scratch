# fit-
# get n_samples, n_features,n_classes
# get mean and variance of each feature vector for each class
# calculate P(y)
# by precalculating these 3 entities, we are fitting the model
# by fixing mean and var, we are assuming a probability distribution that we get from train data,
# predict
# in this formula, P(y|X) = P(y).pi(i=0 to n)[P(xi|y)]/P(X)
# all terms containing X or its components are from test data
# class conditionals, P(xi|y) is calculated using pdf
# the expression for P(y|X) needs laplace smoothening because if P(xi|y)-->0
# it might lead to P(y|X) getting close to 0. 
# 
import numpy as np

class naiveBayesClassifier():

    def fit(self,X,y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # create 0 vectors for mean, variance and priors or P(y)
        self._mean = np.zeros((n_classes,n_features), dtype=np.float64)
        self._var = np.zeros((n_classes,n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[c==y]
            self._mean[idx,:] = np.mean(X_c, axis = 0)
            self._var[idx,:] = np.var(X_c, axis = 0)
            self._priors[idx] = X_c.shape[0]/float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        posteriors = []
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            class_conditionals = np.sum(np.log(self._pdf(idx,x)))
            posterior = prior + class_conditionals
            posteriors.append(posterior)
        return self._classes[np.argmax(posteriors)]      
    
    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2 ) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    

if __name__ == "__main__":
    
        # Imports
        from sklearn.model_selection import train_test_split
        from sklearn import datasets

        def accuracy(y_true, y_pred):
            accuracy = np.sum(y_true == y_pred) / len(y_true)
            return accuracy

        X, y = datasets.make_classification(
            n_samples=1000, n_features=10, n_classes=2, random_state=123
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=123
        )

        nb = naiveBayesClassifier()
        nb.fit(X_train, y_train)
        predictions = nb.predict(X_test)

        print("Naive Bayes classification accuracy", accuracy(y_test, predictions))

