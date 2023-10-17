## Algorithm Fundamentals

The Naive Bayes classifier is based on Bayes' theorem and the "naive" assumption of feature independence. Key concepts include:

- Bayes' Theorem: A statistical formula used to update the probability for a hypothesis as more evidence or information becomes available.
- Conditional Probability: The likelihood of an event occurring given that another event has occurred.
- Independence Assumption: Naive Bayes assumes that features are conditionally independent given the class label.

# Gaussian Naive Bayes:

This variant assumes that the continuous features follow a Gaussian (normal) distribution. It's suitable for data where the continuous features approximate a bell-shaped curve.
The choice of which variant to use depends on the type of data you have and the distribution that best represents the data's characteristics. It's important to select the appropriate variant to avoid introducing modeling assumptions that do not match the data.

In practice, Gaussian Naive Bayes is commonly used for continuous numerical data when the features exhibit a roughly Gaussian distribution. However, if the distributional assumption does not hold, other variants like Multinomial or Bernoulli Naive Bayes can be better choices depending on the data's nature.

It's important to note that Naive Bayes may not perform well in situations where the independence assumption is grossly violated or when feature interactions play a significant role. In such cases, more advanced models like decision trees, random forests, or neural networks may be more appropriate. Nonetheless, Naive Bayes remains a valuable and widely used algorithm in machine learning, especially for certain types of classification problems where its assumptions align well with the data characteristics.


## Implementation Details

The script defines a `naiveBayesClassifier` class with the following methods:

- `fit`: Fits the Naive Bayes model to the training data, calculating the mean, variance, and class priors.
- `predict`: Makes predictions for a set of input data using the Naive Bayes algorithm. It calculates the posterior probabilities and selects the class with the highest probability.


# fit

Get n_samples, n_features,n_classes.
Get mean and variance of each feature vector for each class calculate \(P(y)\).
By precalculating these 3 entities, we are fitting the model. By fixing mean and var, we are assuming a probability distribution that we get from train data.

# predict

In this formula, 
\(P(y|X) = P(y).pi(i=0 to n)[P(xi|y)]/P(X)\)
All terms containing X or its components are from test data.
Class conditional probability, \(P(xi|y)\) is calculated using PDF.
The expression for \(P(y|X)\) needs laplace smoothening because if \(P(xi|y)-->0\)
it might lead to \(P(y|X)\) getting close to 0. 
\(y = argmax of predicted P(y)\) (posterior probability)

## How to Use

1. Import the script and create an instance of the `naiveBayesClassifier` class.
2. Prepare your data as NumPy arrays, where `X` represents the features, and `y` is the class labels.
3. Use the `fit` method to train the model with your training data.
4. Use the `predict` method to make predictions on new data.
5. Evaluate the accuracy of the predictions to assess the model's performance.

## Dependencies

This script requires the following Python libraries:

- NumPy

## Possible Variations and Modifications
The Naive Bayes algorithm can be adapted and extended in various ways to suit different use cases:

* Laplace Smoothing: Implement Laplace (add-one) smoothing to handle cases where certain feature-class combinations have zero counts.
* Different Distributions: Experiment with different probability distributions for modeling the likelihood, such as Multinomial or Bernoulli Naive Bayes.
* Text Classification: Adapt the algorithm for text classification tasks by using the Multinomial Naive Bayes model.
* Feature Engineering: Enhance the feature set by considering feature engineering techniques like TF-IDF or word embeddings for text data.
* Real-world Applications: Apply Naive Bayes to real-world problems such as spam email detection, sentiment analysis, or document classification.
