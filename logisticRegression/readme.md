## Algorithm Fundamentals

Logistic regression is a statistical model that uses the logistic function (sigmoid) to model the probability of a binary target variable. It's based on the following key concepts:
- Linear combination of features and weights.
- Sigmoid activation function for mapping to probabilities.
- Gradient descent for optimizing the model.

Logistic regression is the classification analog of linear regression, i.e., the aim is to solve a true or false (or any pair of outcomes) type of problem instead of finding the best fitting line. We try to fit an S curve (the sigmoid curve) instead of a straight line to the data. 

The loss/ cost function used is the binary cross entropy loss. Binary Cross-Entropy is commonly used for binary classification tasks, where you're predicting one of two classes (0 or 1). It measures the dissimilarity between the true binary labels and the predicted probabilities. 

The expressions for the derivative of this cost function happens to be the same as that in the case of linear regression. Only the expressions for y_predicted are different. The rest is exactly the same as linear regresion.

reference to understand the cost function and its deravative: https://youtu.be/2BkqApHKwn0?si=Y4EytirbyyTD_3aD


## Implementation Details

The script defines a `LogisticRegression` class with the following methods:

- `__init__`: Initializes the logistic regression model with user-defined learning rate and the number of iterations for training.
- `sigmoid`: Computes the sigmoid function for mapping linear predictions to probabilities.
- `fit`: Fits the logistic regression model to input data using gradient descent. It initializes the model parameters, updates weights and bias through iterations, and minimizes the binary cross-entropy loss.
- `predict`: Makes binary predictions (0 or 1) based on the trained model, by comparing the probability of the input data being in class 1 to a threshold (0.5 by default).

## How to Use

1. Import the script and create an instance of the `LogisticRegression` class, specifying the learning rate and the number of iterations.
2. Prepare your data as NumPy arrays, where `X` represents the features, and `y` is the binary target variable (0 or 1).
3. Use the `fit` method to train the model with your data.
4. Use the `predict` method to make predictions on new data.


## Dependencies

This script requires the following Python libraries:

- NumPy
- Scikit-learn (for dataset and train-test split)

## Possible Variations and Modifications with Use Cases

Logistic regression can be extended and modified for various use cases. Here are a few examples:

* Multiclass classification: Extend the model for more than two classes using one-vs-all or softmax.
* Regularization: Add L1 or L2 regularization to prevent overfitting.
* Feature engineering: Create and select meaningful features for better performance.
* Real-world applications: Apply logistic regression to problems like spam classification, customer churn prediction, and more.


## Possible Variations and Modifications with Use Cases

Logistic regression can be extended and modified for various use cases. Here are a few examples:

* Multiclass classification: Extend the model for more than two classes using one-vs-all or softmax.
* Regularization: Add L1 or L2 regularization to prevent overfitting.
* Feature engineering: Create and select meaningful features for better performance.
* Real-world applications: Apply logistic regression to problems like spam classification, customer churn prediction, and more.
