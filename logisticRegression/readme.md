## Algorithm Fundamentals

Logistic regression is a statistical model that uses the logistic function (sigmoid) to model the probability of a binary target variable. It's based on the following key concepts:
- Linear combination of features and weights.
- Sigmoid activation function for mapping to probabilities.
- Gradient descent for optimizing the model.

Logistic regression is the classification analog of linear regression, i.e., the aim is to solve a true or false (or any pair of outcomes) type of problem instead of finding the best fitting line. We try to fit an S curve (the sigmoid curve) instead of a straight line to the data. 

The loss/ cost function used is the binary cross entropy loss. Binary Cross-Entropy is commonly used for binary classification tasks, where you're predicting one of two classes (0 or 1). It measures the dissimilarity between the true binary labels and the predicted probabilities. 

The expressions for the derivative of this cost function happens to be the same as that in the case of linear regression. Only the expressions for y_predicted are different. The rest is exactly the same as linear regresion.

reference to understand the cost function and its deravative: https://youtu.be/2BkqApHKwn0?si=Y4EytirbyyTD_3aD

## Intuition and Math behind it

## Math and Intuition Behind Logistic Regression

The mathematical foundation of Logistic Regression includes the following key elements:

* **Log-Odds:** Logistic Regression models the log-odds of the probability of a data point belonging to a particular class. The log-odds (logit) is defined as:
  \[logit(p) = \ln\left(\frac{p}{1-p}\right)\]
  where \(p\) is the probability of a positive outcome.

* **Sigmoid Function:** The sigmoid function transforms the log-odds into a probability between 0 and 1:
  \[P(Y=1|X) = \frac{1}{1 + e^{-\left(\beta_0 + \beta_1X\right)}}\]
  where \(P(Y=1|X)\) is the probability that the dependent variable \(Y\) is 1 given the predictor \(X\), and \(\beta_0\) and \(\beta_1\) are model parameters.

* **Log-Likelihood Function:** Logistic Regression optimizes the log-likelihood function to find the best-fitting model parameters. The log-likelihood is calculated based on the predicted probabilities for the observed data.

* **Cost Function:** The logistic regression cost function, often referred to as the cross-entropy loss or log loss, measures the difference between the predicted probabilities and the actual outcomes.

* **Gradient Descent:** Logistic Regression parameters are estimated using optimization techniques like gradient descent. The gradient of the cost function is used to update the model's weights iteratively.

* **Regularization Terms:** Regularization terms, such as L1 or L2 regularization, can be added to the cost function to prevent overfitting by penalizing large parameter values.

In summary, Logistic Regression models the probability of a binary outcome using the sigmoid function and linear decision boundaries. Its mathematical foundation involves log-odds, likelihood optimization, and cost functions. Regularization techniques can be incorporated to improve the model's performance.

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


## Possible Variations and Modifications

Logistic Regression has variations such as:
* Regularized Logistic Regression (L1 or L2 regularization).
* Multinomial Logistic Regression for multiclass classification.
* Custom loss functions or optimization methods.

## Advantages and Limitations

# Advantages:

* Simple, interpretable, and quick to train.
* Suitable for binary classification problems.
* Works well when the classes are linearly separable.

# Limitations:

* Assumes a linear decision boundary.
* Sensitive to outliers.
* May overfit with many features.

## Ideal Use Cases

Logistic Regression is ideal for binary classification problems, such as spam detection, medical diagnosis, and customer churn prediction. 

Its interpretability allows to understand and trust its predictions, making it a valuable tool for initial model development and as a benchmark. 
