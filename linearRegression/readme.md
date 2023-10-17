## Algorithm Fundamentals

Linear regression aims to find a linear relationship between input features and a target variable. Key concepts include:
- Linear relationship: Predictions are made by summing the weighted features.
- Gradient descent: An optimization technique used to find the best weights for the model.
- Mean squared error (MSE): A common loss function used to measure the error of the model's predictions.

## Intuition and Math Behind It

Linear Regression minimizes the cost function using gradient descent. It aims to reduce the error between predicted values and actual target values by iteratively updating the weights `w` and the bias `b`. The following formulas help in updating parameters during training:

- `y_predicted = wx + b`: Linear equation for prediction.
- `mean_squared_error = (1/n) * Σ(y_true - y_predicted)^2`: Cost function to minimize.
- `dJ_dw = (1/n) * X^T * (y_predicted - y_true)`: Gradient of the cost function with respect to weights `w`.
- `dJ_db = (1/n) * Σ(y_predicted - y_true)`: Gradient of the cost function with respect to bias `b`.
- Parameter updates: `w -= learning_rate * dJ_dw`, `b -= learning_rate * dJ_db`.

Aim is to fit the best straight line to a given dataset showing a relationship between 2 n-dimensional entities. 
The equation of a straight line is-
`y = slope*x + intercept`

So the aim is to find the best slope and intercept for a given set of y and x this can be done by iteratively minimizing the difference. This difference that needs to be minimized is called the cost function. between y and the model mx+c.

In case of n dimensional matrices,  
`matrix Y = dot(weight matrix W, ; X)\) + bias matrix`

So in this case, we need to find the best weight matrix and the bias matrix the cost function in this case would be-
`J = (y - (wx+b))^2`
J, here is MSE- the mean squared error.
We need to find the best `w` and `b` such that `J` is minimum.
`dj/dw shape - (n_features,n_samples)`
`dj_dw = [gradient_w1, gradient_w2, gradient_w3, ..., gradient_wn]`
n = n_features
Each gradient_wX represents the gradient of the cost function with respect to the weight wX,
where X is the feature number. These gradients guide the updates of the respective weights
during the optimization process to minimize the cost function and fit the data.

side note: normalization can be used if the dataset has input features having a high range
mse is used as J. But any other error can be used as the cost function. 
mse is also used to evaluate the model. Mse assess the accuracy of individual predictions. 
R2 score how well the regression model explains the variability in the target variable.
This code can be modified and optimized by using appropriate normalization techniques and 
different cost functions and evaluation scores (for example, z-score, etc)
depending on the nature of the dataset . 

## Implementation Details

The script defines a `LinearRegression` class with the following methods:

- `__init__`: Initializes the linear regression model with user-defined learning rate and the number of iterations for training.
- `fit`: Fits the linear regression model to input data using gradient descent. It initializes the model parameters (weights and bias), updates them through iterations, and minimizes the mean squared error.
- `predict`: Makes predictions for input data using the trained model, resulting in continuous target values.

## How to Use

1. Import the script and create an instance of the `LinearRegression` class, specifying the learning rate and the number of iterations.
2. Prepare your data as NumPy arrays, where `X` represents the features, and `y` is the continuous target variable.
3. Use the `fit` method to train the model with your data.
4. Use the `predict` method to make predictions on new data.

## Dependencies

This script requires the following Python libraries:

- NumPy
- Matplotlib
- Scikit-learn (for dataset creation and train-test split)

## Possible Modifications and Variations

Linear regression can be extended and modified for various use cases. Here are a few examples:

* Multiple Features: In this example, we used linear regression with a single feature. You can extend it to handle multiple features by updating the model accordingly.
* Polynomial Regression: You can extend linear regression to polynomial regression by introducing higher-order terms, allowing it to model more complex relationships.
* Regularization: Add L1 or L2 regularization to prevent overfitting when dealing with high-dimensional data.
* Feature Engineering: Consider feature engineering techniques to create more meaningful features for better predictive performance.
* Real-world Applications: Apply linear regression to various real-world problems, such as predicting housing prices, sales, or any continuous target variable.

## Advantages and Limitations

# Advantages:
* Simplicity and interpretability.
* Quick training and prediction.
* Suitable for problems with linear relationships.

# Limitations:
* Limited to linear relationships.
* Sensitive to outliers.
* May overfit with many features.

## Ideal Use Cases

Linear Regression is suitable for scenarios where the relationship between input features and target variables is approximately linear. It is commonly used in finance, economics, social sciences, and many other fields for predicting numerical values based on historical data.
