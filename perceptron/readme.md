## Algorithm Fundamentals

The Perceptron classifier is a linear binary classification algorithm. Key concepts include:

- Linear Decision Boundary: The Perceptron learns a linear decision boundary to separate data points into two classes.
- Activation Function: The Perceptron uses a step function as its activation function. If the weighted sum of input features is greater than or equal to zero, it predicts class 1; otherwise, it predicts class 0.
- Learning Rate: The learning rate controls the step size during parameter updates. It influences the convergence speed of the Perceptron.


Perceptron is similar to regression models, except it does not use gradient descent and it updates based on true difference between true and predicted vectors, scaled by feature vectors and moderated in a similar fashion using learning rate. It's a binary classifier, so the y_predicted and y_true vectors have binary values. It is a very basic algorithm for classification but these stacked together can perform useful computations. These arrangements form the basic backbone of neural networks.

## Math Behind It

The Perceptron's core idea is to learn a linear decision boundary by iteratively adjusting its weights and bias. Here's the intuition and mathematical foundation:

- **Linear Output (z)**: The Perceptron calculates a linear combination of input features as `z = (x_1 * w_1) + (x_2 * w_2) + ... + b`, where `x_i` represents features, `w_i` are weights, and `b` is the bias.

- **Activation Function**: It applies the unit step function to the linear output, which makes predictions. If `z >= 0`, it predicts class 1; otherwise, it predicts class 0.

- **Parameter Updates**: During training, the Perceptron adjusts its weights and bias based on the difference between the true class and predicted class. The update is determined by the learning rate: `update = learning_rate * (y_true - y_predicted)`. The weights and bias are updated for each sample in the dataset.

## Implementation Details

The script defines a `Perceptron` class with the following methods:

- `__init__`: Initializes the Perceptron with user-defined learning rate and the number of iterations for training.
- `fit`: Fits the Perceptron model to input data by adjusting the weights and bias through multiple iterations.
- `predict`: Makes predictions for input data using the trained model, resulting in binary class predictions.

## How to Use

1. Import the script and create an instance of the `Perceptron` class, specifying the learning rate and the number of iterations.
2. Prepare your data as NumPy arrays, where `X` represents the features, and `y` is the binary class labels.
3. Use the `fit` method to train the model with your data.
4. Use the `predict` method to make binary class predictions on new data.

## Dependencies

This script requires the following Python libraries:

- NumPy
- Matplotlib
- Scikit-learn (for dataset creation and train-test split)

## Possible Variations and Modifications

The Perceptron algorithm can be extended and modified in various ways to handle more complex problems or improve performance:

* Multiclass Classification: Extend the Perceptron to handle multiclass classification problems using techniques like one-vs-all.
* Stochastic Gradient Descent: Implement stochastic gradient descent for online learning and faster convergence.
* Regularization: Add L1 or L2 regularization to prevent overfitting and improve generalization.
* Feature Scaling: Apply feature scaling techniques like Min-Max scaling for better convergence.
* Custom Activation Functions: Experiment with different activation functions for non-linear decision boundaries.