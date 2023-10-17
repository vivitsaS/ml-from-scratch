## Algorithm Fundamentals

k-NN is a non-parametric and instance-based machine learning algorithm. Key concepts include:

- Euclidean Distance: Measuring the similarity between data points.
- Majority Vote: Classifying new data points based on the majority class among their k-nearest neighbors.

KNN is a simple yet powerful concept that relies on the proximity of data points in a feature space to make predictions. It's versatile and easy to understand, making it a valuable tool in various machine learning tasks, particularly for small to medium-sized datasets with clear patterns. However, it has limitations, such as computational efficiency in high-dimensional spaces and sensitivity to noisy data.
The prediction for a given `(x,y)` is the most commonly occuring result- falling in `k` unit radius around the point `(x,y)` the chosen average euclidaen distance = `k`

## Intuition and Math behind it

k-Nearest Neighbors (KNN) is a simple yet powerful classification algorithm that makes predictions based on the majority class among its k-nearest data points. The core intuition behind KNN can be summarized as follows:

* **Similarity-Based Classification:** KNN assumes that similar data points tend to belong to the same class. It measures similarity using a distance metric, commonly the Euclidean distance.
* **Local Decision Boundary:** Instead of a global decision boundary, KNN uses a local decision boundary around each data point. This flexibility enables KNN to handle complex and nonlinear decision boundaries.
* **Voting Mechanism:** KNN employs a voting mechanism, where each of the k-nearest neighbors 'votes' for their respective class. The class with the most votes is assigned to the data point.
* **Parameter Selection:** The choice of the parameter k (number of neighbors) is critical. A smaller k may lead to overfitting, while a larger k can result in underfitting.

The mathematical foundation of KNN includes the following key elements:

* **Distance Metric:** KNN uses a distance metric (typically Euclidean distance) to measure the similarity between data points. Given two data points A and B, the Euclidean distance is computed as:
  \[d(A, B) = \sqrt{\sum_{i=1}^{n} (A_i - B_i)^2}\]
  Where \(A_i\) and \(B_i\) are the feature values for the ith dimension.

* **KNN Algorithm:**
    1. Compute the distance between the new data point and all training data points.
    2. Select the k-nearest data points based on the computed distances.
    3. Count the frequency of each class among these k neighbors.
    4. Assign the class with the highest frequency as the predicted class for the new data point.

* **Weighted Voting:** In some cases, you can assign weights to neighbors based on their proximity. Closer neighbors have a greater influence on the prediction.

* **Selecting the Optimal k:** The choice of the value for k is crucial. Too small values of k can lead to noisy predictions, while too large values can lead to overly smooth decision boundaries. Cross-validation techniques are often used to determine the optimal k.

* **Normalization:** Feature scaling is essential in KNN to ensure that no single feature dominates the distance calculation.

* **Handling Categorical Features:** KNN can be extended to handle categorical features by using appropriate distance metrics, such as Hamming distance for binary data.

In summary, KNN's mathematical foundation is rooted in the concept of distance and similarity, and it classifies data points based on the majority class of their k-nearest neighbors. Parameter selection and data preprocessing play crucial roles in the performance of the KNN algorithm.

## Implementation Details

The script defines a `KNN` class with the following methods:

- `__init__`: Initializes the k-NN model with a user-defined value of `k`, which represents the number of nearest neighbors to consider during classification.
- `fit`: Fits the k-NN model to the training data, storing the training dataset and its labels.
- `predict`: Makes predictions for a set of input data using the k-NN algorithm. For each data point, it computes the Euclidean distances to all training data points, identifies the k-nearest neighbors, and performs a majority vote to determine the predicted class.

## How to Use

1. Import the script and create an instance of the `KNN` class, specifying the number of neighbors `k`.
2. Prepare your data as NumPy arrays, where `X` represents the features, and `y` is the corresponding class labels.
3. Use the `fit` method to train the model with your training data.
4. Use the `predict` method to make predictions on new data.
5. Evaluate the accuracy of the predictions to assess the model's performance.

## Dependencies

This script requires the following Python libraries:

- NumPy
- Matplotlib
- Scikit-learn

## Possible Modifications and Variations

k-NN is a versatile algorithm with various modifications and use cases:

* Distance Metrics: You can use different distance metrics other than Euclidean, such as Manhattan, Minkowski, or customized distance functions.
* Feature Scaling: Applying feature scaling or normalization can improve k-NN's performance when features have different scales.
* Optimizing k: Experiment with different values of k to find the optimal number of neighbors for your specific dataset.
* Weighted k-NN: Weight the influence of neighbors based on their distance, giving closer neighbors more importance.
* Distance-Weighted Voting: Assign different weights to neighbors based on their distance when performing the majority vote.

## Advantages and Limitations

# Advantages

* Simplicity: KNN is easy to understand and implement, making it an excellent choice for beginners in machine learning and data science.
* Non-parametric: KNN is a non-parametric algorithm, which means it doesn't make strong assumptions about the underlying data distribution. It can handle data with complex patterns and shapes.
* Adaptability: KNN can be used for classification and regression tasks. In classification, it assigns the majority class among the k-nearest neighbors, while in regression, it calculates the average of the target values of those neighbors.
* Robustness: It works well with noisy data because it relies on the local neighborhood's information rather than the entire dataset.
* No Training Period: KNN doesn't require a training phase. It stores the training data and performs predictions on the fly, making it suitable for dynamic environments where data changes frequently.
* Interpretability: KNN provides transparent and intuitive results. It's easy to explain why a particular prediction was made based on the nearest neighbors.

# Limitations

* Computational Cost: KNN can be computationally expensive, especially with large datasets. The algorithm needs to calculate distances between the query point and all training data points.
* Memory Usage: It stores the entire training dataset, which can be memory-intensive for large datasets.
* Sensitive to Feature Scaling: KNN is sensitive to the scale of features. Features with large values may dominate the distance calculation, leading to biased results. Standardization or normalization is often required.
* Curse of Dimensionality: In high-dimensional spaces, the distance between data points tends to become similar, making it harder to identify relevant neighbors and affecting the algorithm's performance.
* Choosing the Right k: Selecting an appropriate value for k can be challenging. A small k may lead to noisy predictions, while a large k may oversmooth the decision boundary.

## Ideal Use Cases

* Image Classification: KNN can be applied to image classification tasks, such as recognizing handwritten digits or identifying objects in images.
* Recommender Systems: KNN is used in collaborative filtering-based recommender systems to suggest products or content based on users' preferences and their similarity to others.
* Anomaly Detection: KNN can be employed to identify anomalies or outliers in datasets, such as fraud detection or network intrusion detection.
* Bioinformatics: It's useful in bioinformatics for tasks like gene expression analysis, identifying disease-related genes, or protein structure classification.
* Text Classification: KNN can be used in text analysis for document classification, sentiment analysis, and spam detection.
* Environmental Data Analysis: KNN can be applied in environmental science for tasks like predicting air quality based on sensor data or classifying plant species using environmental features.
