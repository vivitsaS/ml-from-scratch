Gaussian Naive Bayes:
This variant assumes that the continuous features follow a Gaussian (normal) distribution. It's suitable for data where the continuous features approximate a bell-shaped curve.
The choice of which variant to use depends on the type of data you have and the distribution that best represents the data's characteristics. It's important to select the appropriate variant to avoid introducing modeling assumptions that do not match the data.
In practice, Gaussian Naive Bayes is commonly used for continuous numerical data when the features exhibit a roughly Gaussian distribution. However, if the distributional assumption does not hold, other variants like Multinomial or Bernoulli Naive Bayes can be better choices depending on the data's nature.
It's important to note that Naive Bayes may not perform well in situations where the independence assumption is grossly violated or when feature interactions play a significant role. In such cases, more advanced models like decision trees, random forests, or neural networks may be more appropriate. Nonetheless, Naive Bayes remains a valuable and widely used algorithm in machine learning, especially for certain types of classification problems where its assumptions align well with the data characteristics.


<!-- fit- -->
get n_samples, n_features,n_classes
get mean and variance of each feature vector for each class
calculate P(y)
by precalculating these 3 entities, we are fitting the model
by fixing mean and var, we are assuming a probability distribution that we get from train data,
<!-- predict -->
in this formula, P(y|X) = P(y).pi(i=0 to n)[P(xi|y)]/P(X)
all terms containing X or its components are from test data
class conditionals, P(xi|y) is calculated using pdf
the expression for P(y|X) needs laplace smoothening because if P(xi|y)-->0
it might lead to P(y|X) getting close to 0. 
y = argmax of predicted P(y) (posterior)

