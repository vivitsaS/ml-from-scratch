Aim is to fit the best straight line to a given dataset showing
a relationship between 2 n-dimensional entities. 
The equation of a staright line is y = slope*x + intercept,
so the aim is to find the best slope and intercept for a given set of y and x
this can be done by iteratively minimizing the difference. This difference that
needs to be minimized is called the cost function.
between y and the model mx+c
In case of n dimensional matrices,  
matrix Y = dot(weight matrix, ; X) + bias matrix
so in this case, we need to find the best weight matrix and the bias matrix
the cost function in this case would be J = (y - (wx+b))^2
we need to find the best w and b such that J is minimum.
dj/dw shape - (n_features,n_samples)
dj_dw = [gradient_w1, gradient_w2, gradient_w3, ..., gradient_wn], n = n_features
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