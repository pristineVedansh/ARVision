# importing libraries
import numpy as np

class Perceptron:
    # N - number of columns in our input vector. In context of bitwise, we'll set it to two.
    # alpha - learning rate
    def __init__(self, N, alpha=0.1):
        # initialize the W and alpha
        self.W = np.random.randn(N + 1) / np.sqrt(N) # divider by sqrt N to scale our weight matrix
        self.alpha = alpha

    # Now, we'll define our step function
    def step(self, x):
        # apply step function
        return 1 if x > 0 else 0

    # fitting our model to the data
    def fit(self, X, y, epochs=10):
        # insert a column of 1's as the last  entry
        # in the feature matrix -- this little trick
        # allows us to treat the bias as the trainable parameter
        X = np.c_[X, np.ones((X.shape[0]))]
        # loop over the desired number of epochs
        for epoch in np.arange(0, epochs):
            # loop over each indivisual data points
            for (x, target) in zip(X, y):
                # take the dot product between input features
                # and the weight matrix, then pass this value
                # through the step function to obtain the prediction
                p = self.step(np.dot(x, self.W))

                # only performs a weight update if our prediction
                # does not match the target
                if p != target:
                    # determine the error
                    error = p - target

                    # update the weight matrix
                    self.W += -self.alpha * error * x

    def predict(self, X, addBias=True):
        # ensure our output is a matrix
        X = np.atleast_2d(X)

        # check to see if bias column should be added
        if addBias:
            # insert a column of 1's as the last entry in the feature matrix(bias)
            X = np.c_[X, np.ones((X.shape[0]))]

        # take the dot product between inpur features and W, and pass value through step function
        return self.step(np.dot(X, self.W))
