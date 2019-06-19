# import packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse

def sigmoid_activation(x):
    # compute sigmoid activation for an input
    return 1.0 / (1 + np.exp(-x))

def predict(X, W):
    # take a dot product between features and weight matrix
    preds = sigmoid_activation(X.dot(W))

    # apply a step function to threshold the outputs to binary class labels
    preds[preds <= 0.5] = 0
    preds[preds > 0] = 1

    # return the predictions
    return preds

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100, help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01, help="learn rate")
args = vars(ap.parse_args())

# generate a 2-class classification problem with 1,000 data points
# where each data point in a 2D feature vector
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))

# insert a columns of 1's as the last entry in the feature
# matrix -- this little trick allows us to teat the bias
# as a trainable parameter withiin the weight matrix
X = np.c_[X, np.ones((X.shape[0]))]

# partition the data into training and testing splits using 50% of
# the data for training and the remaining 50% of testing
(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=42)

# initialize our weight matrix and list of losses
print("[INFO] training...")
W = np.random.randn(X.shape[1], 1)
losses = []

# loop over desire number of epochs
for epoch in np.arange(0, args["epochs"]):
    # take the dot product between W and X
    preds = sigmoid_activation(trainX.dot(W))

    # now we need to determine error
    error = preds - trainY
    loss = np.sum(error ** 2)
    losses.append(loss)

    # the gradient descent update is the dot product between
    # our features and the error of predictions
    gradient = trainX.T.dot(error)

    # in the update stage, all we need to do is "nudge" the weight
    # matrix in the negative direction of the gradient (hence the
    # term "gradient descent" by taking a small step towards a set
    # towards a set of "more optimal" parameters)
    W += -args["alpha"] * gradient

    # check to see if an update should be displayed
    if epoch == 0 or (epoch + 1) % 5 == 0:
        print("[INFO] epoch={}, loss={:.7f}".format(int(epoch + 1), loss))

# evaluate our model
print("[INFO] evaluating...")
preds = predict(testX, W)
print(classification_report(testY, preds))

# plot the (testing) classification data
plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=testY, s=30)

# figure that plots the loss over time
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epochs"]), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()
