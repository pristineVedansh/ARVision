# importing the packages
from sklearn.preprocessing import LabelBinarizer # one-hot encode integer labels as vector labels
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential # our network will be feed forward and layer will be added sequentially on top of each other
from keras.layers.core import Dense # fully-connected layers
from keras.optimizers import SGD # SGD will optimize the parameter of our network
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

# grab the mnist dataset
print("[INFO] loading MNIST dataset...")
dataset = datasets.fetch_openml('mnist_784')

# scale the pixel intensities between [0, 1.0]
# split the dataset
data  = dataset.data.astype("float") / 255.0
(trainX, testX, trainY, testY) = train_test_split(data, dataset.target, test_size=0.25)

# convert the labels from integers to vector
# each data point in MNIST has label [0, 9] 
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# define 784-256-128-10 architecture using keras
model = Sequential()
# input shape is set to 784(the dimensionality of each data points)
# we then learn 256 weight in this layer
model.add(Dense(256, input_shape=(784,), activation="sigmoid")) 
model.add(Dense(128, activation="sigmoid"))
# we are using softmax to obtain normalized class probabilities
model.add(Dense(10, activation="sigmoid"))

# train the model using SGD
print("[INFO] training network...")
# SGD optimizer with learning rate of 0.01
sgd = SGD(0.01)
# using cross-entropy loss function
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
# we are being lenient and using test data as validation data
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=128)

# evaluating the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=128)
# argmax gives class with the largest probability
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(x) for x in lb.classes_])) 

# plotting the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])