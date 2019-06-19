# importing libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from nn.conv.lenet import LeNet
from imutils import paths
from matplotlib.pyplot import plt 
import numpy as np 
import argparse
import imutils
import cv2
import os

# parsing arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input datset")
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())

# initialize list of data and labels
data = []
labels = []

# loop over the input image
for imagePath in sorted(list(paths.list_images(args["datset"]))):
    # load, preprocess, and store
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, width=28)
    image = img_to_array(image)
    data.append(image)

    # extract the labels and update the label list
    label = imagePath.split(os.path.sep)[-3]
    label = "smiling" if label == "positives" else "not_smiling"
    labels.append(label)

# scale the raw pixel
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# convert the labels from integers to vectors
le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 2)

# account for skew in labeled data
classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals

# partition the data into training and testing splits using 80% of
# data for training and remaining 20% for training
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

