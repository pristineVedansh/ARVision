from keras.applications import ResNet50, InceptionV3, Xception, VGG16, VGG19, imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img
import numpy as np 
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-model", "--model", type=str, default="vgg16")
args = vars(ap.parse_args())

# dictionary of models
MODELS = {
    "vgg16": VGG16,
    "vgg19": VGG19,
    "inception": InceptionV3,
    "xception": Xception, #TensorFlowONLY
    "resnet": ResNet50
}

# ensure if valid model was supplied via command line argument
if args["model"] not in MODELS.keys():
    raise AssertionError("The --model command line argument should ""be a key in the 'MODELS' dictionary")

