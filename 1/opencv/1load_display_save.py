#importing the libraries
from __future__ import print_function
import argparse #argparse will handle our command line arguments
import cv2

#the only argument we need is an image path on our disk
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
               help = "Path to the image")
args = vars(ap.parse_args())

#load image off the disk and imread returns numpy array representing image
image = cv2.imread(args["image"])
print("width: {} pixels".format(image.shape[1]))
print("height: {} pixels".format(image.shape[0]))
print("channels: {}".format(image.shape[2]))

cv2.imshow("Image", image)
cv2.waitKey(0) #pauses the execution of the script until we press any key
cv2.imwrite("newimage.jpg", image)