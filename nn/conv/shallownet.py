# ShallowNet
# INPUT => CONV => RELU => FC
# import packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes):
        # width - width of the input image
        # height - height of the input image OR number of rows
        # depth - number of channels in the input image
        # classes - For CIFAR-10, classes - 10
        # initialize the model along with input shape to be 
        # 'channel last'
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channel first", update the input shape
        if K.image_data_format() == "channel_first":
            inputShape = (depth, height, width)

        # define the first and only CONV=>RELU layer
        # layer will have 32 filters(K) each of which are 3 X 3(square F x F)
        # "same" padding to ensure the output of convolution operation matches the input
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        # in order to apply our fully-connected layer, we need to flatten
        # multidimensional representation into 1D
        model.add(Flatten())
        # Dense layer is connected using same number of node as output class labels
        model.add(Dense(classes)) 
        model.add(Activation("softmax"))

        return model



