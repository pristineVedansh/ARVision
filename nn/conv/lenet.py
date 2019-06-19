# LeNet with RELU activation instead of CONV
# INPUT => CONV => RELU => POOL => CONV => RELU => POOL => FC => RELU => FC
# import packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

class LeNet:
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

        # first set of layers CONV => RELU => POOL
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        # we apply 2 X 2 pooling with 2 X 2 stride, thereby decreasing the output volume by 75%
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of layers CONV => RELU => POOL
        model.add(Conv2D(50, (5, 5), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        # we apply 2 X 2 pooling with 2 X 2 stride, thereby decreasing the output volume by 75%
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # first and only set of FC => RELU layers
        # flattended input volume and a fully connected layer with 500 nodes can be applied
        model.add(Flatten())
        model.add(Dense(500))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return model
        return model




