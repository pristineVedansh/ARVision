from keras.preprocessing.image import img_to_array

class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        # stores the image data format
        self.dataFormat = dataFormat

    def preprocess(self, image):
        # apply the keras utility function that correctly rearranges
        #  the dimesnions  of images
        return img_to_array(image, data_format=self.dataFormat)
