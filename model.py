import numpy as np

from tensorflow.python.keras.applications.vgg16 import VGG16

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Flatten

from tensorflow.python.keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

class VGG_model():

    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        base_model = VGG16(weights='imagenet')
        top_model = Sequential()
        top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
        return Model(
            inputs=base_model.input, outputs=top_model(base_model.output))

    def predict(self, img):
        
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        return np.squeeze(self.model.predict(x))
