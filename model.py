import numpy as np

from keras.models import Model, Sequential
from keras.layers import Flatten

from keras.applications.vgg16 import VGG16 as DL
from keras.applications.vgg16 import preprocess_input

#from keras.applications.inception_resnet_v2 import InceptionResNetV2 as DL
#from keras.applications.inception_resnet_v2 import preprocess_input

#from keras.applications.xception import Xception as DL
#from keras.applications.xception import preprocess_input

#from keras.applications.densenet import DenseNet169 as DL
#from keras.applications.densenet import preprocess_input


class layer_model():

    def __init__(self):
        self.model = DL(weights='imagenet')

    def predict(self, x):

        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        pred = np.squeeze(self.model.predict(x))
        return pred
