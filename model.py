import numpy as np

# One can change the model here and it should work

#from keras.applications.vgg16 import VGG16 as DL
#from keras.applications.vgg16 import preprocess_input

from keras.applications.inception_resnet_v2 import InceptionResNetV2 as DL
from keras.applications.inception_resnet_v2 import preprocess_input

#from keras.applications.xception import Xception as DL
#from keras.applications.xception import preprocess_input

#from keras.applications.densenet import DenseNet169 as DL
#from keras.applications.densenet import preprocess_input

class layer_model():

    def __init__(self):
        # Skim off the top of the model before final connected layer
        self.model = DL(
            weights='imagenet',
            include_top=False,
            pooling='max',
        )

    def predict(self, x):

        x = np.expand_dims(x, axis=0).astype(float)
        x = preprocess_input(x)
        pred = np.squeeze(self.model.predict(x))

        print (pred.shape)

        return pred
