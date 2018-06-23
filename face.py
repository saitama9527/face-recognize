import warnings
warnings.filterwarnings('ignore')
import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.models import load_model
import itertools
from PIL import Image

batch_size = 128
num_classes = 7


epochs = 20

img_rows, img_cols = 224, 224

train_path = './face'
#valid_path = './Case/test_val'
#test_path = './Case/test_test'

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224),classes=['1', '2','3','4','5','6','7'], batch_size=16)
#valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224,224), classes=['boy', 'girl','cat','bird','dog','bear','rab'], batch_size=128)
#test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224), classes=['boy', 'girl','cat','bird','dog','bear','rab'], batch_size=128)

#vgg16_model = keras.applications.vgg16.VGG16()

#model = Sequential()
#model.summary()

#for layer in vgg16_model.layers:
#    model.add(layer)
#model.summary()

#model.layers.pop()

#for layer in model.layers:
#    layer.trainable = False

#model.add(Dense(7, activation='softmax'))

#model.compile(Adam(lr=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
#model.fit_generator(train_batches, validation_data=valid_batches, epochs=10, verbose=1)

#model.save("line_test_1.h5")
