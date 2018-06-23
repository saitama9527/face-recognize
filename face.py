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
from keras.preprocessing.image import img_to_array
from keras.layers.convolutional import *
from keras.models import load_model
import itertools
from PIL import Image
import os

batch_size = 128
num_classes = 7


epochs = 20

img_rows, img_cols = 224, 224

train_path = './face/database'
#valid_path = './Case/test_val'
#test_path = './Case/test_test'

for imgname in os.listdir(train_path):

    print(imgname)
    img = Image.open(train_path+imgname)
    arr = np.asarray(img, dtype= np.float32)

    print(img.size,arr.shape)

    arr = img_to_array(img)

    print(img.size, arr,shape)

#train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224),classes=['1', '2','3','4','5','6','7'], batch_size=16)


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
