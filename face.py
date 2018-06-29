import warnings
warnings.filterwarnings('ignore')
import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import img_to_array
from keras.layers.convolutional import *
from keras.models import load_model
from keras.layers import Dropout, Dense, Activation, Convolution2D, MaxPooling2D, Flatten
import itertools
from PIL import Image
import os

batch_size = 10
nb_classes = 40
nb_epoch = 12
# input image dimensions
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3


epochs = 20
row,col=64,128
train_path = './data/train'
valid_path = './data/valid'
#test_path = './Case/test_test'

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(row,col), batch_size=16)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(row,col), batch_size=16)


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

model = Sequential()
model.add(Convolution2D(nb_filters, nb_conv, nb_conv,border_mode='valid',input_shape=(row,col,3)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Activation('softmax'))

########################
model.compile(optimizer=Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#model.compile(Adam(lr=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_batches, validation_data=valid_batches, epochs=10, verbose=1)

model.save("face_test_1.h5")