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


epochs = 20

img_rows, img_cols = 224, 224

train_path = './face/database/train'
valid_path = './face/database/val'
#test_path = './Case/test_test'

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(110,220), batch_size=64)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(110,220), batch_size=64)


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
model.add(Convolution2D(
                        nb_filter = 64,
                        nb_row = 5,
                        nb_col = 5,
                        border_mode = 'same',
                        input_shape=(110,220,3)
                        )
          )
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(MaxPooling2D(
                       pool_size = (2,2),
                       strides = (2,2),
                       border_mode = 'same',
                       )
          )

model.add(Convolution2D(128, 5, 5, border_mode = 'same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(MaxPooling2D(2, 2, border_mode = 'same'))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

model.add(Dense(50))
model.add(Activation('softmax'))


########################
model.compile(optimizer=Adam(lr=0.00001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#model.compile(Adam(lr=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_batches, validation_data=valid_batches, epochs=10, verbose=1)

model.save("face_test_1.h5")
