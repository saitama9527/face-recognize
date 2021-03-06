›from PIL import Image
import numpy
import os,glob
import shutil
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

drow=168
dcol=120

fs="s01_01.jpg"
face_data = numpy.empty((50*13,drow,dcol,3))



for row in range(50):
    for col in range(13):
        if col==4:
            continue
        if col==8:
            continue
        p=row+1
        n=col+1
        fs=fs[:1]+'{:0>2d}'.format(p)+fs[3]+'{:0>2d}'.format(n)+fs[6:]
        print (fs)
        img = Image.open(fs)
        img = img.resize((dcol, drow), Image.BILINEAR)
        img_ndarray = numpy.asarray(img, dtype='float64')/ 255
        face_data[row*13+col] =img_ndarray

X = face_data
print (X.shape)
face_label = numpy.empty(650)
for i in range(650):
    face_label[i]= int(i/13)
y = face_label
print (y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 30)
print (X_train.shape)
print (X_test.shape)
print (y_train.shape)
print (y_test.shape)

print("Changing format......")

X_train = X_train.reshape(-1, 3,dcol, drow)
X_test = X_test.reshape(-1, 3,dcol, drow)
y_train = np_utils.to_categorical(y_train, num_classes=50)
y_test = np_utils.to_categorical(y_test, num_classes=50)

print (X_train.shape)
print (X_test.shape)
print (y_train.shape)
print (y_test.shape)

print("Changing succeeded!")
os.system("pause")

model = Sequential()
model.add(Convolution2D(nb_filter = 64,nb_row = 5,nb_col = 5,border_mode = 'same',input_shape=(3,dcol,drow)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2),strides = (2,2),border_mode = 'same',))
model.add(Convolution2D(128, 5, 5, border_mode = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(2, 2, border_mode = 'same'))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))


model.add(Dense(50))
model.add(Activation('softmax'))

adam = Adam(lr = 1e-4)

model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
model.fit(X_train, y_train, epochs=30, batch_size=32,)

print('\nTesting ------------')
loss, accuracy = model.evaluate(X_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)

model.save('cnn_model.h5')
print('\nSuccessfully saved as cnn_model_1.h5')

