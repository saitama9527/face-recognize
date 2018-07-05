from PIL import Image
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
from keras.models import load_model

drow=168
dcol=120

model = load_model('cnn_model.h5')

fs="s01_05.jpg"
face_data = numpy.empty((50*2,drow,dcol,3))


for row in range(50):
    for col in range(2):
        p=row+1
        if col==0:
            n=5
        else:
            n=9
        
        fs=fs[:1]+'{:0>2d}'.format(p)+fs[3]+'{:0>2d}'.format(n)+fs[6:]
        print (fs)
        img = Image.open(fs)
        img = img.resize((dcol, drow), Image.BILINEAR)
        img_ndarray = numpy.asarray(img, dtype='float64')/ 255
        face_data[row*2+col] =img_ndarray

X = face_data
print (X.shape)
face_label = numpy.empty(100)
for i in range(100):
    face_label[i]= int(i/2)
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

print('\nTesting ------------')
loss, accuracy = model.evaluate(X_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)

