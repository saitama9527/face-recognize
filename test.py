# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 01:10:22 2018

@author: s7856
"""

from PIL import Image
import numpy
import os,glob
import shutil
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

fs="s01_01.jpg"
face_data = numpy.empty((50*15,220,110))

for row in range(50):
    for col in range(15):
        p=row+1
        n=col+1
        fs=fs[:1]+'{:0>2d}'.format(p)+fs[3]+'{:0>2d}'.format(n)+fs[6:]
        print (fs)
        img = Image.open(fs)
        img = img.resize((110, 220), Image.BILINEAR)
        img = img.convert('L')
        img_ndarray = numpy.asarray(img, dtype='float64')/ 255
        face_data[row*15+col] =img_ndarray
        
x = face_data
print (x.shape)
face_label = numpy.empty(750)
for i in range(750):
  face_label[i]= int(i/15)
y = face_label
print (y.shape)
