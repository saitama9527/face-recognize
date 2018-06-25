# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 01:10:22 2018

@author: s7856
"""

from PIL import Image
import numpy
import cPickle
import cv2
import os,glob
import shutil
fs='/data/teamp8/ML/face/s01_01.jpg'
face_data = numpy.empty((750,220,110))
for row in range(1,51):

    for col in range(1,16):
        if col==14: 
          
          fs=fs[:22]+'{:0>2d}'.format(row)+fs[24]+'{:0>2d}'.format(col)+fs[27:]
          newname='s'+'{:0>2d}'.format(row)+'_09.jpg' 
          print newname
          shutil.copy2(fs,newname)
        