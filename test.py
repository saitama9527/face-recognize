# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 01:10:22 2018

@author: s7856
"""

from PIL import Image
import numpy
import cPickle
fs='/data/teamp8/ML/face/s01_01'
print fs[22:24] 
print fs[25:27]
n=2
num= str.format('%02d',n)
fs=fs[:22]+'{:0>2d}'.format(n)+fs[24:]
print fs