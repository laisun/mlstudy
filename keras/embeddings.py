#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os,sys

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Reshape,Merge

import numpy as np

input_array = np.random.randint(1000, size=(1,32*10))

model = Sequential()
model.add(Embedding(1000, 64, input_length=32*10))
# the model will take as input an integer matrix of size (batch, input_length).
# the largest integer (i.e. word index) in the input should be no larger than 999 (vocabulary size).
# now model.output_shape == (None, 10, 64), where None is the batch dimension.
model.add(Reshape((32*10,64),input_shape=(64,)))

model.compile('rmsprop', 'mse')

output_array = model.predict(input_array)

print output_array.shape
print output_array


