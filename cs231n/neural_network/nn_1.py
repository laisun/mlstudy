#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os,sys
import numpy as np

def sigmoid(W,b,x):
  sum_ = np.dot(x,W) + b
  return 1.0 / (1.0 + np.exp(-sum_))

def tanh(W,b,x):
  return 2*sigmoid(W,b,2*x) - 1

def relu(W,b,x):
  sum_ = np.dot(x,W) + b
  return np.maximum(0,sum_)

def weight_variable(shape):
  initial = np.random.standard_normal(size=shape)
  return initial

def bias_variable(shape):
  initial = np.random.uniform(-0.1,0.1,shape)
  return initial

class Neuron(object):
  def __init__(self,size):
    self.weights = weight_variable(size)
    self.bias = bias_variable(1)
  def forward(self,inputs):
    '''
    inputs: [n_samples,size]
    sum the inputs and make a sigmoid activation .
    '''
    return tanh(self.weights,self.bias,inputs)

shape = (3,10)
inputs = np.random.uniform(-0.1,0.1,shape)

neron = Neuron(shape[-1])
output = neron.forward(inputs)

print(output)


