#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os,sys
import csv

from sklearn import preprocessing
from sklearn import metrics
import numpy as np
import collections
from sk_data_reprecessing import DataPreprocessing

cates = ["workclass", "education", "marital_status", "occupation",
         "relationship", "race", "gender", "native_country"
]

numbers = [ "education_num", "capital_gain", "capital_loss",
           "hours_per_week"
]

bucket = ["age"]
boundaries = [ [18, 25, 30, 35, 40, 45,50, 55, 60, 65] ]
y_column = 'income_bracket'
v_to_idx = { 
             '>50K':0,
             '>50K.':0,
             '<=50K':1,
             '<=50K.':1
}

adult_data_processing = DataPreprocessing(
        cates,
        numbers,
        y_column,
        v_to_idx,
        bucket,
        boundaries
)

adult_data_processing.catefeat_v_to_id_mapping('../data/adult/adult.data')
def read_adult_data():
  y_train,X_train,y_test,X_test = \
      adult_data_processing.read_train_test(
           '../data/adult/adult.data',
           '../data/adult/adult.test'
  )    
  return y_train,X_train,y_test,X_test

def to_svm(X,Y):
  s = "fea_"
  for i in range(len(X)):
    x = X[i]
    y = Y[i]
    li = []
    li.append(str(y))
    for j in range(len(x)):
      li.append(str(j+1) + ":" + str(x[j]))
    print " ".join(li)


