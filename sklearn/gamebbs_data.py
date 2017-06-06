#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os,sys
import csv

from sklearn import preprocessing
from sklearn import metrics
import numpy as np
import collections
from sk_data_reprecessing import DataPreprocessing

cates = ["topic"] 
numbers = ["entity_count","entity_rate"]

y_column = 'cate'
v_to_idx = { 
		'0':0,
		'1':1,
		'2':2
}

adult_data_processing = DataPreprocessing (
		cates,
		numbers,
		y_column,
		v_to_idx
)

adult_data_processing.catefeat_v_to_id_mapping('./data/alonsoli_task_2017_05_16_bbsscore.train.data.feas')

def read_adult_data():
  y_train,X_train,y_test,X_test = \
      adult_data_processing.read_train_test(
       './data/alonsoli_task_2017_05_16_bbsscore.train.data.feas_train.txt',
	   './data/alonsoli_task_2017_05_16_bbsscore.train.data.feas_test.txt'
  )	
  return y_train,X_train,y_test,X_test



