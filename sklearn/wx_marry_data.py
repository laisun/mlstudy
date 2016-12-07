#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os,sys
import csv

from sklearn import preprocessing
from sklearn import metrics
import numpy as np
import collections
from sk_data_reprecessing import DataPreprocessing

cates = ["fsex"]  #fage
numbers = ["fage","baby_cnt_201610", "readmsg_ratio_d84", 
           "readmsg_ratio_d96","readmsg_ratio_d101","readmsg_ratio_d103",
	   "sharesns_ratio_d95","sharesns_ratio_d97","sharesns_ratio_d100",
	   "sharesns_ratio_d120","cnt"
	   ]

BUCKET_COLUMNS = ['fage']
bucket_boundaries = [[18, 25, 30, 35, 40, 45]]
y_column = 'status'
v_to_idx = { 
             '1':0,
	     '2':1
	   }

data_processing = DataPreprocessing(
                            cates,
			    numbers,
			    y_column,
                            v_to_idx
			)

def read_marry_data():
  y_train,X_train,y_test,X_test = \
      data_processing.read_train_test(
           './data/marry/wx_marry.train',
	   './data/marry/wx_marry.test'
      )	
  return y_train,X_train,y_test,X_test

def read_dev_marry_data():
  X_dev = data_processing.read_dev_data(
         './data/marry/good_bad_with_beizhu.csv'
         )  
  return X_dev

def read_src_marry_data():
  rows = []
  for row in csv.DictReader(
       open('./data/marry/good_bad_with_beizhu.csv')):
    rows.append(row)
  return rows  
