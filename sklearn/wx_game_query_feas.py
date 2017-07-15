#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os,sys
import csv

from sklearn import preprocessing
from sklearn import metrics
import numpy as np
import collections
from sk_data_reprecessing import DataPreprocessing

cates = []
numbers = ['count','rate','c2']
y_column = 'tag'
v_to_idx = { 
             '0':0,
	         '1':1,
	   }

adult_data_processing = DataPreprocessing(
                            cates,
			    numbers,
			    y_column,
                            v_to_idx
			)

path='/home/lx/work/mmdataapp/wxalonso/wxmlcls/data/commoncls/game_user_train/'
adult_data_processing.catefeat_v_to_id_mapping( path + 'alonsoli_zx_game_querys_201704_05_06_unions_game._allcsv.csv_tag_querys.2.2.csv')
def read_adult_data():
  y_train,X_train,y_test,X_test = \
      adult_data_processing.read_train_test(
           path + 'alonsoli_zx_game_querys_201704_05_06_unions_game._allcsv.csv_tag_querys.2.2.csv_train.txt',
	       path + 'alonsoli_zx_game_querys_201704_05_06_unions_game._allcsv.csv_tag_querys.2.2.csv_test.txt'
      )	
  return y_train,X_train,y_test,X_test

