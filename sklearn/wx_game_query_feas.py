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
numbers = "cnt_1,cnt_2,cnt_3,cnt_4,cnt_5,cnt_6,cnt_7,cnt_8,cnt_9,cnt_10,cnt_11,cnt_12,cnt_13,cnt_14,cnt_15,cnt_16,cnt_17,cnt_18,cnt_19,cnt_20,cnt_21,cnt_22,cnt_23,cnt_24,cnt_25,cnt_26,cnt_27,cnt_28,cnt_29,p".split(",")
numbers = "p"
y_column = 'y'
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

path = '/home/lx/work/mmdataapp/wxalonso/wxqueryanlysis/data/game_user_train/'
adult_data_processing.catefeat_v_to_id_mapping( path + 'alonsoli_zx_game_querys_201704_05_06_unions_with_p.csv_with_query_cates.2.new_feas.csv')
def read_adult_data():
  y_train,X_train,y_test,X_test = \
      adult_data_processing.read_train_test(
           path + 'alonsoli_zx_game_querys_201704_05_06_unions_with_p.csv_with_query_cates.2.new_feas.csv_train.txt',
	       path + 'alonsoli_zx_game_querys_201704_05_06_unions_with_p.csv_with_query_cates.2.new_feas.csv_test.txt'
      )	
  return y_train,X_train,y_test,X_test



