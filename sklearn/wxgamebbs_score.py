#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os,sys
import csv

from sklearn import preprocessing
from sklearn import metrics
import numpy as np
import collections
from sk_data_reprecessing import DataPreprocessing

cates = ["has_game_name"]
numbers = "game_count,title_BUILDING,title_GAME,title_HERO,title_HEROROLE,title_HERO_ALIAS,title_MATCH,title_SPW,content_BUILDING,content_GAME,content_HERO,content_HEROROLE,content_HERO_ALIAS,content_MATCH,content_SPW".split(",")
y_column = 'cate'
v_to_idx = { 
             '好':0,
	         '中':1,
             '差':2,
             '无关':3
	   }

adult_data_processing = DataPreprocessing(
                cates,
			    numbers,
			    y_column,
                v_to_idx
			)

path = '/home/lx/work/mmdatatextclsbroker/texttag/datas/gamebbs/score/' + 'rijialiu_task_2017_05_18.csv.feas.csv.csv'
adult_data_processing.catefeat_v_to_id_mapping( path )
def read_adult_data():
  y_train,X_train,y_test,X_test = \
      adult_data_processing.read_train_test(
           path + '_train.txt',
	       path + '_test.txt'
      )	
  return y_train,X_train,y_test,X_test



