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
numbers = "cnt_10101,day_cnt_10101,cnt_10102,day_cnt_10102,cnt_10103,day_cnt_10103,cnt_10104,day_cnt_10104,cnt_10105,day_cnt_10105,cnt_10106,day_cnt_10106,cnt_10107,day_cnt_10107,cnt_10108,day_cnt_10108,cnt_10201,day_cnt_10201,cnt_10202,day_cnt_10202,cnt_10203,day_cnt_10203,cnt_10204,day_cnt_10204,cnt_10205,day_cnt_10205,cnt_10206,day_cnt_10206,cnt_10207,day_cnt_10207,cnt_10208,day_cnt_10208,cnt_10209,day_cnt_10209,cnt_10210,day_cnt_10210,cnt_10211,day_cnt_10211,cnt_10301,day_cnt_10301,cnt_10302,day_cnt_10302,cnt_10303,day_cnt_10303,cnt_10401,day_cnt_10401,cnt_10402,day_cnt_10402,cnt_10403,day_cnt_10403,cnt_10404,day_cnt_10404,cnt_10405,day_cnt_10405,cnt_10406,day_cnt_10406,cnt_10407,day_cnt_10407,cnt_10408,day_cnt_10408,cnt_20101,day_cnt_20101,cnt_20102,day_cnt_20102,cnt_20103,day_cnt_20103,cnt_20104,day_cnt_20104,cnt_20105,day_cnt_20105,cnt_20201,day_cnt_20201,cnt_20202,day_cnt_20202,cnt_20203,day_cnt_20203,cnt_20204,day_cnt_20204,cnt_20301,day_cnt_20301,cnt_20302,day_cnt_20302,cnt_20303,day_cnt_20303,cnt_20304,day_cnt_20304,cnt_20305,day_cnt_20305,cnt_20306,day_cnt_20306,cnt_20307,day_cnt_20307,cnt_20308,day_cnt_20308,cnt_20309,day_cnt_20309,cnt_20310,day_cnt_20310,cnt_30101,day_cnt_30101,cnt_30102,day_cnt_30102,cnt_30103,day_cnt_30103,cnt_30104,day_cnt_30104,cnt_30105,day_cnt_30105,cnt_30106,day_cnt_30106,cnt_30107,day_cnt_30107,cnt_30201,day_cnt_30201,cnt_30202,day_cnt_30202,cnt_30203,day_cnt_30203,cnt_30204,day_cnt_30204,cnt_30205,day_cnt_30205,cnt_30206,day_cnt_30206,cnt_30207,day_cnt_30207,cnt_30301,day_cnt_30301,cnt_30302,day_cnt_30302,cnt_30401,day_cnt_30401,cnt_30402,day_cnt_30402,cnt_30403,day_cnt_30403,cnt_30501,day_cnt_30501,cnt_30601,day_cnt_30601,cnt_30602,day_cnt_30602,cnt_30603,day_cnt_30603,cnt_40101,day_cnt_40101,cnt_40102,day_cnt_40102,cnt_40103,day_cnt_40103,cnt_40104,day_cnt_40104,cnt_40105,day_cnt_40105,cnt_40106,day_cnt_40106,cnt_40107,day_cnt_40107,cnt_40201,day_cnt_40201,cnt_40202,day_cnt_40202,cnt_40203,day_cnt_40203,cnt_40301,day_cnt_40301,cnt_40401,day_cnt_40401,cnt_50101,day_cnt_50101,cnt_50102,day_cnt_50102,cnt_50103,day_cnt_50103,cnt_50201,day_cnt_50201,cnt_50202,day_cnt_50202,cnt_50203,day_cnt_50203,cnt_50204,day_cnt_50204,cnt_50205,day_cnt_50205,cnt_50206,day_cnt_50206,cnt_50207,day_cnt_50207,cnt_50301,day_cnt_50301,cnt_50302,day_cnt_50302,cnt_50303,day_cnt_50303,cnt_50401,day_cnt_50401,cnt_50402,day_cnt_50402,cnt_50403,day_cnt_50403,cnt_50501,day_cnt_50501,cnt_50502,day_cnt_50502,cnt_50503,day_cnt_50503,cnt_50601,day_cnt_50601,cnt_50602,day_cnt_50602,cnt_50701,day_cnt_50701,cnt_50702,day_cnt_50702,cnt_50703,day_cnt_50703,cnt_60101,day_cnt_60101,cnt_60102,day_cnt_60102,cnt_60103,day_cnt_60103,cnt_60201,day_cnt_60201,cnt_60202,day_cnt_60202,cnt_70101,day_cnt_70101,cnt_70102,day_cnt_70102,cnt_70201,day_cnt_70201,cnt_70202,day_cnt_70202,cnt_70203,day_cnt_70203,cnt_70204,day_cnt_70204,cnt_70205,day_cnt_70205,cnt_70206,day_cnt_70206,cnt_70207,day_cnt_70207,cnt_70208,day_cnt_70208,cnt_70301,day_cnt_70301,cnt_70302,day_cnt_70302,cnt_1,day_cnt_1,cnt_2,day_cnt_2".split(',') 

y_column = 'gbie'
v_to_idx = { 
             '0':0,
	         '1':1,
}

dp = DataPreprocessing (
                cates,
			    numbers,
			    y_column,
                v_to_idx
)

path='/home/lx/work/mmdataapp/wxalonso/wxqueryanlysis/data/zx_user_train/'
dp.catefeat_v_to_id_mapping( path + 'alonso_zx_test_interest_feas_201704.csv')

def read_adult_data():
  y_train,X_train,y_test,X_test = \
      dp.read_train_test (
           path + 'alonso_zx_test_interest_feas_201704.csv_train.txt',
	       path + 'alonso_zx_test_interest_feas_201704.csv_test.txt'
  )	
  return y_train,X_train,y_test,X_test

