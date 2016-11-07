#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os,sys
import csv

from sklearn import preprocessing
from sklearn import metrics
import numpy as np
import collections

CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                    "relationship", "race", "gender", "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",
                   "hours_per_week"]

label = 'income_bracket'
feats = CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS

def number_to_bucket(x,boundaries = []):
  if len(boundaries) == 0: return x
  if x <= boundaries[0]: return 0
  for i in range(len(boundaries)-1):
    if x < boundaries[i+1] and x >= boundaries[i]:
      return i+1
  if x > boundaries[len(boundaries)-1]:
    return len(boundaries)
  
  return -1

def v_to_idx(value_list):
  d = dict()
  v_to_idx = dict()
  for v in value_list:
    if v not in d: 
      v_to_idx[v] = len(v_to_idx)
      d[v] = 1
  return v_to_idx

# 类别属性的值到idx的映射
def feat_value_to_idx_mapping(csv_file,cate_columns=[]):
  # 每个类别属性的所有取值
  cate_feats_count = collections.defaultdict(list)
  for row in csv.DictReader(open(csv_file)):
    for feat in cate_columns:
      cate_feats_count[feat].append(row[feat].strip())
  
  cate_feat_id_mapping = collections.defaultdict(dict)
  for feat in cate_columns:
    cate_feat_id_mapping[feat] = v_to_idx(cate_feats_count[feat])
  return cate_feat_id_mapping

def read_data(csv_file):
  y,X_num,X_cate = [],[],[]
  cate_feat_id_mapping = feat_value_to_idx_mapping(csv_file,CATEGORICAL_COLUMNS)

  for row in csv.DictReader(open(csv_file)):
    x1 = []
    for feat in CONTINUOUS_COLUMNS:
      w = float(row[feat].strip()) 	
      if feat=='age':
        x1.append(number_to_bucket(w,[18, 25, 30, 35, 40, 45,50, 55, 60, 65]))
      else:
	x1.append(w)
    X_num.append(x1)
    x2 = [ cate_feat_id_mapping[feat][row[feat].strip()] 
             for feat in CATEGORICAL_COLUMNS ]
    X_cate.append(x2)
    if ">50K" in row[label]:
      y.append(1)
    else:
      y.append(0)
  
  return y,X_num,X_cate

def read_adult_data():
  y,X_num,X_cate = read_data('./data/adult.data')
  y_test,X_test_num,X_test_cate = read_data('./data/adult.test')

  # data processing,feature transform
  X_num = np.array(X_num)
  X_test_num = np.array(X_test_num)

  #X = preprocessing.scale(X)
  #X_test = preprocessing.scale(X_test)
  scaler = preprocessing.StandardScaler().fit(X_num)
  scaler = preprocessing.StandardScaler().fit(X_test_num)
  X_num = scaler.transform(X_num)
  X_test_num = scaler.transform(X_test_num)

  enc = preprocessing.OneHotEncoder()
  enc.fit(X_cate)
  X_cate = enc.transform(X_cate).toarray()
  X_test_cate = enc.transform(X_test_cate).toarray()

  X = []
  for i in range(X_num.shape[0]):
    X.append(list(X_num[i]) + list(X_cate[i]))

  X_test = []
  for i in range(X_test_num.shape[0]):
    X_test.append(list(X_test_num[i]) + list(X_test_cate[i]))
  
  return y,X,y_test,X_test

