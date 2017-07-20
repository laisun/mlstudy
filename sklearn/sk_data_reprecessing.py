#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os,sys
import csv

from sklearn import preprocessing
from sklearn import metrics
import numpy as np
import collections

def set_to_dict(value_set):
  li = list(value_set)
  return dict((v, k) for k, v in enumerate(li))

def bucket_encode(column_value, boundaries=[]):
  if column_value < boundaries[0]: return 0  
  for i in range(1,len(boundaries) - 1):
    if column_value >= boundaries[i] and column_value < boundaries[i+1]:
      return i    
  if column_value >= boundaries[len(boundaries) - 1]:
      return len(boundaries) - 1
  return 0

class DataPreprocessing(object):
  def __init__(self,
           CATEGORICAL_COLUMNS, # 离散值属性
           CONTINUOUS_COLUMNS,  # 连续值属性
           y_column,            # y 标签属性
           y_to_idx,            # y 取值到整数的映射
           BUCKET_COLUMNS = [], # 分桶属性，例如年龄分成各个年龄段
           bucket_boundaries = []): # 分桶属性每个桶的边界
    """process the cate features and number features respectively.
    
    Parameters
    -----------
    CATEGORICAL_COLUMNS : list of cate_column names.
    CONTINUOUS_COLUMNS : list of number_column names.
    y_column : str 
         y column name.
    y_to_idx : dict
         the mapping of y label name to id. 
     e.g. {'male':0,'female':1 }

    """
    self.CATEGORICAL_COLUMNS = CATEGORICAL_COLUMNS
    self.CONTINUOUS_COLUMNS = CONTINUOUS_COLUMNS
    self.BUCKET_COLUMNS = BUCKET_COLUMNS
    self.bucket_boundaries = bucket_boundaries
    self.y_column = y_column
    self.y_to_idx = y_to_idx 

  def catefeat_v_to_id_mapping(self,csv_file):
    """get the value to id mapping for all of the cate columns.
       获取每个离散属性的< 属性值->id > 的映射
    """
    cate_feats_count = collections.defaultdict(set)
    for row in csv.DictReader(open(csv_file)):
      for feat in self.CATEGORICAL_COLUMNS:    
        cate_feats_count[feat].update([row[feat].strip().lower()])

    self.cate_feat_id_mapping = collections.defaultdict(dict)
    for feat in self.CATEGORICAL_COLUMNS:
      self.cate_feat_id_mapping[feat] = set_to_dict(cate_feats_count[feat])


  def read_data(self, csv_file, y_column=None):
    """read the csv data file ,to get y labels list 
    and category column values and number column values.
    """

    y,X_num,X_cate = [],[],[]

    for row in csv.DictReader(open(csv_file)):
      # number features
      x1 = []
      for feat in self.CONTINUOUS_COLUMNS:
        w = 0.0
        if row[feat].strip() != 'NULL' and row[feat].strip() != '':
          w = float(row[feat].strip()) 
        x1.append(w)

      # cate features
      x2 = [ self.cate_feat_id_mapping[feat][row[feat].strip().lower()] 
                           for feat in self.CATEGORICAL_COLUMNS ]

      for i in range(len(self.BUCKET_COLUMNS)):
        feat = self.BUCKET_COLUMNS[i]  
        bucket_value = bucket_encode(int(row[feat].strip()),
                                 self.bucket_boundaries[i])  
        x2.append(bucket_value)

      X_num.append(x1)
      X_cate.append(x2)

      # y label
      if y_column:
        v = row[y_column].strip()
        if v in self.y_to_idx.keys():
          y.append(self.y_to_idx[v])

    return y,X_num,X_cate

  def read_train_test(self,train_data_file,test_data_file):
    y_train,X_num,X_cate = self.read_data(train_data_file,self.y_column) 
    y_test,X_test_num,X_test_cate = self.read_data(test_data_file,self.y_column) 

    # 连续值处理: 标准化缩放
    X_all_num = np.array(X_num + X_test_num)
    X_num = np.array(X_num)
    X_test_num = np.array(X_test_num)

    self.num_scaler = preprocessing.MinMaxScaler()#StandardScaler()
    self.num_scaler.fit(X_all_num)
    
    X_num = self.num_scaler.transform(X_num)
    X_test_num = self.num_scaler.transform(X_test_num)

    # 离散值处理: One-hot-encoding 
    if len(self.CATEGORICAL_COLUMNS) > 0:
      self.cate_enc = preprocessing.OneHotEncoder()
      self.cate_enc.fit( X_cate + X_test_cate )
      X_cate = self.cate_enc.transform(X_cate).toarray()
      X_test_cate = self.cate_enc.transform(X_test_cate).toarray()
    else:
      X_cate = None
      X_test_cate = None
    X_train = self.add_num_cate_values(X_num, X_cate)
    X_test = self.add_num_cate_values(X_test_num , X_test_cate)
      
    return y_train,X_train,y_test,X_test

  def read_dev_data(self, dev_csv):
    y_dev,X_num,X_cate = self.read_data(dev_csv,None) 
    X_num = self.num_scaler.transform(X_num)
    X_cate = self.cate_enc.transform(X_cate).toarray()
    X_dev = self.add_num_cate_values(X_num , X_cate) 
    return X_dev,y_dev

  def add_num_cate_values(self, X_num, X_cate):
    X = []
    if X_cate is None:
      shape1 = X_num.shape[0]
      for i in range(shape1):
        x = list(X_num[i])
        X.append(x)
      return X

    assert(X_num.shape[0] == X_cate.shape[0])
    shape1 = X_num.shape[0]
    for i in range(shape1):
      x = []
      if len(self.CONTINUOUS_COLUMNS) > 0:
        x += list(X_num[i])
      if len(self.CATEGORICAL_COLUMNS) > 0:
        x += list(X_cate[i])
      X.append(x)
    return X
