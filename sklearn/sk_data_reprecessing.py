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

def bucket_encode(column_value,boundaries=[]):
  if column_value < boundaries[0]: return 0  
  for i in range(1,len(boundaries) - 1):
    if column_value >= boundaries[i] and column_value < boundaries[i+1]:
      return i	
  if column_value >= boundaries[len(boundaries) - 1]:
      return len(boundaries) - 1
  return 0

class DataPreprocessing(object):
  def __init__(self,
               CATEGORICAL_COLUMNS,
	       CONTINUOUS_COLUMNS,
	       y_column,
	       y_to_idx,
	       BUCKET_COLUMNS = [],
	       bucket_boundaries = []):
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
    """
    cate_feats_count = collections.defaultdict(set)
    for row in csv.DictReader(open(csv_file)):
      for feat in self.CATEGORICAL_COLUMNS:	
        cate_feats_count[feat].update(row[feat].strip())

    cate_feat_id_mapping = collections.defaultdict(dict)
    for feat in self.CATEGORICAL_COLUMNS:
      cate_feat_id_mapping[feat] = set_to_dict(cate_feats_count[feat])

    return cate_feat_id_mapping

  def read_data(self,csv_file,y_column=None):
    """read the csv data file ,to get y labels list 
    and category column values and number column values.
    """

    y,X_num,X_cate = [],[],[]
    cate_feat_id_mapping = self.catefeat_v_to_id_mapping(csv_file)
    
    count = 0
    for row in csv.DictReader(open(csv_file)):
      count += 1
      # number features
      x1 = []
      for feat in self.CONTINUOUS_COLUMNS:
	w = 0.0
	if row[feat].strip() != 'NULL' and row[feat].strip() != '':
	  w = float(row[feat].strip()) 
	x1.append(w)

      # cate features
      x2 = [ cate_feat_id_mapping[feat][row[feat].strip()] 
             for feat in self.CATEGORICAL_COLUMNS ]
      for i in range(len(self.BUCKET_COLUMNS)):
	feat = self.BUCKET_COLUMNS[i]  
	bucket_value = bucket_encode(int(row[feat].strip()),
	                             self.bucket_boundaries[i])  
        x2.append(bucket_value)

      # y label
      if y_column:
        v = row[y_column].strip()
        if v in self.y_to_idx.keys():
          y.append(self.y_to_idx[v])
          X_num.append(x1)
          X_cate.append(x2)
      else:
          X_num.append(x1)
	  X_cate.append(x2)
    return y,X_num,X_cate

  def read_train_test(self,train_data_file,test_data_file):
    y_train,X_num,X_cate = self.read_data(train_data_file,self.y_column) 
    y_test,X_test_num,X_test_cate = self.read_data(test_data_file,self.y_column) 

    # data processing,feature transform
    # number feature processing
    X_all_num = np.array(X_num + X_test_num)
    X_num = np.array(X_num)
    X_test_num = np.array(X_test_num)
    self.num_scaler = preprocessing.StandardScaler().fit(X_num)
    
    X_num = self.num_scaler.transform(X_num)
    X_test_num = self.num_scaler.transform(X_test_num)

    # cate features processing
    self.cate_enc = preprocessing.OneHotEncoder()
    self.cate_enc.fit( X_cate+X_test_cate )
    
    X_cate = self.cate_enc.transform(X_cate).toarray()
    X_test_cate = self.cate_enc.transform(X_test_cate).toarray()

    X_train = []
    for i in range(X_num.shape[0]):
      X_train.append(list(X_num[i]) + list(X_cate[i]))

    X_test = []
    for i in range(X_test_num.shape[0]):
      X_test.append(list(X_test_num[i]) + list(X_test_cate[i]))
  
    return y_train,X_train,y_test,X_test

  def read_dev_data(self,dev_csv):
    y_dev,X_num,X_cate = self.read_data(dev_csv,None) 
    X_num = self.num_scaler.transform(X_num)
    X_cate = self.cate_enc.transform(X_cate).toarray()

    X_dev = []
    for i in range(X_num.shape[0]):
      X_dev.append(list(X_num[i]) + list(X_cate[i]))	
    return X_dev,y_dev

