#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os,sys
import csv

from sklearn import preprocessing
from sklearn import metrics
import numpy as np
import collections

def v_to_idx(value_list):
  li = list(set(value_list))
  return dict((v, k) for k, v in enumerate(li))

class DataPreprocessing(object):
  def __init__(self,CATEGORICAL_COLUMNS,CONTINUOUS_COLUMNS,y_column,y_to_idx):
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
    self.y_column = y_column
    self.y_to_idx = y_to_idx 

  def catefeat_v_to_id_mapping(self,csv_file):
    """get the value to id mapping for all of the cate columns.
    """

    cate_feats_count = collections.defaultdict(list)
    for row in csv.DictReader(open(csv_file)):
      for feat in self.CATEGORICAL_COLUMNS:	
        cate_feats_count[feat].append(row[feat].strip())

    cate_feat_id_mapping = collections.defaultdict(dict)
    for feat in self.CATEGORICAL_COLUMNS:
      cate_feat_id_mapping[feat] = v_to_idx(cate_feats_count[feat])

    return cate_feat_id_mapping

  def read_data(self,csv_file):
    """read the csv data file ,to get y labels list 
    and category column values and number column values.
    """

    y,X_num,X_cate = [],[],[]
    cate_feat_id_mapping = self.catefeat_v_to_id_mapping(csv_file)

    for row in csv.DictReader(open(csv_file)):
      # number features
      x1 = []
      for feat in self.CONTINUOUS_COLUMNS:
        w = float(row[feat].strip()) 
	x1.append(w)
      X_num.append(x1)

      # cate features
      x2 = [ cate_feat_id_mapping[feat][row[feat].strip()] 
             for feat in self.CATEGORICAL_COLUMNS ]
      X_cate.append(x2)
      
      # y label 
      v = row[self.y_column].strip()
      y.append(self.y_to_idx[v])

    return y,X_num,X_cate

  def read_train_test(self,train_data_file,test_data_file):
    y_train,X_num,X_cate = self.read_data(train_data_file) 
    y_test,X_test_num,X_test_cate = self.read_data(test_data_file) 

    # data processing,feature transform
    # number feature processing
    X_all_num = np.array(X_num + X_test_num)
    X_num = np.array(X_num)
    X_test_num = np.array(X_test_num)
    scaler = preprocessing.StandardScaler().fit(X_num)
    X_num = scaler.transform(X_num)
    X_test_num = scaler.transform(X_test_num)

    # cate features processing
    enc = preprocessing.OneHotEncoder()
    enc.fit( X_cate+X_test_cate )
    X_cate = enc.transform(X_cate).toarray()
    X_test_cate = enc.transform(X_test_cate).toarray()

    X_train = []
    for i in range(X_num.shape[0]):
      X_train.append(list(X_num[i]) + list(X_cate[i]))

    X_test = []
    for i in range(X_test_num.shape[0]):
      X_test.append(list(X_test_num[i]) + list(X_test_cate[i]))
  
    return y_train,X_train,y_test,X_test


