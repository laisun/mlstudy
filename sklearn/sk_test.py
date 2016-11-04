#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os,sys
import csv

import sklearn
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.cross_validation import train_test_split,cross_val_score
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
    x1 = [ float(row[feat].strip()) for feat in CONTINUOUS_COLUMNS ]
    X_num.append(x1)
    x2 = [ cate_feat_id_mapping[feat][row[feat].strip()] 
             for feat in CATEGORICAL_COLUMNS ]
    X_cate.append(x2)
    if ">50K" in row[label]:
      y.append(1)
    else:
      y.append(0)
  
  return y,X_num,X_cate

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

# model training
X = []
for i in range(X_num.shape[0]):
  X.append(list(X_num[i]) + list(X_cate[i]))

X_test = []
for i in range(X_test_num.shape[0]):
  X_test.append(list(X_test_num[i]) + list(X_test_cate[i]))

#clf = RandomForestClassifier(n_estimators=30)
clf = sklearn.svm.SVC()#sklearn.linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf = clf.fit(X, y)

# evalutation    
scores = cross_val_score(clf,X_test,y_test,cv=3,scoring='average_precision')
print "cross_val_scor",scores

y_pred = clf.predict(X_test)
print "average_precision_score=",metrics.average_precision_score(y_test, y_pred)
print "roc_auc_score=",metrics.roc_auc_score(y_test, y_pred)
   
 
