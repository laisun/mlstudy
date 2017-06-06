#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os,sys
import csv

import sklearn
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing,linear_model

from sklearn import metrics

from sklearn.metrics import confusion_matrix

import numpy as np
import collections

from gamebbs_data import *
y,X,y_test,X_test = read_adult_data()

def print_confusion_matrix(y_test,y_pred):
  labels = list(set(y_pred))
  conf_mat = confusion_matrix(y_test, y_pred, labels = labels)
  print "confusion_matrix(left labels: y_true, up labels: y_pred):"  
  print "labels\t",  
  for i in range(len(labels)):  
      print labels[i],"\t",  
  print   
  for i in range(len(conf_mat)): 
    print i,"\t",  
    for j in range(len(conf_mat[i])):  
      print conf_mat[i][j],'\t',  
    print   
  print  

def test_scores(clf):
  clf = clf.fit(X, y)
  y_pred = clf.predict(X_test)
  print_confusion_matrix(y_test,y_pred)
  print metrics.classification_report(y_test,y_pred)
  

# 1. 基础分类器
clf_svm = sklearn.svm.SVC()
clf_rf = RandomForestClassifier(n_estimators=10)
clf_gbdt = GradientBoostingClassifier(n_estimators=10)

print "SVM :", test_scores(clf_svm)
print "RF  :", test_scores(clf_rf)
print "GBDT : " ,test_scores(clf_gbdt)


