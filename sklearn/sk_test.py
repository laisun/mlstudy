#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os,sys
import csv

import sklearn
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split,cross_val_score
from sklearn import preprocessing,linear_model
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel

import numpy as np
import collections

from adult_data import *

y,X,y_test,X_test = read_adult_data()

def test_scores(clf):
  clf = clf.fit(X, y)
  y_pred = clf.predict(X_test)
  
  cross_val_scores = cross_val_score(clf,X_test,y_test,cv=3,scoring='average_precision')
  average_precision_score = metrics.average_precision_score(y_test, y_pred)
  roc_auc_score = metrics.roc_auc_score(y_test, y_pred)
  return cross_val_scores,average_precision_score,roc_auc_score

clf_rf = RandomForestClassifier(n_estimators=100)
clf_gbdt = GradientBoostingClassifier(n_estimators=100)
clf_svm = sklearn.svm.SVC()
clf_lr = sklearn.linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)

clf_fc = Pipeline([
        ('feature_selection', SelectFromModel(RandomForestClassifier())),
        ('classification', sklearn.linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6))
     ])
clf_fc.fit(X, y)

print "FC :",test_scores(clf_fc)
'''
print "LR : ",test_scores(clf_lr)
print "SVM : ",test_scores(clf_svm)
print "RF : ",test_scores(clf_rf)
print "GBDT : " ,test_scores(clf_gbdt)

'''

from sk_tree_lr import RF_LR 
clf_rf_lr = RF_LR(tree='gbdt')
clf_rf_lr.fit(X, y)
print clf_rf_lr.scores(X_test,y_test) 
