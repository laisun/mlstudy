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

from sklearn.neural_network import BernoulliRBM,MLPClassifier
from sklearn import metrics

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
  precision_score = metrics.precision_score(y_test, y_pred)

  return cross_val_scores,average_precision_score,roc_auc_score,precision_score
'''
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
print "LR : ",test_scores(clf_lr)
print "SVM : ",test_scores(clf_svm)
print "RF : ",test_scores(clf_rf)
print "GBDT : " ,test_scores(clf_gbdt)


from sk_tree_lr import RF_LR,GBDT_LR
clf_rf_lr = RF_LR()
clf_rf_lr.fit(X, y)
print "RF_LR : ",clf_rf_lr.scores(X_test,y_test)

clf_rf_lr = GBDT_LR()
clf_rf_lr.fit(X, y)
print "GBDT_LR : ",clf_rf_lr.scores(X_test,y_test)
'''

# neural network
# require sklearn >= 0.18.0
mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
mlp.fit(X,y)
y_pred = mlp.predict(X_test)
print metrics.classification_report(y_test,y_pred)
print metrics.confusion_matrix(y_test,y_pred)


# nn + LR
logistic = linear_model.LogisticRegression()
rbm = BernoulliRBM(random_state=0, verbose=True)
rbm2 = BernoulliRBM(random_state=0, verbose=True)
classifier = Pipeline(
       steps=
         [
	    ('rbm', rbm),
#	    ('rbm2',rbm2),
	    ('logistic', logistic)
	 ])

rbm.learning_rate = 0.01
rbm.n_iter = 200 
rbm.n_components = 256 
rbm2.n_components = 50
rbm2.learning_rate = 0.01
rbm2.n_iter = 50
logistic.C = 1000.0

# Training RBM-Logistic Pipeline
classifier.fit(X, y)
print()
print("Logistic regression using RBM features:\n%s\n" % (
		metrics.classification_report(
		    y_test,
		    classifier.predict(X_test))))


