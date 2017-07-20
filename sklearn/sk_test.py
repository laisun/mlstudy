#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os,sys
import csv
import math
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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

#from adult_data import *
#from wx_marry_data import *
from wx_game_query_feas import *
#from wx_zx_querys_data import *
from roc_ks_curve import *

y,X,y_test,X_test = read_adult_data()

def test_scores(clf):
  clf = clf.fit(X, y)
  y_pred = clf.predict(X_test)
  y_proba = clf.predict_proba(X_test)  
  y_proba = [p[1] for p in y_proba]

  roc_auc_score = metrics.roc_auc_score(y_test,y_proba)
  precision_score = metrics.precision_score(y_test, y_pred)
  recall_score = metrics.recall_score(y_test, y_pred)
  accuracy_score = metrics.accuracy_score(y_test, y_pred)
  print metrics.classification_report(y_test,y_pred)

  plot_ks(y_test,y_proba)
  plot_auc(y_test,y_proba)

  return "auc = {:g}, precison = {:g}, recall = {:g}, acc = {:g}".format\
	 (roc_auc_score,precision_score,recall_score,accuracy_score)

clf_rf = RandomForestClassifier(n_estimators=10)
clf_gbdt = GradientBoostingClassifier(n_estimators=10)
clf_svm = sklearn.svm.SVC()
clf_lr = sklearn.linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6)

clf_fc = Pipeline([
        ('feature_selection', SelectFromModel(RandomForestClassifier())),
        ('classification', sklearn.linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6))
])

'''
print "LR  :",test_scores(clf_lr)
#print "SVM :",test_scores(clf_svm)
print "RF  :",test_scores(clf_rf)
print "GBDT:",test_scores(clf_gbdt)
print "FC  :",test_scores(clf_fc)
'''

from sk_tree_lr import RF_LR,GBDT_LR
clf_rf_lr = RF_LR(n_estimator=20)
print "RF_LR:", test_scores(clf_rf_lr)

clf_rf_lr = GBDT_LR(n_estimator=20)
print "GBDT_LR:", test_scores(clf_rf_lr)

# neural network
# require sklearn >= 0.18.0
mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))

mlp.fit(X,y)
y_pred = mlp.predict(X_test)
print "MLP classifier "
print metrics.classification_report(y_test,y_pred)
print metrics.confusion_matrix(y_test,y_pred)
print 

# nn + LR
logistic = linear_model.LogisticRegression()
rbm = BernoulliRBM(random_state=0, verbose=True)
classifier = Pipeline(
       steps=
         [
	    ('rbm', rbm),
	    ('logistic', logistic)
	 ])

rbm.learning_rate = 0.01
rbm.n_iter = 200 
rbm.n_components = 256 
logistic.C = 1.0

# Training RBM-Logistic Pipeline
classifier.fit(X, y)
print()
print("Logistic regression using RBM features:\n%s\n" % (
		metrics.classification_report(
		    y_test,
		    classifier.predict(X_test))))
			
'''
from keras.models import Sequential
from keras.layers import Dense,Dropout

model = Sequential()
model.add(
            Dense(
               input_dim = len(X[0]),
               output_dim = 200, 
               activation='tanh',
               init='uniform')
)
model.add(Dropout(0.5))
model.add(Dense(30,activation='tanh'))
model.add(Dense(1,activation='sigmoid'))

model.compile(
      loss='binary_crossentropy',
      optimizer='adadelta',
      metrics=['accuracy']
)

model.fit(X,y,
          nb_epoch=20,
          batch_size=16
)

print model.evaluate(X_test,y_test,batch_size=16)

'''
