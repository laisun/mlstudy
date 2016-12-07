#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""
sklearn >= 0.17.1

"""

import os,sys

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing,linear_model
from sklearn import metrics
import numpy as np

class RF_LR(object):
  """Tree model plus Logistic Regression.

  Parameters
  ----------
  trees : the tree model to make feature transformation.
          
  encoder : the feature encoder to encode the result of trees.

  lr : the classification model,using the result of encoder.

  """

  def __init__(self,n_estimator=100,max_depth=3):
    """ make a initiation of the models.

    Parameters
    -----------
    n_estimator : the number of sub-trees.
    max_depth : the max depth of each sub-tree.

    """

    self.trees = RandomForestClassifier(
                           n_estimators=n_estimator,
			   max_depth=max_depth
            		)
    self.encoder = preprocessing.OneHotEncoder()
    self.lr = linear_model.LogisticRegression()

  def fit(self,X_train,y_train):
    """train the models.

    Parameters
    -----------
    X_train : array with shape = [n_samples, n_features],
              The input samples. Internally, its dtype will 
	      be converted to ``dtype=np.float32``.
    y_train : array with shape = [n_samples, 1] ,its dtype is int.

    """

    '''train the trees and encoder'''
    self.trees.fit(X_train, y_train)
    self.encoder.fit(self.trees.apply(X_train))
    
    '''get the tree features and encode'''
    X_train_encodes = self.encoder.transform(
                          self.trees.apply(X_train)
		       ).toarray()
  
    '''merge the source features as the final features'''
    X_train_merge = []
    for i in range(0,len(X_train_encodes)):
      X_train_merge.append(list(X_train[i]) + list(X_train_encodes[i]))	
    
    self.lr.fit(
             X_train_merge, 
             y_train
    )
  
  def predict(self,X_test):
    """Predict class for X.
      the predicted class is the one with highest probability.

    Parameters
    ----------
    X_test : array with shape = [n_samples, n_features],its dtype is float32.

    Returns
    -------
    y : array of shape = [n_samples,1]

    """
    
    X_test_encodes = self.encoder.transform(
                          self.trees.apply(X_test)
		      ).toarray()

    X_test_merge = []
    for i in range(0,len(X_test_encodes)):
      X_test_merge.append(list(X_test[i]) + list(X_test_encodes[i]))	
    y_pred = self.lr.predict(X_test_merge)
    return y_pred

  def predict_proba(self,X_test):
    X_test_encodes = self.encoder.transform(
                          self.trees.apply(X_test)
		      ).toarray()

    X_test_merge = []
    for i in range(0,len(X_test_encodes)):
      X_test_merge.append(list(X_test[i]) + list(X_test_encodes[i]))	
    y_pred = self.lr.predict_proba(X_test_merge)

    return y_pred

  def scores(self,X_test,y_test):
    """make a evaluation of the model.

    Parameters
    -----------
    X_test : array with shape = [n_samples, n_features],
              The input samples. Internally, its dtype will 
	      be converted to ``dtype=np.float32``.
    y_test : array with shape = [n_samples, 1] ,its dtype is int.
   

    Returns
    -------
    
    average_precision_score : float
       The the area under the precision-recall curve.
    
    roc_auc_score : float
        Area Under the Curve (AUC).
    
    precision_score: float
        The precision, TP/(TP+FP)

    """

    y_pred = self.predict(X_test)
    y_pred_proba = self.predict_proba(X_test)

    average_precision_score = metrics.average_precision_score(y_test, y_pred)
    roc_auc_score = metrics.roc_auc_score(y_test, y_pred_proba)
    precision_score = metrics.precision_score(y_test, y_pred)
    recall_score = metrics.recall_score(y_test, y_pred)
    accuracy_score = metrics.accuracy_score(y_test, y_pred)

    score_dict = {
	      "average_precision_score" : average_precision_score,
              "roc_auc_score" : roc_auc_score,
	      "precision_score" : precision_score,
	      "recall_score" : recall_score,
	      "accuracy_score" : accuracy_score 
	    }
    s = ""
    for m,v in score_dict.items():
      s += "{} = {:g}".format(m,v)
    return score_dict,s  

  def classification_report(self,X_test,y_test):
    """the report includes: precision recall f1_score et al.
    """

    y_pred = self.predict(X_test)
    return metrics.classification_report(y_test, y_pred)

class GBDT_LR(object):
  """Tree model plus Logistic Regression.

  Parameters
  ----------
  trees : the tree model to make feature transformation.
          
  encoder : the feature encoder to encode the result of trees.

  lr : the classification model,using the result of encoder.

  """

  def __init__(self,n_estimator=100):
    """ make a initiation of the models.

    Parameters
    -----------
    n_estimator : the number of sub-trees.
    max_depth : the max depth of each sub-tree.

    """

    self.trees = GradientBoostingClassifier(n_estimators=n_estimator)
    self.encoder = preprocessing.OneHotEncoder()
    self.lr = linear_model.LogisticRegression()

  def fit(self,X_train,y_train):
    """train the models.

    Parameters
    -----------
    X_train : array with shape = [n_samples, n_features],
              The input samples. Internally, its dtype will 
	      be converted to ``dtype=np.float32``.
    y_train : array with shape = [n_samples, 1] ,its dtype is int.

    """

    self.trees.fit(X_train, y_train)
    self.encoder.fit(self.trees.apply(X_train)[:, :, 0])

    X_train_encodes = self.encoder.transform(
                           self.trees.apply(X_train)[:, :, 0]
			).toarray()

    X_train_merge = []
    for i in range(0,len(X_train_encodes)):
      X_train_merge.append(list(X_train[i]) + list(X_train_encodes[i]))

    self.lr.fit(
            X_train_merge,
            y_train
	 )
  
  def predict(self,X_test):
    """Predict class for X.
      the predicted class is the one with highest probability.

    Parameters
    ----------
    X_test : array with shape = [n_samples, n_features],its dtype is float32.

    Returns
    -------
    y : array of shape = [n_samples,1]

    """

    X_test_encodes = self.encoder.transform(
                          self.trees.apply(X_test)[:, :, 0]
		      ).toarray()

    X_test_merge = []
    for i in range(0,len(X_test_encodes)):
      X_test_merge.append(list(X_test[i]) + list(X_test_encodes[i]))	
    y_pred = self.lr.predict(
                  X_test_merge
	       )

    return y_pred


  def predict_proba(self,X_test):
    """Predict class for X.
      the predicted class is the one with highest probability.

    Parameters
    ----------
    X_test : array with shape = [n_samples, n_features],its dtype is float32.

    Returns
    -------
    y : array of shape = [n_samples,1]

    """

    X_test_encodes = self.encoder.transform(
                          self.trees.apply(X_test)[:, :, 0]
		      ).toarray()

    X_test_merge = []
    for i in range(0,len(X_test_encodes)):
      X_test_merge.append(list(X_test[i]) + list(X_test_encodes[i]))	
    y_pred = self.lr.predict_proba(
                  X_test_merge
	       )

    return y_pred

  def scores(self,X_test,y_test):
    """make a evaluation of the model.

    Parameters
    -----------
    X_test : array with shape = [n_samples, n_features],
              The input samples. Internally, its dtype will 
	      be converted to ``dtype=np.float32``.
    y_test : array with shape = [n_samples, 1] ,its dtype is int.
   

    Returns
    -------
    
    average_precision_score : float
       The the area under the precision-recall curve.
    
    roc_auc_score : float
        Area Under the Curve (AUC).
    
    precision_score: float
        The precision, TP/(TP+FP)

    """

    y_pred = self.predict(X_test)
    y_pred_proba = self.predict_proba(X_test)

    average_precision_score = metrics.average_precision_score(y_test, y_pred)
    roc_auc_score = metrics.roc_auc_score(y_test, y_pred_proba)
    precision_score = metrics.precision_score(y_test, y_pred)
    recall_score = metrics.recall_score(y_test, y_pred)
    accuracy_score = metrics.accuracy_score(y_test, y_pred)


    score_dict = {
	      "average_precision_score" : average_precision_score,
              "roc_auc_score" : roc_auc_score,
	      "precision_score" : precision_score,
	      "recall_score" : recall_score,
	      "accuracy_score" : accuracy_score 
	    }
    s = ""
    for m,v in score_dict.items():
      s += "{} = {:g}".format(m,v)
    return score_dict,s  
  def classification_report(self,X_test,y_test):
    """the report includes: precision recall f1_score et al.
    """

    y_pred = self.predict(X_test)
    return metrics.classification_report(y_test, y_pred)
