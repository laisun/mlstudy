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

    self.trees = RandomForestClassifier(n_estimators=n_estimator,max_depth=max_depth)
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
    self.encoder.fit(self.trees.apply(X_train))

    self.lr.fit(
             self.encoder.transform(self.trees.apply(X_train)), 
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

    y_pred = self.lr.predict(
                 self.encoder.transform(self.trees.apply(X_test))
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
    average_precision_score = metrics.average_precision_score(y_test, y_pred)
    roc_auc_score = metrics.roc_auc_score(y_test, y_pred)
    precision_score = metrics.precision_score(y_test, y_pred)

    return average_precision_score,roc_auc_score,precision_score

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

    self.lr.fit(
            self.encoder.transform(self.trees.apply(X_train)[:, :, 0]), 
            y_train)
  
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

    y_pred = self.lr.predict(
                 self.encoder.transform(self.trees.apply(X_test)[:, :, 0])
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
    average_precision_score = metrics.average_precision_score(y_test, y_pred)
    roc_auc_score = metrics.roc_auc_score(y_test, y_pred)
    precision_score = metrics.precision_score(y_test, y_pred)

    return average_precision_score,roc_auc_score,precision_score

  def classification_report(self,X_test,y_test):
    """the report includes: precision recall f1_score et al.
    """

    y_pred = self.predict(X_test)
    return metrics.classification_report(y_test, y_pred)
