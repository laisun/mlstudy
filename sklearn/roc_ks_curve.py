
#-*- coding:utf-8 -*-

import sklearn
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cnf_matrix, y_classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(y_classes))
    plt.xticks(tick_marks, y_classes, rotation=45)
    plt.yticks(tick_marks, y_classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_c(y_test, y_pred):
  cnf_matrix = confusion_matrix(y_test, y_pred)
  np.set_printoptions(precision=2)
  # Plot non-normalized confusion matrix
  plt.figure()
  plot_confusion_matrix(cnf_matrix, classes=['1', '0' ],
                      title='Confusion matrix, without normalization')

  # Plot normalized confusion matrix
  plt.figure()
  plot_confusion_matrix(cnf_matrix, classes=['1','0'], normalize=True,
                      title='Normalized confusion matrix')

  plt.show()


def plot_auc(y_test,y_pred_prob):
  fpr, tpr, thresholds = metrics.roc_curve(y_test,y_pred_prob)
  roc_auc = metrics.auc(fpr,tpr)

  # Plot ROC
  plt.title('auc')
  plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
  plt.legend(loc='lower right')
  plt.plot([0,1],[0,1],'r--')
  plt.xlim([-0.1,1.0])
  plt.ylim([-0.1,1.01])
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.show()


def plot_ks(y_test,y_pred_prob):
  fpr, tpr, thresholds = metrics.roc_curve(y_test,y_pred_prob)
  l = len(fpr)
  x = [i*1.0 / l for i in xrange(l) ]
  ks = max([math.fabs(fpr[i] - tpr[i]) for i in xrange(l)])
  plt.title('KS = $max(tpr - fpr)$ = %.2f' % ks)
  plt.plot(x, fpr, 'b',label='fpr' )
  plt.plot(x, tpr, 'r',label='tpr' )
  plt.legend(loc='upper left')
  plt.show()

  
