
#-*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler

# Read in our input data
df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test.csv')

# This prints out (rows, columns) in each dataframe
print('Train shape:', df_train.shape)
print('Test shape:', df_test.shape)
print('Columns:', df_train.columns)

y_train = df_train['target'].values
id_train = df_train['id'].values
id_test = df_test['id'].values

# We drop these variables as we don't want to train on them
# The other 57 columns are all numerical and can be trained on without preprocessing
x_train = df_train.drop(['target', 'id'], axis=1)
x_test = df_test.drop(['id'], axis=1)

# Take a random 20% of the dataset as validation data
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)

print('Train samples: {} Validation samples: {}'.format(len(x_train), len(x_valid)))



# Define the gini metric - from https://www.kaggle.com/c/ClaimPredictionChallenge/discussion/703#5897
def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)

def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

# Create an XGBoost-compatible metric from Gini
def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    roc_auc_score = metrics.roc_auc_score(labels, preds)
    return [('gini', gini_score), ('auc', roc_auc_score)]

# 子模型1. xgboost
# Convert our data into XGBoost format
d_train = xgb.DMatrix(x_train, y_train)
d_valid = xgb.DMatrix(x_valid, y_valid)
d_test = xgb.DMatrix(x_test)

# Set xgboost parameters
params = {}
params['objective'] = 'binary:logistic'
params['eta'] = 0.02
params['silent'] = True
params['max_depth'] = 6
params['subsample'] = 0.9
params['colsample_bytree'] = 0.9
# This is the data xgboost will test on after eachboosting round
watchlist = [(d_train, 'train'), (d_valid, 'valid')]

# Train the model! We pass in a max of 10,000 rounds (with early stopping after 100)
# and the custom metric (maximize=True tells xgb that higher metric is better)
mdl = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=100, feval=gini_xgb, maximize=True, verbose_eval=10)

# 子模型2. GBDT
clf_gbdt = GradientBoostingClassifier(n_estimators=20)
clf_gbdt.fit(x_train, y_train)
y_proba_gbdt = [p[1] for p in clf_gbdt.predict_proba(x_valid) ]
print "GBDT auc=", metrics.roc_auc_score(y_valid, y_proba_gbdt)
print "GBDT gini=", gini_normalized(y_valid, y_proba_gbdt)

# 子模型3. GBDT_LR
from sk_tree_lr import *
clf_gbdt_lr = GBDT_LR(n_estimator = 20)
clf_gbdt_lr.fit( np.array(x_train).tolist(), y_train)
y_proba = clf_gbdt_lr.predict_proba( np.array(x_valid).tolist() )
y_proba_gbdt_lr = [p[1] for p in y_proba ]
print "GBDT_LR auc=", metrics.roc_auc_score( y_valid, y_proba_gbdt_lr )
print "GBDT_LR gini=",gini_normalized( y_valid, y_proba_gbdt_lr )

# 模型组合
p_train_df = pd.DataFrame()
p_train_df['xgb'] = mdl.predict(d_train)
p_train_df['gbdt'] = [p[1] for p in clf_gbdt.predict_proba(x_train) ]
p_train_df['gbdt_lr'] = [p[1] for p in clf_gbdt_lr.predict_proba( np.array(x_train).tolist() ) ]

lr = LogisticRegression()
lr.fit(p_train_df, y_train)

p_valid_df = pd.DataFrame()
p_valid_df['xgb'] = mdl.predict(d_valid)
p_valid_df['gbdt'] = y_proba_gbdt
p_valid_df['gbdt_lr'] = y_proba_gbdt_lr
y_proba_lr = [p[1] for p in lr.predict_proba( p_valid_df )]
print "lr auc=", metrics.roc_auc_score(y_valid, y_proba_lr)
print "lr gini=",gini_normalized(y_valid, y_proba_lr)


# 生产环境预测
# Predict on our test data
p_test_df = pd.DataFrame()
p_test_df['xgb'] = mdl.predict(d_test)
p_test_df['gbdt'] = [p[1] for p in clf_gbdt.predict_proba(x_test) ]
p_test_df['gbdt_lr'] = [p[1] for p in clf_gbdt_lr.predict_proba( np.array(x_test).tolist() ) ]
p_test = [p[1] for p in lr.predict_proba( p_test_df )]

# Create a submission file
sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = p_test
sub.to_csv('xgb1.csv', index=False)

print(sub.head())
