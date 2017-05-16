# -*- coding: utf-8 -*-
"""
Created on Mon May 15 09:38:29 2017

@author: chengchaoyang
"""

import pandas as pd 
#import numpy as np
#import scipy.io as sio
#from collections import Counter
#from sklearn.model_selection import train_test_split
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn import neighbors
#from sklearn import datasets, linear_model
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
#from imblearn.over_sampling import     SMOTE
from sklearn import datasets
from transf import classify, transf,roccrossvalid
from sklearn import neighbors
from sklearn.model_selection import cross_val_predict
# # import some data to play with
# iris = datasets.load_iris()
# clf = neighbors.KNeighborsClassifier(n_neighbors=1)
# predicted = cross_val_predict(clf, iris.data, iris.target, cv=10)
# metrics.accuracy_score(iris.target, predicted)

data = pd.read_csv('data_for_student_case.csv')
dfcs=transf(data)
dffeatures = dfcs[['issuercountrycode','shopperinteraction','amountEUR','bookingdate','cardverificationcodesupplied','cvcresponsecode']]
# dffeatures = dfcs[['issuercountrycode','shopperinteraction','amountEUR','bookingdate','cardverificationcodesupplied','cvcresponsecode']]
labels = dfcs['simple_journal']
roccrossvalid(dffeatures,labels,'KNN(K=1)')
# classify(dffeatures,labels,'KNN(K=1)')#with smote
#1nn_sm_6f
# TP: 57
# FP: 316
# FN: 15
# TN: 44314
#2nn_sm_6f
# TP: 55
# FP: 250
# FN: 13
# TN: 44384
#neural_sm_6f
# TP: 60
# FP: 1604
# FN: 3
# TN: 43035
#neural_time_sm
# TP: 210
# FP: 17129
# FN: 116
# TN: 27247
#neural_time_sm_0.9
# TP: 256
# FP: 4576
# FN: 62
# TN: 17457
# neural_time_sm_0.9
# TP: 155
# FP: 13554
# FN: 178
# TN: 97868