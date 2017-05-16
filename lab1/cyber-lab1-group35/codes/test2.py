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


data = pd.read_csv('data_for_student_case.csv')
dfcs=transf(data)
dffeatures = dfcs[['issuercountrycode','shopperinteraction','amountEUR','bookingdate','cardverificationcodesupplied','cvcresponsecode']]
labels = dfcs['simple_journal']

#on clfname input LR, neural,1nn for linear regression, neural network(slow),nearest nrighbor
#with smote, do classification on time-based split train-test set
#classify(dffeatures,labels,'clfname')

#on clfname input LR, neural,1nn for linear regression, neural network(slow),nearest nrighbor
#with smote do 10folds cv on random split train-test set
#roccrossvalid(dffeatures,labels,'clfname')
