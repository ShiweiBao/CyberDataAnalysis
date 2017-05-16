#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 18:50:30 2017

@author: chengchaoyang
"""
import pandas as pd 
import numpy as np
import scipy.io as sio
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors
from sklearn import datasets, linear_model
from sklearn import metrics
from sklearn.metrics import roc_curve, auc,accuracy_score
from imblearn.over_sampling import     SMOTE
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
#
import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.datasets import make_classification
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
##-----Let's do the classification-------##
##-----try 3 classifiers with both SMOTED and UNSMOTED data------##


def classify(dffeatures,labels,clfname):
    if clfname=='1nn':
        clf = neighbors.KNeighborsClassifier(n_neighbors=1)
    elif clfname=='neural':
        clf=MLPClassifier(hidden_layer_sizes=(300, ))
    elif clfname=='LR':
        clf = linear_model.LinearRegression()
    else:
        print('input LR, 1nn or neural !')
    TP, FP, FN, TN = 0, 0, 0, 0
    x_array = np.array(dffeatures)
    y_array = np.array(labels)
    usx = x_array
    usy = y_array
    # x_train, x_test, y_train, y_test = train_test_split(usx, usy, test_size = 0.2)#test_size: proportion of train/test data
    propotion=0.5
    length=int(propotion*len(usx))
    x_train=usx[:length,[0,1,2,4,5]]
    y_train=usy[:length]
    x_test=usx[length:,[0,1,2,4,5]]
    y_test=usy[length:]
    #--- classifier 1:  KNN

    #---classifier 2: Linear Regression
    # Create linear regression object
    # clf= linear_model.LinearRegression()

    # Train the model using the training sets
#     regr.fit(x_train, y_train)

     #classifier3: Neural Network
    # clf=MLPClassifier(hidden_layer_sizes=(300, ))


    ## -----do SMOTE on training data here---------## 
    ## -----remember : split the data first, then do SMOTE on train set. NEVER do test on modified set.
    ## uncommit the following sentence to do SMOTE
    x_train, y_train = dosm(x_train, y_train)
    
    
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    #using 10 cross validation
    # y_predict = cross_val_predict(clf, x_train, y_train, cv=10)

    for i in range(len(y_predict)):
        if y_test[i]==1 and y_predict[i]==1:
            TP += 1
        if y_test[i]==0 and y_predict[i]==1:
            FP += 1
        if y_test[i]==1 and y_predict[i]==0:
            FN += 1
        if y_test[i]==0 and y_predict[i]==0:
            TN += 1
    print ('TP: '+ str(TP))
    print ('FP: '+ str(FP))
    print ('FN: '+ str(FN))
    print ('TN: '+ str(TN))
    print ('accuracy'+str(accuracy_score(y_test,y_predict)))
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for '+ clfname)
    plt.legend(loc="lower right")
    ##------uncommited the sentence below to save the fig locally
#     plt.savefig('ROC-KNN', ext='png', dpi=150)
    plt.show()

    return

##---SMOTE function: oversample the minor class to same size with major class, just for train set
def dosm(x_train, y_train):
    print('Original dataset shape {}'.format(Counter(y_train)))
    sm = SMOTE()
    X_res, y_res = sm.fit_sample(x_train, y_train)
    print('Resampled dataset shape {}'.format(Counter(y_res)))
    x_train = X_res
    y_train = y_res
    return x_train, y_train





def perday(df,DateName):

     df[DateName]=pd.to_datetime(df[DateName])
     df=df.sort_values(by=DateName)
     df[DateName]=df[DateName].dt.date
     return df
     

 #cardverificationresponsesupplied: did the shopper provide his 3 digit CVC/CVV2 code?
 #cvcresponsecode: Validation result of the CVC/CVV2 code: 0 = Unknown, 1=Match, 2=No Match, 3-6=Not checked     

def transf(df):
    currencyconvert = {"AUD":0.680237,"MXN":0.0487438,"NZD":0.630313,"GBP":1.18285,"SEK":0.103558}
    amountEUR = []
    df=df.dropna()
    for i in df.index:
        if df['currencycode'][i] == "MXN":
            amountEUR.append(currencyconvert["MXN"]*df['amount'][i])
#         print(datacharge.loc[i,['currencycode']])
        if df['currencycode'][i] == "AUD":
            amountEUR.append(currencyconvert["AUD"]*df['amount'][i])

        if df['currencycode'][i] == "NZD":
            amountEUR.append(currencyconvert["NZD"]*df['amount'][i])

        if df['currencycode'][i] == "GBP":
            amountEUR.append(currencyconvert["GBP"]*df['amount'][i])

        if df['currencycode'][i] == "SEK":
            amountEUR.append(currencyconvert["SEK"]*df['amount'][i])
#amountEUR
    df['amountEUR']=amountEUR
    df=df.loc[:,['issuercountrycode','shopperinteraction','amountEUR','bookingdate','simple_journal','cardverificationcodesupplied','cvcresponsecode']]
    dfcs=df[(df['simple_journal']=='Chargeback') | (df['simple_journal']=='Settled')]


#from datetime to date difference
    dfcs=perday(dfcs,'bookingdate')
    firstday=dfcs.loc[17903,'bookingdate']
    dfcs['bookingdate'] = dfcs['bookingdate']-firstday
    dfcs['bookingdate']=dfcs['bookingdate'] / np.timedelta64(1, 'D')

    dfcs.loc[dfcs['cardverificationcodesupplied']==False,'cardverificationcodesupplied']=0.0
    dfcs.loc[dfcs['cardverificationcodesupplied']==True,'cardverificationcodesupplied']=1.0
    dfcs=dfcs[(dfcs['cardverificationcodesupplied']==0) | (dfcs['cardverificationcodesupplied']==1)]
    dfcs.loc[:,'cardverificationcodesupplied']=pd.to_numeric(dfcs.loc[:,'cardverificationcodesupplied'])

    y = dfcs[dfcs['simple_journal'] == 'Settled']['issuercountrycode'].value_counts()
    for i in range(len(y)):
        dfcs.loc[dfcs['issuercountrycode'] == y.index[i], 'issuercountrycode'] = i

    dfcs.loc[:, 'issuercountrycode'] = pd.to_numeric(dfcs.loc[:, 'issuercountrycode'])

    dfcs.loc[dfcs['shopperinteraction'] == 'Ecommerce', 'shopperinteraction'] = 0.0
    dfcs.loc[dfcs['shopperinteraction'] == 'ContAuth', 'shopperinteraction'] = 1.0
    dfcs.loc[dfcs['shopperinteraction'] == 'POS', 'shopperinteraction'] = 2.0
    dfcs.loc[:, 'shopperinteraction'] = pd.to_numeric(dfcs.loc[:, 'shopperinteraction'])

    dfcs.loc[dfcs['simple_journal']=='Chargeback','simple_journal']=1.0
    dfcs.loc[dfcs['simple_journal']=='Settled','simple_journal']=0.0
    dfcs.loc[:,'simple_journal']=pd.to_numeric(dfcs.loc[:,'simple_journal'])



    dfcs.loc[:,'cvcresponsecode']=pd.to_numeric(dfcs.loc[:,'cvcresponsecode'])
    
    return dfcs
def roccrossvalid(dffeatures,labels,clfname):
    if clfname=='1nn':
        classifier = neighbors.KNeighborsClassifier(n_neighbors=1)
    elif clfname=='neural':
        classifier=MLPClassifier(hidden_layer_sizes=(300, ))
    elif clfname=='LR':
        classifier = linear_model.LinearRegression()
    else:
        print('input LR, 1nn or neural !')
    # Run classifier with cross-validation and plot ROC curves
    X = dffeatures.values
    y = labels.values
    # X, y = X[y != 2], y[y != 2]
    n_samples, n_features = X.shape

    cv = StratifiedKFold(n_splits=10)
    # classifier = neighbors.KNeighborsClassifier(n_neighbors=1)
    # classifier = MLPClassifier(hidden_layer_sizes=(300,))
    # classifier = svm.SVC(kernel='linear', probability=True,
    #                      random_state=random_state)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
    lw = 2

    i = 0
    for (train, test), color in zip(cv.split(X, y), colors):
        x_train, y_train = dosm(X[train], y[train])

        probas_ = classifier.fit(x_train, y_train).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=lw, color=color,
                 label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
             label='baseline')

    mean_tpr /= cv.get_n_splits(X, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='k', linestyle='-',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for '+ clfname)
    plt.legend(loc="lower right")
    plt.show()
