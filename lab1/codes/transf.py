#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 18:50:30 2017

@author: chengchaoyang
"""
import pandas as pd 
import numpy as np
import scipy.io as sio




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
    df=df.loc[:,['amountEUR','bookingdate','simple_journal','cardverificationcodesupplied','cvcresponsecode']]
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
    
    dfcs.loc[dfcs['simple_journal']=='Chargeback','simple_journal']=1.0
    dfcs.loc[dfcs['simple_journal']=='Settled','simple_journal']=0.0
    dfcs.loc[:,'simple_journal']=pd.to_numeric(dfcs.loc[:,'simple_journal'])

    dfcs.loc[:,'cvcresponsecode']=pd.to_numeric(dfcs.loc[:,'cvcresponsecode'])
    
    


    return dfcs

