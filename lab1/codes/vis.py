#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 10:51:13 2017

@author: chengchaoyang
"""
import pandas as pd 
import numpy as np
import seaborn as sns;sns.set()
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.style.use('ggplot')
from transf import transf,perday
src='/Users/chengchaoyang/GoogleCloud/Class/1cyber/assignment1/data_for_student_case.csv'
#data=pd.read_csv(src)
#data=data[(data['simple_journal']=='Chargeback') | (df['simple_journal']=='Settled')]
#df=perday(data,'bookingdate')
#data = data[data['issuercountrycode'] != 'NA']
#data=data[data['issuercountrycode']!='NA']
#x=data[data['simple_journal']=='Chargeback']['issuercountrycode'].value_counts()
#y=data[data['simple_journal']=='Settled']['issuercountrycode'].value_counts()
#for i in range(len(y)):   
#    data.loc[data['issuercountrycode']==y.index[i],'issuercountrycode']=i*1.0
#data.fillna(int(data['issuercountrycode'].mean()),inplace=True)
#data.loc[:, 'issuercountrycode'] = pd.to_numeric(data.loc[:, 'issuercountrycode'])
#dfcs.loc[dfcs['simple_journal']=='Settled','simple_journal']=0.0
#dfcs.loc[:,'simple_journal']=pd.to_numeric(dfcs.loc[:,'simple_journal'])
#df = pd.Series(np.array([0.1,0.2,0.3,0.4]), index=['a', 'b', 'c', 'd']) 
#plt.figure()
#df.plot.pie(labels=['AA', 'BB', 'CC', 'DD'], colors=['r', 'g', 'b', 'c'],autopct='%.2f', fontsize=20, figsize=(6, 6))
#plt.title('xxxx')  

#df=pd.read_csv(src)
#df=transf(df)

#=========================pie chart=====================================================

#df=df[df['cardverificationcodesupplied']==True]

#cb=df[df['simple_journal']=='Chargeback']
#cb = cb.rename(columns={'cvcresponsecode': 'cvc'})
#cb_dic=cb['cvc'].value_counts()
#cb_dic.plot.pie(labels=['unSupplied','Supplied'],subplots=True,figsize=(4,4),autopct='%.2f')

#st=df[df['simple_journal']=='Settled']
#st = st.rename(columns={'cvcresponsecode': 'cvc'})
#st_dic=st['cvc'].value_counts()
#st_dic.plot.pie(labels=['Supplied','unSupplied','others',' ',' '],subplots=True,figsize=(4,4),autopct='%.2f')
#plt.title('Proportion of Verification Code supplied in Settle')

#==============================================================================

#====================percent bar==========================================================
#df=pd.read_csv(src)
#num_cb=df[df['simple_journal']=='Chargeback']['issuercountrycode'].value_counts()
#per_country={}.fromkeys(num_cb.index)
# #print (country.keys())
#for c in num_cb.index:
#    per_country[c]=num_cb[c]/df[df['issuercountrycode']==c]['issuercountrycode'].count()
#print (per_country)
#sns.barplot(list(per_country.keys()),list(per_country.values()))
#==============================================================================

#=============settled heat=================================================================
  #df['bookingdate']=pd.to_datetime(df['bookingdate'])
  #print (df.ix[1,'bookingdate'].date())
#df=pd.read_csv(src)
#currencyconvert = {"AUD":0.680237,"MXN":0.0487438,"NZD":0.630313,"GBP":1.18285,"SEK":0.103558}
#amountEUR = []
#for i in df.index:
#        if df['currencycode'][i] == "MXN":
#            amountEUR.append(currencyconvert["MXN"]*df['amount'][i])
##         print(datacharge.loc[i,['currencycode']])
#        if df['currencycode'][i] == "AUD":
#            amountEUR.append(currencyconvert["AUD"]*df['amount'][i])
#
#        if df['currencycode'][i] == "NZD":
#            amountEUR.append(currencyconvert["NZD"]*df['amount'][i])
#
#        if df['currencycode'][i] == "GBP":
#            amountEUR.append(currencyconvert["GBP"]*df['amount'][i])
#
#        if df['currencycode'][i] == "SEK":
#            amountEUR.append(currencyconvert["SEK"]*df['amount'][i])
##amountEUR
#df['amountEUR']=amountEUR
#df=df.loc[:,['creationdate','simple_journal','amountEUR']]
#df=perday(df,'creationdate')
#datacharge=df[df['simple_journal']=='Settled']
#  #result=datacharge.sort([''])
#a=datacharge.groupby('creationdate')['simple_journal'].size()
#a=pd.DataFrame(a,columns=['SettledSize'])
#b=datacharge.groupby('creationdate')['amountEUR'].mean()
#  #b=pd.DataFrame(b,columns=['amountMean'])#fail with getting NaN
#b=pd.DataFrame(b)
#  #c=pd.concat(a,b) #fail with getting NaN
#c=a.join(b,how='outer')
#c['creationdate'] = c.index
##p=pd.pivot_table(c,index=['creationdate'],columns=['SettledSize'],values=['amountEUR'])
#p=c.pivot('creationdate','SettledSize','amountEUR')
#ax=sns.heatmap(p,xticklabels=10,yticklabels=10)
#==============================================================================

#=================chargeback heat=============================================================
df=pd.read_csv(src)
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
  #df['bookingdate']=pd.to_datetime(df['bookingdate'])
  #print (df.ix[1,'bookingdate'].date())
df=df.loc[:,['creationdate','simple_journal','amountEUR']]
df=perday(df,'creationdate')
datacharge=df[df['simple_journal']=='Chargeback']
  #result=datacharge.sort([''])
a=datacharge.groupby('creationdate')['simple_journal'].size()
a=pd.DataFrame(a,columns=['chargebackSize'])
b=datacharge.groupby('creationdate')['amountEUR'].mean()
  #b=pd.DataFrame(b,columns=['amountMean'])#fail with getting NaN
b=pd.DataFrame(b)
  #c=pd.concat(a,b) #fail with getting NaN
c=a.join(b,how='outer')
c['creationdate'] = c.index
#p=pd.pivot_table(c,index=['creationdate'],columns=['chargebackSize'],values=['amountEUR'])
p=c.pivot('creationdate','chargebackSize','amountEUR')
ax=sns.heatmap(p,yticklabels=10)
#==============================================================================

