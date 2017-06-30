#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 23:03:06 2017

@author: clark
"""
from __future__ import division
import pandas as pd
import pylab as plt
import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import cdist, pdist
import collections
def elbow(data):  
   cluster_range = range( 2, 10)   
   cluster_errors = []
   for num_clusters in cluster_range:
       clusters = KMeans( num_clusters )
       clusters.fit( Sport )
       cluster_errors.append( clusters.inertia_ )
   clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )  
   plt.figure(figsize=(12,6))
   plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )

    
def Millisec(diff):
    return (diff.days * 86400000) + (diff.seconds * 1000) + (diff.microseconds / 1000)
#def freqDis(s,t):
#    for i in range(0,len(s)):
#        
#        if (str(int(s0[0]))+str(int(s1[0]))+str(int(s2[0]))) in 
#df = pd.read_csv('capture20110810.binetflow')
#df = pd.read_csv('capture20110811.binetflow')
#df = pd.read_csv('capture20110815-3.binetflow')
df = pd.read_csv('capture20110817.binetflow')

#df.head(5)
#df.shape
df2 = df[df['Label'].str.contains("Background")==False]
# delete the 
df3 = df2[df2['Label'].str.contains("To")==False]

df3.loc[df3['Label'].str.contains("Botnet"), 'biLabel'] = '1'
df3.loc[df3['Label'].str.contains("Normal"), 'biLabel'] = '0'

df3.drop('dTos',axis=1,inplace=True)#too many NaN
df3.drop('Label',axis=1,inplace=True)
#df3.drop('StartTime',axis=1,inplace=True)
# df3.drop('SrcAddr',axis=1,inplace=True)
df3.drop('DstAddr',axis=1,inplace=True)
df3.dropna(inplace=True)
df3[:1]

#for col in ['Proto', 'State','Dir','Sport','Dport','SrcAddr']:
#    df3[col] = df3[col].astype('category').cat.codes
for col in ['Proto', 'State','Dir','Sport','Dport']:
    df3[col] = df3[col].astype('category').cat.codes
#df_label = df3['biLabel']
#df3.drop('biLabel',axis=1,inplace=True)

Sport=np.reshape(df3['Sport'],[len(df3['Sport'].values),1])
#elbow(Sport)   #3 clusters
#
Dport=np.reshape(df3['Dport'],[len(df3['Dport'].values),1])
#elbow(Dport)# cluster=3
#
State=np.reshape(df3['State'],[len(df3['State'].values),1])
#elbow(State)#cluster=3
#
TotPkts=np.reshape(df3['TotPkts'],[len(df3['TotPkts'].values),1])
#elbow(TotPkts)#cluster=3
#
TotBytes=np.reshape(df3['TotBytes'],[len(df3['TotBytes'].values),1])
#elbow(TotBytes)#cluster=3
#
SrcBytes=np.reshape(df3['SrcBytes'],[len(df3['SrcBytes'].values),1])
#elbow(SrcBytes)#cluster=3


Sport3 = KMeans(n_clusters=3, random_state=0).fit(Sport)   
df3['SportLabel']=Sport3.labels_  
Dport3 = KMeans(n_clusters=3, random_state=0).fit(Dport)   
df3['DportLabel']=Dport3.labels_  
State3 = KMeans(n_clusters=3, random_state=0).fit(State)   
df3['StateLabel']=State3.labels_
TotPkts3 = KMeans(n_clusters=3, random_state=0).fit(TotPkts)   
df3['TotPktsLabel']=TotPkts3.labels_
TotBytes3 = KMeans(n_clusters=3, random_state=0).fit(TotBytes)   
df3['TotBytesLabel']=TotBytes3.labels_
SrcBytes3 = KMeans(n_clusters=3, random_state=0).fit(SrcBytes)   
df3['SrcBytesLabel']=SrcBytes3.labels_


df3['Time']=pd.to_datetime(df3['StartTime'])
#df4=df3[df3['SrcAddr']=='147.32.84.165']#botnet
#df4=df3[df3['SrcAddr']=='147.32.84.170']#normal
#df4=df3[df3['SrcAddr']=='147.32.84.192']#botnet
df4=df3[df3['SrcAddr']=='147.32.84.134']#normal

millisecond=np.zeros(len(df4))
for i in range(0,len(df4)):
    if i==0:
        millisecond[i]=0
    else:
        millisecond[i]=Millisec(df4['Time'].iloc[i]-df4['Time'].iloc[i-1])
#df3['millisecond']=millisecond
#Proto3  Dir4  SportLabel3 DportLabel3 StateLabel3 TotPktsLabel3 
#TotBytesLabel3 SrcBytesLabel3         8 features
code=np.zeros(len(df4))
space=3*4*3**6
features=['Proto','Dir','SportLabel','DportLabel','StateLabel','TotPktsLabel','TotBytesLabel','SrcBytesLabel']
lfeature=[3,4,2,3,3,3,3,3,3]
for i in range(0,len(df4)):
    code[i]=0
    for j in range(0,8) :
        code[i]=code[i]+int(df4[features[j]].iloc[i])*(space/lfeature[j])
        
#event=[code,millisecond]   
df_event=pd.DataFrame()
df_event['Code']=code
df_event['Time']=millisecond
print df_event.head(5)
threshold=40
countlist=[]
duration=0
count=0

for m in range(0,df_event.shape[0]-2):
    duration=0
    count=0
    for j in range(m+1,int(df_event.shape[0]-1)):
        duration=duration+df_event['Time'][j]
        if duration>threshold:
            break
        count=count+1
    if count > 1:
        for i in range(0, count-1):
            countlist.append((df_event['Code'][m+i],df_event['Code'][m+i+1],df_event['Code'][m+i+2]))
            
counts=collections.Counter(countlist)
state=np.zeros((10,3))
freq=np.zeros(10)
for i in range(0,10):
    state[i]=counts.most_common(10)[i][0]
    freq[i]=counts.most_common(10)[i][1]/len(countlist)

s0=np.zeros(10)
s1=np.zeros(10)
s2=np.zeros(10)
for i in range(0,10):
    s0[i]=state[i][0]  
    s1[i]=state[i][1] 
    s2[i]=state[i][2]  

df_model17134=pd.DataFrame() 
df_model17134['s0']=s0
df_model17134['s1']=s1
df_model17134['s2']=s2
df_model17134['freq']=freq
#147.32.84.165    40948   1 all 1 below
#147.32.84.192
#147.32.84.204
#147.32.84.191
#147.32.84.206
#147.32.84.170    18438   0 all zero below
#147.32.84.164     7654   
#147.32.84.134     3808   
#147.32.87.36       260
#147.32.80.9         83   
#147.32.86.187        4   
#147.32.87.252        3
#147.32.87.11         2
#147.32.86.140        1
#147.32.86.111        1
     