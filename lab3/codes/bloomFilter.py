#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 17:11:18 2017

@author: clark
"""
    
from bitarray import bitarray
import mmh3
import numpy as np 
import pandas as pd
class BloomFilter:
    
    def __init__(self, size, hash_count):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = bitarray(size)
        self.bit_array.setall(0)
    def add(self, string):
        for seed in xrange(self.hash_count):
            result = mmh3.hash(string, seed) % self.size
            self.bit_array[result] = 1                
    def lookup(self, string):
        for seed in xrange(self.hash_count):
            result = mmh3.hash(string, seed) % self.size
            if self.bit_array[result] == 0:
                return "Nope"
        return "Probably"
# k is number of hash function, set 5
# df = pd.read_csv('./ftrain.txt')
# a=df['115.177.11.215'].unique()
# items number is 361423, make it 361500
#bit array is 1e4(fp 1.0000) 1e5(fp 0.9999) 1e6(0.4082) 5e6(0.0026)
bf = BloomFilter(1000000, 5)
 
lines = open("./src_ip.txt").read().splitlines()
ftrain=open('./ftrain.txt').read().splitlines()
ftest=open('./ftest.txt').read().splitlines()
sumline=3414011
halfline=np.round(sumline/2)#1707005

train=pd.read_csv('./ftrain.txt')
test=pd.read_csv('./ftest.txt')

truni=train['115.177.11.215'].unique()#361423
teuni=test['42.117.99.84'].unique()#251098

true=np.zeros(len(teuni))
for  i in range(0,len(teuni)):
    print ('i{}'.format(i))
    if (teuni[i] in truni):
        true[i]=1
    else :
        true[i]=0

for str in truni:    
    bf.add(str)
    
pred=np.zeros(len(teuni))  
for j in range(0,len(teuni)):
    print ('j{}'.format(j))
    if bf.lookup(teuni[j]) == 'Probably':
        pred[i]=1
    else:
        pred[i]=0        

for i in range(0,len(true)):
    if (true[i]==0 and pred[i]==1):
        fp=fp+1
    else:
        pass