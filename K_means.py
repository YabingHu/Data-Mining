# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 13:13:12 2018

@author: yabinghu
"""
#%% 

#from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from random import *
from copy import deepcopy

df = pd.read_csv('yelp.csv')
names=list(df)
f1=df[names[2]].values#review counts
f2=df[names[4:7]].values.sum(axis=1)#useful funny cool
f3=df[names[7]].values#fans
f4=df[names[10:]].values.sum(axis=1)#comliment_hot and all other comliments
f5=df[names[8]].values#elite
for i,item in enumerate(f5):
    if item=='None':
        f5[i] = 0
    else:
       f5[i] = item.count(',')+1
X = np.array(list(zip(f1, f2,f3,f4,f5)))
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)

del f1,f2,f3,f4,f5


#%% k_means and k_means_++
def distance(X,C):
    error=[]
    for i in range(len(X)):
        distances=np.sqrt(np.square(X[i] - C).sum(axis=1))
        error.append(min(distances))
    avg_dist=sum(error)/len(error)
    min_dist=(min(error))
    max_dist=(max(error))
    return avg_dist, min_dist,max_dist


def online_k_means(X,k):
    C=X[np.random.randint(0, len(X), size=k)]
    m, n = X.shape  
    diff=1
    i=1
    count = np.zeros(k)
    while  diff !=0 and i <1000:
        C_prev=deepcopy(C)
        I=np.random.choice(m, 1000, replace=False)
        for t in I:
            distances=np.sqrt(np.square(X[t] - C).sum(axis=1))
            cluster = np.argmin(distances)
            count[cluster] += 1
            eta=1/count[cluster]
            C[cluster] = C[cluster]+ eta*(X[t]-C[cluster])
        diff=sum(np.sqrt(np.square(C_prev - C).sum(axis=1)))/k
        if diff< 0.0001:
            diff=0
        i=i+1
    return C

k=[5,50,100,250,500]
plot_data=[]
for i in k:
    C_km=online_k_means(X,i)
    avg_dist, min_dist,max_dist=distance(X, C_km)
    plot_data.append(avg_dist)
    plot_data.append(min_dist)
    plot_data.append(max_dist)

 
def k_means_pp(X,k):
    m, n = X.shape  
    C=X[np.random.randint(0, len(X), size=1)]
    Dist=np.array(np.square(X - C).sum(axis=1))
    p_x=Dist/sum(Dist)
    for i in range(1,k):
        ind=np.random.choice(np.arange(0, m), p=p_x)
        C=np.vstack([C, X[ind]])
        D=np.array(np.square(X - C[i]).sum(axis=1))
        Dist=np.vstack([Dist,D])# history distance
        D=Dist.min(axis=0)
        p_x=D/sum(D)

    diff=1
    i=1
    count = np.zeros(k)
    while  diff !=0 and i <1000:
        C_prev=deepcopy(C)
        I=np.random.choice(m, 1000, replace=False)
        for t in I:
            distances=np.sqrt(np.square(X[t] - C).sum(axis=1))
            cluster = np.argmin(distances)
            count[cluster] += 1
            eta=1/count[cluster]
            C[cluster] = C[cluster]+ eta*(X[t]-C[cluster])
        diff=sum(np.sqrt(np.square(C_prev - C).sum(axis=1)))/k
        if diff< 0.0001:
            diff=0
        i=i+1
    return C
    
plot_data_kmpp=[]
for i in k:
    C_kmpp=k_means_pp(X,i)
    avg_dist, min_dist,max_dist=distance(X, C_kmpp)
    plot_data_kmpp.append(avg_dist)
    plot_data_kmpp.append(min_dist)
    plot_data_kmpp.append(max_dist)
  


    