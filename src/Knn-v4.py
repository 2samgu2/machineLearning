#!/usr/bin/env python 3.6
# -*- coding: utf-8 -*-
"""
Created on Saturday Sep 16 16:58:58 2017

@author: Hans - Clément - Ali
"""
#----------Import_module----------------------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn import neighbors
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

try:
	import ConfigParser as conf

except:
	import configparser as conf

config = conf.ConfigParser()
config.readfp(open('../configuration.ini','r'))
xtrain= config.get('Data', 'xtrain')

genes = pd.read_table("../data/xtrain.txt", header=None)
labels = pd.read_table("../data/ytrain.txt", header=None)
ncol = genes.shape[1] 

X = genes[genes.columns[np.arange(1, ncol)]]
Y = np.array(labels).reshape(ncol-1)

# Ici il faut transposer X
X = X.T

## =================================================================================================
## knn

n = len(X[0])

# Keep only genes with high correlation with Y
rho = np.zeros(ncol)
for k in range(ncol) :
    c = np.corrcoef(X[k],Y)
    rho[k] = c[0,1]

w = np.nonzero(abs(rho)>=.2)[0]


Xw = X[w]
B = 100
n_neighbors = np.arange(1,7)
ErrClassif = np.zeros([len(n_neighbors),B])

for b in range(B): 
    itrain,itest=train_test_split(range(0,n-1),test_size=0.25)
    Xtrain = Xw.iloc[itrain]
    ytrain = Y[np.asarray(itrain)] # because itrain is a list
                          # and y is indexed from 6 to ... 
    ytest = Y[np.asarray(itest)] # because itest is a list
    
    for i in n_neighbors:
        clf = neighbors.KNeighborsClassifier(i)
        clf.fit(Xtrain, ytrain)
        yhat = clf.predict(Xw.iloc[itest])
        ErrClassif[i-1,b] = np.mean(ytest!=yhat)

plt.boxplot(ErrClassif.T,labels=n_neighbors)
plt.ylim(-0.1,1)
plt.ylabel('Mean classification error')
plt.xlabel('nb of neighbors')
plt.show()


