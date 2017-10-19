#!/usr/bin/env python 3.6
# -*- coding: utf-8 -*-
"""
Created on Saturdau Sep 16 16:58:58 2017

@author: Hans - Clément - Ali
"""
#----------Import_module----------------------------------------------------------------------------


import numpy as np
import pandas as pd
from sklearn import neighbors
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

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

#===================================================================================================

def ROC(y_test,y_score,methodName=" ",plot=True):

    ntest = np.size(y_test,0)
    B = np.size(y_test,1)
    fpr, tpr, _ = roc_curve(np.reshape(y_test,B*ntest), np.reshape(y_score,B*ntest))
    roc_auc = auc(fpr, tpr)

    if plot:
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(methodName)
        plt.legend(loc="lower right")
        plt.show()
        
    return(roc_auc)

    
## =================================================================================================
## Gradient boosting

# Keep only genes with high correlation with Y

rho = np.zeros(ncol)
for k in range(ncol) :
	c = np.corrcoef(X.iloc[k],Y)
	rho[k] = c[0,1]

# if no preselection of gene is done
n = len(X.iloc[0])
B = 100
test_size = .1
ntest = 19#round(.1*n) # size of test sample
y_score_GB = np.zeros([ntest,B]) 
y_test = np.zeros([ntest,B]).astype(int)

for b in range(B): 
    itrain,itest=train_test_split(range(0,n-1),test_size=test_size)
    Xtrain = X.iloc[itrain]
    ytrain = Y[np.asarray(itrain)]
    w = np.nonzero(abs(rho)>.5)[0]
    ytest = Y[np.asarray(itest)] # because itest is a list
    y_test[:,b] = ytest
    clf = GradientBoostingClassifier(loss='exponential',n_estimators=5000,max_features=30)
    clf.fit(Xtrain, ytrain)
    y_score_GB[:,b] = clf.predict_proba(X.iloc[itest])[:,1]

ROC(y_test,y_score_GB,"GB")

# with preselection
n = len(X)
B = 50
test_size = .1
ntest = round(.1*n) # size of test sample

y_score_GB = np.zeros([ntest,B]) 
y_test = np.zeros([ntest,B]) 

for b in range(B): 
    itrain,itest=train_test_split(range(0,n-1),test_size=test_size)
    Xtrain = X.iloc[itrain]
    ytrain = Y[np.asarray(itrain)]
    rho = np.zeros(ncol) # correlated gene should be selected here
                  # and not outside the cross-validation loop
    for k in range(ncol) : 
        c = np.corrcoef(Xtrain[k],ytrain)
        rho[k] = c[0,1]
    w = np.nonzero(abs(rho)>.15)[0]
    ytest = Y[np.asarray(itest)] # because itest is a list
    y_test[:,b] = ytest
    clf = GradientBoostingClassifier(loss='exponential',n_estimators=200,max_features=15)
    clf.fit(Xtrain[w], ytrain)
    y_score_GB[:,b] = clf.predict_proba(X[w].iloc[itest])[:,1]

ROC(y_test,y_score_GB,"GB")






