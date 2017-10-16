#!/usr/bin/env python 3.6
# -*- coding: utf-8 -*-
"""
Created on Saturdau Sep 16 16:58:58 2017

@author: Hans - Clément - Ali
"""
#----------Import_module-----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn import neighbors

try:
	import ConfigParser as conf

except:
	import configparser as conf

# KNN =+> on va voir quel sont les individus qui se ressemble et on va prendre des décision


#-------------------Fontion------------------------------------
def plot(genes) :
    plt.pcolor(genes[np.arange(100)])
    plt.colorbar()
    plt.title('Example of gene expressions')
    plt.ylabel('samples')
    plt.xlabel('genes')
    plt.show()

def geneHightCorrelation(G,Y) :
	ncol = G.shape[1]
	rho = np.zeros(ncol)
	for k in range (ncol) :
		#print (len(G[1:,k]))
		#print (len(Y))
		#print (G[1:,k], Y)

		c = np.corrcoef(G[1:,k].astype(float), Y.astype(float))
		rho[k] = c [0,1]
	#print (rho)
	w = np.nonzero(abs(rho)>.1)[0] # On sélecionne uniquement les genes qui ont un
							# coefficient de corrélation > 0.1
	#print (len(w))
	#print (w)
	return (rho, w)

def knn (G,Y) :
	w = geneHightCorrelation(G,Y)[1]
	n = len (X[0]) # Nbre d'échantillon
	Xw = X[w] # Recupère les valeurs d'expression des gènes avec un coeff > 0.1
	#print (n)
	Xw = Xw[1:]


	b=100
	n_neighbors = np.arange(1,7)
	ErrClassif = np.zeros([len(n_neighbors),b])
	#print (ErrClassif)


	for i in range (b) :
		itrain, itest = train_test_split(range(0, n-1), test_size = 0.25)
		Xtrain = Xw.iloc[itrain]
		ytrain = Y[np.asarray(itrain)] # because itrain is a list
	                          # and y is indexed from 6 to ...
		ytest = Y[np.asarray(itest)] # because itest is a list

		for j in n_neighbors:
			clf = neighbors.KNeighborsClassifier(j)
			clf.fit(Xtrain, ytrain)
			yhat = clf.predict(Xw.iloc[itest])
			#print (yhat)


			ErrClassif[j-1,99] = np.mean(ytest!=yhat)
			#print (ErrClassif)
	return (ErrClassif, n_neighbors)



"""
# Best result for 1 neighbor
ibest = 1
ntest = 10 # 10 because len(itest) = 10
y_score = np.zeros([ntest,B]) # 10 because len(itest) = 10
y_test = np.zeros([ntest,B]) # 10 because len(itest) = 10

for b in range(B):
    itrain,itest=train_test_split(range(0,n-1),test_size=0.25)
    Xtrain = Xw.iloc[itrain]
    ytrain = Y[np.asarray(itrain)] # because itrain is a list
                          # and y is indexed from 6 to ...
    ytest = Y[np.asarray(itest)] # because itest is a list
    y_test[:,b] = ytest
    clf = neighbors.KNeighborsClassifier(ibest)
    clf.fit(Xtrain, ytrain)
    y_score[:,b] = clf.predict_proba(Xw.iloc[itest])[:,1]



ROC(y_test,y_score,"kNN, 1 neighbor")

"""





#----------------Menu Principale----------------------------------

config = conf.ConfigParser()
config.readfp(open('../configuration.ini','r'))
xtrain= config.get('Data', 'xtrain')
path_xtrain = str(xtrain)

gene = pd.read_table("../data/xtrain.txt", header=None)
labels = pd.read_table("../data/ytrain.txt", header=None)
ncol = gene.shape[1]

X = gene.T

Y = np.array(labels).reshape(184)
G = np.array(X)

geneHightCorrelation(G,Y)

ErrClassif , n_neighbors = knn (G,Y)

#plt.boxplot(ErrClassif.T,labels=n_neighbors)
plt.plot(ErrClassif.T)
plt.ylim(0,1)
plt.ylabel('Mean classification error')
plt.xlabel('nb of neighbors')
#plt.plot(rho)
plt.show()
