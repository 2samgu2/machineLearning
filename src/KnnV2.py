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


def geneHightCorrelation(gene, ncol, labels) :
    #print (len(ncol))
    L=len (labels)
    #rho = np.zeros(ncol)
    #print (rho)
    for k in range(ncol):

        c = np.corrcoef(np.reshape(genes[k],[L,1]),np.reshape(labels,[L,1]))
        print (c)
        break

        #print (c)
        #rho[k] = c[0, 1]
        #print (rho[k])
    #w = np.nonzero(abs(rho)>0.5)[0]
    #print (len(w))


#----------------Menu Principale----------------------------------

config = conf.ConfigParser()
config.readfp(open('../configuration.ini','r'))
xtrain= config.get('Data', 'xtrain')
path_xtrain = str(xtrain)

gene = pd.read_table("../data/xtrain.txt", header=None)
labels = pd.read_table("../data/ytrain.txt", header=None)
ncol = gene.shape[1]

#print (ncol)
geneT = gene[gene.columns[np.arange(1, ncol)]]
#print(geneT)
genes = geneT.T
#print (genes)

ncol = genes.shape[1]
nrow = genes.shape[0]
#print (ncol) # Nombre d'échantillon
#print(nrow) # Nombre de genes

#print (type(labels))
#print (ncol)
#print (len (labels))
#plot(genes)
#knn(genes)

geneHightCorrelation(genes, ncol, labels)
