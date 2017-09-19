#!/usr/bin/env python 3.6
# -*- coding: utf-8 -*-
"""
Created on Saturdau Sep 16 16:58:58 2017

@author: Hans - Clément - Ali
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

try:
	import ConfigParser as conf
	
except:
	import configparser as conf

config = conf.ConfigParser()
config.readfp(open('../configuration.ini','r'))
xtrain= config.get('Data', 'xtrain')
path_xtrain = str(xtrain)

gene = pd.read_table("../data/xtrain.txt")
ncol = gene.shape[1]
geneT = gene[gene.columns[np.arange(1, ncol)]]
genes = geneT.T

ncol = genes.shape[1]
nrow = genes.shape[0]
pca = PCA (n_components=2)
pca.fit(genes)
plt.plot(pca.explained_variance_ratio_)
plt.ylabel("Explained variance")
plt.xlabel("Number of components")
#plt.show()

   
pca2 = PCA (n_components=2) # conserver les deux premiers composants principaux
pca2.fit(genes) # adapter le modèle

# transformer les données sur les deux premiers composants principaux
x_pca = pca2.transform(genes)
print ("Original shape : %s" % str((genes).shape))
print ("Reduced shape : %s" % str((x_pca).shape))

# plot fist vs second principal component, color by class

plt.figure(figsize=(10, 10))
plt.scatter(x_pca[:, 0], x_pca[:, 1])
plt.gca().set_aspect("equal")
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
plt.show()