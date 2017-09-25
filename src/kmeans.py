#!/usr/bin/env python 3.6
# -*- coding: utf-8 -*-
"""
@author: Hans - Clément - Ali
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

try:
	import ConfigParser as conf
	
except:
	import configparser as conf

config = conf.ConfigParser()
config.readfp(open('../configuration.ini','r'))
xtrain= config.get('Data', 'xtrain')
path_xtrain = str(xtrain)

gene = pd.read_table("../data/xtrain.txt", header=None)

ncol = gene.shape[1]
gene = gene[gene.columns[np.arange(1, ncol)]]


#print(gene.ix[:, 0:1]) # Select colonne
#print(gene[0])
#print(gene.columns[2])
#print(gene.columns[np.arange(ncol-1)])
#print(gene.loc[[1],:]) #get line


from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(gene, 'ward')
# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
