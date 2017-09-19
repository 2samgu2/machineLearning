#!/usr/bin/env python 3.6
# -*- coding: utf-8 -*-
"""
Created on Saturdau Sep 16 16:58:58 2017

@author: Hans - Clément - Ali
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

plt.pcolor(genes[np.arange(100)])
plt.colorbar()
plt.title('Example of gene expressions')
plt.ylabel('samples')
plt.xlabel('genes')
plt.show()
