#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 12:47:50 2018

@author: yajie

This file is for partitioning the ground truth images into superpixels. 
"""

import numpy as np
import pickle
from sklearn.cluster import Birch#, AffinityPropagation
import os

t='keypoint2d' #T_t

indexMatrix=np.linspace(0,4095,4096,dtype=np.int).reshape((64,64))

# get the indices of superpixel
index=[]
for i in range(16):
    for j in range(16):
        idx=np.ndarray.flatten(indexMatrix[i*4:(i+1)*4,j*4:(j+1)*4])
        index.append(idx)

index=np.array(index)    

with open('../data/all_64_bilinear/'+t+'_all.dat','rb') as ff:
    data_all=pickle.load(ff)

if not os.path.exists('../data/partQuanBirch/'+t+'/sp4_256'):
    os.mkdir('../data/partQuanBirch/'+t+'/sp4_256')

#os.mkdir('../data/partQuanBirch/'+t+'/sp4_256')   
for i in range(256):
    data=np.array([x[index[i,:]] for x in data_all])

    brc = Birch(branching_factor=50, n_clusters=256, threshold=0.001, compute_labels=True)
    brc.fit(data)    
    labels=brc.predict(data)
    np.save('../data/partQuanBirch/'+t+'/sp4_256/'+str(i)+'_labels_sp4.npy',labels)
    
    print(str(i)+' has been done')
       
