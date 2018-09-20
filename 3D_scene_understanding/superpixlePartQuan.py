#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 12:47:50 2018

@author: yajie

This file is for quantizing each pixel into 16 clusters.
"""

import numpy as np
import pickle
from sklearn.cluster import Birch, AffinityPropagation
import os

t='keypoint'

with open('../data/'+t+'_all_bicubic_float.dat','rb') as ff:
    data_all=pickle.load(ff)

if not os.path.exists('../data/partQuanBirch/'+t+'/sp1_16_bicubic_float_c5'):
    os.mkdir('../data/partQuanBirch/'+t+'/sp1_16_bicubic_float_c5')

#os.mkdir('../data/partQuanBirch/'+t+'/SuperPixle_1')   
for i in range(4096):
    gd=data_all[:,i]

    data=gd.reshape(-1, 1)

    brc = Birch(branching_factor=50, n_clusters=5, threshold=0.01, compute_labels=True)
    brc.fit(data)    
    labels=brc.predict(data)
    np.save('../data/partQuanBirch/'+t+'/sp1_16_bicubic_float_c5/'+str(i)+'_labels_sp1.npy',labels)
    
    print(str(i)+' has been done')
       
