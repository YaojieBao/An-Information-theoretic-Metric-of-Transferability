#!/usr/bin/env python4
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 11:30:29 2018

@author: yajie

This file is for Calculate H-score for each pixel or superpixel. 
"""

#import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.preprocessing import normalize
#from sklearn.cluster import Birch, AgglomerativeClustering, AffinityPropagation, DBSCAN

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/4., 0.9*height, '%s' % round(float(height),2),rotation=90)


def getCov(X):
    X_mean=X-np.mean(X,axis=0,keepdims=True)
    cov = np.divide(np.dot(X_mean.T, X_mean), len(X)-1) 
    return cov

def getDiffNN(f,Z):
    #Z=np.argmax(Z, axis=1)
    Covf=getCov(f)
    alphabetZ=list(set(Z))
    g=np.zeros_like(f)
    for z in alphabetZ:
        Ef_z=np.mean(f[Z==z, :], axis=0)
        g[Z==z]=Ef_z
    
    Covg=getCov(g)
    dif=np.trace(np.dot(np.linalg.pinv(Covf,rcond=1e-15), Covg))
    return dif

def getDiffNNCov(f,inverse,Z):
    #Z=np.argmax(Z, axis=1)
    
    alphabetZ=list(set(Z))
    g=np.zeros_like(f)
    for z in alphabetZ:
        Ef_z=np.mean(f[Z==z, :], axis=0)
        g[Z==z]=Ef_z
    
    Covg=getCov(g)
    dif=np.trace(np.dot(inverse, Covg))
    return dif



import datetime

starttime = datetime.datetime.now()

#long running

t='keypoint2d' # T_t

list_of_tasks = 'edge2d edge \
keypoint2d keypoint \
reshade depth \
class_1000 class_places'

list_of_tasks = list_of_tasks.split()
#label=np.load('../data/partQuanBirch/'+t+'/labels.npy')
score=np.zeros((len(list_of_tasks),4096))
for i,s in enumerate(list_of_tasks):
    # load features of T_s
    fname='../data/'+s+'_fea.dat'
    print('source task is '+s)
    with open(fname, 'rb') as fr:
        fea=pickle.load(fr)
    
    #fea=np.random.rand(22045,2048)
        
    
    Covf=getCov(fea)
    inverse=np.linalg.pinv(Covf,rcond=1e-15)
    hscore=[]
    # Calculate H-score for each pixel
    for k in range(4096):
        labels=np.load('../data/partQuanBirch/'+t+'/sp1_16_bicubic_float/'+str(k)+'_labels_sp1.npy')
        #labels=label[:,k]
        #Hscore = getDiffNN(fea,labels)
        Hscore = getDiffNNCov(fea,inverse,labels)
        print('calculated:',k)
        hscore.append(Hscore)
    
    score[i,:]=np.array(hscore)
    np.save('../data/partQuanBirch/'+t+'/'+s+'_score_sp1_bicubic_float.npy',np.array(hscore))
    #np.save('../data/partQuanBirch/'+t+'/'+s+'_score_sp1.npy',np.array(hscore))
    print('saved:',s)
np.save('../data/partQuanBirch/'+t+'/score_sp1_16_bicubic_float.npy',score)
#np.save('../data/partQuanBirch/'+t+'/score_sp1_18.npy',score)                        

endtime = datetime.datetime.now()
print((endtime - starttime).seconds)










    
