#!/usr/bin/env python4
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 11:30:29 2018

@author: yajie

Given a target task, this file is for computing H-scores for 2nd order transfer tasks.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.cluster import Birch, AgglomerativeClustering, AffinityPropagation, DBSCAN


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


t='depth'

list_of_tasks = 'edge2d edge \
keypoint2d keypoint \
reshade depth \
points \
class_1000 class_places'

list_of_tasks = list_of_tasks.split()

score=[]
for i,s in enumerate(list_of_tasks):
    
    fname='../data/fea/'+s+'_fea.dat'
    print('source task 1 is '+s)
    with open(fname, 'rb') as fr:
        fea1=pickle.load(fr)

    if i==len(list_of_tasks)-1:
        break
    else:
        for j in np.linspace(i+1,len(list_of_tasks)-1, len(list_of_tasks)-i-1, dtype=np.int):  
            
            fname='../data/fea/'+list_of_tasks[j]+'_fea.dat'
            print('source task 2 is '+list_of_tasks[j])
            with open(fname, 'rb') as frr:
                fea2=pickle.load(frr)
    
            fea=np.concatenate((fea1,fea2),axis=1)
            Covf=getCov(fea)
            inverse=np.linalg.pinv(Covf,rcond=1e-15)
            hscore=[]
            for k in range(256):
                labels=np.load('../data/partQuanBirch/'+t+'/sp4_256/'+str(k)+'_labels_sp4.npy')
        
                #Hscore = getDiffNN(fea,labels)
                Hscore = getDiffNNCov(fea,inverse,labels)
                print('calculated:',k)
                hscore.append(Hscore)
    
            score.append(np.array(hscore))
            np.save('../data/partQuanBirch/'+t+'/order2/'+s+'_'+list_of_tasks[j]+'_score_sp4_order2.npy',np.array(hscore))
            print('saved:',s)
            
np.save('../data/partQuanBirch/'+t+'/score_sp4_256_order2.npy',np.array(score))            
'''
score=[]
for i,s in enumerate(list_of_tasks):
    
    fname='../data/fea/'+s+'_fea.dat'
    print('source task 1 is '+s)
    with open(fname, 'rb') as fr:
        fea1=pickle.load(fr)

    if i==len(list_of_tasks)-1:
        break
    else:
        for j in np.linspace(i+1,len(list_of_tasks)-1, len(list_of_tasks)-i-1, dtype=np.int):  
            
            fname='../data/fea/'+list_of_tasks[j]+'_fea.dat'
            print('source task 2 is '+list_of_tasks[j])
            with open(fname, 'rb') as frr:
                fea2=pickle.load(frr)
    
            fea=np.concatenate((fea1,fea2),axis=1)

            with open('../data/labels_81/'+t+'_labels_81.dat', 'rb') as ff:
                labels=pickle.load(ff)
    
            Hscore = getDiffNN(fea,labels)
            print('calculated:',j)
            
    
            score.append(Hscore)
            
            
            
np.save('../data/'+t+'_score_order2.npy',np.array(score)) 
'''








    
