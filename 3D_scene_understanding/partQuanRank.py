#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 18:53:26 2018

@author: yajie

This file is for ranking the source tasks according to the H-scores.
"""

import numpy as np

t='edge' # target task

score=np.load('../data/partQuanBirch/'+t+'/score_sp1_16.npy')
#score=np.load('../data/partQuanBirch/'+t+'/edge_mask_all/score_18.npy')

list_of_tasks = 'edge2d edge \
keypoint2d keypoint \
reshade depth \
points \
class_1000 class_places'

list_of_tasks = list_of_tasks.split()

# method 1: AHP

n=len(list_of_tasks)

win = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        win[i,j]=np.sum(np.ndarray.flatten(score[i,:])>=np.ndarray.flatten(score[j,:]))
        win[j,i]=win[i,j]
        
w,v=np.linalg.eigh(win.T)
rankValue=v[:,np.argmax(w)]
ranking=np.argsort(-np.abs(rankValue))
list_of_tasks=np.array(list_of_tasks)
sourceRank=list_of_tasks[ranking]
print('AHP:',sourceRank)

# method 2: mean 

rankValue=np.mean(score,axis=1)
ranking=np.argsort(-np.abs(rankValue))
list_of_tasks=np.array(list_of_tasks)
sourceRank=list_of_tasks[ranking]
print('mean:',sourceRank)

'''
# Plot heatmap of H-score of a T_s to the T_t
for i in range(len(list_of_tasks)):
    s=score[i,:].reshape((64,64))
    plt.figure()
    plt.imshow(s)
    plt.savefig('../data/partQuanBirch/'+t+'/viz/'+list_of_tasks[i]+'_sp1.png')
'''    
























