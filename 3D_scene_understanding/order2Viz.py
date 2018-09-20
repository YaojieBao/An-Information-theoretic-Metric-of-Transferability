#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 11:14:16 2018

@author: yajie

Given a target task, this file is for showing the heat-map of 2nd order transfer H-scores. 
"""

import os
import numpy as np
import matplotlib.pyplot as plt

t='depth'

file=os.listdir('../data/partQuanBirch/'+t+'/order2/') 

for f in file:
    if f.split('.')[1]=='png':
        continue
    else:
        score=np.load('../data/partQuanBirch/'+t+'/order2/'+f)
        
        score=score.reshape((16,16))
        
        plt.figure()
        plt.imshow(score)#,cmap ='gray')
        plt.savefig('../data/partQuanBirch/'+t+'/order2/'+f.split('.')[0]+'.png')
    
   
list_of_tasks = 'edge2d edge \
keypoint2d keypoint \
reshade depth \
points \
class_1000 class_places'

list_of_tasks = list_of_tasks.split()

os.mkdir('../data/partQuanBirch/'+t+'/order2_pixel/')

score_all=np.load('../data/partQuanBirch/'+t+'/score_sp4_256.npy')
for i in range(len(list_of_tasks)):
    
    score=score_all[i].reshape((16,16))
    plt.figure()
    plt.imshow(score)#,cmap ='gray')
    plt.savefig('../data/partQuanBirch/'+t+'/order2/'+list_of_tasks[i]+'.png')
    
    
'''
count=-1
for i,s in enumerate(list_of_tasks):
    
    if i==len(list_of_tasks)-1:
        break
    else:
        for j in np.linspace(i+1,len(list_of_tasks)-1, len(list_of_tasks)-i-1, dtype=np.int):  
            
            count+=1
            score_all=np.load('../data/partQuanBirch/'+t+'/score_sp4_256_order2.npy')
            score=score_all[count].reshape((16,16))
            
            plt.figure()
            plt.imshow(score)#,cmap ='gray')
            plt.savefig('../data/partQuanBirch/'+t+'/order2/'+s+'_'+list_of_tasks[j]+'_score_sp4_order2.png')
            
'''            

    

            
            
            
