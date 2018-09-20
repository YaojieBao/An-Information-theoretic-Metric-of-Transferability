#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 19:52:40 2018

@author: yajie
"""

import numpy as np
import pickle
   
'''
list_of_tasks = 'autoencoder curvature denoise edge2d edge3d \
keypoint2d keypoint3d colorization \
reshade rgb2depth rgb2mist rgb2sfnorm \
room_layout segment25d segment2d vanishing_point_well_defined \
segmentsemantic class_1000 class_places inpainting_whole'
'''
list_of_tasks = 'edge2d edge3d \
keypoint2d keypoint3d \
reshade rgb2depth \
vanishing_point_well_defined \
class_1000 class_places'

list_of_tasks = list_of_tasks.split()
n=len(list_of_tasks)

target_tasks = 'edge2d edge3d \
keypoint2d keypoint3d \
reshade rgb2depth \
vanishing_point_well_defined \
class_1000 class_places'

target_tasks = target_tasks.split()


with open('../data/all_affinities_16k.pkl', 'rb') as f:
    affinities = pickle.load(f)

aff=np.zeros((len(target_tasks),n))
for i,t in enumerate(target_tasks):
    for j,k in enumerate(list_of_tasks):
        if t+'__'+k in affinities:
            aff[i,j]=affinities[t+'__'+k]
        else:
            continue
            
np.save('../data/groundtruth/AffinitiesPart9x9.npy',aff)
import matplotlib.pyplot as plt
plt.figure()
index=np.arange(len(list_of_tasks))
plt.xticks(index,list_of_tasks,rotation=90)
idx=range((len(target_tasks)))
plt.yticks(idx,target_tasks)
plt.imshow(aff)
plt.savefig('../data/groundtruth/AffinitiesPart9x9.png')        

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/4., 0.9*height, '%s' % round(float(height),4),rotation=90)
        
for i in range(len(target_tasks)):        
    plt.figure()
    rects=plt.bar(index,aff[i,:],fill=False,edgecolor='b')
    plt.xticks(index,list_of_tasks,rotation=90)
    plt.ylabel('AHP')
    plt.title('transfer to '+target_tasks[i]+' from the other tasks')     
    autolabel(rects)
    plt.savefig('../data/groundtruth/to_'+target_tasks[i]+'_src9.png')
    plt.show() 


    
''' 
with open('../data/win_rates_16k.pkl', 'rb') as f:
    win = pickle.load(f)
    
win_rates=win['win_rates'] 
w=np.zeros((len(target_tasks),n))     
for i,t in enumerate(target_tasks):
    for j,k in enumerate(list_of_tasks):
        Task=win_rates[t]
        w[i,j]=Task[(k,)][(t,)]
    
    
import matplotlib.pyplot as plt
plt.figure()
index=np.arange(len(list_of_tasks))
plt.xticks(index,list_of_tasks,rotation=90)
idx=range((len(target_tasks)))
plt.yticks(idx,target_tasks)
plt.imshow(w)        

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/4., 0.9*height, '%s' % round(float(height),3),rotation=90)
        
for i,t in enumerate(target_tasks):        
    plt.figure()
    rects=plt.bar(index,w[i,:],fill=False,edgecolor='b')
    plt.xticks(index,list_of_tasks,rotation=90)
    plt.ylabel('win_rates')
    plt.title('transfer to '+t+' from the other tasks')     
    autolabel(rects)
    plt.savefig('../data/win_rates/to_'+t+'.png')
    plt.show()     
  
    
with open('../data/avg_losses_16k.pkl', 'rb') as f:
    loss = pickle.load(f)    
'''

    
    
    
    
    
    
    
    
    
    
    
    
    
    