"""
This file is to extract features of trained encoders in Zamir et al. (2018). 
"""
from __future__ import absolute_import, division, print_function

import argparse
import importlib
import itertools
import matplotlib
matplotlib.use('Agg')
import time
from   multiprocessing import Pool
import numpy as np
import os
import pdb
import pickle
import subprocess
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
import threading
import scipy.misc
from skimage import color
import init_paths
from models.sample_models import *
from lib.data.synset import *
import scipy
import skimage
import skimage.io
import transforms3d
import math
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from task_viz import *
import random
import utils
import models.architectures as architectures
from   data.load_ops import resize_rescale_image
from   data.load_ops import rescale_image
import lib.data.load_ops as load_ops
import pickle

tf.logging.set_verbosity(tf.logging.ERROR)

list_of_tasks = 'autoencoder curvature denoise edge2d edge3d \
keypoint2d keypoint3d colorization jigsaw \
reshade rgb2depth rgb2mist rgb2sfnorm \
room_layout segment25d segment2d vanishing_point \
segmentsemantic class_1000 class_places inpainting_whole'
list_of_tasks = list_of_tasks.split()

def generate_cfg(task):
    repo_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    CONFIG_DIR = os.path.join(repo_dir, 'experiments/final', task)
    ############## Load Configs ##############
    import utils
    import data.load_ops as load_ops
    from   general_utils import RuntimeDeterminedEnviromentVars
    cfg = utils.load_config( CONFIG_DIR, nopause=True )
    RuntimeDeterminedEnviromentVars.register_dict( cfg )
    cfg['batch_size'] = 1
    if 'batch_size' in cfg['encoder_kwargs']:
        cfg['encoder_kwargs']['batch_size'] = 1
    cfg['model_path'] = os.path.join( repo_dir, 'temp', task, 'model.permanent-ckpt' )
    cfg['root_dir'] = repo_dir
    return cfg

'''
fileset=os.listdir('../val/') 
for f in fileset:
    file=os.listdir('../val/'+f) 
    if 'features' in file:
        continue
    elif 'softmax_1000' in file:
        os.mkdir(file+'/features/')   
'''        

import general_utils
from   general_utils import RuntimeDeterminedEnviromentVars

fileset=os.listdir('../val/') 
for f in fileset:
    file=os.listdir('../val/'+f) 
    if 'features' in file: 
        print(f+'is processing')
        for k in [3,4,5,6,9,10,18,19]: #range(len(list_of_tasks)):
            taskNo=k
            imgPath='../val/'+f+'/rgb/'
            storePath='../val/'+f+'/features/'
            task=list_of_tasks[taskNo]
            # Read raw images
            #predict=[]
            feature=[]
            imgset=os.listdir(imgPath)
            count=0
            for im_name in imgset:
            
                img = load_raw_image_center_crop( imgPath+im_name )
                img = skimage.img_as_float(img)
                #scipy.misc.toimage(np.squeeze(img), cmin=0.0, cmax=1.0).save(im_name)
                
                tf.logging.set_verbosity(tf.logging.ERROR)
                       
                task = list_of_tasks[taskNo]
                if task not in list_of_tasks:
                    raise ValueError('Task not supported')
            
                cfg = generate_cfg(task)
            
                # Since we observe that areas with pixel values closes to either 0 or 1 sometimes overflows, we clip pixels value
                low_sat_tasks = 'autoencoder curvature denoise edge2d edge3d \
                keypoint2d keypoint3d \
                reshade rgb2depth rgb2mist rgb2sfnorm \
                segment25d segment2d'.split()
                if task in low_sat_tasks:
                    cfg['input_preprocessing_fn'] = load_ops.resize_rescale_image_low_sat
            
                if task == 'jigsaw' :
                    img = cfg[ 'input_preprocessing_fn' ]( img, target=cfg['target_dict'][random.randint(0,99)], 
                                                            **cfg['input_preprocessing_fn_kwargs'] )
                else:
                    img = cfg[ 'input_preprocessing_fn' ]( img, **cfg['input_preprocessing_fn_kwargs'] )
            
                img = img[np.newaxis,:]
            
                if task == 'class_places' or task == 'class_1000':
                    synset = get_synset(task)
            
                print("Doing {task}".format(task=task))
                general_utils = importlib.reload(general_utils)
                tf.reset_default_graph()
                training_runners = { 'sess': tf.InteractiveSession(), 'coord': tf.train.Coordinator() }
            
                ############## Set Up Inputs ##############
                # tf.logging.set_verbosity( tf.logging.INFO )
                setup_input_fn = utils.setup_input
                inputs = setup_input_fn( cfg, is_training=False, use_filename_queue=False )
                RuntimeDeterminedEnviromentVars.load_dynamic_variables( inputs, cfg )
                RuntimeDeterminedEnviromentVars.populate_registered_variables()
                start_time = time.time()
            
                ############## Set Up Model ##############
                model = utils.setup_model( inputs, cfg, is_training=False )
                m = model[ 'model' ]
                model[ 'saver_op' ].restore( training_runners[ 'sess' ], cfg[ 'model_path' ] )
                '''
                # encoder (extract features)
                predicted, representation = training_runners['sess'].run( 
                        [ m.decoder_output,  m.encoder_output ], feed_dict={m.input_images: img} )
                
                '''
                representation = training_runners['sess'].run(m.encoder_output, feed_dict={m.input_images: img} )
                '''
                if task == 'segment2d' or task == 'segment25d':
                    segmentation_pca(predicted, storePath+task+im_name.split('.')[0]+'.png')
            
                if task == 'colorization':
                    single_img_colorize(predicted, img , storePath+task+im_name.split('.')[0]+'.png')
            
                
                if task == 'curvature':
                    curvature_single_image(predicted, storePath+task+im_name.split('.')[0]+'.png')
            
            
                just_rescale = ['autoencoder', 'denoise', 'edge2d', 
                                'edge3d', 'keypoint2d', 'keypoint3d',
                                'reshade', 'rgb2sfnorm' ]
            
                if task in just_rescale:
                    simple_rescale_img(predicted, storePath+task+im_name.split('.')[0]+'.png')
            
                
                just_clip = ['rgb2depth', 'rgb2mist']
                if task in just_clip:
                    depth_single_image(predicted, storePath+task+im_name.split('.')[0]+'.png')
            
                
                if task == 'inpainting_whole':
                    inpainting_bbox(predicted, storePath+task+im_name.split('.')[0]+'.png')
            
                    
                if task == 'segmentsemantic':
                    semseg_single_image( predicted, img, storePath+task+im_name.split('.')[0]+'.png')
            
            
                if task in ['class_1000', 'class_places']:
                    classification(predicted, synset, storePath+task+im_name.split('.')[0]+'.png')
            
                
                if task == 'vanishing_point':
                    _ = plot_vanishing_point_smoothed(np.squeeze(predicted), (np.squeeze(img) + 1. )/2., storePath+task+im_name.split('.')[0]+'.png', [])
            
                
                if task == 'room_layout':
                    mean = np.array([0.006072743318127848, 0.010272365569691076, -3.135909774145468, 
                                    1.5603802322235532, 5.6228218371102496e-05, -1.5669352793761442,
                                                5.622875878174759, 4.082800262277375, 2.7713941642895956])
                    std = np.array([0.8669452525283652, 0.687915294956501, 2.080513632043758, 
                                    0.19627420479282623, 0.014680602791251812, 0.4183827359302299,
                                                3.991778013006544, 2.703495278378409, 1.2269185938626304])
                    predicted = predicted * std + mean
                    plot_room_layout(np.squeeze(predicted), (np.squeeze(img) + 1. )/2., storePath+task+im_name.split('.')[0]+'.png', [], cube_only=True)
            
                
                if task == 'jigsaw':
                    predicted = np.argmax(predicted, axis=1)
                    perm = cfg[ 'target_dict' ][ predicted[0] ]
                    show_jigsaw((np.squeeze(img) + 1. )/2., perm, storePath+task+im_name.split('.')[0]+'.png')
            
                '''            
                ############## Clean Up ##############
                training_runners[ 'coord' ].request_stop()
                training_runners[ 'coord' ].join()
                #print("Done: {}".format(config_name))
            
                ############## Reset graph and paths ##############            
                tf.reset_default_graph()
                training_runners['sess'].close()
            
                #predict.append(np.squeeze(predicted))
                feature.append(np.squeeze(representation))
                
                count+=1
                print(str(count)+'has been processed!')
                       
            with open(storePath+task+'_fea.dat','wb') as fw:
                pickle.dump(feature,fw)
            print("data written to",storePath+task+'_fea.dat')
            '''
            with open(storePath+task+'_pre.dat','wb') as f:
                pickle.dump(predict,f)
            print("data written to",storePath+task+'_pre.dat')
            '''
    else:
        continue
    
