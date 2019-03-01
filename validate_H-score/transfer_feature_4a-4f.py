"""
This file is for extracting features from pre-trained resnet-50 layer4a to 4f. 
"""
import h5py
from keras.applications.resnet50 import ResNet50
from keras.models import Model 
from keras.layers import Input
from keras.layers import AveragePooling2D
from keras.backend import clear_session
#from keras.utils import np_utils
import os

#from keras.layers import Flatten, Dense
#from keras.optimizers import Adam

import numpy as np
from skimage.transform import resize
'''
pic_dir_out = './224_resnet50_cifar/'

f = h5py.File('./pic_out/Cifar100_color_data_train_224X224_p1.h5','r')
X_train = f['X_train_p1'][:]
#y_train1 = f['y_train_p1'][:]
f.close()
'''
from keras.datasets import cifar100
(X_train, y_train), (X_test, y_test) =cifar100.load_data(label_mode='fine')
X_train = np.concatenate((X_test,X_train),axis=0)

alphabet = [chr(i) for i in range(97,123)]
# For the limited memory capacity, features are extracted part by part. 
for j in range(6):
    for i, k in enumerate(np.linspace(25, 40, 6)):
        clear_session()

        input_tensor = Input(shape=(224, 224, 3))

        base_model = ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=False)
        layer_name = 'activation_'+str(int(k))
        x = base_model.get_layer(layer_name).output

        feature = AveragePooling2D((14, 14), name='avg_pool')(x)
        model = Model(inputs=base_model.input, outputs=feature)
       
        X_train_p = np.array([resize(x,(224,224)) for x in X_train[j*10000:(j+1)*10000]], dtype=np.float32)
        resnet50_train_output = model.predict(X_train_p)

        file_name = os.path.join(pic_dir_out,'resnet50_train_4'+alphabet[i]+'_output_p'+str(j)+'.h5')

        f = h5py.File(file_name,'w')          
        f.create_dataset('resnet50_train_4'+alphabet[i]+'_output_p'+str(j), data = resnet50_train_output)
        f.close()

        print('resnet50_train_4'+alphabet[i]+'_output_p'+str(j)+' is done!')
