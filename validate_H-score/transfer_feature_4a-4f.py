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

pic_dir_out = './224_resnet50_cifar/'

f = h5py.File('./pic_out/Cifar100_color_data_train_224X224_p1.h5','r')
X_train = f['X_train_p1'][:]
#y_train1 = f['y_train_p1'][:]
f.close()

alphabet = [chr(i) for i in range(97,123)]

for i, k in enumerate(np.linspace(25, 40, 6)):
    clear_session()

    input_tensor = Input(shape=(224, 224, 3))
    
    base_model = ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=False)
    layer_name = 'activation_'+str(int(k))
    x = base_model.get_layer(layer_name).output
    
    feature = AveragePooling2D((14, 14), name='avg_pool')(x)
    model = Model(inputs=base_model.input, outputs=feature)
    
    resnet50_train_output = model.predict(X_train)
    
    file_name = os.path.join(pic_dir_out,'resnet50_train_4'+alphabet[i]+'_output_p3'+'.h5')
    
    f = h5py.File(file_name,'w')          
    f.create_dataset('resnet50_train_4'+alphabet[i]+'_output_p3', data = resnet50_train_output)
    f.close()
    
    print('resnet50_train_4'+alphabet[i]+'_output is done!')
