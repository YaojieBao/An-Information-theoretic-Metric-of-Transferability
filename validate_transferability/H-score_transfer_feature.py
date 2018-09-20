"""
This file is for computing the H-score of the transferred feature(layer 4f) for each target task. 
"""
import os
import h5py

from keras.models import Model
from keras.layers import Input, Dense

import copy
import numpy as np
#from sklearn.cross_validation import train_test_split

from keras.datasets import cifar100
(_, y_train), (_, y_test) =cifar100.load_data(label_mode='fine')
(_, y_train_coarse), (_, y_test_coarse) =cifar100.load_data(label_mode='coarse')


def transformLabels(y_train_i, y_train_5):
    num = len(y_train_5)
    seats = list(range(num))
    for k, y in enumerate(y_train_5):
        if y<num:
            seats.remove(y)
            y_train_5.remove(y)
    
    #assert(len(y_train_5)==len(seats))
    for k, y in enumerate(y_train_5):
        y_train_i[y_train_i==y]=seats[k]
        
    return y_train_i


def getData(i, class_num_list, X_train, y_train, y_train_coarse, y_train_5):
    indices_i = [y_train==k for k in range(class_num_list[i], class_num_list[i+1])]
    indices = np.concatenate(indices_i,axis=1)
    indices = np.sum(indices, axis=1)
    indices_i = [bool(i) for i in indices]

    X_train_i = X_train[indices_i]

    y_train_i = y_train[indices_i]
    
    y_train_i = transformLabels(y_train_i, y_train_5)
    
    return X_train_i, y_train_i


def getCov(X):
    X_mean=X-np.mean(X,axis=0,keepdims=True)
    cov = np.divide(np.dot(X_mean.T, X_mean), len(X)-1) 
    return cov
    

def getDiffNN(f,Z):
    #Z=np.argmax(Z, axis=1)
    Covf=getCov(f)
    alphabetZ=list(set(Z.reshape((-1,))))
    g=np.zeros_like(f)
    for z in alphabetZ:
        Ef_z=np.mean(f[np.reshape(Z==z, (-1,))], axis=0)
        g[np.reshape(Z==z, (-1,))]=Ef_z
    
    Covg=getCov(g)
    dif=np.trace(np.dot(np.linalg.pinv(Covf,rcond=1e-9), Covg))
    return dif

pic_dir_out = '/home/yang/Desktop/H-scoreVSlogloss/224_resnet50_cifar/'
alphabet = [chr(i) for i in range(97,123)] 
i = 5
file_name = os.path.join(pic_dir_out,'resnet50_train_4'+alphabet[i]+'_output_p1'+'.h5')
#if os.path.exists(file_name):
f = h5py.File(file_name,'r')
resnet50_train_output_p1 = f['resnet50_train_4'+alphabet[i]+'_output_p1'][:]
#resnet50_train_labels = f['resnet50_train_4'+alphabet[i]+'_labels'][:]
f.close()

file_name = os.path.join(pic_dir_out,'resnet50_train_4'+alphabet[i]+'_output_p2'+'.h5')
#if os.path.exists(file_name):
f = h5py.File(file_name,'r')
resnet50_train_output_p2 = f['resnet50_train_4'+alphabet[i]+'_output_p2'][:]
#resnet50_train_labels = f['resnet50_train_4'+alphabet[i]+'_labels'][:]
f.close()

file_name = os.path.join(pic_dir_out,'resnet50_train_4'+alphabet[i]+'_output_p3'+'.h5')
#if os.path.exists(file_name):
f = h5py.File(file_name,'r')
resnet50_train_output_p3 = f['resnet50_train_4'+alphabet[i]+'_output_p3'][:]
#resnet50_train_labels = f['resnet50_train_4'+alphabet[i]+'_labels'][:]
f.close()

file_name = os.path.join(pic_dir_out,'resnet50_train_4'+alphabet[i]+'_output_p4'+'.h5')
#if os.path.exists(file_name):
f = h5py.File(file_name,'r')
resnet50_train_output_p4 = f['resnet50_train_4'+alphabet[i]+'_output_p4'][:]
#resnet50_train_labels = f['resnet50_train_4'+alphabet[i]+'_labels'][:]
f.close()

file_name = os.path.join(pic_dir_out,'resnet50_train_4'+alphabet[i]+'_output_p5'+'.h5')
#if os.path.exists(file_name):
f = h5py.File(file_name,'r')
resnet50_train_output_p5 = f['resnet50_train_4'+alphabet[i]+'_output_p5'][:]
#resnet50_train_labels = f['resnet50_train_4'+alphabet[i]+'_labels'][:]
f.close()

file_name = os.path.join(pic_dir_out,'resnet50_train_4'+alphabet[i]+'_output'+'.h5')
#if os.path.exists(file_name):
f = h5py.File(file_name,'r')
resnet50_train_output_p0 = f['resnet50_train_4'+alphabet[i]+'_output'][:]
#resnet50_train_labels = f['resnet50_train_4'+alphabet[i]+'_labels'][:]
f.close()

X_train = np.concatenate((resnet50_train_output_p1,resnet50_train_output_p2,resnet50_train_output_p3,resnet50_train_output_p4,resnet50_train_output_p5))
X_train = np.squeeze(X_train)
X_test = np.squeeze(resnet50_train_output_p0)

from keras.optimizers import SGD
from keras.backend import clear_session

trn_accs = []
tst_accs = []
trn_loss = []
score = []
class_num_list=[2,5,10,20,40]
for i in range(len(class_num_list)-1):
    class_num = class_num_list[i+1]-class_num_list[i] 
    print(class_num)
    clear_session()
    input_tensor = Input(shape=(1024,))
    #x = Dense(1024, activation='relu')(x)
    predictions = Dense(class_num, activation='softmax')(input_tensor)   
    
    model = Model(inputs=input_tensor, outputs=predictions)
    model.compile(optimizer=SGD(lr=1e-4, momentum=0.9), loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    
    indices_i = [y_train==i for i in range(class_num_list[i], class_num_list[i+1])]
    indices = np.concatenate(indices_i,axis=1)
    indices = np.sum(indices, axis=1)
    indices = [bool(i) for i in indices]
    y_train_i = y_train[indices]

    y_train_5 = list(set(y_train_i.reshape(-1,)))
    y_test_5 = copy.deepcopy(y_train_5)
    X_train_i, y_train_i = getData(i, class_num_list, X_train, y_train, y_train_coarse, y_train_5)
    X_test_i, y_test_i = getData(i, class_num_list, X_test, y_test, y_test_coarse, y_test_5)

    history_callback = model.fit(X_train_i,y_train_i,epochs=100, batch_size=64, validation_data=(X_test_i, y_test_i))
    trn_acc=history_callback.history['acc']
    trn_accs.append(trn_acc[-1])

    loss_history = history_callback.history['loss']
    trn_loss.append(loss_history[-1])

    tst_acc = history_callback.history['val_acc'][-1]
    print(tst_acc)
    tst_accs.append(tst_acc)

    hscore = getDiffNN(X_train_i, y_train_i)
    score.append(hscore)

np.save('../outputs/trn_accs_p.npy', trn_accs)
np.save('../outputs/tst_accs_p.npy', tst_accs)
np.save('../outputs/trn_loss_p.npy', trn_loss)
np.save('../outputs/score_p.npy', score)
