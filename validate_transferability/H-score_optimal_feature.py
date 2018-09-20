"""
This file is for computing H_{T}(f_{T_{opt}}) for each target task. 
"""
import tensorflow as tf
import keras
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Input, Dense, GlobalAveragePooling2D, Lambda, AveragePooling2D, Flatten
from keras import backend as K
from keras.utils.np_utils import to_categorical
import copy
import numpy as np
#from sklearn.cross_validation import train_test_split

from keras.datasets import cifar100
(X_train, y_train), (X_test, y_test) =cifar100.load_data(label_mode='fine')
(_, y_train_coarse), (_, y_test_coarse) =cifar100.load_data(label_mode='coarse')
from keras.optimizers import SGD
from keras.backend import clear_session

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
    from skimage.transform import resize
    X_train_i = np.array([resize(x, (224,224)) for x in X_train_i], dtype=np.float32)
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
    dif=np.trace(np.dot(np.linalg.pinv(Covf,rcond=1e-10), Covg))
    return dif


trn_accs = []
tst_accs = []
trn_loss = []
score = []
class_num_list=[2,5,10,20,40]
for i in range(len(class_num_list)-1):
    class_num = class_num_list[i+1]-class_num_list[i]
    print(class_num)
    clear_session()
    input_tensor = Input(shape=(224, 224, 3)) 
    # create the base pre-trained model
    base_model = ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(class_num, activation='softmax')(x)
    
    f = base_model.get_layer('activation_40').output
    f = AveragePooling2D((14, 14), name='avg_pool')(f)
    feature = Flatten()(f)
    model_fea = Model(inputs=base_model.input, outputs=feature)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    for layer in model.layers:
       layer.trainable = True
    
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
        
    history_callback = model.fit(X_train_i,y_train_i,epochs=5, batch_size=32, validation_data=(X_test_i, y_test_i))
    trn_acc = history_callback.history['acc'][-1]
    trn_accs.append(trn_acc)
    
    tst_acc=history_callback.history['val_acc'][-1]
    tst_accs.append(tst_acc)
    
    loss_history = history_callback.history["loss"][-1]
    trn_loss.append(loss_history)
    
    feature_4f = model_fea.predict(X_train_i)
    hscore = getDiffNN(feature_4f, y_train_i)
    score.append(hscore)
    
np.save('../outputs/trn_accs_a.npy', trn_accs)
np.save('../outputs/tst_accs_a.npy', tst_accs)
np.save('../outputs/trn_loss_a.npy', trn_loss)
np.save('../outputs/score_a.npy', score)
