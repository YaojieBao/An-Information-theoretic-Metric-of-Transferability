"""
This file is for showing the relationship between H-score and transfer performance under the scenario of source task selection. 
"""
import h5py

from keras.models import Model 
from keras.layers import Input

from keras.utils import np_utils
#from keras.utils import np_utils
import os
import matplotlib.pyplot as plt
#from keras.layers import Flatten, Dense
#from keras.optimizers import Adam

import numpy as np

def getCov(X):
    X_mean=X-np.mean(X,axis=0,keepdims=True)
    cov = np.divide(np.dot(X_mean.T, X_mean), len(X)-1) 
    return cov
    
# function to compute H-score
def getDiffNN(f,Z):
    Z=np.argmax(Z, axis=1)
    Covf=getCov(f)
    alphabetZ=list(set(Z))
    g=np.zeros_like(f)
    for z in alphabetZ:
        Ef_z=np.mean(f[Z==z, :], axis=0)
        g[Z==z]=Ef_z
    
    Covg=getCov(g)
    dif=np.trace(np.dot(np.linalg.pinv(Covf,rcond=1e-10), Covg))
    return dif

from keras.layers import Flatten, Dense
from keras.optimizers import Adam

def split_samples(X, split=0.9):
    train_indices = np.random.choice(len(X), int(len(X)*split), replace=False)
    test_indices = np.array(list(set(range(len(X))) - set(train_indices)))
    X_train = X[train_indices]
    X_test = X[test_indices]
    '''
    Y_train = Y[train_indices]
    Y_test = Y[test_indices]
    
    print( "Split ratio:",split ,len(train_indices) ,"training samples,",
           len(test_indices),"testing samples")
    '''
    return X_train, X_test, train_indices, test_indices #Y_train,  Y_test,

# read labels 
from keras.datasets import cifar100
(_, y_train), (_, y_test) =cifar100.load_data(label_mode='fine')
y_train = np.concatenate((y_train,y_test))

alphabet = [chr(i) for i in range(97,123)] 
pic_dir_out = './224_resnet50_cifar/'
num_classes=100
lossr=np.zeros((6,6))
transferr = np.zeros((6,6))  
accr=np.zeros((6,2,6))  
for j,k in enumerate(np.linspace(10000,60000,6,dtype=np.int)):
    y_s = y_train[:k]
    y_train_c = np_utils.to_categorical(y_s, num_classes)
    
    y_trn, y_tst, trn, tst=split_samples(y_train_c, split=0.9)
    
    input_tensor = Input(shape=(1, 1, 1024))
    x = Flatten()(input_tensor)
    #x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)   
    model = Model(inputs=input_tensor, outputs=predictions)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy',metrics=['accuracy'])
    
    hscore=[] 
    loss=[] 
    trn_accs=[]
    tst_accs=[]  
    for i in range(6):
        # read data 
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
        
        resnet50_train_output=np.concatenate((resnet50_train_output_p1,resnet50_train_output_p2,resnet50_train_output_p3,resnet50_train_output_p4,resnet50_train_output_p5,resnet50_train_output_p0))
        resnet50_train_output_squeeze = np.squeeze(resnet50_train_output[:k])
        score = getDiffNN(resnet50_train_output_squeeze[trn], y_trn)
        hscore.append(score)
        
        history_callback = model.fit(resnet50_train_output[trn], y_trn, epochs=100, batch_size=2048) 
        trn_predict=model.predict(resnet50_train_output[trn])
        trn_acc=np.mean(np.argmax(trn_predict,axis=1)==np.argmax(y_trn,axis=1))
        trn_accs.append(trn_acc)
        tst_predict=model.predict(resnet50_train_output[tst])
        tst_acc=np.mean(np.argmax(tst_predict,axis=1)==np.argmax(y_tst,axis=1))
        tst_accs.append(tst_acc)
        loss_history = history_callback.history["loss"]
        loss.append(loss_history[-1])
           
    lossr[j,:]=loss
    transferr[j,:]=hscore
    accr[j,0,:]=trn_accs
    accr[j,1,:]=tst_accs
    
    plt.figure()
    plt.plot(hscore,loss,'b-',marker='o',label='log loss')
    plt.title('log-loss vs transferability for 4a to 4f')
    plt.xlabel('transferability')#('cardinality of z')#
    plt.ylabel('log loss')
    #plt.legend(loc='upper right')
    plt.show()    
        
    plt.figure()
    plt.plot(hscore,trn_accs,'b-',marker='o',label='training accuracy')
    plt.plot(hscore,tst_accs,'r-',marker='o',label='testing accuracy')
    plt.title('accuracy vs transferability for 4a to 4f')
    plt.xlabel('transferability')#('cardinality of z')#
    plt.ylabel('accuracy')
    plt.legend(loc='upper left')
    plt.show() 
    
np.save('./lossr_123450_1.npy',lossr)
np.save('./transferability_123450_1.npy',transferr)
np.save('./accuracy_123450_1.npy',accr)

plt.figure()
plt.plot(transferr[0,:],lossr[0,:],'k-',marker='o',label='num=10000')
plt.plot(transferr[1,:],lossr[1,:],'r-',marker='o',label='num=20000')
plt.plot(transferr[2,:],lossr[2,:],'y-',marker='o',label='num=30000')
plt.plot(transferr[3,:],lossr[3,:],'g-',marker='o',label='num=40000')
plt.plot(transferr[4,:],lossr[4,:],'c-',marker='o',label='num=50000')
plt.plot(transferr[5,:],lossr[5,:],'b-',marker='o',label='num=60000')
plt.title('log-loss vs transferability for 4a to 4f')
plt.xlabel('transferability')#('cardinality of z')#
plt.ylabel('log loss')
plt.legend(loc='upper right')
plt.show()

plt.figure()
plt.plot(transferr[0,:],accr[0,0,:],'k-',marker='o',label='num=10000')
plt.plot(transferr[1,:],accr[1,0,:],'r-',marker='o',label='num=20000')
plt.plot(transferr[2,:],accr[2,0,:],'y-',marker='o',label='num=30000')
plt.plot(transferr[3,:],accr[3,0,:],'g-',marker='o',label='num=40000')
plt.plot(transferr[4,:],accr[4,0,:],'c-',marker='o',label='num=50000')
plt.plot(transferr[5,:],accr[5,0,:],'b-',marker='o',label='num=60000')
plt.title('training accuracy vs transferability for 4a to 4f')
plt.xlabel('transferability')#('cardinality of z')#
plt.ylabel('training accuracy')
plt.legend(loc='lower right')
plt.show()

plt.figure()
plt.plot(transferr[0,:],accr[0,1,:],'k-',marker='o',label='num=10000')
plt.plot(transferr[1,:],accr[1,1,:],'r-',marker='o',label='num=20000')
plt.plot(transferr[2,:],accr[2,1,:],'y-',marker='o',label='num=30000')
plt.plot(transferr[3,:],accr[3,1,:],'g-',marker='o',label='num=40000')
plt.plot(transferr[4,:],accr[4,1,:],'c-',marker='o',label='num=50000')
plt.plot(transferr[5,:],accr[5,1,:],'b-',marker='o',label='num=60000')
plt.title('testing accuracy vs transferability for 4a to 4f')
plt.xlabel('transferability')#('cardinality of z')#
plt.ylabel('testing accuracy')
plt.legend(loc='lower right')
plt.show()
