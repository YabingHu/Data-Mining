# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 15:48:39 2018

@author: Yabing Hu and Yang Xiao
"""
#%%
#Problem 1
import numpy as np
import math  
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_rcv1
rcv1 = fetch_rcv1()

#part a Make a new label vector, where each article has a 1 if it has been classified as CCAT and a -1 otherwise.
target=rcv1.target.toarray()
result=np.where(target[:,33]==1)[0]
target_new=np.zeros((target.shape[0],1))-1
target_new[result]=1


#part b split to test and training data
x_train=rcv1.data[:100000]
x_test=rcv1.data[100000:]
y_train=target_new[:100000]
y_test=target_new[100000:]
y_train_np1=(y_train+1)/2
y_test_np1=(y_test+1)/2
del target, result
#%%
#Problem 2 Using PEGASOS, train an SVM on the training articles.
#The best model with tunned parameters
def Pegasos(x_train,y_train,T,lambda1,k):
    m, n = x_train.shape    
    w=np.zeros((n,1)) 
    error=[]
    for t in range(1,T+1):
        I=np.random.choice(m, k, replace=False)
        A_plus=np.where(np.multiply(x_train[I] * w, y_train[I]) < 1)
        eta=1.0/(lambda1*t)
        w=(1 - eta * lambda1) * w + (eta / k) * x_train[np.array(I)[A_plus[0].tolist()]].T * y_train[np.array(I)[A_plus[0].tolist()]]
        w=min(1.0, 1.0 / math.sqrt(lambda1 * np.square(w).sum())) * w
        y_pred=np.sign(x_train*w)
        hit=np.where(y_train==y_pred)[0]
        error.append(1-len(hit)/y_train.shape[0])
    return w,error 

w,error_pegasos=Pegasos(x_train,y_train,1000,0.001,100)
plt.plot(error_pegasos,label=' k=100')
plt.legend(loc='best')
plt.title('training error vs iterations when lambda=10e-3')
plt.xlabel('iterations')
plt.ylabel('error')

'''  
#Code for tuning parameters

def Pegasos(x_train,y_train,T,lambda1,k):
    m, n = x_train.shape    
    w=np.zeros((n,1)) 
    error=[]
    for t in range(1,T+1):
        I=np.random.choice(m, k, replace=False)
        A_plus=np.where(np.multiply(x_train[I] * w, y_train[I]) < 1)
        eta=1.0/(lambda1*t)
        w=(1 - eta * lambda1) * w + (eta / k) * x_train[np.array(I)[A_plus[0].tolist()]].T * y_train[np.array(I)[A_plus[0].tolist()]]
        w=min(1.0, 1.0 / math.sqrt(lambda1 * np.square(w).sum())) * w
        y_pred=np.sign(x_train*w)
        hit=np.where(y_train==y_pred)[0]
        error.append(1-len(hit)/y_train.shape[0])
    return error


error1=Pegasos(x_train,y_train,1000,0.01,10)#change the arguments for different cases
error2=Pegasos(x_train,y_train,1000,0.01,100)
error3=Pegasos(x_train,y_train,1000,0.01,500)
error4=Pegasos(x_train,y_train,1000,0.01,1000)    
fig = plt.figure()
plt.plot(error1,label=' k=10')
plt.plot(error2,label=' k=100')
plt.plot(error3,label=' k=500')
plt.plot(error4,label=' k=1000')
plt.legend(loc='best')
plt.title('training error vs iterations when lambda=10e-2')
plt.xlabel('iterations')
plt.ylabel('error')
fig.savefig('test.jpg')
'''

#%%
#Problem 3 Using Adagrad, train an SVM on the training articles.
#The best model with tunned parameters
def Adagrad(x_train,y_train,T,lambda1,k):
    m, n = x_train.shape    
    w=np.zeros((n,1)) 
    gti=np.ones((n,1)) 
    error=[]
    for t in range(1,T+1):
        I=np.random.choice(m, k, replace=False)
        A_plus=np.where(np.multiply(x_train[I[0 : k]] * w, y_train[I[0 : k]]) < 1)
        eta=1.0/(lambda1*t)
        grad=-(1/k)*x_train[np.array(I)[A_plus[0].tolist()]].T * y_train[np.array(I)[A_plus[0].tolist()]]
        gti+=np.square(grad)
        w = w - eta*grad / np.sqrt(gti)
        w=min(1.0, 1.0 / math.sqrt(lambda1 * (gti*np.square(w)).sum())) * w
        y_pred=np.sign(x_train*w)
        hit=np.where(y_train==y_pred)[0]
        error.append(1-len(hit)/y_train.shape[0])
    return w,error

w,error_adagrad=Adagrad(x_train,y_train,1000,0.001,100)
plt.plot(error_pegasos,label=' pegasos')
plt.plot(error_adagrad,label=' adagrad')
plt.legend(loc='best')
plt.title('training error vs iterations when lambda=10e-3 for PEGASOS and ADAGRAD')
plt.xlabel('iterations')
plt.ylabel('error')

'''
#Code for tuning parameters
def Adagrad(x_train,y_train,T,lambda1,k):
    m, n = x_train.shape    
    w=np.zeros((n,1)) 
    gti=np.ones((n,1)) 
    error=[]
    for t in range(1,T+1):
        I=np.random.choice(m, k, replace=False)
        A_plus=np.where(np.multiply(x_train[I[0 : k]] * w, y_train[I[0 : k]]) < 1)
        eta=1.0/(lambda1*t)
        #eta=1.0/np.square(T)
        grad=-(1/k)*x_train[np.array(I)[A_plus[0].tolist()]].T * y_train[np.array(I)[A_plus[0].tolist()]]
        gti+=np.square(grad)
        w = w - eta*grad / np.sqrt(gti)
        w=min(1.0, 1.0 / math.sqrt(lambda1 * (gti*np.square(w)).sum())) * w
        y_pred=np.sign(x_train*w)
        hit=np.where(y_train==y_pred)[0]
        error.append(1-len(hit)/y_train.shape[0])
    return error
error1=Adagrad(x_train,y_train,1000,00.1,10)
error2=Adagrad(x_train,y_train,1000,00.1,100)
error3=Adagrad(x_train,y_train,1000,00.1,500)
error4=Adagrad(x_train,y_train,1000,00.1,1000)    
fig = plt.figure()
plt.plot(error1,label=' k=10')
plt.plot(error2,label=' k=100')
plt.plot(error3,label=' k=500')
plt.plot(error4,label=' k=1000')
plt.legend(loc='best')
plt.title('training error vs iterations when lambda=10e-1')
plt.xlabel('iterations')
plt.ylabel('error')
fig.savefig('test.jpg')
'''
#%%
#Problem 4 Neural Network
#part a
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
m, n = x_train.shape   

model = Sequential()
model.add(Dense(100, input_dim=n, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
result_1=model.fit(x_train, y_train_np1,epochs=5,batch_size=128)
print("Finsh training 1 hidden layer neural network")

model = Sequential()
model.add(Dense(100, input_dim=n, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
result_2=model.fit(x_train, y_train_np1,epochs=5,batch_size=128)
print("Finsh training 2 hidden layer neural network")

model = Sequential()
model.add(Dense(100, input_dim=n, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
result_3=model.fit(x_train, y_train_np1,epochs=5,batch_size=128)
print("Finsh training 3 hidden layer neural network")


plt.plot(1-np.asarray(result_1.history['acc']),'ro',label=' 1 hidden layer')
plt.plot(1-np.asarray(result_2.history['acc']),'bo',label=' 2 hidden layer')
plt.plot(1-np.asarray(result_3.history['acc']),'go',label=' 3 hidden layer')
plt.legend(loc='best')
plt.title('training error for three neural network')
plt.xlabel('number of epoch')
plt.ylabel('error')
#%%

# Problem 4 part b
#below is the best model
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
#from keras.layers.core import Dense, Dropout, Activation  
m, n = x_train.shape    
model = Sequential()
model.add(Dense(100, input_dim=n, activation='tanh'))
#model.add(Dropout(0.5))
model.add(Dense(100, activation='tanh'))
#model.add(Dropout(0.5))
model.add(Dense(100, activation='tanh'))
#model.add(Dropout(0.5))
#model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
result=model.fit(x_train, y_train_np1,epochs=5,batch_size=128)


#%%
#Problem 5
# Final Pegasos model
def Pegasos(x_train,y_train,T,lambda1,k):
    m, n = x_train.shape    
    w=np.zeros((n,1)) 
    for t in range(1,T+1):
        I=np.random.choice(m, k, replace=False)
        A_plus=np.where(np.multiply(x_train[I] * w, y_train[I]) < 1)
        eta=1.0/(lambda1*t)
        w=(1 - eta * lambda1) * w + (eta / k) * x_train[np.array(I)[A_plus[0].tolist()]].T * y_train[np.array(I)[A_plus[0].tolist()]]
        w=min(1.0, 1.0 / math.sqrt(lambda1 * np.square(w).sum())) * w
    return w

def accuracy_test(w,x_train,y_train):
    y_pred=np.sign(x_train*w)
    hit=np.where(y_train==y_pred)[0]
    accuracy=len(hit)/y_train.shape[0]
    return accuracy

# Final Adagrad model
def Adagrad(x_train,y_train,T,lambda1,k):
    m, n = x_train.shape    
    w=np.zeros((n,1)) 
    gti=np.ones((n,1)) 
    for t in range(1,T+1):
        I=np.random.choice(m, k, replace=False)
        A_plus=np.where(np.multiply(x_train[I[0 : k]] * w, y_train[I[0 : k]]) < 1)
        eta=1.0/(lambda1*t)
        grad=-(1/k)*x_train[np.array(I)[A_plus[0].tolist()]].T * y_train[np.array(I)[A_plus[0].tolist()]]
        gti+=np.square(grad)
        w = w - eta*grad / np.sqrt(gti)
        w=min(1.0, 1.0 / math.sqrt(lambda1 * (gti*np.square(w)).sum())) * w
    return w

w_p=Pegasos(x_train,y_train,1000,0.001,100)
error_p_test=1-accuracy_test(w_p,x_test,y_test)
print("test error for PEGASOS is",error_p_test)
w_a=Adagrad(x_train,y_train,1000,0.001,100)
error_a_test=1-accuracy_test(w_a,x_test,y_test)
print("test error for ADAGRAD is",error_a_test)

#Final model for NN
m, n = x_train.shape    
model = Sequential()
model.add(Dense(100, input_dim=n, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',optimizer=sgd, metrics=['accuracy'])
result=model.fit(x_train, y_train_np1,epochs=5,batch_size=128)
classes = model.predict(x_test,batch_size=128)

def error_nn(y_ture,classes):
    y_pred=np.rint(classes)
    hit=np.where(y_ture==y_pred)[0]
    error_nn=1-len(hit)/y_ture.shape[0]
    return error_nn

error_n_test=error_nn(y_test_np1,classes)
print("test error for neural network is",error_n_test)