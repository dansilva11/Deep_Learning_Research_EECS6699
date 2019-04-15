# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 20:01:53 2019

@author: Rohan
"""
import math
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense
import numpy as np


def initWeights(L,W,d):
    if L==2:
        w_i=W/(d+1)
    else:
        w_i = ((1-d-L) + math.sqrt((d+L-1)**2 - 4*(L-2)*(1-W)))/(2*L-4)
    w_i=round(w_i)
    return w_i
    
    

def compModel(d,w_i,L):
    model = Sequential()
    # Hidden Layers
    model.add(Dense(w_i, activation='relu', kernel_initializer='RandomUniform', input_dim=d))
    for i in range(0,L-2):
        #Hidden Layers
        model.add(Dense(w_i, activation='relu', kernel_initializer='RandomUniform'))

    #Output Layer
    model.add(Dense(1, activation='relu', kernel_initializer='RandomUniform'))

    # model = tf.keras.models.Sequential([layer0, layer1, layer3])
    model.count_params()
    model.summary()
    model.compile(optimizer='sgd', loss='mean_squared_error',metrics=['binary_accuracy'])
    
    return model
        
def buildSubModel(x_train,y_train,L,VC,d,ld,x,epochs):
    W = 0

    while True:
        W+=.1
        error=VC-W*L*math.log(W)
        if error<1:
            break
    W=round(W)
        
    w_i = initWeights(L,W,d)
        
    model = compModel(d,w_i,L)
                
    # Train the model
    his = model.fit(x_train, y_train, epochs=epochs)
    
    plt.subplot(2,ld,x)
    plt.plot(his.history['loss'],label='VCdim='+str(VC))
    plt.title('Layer Depth = '+str(L))
    if x==1:
        plt.legend()
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.subplot(2, ld, x+ld)
    plt.plot(his.history['binary_accuracy'])
    # plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    
    return his

def getLWeights(L,w_s):
    wts = np.ones(L)
    wts[0] = d*w_s[0]
    for i in range(L):
        if (i > 0) and (i < L-2):
            wts[i] = (w_s[i-1]+1)*w_s[i]
        wts += w_s[i]+1
    return wts

def customWeights(L,W,d,Lbar,pace):
    w_s = np.ones(L) 
    #start with evenly distributed nodes
    w_s = w_s * Def_Model.initWeights(L,W,d)
    #weights per layer
    wts = getLWeights(L,w_s)
    mylbar = round(sum(np.cumsum(wts)/sum(wts)))
    print('Starting Weights: ',str(sum(wts)))
    notReady = True
    
    if Lbar == mylbar:
        return w_s
    else:
        while(notReady):
            #print('NotReady:: moving lbar ',str(mylbar),' to target ',str(Lbar))
            #print(w_s)
            cL = L-Lbar #concentrated layer
            #if mylbar != Lbar: 
            deltas = pace*np.ones(L) #take pace nodes from each layer
            deltas[cL] = 0 #dont take nodes from cL
            for i in range(L):
                if(0 >= w_s[i] - deltas[i]): #if too many nodes taken, keep at least 1
                    deltas[i] = w_s[i]-1
                w_s[i] = max(1,w_s[i]-deltas[i])
            
            if(sum(deltas)==0): #add pace nodes to cL if non taken
                w_s[cL] += pace    
            w_s[cL] += sum(deltas) #add balance of nodes onto cL
            
            wts = getLWeights(L,w_s)
            if(sum(wts) < W): #if there are still weights to be added add a node to all layers
                for i in range(L): w_s[i] += 1 
            if(sum(wts) > 1.10*W): #if there are too many weights
                for i in range(L): w_s[i] = max(1,w_s[i]-1) 
            wts = getLWeights(L,w_s)
            print('Current Weights: ',str(sum(wts)))
                
                
            mylbar = round(sum(np.cumsum(wts)/sum(wts)))
            if((mylbar == Lbar) & ((sum(wts)/W - 1) > -0.01)):
                notReady = True
                print('Total Weights: ',str(sum(wts)),' with Lbar: ',str(mylbar))
                return w_s                    
    return w_s
