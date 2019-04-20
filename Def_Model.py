import math
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense
import numpy as np

def initWeights(L,W,d):

    # If 2-Layer NN, # nodes/hidden layer = weight / (dim + 1)
    if L == 2:
        w_i = W/(d+1)

    # Else, # nodes/hidden layer = 
    else:
        # Previous: w_i = -(math.sqrt(d**2+2*d+4*(L-2)*W+1)+d+1)/(4-2*L)
        w_i = ((1-d-L) + math.sqrt((d+L-1)**2 - 4*(L-2)*(1-W)))/(2*L-4)

    # Round hidden nodes value to integer
    w_i=round(w_i)
    return w_i

def compModel(d,w_i,L):
    # Setup a Sequential NN
    model = Sequential()

    # Input Layer
    model.add(Dense(w_i, activation='relu', kernel_initializer='RandomUniform', input_dim=d))

    # Hidden Layers
    for i in range(0,L-2):
        model.add(Dense(w_i, activation='relu', kernel_initializer='RandomUniform'))

    # Output Layer
    model.add(Dense(1, activation='relu', kernel_initializer='RandomUniform'))

    # Print Model Details
    model.count_params()
    model.summary()
    
    # Compile Model
    model.compile(optimizer='sgd', loss='mean_squared_error',metrics=['binary_accuracy'])
    
    return model
        
def buildSubModel(x_train,y_train,L,VC,d,ld,x,epochs):

    # W - # Parameters
    W = 0

    while True:
        W += .1

        # Error = VC-Dim - Current VC upper bound (Source: Bartlett et al. 2017)
        error = VC - W*L*math.log(W)
        if error < 1:
            break

    # Round parameter count to integer
    W = round(W)
        
    w_i = initWeights(L,W,d)
        
    model = compModel(d,w_i,L)
                
    # Train the model
    his = model.fit(x_train, y_train, epochs=epochs)

    # [Out-of-Scope] Test the model
    # loss, acc = model.evaluate(x_test, y_test)
    # print("loss=", loss, "acc=", acc)

    # Setup Graphs
    plt.subplot(2,ld,x)
    plt.plot(his.history['loss'],label='VCdim='+str(VC))
    plt.title('Layer Depth = '+str(L))
    
    # Define Legend
    if x == 1:
        plt.legend()
    plt.ylabel('loss')
    plt.xlabel('epoch')

    # Plot Graphs
    plt.subplot(2, ld, x+ld)
    plt.plot(his.history['binary_accuracy'])
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
