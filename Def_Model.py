import math
import tensorflow as tf
import numpy as np
import time

from keras import Sequential
from keras.layers import Dense
from keras.callbacks import History
import matplotlib.pyplot as plt
from decimal import Decimal
from Def_Lbar import customWeights
from Def_GMatrix import initGMatrix, dynamicGMatrix, calcLambdaMin, maximalDist

# References
# [1] Bartlett et al. ”Nearly-tight VC-dimension and pseudodimension bounds for piecewise 
#     Linear NNs”, Conference Paper at Conference on Learning Theory 2017, October 2017.
# [2] Weinan et al. " A Comparative Analysis of the Optimization and Generalization Property
#	  of Two-layer Neural Network and Random Feature Models Under Gradient Descent Dynamics"
#	  April 2019.

# -- Variable Definitions -- 
#   d = Input Data Dimensions
#   L = Network Depth (# layers, excl. Input Layer)
#   VC = Network VC Dimension 
#   W = # Parameters
#   hidden_nodes = # Hidden Nodes per layer

# Calculate hidden node count per layer (evenly distributed)
def initNodes(L, W, d):
    # Case: 2-Layer NN (Control)
    if L == 2:
        hidden_nodes = W/(d+1)

    # Case: Variant NNs
    else:
        # Original Formula: W = (n*d*hidden_nodes)+(L-2)*(hidden_nodes^2)+hidden_nodes
        hidden_nodes = ((1-d-L) + math.sqrt((d+L-1)**2 - 4*(L-2)*(1-W)))/(2*L-4)

    hidden_nodes = round(hidden_nodes)
    return hidden_nodes

# Calculate parameter count
def initParams(VC, L):
    W = 0
    while True:
        W += .1

        # Error = VC-Dim - Current VC upper bound (Source: [1])
        error = VC - W*L*math.log(W)
        if error < 1:
            break

    W = round(W)

    return W

# Setup model architecture
def compModel(d, w_i, L):
    # Setup a Sequential NN and Input layer
    model = Sequential()

    # Setup 1st Hidden Layer
    model.add(Dense(w_i, activation='relu', kernel_initializer='RandomUniform',
        input_dim=d))

    # Setup other Hidden Layers (2nd+)
    for i in range(0,L-2):
        model.add(Dense(w_i, activation='relu', kernel_initializer='RandomUniform', batch_input_shape=(None, 784)))

    # Setup Output Layer
    model.add(Dense(1, activation='relu', kernel_initializer='RandomUniform'))

    # Print Model Details
    model.count_params()
    model.summary()

    # Configure Model for Training
    model.compile(optimizer='sgd', loss='mean_squared_error',
        metrics=['binary_accuracy'])

    return model

def compCustomModel(d, w_s, L):
    # Setup a Sequential NN
    model = Sequential()

    # Setup Input Layer
    model.add(Dense(int(w_s[0]), activation='relu', kernel_initializer='RandomUniform',
        input_dim=d))

    # Setup Hidden Layers
    if(L > 2):
        for i in range(1,L-2):
            model.add(Dense(int(w_s[i]), activation='relu', kernel_initializer='RandomUniform'))

    # Setup Output Layer
    model.add(Dense(1, activation='relu', kernel_initializer='RandomUniform'))

    # Print Model Details
    model.count_params()
    model.summary()

    # Configure Model for Training
    model.compile(optimizer='sgd', loss='mean_squared_error',
        metrics=['binary_accuracy'])

    return model
    

# Build/Train Model and Plot Training Loss
def buildSubModel(x_train, y_train, L, VC, d, x, epochs):

    # Build Model
    W = initParams(VC, L)
    hidden_nodes = initNodes(L, W, d)
    model = compModel(d, hidden_nodes, L)

    # Setup Gram (Infinity) Matrix
    G_Matrix = {}
    weight_matrix_0 = model.get_weights()
    H0 = initGMatrix(x_train, weight_matrix_0, hidden_nodes)


    # Train Model
    start = time.time()
    loss_history = model.fit(x_train, y_train, batch_size=len(x_train), epochs=epochs, verbose=0)
    end = time.time()
    elapsed = end - start

    # Calculate End Gram Matrix and Lambda Min
    weight_matrix = model.get_weights()
    H = dynamicGMatrix(x_train, weight_matrix, hidden_nodes, L)
    M = maximalDist(weight_matrix, weight_matrix_0)
    lambda_min = calcLambdaMin(H)

    G_Matrix['gram_matrix'] = [H0,H]
    G_Matrix['max_dist'] = M
    G_Matrix['lambda_min'] = lambda_min

    end = time.time()
    elapsed = end - start 


    # Setup Graphs
    node_string = ''
    i = 1
    while i < L:
        node_string = node_string + r'$h_%s$ = '%i + str(hidden_nodes) +', '
        i+=1

    # Build Sub-Graphs
    key_str = "$VC_{max}$: upper bound of VC dimension \n$W$: total # of network parameters \n $h_i$: number of nodes in hidden layer $i$"
    ax =plt.subplot(2, 1, 1)
    plt.plot(loss_history.history['loss'], label=(r'$\bf VC_{max}$ = '+r'%.2E' % Decimal(str(VC)) + r'   $\bf W$ = ' + '%.2E' % Decimal(str(W)) +'   '+node_string + r'   $\bf train time$ = ' + r'%.2E' % Decimal(str(elapsed))))
    # plt.title('Network Depth = '+str(L),fontsize=15,loc='left')

    plt.legend(loc='upper right', markerscale=50, bbox_to_anchor=(1, 1.35),
      ncol=1, fancybox=True, shadow=True, fontsize=10,title=r'$\bf Network Parameters$'+'\n          (depth='+str(L)+')')

    if x == 1:
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        plt.text(0, 1.25, key_str, fontsize=10,
                 horizontalalignment='left',
                verticalalignment='top', bbox=props, transform=ax.transAxes)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.grid(color='gray', linestyle='--', linewidth=.5)
    plt.subplot(2, 1, 2)
    plt.plot(loss_history.history['binary_accuracy'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.grid(color='gray', linestyle='--', linewidth=.5)
    # plt.subplot(3, num_depths, x+num_depths+num_depths)
    # plt.plot(loss_history.history['lambda_min'])
    # plt.ylabel('lambda_min')
    # plt.xlabel('epoch')
    # plt.grid(color='gray', linestyle='--', linewidth=.5)

    return loss_history, G_Matrix, W, hidden_nodes

# Build/Train Model and Plot Training Loss
def buildCustomModel(x_train, y_train, L, VC, d, concenLayer, x, epochs, minw):
    import time
    # Build Model
    W = initParams(VC, L)
    cw = customWeights(L, W, d, concenLayer, minw)
    w_s = cw[0]  # pull the custom number of weights per layer as an array
    lbar = cw[1]  # pull the defined network's assumed Lbar
    model = compCustomModel(d, w_s, L)  # build the network given the fixed number of weights per layer
    hidden_nodes = sum(w_s)
    # Setup Gram (Infinity) Matrix
    G_Matrix = {}
    weight_matrix_0 = model.get_weights()
    H0 = initGMatrix(x_train, weight_matrix_0, hidden_nodes)

    # Train Model
    start = time.time()
    loss_history = model.fit(x_train, y_train, batch_size=len(x_train), epochs=epochs, verbose=0)
    end = time.time()
    elapsed = end - start

    # Calculate End Gram Matrix and Lambda Min
    weight_matrix = model.get_weights()
    H = dynamicGMatrix(x_train, weight_matrix, hidden_nodes, L)
    M = maximalDist(weight_matrix, weight_matrix_0)
    lambda_min = calcLambdaMin(H)

    G_Matrix['gram_matrix'] = [H0, H]
    G_Matrix['max_dist'] = M
    G_Matrix['lambda_min'] = lambda_min

    # Setup Graphs
    node_string = ''
    i = 1
    while i < L:
        node_string = node_string + r'$h_%s$ = ' % i + str(hidden_nodes) + ', '
        i += 1

    # Build Sub-Graphs
    key_str = "$VC_{max}$: upper bound of VC dimension \n$W$: total # of network parameters \n $h_i$: number of nodes in hidden layer $i$"
    ax = plt.subplot(2, 1, 1)
    plt.plot(loss_history.history['loss'], label=(
                r'$\bf VC_{max}$ = ' + r'%.2E' % Decimal(str(VC)) + r'   $\bf W$ = ' + '%.2E' % Decimal(
            str(W)) + '   ' + node_string + r'   $\bf train time$ = ' + r'%.2E' % Decimal(str(elapsed))))
    # plt.title('Network Depth = '+str(L),fontsize=15,loc='left')

    plt.legend(loc='upper right', markerscale=50, bbox_to_anchor=(1, 1.35),
               ncol=1, fancybox=True, shadow=True, fontsize=10,
               title=r'$\bf Network Parameters$' + '\n          (depth=' + str(L) + ')')

    if x == 1:
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        plt.text(0, 1.25, key_str, fontsize=10,
                 horizontalalignment='left',
                 verticalalignment='top', bbox=props, transform=ax.transAxes)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.grid(color='gray', linestyle='--', linewidth=.5)
    plt.subplot(2, 1, 2)
    plt.plot(loss_history.history['binary_accuracy'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.grid(color='gray', linestyle='--', linewidth=.5)
    # plt.subplot(3, num_depths, x+num_depths+num_depths)
    # plt.plot(loss_history.history['lambda_min'])
    # plt.ylabel('lambda_min')
    # plt.xlabel('epoch')
    # plt.grid(color='gray', linestyle='--', linewidth=.5)

    return loss_history, G_Matrix, W, w_s