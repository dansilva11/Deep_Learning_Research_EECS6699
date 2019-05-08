import math
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense
import numpy as np

# Custom Modules
#from Def_Model import initNodes

# -- Variable Definitions -- 
#   d = Input Data Dimensions
#   L = Network Depth (# layers, excl. Input Layer)
#   hL= Number of hidden layers in a network (excl. Input and Output)
#   W = # Parameters
#   w_s = List of # Hidden Nodes/Layer (Custom value per Layer)


#d and L are assumed to be positive integers, w_s is an array of positive integers 
def getLWeights(d,L,w_s): 
    #returns an array of input parameters per layer 
    
    # Setup 1-D array to sum weights across all layers
    #   set an extra layer to grab the output weights
    layer_weights = np.ones(L+1) 
    
    layer_weights[0] = d*w_s[0] #first layer takes from input
    
    # Define # weights/layer and incrementally sum weights across prev layers
    for i in range(L+1):  
        #move from second layer through to the last hidden layer 
        if (i > 0) and (i <= L-1): 
            # Weights per Hidden Layer = # Nodes in Prev Layer (+1 for Bias) x # Nodes in Curr Layer
            layer_weights[i] = (w_s[i-1]+1)*w_s[i]
        #once you are in the output layer, count the nodes plus bias 
        elif(i > L-1): 
            layer_weights[i] = w_s[i-1]+1
            
    return layer_weights


#all inputs are assumed to be positive integers 
def customWeights(hL,W,d,cL,minw):
    np.set_printoptions(suppress=True)
    print("Generating network with concentrated layer ",str(cL))
    
    # Initialize 1-D array of hidden nodes per layer
    layer_nodes = np.ones(hL) 

    # With a single hidden layer, number of nodes is found in a single term
    if(hL==1):
        cW = round((W-1)/(d+1))
    # With multiple hidden layers, the solution depends on the concentrated layer 
    #   use closed form solution to approximate root 
    else: 
        if(cL==1): # When the concentraded layer is the first hidden layer
            cW = round((W - max(0,hL-3)*(minw+1)*minw - (minw+1) - minw)/(d + minw))
        elif(cL==hL): # When the concentrated layer is the last hidden layer
            cW = round((W - d*minw - max(0,hL-3)*(minw+1)*minw - 1)/(minw + 1))
        else: # When the concentrated layer is somewhere in the middle 
            cW = round((W - d*minw - max(0,hL-3)*(minw+1)*minw - (minw+1) - minw)/(2*minw + 1))
    
    # If minw is too high we can end up with negative cW 
    #   so we enforce at least minw nodes at the concentrated layer 
    cW = max(cW,minw) 
    
    # Convert cL to a 0 base index for array assignment 
    cL -= 1 
    
    # Initialize all layers to minw 
    layer_nodes = layer_nodes * minw 
    # Step the concentrated layer up to cW number of nodes
    layer_nodes[cL] = cW 
    # Get the weights per layer given the nodes assignment 
    layer_weights = getLWeights(d,hL,layer_nodes) 
    # Total number of weights in the network (from input through to output)
    sum_weights = sum(layer_weights)
    
    # Compute a-priori Lbar given Bartlett's definition 
    lbar = round(sum(np.cumsum(layer_weights)/sum_weights)) 
    print('Weights: ',str(sum_weights), " with Lbar = ",str(lbar))    
    
    # Return both the set of nodes per layer (architecture) along with the resultant Lbar 
    return (layer_nodes,lbar)