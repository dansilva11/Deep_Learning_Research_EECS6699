import math
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense
import numpy as np

# Custom Modules
from Def_Model import initNodes

# -- Variable Definitions -- 
#   d = Input Data Dimensions
#   L = Network Depth (# layers, excl. Input Layer)
#   W = # Parameters
#   w_s = List of # Hidden Nodes/Layer (Custom value per Layer)
#   sum_weights = Array of incremental sum of weights in previous layers 

def getLWeights(L, w_s, d):

	# Setup 1-D array to sum weights across all layers
	layer_weights = np.ones(L)

	# Define # nodes for input layer
	layer_weights[0] = d*w_s[0]

	# Define # weights/layer and incrementally sum weights across prev layers
	for i in range(L):
		if (0 < i < L-2):
			# Weights per Hidden Layer = # Nodes in Prev Layer (+1 for Bias) x # Nodes in Curr Layer
			layer_weights[i] = (w_s[i-1]+1)*w_s[i]
	
	# Increment total # weights from last layer to output
	layer_weights += w_s[i]
	
	return layer_weights

def customWeights(L, W, d, Lbar, pace):

	# Initialize 1-D array of hidden nodes per layer
	w_s = np.ones(L) 

	# Start with evenly distributed nodes
	w_i = initNodes(L, W, d)
	w_s = w_s * w_i

	# Setup 1-D Sum of total # weights in network
	layer_weights = getLWeights(L, w_s, d)

	np.set_printoptions(suppress=True)

	# Cumulative sum of flattened 
	mylbar = round(sum(np.cumsum(sum_weights)/sum(sum_weights)))
	print('Starting Weights: ',str(sum(sum_weights)))
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
			
			sum_weights = getLWeights(L, w_s, d)
			if(sum(sum_weights) < W): #if there are still weights to be added add a node to all layers
				for i in range(L): w_s[i] += 1 
			if(sum(sum_weights) > 1.10*W): #if there are too many weights
				for i in range(L): w_s[i] = max(1,w_s[i]-1) 
			sum_weights = getLWeights(L, w_s, d)
			print('Current Weights: ',str(sum(sum_weights)))
				
				
			mylbar = round(sum(np.cumsum(sum_weights)/sum(sum_weights)))
			if((mylbar == Lbar) & ((sum(sum_weights)/W - 1) > -0.01)):
				notReady = False
				print('Total Weights: ',str(sum(sum_weights)),' with Lbar: ',str(mylbar))
				return w_s                    
	return w_s