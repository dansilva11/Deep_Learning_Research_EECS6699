import math
import tensorflow as tf
import numpy as np

from keras import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

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
	# Setup a Sequential NN
	model = Sequential()

	# Setup Input Layer
	model.add(Dense(w_i, activation='relu', kernel_initializer='RandomUniform', 
		input_dim=d))

	# Setup Hidden Layers
	for i in range(0,L-2):
		model.add(Dense(w_i, activation='relu', kernel_initializer='RandomUniform'))

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
def buildSubModel(x_train, y_train, L, VC, d, num_depths, x, epochs):

	# Build Model
	W = initParams(VC, L)
	hidden_nodes = initNodes(L, W, d)
	model = compModel(d, hidden_nodes, L)
				
	# Train Model
	loss_history = model.fit(x_train, y_train, epochs=epochs)

	# Setup Graphs
	plt.subplot(2, num_depths, x)
	plt.plot(loss_history.history['loss'], label='VCdim='+str(VC))
	plt.title('Layer Depth = '+str(L))
	if x == 1:
		plt.legend()
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.subplot(2, num_depths, x+num_depths)
	plt.plot(loss_history.history['binary_accuracy'])
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	
	return loss_history