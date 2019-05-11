import math
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense
import numpy as np
from scipy.sparse.linalg import eigs
# -- Variable Definitions -- 
#   G = (Square) Gram Matrix of d x d dimensions
#   lambda_min = Smallest, real eigenvalue of Gram Matrix

def maximalDist(weight_matrix, weight_matrix_0):
	weight_matrix_0 = weight_matrix_0[::2]
	weight_matrix = weight_matrix[::2]

	# Gram Matrix for other Hidden Layers
	M = []
	for weights0, weights1 in zip(weight_matrix_0[1:], weight_matrix[1:]):
		part_1 = np.linalg.norm(weights1 - weights0)
		M.append(part_1)

	return M

def calcLambdaMin(G):
	eigen_values, eigen_vectors = eigs(G, k=1)
	return eigen_values.real

def calcGMatrix(x, weight_matrix, hidden_nodes):
	# 1st Hidden Layer
	w0 = weight_matrix[0]

	# 1st Layer Weight Matrix (m,d) * Input Data (T) (d,n) = (m,n)
	part_1 = w0.T.dot(x.T)

	# ReLU Activation: Max[Zero Matrix(m,n), Sign of Part 1 (m,n)] = (m,n)
	part_2 = np.zeros(part_1.shape)
	part_3 = np.maximum(part_2,np.sign(part_1))

	return part_3

def initGMatrix(x, weight_matrix, hidden_nodes):
	# Get rid of the bias parameters
	weight_matrix = weight_matrix[::2]

	part_3 = calcGMatrix(x, weight_matrix, hidden_nodes)

	# Expectation: Covariance Matrix of (m,n) = (m,m)
	G = np.cov(part_3)

	# Gram Matrix for other Hidden Layers
	for weights in weight_matrix[1:]:
		
		# ith Layer Weight Matrix (m,m) * Previous Layer (T) (m,m) = (m,m)
		part_1 = weights.T.dot(G.T)

		# ReLU Activation: Max[Zero Matrix(m,m), Sign of Part 2 (m,m)] = (m,m)
		part_2 = np.zeros(part_1.shape)
		part_3 = np.maximum(part_2,np.sign(part_1))

		# Expectation: Covariance Matrix of (m,m) = (m,m)
		G = np.cov(part_3)

	# Gram Matrix: Previous Covariance Matrix * [Input Data (n,d) * Input Data (T) (d,n)] = (n,n)
	return x.dot(x.T)*G

def dynamicGMatrix(x, weight_matrix, hidden_nodes, L):
    # Get rid of the bias parameters
    weight_matrix = weight_matrix[::2]
    part_3 = calcGMatrix(x, weight_matrix, hidden_nodes)

    # Gram Matrix for other Hidden Layers
    i = 1

    for weights in weight_matrix[1:-1]:
        if L > 2:
            # ith Layer Weight Matrix (m,m) * Previous Layer (m,n) = (m,n)
            part_1 = weights.T.dot(part_3)

            # ReLU Activation: Max[Zero Matrix(m,n), Sign of Part 2 (m,n)] = (m,n)
            part_2 = np.zeros(part_1.shape)
            part_3 = np.maximum(part_2,np.sign(part_1))

        else:
            # ith Layer Weight Matrix (m,m) * Previous Layer (T) (m,m) = (m,m)
            part_1 = weights.T.dot(part_3.T)

            # ReLU Activation: Max[Zero Matrix(m,m), Sign of Part 2 (m,m)] = (m,m)
            part_2 = np.zeros(part_1.shape)
            part_3 = np.maximum(part_2,np.sign(part_1))

        i += 1

    # Final Gram Matrix:
    #    [Input Data (n,d) * Input Data (T) (d,n)]
    #        * Sum of Previous Layer ReLU Outputs (1) * 1/Hidden Nodes (1) = (n,n)
    return x.dot(x.T)*(np.sum(part_3))/(hidden_nodes*L)