import tensorflow as tf
import numpy as numpy
import matplotlib.pyplot as plt
import numpy as np
import Data_Gen
import Def_Model
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense
import math

# References
# [1] Bartlett et al. ”Nearly-tight VC-dimension and pseudodimension bounds for piecewise Linear NNs”, 
#     Conference  Paper  at  Conference  on  Learning  Theory  2017,  October 2017.

def modelsInit(x_train, y_train, weight_counts, depths, VCdims, n, d, epochs):

	# Setup Plot
	fig = plt.figure()
	x = 1
	
	# Calculate VC-Dim for each weight count
	VCdims=[]
	for m in weight_counts:
		# VC-Dim = weight count x 2 x log (weight count)
	    VCdims.append(m*2*math.log(m))

    # Generate convergence rate graph for each depth/VC-dim combination
    for L in depths:
        for VC in VCdims:
	        Def_Model.buildSubModel(x_train, y_train, L, VC, d, len(depths), x, epochs)
	    x += 1

	plt.show()

def main():
    # n - Number of Samples
    n = 1000

    # d - Dimensions
    d = 1000

	# Training Epochs
    epochs = 50

    # X - Sample Set from a d-Dimension hypersphere
    # Y - Label Set randomly generated from a Standard Gaussian Dist
    X,Y = Data_Gen.synthesize(n,d,5)

    # Split-up sample set into train/test sets
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0)

    # Setup model attributes - A
    weight_counts = [50000,100000,200000,400000,800000]
    depths = [2,3,4,5,6]

    # Calculate VC-Dim for each weight count
    VCdims=[]
    for m in weight_counts:
        # VC-Dim = weight count x 2 x log (weight count)
        VCdims.append(m*2*math.log(m))

    modelsInit(x_train, y_train, weight_counts, depths, VCdims, n, d)

    # Setup model attributes - B
	L = 10
	W = 50000

	# Setup list of weighted \bar{L}
	LBars = [round((L/2)+1), L, 1, round((L/2)+1)+1]
	
	for LBar in LBars:
		Def_Model.customWeights(L,W,d,Lbar=LBar,pace=5)

if __name__ == '__main__':
    main()