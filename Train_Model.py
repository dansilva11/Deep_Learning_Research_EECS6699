import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split

# Custom Modules
from Data_Gen import synthesize
from Def_Model import buildSubModel
from Def_Lbar import customWeights

# References
# [1] Bartlett et al. ”Nearly-tight VC-dimension and pseudodimension bounds for piecewise 
#     Linear NNs”, Conference Paper at Conference on Learning Theory 2017, October 2017.
# [2] Weinan et al. " A Comparative Analysis of the Optimization and Generalization Property
#	  of Two-layer Neural Network and Random Feature Models Under Gradient Descent Dynamics"
#	  April 2019.

# -- Variable Definitions -- 
# 	d = Input Data Dimensions
#   L = Network Depth (# layers, excl. Input Layer)
# 	n = # Samples
#   VC = Network VC Dimension
#   W = # Parameters
#	X - Sample set from a d-Dimension hypersphere
#	Y - Label set randomly generated from a Std Gaussian Dist

# Creates models and plots performance based on weight count and depth input lists
def main():
	# Setup network/training variables
	n = 1000
	d = 1000
	epochs = 50
	X,Y = synthesize(n, d, 5)

	# Split-up sample set into train/test sets
	x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0)

	# Define target depths
	depths = [2,3,4,5]

	# Setup Weight Counts: # Hidden nodes (m) = 
	#	(1) 5x # Samples
	#	(2) 10 x # Samples (n)
	#	(3) (# Samples (n))^2 (Source: [2])
	weight_counts = [5*n, 10*n,n**2]

	# Setup plot/counter
	fig = plt.figure()
	x = 1

	# Create convergence rate graph
	loss_histories = list()
	for L in depths:
		# Calculate VC-Dim for each weight count
		VCdims=[]
		for W in weight_counts:
			VCdims.append(W*L*math.log(W))

		for VC in VCdims:
			loss_histories.append(buildSubModel(x_train, y_train, L, VC, d, len(depths), x, epochs))
		x += 1

	plt.show()

	# [In Progress] Setup variables for LBar calculations
	# L = 3
	# W = 50000
	# LBars = [round((L/2)+1), L, 1, round((L/2)+1)+1]
	
	# for LBar in LBars:
	# 	customWeights(L, W, d, Lbar=LBar, pace=5)

if __name__ == '__main__':
	main()