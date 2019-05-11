import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Custom Modules
from Data_Gen import synthesize
from Def_Model import buildSubModel, buildCustomModel, initParams
from Def_Lbar import customWeights
from Def_GMatrix import calcGMatrix, calcLambdaMin

# References
# [1] Bartlett et al. ”Nearly-tight VC-dimension and pseudodimension bounds for piecewise 
#     Linear NNs”, Conference Paper at Conference on Learning Theory 2017, October 2017.
# [2] Weinan et al. " A Comparative Analysis of the Optimization and Generalization Property
#	  of Two-layer Neural Network and Random Feature Models Under Gradient Descent Dynamics"
#	  April 2019.

# -- Variable Definitions -- 
# 	 d = Input Data Dimensions
#    L = Network Depth (# layers, excl. Input Layer)
# 	 n = # Samples
#    epochs = # of epochs to train each network 
#    depths = List of depths to attempt in phase 2 & 3 
#    VC = Network VC Dimension
#    W = # Parameters
#	 X - Sample set from a d-Dimension hypersphere
#	 Y - Label set randomly generated from a Std Gaussian Dist
#    G = (Square) Gram Matrix of d x d dimensions
#    lambda_min = Smallest, real eigenvalue of Gram Matrix

# Creates models and plots performance based on weight count and depth input lists
def main(n=1000, d=10, epochs=500, depths=[2]):
    # Setup network/training variables
    X,Y = synthesize(n, d)

    # Split-up sample set into train/test sets
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0)

    ####need to edit this comment with final setting###
    # Setup Weight Counts: # Hidden nodes (m) =
    #	(1) 5x # Samples
    #	(2) 10 x # Samples (n)
    #	(3) (# Samples (n))^2 (Source: [2])
    # param_counts = [n, n**1.25, n**1.5, n**1.6, n**1.65, n**1.7, n**1.75,n**1.8, n**1.85, n**1.9, n**1.95, n**2]
    param_counts = [n, n**1.25, n**1.5]

    # Setup plot/counter
    fig = plt.figure()
    plts = len(depths) # total number of sub-plots 
    
    # ########################### PHASE 1 ###########################
    # #   Intention is to use a 2 layer network and find the minimal 
    # #       number of nodes (and corresponding minimal VCDim) 
    # #       to capture reasonable overparameterization benefits 
    # ###############################################################
    
    x = 1 # Phase sub-plot position out of 3
    
    # Calculate VC-Dim for each weight count for a 2 layer network these input VC dimensions will be held constant
    # as L is increased
    VCdims = []
    for W in param_counts:
        VCdims.append(W * 2 * math.log(W))

    # Create convergence rate graph
    loss_histories = list()
    
    L = 2 # Fix depth L = 2
    # Pass through each VCDim, construct the corresponding network, and train 
    for i in range(0,len(VCdims)):
        VC = VCdims[i]
        W = param_counts[i]
        loss_histories.append(buildSubModel(x_train, y_train, L, VC, d, plts, x, epochs, W))

    # Show phase sub-plot 
    plt.show()

    # ########################### PHASE 2 ###########################
    # #   Intention is use a fixed minimal VCDim from Phase 1 and  
    # #       introduce depth. With fixed VCDim we will construct 
    # #       an equally weighted CNN and observe training 
    # #       acceleration at each step. 
    # ###############################################################
    
    # z = 2 #phase sub-plot position out of 3
    
    # # Assume you have a way to pick some minimal VCDim
    # VCstar = max(VCdims)
    
    # # Create convergence rate graph 
    # depth_histories = list()
    
    # # Pass through each depth in depths, construct corresponding network, and train 
    # for L in depths:
    #     Wd = initParams(VCstar,L)
    #     depth_histories.append(buildSubModel(x_train, y_train ,L, VCstar, d, plts, z, epochs, Wd))
    
    # # Show phase sub-plot 
    # plt.show()
    
    # ########################### PHASE 3 ###########################
    # #   Intention is use a fixed maximal depth L from Phase 2 and
    # #       vary the position of weights. VCDim will remain fixed 
    # #       from Phase 1. Variation will concentrate weights in a 
    # #       single layer with minimal number of nodes (minw) in all
    # #       other layers. This contrasts with Phase 2's equally 
    # #       weighted initialization. 
    # ###############################################################
    
    # c = 3 #phase sub-plot position out of 3
    
    # # Assume you have a way to pick some maximal L 
    # L = max(depths)
    
    # # Create convergence rate graph 
    # lbar_histories = list()
    
    # # Pass through each concentrated layer cL, construct corresponding network, and train 
    # for cL in range(1+L):
    #     if(cL>0): #cL starts from index 1
    #         lbar_histories.append(buildCustomModel(x_train, y_train, L, VCstar, d, cL, plts, c, epochs, 5))
    
    # # Show phase sub-plot 
    # plt.show()
    
    return ;

if __name__ == '__main__':
	main(n=1000,d=10,epochs=2000,depths=[2,3])
    
    