import matplotlib.pyplot as plt
import numpy as np
import math, random
from mpl_toolkits.mplot3d import Axes3D

# Generate random vectors with d components selected from uniform distribution 
#   between -1 and 1 keep the first n vectors with magnitude <= 1
def synthesize(n, d):
    X = np.random.randn(n, d)
    Y = np.zeros(n)

    for i in range(0,len(X)):
        # Define Vector Magnitude from Standard Gaussian
        mag = random.uniform(0, 1)

        # For each sample, divide by L-2 Norm
        X[i] = X[i] / np.linalg.norm(X[i])

        # Project sample into d-dimensional space
        X[i] = X[i]*mag**(1/d)
        
        # Assign labels (y) from Standard Gaussian
        y = round(np.random.uniform(0, 1))
        Y[i]=y

    cdict = {1: 'red', 0: 'blue'}

    # Data Visualization
    # For 3D plot (dont forget to set d=3)
    # if d==3:
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     for g in np.unique(Y):
    #         ix = np.where(Y == g)
    #         ax.scatter(X[ix,0],X[ix,1],X[ix,2],s=2,c=cdict[g],label='y= '+str(int(g)))
    #     plt.title(r'$y_i=sgn(sin(6\pi||\bar{x}_i||))$ with Gaussian Noise ($\sigma^2$='+str(noise_variance)+')')
    #     plt.xlabel('$x_1$')
    #     plt.ylabel('$x_2$')
    #     plt.ylabel('$x_3$')
    #     plt.legend()
    #     plt.show()
    # For 2D plot
    # elif d==2:
    #     fig, ax = plt.subplots()
    #     for g in np.unique(Y):
    #         ix = np.where(Y == g)
    #         ax.scatter(X[ix,0],X[ix,1],s=2,c=cdict[g],label='y= '+str(g))
    #     plt.title(r'$y_i=sgn(sin(6\pi||\bar{x}_i||))$ with Gaussian Noise ($\sigma^2$='+str(noise_variance)+')')
    #     plt.xlabel('$x_1$')
    #     plt.ylabel('$x_2$')
    #     plt.legend()
    #     plt.show()
    #
    # fig, ax = plt.subplots()
    # cm = plt.cm.get_cmap('RdYlBu')
    # sc = ax.scatter(X[:, 0], X[:, 1], s=2, c=Y, cmap=cm)
    # clb = plt.colorbar(sc)
    # clb.set_label('y',rotation=0)
    # plt.xlabel('$x_1$')
    # plt.ylabel('$x_2$')
    # plt.show()

    return X, Y