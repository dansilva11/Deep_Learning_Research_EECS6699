import matplotlib.pyplot as plt
import numpy as np
import math
import random
from mpl_toolkits.mplot3d import Axes3D
def synthesize(n,d):
    """
    generate random vectors with d components selected from uniform distribution between -1 and 1
    keep the first n vectors with magnitude <= 1
    """
    # X=np.empty((0,d))
    # while len(X)<n:
    #     x=np.random.rand(1,d)*2-1
    #     if np.linalg.norm(x)<=1:
    #         X=np.vstack((X, x))
    #         print(len(X))
    X = np.random.randn(n, d)


    Y=np.zeros(n)

    # sigma=math.sqrt(noise_variance)
    for i in range(0,len(X)):
        mag = random.uniform(0, 1)
        X[i] = X[i] / np.linalg.norm(X[i])
        X[i] = X[i]*mag**(1/d)
        #assign y labels
        y = round(np.random.uniform(0, 1))
        # y=np.random.normal(math.sin(np.linalg.norm(X[i])*6*math.pi), sigma)
        # if y>0: y=1
        # else:   y=0
        Y[i]=y

    cdict = {1: 'red', 0: 'blue'}

    #For 3D plot (dont forget to set d=3)
    if d==3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for g in np.unique(Y):
            ix = np.where(Y == g)
            ax.scatter(X[ix,0],X[ix,1],X[ix,2],s=2,c=cdict[g],label='y= '+str(int(g)))
        plt.title(r'$y_i=sgn(sin(6\pi||\bar{x}_i||))$ with Gaussian Noise ($\sigma^2$='+str(noise_variance)+')')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.ylabel('$x_3$')
        plt.legend()
        plt.show()

    #For 2D plot
    if d==2:
        fig, ax = plt.subplots()
        cm = plt.cm.get_cmap('RdBu')
        # ax.scatter(X[:, 0], X[:, 1], s=2, c=Y, label='y', cmap=cm)
        for g in np.unique(Y):
            ix = np.where(Y == g)
            ax.scatter(X[ix,0],X[ix,1],s=2,c=cdict[g],label='y= '+str(g),cmap=cm)
        # plt.title(r'$y_i=sgn(sin(6\pi||\bar{x}_i||))$ with Gaussian Noise ($\sigma^2$='+str(noise_variance)+')')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.legend()
        plt.show()



    return X, Y

# synthesize(n=10000,d=2,noise_variance=5)