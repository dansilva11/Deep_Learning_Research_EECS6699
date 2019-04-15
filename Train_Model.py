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

fig = plt.figure()

n=1000
d=1000
epochs=50

X,Y=Data_Gen.synthesize(n,d)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0)

weight_counts=[50000]
VCdims=[]
for m in weight_counts:
    VCdims.append(m*2*math.log(m))
depths = [2,3]


his_vec = list()
x=1
for L in depths:
    for VC in VCdims:
        his_vec.append(Def_Model.buildSubModel(x_train,y_train,L,VC,d,len(depths),x,epochs))
    x += 1

plt.show()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#find optimal VC and layers for our analysis (lowest # of nodes with good accuracy)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Pass weight concentration through the layers 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

L = 10
W = 50000
#equally distributed
Def_Model.customWeights(L,W,d,Lbar=round((L/2)+1),pace=5)
#other settings
Def_Model.customWeights(L,W,d,Lbar=L,pace=5)
Def_Model.customWeights(L,W,d,Lbar=1,pace=5)
Def_Model.customWeights(L,W,d,Lbar=round((L/2)+1)+1,pace=5)

