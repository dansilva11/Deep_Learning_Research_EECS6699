import tensorflow as tf
import numpy as numpy
import matplotlib.pyplot as plt
import numpy as np
import Data_Gen
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense
import math
n=1000
d=1000
# mnist = tf.keras.datasets.mnist
# (x_train, y_train),(x_test, y_test) = mnist.load_data()
X,Y=Data_Gen.synthesize(n,d)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0)

# x_test = X[sample_size+1:sample_size+1+test_size]
# y_test = Y[sample_size+1:sample_size+1+test_size]
weight_counts=[50000,100000,200000,400000,800000]
VCdims=[]
for m in weight_counts:
    VCdims.append(m*2*math.log(m))
depths = [2,3,4,5,6]
# f, ax = plt.subplots(2,2, sharex='col')
fig = plt.figure()
x=1
for L in depths:
    for VC in VCdims:
        W = 0

        while True:
            W+=.1
            error=VC-W*L*math.log(W)
            if error<1:
                break
        W=round(W)
        model = Sequential()
        if L==2:
            w_i=W/(d+1)
        else:
            w_i=-(math.sqrt(d**2+2*d+4*(L-2)*W+1)+d+1)/(4-2*L)
            # q=(L-2)*(w_i**2)+(d+1)*w_i
        w_i=round(w_i)
        model = Sequential()
        # Hidden Layers
        model.add(Dense(w_i, activation='relu', kernel_initializer='RandomUniform', input_dim=d))
        for i in range(0,L-2):
            #Hidden Layers
            model.add(Dense(w_i, activation='relu', kernel_initializer='RandomUniform'))

        #Output Layer
        model.add(Dense(1, activation='relu', kernel_initializer='RandomUniform'))

        # model = tf.keras.models.Sequential([layer0, layer1, layer3])
        model.count_params()
        model.summary()
        model.compile(optimizer='sgd', loss='mean_squared_error',metrics=['binary_accuracy'])

        epochs=150
        # Train the model
        history = model.fit(x_train, y_train, epochs=epochs)
        # loss, acc = model.evaluate(x_test, y_test)
        # print("loss=", loss, "acc=", acc)


        plt.subplot(2,len(depths),x)
        plt.plot(history.history['loss'],label='VCdim='+str(VC))
        plt.title('Layer Depth = '+str(L))
        if x==1:
            plt.legend()
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.subplot(2, len(depths), x+len(depths))
        plt.plot(history.history['binary_accuracy'])
        # plt.legend()
        plt.ylabel('accuracy')
        plt.xlabel('epoch')

    x += 1



# plt.title('model loss')
# ax[0].legend()
# label=['loss','acc']
# i=0
# for x in ax.flat:
#     x.set(ylabel=label[i])
#     i+=1
#
# plt.xlabel('epoch')

plt.show()