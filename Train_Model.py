import tensorflow as tf
import numpy as numpy
import matplotlib.pyplot as plt
import numpy as np
import Data_Gen

# mnist = tf.keras.datasets.mnist
# (x_train, y_train),(x_test, y_test) = mnist.load_data()
X,Y=Data_Gen.synthesize(n=10000,d=1000,noise_variance=0)
sample_size = 1000
test_size = sample_size / 10.0
test_size = int(test_size)

x_train = X[:sample_size]
y_train = Y[:sample_size]

x_test = X[sample_size+1:sample_size+1+test_size]
y_test = Y[sample_size+1:sample_size+1+test_size]

layer0 = tf.keras.layers.Flatten(input_shape=np.shape(x_train[0]))
layer1 = tf.keras.layers.Dense(32768, activation = tf.nn.relu)
layer2 = tf.keras.layers.Dense(1000, activation = tf.nn.relu)
layer3 = tf.keras.layers.Dense(len(np.unique(y_train)), activation = tf.nn.sigmoid)

model = tf.keras.models.Sequential([layer0, layer1, layer2, layer3])
model.count_params()
model.summary()
model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)
loss, acc = model.evaluate(x_test, y_test)
print("loss=", loss, "acc=", acc)