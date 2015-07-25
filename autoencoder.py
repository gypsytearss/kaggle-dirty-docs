from __future__ import absolute_import
import theano
import keras
from keras.layers import containers
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import numpy as np
from scipy import misc
import os
import load_data

encoder = containers.Sequential([Dense(540*420, 270*210), Dense(270*210, 135*105)])
decoder = containers.Sequential([Dense(135*105, 270*210), Dense(270*210, 540*420)])

autoencoder = Sequential()
autoencoder.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=True))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.0, nesterov=True)
autoencoder.compile(loss='categorical_crossentropy', optimizer=sgd)

batch_size = 12
nb_epoch = 20

data, X_test = load_data()
X_train, Y_train = data[0][:140,:], data[1][:140,:]
X_test, Y_test = data[0][141,:], data[1][:141,:]

autoencoder.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch)

Y_test = autoencoder.predict_classes(X_test, batch_size=1, verbose=True)
Y_test = Y_test.reshape((420,540))
print Y_test.tolist()

misc.imsave('nudie.png', Y_test)