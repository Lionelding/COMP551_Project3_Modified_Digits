#http://blog.christianperone.com/2015/08/convolutional-neural-networks-and-feature-extraction-with-python/

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from urllib import urlretrieve
import cPickle as pickle
import os
import gzip
import numpy as np
import theano
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import dill
import sys
sys.setrecursionlimit(20000)

def load_dataset():    
    data = np.fromfile('train_x.bin', dtype='uint8')
    data = data.reshape((100000,60,60))
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                data[i][j][k] = 0 if data[i][j][k] < 255 else 1
    
    data = data.astype(np.float32)
    
    fittedData = np.zeros((data.shape[0], 28, 28))
    for i in range(data.shape[0]):
        temp = np.zeros((30,30))
        for j in range(temp.shape[0]):
            for k in range(temp.shape[1]):
                temp[j][k] = (data[i][j*2][k*2] + data[i][j*2][k*2+1] + data[i][j*2+1][k*2] + data[i][j*2+1][k*2+1]) / 4
        
        fittedData[i] = temp[1:29,1:29]
    
    data = fittedData.reshape((-1,1,28,28))
    
    y = []
    with open('train_y.csv', 'r') as f:
        y = f.readlines()[1:]
    for i in range(len(y)):
        y[i] = int(y[i].split(',')[1])
    y = np.asarray(y)
    
    X_train = data[:95000]
    X_test = data[95000:]
    y_train = y[:95000]
    y_test = y[95000:]
    return X_train, y_train, X_test, y_test
    
X_train, y_train, X_test, y_test = load_dataset()

net1 = NeuralNet(
    layers=[('input', layers.InputLayer),
            ('conv2d1', layers.Conv2DLayer),
            ('maxpool1', layers.MaxPool2DLayer),
            ('conv2d2', layers.Conv2DLayer),
            #('maxpool2', layers.MaxPool2DLayer),
            ('dropout1', layers.DropoutLayer),
            ('dense', layers.DenseLayer),
            #('dropout2', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
    # input layer
    input_shape=(None, 1, 28, 28),
    # layer conv2d1
    conv2d1_num_filters=32,
    conv2d1_filter_size=(5, 5),
    conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d1_W=lasagne.init.GlorotUniform(),  
    # layer maxpool1
    maxpool1_pool_size=(2, 2), 
    # layer conv2d2
    conv2d2_num_filters=144,
    conv2d2_filter_size=(5, 5),
    conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
    # layer maxpool2
#    maxpool2_pool_size=(2, 2),
    # dropout1
    dropout1_p=0.25,    
    # dense
    dense_num_units=256,
    dense_nonlinearity=lasagne.nonlinearities.rectify,    
    # dropout2
#    dropout2_p=0.25,    
    # output
    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=19,
    # optimization method params
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,
    max_epochs=50,
    verbose=1,
    )
# Train the network
nn = net1.fit(X_train, y_train)

with open('cnn.pkl', 'wb') as f:
    dill.dump(nn, f)

preds = net1.predict(X_test)

cm = confusion_matrix(y_test, preds)
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()