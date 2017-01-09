#https://triangleinequality.wordpress.com/2014/08/12/theano-autoencoders-and-mnist/

import numpy as np
import theano as th
from theano import tensor as T
from numpy import random as rng
 
class AutoEncoder(object):
    def __init__(self, X, hidden_size, activation_function,
                 output_function):
        #X is the data, an m x n numpy matrix
        #where rows correspond to datapoints
        #and columns correspond to features.
        assert type(X) is np.ndarray
        assert len(X.shape)==2
        self.X=X
        self.X=th.shared(name='X', value=np.asarray(self.X, 
                         dtype=th.config.floatX),borrow=True)
        #The config.floatX and borrow=True stuff is to get this to run
        #fast on the gpu. I recommend just doing this without thinking about
        #it until you understand the code as a whole, then learning more
        #about gpus and theano.
        self.n = X.shape[1]
        self.m = X.shape[0]
        #Hidden_size is the number of neurons in the hidden layer, an int.
        assert type(hidden_size) is int
        assert hidden_size > 0
        self.hidden_size=hidden_size
        initial_W = np.asarray(rng.uniform(
                 low=-4 * np.sqrt(6. / (self.hidden_size + self.n)),
                 high=4 * np.sqrt(6. / (self.hidden_size + self.n)),
                 size=(self.n, self.hidden_size)), dtype=th.config.floatX)
        self.W = th.shared(value=initial_W, name='W', borrow=True)
        self.b1 = th.shared(name='b1', value=np.zeros(shape=(self.hidden_size,),
                            dtype=th.config.floatX),borrow=True)
        self.b2 = th.shared(name='b2', value=np.zeros(shape=(self.n,),
                            dtype=th.config.floatX),borrow=True)
        self.activation_function=activation_function
        self.output_function=output_function
                     
    def train(self, n_epochs=100, mini_batch_size=1, learning_rate=0.1):
        index = T.lscalar()
        x=T.matrix('x')
        params = [self.W, self.b1, self.b2]
        hidden = self.activation_function(T.dot(x, self.W)+self.b1)
        output = T.dot(hidden,T.transpose(self.W))+self.b2
        output = self.output_function(output)
         
        #Use cross-entropy loss.
        L = -T.sum(x*T.log(output) + (1-x)*T.log(1-output), axis=1)
        cost=L.mean()       
        updates=[]
         
        #Return gradient with respect to W, b1, b2.
        gparams = T.grad(cost,params)
         
        #Create a list of 2 tuples for updates.
        for param, gparam in zip(params, gparams):
            updates.append((param, param-learning_rate*gparam))
         
        #Train given a mini-batch of the data.
        train = th.function(inputs=[index], outputs=[cost], updates=updates,
                            givens={x:self.X[index:index+mini_batch_size,:]})
                             
 
        import time
        start_time = time.clock()
        for epoch in xrange(n_epochs):
            print "Epoch:",epoch
            for row in xrange(0,self.m, mini_batch_size):
                train(row)
        end_time = time.clock()
        print "Average time per epoch=", (end_time-start_time)/n_epochs
                    
    def get_hidden(self,data):
        x=T.dmatrix('x')
        hidden = self.activation_function(T.dot(x,self.W)+self.b1)
        transformed_data = th.function(inputs=[x], outputs=[hidden])
        return transformed_data(data)
     
    def get_weights(self):
        return [self.W.get_value(), self.b1.get_value(), self.b2.get_value()]
 

print 'Opening data'
x = np.fromfile('train_x.bin', dtype='uint8')
x = x.reshape((100000,3600))[:10000]

for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        x[i][j] = 0 if x[i][j] < 255 else 1

x = x.astype(np.float32)

y = []
with open('train_y.csv', 'r') as f:
    y = f.readlines()[1:]
for i in range(len(y)):
    y[i] = y[i].split(',')[1]
    
data = ((x[:8000], np.asarray(y[:8000])), (x[8000:9000], np.asarray(y[8000:9000])), (x[9000:], np.asarray(y[9000:])))

print 'Creating autoencoder'
dataAE = data[0][0]
activation_function = T.nnet.sigmoid
output_function=activation_function
AE = AutoEncoder(dataAE, 100, activation_function, output_function)
AE.train(20,200)
print 'Writing weights to file'
W=AE.get_weights()
W1 = W[0]
W2 = np.transpose(W1)
B1 = W[1]
B2 = W[2]

with open('outputfile', 'w') as f:
    for i in range(W1.shape[0]):
        f.write(str(W1[i][0]))
        for j in range(1, W1.shape[1]):
            f.write(',')
            f.write(str(W1[i][j]))
        f.write('\n')
    
    f.write('\n')    
    f.write(str(B1[0]))
    for i in range(1, B1.shape[0]):
        f.write(',')
        f.write(str(B1[i]))
    f.write('\n')
    
import matplotlib.pyplot as plt


for i in range(x.shape[0]):
    before = x[i] * 256
    for j in range(before.shape[0]):
        before[j] = max(0, before[j])
    before = before.astype(np.uint8)
    before = before.reshape(60,60)
    
    fig = plt.figure()
    a = fig.add_subplot(1,2,1)
    a.set_title('Before')
    plt.imshow(before)
    
    after = np.dot(x[i],W1) + B1
    after = np.dot(after, W2) + B2
    for j in range(after.shape[0]):
        after[j] = max(0, after[j])
    after = after.astype(np.uint8)
    after = after.reshape(60,60)
    fig.add_subplot(1,2,2)
    imgplot = plt.imshow(after)
    a.set_title('After')
    plt.show()