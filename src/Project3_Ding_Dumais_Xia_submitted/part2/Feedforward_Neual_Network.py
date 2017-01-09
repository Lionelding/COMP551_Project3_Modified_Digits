from matplotlib import pylab as plt
from time import time
import scipy.misc
import numpy as np
import csv
from neuron import Neuron
from layer import Layer
from network import Network
########################################################################################
# Set the number of training and testing examples
train_num=5
testing_num=5

########################################################################################
## Load the train_x to a List
print "_"*100
print "------Loading Data-----"
print ''
print 'Loading the train_x'
t0 = time()
#train_in = np.fromfile("train_x.bin", dtype='uint8')
train_in = np.fromfile("train_in_modified3000_cropped.bin", dtype='uint8')
print np.shape(train_in)
train_in = train_in.reshape((3000,50,50))
#scipy.misc.imshow(train_x[1]) 

print 'The total size of train_in is : '+str(np.shape(train_in))
print 'The shape of train_in is : '+str(np.shape(train_in[1]))
print 'The image storage type is : '+str(type(train_in[1]))
duration = time() - t0
print("done in %fs " % (duration))
print ''

print 'Loading the train_y'
## Load the train_out to a List
t0 = time()
with open('train_y.csv', 'rb') as b:
    reader2 = csv.reader(b)
    your_list2 = list(reader2)
train_out_num=len(your_list2)
goodtarget=[]
for i in range(0,train_out_num):
	goodtarget.extend(your_list2[i])
print "The shape of train_out is : "+str(np.shape(goodtarget))
print 'Number of training_out results is: ' +str(train_out_num)
duration = time() - t0
print("done in %fs " % (duration))
print ''
print "_"*100

########################################################################################
# Define variables 
# Two input and ouput for testing 
Input=[0.05,0.1]
target=[0.01, 0.99]

# Hyper-parameter
hidden_num=1
hidden_neuron_num=500
outer_neuron_num=1
hidden_weight=[0.003]*2500*hidden_neuron_num
outer_weight=[0.003]*hidden_neuron_num
L_rate=0.05
thres=1
iteration=5


########################################################################################

newFileBytes = []


########################################################################################
Test_network=Network(thres, L_rate, hidden_num, hidden_neuron_num, outer_neuron_num, hidden_weight, outer_weight )

for j in range(0, train_num):

	temp_target=[]
	temp_target.append(int(goodtarget[j])/10.0)
	print "Training on image : "+str(j)

	print 'The image ' +str(j)+' shape : '+str(np.shape(train_in[j]))
	temp_train=train_in[j].reshape(2500,1)

	for i in range(0, iteration):
		
		Test_network.Network_backward(temp_train, temp_target)
		print(i, round(Test_network.get_Network_error(temp_target),9))
	print ''
	print '_'*100

# temp_train=[]
# train_in=train_in[1].reshape(3600,1)

# for i in range (0, 3600):
# 	if train_in[i]==255:
# 		train_in[i]=1

# for i in range (0, 3600):
# 	temp_train.extend(train_in[i])

# print np.shape(temp_train)

# temp_target=[0.7]
# for i in range(0, iteration):
# 	Test_network.train(temp_train, temp_target)
# 	print(i, round(Test_network.get_Network_error(temp_target),9))
# print ''
# print '_'*100

########################################################################################
# Test the system
overall_error=0.0
for i in range(0, testing_num):

	temp_target=[]
	temp_target.append(int(goodtarget[i])/10.0)

	temp_train=train_in[j].reshape(2500,1)
	overall_error=Test_network.test(temp_train, temp_target[0])+overall_error

overall_error=overall_error/testing_num
print 'The Overal Error Rate is : ' +str(overall_error)








