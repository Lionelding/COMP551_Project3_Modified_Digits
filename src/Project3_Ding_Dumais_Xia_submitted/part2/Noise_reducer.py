from matplotlib import pylab as plt
from time import time
import scipy.misc
import numpy as np
import csv
from neuron import Neuron
from layer import Layer
from network import Network

train_num=3000
testing_num=5

########################################################################################
## Load the train_x to a List
print "_"*100
print "------Loading Data-----"
print ''
print 'Loading the train_x'
t0 = time()
#train_in = np.fromfile("train_x.bin", dtype='uint8')
train_in = np.fromfile("train_in_modified3000.bin", dtype='uint8')
train_in = train_in.reshape((train_num,60,60))
#scipy.misc.imshow(train_x[1]) 

print 'The total size of train_in is : '+str(np.shape(train_in))
print 'The shape of train_in is : '+str(np.shape(train_in[1]))
print 'The image storage type is : '+str(type(train_in[1]))
duration = time() - t0
print("done in %fs " % (duration))
print ''

########################################################################################
# Take the noise off 
newFileBytes = []

# for k in range(0, train_num):
# 	for i in range(0, 60):
# 		for j in range(0, 60):
# 			if train_in[k][i,j]!=255:
# 				train_in[k][i,j]=0
# 	newFileBytes.append(train_in[k])

# #cipy.misc.imshow(train_in[5]) 

# a = np.array(newFileBytes)
# a = a.reshape((3600*train_num,1))

# print np.shape(a)
# newFileByteArray = bytearray(a)
# # make file
# newFile = open ("train_in_modified3000.bin", "wb")
# # write to file
# newFile.write(newFileByteArray)

for k in range(0, train_num):
	train_in_cropped=train_in[k][5:55,5:55]
	newFileBytes.append(train_in_cropped)

a = np.array(newFileBytes)
a = a.reshape((2500*train_num,1))
newFileByteArray = bytearray(a)
# make file
newFile = open ("train_in_modified3000_cropped.bin", "wb")
# write to file
newFile.write(newFileByteArray)