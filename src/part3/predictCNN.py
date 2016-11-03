import numpy as np
from nolearn.lasagne import NeuralNet
import dill
import sys
sys.setrecursionlimit(20000)

data = np.fromfile('test_x.bin', dtype='uint8')
data = data.reshape((20000,60,60))

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

cnn = None
with open('cnn_preliminary-2.pkl', 'rb') as f:
    cnn = dill.load(f)

prediction = cnn.predict(data)

with open('predictions.csv', 'w') as f:
    f.write('Id,Prediction\n')
    for i in range(len(prediction)):
        f.write(str(i) + ',' + str(prediction[i]) + '\n')