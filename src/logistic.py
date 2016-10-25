import numpy as np
import csv
from sklearn import svm, linear_model, naive_bayes 
import scipy.misc # to visualize only
from scipy.ndimage.interpolation import zoom

def generate_Y():
	f1='../data/train_y.csv'
	new_rows = []
	i=0
	with open(f1, 'rb') as f:
		reader = csv.reader(f)
		for row in reader:
			new_rows.append(row)
			print i
			i=i+1
			#
	global Y_cate
	Y_cate = new_rows[0]
	del new_rows[0]
	Y = np.array(new_rows)
	Y=Y.astype(int)
	Y=Y[:,1]
	return Y

def accuracy(gold, predict):
	assert len(gold) == len(predict)
	corr = 0
	for i in xrange(len(gold)):
		if int(gold[i]) == int(predict[i]):
			corr += 1
	acc = float(corr) / len(gold)
	print 'Accuracy %d / %d = %.4f' % (corr, len(gold), acc)
	return acc

def small(a):
	hold = 200;
	small_a = zoom(a, 0.5)
	#for i,r in enumerate(small_a):
	#	for j,c in enumerate(r):
	#		if(c<hold):
	#			small_a[i,j]=0;
	return small_a

def write_predictions(Y_te,txt):
	Yw=[]
	for i,p in enumerate(Y_te):
		Yw.append([i,Y_te[i]])
	#
	with open(txt, 'wb') as csvfile:
		spamwriter = csv.writer(csvfile)
		spamwriter.writerow(['Id','Prediction'])
		spamwriter.writerows(Yw)


'''
X_tr = np.fromfile('../data/train_x.bin', dtype='uint8')
X_tr = X_tr.reshape((100000,60,60))
#scipy.misc.imshow(X_tr[1]) # to visualize only
X = np.zeros((100000,30,30))
for i,img in enumerate(X_tr):
	X[i] = small(img)
	print i

Y_tr = generate_Y()
X=X.reshape((100000,900))
logistic = linear_model.LogisticRegression(C=1e5)
logistic.fit(X, Y_tr)

import pickle
with open('../data/logi0-100000', 'wb') as output:
	pickle.dump(logistic, output, pickle.HIGHEST_PROTOCOL)

with open('../data/logi0-100000', 'rb') as input:
	logistic=pickle.load(input)

X_te = np.fromfile('../data/test_x.bin', dtype='uint8')

X_te = X_te.reshape((20000,60,60))
#scipy.misc.imshow(X_tr[1]) # to visualize only
X = np.zeros((20000,30,30))
for i,img in enumerate(X_te):
	X[i] = small(img)
	print i

X=X.reshape((20000,900))

Y_te=logistic.predict(X)

write_predictions(Y_te,"../data/logistic_half.csv")










a = X_tr[8].copy()
a = a[5:55,5:55]
scipy.misc.imshow(a)

hold = 200;
small_a = zoom(X_tr[6], 0.5)
for i,r in enumerate(small_a):
	for j,c in enumerate(r):
		if(c<hold):
			small_a[i,j]=0;


scipy.misc.imshow(small_a)


Y_tr = generate_Y()
logistic = linear_model.LogisticRegression(C=1e5)
logistic.fit(X_tr, Y_tr)

import pickle
with open('../data/logi0-100000', 'wb') as output:
	pickle.dump(logistic, output, pickle.HIGHEST_PROTOCOL)

with open('../data/logi0-100000', 'rb') as input:
	Y_te=pickle.load(input)


accuracy(logistic.predict(X_te),Y_te)

import Image
import numpy as np

image=X[2]
image = small(X[2])
hold = 200;
for i,r in enumerate(image):
	for j,c in enumerate(r):
		if(c<hold):
			image[i,j]=0;


image_data_bw = image_data.max(axis=2)
non_empty_columns = np.where(image_data_bw.max(axis=0)>0)[0]
non_empty_rows = np.where(image_data_bw.max(axis=1)>0)[0]
cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))

image_data_new = image_data[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1 , :]

new_image = Image.fromarray(image_data_new)
new_image.save('../data/L_2d_cropped.png')
'''