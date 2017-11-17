#from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as k
#print(k.image_dim_ordering())
k.set_image_dim_ordering('th')
from keras.models import load_model
import sys
print ("loading model...")
model = load_model('model08.h5')
#model = load_model(sys.argv[2])

'''
#string = sys.argv[1] + 'test.p'
string = 'test.csv' 
print ("loading testing data....")
test = pickle.load(open(string,"r"))
#print(len(test))
ID = test['ID']
X_test = test['data']
X_test = numpy.asarray(X_test)
X_test = X_test.reshape(10000,48,48,1)
'''


import csv
import numpy as np
# import testing data
# image resolution is 48*48
file=open(sys.argv[1],'r')
test_X=[]
test_Y=[]
line_number = 0
for line in file:
	if (line_number > 0) :
		#train_Y.append(line[1])
		temp=line.strip('\n').split(',')
		#temp1 = temp[0]
		#output = [0,0,0,0,0,0,0]
		#output[int(temp[0])] = 1
		test_Y.append(temp[0])
		#temp2 = temp[1].split(' ')
		test_X.append(temp[1].split(' '))
		#train_X.append(line[2])
	line_number+=1
file.close()
test_X = np.asarray(test_X)
test_X = test_X.astype(np.float)
test_X = np.reshape(test_X,(len(test_Y),48,48,1))
test_Y = np.asarray(test_Y)
#train_Y = train_Y.astype(np.int)

#Y_test=model.predict_classes(test_X,batch_size=20)
y_proba = model.predict(test_X,batch_size=20)
Y_test = y_proba.argmax(axis=-1)
Y_test.shape=(len(test_Y),1)

#file=open(sys.argv[3],'w')
file=open(sys.argv[2],'w')
csv.writer(file).writerow(['id','label'])
for i in range(len(test_Y)):
    csv.writer(file).writerow([str(i),Y_test[i][0]])
file.close()
