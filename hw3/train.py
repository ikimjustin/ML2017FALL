'''
from keras.models import Sequential
model = Sequential()

from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as k
print(k.image_dim_ordering())
k.set_image_dim_ordering('th')

model.add(Convolution2D( 30,3,3,input_shape=(1,48,48)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))
model.add(Convolution2D( 60,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Convolution2D( 60,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(output_dim=500))
model.add(Activation("sigmoid"))
#model.add(Dropout(0.5))
model.add(Dense(output_dim=500))
model.add(Activation("sigmoid"))
#model.add(Dense(output_dim=1000))
#model.add(Activation("sigmoid"))
model.add(Dense(output_dim=7))
model.add(Activation("softmax"))


#from keras.optimizers import SGD
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01,momentum=0.9,nesterov=True))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
'''
#!/usr/bin/env python
# -- coding: utf-8 --
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta , Adagrad
from keras.callbacks import EarlyStopping
from keras.models import load_model

input_img = Input(shape=(48, 48, 1))

block1 = Conv2D(64, (5, 5), padding='valid', activation='relu')(input_img)
block1 = ZeroPadding2D(padding=(2, 2), data_format='channels_last')(block1)
block1 = MaxPooling2D(pool_size=(5, 5), strides=(2, 2))(block1)
block1 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block1)

block2 = Conv2D(64, (3, 3), activation='relu')(block1)
block2 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block2)

block3 = Conv2D(64, (3, 3), activation='relu')(block2)
block3 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block3)
block3 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block3)

block4 = Conv2D(128, (3, 3), activation='relu')(block3)
block4 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block4)

block5 = Conv2D(128, (3, 3), activation='relu')(block4)
block5 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block5)
block5 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block5)
block5 = Flatten()(block5)

fc1 = Dense(1024, activation='relu')(block5)
fc1 = Dropout(0.5)(fc1)

fc2 = Dense(1024, activation='relu')(fc1)
fc2 = Dropout(0.5)(fc2)



predict = Dense(7)(fc2)
predict = Activation('softmax')(predict)
model = Model(inputs=input_img, outputs=predict)

    # opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # opt = Adam(lr=1e-3)
#opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#opt = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08) ###modified!!!
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()


model = load_model('model06.h5')

import csv
import numpy as np
import sys
# import training data
# image resolution is 48*48
file=open(sys.argv[1],'r')
train_X=[]
train_Y=[]
line_number = 0
for line in file:
	if (line_number > 0) :
		#train_Y.append(line[1])
		temp=line.strip('\n').split(',')
		#temp1 = temp[0]
		output = [0,0,0,0,0,0,0]
		output[int(temp[0])] = 1
		train_Y.append(output)
		#temp2 = temp[1].split(' ')
		train_X.append(temp[1].split(' '))
		#train_X.append(line[2])
	line_number+=1
file.close()
train_X = np.asarray(train_X)
train_X = train_X.astype(np.float)
train_X = np.reshape(train_X,(len(train_Y),48,48,1))
train_Y = np.asarray(train_Y)
#train_Y = train_Y.astype(np.int)






# training model

earlystop = EarlyStopping(monitor='valid_acc', min_delta=0.0, patience=6, verbose=0, mode='auto')
model.fit(train_X,train_Y, nb_epoch=50,batch_size=128,validation_split=0.05,shuffle=True,callbacks=[earlystop])#35,50

#loss_and_metrics = model.evaluate(X_train, Y_train, batch_size=20)validation_split=0.10,shuffle=True
#classed = model.predict_classes(X_train, batch_size=20)
#proba = model.predict_proba(X_train, batch_size=20)

#print('Accuracy of Testing Set:', loss_and_metrics[1] )

from keras.models import load_model
model.save('model09.h5')
#model.save('model'+i+'.h5')
