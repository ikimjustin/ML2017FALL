import sys, csv, tensorflow, pickle, keras.backend.tensorflow_backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Input, Flatten, Concatenate
from sklearn.cluster import KMeans
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.initializers import RandomNormal

#from __future__ import print_function

from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam

from sklearn.cluster import KMeans


import numpy as np


# build model
input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

# build encoder
encoder = Model(input=input_img, output=encoded)

# build autoencoder
adam = Adam(lr=1e-5)
autoencoder = Model(input=input_img, output=decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()


# load images
train_num = 130000
X = np.load(sys.argv[1])
X = X.astype('float32') / 255.
X = np.reshape(X, (len(X), -1))
x_train = X[:train_num]
x_val = X[train_num:]
x_train.shape, x_val.shape
#training code
'''
if sys.argv[1] == "train":

	# train autoencoder
	autoencoder.fit(x_train, x_train,
                	epochs=100,
                	batch_size=128,
                	shuffle=True,
                	validation_data=(x_val, x_val))

	# after training, use encoder to encode image, and feed it into Kmeans
	encoded_imgs = encoder.predict(X)
	encoded_imgs = encoded_imgs.reshape(encoded_imgs.shape[0], -1)
	kmeans = KMeans(n_clusters=2, random_state=0).fit(encoded_imgs).labels_
	with open('mypickle', 'wb') as f:
		pickle.dump(kmeans, f)
'''

#testing
#if sys.argv[1] == "test":
# get test cases
import pandas as pd
f = pd.read_csv(sys.argv[2])
IDs, idx1, idx2 = np.array(f['ID']), np.array(f['image1_index']), np.array(f['image2_index'])
with open('mypickle', 'rb') as f:
	cluster = pickle.load(f)
# predict
o = open(sys.argv[3], 'w')
o.write("ID,Ans\n")
for idx, i1, i2 in zip(IDs, idx1, idx2):
	p1 = cluster[i1]
	p2 = cluster[i2]
	if p1 == p2:
		pred = 1  # two images in same cluster
	else: 
		pred = 0  # two images not in same cluster
	o.write("{},{}\n".format(idx, pred))
o.close()
