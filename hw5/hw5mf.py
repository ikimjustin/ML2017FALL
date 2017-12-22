import sys, csv, tensorflow, keras.backend.tensorflow_backend
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Embedding, Reshape, Merge
import sys, os
import keras
import pandas as pd
#import readline
import numpy as np
import csv
from keras import regularizers
from keras.models import Model
from keras.layers import Input, GRU, LSTM, Dense, Dropout, Bidirectional, Flatten, Dot, Add
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

# training argument 
epochs = 30#30
validation_split = 0.1
dropout_rate = 0.3
gpu_fraction = 0.5
#for dnn
dimension = 480 #240

	
# set GPU utilization
keras.backend.tensorflow_backend.set_session(tensorflow.Session(config=tensorflow.ConfigProto(gpu_options=tensorflow.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction))))



#MF

def mfmodel(users, items, latent_dim=240):
	user_input = Input(shape=[1])
	item_input = Input(shape=[1])
	user_vec = Embedding(users, latent_dim, embeddings_initializer='random_normal')(user_input)
	user_vec = Flatten()(user_vec)
	item_vec = Embedding(items, latent_dim, embeddings_initializer='random_normal')(item_input)
	item_vec = Flatten()(item_vec)
	user_bias = Embedding(users, 1, embeddings_initializer='zeros')(user_input)
	user_bias = Flatten()(user_bias)
	item_bias = Embedding(items, 1, embeddings_initializer='zeros')(item_input)
	item_bias = Flatten()(item_bias)
	r_hat = Dot(axes=1)([user_vec, item_vec])
	r_hat = Add()([r_hat, user_bias, item_bias])
	model = keras.models.Model([user_input, item_input], r_hat)
	model.compile(loss='mse', optimizer='adamax') #opt=sgd
	return model
	
	
# set the UV decomposition model
print('initial model...')
def dnnmodel(users, items, factors, dropout_rate):
	user = Sequential()
	user.add(Embedding(users, factors, input_length=1))
	user.add(Reshape((factors,)))
	user.summary()
	
	item = Sequential()
	item.add(Embedding(items, factors, input_length=1))
	item.add(Reshape((factors,)))
	item.summary()
	
	model = Sequential()
	model.add(Merge([user, item], mode='concat'))
	model.add(Dropout(dropout_rate))
	model.add(Dense(factors, activation='relu'))
	model.add(Dropout(dropout_rate))
	model.add(Dense(1, activation='linear'))
	model.compile(loss = 'mse', optimizer = 'adamax')
	return model
			  



# training
if sys.argv[1] == "train":
	# load training data
	input_data = pd.read_csv("train.csv", sep = ',', encoding='utf-8', usecols=['UserID', 'MovieID', 'Rating'])
	max_user = input_data['UserID'].drop_duplicates().max()
	max_movie = input_data['MovieID'].drop_duplicates().max()
	input_data = input_data.sample(frac = 1., random_state = 168464)
	users = input_data['UserID'].values - 1
	movies = input_data['MovieID'].values - 1
	ratings = input_data['Rating'].values

	a=np.mean(ratings)
	b=np.std(ratings)
	print(max_user)
	print(max_movie)

	ratings=(ratings-a)/b
	#model = dnnmodel(max_user, max_movie, dimension, dropout_rate)
	model = mfmodel(max_user, max_movie)

	model.summary()

	print('start training')	
	earlystopping = EarlyStopping('val_loss', patience = 3, verbose = 1)
	checkpoint = ModelCheckpoint("best_model", verbose = 1 ,save_best_only = True, monitor = 'val_loss')
	history = model.fit([users, movies], ratings, validation_split = validation_split, batch_size = 128, epochs = epochs, callbacks=[checkpoint,earlystopping])

	
# testing
if sys.argv[1] == "test":
	# load testing data
	print('start to test data')
	input_data = pd.read_csv(sys.argv[2], sep = ',', encoding='utf-8', usecols=['UserID', 'MovieID'])
	
	
	max_user = input_data['UserID'].drop_duplicates().max()
	max_movie = input_data['MovieID'].drop_duplicates().max()
	#input_data = input_data.sample(frac = 1., random_state = 168464)
	users = input_data['UserID'].values - 1
	movies = input_data['MovieID'].values - 1
	print(max_user)
	print(max_movie)



	#model = dnnmodel(max_user, max_movie, dimension, dropout_rate)
	model = mfmodel(max_user, max_movie)
	
	model.load_weights('best_model')	
	output = model.predict([users, movies])
	file=open(sys.argv[3],'w')
	csv.writer(file).writerow(['TestDataID', 'Rating'])
	a=3.58171208604
	b=1.11689766115

	for i, j in enumerate(output):
		csv.writer(file).writerow([str(i+1), str((j[0]*b)+a)])
	file.close()


'''
	model.load_weights('best_model')
	output = model.predict([users, movies])
	with open("output.csv", 'w', encoding='utf-8') as f:
		spamwriter = csv.writer(f, delimiter=',')
		spamwriter.writerow(['TestDataID', 'Rating'])
		for i, j in enumerate(output):  
			spamwriter.writerow([str(i+1), str(j[0])])
'''
