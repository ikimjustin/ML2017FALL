import sys, argparse, os
import keras
import _pickle as pk
import readline
import numpy as np
import csv
from keras import regularizers
from keras.models import Model
from keras.layers import Input, GRU, LSTM, Dense, Dropout, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

import keras.backend.tensorflow_backend as K
import tensorflow as tf

from util import DataManager
parser = argparse.ArgumentParser(description='Sentiment classification')
parser.add_argument('model')
parser.add_argument('action', choices=['train','test','semi'])

#for testing!!
#parser.add_argument('outputpath')
# training argument
parser.add_argument('--batch_size', default=512, type=float)#128
parser.add_argument('--nb_epoch', default=20, type=int)
parser.add_argument('--val_ratio', default=0.05, type=float)
parser.add_argument('--gpu_fraction', default=0.5, type=float) #0.3
parser.add_argument('--vocab_size', default=20000, type=int)#20000
parser.add_argument('--max_length', default=40,type=int)

# model parameter
parser.add_argument('--loss_function', default='binary_crossentropy')
parser.add_argument('--cell', default='LSTM', choices=['LSTM','GRU'])
parser.add_argument('-emb_dim', '--embedding_dim', default=256, type=int)#128
parser.add_argument('-hid_siz', '--hidden_size', default=512, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)#0.3
parser.add_argument('-lr','--learning_rate', default=0.000000001,type=float)
parser.add_argument('--threshold', default=0.17,type=float)##0.1

# output path for your prediction
parser.add_argument('--result_path', default='result.csv',)
parser.add_argument('--train_path', default='result.csv',)
parser.add_argument('--semi_path', default='result.csv',)

# put model in the same directory

parser.add_argument('--load_model', default = None)
parser.add_argument('--save_dir', default = 'model/')

args = parser.parse_args()

#train_path = 'training_label.txt'
train_path = args.train_path
#test_path = sys.argv[1]
#test_path = 'data/testing_data.txt'
#semi_path = 'data/training_nolabel.txt'
semi_path = args.semi_path
#semi_path = 'testing_data.txt'
#outputpath=sys.argv[2]

# build model
def simpleRNN(args):
    inputs = Input(shape=(args.max_length,))

    # Embedding layer
    embedding_inputs = Embedding(args.vocab_size, 
                                 args.embedding_dim, 
                                 trainable=True)(inputs)
    # RNN 
    return_sequence = False
    dropout_rate = args.dropout_rate
    if args.cell == 'GRU':
        RNN_cell = GRU(args.hidden_size, 
                       return_sequences=return_sequence, 
                       dropout=dropout_rate)
    elif args.cell == 'LSTM':
        RNN_cell = LSTM(args.hidden_size, 
                        return_sequences=return_sequence, 
                        dropout=dropout_rate)

    RNN_output = RNN_cell(embedding_inputs)

    # DNN layer
    outputs = Dense(args.hidden_size//2, 
                    activation='relu',
                    kernel_regularizer=regularizers.l2(0.1))(RNN_output)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Dense(1, activation='sigmoid')(outputs)
        
    model =  Model(inputs=inputs,outputs=outputs)

    # optimizer
    #adam = Adam()
    #new = rmsprop()
    print ('compile model...')

    # compile model
    model.compile( loss=args.loss_function, optimizer='rmsprop', metrics=[ 'accuracy',])
    
    return model
def main():
    # limit gpu memory usage
    def get_session(gpu_fraction):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))  
    K.set_session(get_session(args.gpu_fraction))
    
    save_path = os.path.join(args.save_dir,args.model)
    if args.load_model is not None:
        load_path = os.path.join(args.save_dir,args.load_model)
 #####read data#####
    dm = DataManager()
    print ('Loading data...')
    if args.action == 'train':
        dm.add_data('train_data', train_path, True)
        #dm.add_data('train_data', train_path, True)
        dm.add_data('semi_data', semi_path, False)
        #dm.add_test('semi_data', semi_path, False)
    else:
        #raise Exception ('Implement your testing parser')
        dm.add_test('test_data', test_path, False)
            
    # prepare tokenizer
    print ('get Tokenizer...')
    if args.load_model is not None:
        # read exist tokenizer
        dm.load_tokenizer(os.path.join(load_path,'token.pk'))
    else:
        # create tokenizer on new data
        dm.tokenize(args.vocab_size)
                            
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if not os.path.exists(os.path.join(save_path,'token.pk')):
        dm.save_tokenizer(os.path.join(save_path,'token.pk')) 

    # convert to sequences
    dm.to_sequence(args.max_length)
  # initial model
    print ('initial model...')
    model = simpleRNN(args)    
    print (model.summary())

    if args.load_model is not None:
        if args.action == 'train':
            print ('Warning : load a exist model and keep training')
        path = os.path.join(load_path,'model.h5')
        if os.path.exists(path):
            print ('load model from %s' % path)
            model.load_weights(path)
        else:
            raise ValueError("Can't find the file %s" %path)
    elif args.action == 'test':
        print ('Warning : testing without loading any model')
  # training
    if args.action == 'train':
        (X,Y),(X_val,Y_val) = dm.split_data('train_data', args.val_ratio)
        earlystopping = EarlyStopping(monitor='val_acc', patience = 3, verbose=1, mode='max')

        save_path = os.path.join(save_path,'model.h5')
        checkpoint = ModelCheckpoint(filepath=save_path, 
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     monitor='val_acc',
                                     mode='max' )
        history = model.fit(X, Y, 
                            validation_data=(X_val, Y_val),
                            epochs=args.nb_epoch, 
                            batch_size=args.batch_size,
                            shuffle= True,
                            callbacks=[checkpoint, earlystopping] )
        #args.action == 'semi':
        (X,Y),(X_val,Y_val) = dm.split_data('train_data', args.val_ratio)

        [semi_all_X] = dm.get_data('semi_data')
        earlystopping = EarlyStopping(monitor='val_acc', patience = 3, verbose=1, mode='max')

        save_path = os.path.join(save_path,'model.h5')
        checkpoint = ModelCheckpoint(filepath=save_path, 
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     monitor='val_acc',
                                     mode='max' )
            # semi-supervised training
        # repeat 10 times
        for i in range(10):
            ##^^def=10
            # label the semi-data
            semi_pred = model.predict(semi_all_X, batch_size=512, verbose=True)#1024
            semi_X, semi_Y = dm.get_semi_data('semi_data', semi_pred, args.threshold, args.loss_function)
            semi_X = np.concatenate((semi_X, X))
            semi_Y = np.concatenate((semi_Y, Y))
            print ('-- iteration %d  semi_data size: %d' %(i+1,len(semi_X)))
            # train
            history = model.fit(semi_X, semi_Y, 
                                validation_data=(X_val, Y_val),
                                epochs=3, 
                                batch_size=args.batch_size,
                                callbacks=[checkpoint, earlystopping] )

            if os.path.exists(save_path):
                print ('load model from %s' % save_path)
                model.load_weights(save_path)
            else:
                raise ValueError("Can't find the file %s" %path)
 # testing
    elif args.action == 'test' :
        #raise Exception ('Implement your testing function')
        [all_X] = dm.get_data('test_data')
        print('load model and predict...')
        save_path = os.path.join(save_path,'model.h5')
        #model=load_model('model.h5')
        pred = model.predict(all_X, batch_size=512, verbose=True)
        pred =np.around(pred)
        file=open(outputpath,'w')
        csv.writer(file).writerow(['id','label'])

        for i in range(len(all_X)):
            csv.writer(file).writerow([str(i),str(int(pred[i]))])
        file.close()

    # semi-supervised training
        # repeat 10 times
        for i in range(10):
            ##^^def=10
            # label the semi-data
            semi_pred = model.predict(semi_all_X, batch_size=512, verbose=True)#1024
            semi_X, semi_Y = dm.get_semi_data('semi_data', semi_pred, args.threshold, args.loss_function)
            semi_X = np.concatenate((semi_X, X))
            semi_Y = np.concatenate((semi_Y, Y))
            print ('-- iteration %d  semi_data size: %d' %(i+1,len(semi_X)))
            # train
            history = model.fit(semi_X, semi_Y, 
                                validation_data=(X_val, Y_val),
                                epochs=3, 
                                batch_size=args.batch_size,
                                callbacks=[checkpoint, earlystopping] )

            if os.path.exists(save_path):
                print ('load model from %s' % save_path)
                model.load_weights(save_path)
            else:
                raise ValueError("Can't find the file %s" %path)
if __name__ == '__main__':
        main()