import os, sys
import numpy as np
from random import shuffle
import argparse
from math import log, floor
import pandas as pd
import csv

# If you wish to get the same shuffle result
# np.random.seed(2401)
'''
def load_data(train_data_path, train_label_path, test_data_path):
    X_train = pd.read_csv(train_data_path, sep=',', header=0)
    X_train = np.array(X_train.values)
    Y_train = pd.read_csv(train_label_path, sep=',', header=0)
    Y_train = np.array(Y_train.values)
    X_test = pd.read_csv(test_data_path, sep=',', header=0)
    X_test = np.array(X_test.values)

    return (X_train, Y_train, X_test)
'''
features = ['age', 'fnlwgt','sex', 'capital_gain', 'capital_loss', ' Federal-gov', ' Local-gov', ' Never-worked', ' Private', ' Self-emp-inc', ' Self-emp-not-inc', ' State-gov', ' Without-pay', '?_workclass', ' 10th', ' 11th', ' 12th', ' 1st-4th', ' 5th-6th', ' 7th-8th', ' 9th', ' Assoc-acdm', ' Assoc-voc', ' Bachelors', ' Doctorate', ' HS-grad', ' Masters', ' Preschool', ' Prof-school', ' Some-college', ' Adm-clerical', ' Armed-Forces', ' Craft-repair', ' Exec-managerial', ' Farming-fishing', ' Handlers-cleaners', ' Machine-op-inspct', ' Other-service', ' Priv-house-serv', ' Prof-specialty', ' Protective-serv', ' Sales', ' Tech-support', ' Transport-moving', '?_occupation', ' Husband', ' Not-in-family', ' Other-relative', ' Own-child', ' Unmarried', ' Wife', ' Amer-Indian-Eskimo', ' Asian-Pac-Islander', ' Black', ' Other', ' White']
def load_feature(train_data_path, train_label_path, test_data_path):
    feature = ['age', 'fnlwgt','sex', 'capital_gain', 'capital_loss', ' Federal-gov', ' Local-gov', ' Never-worked', ' Private', ' Self-emp-inc', ' Self-emp-not-inc', ' State-gov', ' Without-pay', '?_workclass', ' 10th', ' 11th', ' 12th', ' 1st-4th', ' 5th-6th', ' 7th-8th', ' 9th', ' Assoc-acdm', ' Assoc-voc', ' Bachelors', ' Doctorate', ' HS-grad', ' Masters', ' Preschool', ' Prof-school', ' Some-college', ' Adm-clerical', ' Armed-Forces', ' Craft-repair', ' Exec-managerial', ' Farming-fishing', ' Handlers-cleaners', ' Machine-op-inspct', ' Other-service', ' Priv-house-serv', ' Prof-specialty', ' Protective-serv', ' Sales', ' Tech-support', ' Transport-moving', '?_occupation', ' Husband', ' Not-in-family', ' Other-relative', ' Own-child', ' Unmarried', ' Wife', ' Amer-Indian-Eskimo', ' Asian-Pac-Islander', ' Black', ' Other', ' White']
    train_num = len(np.array(pd.read_csv(train_data_path, sep=',',usecols=[feature[0]], header=0).values))
    test_num = len(np.array(pd.read_csv(test_data_path, sep=',',usecols=[feature[0]], header=0).values))
    X_train = np.zeros((train_num,len(feature)*2))
    X_test = np.zeros((test_num,len(feature)*2))
    for i in range(0, len(feature)):
        X_train[:,i] = (np.array(pd.read_csv(train_data_path, sep=',',usecols=[feature[i]], header=0).values)).reshape(train_num)
        #X_train = np.array(X_train.values)
        X_test[:,i] = (np.array(pd.read_csv(test_data_path, sep=',',usecols=[feature[i]], header=0).values)).reshape(test_num)
        #X_test = np.array(X_test.values)
    for i in range(0, len(feature)):
        X_train[:,len(feature)+i] = (np.array(pd.read_csv(train_data_path, sep=',',usecols=[feature[i]], header=0).values)).reshape(train_num)
        X_train[:,len(feature)+i] = np.square(X_train[:,len(feature)+i])
        #X_train = np.array(X_train.values)
        X_test[:,len(feature)+i] = (np.array(pd.read_csv(test_data_path, sep=',',usecols=[feature[i]], header=0).values)).reshape(test_num)
        X_test[:,len(feature)+i] = np.square(X_test[:,len(feature)+i])
        #X_test = np.array(X_test.values)
        '''
    for i in range(0, len(feature)):
        X_train[:,2*len(feature)+i] = (np.array(pd.read_csv(train_data_path, sep=',',usecols=[feature[i]], header=0).values)).reshape(train_num)
        X_train[:,2*len(feature)+i] = np.power(X_train[:,len(feature)+i],3)
        #X_train = np.array(X_train.values)
        X_test[:,2*len(feature)+i] = (np.array(pd.read_csv(test_data_path, sep=',',usecols=[feature[i]], header=0).values)).reshape(test_num)
        X_test[:,2*len(feature)+i] = np.power(X_test[:,len(feature)+i],3)
        #X_test = np.array(X_test.values)
        '''
    Y_train = pd.read_csv(train_label_path, sep=',', header=0)
    Y_train = np.array(Y_train.values)
    
    return (X_train, Y_train, X_test)


def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def normalize(X_all, X_test):
    # Feature normalization with train and test X
    X_train_test = np.concatenate((X_all, X_test))
    mu = (sum(X_train_test) / X_train_test.shape[0])
    sigma = np.std(X_train_test, axis=0)
    mu = np.tile(mu, (X_train_test.shape[0], 1))
    sigma = np.tile(sigma, (X_train_test.shape[0], 1))
    #X_train_test_normed = (X_train_test - mu) / sigma
    X_train_test_normed = (X_train_test ) / sigma
    # Split to train, test again
    X_all = X_train_test_normed[0:X_all.shape[0]]
    X_test = X_train_test_normed[X_all.shape[0]:]
    return X_all, X_test

def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    valid_data_size = int(floor(all_data_size * percentage))

    X_all, Y_all = _shuffle(X_all, Y_all)

    X_train, Y_train = X_all[0:valid_data_size], Y_all[0:valid_data_size]
    X_valid, Y_valid = X_all[valid_data_size:], Y_all[valid_data_size:]

    return X_train, Y_train, X_valid, Y_valid

def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-8, 1-(1e-8))

def valid(w, b, X_valid, Y_valid):
    valid_data_size = len(X_valid)

    z = (np.dot(X_valid, np.transpose(w)) + b)
    y = sigmoid(z)
    y_ = np.around(y)
    result = (np.squeeze(Y_valid) == y_)
    print('Validation acc = %f' % (float(result.sum()) / valid_data_size))
    return

def train(X_all, Y_all):

    # Split a 10%-validation set from the training set
    valid_set_percentage = 0.1
    X_train, Y_train, X_valid, Y_valid = split_valid_set(X_all, Y_all, valid_set_percentage)

    # Initiallize parameter, hyperparameter
    feature = ['age', 'fnlwgt','sex', 'capital_gain', 'capital_loss', ' Federal-gov', ' Local-gov', ' Never-worked', ' Private', ' Self-emp-inc', ' Self-emp-not-inc', ' State-gov', ' Without-pay', '?_workclass', ' 10th', ' 11th', ' 12th', ' 1st-4th', ' 5th-6th', ' 7th-8th', ' 9th', ' Assoc-acdm', ' Assoc-voc', ' Bachelors', ' Doctorate', ' HS-grad', ' Masters', ' Preschool', ' Prof-school', ' Some-college', ' Adm-clerical', ' Armed-Forces', ' Craft-repair', ' Exec-managerial', ' Farming-fishing', ' Handlers-cleaners', ' Machine-op-inspct', ' Other-service', ' Priv-house-serv', ' Prof-specialty', ' Protective-serv', ' Sales', ' Tech-support', ' Transport-moving', '?_occupation', ' Husband', ' Not-in-family', ' Other-relative', ' Own-child', ' Unmarried', ' Wife', ' Amer-Indian-Eskimo', ' Asian-Pac-Islander', ' Black', ' Other', ' White']
    w = np.zeros((len(feature)*2,))
    b = np.zeros((1,))
    l_rate = 0.0001
    batch_size = 32 #32
    train_data_size = len(X_train)
    step_num = int(floor(train_data_size / batch_size))
    epoch_num = 1000
    save_param_iter = 50

    # Start training
    total_loss = 0.0
    for epoch in range(1, epoch_num):
        # Do validation and parameter saving
        if (epoch) % save_param_iter == 0:
            print('=====Saving Param at epoch %d=====' % epoch)
            #if not os.path.exists(save_dir):
            #    os.mkdir(save_dir)
            #np.savetxt(os.path.join(save_dir, 'w'), w)
            #np.savetxt(os.path.join(save_dir, 'b'), [b,])

            print('epoch avg loss = %f' % (total_loss / (float(save_param_iter) * train_data_size)))
            total_loss = 0.0
            valid(w, b, X_valid, Y_valid)

        # Random shuffle
        X_train, Y_train = _shuffle(X_train, Y_train)

        # Train with batch
        for idx in range(step_num):
            X = X_train[idx*batch_size:(idx+1)*batch_size]
            Y = Y_train[idx*batch_size:(idx+1)*batch_size]

            z = np.dot(X, np.transpose(w)) + b
            y = sigmoid(z)

            cross_entropy = -1 * (np.dot(np.squeeze(Y), np.log(y)) + np.dot((1 - np.squeeze(Y)), np.log(1 - y)))
            total_loss += cross_entropy

            w_grad = np.sum(-1 * X * (np.squeeze(Y) - y).reshape((batch_size,1)), axis=0)
            b_grad = np.sum(-1 * (np.squeeze(Y) - y))

            # SGD updating parameters
            w = w - l_rate * w_grad
            b = b - l_rate * b_grad

    return w,b

def infer(X_test, save_dir, output_dir):
    test_data_size = len(X_test)

    # Load parameters
    print('=====Loading Param from %s=====' % save_dir)
    w = np.loadtxt(os.path.join(save_dir, 'w'))
    b = np.loadtxt(os.path.join(save_dir, 'b'))

    # predict
    z = (np.dot(X_test, np.transpose(w)) + b)
    y = sigmoid(z)
    y_ = np.around(y)

    print('=====Write output to %s =====' % output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_path = os.path.join(output_dir, 'log_prediction.csv')
    with open(output_path, 'w') as f:
        f.write('id,label\n')
        for i, v in  enumerate(y_):
            f.write('%d,%d\n' %(i+1, v))

    return


# Load feature and label
#X_all, Y_all, X_test = load_data(sys.argv[3], sys.argv[4], sys.argv[5])
X_all, Y_all, X_test = load_feature(sys.argv[3], sys.argv[4], sys.argv[5])
#X_all, Y_all, X_test = load_data('X_train', 'Y_train', 'X_test')
#X_all, Y_all, X_test = load_feature('X_train', 'Y_train', 'X_test')
# Normalization
X_all, X_test = normalize(X_all, X_test)


w,b =train(X_all, Y_all)
x = X_test.T
a=np.dot(w,x)+b
y = sigmoid(a)
y_ = np.around(y)
'''
'''
y_ = 0
tree_num = 20
for i in range(tree_num):
    w,b = train(X_all, Y_all)
    x = X_test.T
    a = np.dot(w,x)+b
    y = sigmoid(a)
    y_QQ = np.around(y)
    y_ = y_ + y_QQ
y_ = np.around(y_/tree_num)
'''
'''
# output
file = open(sys.argv[6],'w')
#file = open("logi_test.csv",'w')
csv.writer(file).writerow(['id','label'])   
for i in range(len(y_)):
    csv.writer(file).writerow([str(i+1),str(int(y_[i]))])
file.close()
