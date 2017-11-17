#!/usr/bin/env python
# -- coding: utf-8 --
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from python_utils import *
import itertools
import numpy as np
import matplotlib.pyplot as plt

def plotconfusionmatrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def read_dataset(data_path):
    train_pixels = load_pickle(data_path)
    for i in range(len(train_pixels)):
        train_pixels[i] = np.fromstring(train_pixels[i], dtype=float, sep=' ').reshape((48, 48, 1))
    return np.asarray(train_pixels)

def get_labels(data_path):
    train_labels = load_pickle(data_path)
    train = []
    for i in range(len(train_labels)):
        train.append(int(train_labels[i]))
    return np.asarray(train)

import csv
import numpy as np
# import training data
# image resolution is 48*48
print ("Loading Data .....")
file=open('train.csv','r')
train_X=[]
train_Y=[]
line_number = 0
for line in file:
    if (line_number > 0) and (line_number<2500) :
        #train_Y.append(line[1])
        temp=line.strip('\n').split(',')
        #temp1 = temp[0]
        #output = [0,0,0,0,0,0,0]
        #output[int(temp[0])] = 1
        train_Y.append(int(temp[0]))
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


print ("Loading model ....")

model_path = 'model03.h5'
emotion_classifier = load_model(model_path)

print ("perdictioning ...")

np.set_printoptions(precision=2)
dev_feats = train_X
predictions = emotion_classifier.predict(dev_feats)
print ("arginggggggggggg")
predictions = predictions.argmax(axis=-1)
#print (predictions)
#te_labels = get_labels('../test_with_ans_labels.pkl')
#print (train_Y)
print ("confusion....")

conf_mat = confusion_matrix(train_Y,predictions)

plt.figure()
plotconfusionmatrix(conf_mat, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
plt.show()

