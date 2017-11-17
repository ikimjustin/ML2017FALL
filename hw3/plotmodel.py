#!/usr/bin/env python
# -- coding: utf-8 --

import argparse
#from keras.utils.vis_utils import plot_model
from keras.models import load_model
from keras.utils import plot_model

print ("loading model...")
model = load_model('model06.h5')
#plot_model(model,to_file='model.png')
model.summary()
plot_model(model, to_file='model.png')