"""
Implementation of https://arxiv.org/pdf/1412.7755.pdf in Keras
Author: Saiprasad Koturwar
DISCLAIMER
Work in progress.
"""
from __future__ import division
from keras.layers import Conv2D,Dense
from keras import backend as K
from keras.models import Model
import keras
from keras import metrics
from keras.datasets import mnist
import numpy as np
import cv2


def get_glimpse_network(filter_size,model_depth,filters,output_size,input_size_image,input_size_loc,hidden_vector_length):
	model_convolve = Sequential()
	for i in range(model_depth):
		if i==0:
			model_convolve.add(Conv2D(filter_size,filters,input_shape=input_size))
		else:
			model_convolve.add(Conv2D(filter_size,filters))	
	model_convolve.add(Flatten())
	model_convolve.add(Dense(output_size))

	model_loc = Sequential()
	model_loc.add(Dense(hiden_vector_length,input_shape=input_size_loc))
	model_loc.add(Dense(output_size))

	return model_convolve,model_loc

def get_reccurent_network(input_size,rnn_hidden_size):
	model_reccurent = Sequential()
	model_reccurent.add(LSTM(rnn_hidden_size,input_shape=input_size,name="rnn_output_1"))
	model_reccurent.add(LSTM(rnn_hidden_size,name="rnn_output_2"))
	return model_reccurent

def get_emission_network(rnn_hidden_size,input_size_loc):
	model_emission = Sequential()
	model_emission.add(Dense(input_size_loc,input_shape=rnn_hidden_size))
	return model_emission

def get_context_network(filter_size,model_depth,filters,rnn_hidden_size):
	model_context = Sequential()
	for i in range(model_depth):
		if i==0:
			model_context.add(Conv2D(filter_size,filters,input_shape=input_size))
		else:
			model_context.add(Conv2D(filter_size,filters))	
	model_context.add(Flatten())
	model_context.add(Dense(rnn_hidden_size))
	return model_context

def get_classification_network(rnn_hidden_size,class_hidden_size,nb_classes):
	model_classify = Sequential()
	model_classify.add(Dense(class_hidden_size,input_shape=rnn_hidden_size))
	model_classify.add(Dense(nb_classes,activation="softmax"))	

