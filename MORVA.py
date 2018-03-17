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
from keras.layers import Input
from keras import metrics
from keras.datasets import mnist
import numpy as np
import cv2


class glimpse_net(object):
	def __init__(self, **kwargs):
        for k, v in kwargs.iteritems():
            setattr(self, k, v)
	def get_glimpse_network(self,glimpse,loc):
		#glimpse network
		for i in range(model_depth):
			if i==0:
				model_convolve = Conv2D(self.filter_size,self.filters)(glimpse)
			else:
				model_convolve = Conv2D(filter_size,filters)(model_convolve)	
		model_convolve = Flatten()(model_convolve)
		model_convolve = Dense(output_size)(model_convolve)
		# location network
		model_loc = Dense(hiden_vector_length)(loc)
		model_loc = Dense(output_size)(model_loc)
		model_out = Merge(model_loc,model_convolve)
		self.out = model_out 

class recurrent_net(object):
	def __init__(self, **kwargs):
        for k, v in kwargs.iteritems():
            setattr(self, k, v)

	def get_reccurent_network(self,model_reccurent):
		model_reccurent_1 = LSTM(rnn_hidden_size,name="rnn_output_1")(model_reccurent)
		model_reccurent_2 = LSTM(rnn_hidden_size,name="rnn_output_2")(model_reccurent_1)
		self.reccurent1 = model_reccurent_1
		self.reccurent2 = model_reccurent_2

class emission_net(object):
	def __init__(self, **kwargs):
        for k, v in kwargs.iteritems():
            setattr(self, k, v)
	def get_emission_network(self,model_emission):
		model_emission_out = Dense(input_size_loc)(model_emission)
		self.out = model_emission_out

class context_net(object):
	def __init__(self, **kwargs):
        for k, v in kwargs.iteritems():
            setattr(self, k, v)
	def get_context_network(self,model_context):
		for i in range(model_depth):
			if i==0:
				model_context_out = Conv2D(filter_size,filters)(model_context)
			else:
				model_context_out = Conv2D(filter_size,filters)(model_context_out)	
		model_context_out = Flatten()(model_context_out)
		model_context_out = Dense(rnn_hidden_size)(model_context_out)
		self.out = model_context_out

class class_net(object):
	def __init__(self, **kwargs):
        for k, v in kwargs.iteritems():
            setattr(self, k, v)
	def get_classification_network(self,model_classify):
		model_classify_out = Dense(self.class_hidden_size)(model_classify)
		model_classify_out = Dense(nb_classes,activation="softmax")(model_classify_out)
		self.out = model_classify_out



