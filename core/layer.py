"""
Implementation of https://arxiv.org/pdf/1412.7755.pdf in Keras
Author: Saiprasad Koturwar
DISCLAIMER
Work in progress.
"""
from __future__ import division
from keras.layers import Conv2D,Dense,Flatten
from keras import backend as K
from keras.models import Model
import keras
from keras.layers import Input,LSTM,Lambda,Cropping2D
from keras import metrics
from keras.datasets import mnist
import tensorflow as tf
import numpy as np
import cv2


class glimpse_net(object):
	def __init__(self, **kwargs):
        	for k, v in kwargs.items():
            		setattr(self, k, v)
	def build_layers(self):
		self.convolve_layers = [];
		for i in range(self.model_depth):
			self.convolve_layers.append(Conv2D(self.filters,self.kernel_size))
		self.convolve_dense = Dense(self.output_size)
		self.loc_dense_1 = Dense(self.hidden_vector_length)
		self.loc_dense_2 = Dense(self.output_size) 		
	def get_glimpse_out(self,glimpse,loc):
		#glimpse network
		glimpse = Lambda(lambda x:tf.image.extract_glimpse(x,[10,10],loc))(glimpse)
		for i in range(self.model_depth):
			if i==0:
				model_convolve = self.convolve_layers[i](glimpse)
			else:
				model_convolve = self.convolve_layers[i](model_convolve)	
		model_convolve = Flatten()(model_convolve)
		model_convolve = Dense(self.output_size)(model_convolve)
		# location network
		model_loc = self.loc_dense_1(loc)
		model_loc = self.loc_dense_2(model_loc)
		model_out = keras.layers.Multiply()([model_loc,model_convolve])
		return model_out 

class recurrent_net(object):
	def __init__(self, **kwargs):
        	for k, v in kwargs.items():
            		setattr(self, k, v)
	def build_layers(self):
		self.recurrent1 = LSTM(self.rnn_hidden_size,name="rnn_output_1")
		self.recurrent2 = LSTM(self.rnn_hidden_size,name="rnn_output_2")
	def get_recurrent_out(self,model_recurrent):
		model_recurrent = Lambda(lambda x:K.reshape(x,(-1,1,self.rnn_hidden_size)))(model_recurrent)
		model_recurrent_1 = self.recurrent1(model_recurrent)
		model_recurrent_1 = Lambda(lambda x:K.reshape(x,(-1,1,self.rnn_hidden_size)))(model_recurrent_1)
		model_recurrent_2 = self.recurrent2(model_recurrent_1)
		return model_recurrent_1,model_recurrent_2

class emission_net(object):
	def __init__(self, **kwargs):
        	for k, v in kwargs.items():
            		setattr(self, k, v)
	def build_layers(self):
		self.dense = Dense(self.input_size_loc)        		
	def get_emission_out(self,model_emission):
		model_emission_out = self.dense(model_emission)
		return model_emission_out

class context_net(object):
	def __init__(self, **kwargs):
        	for k, v in kwargs.items():
            		setattr(self, k, v)
	def build_layers(self):
		self.conv_layers = []
		for i in range(self.model_depth):
			self.conv_layers.append(Conv2D(self.filters,self.kernel_size))	
		self.dense = Dense(self.rnn_hidden_size)    		
	def get_context_out(self,model_context):
		for i in range(self.model_depth):
			if i==0:
				model_context_out = self.conv_layers[i](model_context)
			else:
				model_context_out = self.conv_layers[i](model_context_out)	
		model_context_out = Flatten()(model_context_out)
		model_context_out = self.dense(model_context_out)
		return model_context_out

class class_net(object):
	def __init__(self, **kwargs):
        	for k, v in kwargs.items():
            		setattr(self, k, v)
	def build_layers(self):
		self.dense1 =  Dense(self.class_hidden_size)
		self.dense2 =  Dense(self.nb_classes,activation="softmax")
	def get_classification_out(self,model_classify):
		model_classify_out = self.dense1(model_classify)
		model_classify_out = self.dense2(model_classify_out)
		return model_classify_out



