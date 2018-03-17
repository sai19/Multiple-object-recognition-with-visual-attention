from __future__ import division
from keras.layers import LSTM,Input,Lambda,Dense,Activation
from keras import backend as K
from keras.models import Model
import keras
from keras import metrics
from keras.datasets import mnist
import numpy as np
import cv2
from core import layer


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train/255
#x_train[x_train>0.5] = 1
#x_train[x_train<=0.5] = 0
#x_train = x_train.reshape(x_train.shape[0], 28*28)

M = 2 # monte carlo samples
nb_classes = 10;
y_train = keras.utils.to_categorical(y_train, nb_classes)
y_test = keras.utils.to_categorical(y_test, nb_classes)
resized_input = Input(shape=(16,16,1))
x_resized = np.resize(x_train,(x_train.shape[0],16,16))
x_resized = np.reshape(x_resized,(x_resized.shape[0],16,16,1))
x_train = np.reshape(x_train,(x_train.shape[0],28,28,1))
image_input = Input(shape=(28,28,1))
pred_input = Input(shape=(nb_classes,))

## build context network
context_param = {}
context_param["kernel_size"] = 3
context_param["filters"] = 64
context_param["rnn_hidden_size"] = 128
context_param["model_depth"] = 3 
context_net = layer.context_net(**context_param)
context_net.build_layers()
## build recurrent network
recurrent_param = {}
recurrent_param["rnn_hidden_size"] = 128
recurrent_net = layer.recurrent_net(**recurrent_param)
recurrent_net.build_layers()

## build emission network
emission_param = {}
emission_param["input_size_loc"] = 2
emission_net = layer.emission_net(**emission_param)
emission_net.build_layers()

## build glimpse network
glimpse_param = {}
glimpse_param["kernel_size"] = 3
glimpse_param["model_depth"] = 3
glimpse_param["filters"] = 64
glimpse_param["hidden_vector_length"] = 128
glimpse_param["output_size"] = 128
glimpse_net = layer.glimpse_net(**glimpse_param)
glimpse_net.build_layers()

## build classification network
class_param = {}
class_param["class_hidden_size"] = 128
class_param["nb_classes"] = nb_classes
class_net = layer.class_net(**class_param)
class_net.build_layers()

for i in range(M):
	if i==0:
		initial_state = context_net.get_context_out(resized_input)
		r1,r2 = recurrent_net.get_recurrent_out(initial_state)
		e = emission_net.get_emission_out(r2)
		g = glimpse_net.get_glimpse_out(image_input,e)
		out = class_net.get_classification_out(r1)
		loss = K.categorical_crossentropy(pred_input,out)
	else:
		r1,r2 = recurrent_net.get_recurrent_out(g)
		e = emission_net.get_emission_out(r2)
		g = glimpse_net.get_glimpse_out(image_input,e)
		out = class_net.get_classification_out(r1)
		loss = loss + K.categorical_crossentropy(pred_input,out)
loss = loss/M		
model = Model([resized_input,image_input,pred_input],[out])
model.add_loss(loss)
model.compile(optimizer="rmsprop")
model.summary()
model.fit([x_resized,x_train,y_train],verbose=1,epochs=10)			 