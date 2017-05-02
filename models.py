import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32, lib.cnmem=0.75"
import copy
import numpy as np
import keras.backend as K

from keras.models import Model,Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.layers.core import Lambda

from code.loading import *

def upsample_hlist(exclude=[]):
    def upsample(input_list):
        output_list = []
        for i,x in enumerate(input_list):
            index = i
            output = x
            #print(output.tag.test_value.shape)
            while index > 0:
                output = K.repeat_elements(K.repeat_elements(output,2,axis=-3),2,axis=-2)
                shape = input_list[index-1].shape
                output = output[:,:shape[1],:shape[2],:]
                index -= 1
            if i not in exclude:
                output_list.append(output)
        return K.concatenate(output_list,axis=-1)
    return upsample


def upsample_hlist_output_shape(exclude=[]):
    def output_shape(input_shapes):
        summation = int(np.sum([shape[-1] for index, shape in enumerate(input_shapes)
                    if index not in exclude]))
        shape = input_shapes[0]
        return (shape[0],shape[1],shape[2],summation)
    return output_shape

def im_flatten(x):
    return x.reshape((x.shape[1]*x.shape[2],x.shape[3]))

def im_flatten_shape(shape):
    return (shape[1]*shape[2],shape[3])

def random_sample(sampling_rate,length=312180):
    rng_state = np.random.get_state()
    choices = np.random.choice(range(length),sampling_rate,replace=False)
    def sample(x):
        return x[choices]
    return sample,choices

def random_sample_shape(sampling_rate):
    def sample_shape(shape):
        return (sampling_rate,shape[1])
    return sample_shape

def expand_dims(x):
    return K.expand_dims(x).reshape((1,x.shape[0],x.shape[1]))

def expand_dims_shape(shape):
    return (1,shape[0],shape[1])

def dense_hc_model(nclasses=5,sampling_rate=500):
    inputs = Input(batch_shape=(1,480,640,3))
    vgg = VGG16(weights='imagenet', include_top=False)
    vgg.layers = vgg.layers[1:]
    X = vgg.layers[0](inputs)
    vgg.layers = vgg.layers[1:]
    indices = [0,3,7,11,15]
    output_list = []
    for index,layer in enumerate(vgg.layers):
        layer._non_trainable_weights = copy.deepcopy(layer._trainable_weights)
        layer._trainable_weights = []
        layer.border_mode = 'same'
        if index in indices:
            X = layer(X)
            output = Conv2D(32,(1,1),activation='relu',border_mode='same')(X)
            output_list.append(output)
        else:
            X = layer(X)

    X = Lambda(upsample_hlist(),output_shape=upsample_hlist_output_shape())(output_list)
    X = Lambda(im_flatten,output_shape=im_flatten_shape)(X)
    X = Dense(40,activation='relu')(X)
    X = Dropout(0.25)(X)
    X = Dense(40,activation='relu')(X)
    X = Dense(nclasses,activation='softmax')(X)
    output = Lambda(expand_dims,output_shape=expand_dims_shape)(X)
    model = Model(inputs=inputs,outputs=output)
    sgd = SGD(lr=0.01, decay=0.0001)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

def full_conv_model(nclasses):
    inputs = Input(batch_shape=(1,None,None,3))
    vgg = VGG16(weights='imagenet', include_top=False)
    vgg.layers = vgg.layers[1:]
    X = vgg.layers[0](inputs)
    vgg.layers = vgg.layers[1:]
    indices = [0,3,7,11,15]
    output_list = []
    for index,layer in enumerate(vgg.layers):
        layer._non_trainable_weights = copy.deepcopy(layer._trainable_weights)
        layer._trainable_weights = []
        layer.border_mode = 'same'
        if index in indices:
            X = layer(X)
            output = Conv2D(32,(3,3),activation='relu',border_mode='same')(X)
            output_list.append(output)
        else:
            X = layer(X)

    X = Lambda(upsample_hlist(),output_shape=upsample_hlist_output_shape())(output_list)
    X = Conv2D(32,(3,3),activation='relu',border_mode='same')(X)
    X = Dropout(0.4)(X)
    output = Conv2D(nclasses,(3,3),activation='softmax',border_mode='same')(X)
    model = Model(inputs=inputs,outputs=output)
    sgd = SGD(lr=0.01, decay=0.0001)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

def feature_model():
    vgg = VGG16(weights='imagenet', include_top=False)
    input = Input(shape=(None,None,3))
    vgg.layers = vgg.layers[1:]
    X = vgg.layers[0](input)
    vgg.layers = vgg.layers[1:]
    indices = [0,3,7,11,15]
    output_list = []
    for index,layer in enumerate(vgg.layers):
        layer._non_trainable_weights = copy.deepcopy(layer._trainable_weights)
        layer._trainable_weights = []
        layer.border_mode = 'same'
        X = layer(X)
        if index in indices:
            output_list.append(X)

    model = Model(inputs=input,outputs=output_list)
    sgd = SGD(lr=0.01, decay=0.0001)
    model.compile(optimizer=sgd, loss='binary_crossentropy') 
    return model

def hc_classifier():
	model = Sequential()
	model.add(Dense(128,input_dim=1472))
	model.add(Dropout(0.25))
	model.add(Dense(128,activation='relu'))
	model.add(Dense(16,activation='relu'))
	model.add(Dense(4,activation='softmax'))
	sgd = SGD(lr=0.0001, decay=0.0001)
	model.compile(optimizer=sgd, loss='categorical_crossentropy') 
	return model
