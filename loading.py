import random
import h5py
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from skimage.io import imread
from scipy.misc import imresize
from skimage.color import gray2rgb,rgb2gray

def pre_process(image,color=False):
    if color:
        image[:,:,0] -= np.mean(image[:,:,0])
        image[:,:,1] -= np.mean(image[:,:,1])
        image[:,:,2] -= np.mean(image[:,:,2])
    else:
        image[:,:,0] -= 127.5
        image[:,:,1] -= 127.5
        image[:,:,2] -= 127.5
    return image

def one_hot_encode(labels,n_values=5):
    shape = labels.shape
    l = labels.flatten()
    if n_values == 6:
        l[l==-1] = 5 
    
    one_hot = np.eye(n_values)[l]
    if len(shape) == 2:
        output = one_hot.reshape(shape[0],shape[1],n_values)
    else:
        output = one_hot.reshape(shape[0],n_values)
    return output

def load_batch():
    hfile = 'data/uhcs.h5'
    validation = ['micrograph596','micrograph75','micrograph360','micrograph579','micrograph1579']
    with h5py.File(hfile, 'r') as f:
        key = random.choice([key for key in list(f.keys()) if not key in validation])
        micrograph = f[key]
        im = gray2rgb(micrograph['image'][...][:-38])
        l = micrograph['labels'][...][:-38]
        l[l==-1]=4

        return im,l

def full_conv_generator(datagen):
    for i in range(100000):
        image,labels = load_batch()
        shape = labels.shape
        rng_state = np.random.get_state()
        labels = datagen.random_transform(np.expand_dims(labels,axis=-1))
        labels = labels.reshape(shape)
         
        labels = one_hot_encode(labels,n_values=6)[:480,:640]
        labels = np.expand_dims(labels,axis=0)

        np.random.set_state(rng_state)
        shape = image.shape
        image = datagen.random_transform(image)
        '''
        image = imresize(image[:,left:right],shape,interp='nearest')
        '''
        image = np.array(image).astype(float)
        image = pre_process(image)
        image = np.expand_dims(image,axis=0)[:,:480,:640]
        yield image,labels

def dense_generator(datagen):
    for i in range(100000):
        image,labels = load_batch()
        shape = labels.shape
        rng_state = np.random.get_state()
        labels = datagen.random_transform(np.expand_dims(labels,axis=-1))
        labels = labels.reshape(shape)
        
        '''
        top = [index for index,label in enumerate(labels[0]) if label !=-1]
        topleft = top[0]
        topright = top[-1]
        bottom = [index for index,label in enumerate(labels[-1]) if label !=-1]
        bottomleft = bottom[0]
        bottomright = bottom[-1]
        left,right = max(bottomleft,topleft),min(bottomright,topright)
        labels = imresize(labels[:,left:right],shape,interp='nearest')
        '''
        labels = one_hot_encode(labels,n_values=5)[:480,:640]
        labels = labels.reshape(-1,labels.shape[-1])
        labels = np.expand_dims(labels,axis=0)

        np.random.set_state(rng_state)
        shape = image.shape
        image = datagen.random_transform(image)
        '''
        image = imresize(image[:,left:right],shape,interp='nearest')
        '''
        image = np.array(image).astype(float)
        image = pre_process(image)
        image = np.expand_dims(image,axis=0)[:,:480,:640]
        yield image,labels

def load_image_batch(datagen):
        image,labels = load_batch()
        shape = labels.shape
        rng_state = np.random.get_state()
        labels = np.expand_dims(labels,axis=-1)
        labels = datagen.random_transform(labels)
        labels = labels.reshape(shape[0],shape[1])
        '''
        top = [index for index,label in enumerate(labels[0]) if label !=-1]
        topleft = top[0]
        topright = top[-1]
        bottom = [index for index,label in enumerate(labels[-1]) if label !=-1]
        bottomleft = bottom[0]
        bottomright = bottom[-1]
        left,right = max(bottomleft,topleft),min(bottomright,topright)
        labels = imresize(labels[:,left:right],shape,interp='nearest')
        '''

        np.random.set_state(rng_state)
        shape = image.shape
        image = datagen.random_transform(image)
        '''
        image = imresize(image[:,left:right],shape,interp='nearest')
        '''
        image = np.array(image).astype(float)
        image = pre_process(image)
        #image = image.transpose(2,0,1)
        image = np.expand_dims(image,axis=0)
        return image[:480,:640],labels[:480,:640]

def get_hypercolumns(hc_list,indices):
	hypercolumns = [np.zeros((len(indices),64)),
			np.zeros((len(indices),128)),
			np.zeros((len(indices),256)),
			np.zeros((len(indices),512)),
			np.zeros((len(indices),512))]

	for i,hc in enumerate(hc_list):
		for j,index in enumerate(indices):
			index = [index[0]/2**i,index[1]/2**i]
			index = tuple(np.array(index).astype(int))
			hypercolumns[i][j] = hc_list[i][0][index]
	return np.concatenate(hypercolumns,axis=1)
			
		
