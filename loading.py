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

def one_hot_encode(labels):
    l = labels.flatten()
    if set(l) != {0,1,2,3}:
        l[l==85] = 1
        l[l==170] = 2
        l[l==255] = 3
    shape = labels.shape
    n_values = 4
    one_hot = np.eye(n_values)[l]
    return one_hot.reshape(shape[0],shape[1],n_values)

def load_batch():
    hfile = 'data/uhcs.h5'
    with h5py.File(hfile, 'r') as f:
        key = random.choice(list(f.keys()))
        micrograph = f[key]
        im = gray2rgb(micrograph['image'][...][:-38])
        l = micrograph['labels'][...][:-38]
        l[l==-1] = 4

        return im,l

def full_conv_generator():
    for i in range(100000):
        image,labels = load_batch()
        yield image,labels

def dense_generator(datagen):
    for i in range(100000):
        image,labels = load_batch()
        shape = labels.shape
        rng_state = np.random.get_state()
        datagen.random_transform(np.expand_dims(labels,axis=0))[0]
        top = [index for index,label in enumerate(labels[0]) if label !=-1]
        topleft = top[0]
        topright = top[-1]
        bottom = [index for index,label in enumerate(labels[-1]) if label !=-1]
        bottomleft = bottom[0]
        bottomright = bottom[-1]
        left,right = max(bottomleft,topleft),min(bottomright,topright)
        labels = imresize(labels[:,left:right],shape,interp='nearest')
        labels = one_hot_encode(labels)
        labels = labels.reshape(-1,labels.shape[-1])
        labels = np.expand_dims(labels,axis=0)

        np.random.set_state(rng_state)
        shape = image.shape
        datagen.random_transform(image)
        image = imresize(image[:,left:right],shape,interp='nearest')
        image = np.array(image).astype(float)
        image = pre_process(image).transpose(2,0,1)
        image = np.expand_dims(image,axis=0)
        yield image,labels
