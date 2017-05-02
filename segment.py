import numpy as np
from sklearn.linear_model import SGDClassifier
from code.models import *
from code.loading import *
from keras.preprocessing.image import ImageDataGenerator

def full_conv():
	datagen = ImageDataGenerator(
		rotation_range=45.,
		fill_mode='constant',
		cval=-1.0,
		horizontal_flip=True,
		vertical_flip=True,	
	)
	model = full_conv_model(6)
	model.fit_generator(full_conv_generator(datagen),1,7500,verbose=2)
	model.save_weights("full_conv_hc.h5")

def dense_hc():
	datagen = ImageDataGenerator(
		rotation_range=45.,
		fill_mode='constant',
		cval=-1.0,
		horizontal_flip=True,
		vertical_flip=True,	
	)
	model = dense_hc_model()
	model.fit_generator(dense_generator(datagen),1,2500,verbose=2)
	model.save_weights("dense_hc.h5")
def full_hc():
	datagen = ImageDataGenerator(
		rotation_range=25.0,
		fill_mode='constant',
		cval= -1.0,
		horizontal_flip=False,
		vertical_flip=False,	
	)
	epochs = 50000
	batch_size = 50
	sampling = batch_size*10
	hc_model = feature_model()
	classifier = SGDClassifier(loss='modified_huber')
	for i in range(epochs):
		image,labels = load_image_batch(datagen)
		hc_list = hc_model.predict(image)
		p = np.array(labels != -1).astype(float).flatten()
		p = p/np.sum(p)
		indices = np.random.choice(range(len(labels.flatten())),
			size=sampling, replace=False,p=p)
		batch = np.array(np.unravel_index(indices,labels.shape)).T
		hc_batch = get_hypercolumns(hc_list,batch)
		#label_batch = one_hot_encode(labels[batch[:,0],batch[:,1]])
		#loss = classifier.train_on_batch(hc_batch,label_batch)
		label_batch = labels[batch[:,0],batch[:,1]]
		split = int(len(hc_batch)*.8)
		hc_train, label_train = hc_batch[:split],label_batch[:split]
		hc_val,label_val = hc_batch[split:],label_batch[split:]
		classifier.partial_fit(hc_train,label_train,classes=[0,1,2,3])
		loss = classifier.score(hc_val,label_val)
		print('Epoch {}: Score = {}'.format(i+1,loss))
			
					
if __name__ == '__main__':
	full_conv()
