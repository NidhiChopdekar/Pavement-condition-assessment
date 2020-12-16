import os,cv2
import numpy as np
import keras
import matplotlib.pyplot as plt
import wandb
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras import backend as K
K.set_image_dim_ordering('th')

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam

from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

PATH = os.getcwd()
# Define data path
data_path = '//Users//prajna//Downloads//sih//alligator'
data_dir_list = os.listdir(data_path)

img_rows=128
img_cols=128
num_channel=1
num_epoch=3

# Define the number of classes
num_classes = 4

labels_name={'l':0,'m':1,'h':2,'na':3}

img_data_list=[]
labels_list = []

for dataset in data_dir_list:
    if not dataset.startswith('.'):
        img_list=os.listdir(data_path+'/'+ dataset)
        img_list.sort()
        print ('Loading the images of dataset-'+'{}\n'.format(dataset))
        label = labels_name[dataset]
        for img in img_list:
            input_img = load_img(data_path + '//'+ dataset + '//'+ img)
            data = img_to_array(input_img)
            samples = expand_dims(data, 0)
            datagen = ImageDataGenerator(width_shift_range=[-200,200])
            it = datagen.flow(samples, batch_size=1)
            for i in range(9):
                batch = it.next()
                input_img = batch[0].astype('uint8')
                input_img = np.full((100,80,3), 12, np.uint8)
                input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
                input_img_resize = cv2.resize(input_img,(128,128))
                img_data_list.append(input_img_resize)
                labels_list.append(label)
img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print (img_data.shape)

labels = np.array(labels_list)
# print the count of number of samples for different classes
print(np.unique(labels,return_counts=True))
# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

if num_channel==1:

	if K.image_dim_ordering()=='th':
		img_data= np.expand_dims(img_data, axis=1)
		print (img_data.shape)
	else:
		img_data= np.expand_dims(img_data, axis=4)
		print (img_data.shape)

else:
	if K.image_dim_ordering()=='th':
		img_data=np.rollaxis(img_data,3,1)
		print (img_data.shape)

USE_SKLEARN_PREPROCESSING=False

if USE_SKLEARN_PREPROCESSING:
	# using sklearn for preprocessing
	from sklearn import preprocessing

	def image_to_feature_vector(image, size=(128, 128)):
		# resize the image to a fixed size, then flatten the image into
		# a list of raw pixel intensities
		return cv2.resize(image, size).flatten()

	img_data_list=[]
	for dataset in data_dir_list:
		img_list=os.listdir(data_path+'//'+ dataset)
		print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
		for img in img_list:
			input_img=cv2.imread(data_path + '//'+ dataset + '//'+ img )
			input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
			input_img_flatten=image_to_feature_vector(input_img,(128,128))
			img_data_list.append(input_img_flatten)

	img_data = np.array(img_data_list)
	img_data = img_data.astype('float32')
	print (img_data.shape)
	img_data_scaled = preprocessing.scale(img_data)
	print (img_data_scaled.shape)

	print (np.mean(img_data_scaled))
	print (np.std(img_data_scaled))

	print (img_data_scaled.mean(axis=0))
	print (img_data_scaled.std(axis=0))

	if K.image_dim_ordering()=='th':
		img_data_scaled=img_data_scaled.reshape(img_data.shape[0],img_rows,img_cols,1)
		print (img_data_scaled.shape)

	else:
		img_data_scaled=img_data_scaled.reshape(img_data.shape[0],1,img_rows,img_cols)
		print (img_data_scaled.shape)


	if K.image_dim_ordering()=='th':
		img_data_scaled=img_data_scaled.reshape(img_data.shape[0],img_rows,img_cols,1)
		print (img_data_scaled.shape)

	else:
		img_data_scaled=img_data_scaled.reshape(img_data.shape[0],1,img_rows,img_cols)
		print (img_data_scaled.shape)

if USE_SKLEARN_PREPROCESSING:
	img_data=img_data_scaled
X_train.shape

input_shape=img_data[0].shape

model = Sequential()

model.add(Convolution2D(64, (3,3),input_shape=(1,128,128),padding="same",data_format='channels_first'))
model.add(Activation('relu'))
model.add(Dropout(rate=0.4))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(Dropout(rate=0.4))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(rate=0.9))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(rate=0.4))
model.add(Dense(4))
model.add(Activation('softmax'))

sgd=SGD(lr=0.001,decay=1e-4,momentum=0.6,nesterov=True)
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

hist = model.fit(X_train, y_train, batch_size=10, epochs=10,validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

model.save("alli_classsifier.h5")

myimage=cv2.imread('path to a test image',0)
myimage=cv2.resize(myimage,(128,128))
m=image.img_to_array(myimage)
m=np.expand_dims(m,axis=0)
images=np.vstack([m])
y_pred=model.predict(images)
print(y_pred)
classes=model.predict_classes(images)
print(classes)
