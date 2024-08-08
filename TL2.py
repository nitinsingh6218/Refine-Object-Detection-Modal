import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
#from tensorflow.keras.applications.xception import Xception
#import pandas as pd
import matplotlib.pyplot as plt
#from tqdm import tqdm
from keras.preprocessing import image
from numpy import expand_dims
#from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.layers import Flatten,Activation,Concatenate,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import glob
#import cv2
def dataset(train_path,test_path):
    datagen = ImageDataGenerator()
    train_set = datagen.flow_from_directory(train_path, class_mode='categorical', batch_size=20,target_size=(224,224))
    test_set = datagen.flow_from_directory(test_path, class_mode='categorical', batch_size=20,target_size=(224,224))
    return train_set,test_set

def load_image_pixels(filename, shape):
    # load the image to get its shape
    image = load_img(filename)
    width, height = image.size
    # load the image with the required size
    image = load_img(filename, target_size=shape)
    # convert to numpy array
    image = img_to_array(image)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0
    # add a dimension so that we have one sample
    image = expand_dims(image, 0)
    return image, width, height
pre_train_model=keras.models.load_model('Yolov3_original.h5')
#model=keras.models.load_model('Yolov3_224_custom_.h5')
#pre_train_model.summary()
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>...\n\n")
for layer in pre_train_model.layers:
    layer.trainable = False
#model = Model(inputs=pre_train_model.inputs,outputs=pre_train_model.layers[-2].output) #outputs=[pre_train_model.layers[-3].output,pre_train_model.layers[-2].output,pre_train_model.layers[-1].output])
#model.summary()
input_w, input_h = 224, 224
# define our new photo
#photo_filename = 'pict3.jpg'
# load and prepare image
#image, image_w, image_h = load_image_pixels(photo_filename, (input_w, input_h))
# make prediction
#yhat = model.predict(image)
#print(yhat.shape)
# summarize the shape of the list of arrays
#print([a.shape for a in yhat])
#for layer in model.layers:
    #layer.trainable = False
#model.summary()

flat1 = Flatten()(pre_train_model.layers[-1].output)
flat2 = Flatten()(pre_train_model.layers[-2].output)
flat3 = Flatten()(pre_train_model.layers[-3].output)
#dd=Concatenate()([flat1,flat2,flat3])
#dd1=Concatenate()([dd,flat3])
class11 = Dense(512, activation='relu')(flat1)
bb11=BatchNormalization()(class11)
#x11 = Dropout(0.5)(bb11)
class12 = Dense(256, activation='relu')(bb11)
bb12=BatchNormalization()(class12)
#x12 = Dropout(0.5)(bb12)
class13 = Dense(128, activation='relu')(bb12)
bb13=BatchNormalization()(class13)
#x13 = Dropout(0.5)(bb13)
class14 = Dense(64, activation='relu')(bb13)
#...............................................................................
class21 = Dense(512, activation='relu')(flat2)
bb21=BatchNormalization()(class21)
#x21 = Dropout(0.5)(bb21)
class22 = Dense(256, activation='relu')(bb21)
bb22=BatchNormalization()(class22)
#x22 = Dropout(0.5)(bb22)
class23 = Dense(128, activation='relu')(bb22)
bb23=BatchNormalization()(class23)
#x23 = Dropout(0.5)(bb23)
class24 = Dense(64, activation='relu')(bb23)
#...........................................................................
class31 = Dense(512, activation='relu')(flat3)
bb31=BatchNormalization()(class31)
#x31 = Dropout(0.5)(bb31)
class32 = Dense(256, activation='relu')(bb31)
bb32=BatchNormalization()(class32)
#x32 = Dropout(0.5)(bb32)
class33 = Dense(128, activation='relu')(bb32)
bb33=BatchNormalization()(class33)
#x33 = Dropout(0.5)(bb33)
class34 = Dense(64, activation='relu')(bb33)
#............................................................................
dd=Concatenate()([class14,class24,class34])
bb4=BatchNormalization()(dd)
class5 = Dense(32, activation='relu')(bb4)
bb5=BatchNormalization()(class5)
class6 = Dense(16, activation='relu')(bb5)
bb6=BatchNormalization()(class6)
output = Dense(10, activation='softmax')(bb6)
'''..................................................................................'''

# define new model
model = Model(inputs=pre_train_model.inputs, outputs=output)
# summarize'''
model.summary()
#yhat=model.predict(image)
#print(yhat.shape)
#print(yhat)'''
train="dataset_3/train1/"
test="dataset_3/test1/"
train_set,test_set=dataset(train,test)
model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=['accuracy'])#loss="categorical_crossentropy",optimizer='adam',metrics=['accuracy']
history=model.fit(train_set,validation_data=test_set,epochs=30,shuffle=True)
#history=pre_train_model.evaluate(train_set)
#print(history)
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.title("Model Accuracy")
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
         
         
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
model.save("Yolov3_224_custom_prll2_typr_512.h5")
print(train_set.class_indices)

