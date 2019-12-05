#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow.keras.layers as Layers
import tensorflow.keras.activations as Actications
import tensorflow.keras.models as Models
import tensorflow.keras.optimizers as Optimizer
import tensorflow.keras.metrics as Metrics
import tensorflow.keras.utils as Utils
from keras.utils.vis_utils import model_to_dot
import os
import matplotlib.pyplot as plot
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix as CM
from random import randint
from IPython.display import SVG
import matplotlib.gridspec as gridspec

def get_images(directory):
    Images = []
    Labels = []  # 0 for Building , 1 for forest, 2 for glacier, 3 for mountain, 4 for Sea , 5 for Street
    label = 0
    
    for labels in os.listdir(directory): #Main Directory where each class label is present as folder name.
        if labels == 'glacier': #Folder contain Glacier Images get the '2' class label.
            label = 2
        elif labels == 'sea':
            label = 4
        elif labels == 'buildings':
            label = 0
        elif labels == 'forest':
            label = 1
        elif labels == 'street':
            label == 5
        elif labels == 'mountain':
            label == 3
        
        for image_file in os.listdir(directory+labels): #Extracting the file name of the image from Class Label folder
            image = cv2.imread(directory+labels+r'/'+image_file) #Reading the image (OpenCV)
            image = cv2.resize(image,(150,150)) #Resize the image, Some images are different sizes. (Resizing is very Important)
            Images.append(image)
            Labels.append(label)
    
    return shuffle(Images,Labels,random_state=817328462) #Shuffle the dataset you just prepared.

def get_classlabel(class_code):
    labels = {2:'glacier', 4:'sea', 0:'buildings', 1:'forest', 5:'street', 3:'mountain'}
    
    return labels[class_code]
	
train_Images, train_Labels = get_images('seg_train') #Extract the training images from the folders.

train_Images = np.array(train_Images) #converting the list of images to numpy array.
train_Labels = np.array(train_Labels)


test_Images, test_Labels = get_images('seg_test') #Extract the test images from the folders.

test_Images = np.array(test_Images) #converting the list of images to numpy array.
test_Labels = np.array(test_Labels)



print("Shape of Images:",train_Images.shape)
print("Shape of Labels:",train_Labels.shape)
print("Shape of Images:",test_Images.shape)
print("Shape of Labels:",test_Labels.shape)

f,ax = plot.subplots(5,5) 
f.subplots_adjust(0,0,3,3)
for i in range(0,5,1):
    for j in range(0,5,1):
        rnd_number = randint(0,len(train_Images))
        ax[i,j].imshow(train_Images[rnd_number])
        ax[i,j].set_title(get_classlabel(Labels[rnd_number]))
        ax[i,j].axis('off')


# In[2]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import os,shutil,math,scipy,cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix,roc_curve,auc
#from IPython.display import display
from PIL import Image
from PIL import Image as pil_image
from time import time
from PIL import ImageDraw
from glob import glob
from tqdm import tqdm
#from skimage.io import imread
#from IPython.display import SVG

from scipy import misc,ndimage
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import imread

from keras import backend as K
from keras import layers
from keras.preprocessing.image import save_img
from keras.utils.vis_utils import model_to_dot
from keras.applications.vgg19 import VGG19,preprocess_input
from keras.applications.xception import Xception
from keras.applications.nasnet import NASNetMobile
from keras.models import Sequential,Input,Model
from keras.layers import Dense,Flatten,Dropout,Concatenate,GlobalAveragePooling2D,Lambda,ZeroPadding2D
from keras.layers import SeparableConv2D,BatchNormalization,MaxPooling2D,Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,SGD
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard,CSVLogger,ReduceLROnPlateau,LearningRateScheduler


# In[3]:


train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True,validation_split=0.3)


# In[4]:


test_datagen = ImageDataGenerator(rescale = 1./255)


# In[5]:


training_set = train_datagen.flow_from_directory('seg_train',target_size=(64,64),batch_size=8,class_mode = 'categorical')


# In[6]:


test_set = test_datagen.flow_from_directory('seg_test',target_size=(64,64),batch_size=8,class_mode = 'categorical')


# In[7]:


training_set.target_size,test_set.target_size


# In[8]:


train_size = training_set.n
test_size = test_set.n

train_size,test_size


# In[10]:
from keras.models import Sequential
from keras.layers import  Dense, Dropout, Flatten, Conv2D, MaxPooling2D

def ConvBlock(model, layers, filters,name):
    for i in range(layers):
        model.add(Conv2D(filters, (3, 3), activation='relu',name=name))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
def vgg16():
    model = Sequential()
    model.add(Lambda(lambda x: x, input_shape=(64, 64, 3)))
    ConvBlock(model, 1, 64,'block_1')
    ConvBlock(model, 1, 128,'block_2')
    ConvBlock(model, 1, 256,'block_3')
    ConvBlock(model, 1, 512,'block_4')
    model.add(Flatten())
    model.add(Dense(1024,activation='relu'))
    model.add(Dense(4096,activation='relu'))
    model.add(Dense(6,activation='softmax'))
    return model

model = vgg16()
model.summary()


# In[11]:


opt = SGD(lr=1e-4,momentum=0.99)
opt1 = Adam(lr=2e-4)

model.compile(
    loss='binary_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)


# In[12]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


from PIL import Image
history = model.fit_generator(training_set,steps_per_epoch = train_size,epochs = 8,validation_data = test_set,validation_steps  = test_size)


# In[ ]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('modelacc')
plt.show()


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('modelloss')
plt.show()

