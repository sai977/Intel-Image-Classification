import pandas as pd
import os,shutil,math,scipy,cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pydot
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from keras.utils.np_utils import to_categorical
from sklearn.metrics import categorical_accuracy, top_k_categorical_accuracy,confusion_matrix,roc_curve,auc
from PIL import Image
from PIL import Image as pil_image
from time import time
from PIL import ImageDraw
from glob import glob
from tqdm import tqdm
from scipy import misc,ndimage
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import imread
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model
from keras import backend as K
from keras import layers
from keras.preprocessing.image import save_img
from keras.utils.vis_utils import model_to_dot
from keras.applications.vgg19 import VGG19,preprocess_input
from keras.models import Sequential,Input,Model
from keras.layers import Dense,Flatten,Dropout,Concatenate,GlobalAveragePooling2D,Lambda,ZeroPadding2D,SeparableConv2D,BatchNormalization,MaxPooling2D,Conv2D
from keras.optimizers import Adam,SGD
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard,CSVLogger,ReduceLROnPlateau,LearningRateScheduler
from IPython.display import SVG

#data generation
train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
valid_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
test_datagen = ImageDataGenerator()

training_set = train_datagen.flow_from_directory('seg_train',target_size=(150,150),batch_size=8,class_mode = 'categorical')
valid_set = valid_datagen.flow_from_directory('seg_test',target_size=(150,150),batch_size=8,class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('seg_pred',target_size=(150,150),batch_size=8,class_mode = None,)

batch_size=8
training_set.target_size,valid_set.target_size,test_set.target_size

train_size = training_set.n
valid_size = valid_set.n
test_size =test_set.n

train_size,valid_size,test_size

# create the base pre-trained model
base_model = VGG19(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(6, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)
# train the model on the new data for a few epochs

#custom metrics 
def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

# Compile the model
model.compile(optimizer=SGD(lr=1e-4,momentum=0.99), loss='categorical_crossentropy', metrics=['accuracy',categorical_accuracy, top_2_accuracy, top_3_accuracy])

model.summary()

#train the model
history = model.fit_generator(training_set,steps_per_epoch = train_size/batch_size,epochs = 12,validation_data = valid_set,validation_steps  = valid_size/batch_size)
#evaluating the model
eval = model.evaluate_generator(generator=valid_set,steps=valid_size/batch_size)
print('accuracy=',eval[1])

y_pred=model.predict_generator(generator=test_set,steps=valid_size//batch_size)
y_pred=np.argmax(y_pred,axis=1)

print('confusion matrix')
print(confusion_matrix(valid_set.classes,y_pred))
#confusion_matrix
cm=confusion_matrix(valid_set.classes,y_pred)
cm_df=pd.DataFrame(cm,index=['buildings','foresst','glacier','mountain','sea','street'],columns=['buildings','foresst','glacier','mountain','sea','street'])

plt.figure(figsize=(5,5))
sns.heatmap(cm_df,annot=True)
plt.title('confusion matrix')
plt.ylabel('true label')
plt.xlabel('predicted label')
plt.savefig('confusion')
plt.show()
#classification_report
print('classification report')
target_names=['buildings','foresst','glacier','mountain','sea','street']

print(classification_report(valid_set.classes,y_pred,target_names=target_names))
#test set prediction
test_set.reset()
pred=model.predict_generator(test_set,steps=test_size/batch_size,verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)

labels = (training_set.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

#saving the predictions to csv
filenames=test_set.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("vgg19results.csv",index=False)


#plotting accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('modelacc')
plt.show()

#plotting loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('modelloss')
plt.show()

