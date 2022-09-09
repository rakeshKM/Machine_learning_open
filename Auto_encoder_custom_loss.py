
import re
import tqdm
import cv2
from ast import literal_eval
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#from sklearn.model_selection import train_test_split
import gzip

import tensorflow as tf
#tf.debugging.set_log_device_placement(True)
from tensorflow import keras
from tensorflow.keras.layers import Input,Conv2D,Conv2DTranspose,MaxPool2D,UpSampling2D,Flatten,Dense,GlobalAveragePooling2D,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator

import os
import base64
import io
from datetime import datetime



import ssl
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['http_proxy'] = 'http://proxy-chain.intel.com:911'
os.environ['HTTP_PROXY'] = 'http://proxy-chain.intel.com:911'
os.environ['https_proxy'] = 'https://proxy-chain.intel.com:912'
os.environ['HTTPS_PROXY'] = 'https://proxy-chain.intel.com:912'


#https://github.com/anikita/ImageNet_Pretrained_Autoencoder/blob/master/autoencoder.py
def Autoencoder_VGG(input_img, TRAINABLE=True):
  #base_model = VGG16(weights='imagenet')
  base_model = VGG16(weights="/home/cdsw/vgg16_weights_tf_dim_ordering_tf_kernels.h5")
  for layer in base_model.layers:
      layer.trainable=TRAINABLE    
  #-------------------encoder---------------------------- 
  #--------(pretrained & trainable if selected)----------
  #    block1
  x=base_model.get_layer('block1_conv1')(input_img)
  x=base_model.get_layer('block1_conv2')(x)
  x=conv1_1_batch_normed = BatchNormalization()(x , training=True)
  x=base_model.get_layer('block1_pool')(x)
  #    block2
  x=base_model.get_layer('block2_conv1')(x)
  x=base_model.get_layer('block2_conv2')(x)
  x=conv1_1_batch_normed = BatchNormalization()(x , training=True)
  x=base_model.get_layer('block2_pool')(x)
  #    block3
  x=base_model.get_layer('block3_conv1')(x)
  x=base_model.get_layer('block3_conv2')(x)
  x=conv1_1_batch_normed = BatchNormalization()(x , training=True)
  x=base_model.get_layer('block3_pool')(x)
  #    block4
  x=base_model.get_layer('block4_conv1')(x)
  x=base_model.get_layer('block4_conv2')(x) 
  x=conv1_1_batch_normed = BatchNormalization()(x , training=True)
  x=base_model.get_layer('block4_pool')(x)

  
  latent_space=x   
  x = Conv2D(512, (3, 3), activation='relu', padding='same',name='latent')(x)
  #--------------decoder (trainable)-----------        
  # Block 4
  x = UpSampling2D((2,2))(x) 
  x=conv1_1_batch_normed = BatchNormalization()(x , training=True)
  x = Conv2DTranspose(512, (3, 3), activation='relu', padding='same', name='dblock4_conv1')(x)
  x = Conv2DTranspose(512, (3, 3), activation='relu', padding='same', name='dblock4_conv2')(x)
   
  # Block 3
  x = UpSampling2D((2,2))(x)
  x=conv1_1_batch_normed = BatchNormalization()(x , training=True)
  x = Conv2DTranspose(256, (3, 3), activation='relu', padding='same', name='dblock3_conv1')(x)
  x = Conv2DTranspose(256, (3, 3), activation='relu', padding='same', name='dblock3_conv2')(x)
       
  # Block 2
  x = UpSampling2D((2,2))(x)  
  x=conv1_1_batch_normed = BatchNormalization()(x , training=True)
  x = Conv2DTranspose(128, (3, 3), activation='relu', padding='same', name='dblock2_conv1')(x)
  x = Conv2DTranspose(128, (3, 3), activation='relu', padding='same', name='dblock2_conv2')(x)

  # Block 1
  x = UpSampling2D((2,2))(x)
  x=conv1_1_batch_normed = BatchNormalization()(x , training=True)
  x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same', name='dblock1_conv1')(x)
  x = Conv2DTranspose(3, (3, 3), activation='relu', padding='same', name='dblock1_conv2')(x)
  
   
  return x,latent_space


def create_mask(X):
  images=[]
  mask_p1=[]
  mask_n1=[]
  for xi in X:
    images_data,bbox_new=xi[0],xi[1]
    images.append(images_data)
    mask=np.ones(images_data.shape)
    y1,x1,y2,x2=bbox_new           #needed to be chnaged
    if x1 != -1.0:
      mask[int(x1):int(x2),int(y1):int(y2)]=0 
      mask_p= np.ones(images_data.shape) - mask
      mask_n=  mask  
    else:
      mask_p=np.zeros(images_data.shape)
      mask_n=np.ones(images_data.shape)
    mask_p1.append(mask_p) 
    mask_n1.append(mask_n) 
  return np.array(images),np.array(mask_p1),np.array(mask_n1)

def assymetric_fun(input_images,positive_mask,negative_mask,recons_images):  
  within_box_image=  input_images*positive_mask
  outside_box_image=  input_images*negative_mask    
  within_box_recon=recons_images*positive_mask
  outside_box_recon=recons_images*negative_mask          
  L1 = tf.reduce_sum(tf.square(tf.subtract(within_box_image, within_box_recon)),axis=[1, 2, 3])
  L2=  tf.reduce_sum(tf.square(tf.subtract(outside_box_image, outside_box_recon)),axis=[1, 2, 3])        
  return L1,L2


### The Custom Loop
# The train_on_batch function


# Compile the model
input_images = Input(shape = (512, 512, 3))
positive_mask= Input(shape = (512, 512, 3))
negative_mask= Input(shape = (512, 512, 3))
recons_images,latent_space=Autoencoder_VGG(input_images)
autoencoder_model=Model(inputs=[input_images,positive_mask,negative_mask], outputs=[recons_images])


#optimizer = tf.keras.optimizers.Adam()
optimizer1 =tf.keras.optimizers.Adam(1e-4)
optimizer2 =tf.keras.optimizers.Adam(1e-4)
def train_on_batch(X):
  with tf.GradientTape(persistent=True) as tape:
    X_original,X_p,X_n=create_mask(X)
    # Forward pass.
    #y = model(X, training=True)
    recon=autoencoder_model([X_original,X_p,X_n],training=True)
    # Loss value for this batch.
    #loss_value = loss(y, y)
    loss_value1_,loss_value2=assymetric_fun(X_original,X_p,X_n,recon)
    loss_value1=-1.0*loss_value1_
    
  # Update the weights of the model to minimize the loss value.
  grads1 = tape.gradient(loss_value1, autoencoder_model.trainable_variables)
  grads2 = tape.gradient(loss_value2, autoencoder_model.trainable_variables)
  
  #optimizer.apply_gradients(zip(grads, autoencoder_model.trainable_variables))
  optimizer1.apply_gradients(zip(grads1, autoencoder_model.trainable_variables))
  optimizer2.apply_gradients(zip(grads2, autoencoder_model.trainable_variables))
  
  #optimizer1.minimize(assymetric_fun1(input_images,positive_mask,recons_images), autoencoder_model.trainable_variables)
  #optimizer2.minimize(assymetric_fun2(input_images,negative_mask,recons_images), autoencoder_model.trainable_variables)
  return loss_value1_+loss_value2

# The validate_on_batch function
def validate_on_batch(X):
  X_original,X_p,X_n=create_mask(X)
  # Forward pass.
  #y = model(X, training=True)
  recon=autoencoder_model([X_original,X_p,X_n],training=False)
  # Loss value for this batch.
  #loss_value = loss(y, y)
  loss_value1,loss_value2=assymetric_fun(X_original,X_p,X_n,recon)
  return loss_value1+loss_value2


#-------load training data

img_file_dir="/home/cdsw/images_augmented_512_individual_annotation_separate_train_val/images/train/" 
pd_levels=pd.read_excel("/home/cdsw/images_augmented_512_individual_annotation_separate_train_val/lebels_train_augmented.xlsx")

img_file_name=pd_levels.Image_file_name
img_lebel=pd_levels.lebel
img_bboxes=pd_levels.bboxes

images_bbox = []
for current_img,lebel,bbox in tqdm.tqdm(zip(img_file_name,img_lebel,img_bboxes)):
  img = cv2.imread(img_file_dir+current_img,cv2.IMREAD_COLOR)
  #print(type(bbox),bbox)
  #numpy_img = img_to_array(img)
  numpy_img= cv2.normalize(img, None, alpha=0, beta=1,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
  try:
    literal_eval(bbox)[0] == -1.0 
    bbox_temp=[-1.0,-1.0,-1.0,-1.0]
  except:
    extract_float=re.findall(r"[-+]?\d*\.\d+|\d+", str(bbox))
    bbox_temp=[float(i) for i in extract_float]
  images_bbox.append([numpy_img,bbox_temp])

X_train=images_bbox

img_file_dir="/home/cdsw/images_augmented_512_individual_annotation_separate_train_val/images/val/" 
pd_levels=pd.read_excel("/home/cdsw/images_augmented_512_individual_annotation_separate_train_val/lebels_val_augmented.xlsx")

img_file_name=pd_levels.Image_file_name
img_lebel=pd_levels.lebel
img_bboxes=pd_levels.bboxes

images_bbox = []
for current_img,lebel,bbox in tqdm.tqdm(zip(img_file_name,img_lebel,img_bboxes)):
  img = cv2.imread(img_file_dir+current_img,cv2.IMREAD_COLOR)
  #print(type(bbox),bbox)
  #numpy_img = img_to_array(img)
  numpy_img= cv2.normalize(img, None, alpha=0, beta=1,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
  try:
    literal_eval(bbox)[0] == -1.0 
    bbox_temp=[-1.0,-1.0,-1.0,-1.0]
  except:
    extract_float=re.findall(r"[-+]?\d*\.\d+|\d+", str(bbox))
    bbox_temp=[float(i) for i in extract_float]
  images_bbox.append([numpy_img,bbox_temp])

X_test=images_bbox

# Training Loop


def create_mask(X):
  images=[]
  mask_p1=[]
  mask_n1=[]
  for xi in X:
    images_data,bbox_new=xi[0],xi[1]
    images.append(images_data)
    mask=np.ones(images_data.shape)
    y1,x1,y2,x2=bbox_new
    if x1 != -1.0:
      mask[int(x1):int(x2),int(y1):int(y2)]=0 
      mask_p= np.ones(images_data.shape) - mask
      mask_n=  mask  
    else:
      mask_p=np.zeros(images_data.shape)
      mask_n=np.ones(images_data.shape)
    mask_p1.append(mask_p) 
    mask_n1.append(mask_n) 
  return np.array(images),np.array(mask_p1),np.array(mask_n1)

batch_size = 4
epochs = 50

train_loss_epoch=[]
val_loss_epoch=[]



for epoch in range(0, epochs):
  train_loss=[]
  for i in range(0, len(X_train) // batch_size):
      X = X_train[i * batch_size:min(len(X_train), (i+1) * batch_size)]
      loss=train_on_batch(X)  
      train_loss.append(loss)
      if i%100 ==0:
        print('\rEpoch [%d/%d] Batch: %d%s' % (epoch + 1, epochs, (i+1), '.' * ((i+1) % 10)), end='')
  print(' training Loss: ' + str(np.mean(train_loss)))   
  val_loss = []
  for i in range(0, len(X_test) // batch_size):
      X = X_test[i * batch_size:min(len(X_test), (i+1) * batch_size)]
      val_loss.append(validate_on_batch(X))  
  print(' Validation Loss: ' + str(np.mean(val_loss)))

  train_loss_epoch.append(np.mean(train_loss))
  val_loss_epoch.append(np.mean(val_loss))


epoch=[i+1 for i in range(len(train_loss_epoch))]
fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
ax.plot(epoch,train_loss_epoch, epoch,val_loss_epoch )
fig.savefig('training_curve_vgg_4.png')   # save the figure to file
plt.close(fig)    # close the figure window
#intresting https://towardsdatascience.com/build-a-simple-image-retrieval-system-with-an-autoencoder-673a262b7921
'''
#to check gradient value
for i in range(len(grads1)):
  print(np.count_nonzero(grads[i].numpy()))

base_models = VGG16(weights="/home/cdsw/vgg16_weights_tf_dim_ordering_tf_kernels.h5")
for layer in base_models.layers:
  print(layer.name)
'''
