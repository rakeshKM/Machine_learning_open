import re
import tqdm
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input,Conv2D,Conv2DTranspose,MaxPool2D,UpSampling2D,Flatten,Dense,Reshape,GlobalAveragePooling2D,BatchNormalization
from tensorflow.keras.models import Model

from ast import literal_eval

#------------------model defination-----------

#define model
input_images = Input(shape = (512, 512, 3))

# Encoder
# batch norm layer helping in vanishing gradient
#https://towardsdatascience.com/how-to-use-batch-normalization-with-tensorflow-and-tf-keras-to-train-deep-neural-networks-faster-60ba4d054b73

# defining conv autoencodeer https://towardsdatascience.com/deep-inside-autoencoders-7e41f319999f
# replacing stride polling with stride https://medium.com/analytics-vidhya/building-a-convolutional-autoencoder-using-keras-using-conv2dtranspose-ca403c8d144e


conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same',name='conv1')(input_images)
conv_2= Conv2D(128, (3, 3), activation='relu', padding='same',name='conv2')(conv_1)
batch_normed_c1 = BatchNormalization()(conv_2 , training=True)
pool_1 = MaxPool2D((2, 2), padding='same')(batch_normed_c1)    

conv_3 = Conv2D(256, (3, 3), activation='relu', padding='same',name='conv3')(pool_1 )
batch_normed_c2 = BatchNormalization()(conv_3  , training=True)
pool_2 = MaxPool2D((2, 2), padding='same')(batch_normed_c2)

conv_4 = Conv2D(512, (3, 3), activation='relu', padding='same',name='conv4')(pool_2)
batch_normed_c3 = BatchNormalization()(conv_4  , training=True)
pool_3 = MaxPool2D((2, 2), padding='same')(batch_normed_c3)

#h=pool_4

#to test why sparshess around middle layer of autoenoder

# towards classifier
#l = Flatten()(e)
#l = Dense(256, activation='softmax')(l)#DECODER
#d = Reshape((7,7,1))(l)

#F = Flatten()(pool_4)
#h = Dense(256, activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-5))(F)
#shape_list=pool_4.get_shape().as_list()
#IF=Reshape((shape_list[1],shape_list[2],shape_list[3]))(h)
#IF=tf.reshape(h,[-1,shape_list[1],shape_list[2],shape_list[3]])

h =Conv2DTranspose(512, (3, 3), activation='relu', padding='same',name='latent')(pool_3)



# Decoder
up_pool_3 = UpSampling2D((2, 2))(h)
batch_normed_d4 = BatchNormalization()(up_pool_3 , training=True)
dconv_4 = Conv2DTranspose(512, (3, 3), activation='relu', padding='same',name='dconv4')(batch_normed_d4)

up_pool_2 = UpSampling2D((2, 2))(dconv_4)
batch_normed_d3 = BatchNormalization()(up_pool_2, training=True)
dconv_3 = Conv2DTranspose(256, (3, 3), activation='relu', padding='same',name='dconv3')(batch_normed_d3)

up_pool_1 = UpSampling2D((2, 2))(dconv_3)
batch_normed_d3 = BatchNormalization()(up_pool_1  , training=True)
dconv_2 = Conv2DTranspose(128, (3, 3), activation='relu',padding='same',name='dconv2')(batch_normed_d3)
dconv_1 = Conv2DTranspose(64, (3, 3), activation='relu',padding='same',name='dconv1')(dconv_2)

r = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same',name='recon')(dconv_1)  # recon pixel value sometime more than 1,so sigmoid at the end 
                                                                        

autoencoder_model=Model(inputs=[input_images], outputs=[r,h])

#---------defining the loss function 



def train_on_batch(X_original,X_p,X_n,optimizer1,optimizer2):  
  
  '''
  recon=autoencoder_model([X_original,X_p,X_n],training=True)    
  within_box_image=  X_original * X_p 
  within_box_recon= recon*X_p
  #l1 = tf.reduce_sum(tf.square(tf.subtract(within_box_image, within_box_recon)),axis=[1, 2, 3])
  def ll1():
    return tf.reduce_sum(tf.keras.losses.MSE(within_box_image, within_box_recon))

  outside_box_image=  X_original * X_n  
  outside_box_recon=recon*X_n 
  #l2=  tf.reduce_sum(tf.square(tf.subtract(outside_box_image, outside_box_recon)),axis=[1, 2, 3])        
  def ll2():
      return tf.reduce_sum(tf.keras.losses.MSE(outside_box_image, outside_box_recon))

  op1=optimizer1.minimize(ll1, autoencoder_model.trainable_variables)  
  op2=optimizer2.minimize(ll2, autoencoder_model.trainable_variables)
  '''
  
  with tf.GradientTape(persistent=True) as tape:
    
    # Forward pass.
    #y = model(X, training=True)
    recon,latent=autoencoder_model([X_original],training=True) 
    
    # Loss value for this batch.
    #loss_value = loss(y, y_)
    within_box_image=   X_original * tf.cast(X_p,tf.float32) 
    within_box_recon=   recon      * tf.cast(X_p,tf.float32)
    outside_box_image=  X_original * tf.cast(X_n,tf.float32)  
    outside_box_recon=  recon      * tf.cast(X_n,tf.float32) 
    l1 = tf.reduce_sum(tf.square(tf.subtract(within_box_image, within_box_recon)),axis=[1, 2, 3])
    l2=  tf.reduce_sum(tf.square(tf.subtract(outside_box_image, outside_box_recon)),axis=[1, 2, 3])        
    loss_value1=-20*l1
    loss_value2= l2
          
  # Update the weights of the model to minimize the loss value.
  grads1 = tape.gradient(loss_value1, autoencoder_model.trainable_variables)
  grads2 = tape.gradient(loss_value2, autoencoder_model.trainable_variables)
  
  #optimizer.apply_gradients(zip(grads, autoencoder_model.trainable_variables))
  optimizer1.apply_gradients(zip(grads1, autoencoder_model.trainable_variables))
  optimizer2.apply_gradients(zip(grads2, autoencoder_model.trainable_variables))
  
  return l1+l2,recon

# The validate_on_batch function
def validate_on_batch(X_original):
  
  # Forward pass.
  #y = model(X, training=True)
  recon,latent=autoencoder_model([X_original],training=False)
  
  # Loss value for this batch.
  #loss_value = loss(y, y)
  loss_value=tf.reduce_sum(tf.square(tf.subtract(X_original,recon)),axis=[1, 2, 3]) 
  
  return loss_value,recon

#-------load training data

img_file_dir="/home/cdsw/images_augmented_512_individual_annotation_separate_train_val/images/train/" 
pd_levels=pd.read_excel("/home/cdsw/images_augmented_512_individual_annotation_separate_train_val/lebels_train_augmented.xlsx")
img_file_name=pd_levels.Image_file_name
img_lebel=pd_levels.lebel
img_bboxes=pd_levels.bboxes
images_bbox_lebels = []
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
  images_bbox_lebels.append([numpy_img,bbox_temp,lebel])
train_data=images_bbox_lebels 

img_file_dir="/home/cdsw/images_augmented_512_individual_annotation_separate_train_val/images/val/" 
pd_levels=pd.read_excel("/home/cdsw/images_augmented_512_individual_annotation_separate_train_val/lebels_val_augmented.xlsx")
img_file_name=pd_levels.Image_file_name
img_lebel=pd_levels.lebel
img_bboxes=pd_levels.bboxes
images_bbox_lebels = []
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
  images_bbox_lebels.append([numpy_img,bbox_temp,lebel])

val_data=images_bbox_lebels

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

#----------------starting training-------
batch_size = 4

epochs = 100

myLearnRate1=1e-4
myLearnRate2=1e-4
#optimizer = tf.keras.optimizers.Adam()
optimizer1 =tf.keras.optimizers.Adam(1e-4)
optimizer2 =tf.keras.optimizers.Adam(1e-4)

train_loss_epoch=[]
val_loss_epoch=[]

X_train,X_test=train_data,val_data

for epoch in range(0, epochs):
  train_loss=[]
  for i in range(0, len(X_train) // batch_size):
      X = X_train[i * batch_size:min(len(X_train), (i+1) * batch_size)]
      Image,Mask_p,Mask_n=create_mask(X)
      
      #https://stats.stackexchange.com/questions/201129/training-loss-goes-down-and-up-again-what-is-happening?newreg=44f7070af879400799f93829fa187c5f
      Lr1= myLearnRate1 #/(1 + (epoch/37))    
      Lr2= myLearnRate2 #/(1 + (epoch/37))   
      optimizer1 =tf.keras.optimizers.Adam(learning_rate=Lr1)
      optimizer2 =tf.keras.optimizers.Adam(learning_rate=Lr2)
      
      loss,recon=train_on_batch(Image,Mask_p,Mask_n,optimizer1,optimizer2)  
      
      train_loss.append(loss)
      if i%500 ==0:
        print('\rEpoch [%d/%d] Batch: %d%s' % (epoch + 1, epochs, (i+1), '.' * ((i+1) % 10)), end='')
  print(' training Loss: ' + str(np.mean(train_loss)))   
  val_loss = []
  for i in range(0, len(X_test) // batch_size):
      X = X_test[i * batch_size:min(len(X_test), (i+1) * batch_size)]
      Image,Mask_p,Mask_n=create_mask(X)
      loss,recon=validate_on_batch(Image)
      val_loss.append(loss)  
  print(' Validation Loss: ' + str(np.mean(val_loss)))
  
  train_loss_epoch.append(np.mean(train_loss))
  val_loss_epoch.append(np.mean(val_loss))
  

#--------------print learning curve
epoch=[i+1 for i in range(len(train_loss_epoch))]
fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
ax.plot(epoch,train_loss_epoch, epoch,val_loss_epoch )
fig.savefig('training_curve_simple2_3.png')   # save the figure to file
plt.close(fig)    # close the figure window
#intresting https://towardsdatascience.com/build-a-simple-image-retrieval-system-with-an-autoencoder-673a262b7921


#------------------ saving the recon image
for i in range(0, len(X_test) // batch_size):
    X = X_test[i * batch_size:min(len(X_test), (i+1) * batch_size)]
    X_original,X_p,X_n=create_mask(X)
    recon,h=autoencoder_model([X_original,X_p,X_n],training=False)
    Image_test,Mask_p,_,recon_test=X_original,X_p,X_n,recon
    
    fig, [[ax1,m1, ax2],[ax3,m2, ax4],[ax5,m3,ax6],[ax7,m4,ax8]] = plt.subplots(nrows=4, ncols=3)
    ax1.imshow(Image_test[0])
    m1.imshow(Mask_p[0])
    ax2.imshow(recon_test[0])
    ax3.imshow(Image_test[1])
    m2.imshow(Mask_p[1])
    ax4.imshow(recon_test[1])
    ax5.imshow(Image_test[2])
    m3.imshow(Mask_p[2])
    ax6.imshow(recon_test[2])
    ax7.imshow(Image_test[3])
    m4.imshow(Mask_p[3])
    ax8.imshow(recon_test[3])
    fig.suptitle('testing_recon_curve_simple2_3_{}'.format(i+1))
    fig.savefig('/home/cdsw/Recon/simple2_3_test/testing_recon_curve_simple2_3_{}.png'.format(i+1))  
    
    plt.close()
    
X_train_recon=X_train[:1200]

for i in range(0, len(X_train_recon) // batch_size):
    X = X_train_recon[i * batch_size:min(len(X_train_recon), (i+1) * batch_size)]
    X_original,X_p,X_n=create_mask(X)
    recon,h=autoencoder_model([X_original,X_p,X_n],training=False)
    Image_train,Mask_p,_,recon_train=X_original,X_p,X_n,recon
    
    fig, [[ax1,m1, ax2],[ax3,m2, ax4],[ax5,m3,ax6],[ax7,m4,ax8]] = plt.subplots(nrows=4, ncols=3)
    ax1.imshow(Image_train[0])
    m1.imshow(Mask_p[0])
    ax2.imshow(recon_train[0])
    ax3.imshow(Image_train[1])
    m2.imshow(Mask_p[1])
    ax4.imshow(recon_train[1])
    ax5.imshow(Image_train[2])
    m3.imshow(Mask_p[2])
    ax6.imshow(recon_train[2])
    ax7.imshow(Image_train[3])
    m4.imshow(Mask_p[3])
    ax8.imshow(recon_train[3])
    fig.suptitle('training_recon_curve_simple2_3_{}'.format(i+1))
    fig.savefig('/home/cdsw/Recon/simple2_3_train/training_recon_curve_simple2_3_{}.png'.format(i+1))  
    
    plt.close()
    
    
  #-------------measuring accuracy----------------
  
  
images_lebel_error = []
for i in range(0, len(X_test)):
  X=X_test[i]
  img=X[0]
  label=X[2]
  img2=np.expand_dims(img, axis=0)
  L2loss,recon=validate_on_batch([img2])  
  images_lebel_error.append([current_img,lebel,L2loss])
  
  recon_diff= img2[0] -recon[0]
  #plt.rcParams["axes.grid"] = False
  fig = plt.figure()
  plt.imshow(recon_diff)  
  fig.suptitle('difference_imagessimple2_3_{}'.format(i+1))
  fig.savefig('/home/cdsw/Recon/simple2_3_diff/difference_images_simple2_3_{}.png'.format(i))  
  plt.close()

Data_lebels_train_original=pd.DataFrame(images_lebel_error,columns=["Image_file_name","lebel",'recon_error'])
Data_lebels_train_original.to_excel("./images_augmented_512_individual_annotation_separate_train_val/simple2_3_recon_error.xlsx")  
