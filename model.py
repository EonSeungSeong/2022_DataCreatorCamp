
import tensorflow as tf
import tensorflow.keras
from keras import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn import metrics
from keras.datasets import cifar10
from keras.utils import to_categorical,load_img,img_to_array
from keras.preprocessing.image import ImageDataGenerator
from tensorflow_addons.optimizers import AdamW,SGDW
from sklearn.metrics import f1_score,confusion_matrix,classification_report
#Convolution Neural Networks Filter Image
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, UpSampling2D
# Convolution Auto Encoder Filter Image
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D,Concatenate ,Conv2DTranspose,GlobalAveragePooling2D,add,ZeroPadding2D,concatenate
from keras import optimizers
from keras.models import Model
from keras import backend as K
from keras import regularizers
from keras.layers import Input,Layer
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D ,LeakyReLU,ReLU, Concatenate,Flatten,BatchNormalization,Dropout,ReLU,LeakyReLU,PReLU
from keras import optimizers
from keras.models import Model
import gc
from sklearn.metrics import f1_score,confusion_matrix,classification_report
import collections
from keras.utils import to_categorical

import tensorflow as tf

model_configs = {} 


def model_fn(x,y,inChannel,n_class):
    act=ReLU()
    input_img = Input(shape=(x,y,inChannel))
    vgg = Conv2D(64, (3,3),activation=act,padding='same',kernel_regularizer=regularizers.l2(0.001))(input_img)
    vgg = Conv2D(64,(3,3),activation=act,padding='same',kernel_regularizer=regularizers.l2(0.001))(vgg)
    vgg = MaxPooling2D(pool_size=(2,2))(vgg)
    vgg = Dropout(0.2)(vgg)

    vgg = Conv2D(128,(3,3),activation=act,padding='same',kernel_regularizer=regularizers.l2(0.001))(vgg)
    vgg = Conv2D(128,(3,3),activation=act,padding='same',kernel_regularizer=regularizers.l2(0.001))(vgg)
    vgg = MaxPooling2D(pool_size=(2,2))(vgg)
    vgg = Dropout(0.2)(vgg)

    vgg1 = Conv2D(256,(3,3),activation=act,padding='same',kernel_regularizer=regularizers.l2(0.001))(vgg)
    vgg = Conv2D(256,(3,3),activation=act,padding='same',kernel_regularizer=regularizers.l2(0.001))(vgg1)
    vgg = Conv2D(256,(3,3),padding='same',kernel_regularizer=regularizers.l2(0.001))(vgg)
    vgg = add([vgg1,vgg])
    vgg = ReLU()(vgg)
    vgg = MaxPooling2D(pool_size=(2,2))(vgg)
    vgg = Dropout(0.2)(vgg)

    vgg2 = Conv2D(256,(3,3),activation=act,padding='same',kernel_regularizer=regularizers.l2(0.001))(vgg)
    vgg = Conv2D(256,(3,3),activation=act,padding='same',kernel_regularizer=regularizers.l2(0.001))(vgg2)
    vgg = Conv2D(256,(3,3),padding='same',kernel_regularizer=regularizers.l2(0.001))(vgg)
    vgg = add([vgg2,vgg])
    vgg = ReLU()(vgg)
    vgg = MaxPooling2D(pool_size=(2,2))(vgg)
    vgg = Dropout(0.2)(vgg)

    vgg3 = Conv2D(512,(3,3),activation=act,padding='same',kernel_regularizer=regularizers.l2(0.001))(vgg)
    vgg = Conv2D(512,(3,3),activation=act,padding='same',kernel_regularizer=regularizers.l2(0.001))(vgg3)
    vgg = Conv2D(512,(3,3),padding='same',kernel_regularizer=regularizers.l2(0.001))(vgg)
    vgg = add([vgg3,vgg])
    vgg = ReLU()(vgg)
    vgg = MaxPooling2D(pool_size=(2,2))(vgg)
    vgg = Dropout(0.2)(vgg)

    vgg3 = Conv2D(512,(3,3),activation=act,padding='same',kernel_regularizer=regularizers.l2(0.001))(vgg)
    vgg = Conv2D(512,(3,3),activation=act,padding='same',kernel_regularizer=regularizers.l2(0.001))(vgg3)
    vgg = Conv2D(512,(3,3),padding='same',kernel_regularizer=regularizers.l2(0.001))(vgg)
    vgg = add([vgg3,vgg])
    vgg = ReLU()(vgg)
    vgg = MaxPooling2D(pool_size=(2,2))(vgg)
    vgg = Dropout(0.2)(vgg)

    vgg = Flatten()(vgg)
    vgg = Dense(4096,activation='relu')(vgg)
    vgg = Dropout(0.2)(vgg)
    vgg = Dense(4096,activation='relu')(vgg)
    vgg = Dropout(0.2)(vgg)
    vgg = Dense(n_class,activation='softmax')(vgg)

    vgg16 = Model(input_img , vgg)
    return vgg16    

def get_classifier(num_classes=20):
    x = 224
    y = 224
    inChannel = 3
    num_class= 20
    model = model_fn(x,y,inChannel, num_class)
    return model




