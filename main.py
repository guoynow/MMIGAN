import jieba
import numpy as np
import tensorflow as tf

from tensorflow.python.keras import Model
from tensorflow.python import keras
from tensorflow.python.ops.variable_scope import get_variable
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences

import imageCNN_fusion

import time
from torch import nn
from torch.nn import MaxPool2d

import pretraining
import textCNN
import textCNN_fusion
import textCNN_ori
import utilss
import imageCNN
import gc
import imageCNN_fusion_unet_image_mask_gan

from textCNN_ori import TextCNN

class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1= tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[3,3],
            padding='same',
            strides=(1, 1),
            activation= tf.nn.relu

        )
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2,2],strides=2)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[3,3],
            padding='same',
            strides=(1, 1),
            activation=tf.nn.relu
        )
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2,2],strides=2)
        self.conv3 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[3, 3],
            padding='same',
            strides=(1, 1),
            activation=tf.nn.relu
        )  
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=1024,activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self,inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)

        return x

class de_CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.unpool1 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.unconv1 = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(1, 1), activation='relu', padding='same', )
        self.unpool2 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.unconv2 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(1, 1), activation='relu', padding='same', )
        self.unpool3 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.unconv3 = tf.keras.layers.Conv2DTranspose(3, (3, 3), strides=(1, 1), activation='relu', padding='same', )
    def call(self, inputs):
        x = self.unpool1(inputs)
        x = self.unconv1(x)
        x = self.unpool2(x)
        x = self.unconv2(x)
        x = self.unpool3(x)
        x = self.unconv3(x)
        return x
class encoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(64, (3, 3), input_shape=(224, 224, 3), padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.conv4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',)
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same',)
        self.conv6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same',)
        self.conv7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same',)
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv8 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same',)
        self.conv9 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same',)
        self.conv10 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same',)
        self.pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv11 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same',)
        self.conv12 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same',)
        self.conv13 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same',)
        self.pool5 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.fla = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(4096, activation='relu')
        self.dense2 = tf.keras.layers.Dense(4096, activation='relu')
        self.dense3 = tf.keras.layers.Dense(1000, activation='relu')
        
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        
        x, mask1 = utilss.max_poo_with_argmax(x, 2)
        self.mask1 = mask1

        x = self.conv3(x)
        x = self.conv4(x)
        
        x, mask2 = utilss.max_poo_with_argmax(x, 2)
        self.mask2 = mask2

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        
        x, mask3 = utilss.max_poo_with_argmax(x, 2)
        self.mask3 = mask3

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        
        x, mask4 = utilss.max_poo_with_argmax(x, 2)
        self.mask4 = mask4

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        
        x, mask5 = utilss.max_poo_with_argmax(x, 2)
        self.mask5 = mask5

        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

text_data = utilss.DataLoader_text()
maxlen = 11
max_features =text_data.max_features
embedding_dims = 100
filter_num =128

kernel_regularizer = None

batch_size = 96

learning_rate = 0.001

model_img = imageCNN_fusion_unet_image_mask_gan.Image_CNN(maxlen, max_features, embedding_dims, filter_num, kernel_regularizer, text_data)

model_img.train_model_img(batch_size)
