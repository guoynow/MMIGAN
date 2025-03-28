import keras
import numpy as np
import tensorflow as tf
from keras.layers import Embedding, Conv1D, Conv2D, Conv2DTranspose, Flatten, GlobalAveragePooling1D, Dense, Concatenate, GlobalMaxPooling2D, UpSampling1D
from keras import Model
import keras.backend as K

import utilss

class Fusion_Model(Model):
    def __init__(self):
        super().__init__()
        
        self.dense_384_en = Dense(384, activation='elu')
        self.dense_128_en_img = Dense(128, activation='elu')
        self.dense_50_en_img = Dense(50, activation='elu')
        self.dense_128_en_text = Dense(128, activation='elu')
        self.dense_50_en_text = Dense(50, activation='elu')

        self.dense_128_de_img = Dense(128, activation='elu')
        self.dense_384_de = Dense(384, activation='elu')
        self.dense_128_de_text = Dense(128, activation='elu')
        self.dense_192 = Dense(192, activation='elu')
        self.dense_512 = Dense(512, activation='elu')
    def call(self, input1, input2):

        res_en_img = self.dense_384_en(input1)
        res_en_img = self.dense_128_en_img(res_en_img)
        res_en_img = self.dense_50_en_img(res_en_img)
        
        res_en_text = self.dense_128_en_text(input2)
        res_en_text = self.dense_50_en_text(res_en_text)
        
        conca = tf.concat((res_en_img, res_en_text), axis=1)
        
        res_de_text = self.dense_128_de_img(conca)
        res_de_text = self.dense_192(res_de_text)

        return res_de_text