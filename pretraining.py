import numpy
import numpy as np

import tensorflow as tf

from tensorflow import keras
import time
from torch import nn
from torch.nn import MaxPool2d

import utilss
from tensorflow.python.keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout, MaxPool1D,\
    Flatten, Conv1DTranspose, UpSampling1D

class encoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.vgg16 = tf.keras.applications.vgg16.VGG16(include_top=False, weights=None,
                                                          input_shape=(32, 32, 3),
                                                          pooling='None')
        
        self.fla = Flatten()
        self.dense1 = Dense(6144, activation='relu')
        self.dense2 = Dense(4096, activation='relu')
        self.dense3 = Dense(2048, activation='relu')
        self.dense4 = Dense(512,activation='relu')

    def call(self,inputs):
        x = self.vgg16(inputs)
        
        x = self.fla(x)
        
        return x
class decoder(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.unpool1 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.unconv1 = tf.keras.layers.Conv2DTranspose(512, (3, 3), strides=(1, 1), activation='relu', padding='same', )
        self.unconv2 = tf.keras.layers.Conv2DTranspose(512, (3, 3), strides=(1, 1), activation='relu', padding='same', )
        self.unconv3 = tf.keras.layers.Conv2DTranspose(512, (3, 3), strides=(1, 1), activation='relu', padding='same', )

        self.unpool2 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.unconv4 = tf.keras.layers.Conv2DTranspose(512, (3, 3), strides=(1, 1), activation='relu', padding='same', )
        self.unconv5 = tf.keras.layers.Conv2DTranspose(512, (3, 3), strides=(1, 1), activation='relu', padding='same', )
        self.unconv6 = tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=(1, 1), activation='relu', padding='same', )

        self.unpool3 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.unconv7 = tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=(1, 1), activation='relu', padding='same', )
        self.unconv8 = tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=(1, 1), activation='relu', padding='same', )
        self.unconv9 = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(1, 1), activation='relu', padding='same', )

        self.unpool4 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.unconv10 = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(1, 1), activation='relu', padding='same', )
        self.unconv11 = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(1, 1), activation='relu', padding='same', )

        self.unpool5 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.unconv12 = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(1, 1), activation='relu', padding='same', )
        self.unconv13 = tf.keras.layers.Conv2DTranspose(3, (3, 3), strides=(1, 1), activation='relu', padding='same', )

        self.redense1 = tf.keras.layers.Dense(units=2048, activation=tf.nn.relu)
        self.redense2 = tf.keras.layers.Dense(units=4096, activation=tf.nn.relu)
        self.redense3 = tf.keras.layers.Dense(units=6144,activation=tf.nn.relu)
        self.redense4 = tf.keras.layers.Dense(units=8192,activation=tf.nn.relu)

    def call(self, inputs):

        x = tf.reshape(inputs, shape=(inputs.shape[0], 1, 1, 512))
        
        x = self.unpool1(x)
        x = self.unconv1(x)
        x = self.unconv2(x)
        x = self.unconv3(x)
        
        x = self.unpool2(x)
        x = self.unconv4(x)
        x = self.unconv5(x)
        x = self.unconv6(x)
        
        x = self.unpool3(x)
        x = self.unconv7(x)
        x = self.unconv8(x)
        x = self.unconv9(x)
        
        x = self.unpool4(x)
        x = self.unconv10(x)
        x = self.unconv11(x)
        
        x = self.unpool5(x)
        x = self.unconv12(x)
        x = self.unconv13(x)
        return x

class Img_Model(nn.Module):
    def __init__(self):
        super(Img_Model, self).__init__()
        self.encoder_optimizer = tf.keras.optimizers.Adam(0.0001)
        self.decoder_optimizer = tf.keras.optimizers.Adam(0.0001)
        self.img_en = encoder()
        self.img_de = decoder()

    def save_model_img(self):
        self.img_en.save('./model/en_vgg16', save_format='tf')
        self.img_de.save('./model/de_vgg16', save_format='tf')

    def train_model_img(self, batch_size):
        display_step = 1
        miss_rate = 0.3
        save_step = 100
        print("Start training...")
        train_ds = utilss.DataLoader_img()
        
        total_batch = int(train_ds.num_train_data // batch_size)

        train_data =train_ds.train_data
        print("total_batch:", total_batch)

        ckpt_en = tf.train.Checkpoint(model=self.img_en, optimizer=self.encoder_optimizer)
        ckpt_en.restore(tf.train.latest_checkpoint('./model/vgg_en'))
        ckpt_manager_en = tf.train.CheckpointManager(ckpt_en, './model/vgg_en', max_to_keep=3)
        ckpt_de = tf.train.Checkpoint(model=self.img_de, optimizer=self.decoder_optimizer)
        ckpt_de.restore(tf.train.latest_checkpoint('./model/vgg_de'))
        ckpt_manager_de = tf.train.CheckpointManager(ckpt_de, './model/vgg_de', max_to_keep=3)
        
        training_epochs = 5001
        for epoch in range(training_epochs):
                
                print("epoch", epoch)
                id = 0
                for batch_index in range(total_batch):
                    
                    batch_data = utilss.get_batch(train_data,batch_size,batch_index,total_batch)
                    
                    batch_data = batch_data.astype(np.float32)
                    batch_data = tf.convert_to_tensor(batch_data)
                    batch_data_norm = batch_data / 255.0
                    id = id + 1
                    with tf.GradientTape() as encoder_tape, tf.GradientTape() as decoder_tape:
                        encoder_res = self.img_en(batch_data_norm)
                        decoder_res = self.img_de(encoder_res)

                        mse = tf.keras.losses.MeanSquaredError()
                        self.loss = tf.sqrt(mse(batch_data_norm, decoder_res))
                        self.gradients_of_en = encoder_tape.gradient(self.loss,
                                                                     self.img_en.trainable_variables)

                        self.encoder_optimizer.apply_gradients(
                            zip(self.gradients_of_en, self.img_en.trainable_variables))

                        self.gradients_of_de = decoder_tape.gradient(self.loss,
                                                                     self.img_de.trainable_variables)
                        self.decoder_optimizer.apply_gradients(
                            zip(self.gradients_of_de, self.img_de.trainable_variables))
                        
                print("Epoch:", '%04d' % (epoch + 1), "g_loss=", "{:.9f}".format(self.loss))
                if epoch % save_step == 0:
                    
                    path_en = ckpt_manager_en.save()
                    path_de = ckpt_manager_de.save()
                    print("model_en saved to %s" % path_en)
                    print("model_de saved to %s" % path_de)
                