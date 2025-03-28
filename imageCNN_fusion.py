import numpy as np

import imageCNN
import fusionModel
from textCNN_ori import text_embedding
from textCNN_ori import text_encoder
import time
from torch import nn
from torch.nn import MaxPool2d

import utilss
import tensorflow as tf
from tensorflow.python.keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout, MaxPool1D,\
    Flatten, Conv1DTranspose, UpSampling1D
from tensorflow.keras import Model

class discriminator_model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        self.D_h1 = Dense(3072, activation='relu')
        self.D_h2 = Dense(3072, activation='relu')
        self.D_prob = Dense(3072, activation='sigmoid')

    def call(self, inputs):
        x = self.D_h1(inputs)
        x = self.D_h2(x)
        x = self.D_prob(x)
        return x

class encoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.vgg16 = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet',
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
        self.res = tf.keras.layers.Dense(3072,activation='sigmoid')

    def call(self, inputs):

        x = tf.reshape(inputs, shape=(tf.shape(inputs)[0], 1, 1, 512))
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

        x = tf.reshape(x,shape=(tf.shape(x)[0],
                                          tf.shape(x)[1]*tf.shape(x)[2]*tf.shape(x)[3]))
        x = self.res(x)
        return x

class Image_CNN(nn.Module):
    def __init__(self,maxlen, max_features, embedding_dims, filter_num, kernel_regularizer, train_ds_text):
        super(Image_CNN, self).__init__()
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.filter_num = filter_num
        self.kernel_regularizer = kernel_regularizer
        self.train_ds_text = train_ds_text

        self.encoder_optimizer = tf.keras.optimizers.Adam(0.00001)
        self.decoder_optimizer = tf.keras.optimizers.Adam(0.00001)
        self.text_encoder_optimizer = tf.keras.optimizers.Adam(0.00001)
        self.emb_optimizer = tf.keras.optimizers.Adam(0.00001)
        self.fusion_optimizer = tf.keras.optimizers.Adam(0.00001)
        self.dis_optimizer = tf.keras.optimizers.Adam(0.00001)

        self.img_en = encoder()
        self.img_de = decoder()

        self.embedding = text_embedding(self.maxlen, self.max_features, self.embedding_dims)
        self.text_en = text_encoder(self.filter_num, self.embedding_dims, self.kernel_regularizer)
        self.fusion = fusionModel.Fusion_Model()
        self.Dis = discriminator_model()

    def save_model_img(self):
        self.img_en.save('./model/en_vgg16', save_format='tf')
        self.img_de.save('./model/de_vgg16', save_format='tf')

    def train_model_img(self,  batch_size):
        miss_rate = 0.8
        hint_rate = 0.9
        display_step = 1
        save_step = 100
        print("Start training...")
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        train_ds_img = utilss.DataLoader_img()
        
        ori_data = train_ds_img.train_data
        miss_data, data_m = utilss.miss_data_gen(miss_rate,ori_data)
        
        data_m_fla = tf.reshape(data_m,shape=(data_m.shape[0],data_m.shape[1]*data_m.shape[2]*data_m.shape[3]))

        data_h_temp = utilss.binary_sampler(hint_rate, ori_data.shape[0], ori_data.shape[1], ori_data.shape[2],
                                            ori_data.shape[3]).astype(np.float32)
        
        data_h = np.array((data_h_temp * data_m)).astype(np.float32)

        train_data = miss_data / 255.0
        
        total_batch = int(train_ds_img.num_train_data / batch_size)
        print("total_batch:", total_batch)

        allTimeInteval = 0
        training_epochs = 1501
        
        startTime = time.time()
        for epoch in range(training_epochs):
                
                id = 0
                totalLoss = 0
                for batch_index in range(total_batch):
                    
                    batch_ori_data, batch_data_img, batch_data_text,batch_data_m, batch_data_h = utilss.get_batch_fifth(
                                                                                ori_data / 255.0,
                                                                                train_data,
                                                                                self.train_ds_text.training_data,
                                                                                data_m,
                                                                                data_h,
                                                                                batch_size,batch_index,total_batch)
                    
                    batch_data_text = tf.reshape(batch_data_text, shape=(batch_data_text.shape[0], batch_data_text.shape[1], 1))
                    batch_data_img = batch_data_img.astype(np.float32)
                    batch_data_img = tf.convert_to_tensor(batch_data_img)
                    
                    id = id + 1
                    with tf.GradientTape() as img_en_tape, tf.GradientTape() as img_de_tape,\
                            tf.GradientTape() as emb_tape, tf.GradientTape() as text_en_tape,\
                            tf.GradientTape(persistent=True) as dis_tape,tf.GradientTape() as fusion_tape:

                        encoder_img_res = self.img_en(batch_data_img)
                        emb = self.embedding(batch_data_text)
                        encoder_text_res = self.text_en(emb)
                        
                        fusion_data = self.fusion(encoder_img_res, encoder_text_res)
                        
                        decoder_res = self.img_de(fusion_data)

                        batch_data_m_fla = tf.reshape(batch_data_m,shape=(batch_data_m.shape[0],
                                                                        batch_data_m.shape[1]*batch_data_m.shape[2]*batch_data_m.shape[3]))
                        batch_data_h_fla = tf.reshape(batch_data_h, shape=(batch_data_h.shape[0],
                                                                           batch_data_h.shape[1] * batch_data_h.shape[
                                                                               2] * batch_data_h.shape[3]))
                        batch_data_img_fla = tf.reshape(batch_data_img,shape=(batch_data_img.shape[0],
                                                                        batch_data_img.shape[1]*batch_data_img.shape[2]*batch_data_img.shape[3]))

                        Hat_x = batch_data_img_fla * batch_data_m_fla + decoder_res * (1 - batch_data_m_fla)
                        D_prob = self.Dis(tf.concat(values=[Hat_x, batch_data_h_fla], axis=1))

                        mse = tf.keras.losses.MeanSquaredError()
                        
                        decoder_res_4 = tf.reshape(decoder_res,shape=(decoder_res.shape[0],32,32,3))
                        self.mse_loss = tf.reduce_sum(
                            (batch_data_m * batch_data_img - batch_data_m * decoder_res_4) ** 2) / tf.reduce_sum(
                            batch_data_m)
                        
                        self.G_loss_temp = -tf.reduce_mean((1 - batch_data_m_fla) * tf.math.log(D_prob + 1e-8))
                        self.loss = self.G_loss_temp + self.mse_loss *100
                        self.D_loss_temp = -tf.reduce_mean(
                            batch_data_m_fla * tf.math.log(D_prob + 1e-8) + (1 - batch_data_m_fla) * tf.math.log(1. - D_prob + 1e-8))
                        imputed_loss = tf.reduce_sum(((1-batch_data_m) * batch_ori_data - (1-batch_data_m) * decoder_res_4) ** 2) / tf.reduce_sum((1-batch_data_m))
                        totalLoss+=imputed_loss
                        
                        self.gradients_of_en = img_en_tape.gradient(self.loss,
                                                                     self.img_en.trainable_variables)
                        self.encoder_optimizer.apply_gradients(
                            zip(self.gradients_of_en, self.img_en.trainable_variables))

                        self.gradients_of_de = img_de_tape.gradient(self.loss,
                                                                     self.img_de.trainable_variables)
                        self.decoder_optimizer.apply_gradients(
                            zip(self.gradients_of_de, self.img_de.trainable_variables))

                        self.gradients_of_fusion = fusion_tape.gradient(self.loss,
                                                                    self.fusion.trainable_variables)
                        self.fusion_optimizer.apply_gradients(
                            zip(self.gradients_of_fusion, self.fusion.trainable_variables))
                        
                        self.gradients_of_emb = emb_tape.gradient(self.loss,
                                                                    self.embedding.trainable_variables)
                        self.emb_optimizer.apply_gradients(
                            zip(self.gradients_of_emb, self.embedding.trainable_variables))
                        
                        self.gradients_of_text_en = text_en_tape.gradient(self.loss,
                                                                  self.text_en.trainable_variables)
                        self.text_encoder_optimizer.apply_gradients(
                            zip(self.gradients_of_text_en, self.text_en.trainable_variables))
                        
                        for i in range(5):
                            self.gradients_of_dis = dis_tape.gradient(self.D_loss_temp,
                                                                      self.Dis.trainable_variables)
                            self.dis_optimizer.apply_gradients(
                                zip(self.gradients_of_dis,self.Dis.trainable_variables))
                        
                print("Epoch:", '%04d' % epoch, "imputed_loss:", (totalLoss / total_batch)**.5)
                print("g_loss=", "{:.9f}".format(self.loss), "d_loss=", "{:.9f}".format(self.D_loss_temp))

        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()))
