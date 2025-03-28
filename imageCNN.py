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
        self.res = tf.keras.layers.Dense(3072, activation='sigmoid')
        
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

        x = tf.reshape(x, shape=(x.shape[0],
                                 x.shape[1] * x.shape[2] * x.shape[3]))
        x = self.res(x)
        return x

class Img_encoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1= tf.keras.layers.Conv2D(
            filters=64, 
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
            filters=128,
            kernel_size=[3, 3],
            padding='same',
            strides=(1, 1),
            activation=tf.nn.relu
        )  
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)  
        self.conv4 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=[3, 3],
            padding='same',
            strides=(1, 1),
            activation=tf.nn.relu
        )  
        self.pool4 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)  
        self.conv5 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=[3, 3],
            padding='same',
            strides=(1, 1),
            activation=tf.nn.relu
        )  
        self.pool5 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)  
        
        self.flatten = tf.keras.layers.Flatten()
        
        self.dense1 = tf.keras.layers.Dense(units=4096,activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=2048,activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=512,activation=tf.nn.relu)

    def call(self,inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.conv5(x)
        x = self.pool5(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)

        return x

class Img_decoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.unpool1 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.unconv1 = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(1, 1), activation='relu', padding='same', )
        self.unpool2 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.unconv2 = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(1, 1), activation='relu', padding='same', )
        self.unpool3 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.unconv3 = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(1, 1), activation='relu', padding='same', )
        self.unpool4 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.unconv4 = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(1, 1), activation='relu', padding='same', )
        self.unpool5 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.unconv5 = tf.keras.layers.Conv2DTranspose(3, (3, 3), strides=(1, 1), activation='relu', padding='same', )
        self.redense1 = tf.keras.layers.Dense(units=2048,activation=tf.nn.relu)
        self.redense2 = tf.keras.layers.Dense(units=4096,activation=tf.nn.relu)
        self.redense3 = tf.keras.layers.Dense(units=6272,activation=tf.nn.relu)
        
    def call(self, inputs):
        x = self.redense1(inputs)
        x = self.redense2(x)
        x = self.redense3(x)
        
        x = tf.reshape(x, shape=(x.shape[0], 7, 7, 128))
        x = self.unpool1(x)
        x = self.unconv1(x)
        x = self.unpool2(x)
        x = self.unconv2(x)
        x = self.unpool3(x)
        x = self.unconv3(x)
        x = self.unpool4(x)
        x = self.unconv4(x)
        x = self.unpool5(x)
        x = self.unconv5(x)
        return x

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

class Img_Model(nn.Module):
    def __init__(self):
        super(Img_Model, self).__init__()
        self.encoder_optimizer = tf.keras.optimizers.Adam(0.00001)
        self.decoder_optimizer = tf.keras.optimizers.Adam(0.00001)
        self.dis_optimizer = tf.keras.optimizers.Adam(0.00001)
        self.img_en = encoder()
        self.img_de = decoder()
        self.Dis = discriminator_model()

    def save_model_img(self):
        self.img_en.save('./model/en_vgg16', save_format='tf')
        self.img_de.save('./model/de_vgg16', save_format='tf')

    def train_model_img(self, batch_size):
        display_step = 1
        miss_rate = 0.7
        hint_rate = 0.9
        save_step = 100
        print("Start training...")
        train_ds = utilss.DataLoader_img()
        
        total_batch = int(train_ds.num_train_data // batch_size)

        ori_data = train_ds.train_data
        miss_data, data_m = utilss.miss_data_gen(miss_rate,ori_data) 
        
        data_m_fla = tf.reshape(data_m, shape=(data_m.shape[0], data_m.shape[1] * data_m.shape[2] * data_m.shape[3]))

        data_h_temp = utilss.binary_sampler(hint_rate, ori_data.shape[0], ori_data.shape[1], ori_data.shape[2],
                                            ori_data.shape[3]).astype(np.float32)
        
        data_h = np.array((data_h_temp * data_m)).astype(np.float32)
        train_data = miss_data /255.0 
        
        print("total_batch:", total_batch)

        ckpt_en = tf.train.Checkpoint(model=self.img_en, optimizer=self.encoder_optimizer)
        ckpt_en.restore(tf.train.latest_checkpoint('./model/vgg_en'))
        ckpt_manager_en = tf.train.CheckpointManager(ckpt_en, './model/vgg_en', max_to_keep=3)
        ckpt_de = tf.train.Checkpoint(model=self.img_de, optimizer=self.decoder_optimizer)
        ckpt_de.restore(tf.train.latest_checkpoint('./model/vgg_de'))
        ckpt_manager_de = tf.train.CheckpointManager(ckpt_de, './model/vgg_de', max_to_keep=3)
        
        training_epochs = 1501
        
        for epoch in range(training_epochs):
                if(epoch == training_epochs - 1):
                    decode_data = train_data
                
                id = 0
                totalLoss = 0
                for batch_index in range(total_batch):
                    
                    batch_ori_data, batch_data, batch_data_m, batch_data_h = utilss.get_batch_fourth(ori_data / 255.0,train_data,data_m,data_h,batch_size,batch_index,total_batch)
                    
                    batch_data = batch_data.astype(np.float32)
                    batch_data = tf.convert_to_tensor(batch_data)
                    
                    id = id + 1
                    with tf.GradientTape() as encoder_tape, tf.GradientTape() as decoder_tape,tf.GradientTape(persistent=True) as dis_tape:
                        encoder_res = self.img_en(batch_data)
                        decoder_res = self.img_de(encoder_res)

                        batch_data_m_fla = tf.reshape(batch_data_m, shape=(batch_data_m.shape[0],
                                                                           batch_data_m.shape[1] * batch_data_m.shape[
                                                                               2] * batch_data_m.shape[3]))
                        batch_data_h_fla = tf.reshape(batch_data_h, shape=(batch_data_h.shape[0],
                                                                           batch_data_h.shape[1] * batch_data_h.shape[
                                                                               2] * batch_data_h.shape[3]))
                        batch_data_img_fla = tf.reshape(batch_data, shape=(batch_data.shape[0],
                                                                               batch_data.shape[1] *
                                                                               batch_data.shape[2] *
                                                                               batch_data.shape[3]))

                        Hat_x = batch_data_img_fla * batch_data_m_fla + decoder_res * (1 - batch_data_m_fla)
                        D_prob = self.Dis(tf.concat(values=[Hat_x, batch_data_h_fla], axis=1))

                        mse = tf.keras.losses.MeanSquaredError()
                        
                        decoder_res_4 = tf.reshape(decoder_res, shape=(decoder_res.shape[0], 32, 32, 3))
                        self.mse_loss = tf.reduce_sum(
                            (batch_data_m * batch_data - batch_data_m * decoder_res_4) ** 2) / tf.reduce_sum(
                            batch_data_m)
                        
                        self.G_loss_temp = -tf.reduce_mean((1 - batch_data_m_fla) * tf.math.log(D_prob + 1e-8))
                        self.loss = self.G_loss_temp + self.mse_loss * 100
                        self.D_loss_temp = -tf.reduce_mean(
                            batch_data_m_fla * tf.math.log(D_prob + 1e-8) + (1 - batch_data_m_fla) * tf.math.log(
                                1. - D_prob + 1e-8))
                        imputed_loss = tf.reduce_sum(((1-batch_data_m) * batch_ori_data - (1-batch_data_m) * decoder_res_4) ** 2) / tf.reduce_sum((1-batch_data_m))
                        totalLoss+=imputed_loss

                        self.gradients_of_en = encoder_tape.gradient(self.loss,
                                                                     self.img_en.trainable_variables)

                        self.encoder_optimizer.apply_gradients(
                            zip(self.gradients_of_en, self.img_en.trainable_variables))

                        self.gradients_of_de = decoder_tape.gradient(self.loss,
                                                                     self.img_de.trainable_variables)
                        self.decoder_optimizer.apply_gradients(
                            zip(self.gradients_of_de, self.img_de.trainable_variables))

                        for i in range(5):
                            self.gradients_of_dis = dis_tape.gradient(self.D_loss_temp,
                                                                      self.Dis.trainable_variables)
                            self.dis_optimizer.apply_gradients(
                                zip(self.gradients_of_dis,self.Dis.trainable_variables))

                print("Epoch:", '%04d' % epoch,"imputed_loss:",(totalLoss / total_batch)**.5)
                print("g_loss=", "{:.9f}".format(self.loss),"d_loss=","{:.9f}".format(self.D_loss_temp))
                
def get_res_img(batch_size):
    enco = keras.models.load_model('./model/en_img_norm', compile=False)
    deco = keras.models.load_model('./model/de_img_norm', compile=False)
    data_loader = utilss.DataLoader_img()
    total_batches1 = int(data_loader.num_train_data//batch_size)
    total_batches2 = int(data_loader.num_train_data4//batch_size)
    res = 0.0
    
    for batch_index in range(total_batches1):
        y = utilss.get_batch(data_loader.train_data, batch_size, batch_index, total_batches1)
        y = y.astype(float)
        y = tf.convert_to_tensor(y)
        y_norm, norm_parameters = utilss.normalization(y)
        
        encoder_res = enco(y_norm)
        encoder_res_fla = tf.keras.layers.Flatten()(encoder_res)
        encoder_res = tf.reshape(encoder_res_fla, [encoder_res_fla.get_shape().as_list()[0], 7, 7, 256])
        decoder_res = deco(encoder_res)
        mse = tf.keras.losses.MeanSquaredError()
        mid = mse(y_norm, decoder_res)
        res = res + mid
        decoder_res = utilss.renormalization(decoder_res, norm_parameters)
        
        print("ori:", y)
        print("res_de:", decoder_res)
        
    for batch_index in range(total_batches2):
        y = utilss.get_batch(data_loader.train_data4, batch_size, batch_index, total_batches2)
        y = y.astype(float)
        y = tf.convert_to_tensor(y)
        y_norm, norm_parameters = utilss.normalization(y)
        
        y2 = utilss.get_batch(data_loader.train_data4_miss, batch_size, batch_index, total_batches2)
        y2 = y2.astype(float)
        y2 = tf.convert_to_tensor(y2)
        y2_norm, norm_parameters2 = utilss.normalization(y2)
        encoder_res = enco(y2_norm)
        encoder_res_fla = tf.keras.layers.Flatten()(encoder_res)
        encoder_res = tf.reshape(encoder_res_fla, [encoder_res_fla.get_shape().as_list()[0], 7, 7, 256])
        decoder_res = deco(encoder_res)
        mse = tf.keras.losses.MeanSquaredError()
        mid = mse(y_norm, decoder_res)
        res = res + mid
        
        decoder_res = utilss.renormalization(decoder_res, norm_parameters2)

    return res, total_batches1+total_batches2

class TextCNN_encoder(tf.keras.Model):
    def __init__(self, maxlen, max_features, embedding_dims,
                 class_num=5,
                 last_activation='softmax'):
        super().__init__()
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims

        self.embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen)
        self.conv1D_3 = Conv1D(128, 3, activation='elu')
        self.conv1D_4 = Conv1D(128, 4, activation='elu')
        self.conv1D_5 = Conv1D(128, 5, activation='elu')
        
        self.concat = Concatenate()
        self.fla = Flatten()
    def call(self, inputs):
        convs = []
        x_ori = self.embedding(inputs)
        print("x_ori:", x_ori.shape)
        x_norm, norm_parameters = utilss.normalization(x_ori)
        
        x = self.conv1D_3(x_norm)

        x = MaxPool1D(pool_size=x_norm.shape[1]-2)(x)
        convs.append(x)

        x = self.conv1D_4(x_norm)
        x = MaxPool1D(pool_size=x_norm.shape[1]-3)(x)
        convs.append(x)

        x = self.conv1D_5(x_norm)
        x = MaxPool1D(pool_size=x_norm.shape[1]-4)(x)
        convs.append(x)

        x = self.concat(convs)
        x = self.fla(x)
        return x, x_ori, norm_parameters, x_norm
class TextCNN_decoder(tf.keras.Model):
    def __init__(self, max_features, x_norm):
        super().__init__()
        self.unconv1 = Conv1DTranspose(10, 3, activation='elu')
        self.unconv2 = Conv1DTranspose(10, 4, activation='elu')
        self.unconv3 = Conv1DTranspose(10, 5, activation='elu')
        self.unpool1 = UpSampling1D(size=max_features-2)
        self.unpool2 = UpSampling1D(size=max_features-3)
        self.unpool3 = UpSampling1D(size=max_features-4)
        self.concat = Concatenate()
        self.dense = Dense(10, activation='elu')
        self.max_features = max_features
        self.x_norm = x_norm

    def call(self, inputs):
        convs = []
        res = tf.split(inputs, axis=2, num_or_size_splits=[128, 128, 128])

        x1 = UpSampling1D(size=self.x_norm.shape[1]-2)(res[0])
        x1 = self.unconv1(x1)
        convs.append(x1)

        x2 = UpSampling1D(size=self.x_norm.shape[1]-3)(res[1])
        x2 = self.unconv2(x2)
        convs.append(x2)

        x3 = UpSampling1D(size=self.x_norm.shape[1]-4)(res[2])
        x3 = self.unconv3(x3)
        convs.append(x3)

        res = self.concat(convs)
        res = self.dense(res)
        return res

class Text_Model(nn.Module):
    def __init__(self):
        super(Text_Model, self).__init__()
        self.train_ds = utilss.DataLoader_text()
        self.encoder_optimizer = tf.keras.optimizers.Adam(0.0001)
        self.decoder_optimizer = tf.keras.optimizers.Adam(0.0001)
        self.text_en = TextCNN_encoder(12, self.train_ds.v_count, 10)
        
    def train_model_text(self, training_epochs, batch_size):
        display_step = 1
        save_step = 10
        print("Start training...")
        train_ds = utilss.DataLoader_text()
        total_batch = int(self.train_ds.training_data.shape[0] // batch_size)
        for epoch in range(training_epochs):
            id = 0
            for batch_index in range(total_batch):
                batch_data = utilss.get_batch(self.train_ds.training_data, batch_size, batch_index, total_batch) 
                
                print(id)
                id = id + 1

                with tf.GradientTape() as encoder_tape, tf.GradientTape(persistent=True) as decoder_tape:
                    batch_data = batch_data.astype(float)
                    batch_data = tf.convert_to_tensor(batch_data)
                    
                    encoder_res, encoder_res_em, norm_parameters, x_norm = self.text_en(batch_data)
                    self.text_de = TextCNN_decoder(self.train_ds.v_count, x_norm)
                    
                    encoder_res = tf.reshape(encoder_res, shape=(encoder_res.shape[0], 1, encoder_res.shape[1]))
                    mse = tf.keras.losses.MeanSquaredError()
                    
                    res_de = utilss.renormalization(self.text_de(encoder_res), norm_parameters)
                    
                    self.loss = tf.sqrt(mse(encoder_res_em, res_de))
                    
                    self.gradients_of_en = encoder_tape.gradient(self.loss,
                                                                 self.text_en.trainable_variables)
                    self.encoder_optimizer.apply_gradients(
                        zip(self.gradients_of_en, self.text_en.trainable_variables))

                    self.gradients_of_de = decoder_tape.gradient(self.loss,
                                                                 self.text_de.trainable_variables)
                    self.decoder_optimizer.apply_gradients(
                        zip(self.gradients_of_de, self.text_de.trainable_variables))
                    
                print("res_em:", encoder_res_em)
                print("res_de:", res_de)
                
            print("Epoch:", '%04d' % (epoch + 1), "g_loss=", "{:.9f}".format(self.loss))
            
            self.save_model_text()
    def save_model_text(self):
        self.text_en.save('./model/en_text', save_format='tf')
        self.text_de.save('./model/de_text', save_format='tf')

def get_text_res(batch_size):

    enco = keras.models.load_model('./model/en_text', compile=False)
    deco = keras.models.load_model('./model/de_text', compile=False)
    data_loader = utilss.DataLoader_text()
    res = 0.0
    total_batches = int(data_loader.training_data.shape[0] // batch_size)
    for batch_index in range(total_batches):
        batch_data = utilss.get_batch(data_loader.training_data, batch_size, batch_index, total_batches)
        batch_data = batch_data.astype(float)
        batch_data = tf.convert_to_tensor(batch_data)
        encoder_res, encoder_res_emm, norm_parameters = enco(batch_data)
        encoder_res = tf.reshape(encoder_res, shape=(encoder_res.shape[0], 1, encoder_res.shape[1]))
        decoder_res = deco(encoder_res)
        decoder_res = utilss.renormalization(decoder_res, norm_parameters)
        mse = tf.keras.losses.MeanSquaredError()
        loss = mse(encoder_res_emm, decoder_res)
        res = res + loss
        print("x_em", encoder_res_emm)
        print("decoder_res:", decoder_res)
    return res, total_batches