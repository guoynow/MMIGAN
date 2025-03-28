import numpy as np
import torch
from keras.layers import concatenate, BatchNormalization
import fusionModel
from textCNN import text_embedding, text_decoder
from textCNN import text_encoder
import time

import utilss
import tensorflow as tf
from tensorflow.python.keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout, MaxPool1D,\
    Flatten, Conv1DTranspose, UpSampling1D

import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from scipy.linalg import sqrtm
import tensorflow as tf
from tensorflow.keras import backend as K
import gc

class discriminator_model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        self.D_h1 = Dense(3072, activation='relu')
        self.D_h2 = Dense(3072, activation='relu')
        self.D_prob = Dense(3072, activation='relu')
        
    def call(self, inputs):
        x = self.D_h1(inputs)
        x = self.D_h2(x)
        x = self.D_prob(x)
        x = tf.where(x > 1, 1.0, x)
        
        return x

class LayerScaler(tf.keras.Model):
    def __init__(self, dim, init_value=1e-6):
        super(LayerScaler, self).__init__()
        self.scaler = tf.Variable(initial_value=tf.ones(dim) * init_value, trainable=True)

    def call(self, x):
        gamma = tf.reshape(self.scaler, (1, 1, 1, -1))
        return x * gamma

class encoder(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.conv1 = tf.keras.layers.Conv2D(64, (3, 3), input_shape=(32, 32, 3), padding='same',
                                   activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.conv4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', )
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', )
        self.conv6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', )
        self.conv7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', )
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv8 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', )
        self.conv9 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', )
        self.conv10 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', )
        self.pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv11 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', )
        self.conv12 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', )
        self.conv13 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', )
        self.pool5 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.fla = tf.keras.layers.Flatten()
        
    def call(self, train_data):

        x = self.conv1(train_data)
        x = BatchNormalization()(x)
        x = self.conv2(x)
        x = BatchNormalization()(x)
        block1_out = x
        x = self.pool1(x)

        x = self.conv3(x)
        x = BatchNormalization()(x)
        x = self.conv4(x)
        x = BatchNormalization()(x)
        block2_out = x
        x = self.pool2(x)

        x = self.conv5(x)
        x = BatchNormalization()(x)
        x = self.conv6(x)
        x = BatchNormalization()(x)
        x = self.conv7(x)
        x = BatchNormalization()(x)
        block3_out = x
        x = self.pool3(x)

        x = self.conv8(x)
        x = BatchNormalization()(x)
        x = self.conv9(x)
        x = BatchNormalization()(x)
        x = self.conv10(x)
        x = BatchNormalization()(x)
        block4_out = x
        x = self.pool4(x)

        x = self.conv11(x)
        x = BatchNormalization()(x)
        x = self.conv12(x)
        x = BatchNormalization()(x)
        x = self.conv13(x)
        x = BatchNormalization()(x)
        block5_out = x
        x = self.pool5(x)

        x = self.fla(x)
        
        return x,block1_out,block2_out,block3_out,block4_out,block5_out
class decoder(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.unpool1 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.unconv1 = tf.keras.layers.Conv2DTranspose(512, (3, 3), activation='relu', padding='same', )
        self.unconv2 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', )
        self.unconv3 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', )

        self.unpool2 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.unconv4 = tf.keras.layers.Conv2DTranspose(512, (3, 3), activation='relu', padding='same', )
        self.unconv5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', )
        self.unconv6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', )

        self.unpool3 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.unconv7 = tf.keras.layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same', )
        self.unconv8 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', )
        self.unconv9 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', )

        self.unpool4 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.unconv10 = tf.keras.layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same', )
        self.unconv11 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', )

        self.unpool5 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.unconv12 = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same', )
        self.unconv13 = tf.keras.layers.Conv2D(3, (3, 3), activation='relu', padding='same', )

        self.redense1 = tf.keras.layers.Dense(units=2048, activation=tf.nn.relu)
        self.redense2 = tf.keras.layers.Dense(units=4096, activation=tf.nn.relu)
        self.redense3 = tf.keras.layers.Dense(units=6144,activation=tf.nn.relu)
        self.redense4 = tf.keras.layers.Dense(units=8192,activation=tf.nn.relu)
        
    def call(self, inputs,block1_out,block2_out,block3_out,block4_out,block5_out):

        x = tf.reshape(inputs, shape=(tf.shape(inputs)[0], 1, 1, 512))
        x = self.unpool1(x)
        x = self.unconv1(x)
        x = BatchNormalization()(x)
        x = concatenate([x, block5_out])
        x = self.unconv2(x)
        x = BatchNormalization()(x)
        x = self.unconv3(x)
        x = BatchNormalization()(x)
        
        x = self.unpool2(x)
        x = self.unconv4(x)
        x = BatchNormalization()(x)
        x = concatenate([x, block4_out])
        x = self.unconv5(x)
        x = BatchNormalization()(x)
        x = self.unconv6(x)
        x = BatchNormalization()(x)
        
        x = self.unpool3(x)
        x = self.unconv7(x)
        x = BatchNormalization()(x)
        x = concatenate([x, block3_out])
        x = self.unconv8(x)
        x = BatchNormalization()(x)
        x = self.unconv9(x)
        x = BatchNormalization()(x)
        
        x = self.unpool4(x)
        x = self.unconv10(x)
        x = BatchNormalization()(x)
        x = concatenate([x, block2_out])
        x = self.unconv11(x)
        x = BatchNormalization()(x)
        
        x = self.unpool5(x)
        x = self.unconv12(x)
        x = BatchNormalization()(x)
        x = concatenate([x, block1_out])
        x = self.unconv13(x)
        x = BatchNormalization()(x)

        return x

class D_encoder(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.conv1 = tf.keras.layers.Conv2D(64, (3, 3), input_shape=(32, 32, 3), padding='same',
                                   activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.conv4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', )
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', )
        self.conv6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', )
        self.conv7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', )
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv8 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', )
        self.conv9 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', )
        self.conv10 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', )
        self.pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv11 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', )
        self.conv12 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', )
        self.conv13 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', )
        self.pool5 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.fla = tf.keras.layers.Flatten()
        
    def call(self, train_data):

        x = self.conv1(train_data)
        x = BatchNormalization()(x)
        x = self.conv2(x)
        x = BatchNormalization()(x)
        block1_out = x
        x = self.pool1(x)
        
        x = self.fla(x)
        
        return x,block1_out
class D_decoder(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.unpool1 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.unconv1 = tf.keras.layers.Conv2DTranspose(512, (3, 3), activation='relu', padding='same', )
        self.unconv2 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', )
        self.unconv3 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', )

        self.unpool2 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.unconv4 = tf.keras.layers.Conv2DTranspose(512, (3, 3), activation='relu', padding='same', )
        self.unconv5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', )
        self.unconv6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', )

        self.unpool3 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.unconv7 = tf.keras.layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same', )
        self.unconv8 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', )
        self.unconv9 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', )

        self.unpool4 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.unconv10 = tf.keras.layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same', )
        self.unconv11 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', )

        self.unpool5 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.unconv12 = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same', )
        self.unconv13 = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same', )

        self.redense1 = tf.keras.layers.Dense(units=2048, activation=tf.nn.relu)
        self.redense2 = tf.keras.layers.Dense(units=4096, activation=tf.nn.relu)
        self.redense3 = tf.keras.layers.Dense(units=6144,activation=tf.nn.relu)
        self.redense4 = tf.keras.layers.Dense(units=8192,activation=tf.nn.relu)
        
        self.dense = tf.keras.layers.Dense(512,activation='relu')

    def call(self, inputs,block1_out):

        x = tf.reshape(inputs, shape=(tf.shape(inputs)[0], 16, 16, 64))
        
        x = self.unpool5(x)
        x = self.unconv12(x)
        x = BatchNormalization()(x)
        x = concatenate([x, block1_out])
        x = self.unconv13(x)
        x = BatchNormalization()(x)

        return x

class Image_CNN(torch.nn.Module):
    def __init__(self,maxlen, max_features, embedding_dims, filter_num, kernel_regularizer, train_ds_text):
        super(Image_CNN, self).__init__()
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.filter_num = filter_num
        self.kernel_regularizer = kernel_regularizer
        self.train_ds_text = train_ds_text
        self.learning_rate = 0.001
        self.encoder_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.decoder_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.text_encoder_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.text_decoder_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.bert_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.emb_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.fusion_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.dis_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.dis_en_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.dis_de_optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        self.img_en = encoder()
        self.img_de = decoder()
        self.Dis_en = D_encoder()
        self.Dis_de = D_decoder()

        self.embedding = text_embedding(self.maxlen, self.max_features, self.embedding_dims)
        self.text_en = text_encoder(self.filter_num, self.embedding_dims, self.kernel_regularizer)
        self.text_de = text_decoder(self.filter_num, self.embedding_dims, self.kernel_regularizer,self.maxlen)
        
        self.Dis = discriminator_model()

    def save_model_img(self):
        self.img_en.save('./model/img_en_vgg16', save_format='tf')
        self.img_de.save('./model/img_de_vgg16', save_format='tf')
        self.Dis_en.save('./model/dis_en', save_format='tf')
        self.Dis_de.save('./model/dis_de', save_format='tf')
        self.text_en.save('./model/text_en', save_format='tf')
        self.text_de.save('./model/text_de', save_format='tf')
        self.Dis.save('./model/Dis', save_format='tf')

    def train_model_img(self,  batch_size):
        miss_rate = 0.5
        hint_rate = 0.9
        display_step = 1
        save_step = 100
        print("Start training...")
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        train_ds_img = utilss.DataLoader_img()
        
        ori_data = train_ds_img.train_data
        miss_data, data_m = utilss.miss_data_gen(miss_rate,ori_data)

        miss_rate_guancezhi = 0.3
        non_zero_indices = np.where(miss_data != 0)
        num_to_zero = int(len(non_zero_indices[0]) * miss_rate_guancezhi)
        zero_indices = np.random.choice(len(non_zero_indices[0]), num_to_zero, replace=False)
        miss_data[non_zero_indices[0][zero_indices], non_zero_indices[1][zero_indices], non_zero_indices[2][zero_indices],non_zero_indices[3][zero_indices]] = 0

        data_h_temp = utilss.binary_sampler(hint_rate, ori_data.shape[0], ori_data.shape[1], ori_data.shape[2],
                                            ori_data.shape[3]).astype(np.float32)
        
        data_h = np.array((data_h_temp * data_m)).astype(np.float32)

        train_data = miss_data / 255.0
        
        total_batch = int(train_ds_img.num_train_data / batch_size)
        print("total_batch:", total_batch)

        allTimeInteval = 0
        training_epochs = 1000
        
        startTime = time.time()
        for epoch in range(training_epochs):
                
                id = 0
                totalLoss = 0
                trainingLoss = 0
                decoder_res_all = []
                ori_data_all = []
                for batch_index in range(total_batch):
                    
                    batch_ori_data, batch_data_img, batch_data_text,batch_data_m, batch_data_h = utilss.get_batch_fifth(
                                                                                ori_data / 255.0,
                                                                                train_data,
                                                                                self.train_ds_text.training_data,
                                                                                
                                                                                data_m,
                                                                                data_h,
                                                                                batch_size,batch_index,total_batch)
                    
                    batch_data_img = batch_data_img.astype(np.float32)
                    batch_data_img = tf.convert_to_tensor(batch_data_img)

                    batch_data_text = tf.reshape(batch_data_text,
                                                 shape=(batch_data_text.shape[0], batch_data_text.shape[1], 1))

                    batch_ori_data = batch_ori_data.astype(np.float32)
                    batch_ori_data = tf.convert_to_tensor(batch_ori_data)
                    
                    id = id + 1
                    with tf.GradientTape() as img_en_tape, tf.GradientTape() as img_de_tape,\
                            tf.GradientTape() as emb_tape, tf.GradientTape(persistent=True) as text_en_tape, \
                            tf.GradientTape() as text_de_tape,tf.GradientTape() as dis_tape,tf.GradientTape(persistent=True) as fusion_tape,\
                            tf.GradientTape(persistent=True) as dis_en_tape,tf.GradientTape(persistent=True) as dis_de_tape:
                        encoder_img_res,block1,block2,block3,block4,block5 = self.img_en(batch_data_img)
                        
                        decoder_res = self.img_de(encoder_img_res,block1,block2,block3,block4,block5)

                        ori_data_all.append(batch_ori_data)
                        decoder_res_all.append(decoder_res)
                        Hat_x = batch_ori_data * batch_data_m + decoder_res * (1 - batch_data_m)
                        
                        temp = tf.concat(values=[Hat_x, batch_data_h], axis=3)
                        D_prob_en,d_block1 = self.Dis_en(temp)
                        D_prob = self.Dis_de(D_prob_en,d_block1)

                        self.mse_loss = tf.reduce_sum(
                            (batch_data_m * batch_ori_data - batch_data_m * decoder_res) ** 2) / tf.reduce_sum(
                            batch_data_m)
                        trainingLoss += self.mse_loss
                        
                        self.G_loss_temp = -tf.reduce_mean((1 - batch_data_m) * tf.math.log(D_prob + 1e-8))
                        self.loss = self.G_loss_temp + self.mse_loss * 1000
                        self.D_loss_temp = -tf.reduce_mean(
                            batch_data_m * tf.math.log(D_prob + 1e-8) + (1 - batch_data_m) * tf.math.log(1. - D_prob + 1e-8))
                        imputed_loss = tf.reduce_sum(
                            ((1-batch_data_m) * batch_ori_data - (1-batch_data_m) * decoder_res) ** 2) / tf.reduce_sum(
                            1-batch_data_m)
                        totalLoss+=imputed_loss
                        
                        self.gradients_of_en = img_en_tape.gradient(self.loss,
                                                                     self.img_en.trainable_variables)
                        self.encoder_optimizer.apply_gradients(
                            zip(self.gradients_of_en, self.img_en.trainable_variables))

                        self.gradients_of_de = img_de_tape.gradient(self.loss,
                                                                     self.img_de.trainable_variables)
                        self.decoder_optimizer.apply_gradients(
                            zip(self.gradients_of_de, self.img_de.trainable_variables))

                        for i in range(5):
                            
                            self.gradients_of_dis_en = dis_en_tape.gradient(self.D_loss_temp,
                                                                      self.Dis_en.trainable_variables)
                            self.dis_en_optimizer.apply_gradients(
                                zip(self.gradients_of_dis_en, self.Dis_en.trainable_variables))

                            self.gradients_of_dis_de = dis_de_tape.gradient(self.D_loss_temp,
                                                                            self.Dis_de.trainable_variables)
                            self.dis_de_optimizer.apply_gradients(
                                zip(self.gradients_of_dis_de, self.Dis_de.trainable_variables))
                        
                decoder_res_all = tf.concat(decoder_res_all, axis=0)
                ori_data_all = tf.concat(ori_data_all, axis=0)
                
                print("Epoch:", '%04d' % epoch,"imputed_loss:","{:.9f}".format((totalLoss / total_batch)**.5),"g_loss_temp:", "{:.9f}".format(self.G_loss_temp),"mse=", "{:.9f}".format(self.mse_loss),"g_loss=","{:.9f}".format(self.loss), "d_loss=", "{:.9f}".format(self.D_loss_temp))

        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()))
        
    def calculate_is(self, images, batch_size=32, splits=10):
        
        model = InceptionV3(weights='imagenet', include_top=True, pooling='avg')
        
        images = tf.image.resize(images, [299, 299], method='bilinear')
        if images.dtype != tf.float32:
            images = tf.cast(images, tf.float32)
        
        n_images = images.shape[0]
        preds = []
        for i in range(0, n_images, batch_size):
            batch = images[i:i + batch_size]
            pred = model(batch)
            preds.append(pred.numpy())
        preds = np.concatenate(preds, axis=0)

        scores = []
        for k in range(splits):
            part = preds[k * (n_images // splits): (k + 1) * (n_images // splits)]
            py = np.mean(part, axis=0)  
            kl = part * (np.log(part) - np.log(py))
            kl = np.sum(kl, axis=1)
            scores.append(np.exp(np.mean(kl)))

        return np.mean(scores), np.std(scores)

    def calculate_fid(self, real_images, gen_images, batch_size=4):
        
        model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

        def preprocess(images):
            images = tf.image.resize(images, [299, 299], method='bilinear')
            if images.dtype != tf.float32:
                images = tf.cast(images, tf.float32)
            return images  

        def get_features(images):
            images = preprocess(images)
            n_images = images.shape[0]
            features = []
            for i in range(0, n_images, batch_size):
                batch = images[i:i + batch_size]
                feat = model(batch)
                features.append(feat.numpy())
            return np.concatenate(features, axis=0)

        real_features = get_features(real_images)
        gen_features = get_features(gen_images)

        mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
        mu_gen, sigma_gen = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)

        diff = mu_real - mu_gen
        cov_sqrt = sqrtm(sigma_real.dot(sigma_gen))
        if np.iscomplexobj(cov_sqrt):
            cov_sqrt = cov_sqrt.real
        fid = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * cov_sqrt)
        return fid
