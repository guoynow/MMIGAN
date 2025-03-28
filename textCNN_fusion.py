import numpy as np
import tensorflow as tf
from keras.layers import Embedding, Conv1D, Conv2D, Conv2DTranspose, GlobalAveragePooling1D, Dense, Concatenate, GlobalMaxPooling2D, UpSampling1D
from keras import Model
import keras.backend as K

import fusionModel
import fusionModel_text
import utilss
from imageCNN import encoder

class text_embedding(Model):
    def __init__(self, maxlen, max_features, embedding_dims):
        super().__init__()
        self.embedding = Embedding(input_dim=max_features, output_dim=embedding_dims, input_length=maxlen)
    def call(self, inputs):
        emb = self.embedding(inputs)  
        emb = tf.transpose(emb, [0, 1, 3, 2])  
        return emb
class text_encoder(Model):
    def __init__(self, filter_num, embedding_dims, kernel_regularizer):
        super(text_encoder, self).__init__()
        self.conv1 = Conv2D(filters=filter_num, kernel_size=(2, embedding_dims), activation='relu',
                   kernel_regularizer=kernel_regularizer)
        self.conv2 = Conv2D(filters=filter_num, kernel_size=(3, embedding_dims), activation='relu',
                            kernel_regularizer=kernel_regularizer)
        self.conv3 = Conv2D(filters=filter_num, kernel_size=(4, embedding_dims), activation='relu',
                            kernel_regularizer=kernel_regularizer)
        self.pool = GlobalMaxPooling2D()  
    def call(self, inputs):
        conca = []
        a1 = self.conv1(inputs)
        b1 = self.pool(a1)  
        conca.append(b1)

        a2 = self.conv2(inputs)
        b2 = self.pool(a2)
        conca.append(b2)

        a3 = self.conv3(inputs)
        b3 = self.pool(a3)
        conca.append(b3)
        
        x = Concatenate()(conca)
        
        return x
class text_decoder(Model):
    def __init__(self, filter_num, embedding_dims, kernel_regularizer, maxlen):
        super(text_decoder, self).__init__()

        self.maxlen = maxlen
        self.filter_num = filter_num
        self.unconv1 = Conv2DTranspose(filters=filter_num, kernel_size=(2, embedding_dims),
                            activation='relu',
                            kernel_regularizer=kernel_regularizer)
        self.unconv2 = Conv2DTranspose(filters=filter_num, kernel_size=(3, embedding_dims),
                                      activation='relu',
                                      kernel_regularizer=kernel_regularizer)
        self.unconv3 = Conv2DTranspose(filters=filter_num, kernel_size=(4, embedding_dims),
                                      activation='relu',
                                      kernel_regularizer=kernel_regularizer)
    def call(self, inputs):
        print("inputs:",np.array(inputs).shape)
        conca = []
        x_pred = []
        res = tf.split(inputs, axis=1, num_or_size_splits=[self.filter_num, self.filter_num, self.filter_num])
        c1 = tf.tile(res[0], [1, self.maxlen - 2 + 1])  
        c1 = tf.reshape(c1, [-1, self.maxlen - 2 + 1, 1,
                           self.filter_num])  
        
        d1 = self.unconv1(c1)
        
        d1 = tf.reduce_mean(d1, axis=-1)
        
        conca.append(d1)

        c2 = tf.tile(res[1], [1, self.maxlen - 3 + 1])  
        c2 = tf.reshape(c2, [-1, self.maxlen - 3 + 1, 1,
                            self.filter_num])  
        
        d2 = self.unconv2(c2)
        d2 = tf.reduce_mean(d2, axis=-1)
        
        conca.append(d2)

        c3 = tf.tile(res[2], [1, self.maxlen - 4 + 1])  
        c3 = tf.reshape(c3, [-1, self.maxlen - 4 + 1, 1,
                             self.filter_num])  
        
        d3 = self.unconv3(c3)
        d3 = tf.reduce_mean(d3, axis=-1)
        
        conca.append(d3)

        for i in range(1, len(conca)):
            conca[0] += conca[i]
        x_pred = conca[0] / len(conca)

        return x_pred
class TextCNN(Model):
    def __init__(self, maxlen, max_features, embedding_dims, class_num, filter_num, kernel_sizes=[2, 3, 4],
                 kernel_regularizer=None,  last_activation='softmax',
                 learning_rate=0.0001, epochs=10, batch_size=128):
        super(TextCNN, self).__init__()
        self.maxlen = maxlen
        self.max_features = max_features
        self.kernel_sizes = kernel_sizes
        self.kernel_regularizer = kernel_regularizer
        self.class_num = class_num
        self.filter_num = filter_num
        self.embedding_dims = embedding_dims
        self.embedding = Embedding(input_dim=max_features, output_dim=embedding_dims, input_length=maxlen)

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        self.en_img_optimizer = tf.keras.optimizers.Adam(0.000001)
        self.fusion_optimizer = tf.keras.optimizers.Adam(0.000001)
        self.en_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.de_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.emb_optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        self.img_en = encoder()
        self.fusion = fusionModel_text.Fusion_Model()
        self.en = text_encoder(self.filter_num, self.embedding_dims, self.kernel_regularizer)

        self.de = text_decoder(self.filter_num, self.embedding_dims, self.kernel_regularizer,
                                    self.maxlen)
        self.text_embedding = text_embedding(self.maxlen, self.max_features, self.embedding_dims)

    def train_model(self, x_train, tokenizer):
        display_step = 1
        save_step = 1
        result_fusion = []
        number_fusion = []
        
        print("Start training...")
        train_ds_img = utilss.DataLoader_img()
        batch_no = int(x_train.shape[0] / self.batch_size)
        ori_x_train = x_train
        x_train = tf.reshape(x_train,shape=(x_train.shape[0],x_train.shape[1],1,1))
        x_train = tf.cast(x_train,dtype=float)

        ckpt_en_img = tf.train.Checkpoint(model=self.img_en, optimizer=self.en_img_optimizer)
        ckpt_manager_en_img = tf.train.CheckpointManager(ckpt_en_img, './model/vgg_en', max_to_keep=3)
        ckpt_en_img.restore(tf.train.latest_checkpoint('./model/vgg_en'))

        ckpt_fusion = tf.train.Checkpoint(model=self.fusion, optimizer=self.fusion_optimizer)
        ckpt_manager_fusion = tf.train.CheckpointManager(ckpt_fusion, './model/fusion_model', max_to_keep=3)
        ckpt_fusion.restore(tf.train.latest_checkpoint('./model/fusion_model'))

        ckpt_en = tf.train.Checkpoint(model=self.en, optimizer=self.en_optimizer)
        ckpt_manager_en = tf.train.CheckpointManager(ckpt_en, './model/text_en', max_to_keep=3)
        ckpt_en.restore(tf.train.latest_checkpoint('./model/text_en'))

        ckpt_de = tf.train.Checkpoint(model=self.de, optimizer=self.de_optimizer)
        ckpt_manager_de = tf.train.CheckpointManager(ckpt_de, './model/text_de', max_to_keep=3)
        ckpt_de.restore(tf.train.latest_checkpoint('./model/text_de'))

        ckpt_emb = tf.train.Checkpoint(model=self.text_embedding, optimizer=self.emb_optimizer)
        ckpt_manager_emb = tf.train.CheckpointManager(ckpt_emb, './model/text_emb', max_to_keep=3)
        ckpt_emb.restore(tf.train.latest_checkpoint('./model/text_emb'))

        for epoch in range(self.epochs):
            
            for i in range(batch_no):
                
                batch_text,batch_img = utilss.get_batch_double(x_train,train_ds_img.train_data,self.batch_size,i,batch_no)

                batch_img = batch_img.astype(np.float32)
                batch_img = tf.convert_to_tensor(batch_img)
                batch_img_norm = batch_img / 255.0

                batch_text = tf.reshape(batch_text, shape=(batch_text.shape[0], batch_text.shape[1], 1,1))
                norm_emb1 = tf.reshape(batch_text,shape=(batch_text.shape[0],batch_text.shape[1]*batch_text.shape[2]*batch_text.shape[3]))
                norm_emb, parameters = utilss.normalization(norm_emb1)
                norm_emb = tf.reshape(norm_emb,shape=(batch_text.shape[0],batch_text.shape[1],batch_text.shape[2],batch_text.shape[3]))

                with tf.GradientTape() as en_tape, tf.GradientTape() as de_tape, tf.GradientTape() as en_img_tape, tf.GradientTape() as fusion_tape:
                    
                    enc_list= self.en(norm_emb)
                    enc_img = self.img_en(batch_img_norm)
                    fusion_data = self.fusion(enc_img,enc_list)

                    dec_list = self.de(fusion_data)
                    x_emb_norm = tf.reshape(norm_emb, shape=[-1, self.maxlen,
                                                             self.embedding_dims])  

                    mse = tf.keras.losses.MeanSquaredError()
                    self.loss = tf.sqrt(mse(x_emb_norm, dec_list))

                    self.en_img_gradients = en_img_tape.gradient(self.loss,
                                                                 self.img_en.trainable_variables)
                    self.en_img_optimizer.apply_gradients(
                        zip(self.en_img_gradients, self.img_en.trainable_variables))

                    self.gradients_of_fusion = fusion_tape.gradient(self.loss,
                                                                    self.fusion.trainable_variables)
                    self.fusion_optimizer.apply_gradients(
                        zip(self.gradients_of_fusion, self.fusion.trainable_variables))

                    self.en_gradients = en_tape.gradient(self.loss,
                                                         self.en.trainable_variables)
                    self.en_optimizer.apply_gradients(
                        zip(self.en_gradients, self.en.trainable_variables))

                    self.de_gradients = de_tape.gradient(self.loss,
                                                         self.de.trainable_variables)
                    self.de_optimizer.apply_gradients(
                        zip(self.de_gradients, self.de.trainable_variables))

                    dec_list = tf.reshape(dec_list,shape=(dec_list.shape[0],dec_list.shape[1]*dec_list.shape[2]))
                    renormalization = utilss.renormalization(dec_list, parameters)
                    renormalization = np.array(renormalization)
                    renormalization = utilss.round_all(renormalization)
                    if (epoch == 499 and i == batch_no - 1):
                        np.savetxt("./x_pred_fusion.txt", renormalization[390], delimiter=',', fmt='%s')
                        np.savetxt("./x_emb123_fusion.txt", norm_emb1[390], delimiter=',', fmt='%s')
                    if (epoch == 499):
                        
                        result_fusion.append(tokenizer.sequences_to_texts(renormalization))
                        number_fusion.append(renormalization)

                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(self.loss))
            np.savetxt("./result_fusion.txt", result_fusion, delimiter=',', fmt='%s')
            np.savetxt("./number_fusion.txt", number_fusion, delimiter=',', fmt='%s')

            if epoch % save_step == 0:
                path_en = ckpt_manager_en.save()
                path_de = ckpt_manager_de.save()
                path_emb = ckpt_manager_emb.save()
                path_img = ckpt_manager_en_img.save()
                path_fusion = ckpt_manager_fusion.save()
                print("model_en saved to %s" % path_en)
                print("model_de saved to %s" % path_de)
                print("model_de saved to %s" % path_emb)
                print("model_de saved to %s" % path_img)
                print("model_de saved to %s" % path_fusion)
