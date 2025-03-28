import os

import numpy as np
import tensorflow as tf
import jieba
from gensim.models.word2vec import Word2Vec

from collections import defaultdict
from keras.preprocessing.text import Tokenizer

from keras_preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences

import textCNN_ori
import transform
import time

import sys
import numpy
import numpy as np

import math

import tensorflow as tf

import transform_train

def max_poo_with_argmax(net, stride):
    _, mask = tf.nn.max_pool_with_argmax(net, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding='SAME')
    mask = tf.stop_gradient(mask)
    net = tf.nn.max_pool(net, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding='SAME')
    return net, mask

def unpool(net, mask, stride):
    ksize = [1, stride, stride, 1]
    input_shape = net.get_shape().as_list()
    
    output_shape = (input_shape[0], input_shape[1]*ksize[1], input_shape[2]*ksize[2], input_shape[3])
    
    one_like_mask = tf.ones_like(mask)
    batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64), shape=[input_shape[0], 1, 1, 1])
    b = one_like_mask * batch_range
    y = mask // (output_shape[2] * output_shape[3])
    x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]
    feature_range = tf.range(output_shape[3], dtype=tf.int64)
    f = one_like_mask * feature_range
    
    update_size = tf.size(net)
    indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, update_size]))
    values = tf.reshape(net, [update_size])
    ret = tf.scatter_nd(indices, values, output_shape)
    return ret

def rmse(predictions, targets):
    
    return tf.sqrt(tf.reduce_mean(tf.subtract(predictions, targets)))

class DataLoader_img():
    def __init__(self):
        t = transform.transfromm()
        self.train_data = t.train_data
        self.num_train_data = self.train_data.shape[0]
        saveData = (self.train_data / 255.0).reshape((5000,32*32*3))
        np.savetxt("flickr30k-255-no.txt",saveData)
class DataLoader_img_train():
    def __init__(self):
        t = transform_train.transfromm()
        self.train_data = t.train_data
        self.num_train_data = self.train_data.shape[0]

class word2vec():

    def generate_training_data(self, corpus):
        
        word_counts = defaultdict(int)

        for row in corpus:
            for word in row:
                
                word_counts[word] += 1

        self.v_count = len(word_counts.keys())
        print(self.v_count)
        
        self.words_list = list(word_counts.keys())
        
        self.word_index = dict((word, i) for i, word in enumerate(self.words_list))
        
        self.index_word = dict((i, word) for i, word in enumerate(self.words_list))

        training_data = []

        for sentence in corpus:
            sent_len = len(sentence)
            id = 0

            for i, word in enumerate(sentence):
                id = id + 1
                w_target = self.word2onehot(sentence[i])
                
                training_data.append(w_target)
                if (sentence[i] == "." and id < 35):
                    for j in range(id, 35):
                        w_target = [0 for k in range(0, self.v_count)]
                        training_data.append(w_target)
                    id = 0
                elif(sentence[i] =="." and id>=35):
                    id =0

        return np.array(training_data, dtype=object), self.v_count

    def word2onehot(self, word):

        word_vec = [0 for i in range(0, self.v_count)]

        word_index = self.word_index[word]

        word_vec[word_index] = 1

        return word_vec

class DataLoader_text():
    def __init__(self):
        f = open('Data/flickr8k.txt')
        
        l = []
        for i in f.readlines():

            i = jieba.lcut(i, cut_all=False)
            for j in i:
                while ' ' in i:
                    i.remove(' ')
                while '.' in i:
                    i.remove('.')
                while '\n' in i:
                    i.remove('\n')
            l.append(i)

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(l)

        sequences = tokenizer.texts_to_sequences(l)

        x_train = sequence.pad_sequences(sequences, padding='post')

        self.training_data = x_train
        self.max_features = len(tokenizer.word_index) +1
        self.tokenizer = tokenizer

class DataSet_Double(object):
    def __init__(self, input):
        self._input = input
        
        self._num_examples = input.shape[0]  
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size, shuffle=True):

        start = self._index_in_epoch

        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._input = self._input[perm0]

        if start + batch_size > self._num_examples:

            self._epochs_completed += 1

            rest_num_examples = self._num_examples - start
            input_rest_part = self._input[start:self._num_examples]

            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._input = self._input[perm]

            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            input_new_part = self._input[start:end]

            return np.concatenate((input_rest_part, input_new_part), axis=0)

        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._input[start:end]\

class DataSet_Double1(object):
    def __init__(self, input,input_noise):
        self._input = input
        self._input_noise = input_noise
        self._num_examples = input.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size, shuffle=True):

        start = self._index_in_epoch
        
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._input = self._input[perm0]
            self._input_noise = self._input_noise[perm0]
        
        if start + batch_size > self._num_examples:
            
            self._epochs_completed += 1
            
            rest_num_examples = self._num_examples - start
            input_rest_part = self._input[start:self._num_examples]
            input_noise_rest_part = self._input_noise[start:self._num_examples]
            
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._input = self._input[perm]
                self._input_noise = self._input_noise[perm]
            
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            input_new_part = self._input[start:end]
            input_noise_new_part = self._input_noise[start:end]
            return np.concatenate((input_rest_part, input_new_part), axis=0),\
                   np.concatenate((input_noise_rest_part, input_noise_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._input[start:end]\
                , self._input_noise[start:end]

def get_batch(train_data, batch_size, now_batch, total_batch):
    if now_batch < total_batch - 1:
        image_batch = train_data[now_batch * batch_size:(now_batch + 1) * batch_size]

    else:
        image_batch = train_data[now_batch * batch_size:]

    return np.array(image_batch)

def get_batch_double(train_data1, train_data2, batch_size, now_batch, total_batch):
    if now_batch < total_batch - 1:
        train_data1_batch = train_data1[now_batch * batch_size:(now_batch + 1) * batch_size]
        train_data2_batch = train_data2[now_batch * batch_size:(now_batch + 1) * batch_size]

    else:
        train_data1_batch = train_data1[now_batch * batch_size:]
        train_data2_batch = train_data2[now_batch * batch_size:]

    return np.array(train_data1_batch), np.array(train_data2_batch)
def get_batch_triboule(train_data1, train_data2, train_data3, batch_size, now_batch, total_batch):
    if now_batch < total_batch - 1:
        train_data1_batch = train_data1[now_batch * batch_size:(now_batch + 1) * batch_size]
        train_data2_batch = train_data2[now_batch * batch_size:(now_batch + 1) * batch_size]
        train_data3_batch = train_data3[now_batch * batch_size:(now_batch + 1) * batch_size]

    else:
        train_data1_batch = train_data1[now_batch * batch_size:]
        train_data2_batch = train_data2[now_batch * batch_size:]
        train_data3_batch = train_data3[now_batch * batch_size:]

    return np.array(train_data1_batch), np.array(train_data2_batch),np.array(train_data3_batch)
def get_batch_fourth(train_data1, train_data2, train_data3, train_data4, batch_size, now_batch, total_batch):
    if now_batch < total_batch - 1:
        train_data1_batch = train_data1[now_batch * batch_size:(now_batch + 1) * batch_size]
        train_data2_batch = train_data2[now_batch * batch_size:(now_batch + 1) * batch_size]
        train_data3_batch = train_data3[now_batch * batch_size:(now_batch + 1) * batch_size]
        train_data4_batch = train_data4[now_batch * batch_size:(now_batch + 1) * batch_size]

    else:
        train_data1_batch = train_data1[now_batch * batch_size:]
        train_data2_batch = train_data2[now_batch * batch_size:]
        train_data3_batch = train_data3[now_batch * batch_size:]
        train_data4_batch = train_data4[now_batch * batch_size:]

    return np.array(train_data1_batch), np.array(train_data2_batch),np.array(train_data3_batch),np.array(train_data4_batch)
def get_batch_fifth(train_data1, train_data2, train_data3, train_data4, train_data5, batch_size, now_batch, total_batch):
    if now_batch < total_batch - 1:
        train_data1_batch = train_data1[now_batch * batch_size:(now_batch + 1) * batch_size]
        train_data2_batch = train_data2[now_batch * batch_size:(now_batch + 1) * batch_size]
        train_data3_batch = train_data3[now_batch * batch_size:(now_batch + 1) * batch_size]
        train_data4_batch = train_data4[now_batch * batch_size:(now_batch + 1) * batch_size]
        train_data5_batch = train_data5[now_batch * batch_size:(now_batch + 1) * batch_size]

    else:
        train_data1_batch = train_data1[now_batch * batch_size:]
        train_data2_batch = train_data2[now_batch * batch_size:]
        train_data3_batch = train_data3[now_batch * batch_size:]
        train_data4_batch = train_data4[now_batch * batch_size:]
        train_data5_batch = train_data5[now_batch * batch_size:]

    return np.array(train_data1_batch), np.array(train_data2_batch),np.array(train_data3_batch),np.array(train_data4_batch),np.array(train_data5_batch)

def normalization(data, parameters=None):

    norm_data = data

    if parameters is None:

        min_arr = tf.reduce_min(norm_data, axis=0)
        max_arr = tf.reduce_max(norm_data + 1e-6, axis=0)

        norm_data = (norm_data - min_arr)/(max_arr - min_arr)

        norm_parameters = {'min_val': min_arr,
                           'max_val': max_arr}
    else:
        min_arr = parameters['min_val']
        max_arr = parameters['max_val']
        norm_data = (norm_data - min_arr) / (max_arr - min_arr)
        norm_parameters = parameters
    return norm_data, norm_parameters

def renormalization(norm_data, norm_parameters):

    min_arr = norm_parameters['min_val']
    max_arr = norm_parameters['max_val']

    min_arr = tf.cast(min_arr, dtype=tf.float32)
    max_arr = tf.cast(max_arr, dtype=tf.float32)
    norm_data = tf.cast(norm_data, dtype=tf.float32)

    renorm_data = norm_data * (max_arr - min_arr) + min_arr

    return renorm_data

def mask_center_percent(p, a, b, c, d):
    misl = round(math.sqrt(1024 * p))
    print(misl)

    start_x = (32 - misl) // 2
    end_x = 32 - start_x
    start_y = (32 - misl) // 2
    end_y = 32 - start_x
    if misl % 2 == 1:
        end_x += 1
        end_y += 1
    data_masked = np.ones((a, b, c, d))
    data_masked[:, start_x:end_x, start_y:end_y, :] = 0

    return data_masked

def binary_sampler(p, a, b, c,d):
    np.random.seed(0)
    unif_random_matrix = np.random.uniform(0., 1., size=[a, b,c,d])
    
    binary_random_matrix = 1 * (unif_random_matrix < p)
    
    return binary_random_matrix

def miss_data_gen(miss_rate, data):
    data_m = binary_sampler(1 - miss_rate, data.shape[0],
                            data.shape[1], data.shape[2], data.shape[3]).astype(np.float32)
    
    miss_data = np.array(data)

    miss_data[data_m == 0] = 0

    return miss_data, data_m

def save_img(data, url):
    if not os.path.exists(url):
        os.makedirs(url)
    data = np.array(data)
    data = data.astype(np.float32)
    data = tf.convert_to_tensor(data)
    ori_list = tf.split(data, num_or_size_splits=data.shape[0], axis=0)
    id = 0
    for i in range(data.shape[0]):
        ima = tf.reshape(ori_list[i], shape=(data.shape[1], data.shape[2], 3))
        ima = tf.cast(ima, dtype=tf.uint8)
        ima = tf.image.encode_jpeg(ima)
        fwrite = tf.io.write_file(url + str(id) + ".jpeg", ima)
        id += 1
def round_all(data):
    for i in range(len(data)):
        for j in range(len(data[0])):
         data[i][j] = round(data[i][j])
         
    return data
def sigmoid(x):
    if np.all(x>=0):
        return 1. / (1 + numpy.exp(-x))
    else:
        return numpy.exp(x)/(1+numpy.exp(x))

def softmax(x):
    e = numpy.exp(x - numpy.max(x))  
    if e.ndim == 1:
        return e / numpy.sum(e, axis=0)
    else:
        return e / numpy.array([numpy.sum(e, axis=1)]).T  
class RBM(object):
    def __init__(self, input=None, n_visible=2, n_hidden=3, \
                 W=None, hbias=None, vbias=None, numpy_rng=None):

        self.n_visible = n_visible  
        self.n_hidden = n_hidden  

        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(1234)

        if W is None:
            a = 1. / n_visible
            initial_W = numpy.array(numpy_rng.uniform(  
                low=-a,
                high=a,
                size=(n_visible, n_hidden)))

            W = initial_W

        if hbias is None:
            hbias = numpy.zeros(n_hidden)  

        if vbias is None:
            vbias = numpy.zeros(n_visible)  

        self.numpy_rng = numpy_rng
        self.input = input
        self.W = W
        self.hbias = hbias
        self.vbias = vbias

    def contrastive_divergence(self, lr=0.1, k=1, input=None):
        if input is not None:
            self.input = input
