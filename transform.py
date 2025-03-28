import tensorflow as tf
import os
import numpy as np
import glob

from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

class transfromm():
    def __init__(self):
        train_path1 = "Data/flickr8k"
        path_list1 = os.listdir(train_path1)
        path_list1.sort(key=lambda x: int(x.split('.')[0]))

        def read_image(path_list):
            images = []
            for i in path_list:
                image_temp = tf.io.read_file(train_path1+"/"+i)
                image_temp = tf.image.decode_jpeg(image_temp)
                print("transform_id:", i)
                images.append(image_temp)
            return np.array(images,dtype=object)

        train_images1 = read_image(path_list=path_list1)
        self.train_data = train_images1
        print("shape:", self.train_data.shape)