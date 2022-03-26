# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 20:11:37 2022

@author: Łukasz Kowalik
"""
import os
import numpy as np
import segmentation_models as sm
import random
from skimage import io
from matplotlib import pyplot as plt
from preprocessing.train_test_val_split import clear_and_ttv_split
from sklearn.preprocessing import MinMaxScaler

"""
Clear previous content of the train/test/val dirs, 
then train/test/val split of the patched data
"""
# clear_and_ttv_split('potsdam_rgb', 256)


"""
Initial sanity check of the data
"""
train_img_dir = os.path.join('..', 'data', 'potsdam_rgb', 'train', 'images')
train_label_dir = os.path.join('..', 'data', 'potsdam_rgb', 'train', 'labels')

img_list = os.listdir(train_img_dir)
label_list = os.listdir(train_label_dir)
n_images = len(img_list)

img_index = random.randint(0, n_images-1)
rand_img = io.imread(os.path.join(train_img_dir,img_list[img_index]))
rand_label = io.imread(os.path.join(train_label_dir,label_list[img_index]))

plt.figure(figsize=(12, 8))
plt.subplot(121)
plt.imshow(rand_img)
plt.title('Image')
plt.subplot(122)
plt.imshow(rand_label)
plt.title('Ground Truth')
plt.show()


"""
Image generator for reading data directly from the drive.
"""
seed = 1998
batch_size = 16
n_classes = 6

scaler = MinMaxScaler()
backbone = 'resnet34'
preprocess_input = sm.get_preprocessing(backbone)


