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
from preprocessing.utils import clear_and_ttv_split, preprocess_data, generate_train_data, rgb_to_2D_label
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
print('Random image: ' + os.path.join(train_img_dir,img_list[img_index]))
print('Random label: ' + os.path.join(train_label_dir,label_list[img_index]))

plt.figure(figsize=(12, 8))
plt.subplot(121)
plt.imshow(rand_img)
plt.title('Image')
plt.subplot(122)
plt.imshow(rand_label)
plt.title('Ground Truth')
plt.show()


"""
Image generator for reading data directly from the drive 
with data augmentation (horizontal + vertical flip methods)
"""
seed = 1998
batch_size = 16
n_classes = 6

scaler = MinMaxScaler()
backbone = 'resnet34'
backbone_input = sm.get_preprocessing(backbone)

train_aug_img_dir = os.path.join('..', 'data', 'potsdam_rgb', 'data_for_augmentation', 'train_images')
train_aug_label_dir = os.path.join('..', 'data', 'potsdam_rgb', 'data_for_augmentation', 'train_labels')
val_aug_img_dir = os.path.join('..', 'data', 'potsdam_rgb', 'data_for_augmentation', 'val_images')
val_aug_label_dir = os.path.join('..', 'data', 'potsdam_rgb', 'data_for_augmentation', 'val_labels')

train_dir = os.path.dirname(train_img_dir)
train_img_gen = generate_train_data(train_aug_img_dir, train_aug_label_dir, batch_size, seed)
val_img_gen = generate_train_data(val_aug_img_dir, val_aug_label_dir, batch_size, seed)

X_raw, y_raw = train_img_gen.__next__()
for i in range(3):
    image = X_raw[i]
    label = y_raw[i]
    X, y = preprocess_data(X_raw[i], y_raw[i], n_classes, scaler, backbone_input)
    plt.subplot(1,2,1)
    plt.imshow(X)
    plt.subplot(1,2,2)
    plt.imshow(np.argmax(y, axis=2))
    plt.show()
    
X_raw, y_raw = val_img_gen.__next__()
for i in range(3):
    image = X_raw[i]
    label = y_raw[i]
    X, y = preprocess_data(X_raw[i], y_raw[i], n_classes, scaler, backbone_input)
    plt.subplot(1,2,1)
    plt.imshow(X)
    plt.subplot(1,2,2)
    plt.imshow(np.argmax(y, axis=2))
    plt.show()
    
    
    
    