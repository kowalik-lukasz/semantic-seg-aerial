# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 21:03:29 2022

@author: Łukasz Kowalik
"""
import os
import numpy as np
import splitfolders
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

"""
Clear previous content of the train/test/val dirs, 
then train/test/val split of the patched data
"""
def clear_and_ttv_split(dataset: str, patch_size: int):
    input_folder = os.path.join('..', 'data', dataset, str(patch_size) + '_patches')
    output_folder = os.path.join('..', 'data', dataset)
    to_clear = [
        os.path.join(output_folder, 'train'),
        os.path.join(output_folder, 'test'),
        os.path.join(output_folder, 'val')
    ]
    
    for directory in to_clear:
        for path, subdirs, files in os.walk(directory):
            if path.endswith('images') or path.endswith('labels'):
                for f in files:
                    if f != '.gitkeep':
                        os.remove(os.path.join(path, f))
                
    splitfolders.ratio(input_folder, output=output_folder, 
                        seed=1998, ratio=(.8, .1, .1), group_prefix=None)


def rgb_to_2D_label(label):
    class_dict = {
        'impervious': [255, 255, 255],
        'building': [0, 0, 255],
        'low_veg': [0, 255, 255],
        'tree': [0, 255, 0],
        'car': [255, 255, 0],
        'clutter': [255, 0, 0]
    }
    label2D = np.zeros(label.shape, dtype=np.uint8)
    label2D[np.all(label==class_dict['impervious'],axis=-1)] = 0
    label2D[np.all(label==class_dict['building'],axis=-1)] = 1
    label2D[np.all(label==class_dict['low_veg'],axis=-1)] = 2
    label2D[np.all(label==class_dict['tree'],axis=-1)] = 3
    label2D[np.all(label==class_dict['car'],axis=-1)] = 4
    label2D[np.all(label==class_dict['clutter'],axis=-1)] = 5
    
    label2D = label2D[:,:,0]
    label2D = np.expand_dims(label2D, axis=2)
    return label2D
    

def preprocess_data(img, label, n_classes, scaler, sm_input):
    img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
    if sm_input:
        img = sm_input(img)
        
    label = rgb_to_2D_label(label)
    label = to_categorical(label, n_classes)
    return (img, label)


def generate_train_data(train_img_path, train_label_path, batch_size, seed):
    gen_args = dict(horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='reflect')
    
    img_datagen = ImageDataGenerator(**gen_args)
    label_datagen = ImageDataGenerator(**gen_args)
    
    img_gen = img_datagen.flow_from_directory(train_img_path,
                                              class_mode=None,
                                              batch_size=batch_size,
                                              seed=seed)
    label_gen = label_datagen.flow_from_directory(train_label_path,
                                                  class_mode=None,
                                                  color_mode='rgb',
                                                  batch_size=batch_size,
                                                  seed=seed)
    
    train_gen = zip(img_gen, label_gen)
    for (img, label) in train_gen:
        yield (img, label)
