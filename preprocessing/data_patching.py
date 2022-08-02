# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 22:09:37 2022

@author: Łukasz Kowalik
"""

import os
import numpy as np
from skimage import io
from patchify import patchify
from PIL import Image

"""
Patchifying source images along with the corresponding masks to 
fragments and saving them to ../data/patched_{images/labels} directory
"""

# Config variables
patch_size = 256
dataset_folder = 'potsdam_rgb_windowed'

root_dir = os.path.join('..', 'data')
for path, subdirs, files in os.walk(root_dir):
    print(path, subdirs, files)
    dirname = path.split(os.path.sep)[-1]
    if dirname == 'images':
        dataset_name = path.split(os.path.sep)[-2]
        if dataset_name == dataset_folder:
            tiles = os.listdir(path)
            for tile_name in tiles:
                if tile_name.endswith('.tif'):
                    tile = io.imread(os.path.join(path, tile_name))
                    SIZE_X = (tile.shape[1]//patch_size) * patch_size
                    SIZE_Y = (tile.shape[0]//patch_size) * patch_size
                    tile = Image.fromarray(tile)
                    tile = tile.crop((0, 0, SIZE_X, SIZE_Y))
                    tile = np.array(tile)
                    
                    print('Patchifying image tile: ', os.path.join(path, 
                                                                    tile_name))
                    patched_tile = patchify(tile, (patch_size, patch_size, 3),
                                            step=128)
                    
                    for i in range(patched_tile.shape[0]):
                        for j in range(patched_tile.shape[1]):
                            single_patch = patched_tile[i,j]
                            single_patch = single_patch[0]
                            patch_path = os.path.join(path, tile_name)[:-4] + \
                            '_' + str(i) + '_' + str(j) + '.png'
                            patch_path = patch_path.replace('images',
                                                            os.path.join(str(patch_size) + '_patches', 
                                                                         'images'))
                            io.imsave(patch_path, single_patch)
                            
    elif dirname == 'labels':
        dataset_name = path.split(os.path.sep)[-2]
        if dataset_name == dataset_folder:
            labels = os.listdir(path)
            for label_name in labels:
                if label_name.endswith('.tif'):
                    label = io.imread(os.path.join(path, label_name))
                    SIZE_X = (label.shape[1]//patch_size) * patch_size
                    SIZE_Y = (label.shape[0]//patch_size) * patch_size
                    label = Image.fromarray(label)
                    label = label.crop((0, 0, SIZE_X, SIZE_Y))
                    label = np.array(label)
                     
                    print('Patchifying label tile: ', os.path.join(path, 
                                                              label_name))
                    patched_label = patchify(label,
                                            (patch_size, patch_size, 3),
                                             step=128)
                     
                    for i in range(patched_label.shape[0]):
                        for j in range(patched_label.shape[1]):
                            single_patch = patched_label[i,j]
                            single_patch = single_patch[0]
                            patch_path = os.path.join(path, label_name)[:-4] + \
                            '_' + str(i) + '_' + str(j) + '.png'
                            patch_path = patch_path.replace('labels',
                                                            os.path.join(str(patch_size)+ '_patches', 
                                                                         'labels'))
                            io.imsave(patch_path, single_patch)