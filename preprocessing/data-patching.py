# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 22:09:37 2022

@author: Łukasz Kowalik
"""

import os
from skimage import io
from sklearn.preprocessing import MinMaxScaler
from patchify import patchify

min_max_scaler = MinMaxScaler()
patch_size = 240

root_dir = os.path.join('..', 'data')
for path, subdirs, files in os.walk(root_dir):
    # print(path, subdirs, files)
    dirname = path.split(os.path.sep)[-1]
    if dirname == 'rgb':
        dataset_name = path.split(os.path.sep)[-3]
        if dataset_name == 'potsdam':
            tiles = os.listdir(path)
            # limit = 2
            for tile_name in tiles:
                # if limit == 0:
                #     break
                if tile_name.endswith('.tif'):
                    # limit-=1
                    tile = io.imread(os.path.join(path, tile_name))
                    
                    print('Patchifying image tile: ', os.path.join(path, 
                                                                    tile_name))
                    patched_tile = patchify(tile, (patch_size, patch_size, 3),
                                            step=patch_size)
                    
                    for i in range(patched_tile.shape[0]):
                        for j in range(patched_tile.shape[1]):
                            single_patch = patched_tile[i,j]
                            single_patch = single_patch[0]
                            patch_path = os.path.join(path, tile_name)[:-4] + \
                            '_' + str(i) + '_' + str(j) + '.png'
                            patch_path = patch_path.replace('images',
                                                            'patched_images')
                            # print(patch_path)
                            io.imsave(patch_path, single_patch)
                            
    elif dirname == 'labels':
        dataset_name = path.split(os.path.sep)[-2]
        if dataset_name == 'potsdam':
            labels = os.listdir(path)
            # limit = 2
            for label_name in labels:
                # if limit == 0:
                #     break
                if label_name.endswith('.tif'):
                    # limit-=1
                    label = io.imread(os.path.join(path, label_name))
                     
                    print('Patchifying label tile: ', os.path.join(path, 
                                                              label_name))
                    patched_label = patchify(label,
                                            (patch_size, patch_size, 3),
                                             step=patch_size)
                     
                    for i in range(patched_label.shape[0]):
                        for j in range(patched_label.shape[1]):
                            single_patch = patched_label[i,j]
                            single_patch = single_patch[0]
                            patch_path = os.path.join(path, label_name)[:-4] + \
                            '_' + str(i) + '_' + str(j) + '.png'
                            patch_path = patch_path.replace('labels',
                                                            'patched_labels')
                            # print(patch_path)
                            io.imsave(patch_path, single_patch)
                     
                     
