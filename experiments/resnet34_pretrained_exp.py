# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 20:11:37 2022

@author: Łukasz Kowalik
"""
import os
import splitfolders

"""
Clear previous content of the train/test/val dirs, 
then train/test/val split of the patched data
"""
input_folder = os.path.join('..', 'data', 'potsdam_rgb', '256_patches')
output_folder = os.path.join('..', 'data', 'potsdam_rgb')
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