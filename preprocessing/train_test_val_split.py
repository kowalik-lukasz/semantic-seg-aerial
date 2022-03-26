# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 21:03:29 2022

@author: Łukasz Kowalik
"""
import os
import splitfolders

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