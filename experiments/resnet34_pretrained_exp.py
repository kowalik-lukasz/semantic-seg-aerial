# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 20:11:37 2022

@author: Łukasz Kowalik
"""
from preprocessing.utils import clear_and_ttv_split

"""
Clear previous content of the train/test/val dirs, 
then train/test/val split of the patched data
"""

clear_and_ttv_split('potsdam_rgb', 256)
