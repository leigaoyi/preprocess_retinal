# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 21:20:16 2019

@author: kasy
"""

import os
import cv2

label_path = './dataset_transfer/DRIVE/train/labels/'

label_list = []
for i in os.listdir(label_path):
    label_list.append(os.path.join(label_path, i))
    
label = cv2.imread(label_list[0])