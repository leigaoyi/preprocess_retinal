#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 09:10:57 2019

@author: robot
"""

#from preprocess.pre_DUNet import my_preprocess
#from preprocess.pre_anotation import my_preprocess
#from preprocess.pre_hiera import my_preprocess
#from preprocess.pre_deep import my_preprocess
from preprocess.pre_multi import my_preprocess
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt



process_method = 'Multi'

dataset_name = 'HRF'

if not os.path.exists('./data/{0}/{1}/train/images/'.format(process_method, dataset_name)):
    os.makedirs('./data/{0}/{1}/train/images/'.format(process_method, dataset_name))
    os.makedirs('./data/{0}/{1}/train/labels/'.format(process_method, dataset_name))
    os.makedirs('./data/{0}/{1}/test/images/'.format(process_method, dataset_name))
    os.makedirs('./data/{0}/{1}/test/labels/'.format(process_method, dataset_name))

train_dir_path = './dataset_transfer/{}/train/'.format(dataset_name)
test_dir_path = './dataset_transfer/{}/test/'.format(dataset_name)

train_img_list = []
train_label_list = []
test_img_list = []
test_label_list = []

count = 0

def extract_whole_patches(img, patch_size=48):
    w_size, h_size = img.shape[0], img.shape[1]
    w_num, h_num = w_size//patch_size, h_size//patch_size
    
    img_zeros = np.zeros([(w_num+1)*patch_size, (h_num+1)*patch_size, 3])
    img_zeros[:w_size, :h_size, :] = img
    
    img_patches = []
    for i in range(w_num+1):
        for j in range(h_num+1):
            crop_patch = img_zeros[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size, :]
            img_patches.append(crop_patch)
    img_patches = np.reshape(img_patches, [-1, patch_size, patch_size, 3])
    return img_patches


for i in os.listdir(os.path.join(train_dir_path, 'images'))[:30]:
    #print(i)
    img_name = i
    label_name = i
    
    img = cv2.imread(os.path.join(train_dir_path, 'images', img_name))
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    label = cv2.imread(os.path.join(train_dir_path, 'labels', label_name))
    
    img_process = my_preprocess(img_RGB)
    img_patches = extract_whole_patches(img_process)
    label_patches = extract_whole_patches(label)
    
    for j in range(len(img_patches)):
        cv2.imwrite('./data/{0}/{1}/train/images/{2}.png'.format(process_method, dataset_name,
                    count), img_patches[j])
        
        cv2.imwrite('./data/{0}/{1}/train/labels/{2}.png'.format(process_method, dataset_name,
                    count), label_patches[j])
        count += 1
        
count = 0
for i in os.listdir(os.path.join(train_dir_path, 'images'))[30:]:
    #print(i)
    img_name = i
    label_name = i
    
    img = cv2.imread(os.path.join(train_dir_path, 'images', img_name))
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    label = cv2.imread(os.path.join(train_dir_path, 'labels', label_name))
    
    img_norm = my_preprocess(img_RGB)
    
    cv2.imwrite('./data/{0}/{1}/test/images/{2}.png'.format(process_method, dataset_name,
                    count), img_norm)
        
    cv2.imwrite('./data/{0}/{1}/test/labels/{2}.png'.format(process_method, dataset_name,
                    count), label)
        
    count += 1