# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 15:23:18 2019

@author: kasy
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage

test_path = '../dataset_transfer/DRIVE/train/images/'
test_list = []

for i in os.listdir(test_path):
    test_list.append(os.path.join(test_path, i))

def rgb_2_gray(img, mode='green'):
    # img is read by opencv
    # so its channel sequence is B-G-R
    # changed to RGB
    if mode == 'normal':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif mode == 'green':
        gray = img[..., 1]
    return gray

def normalize_contrast(gray_img):
    mean = np.mean(gray_img)
    std = np.std(gray_img)
    
    gray_img = (gray_img - mean)/std
    min_img = np.min(gray_img)
    max_img = np.max(gray_img)
    
    gray_img = (gray_img - min_img)/(max_img-min_img)*255.
    
    return gray_img

def clahe(gray_img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgs_equalized = clahe.apply(np.array(gray_img, dtype=np.uint8))
    return imgs_equalized

def histogram_equl(gray_img):
    gray_equal = cv2.equalizeHist(np.array(gray_img, dtype=np.uint8))
    
    return gray_equal

def gaussian(gray_img):
    gray_gaussian = skimage.filters.gaussian(gray_img, sigma=0.4)
    return gray_gaussian

#img = cv2.imread(test_list[12])
#
#gray = rgb_2_gray(img)
#gray_norm = normalize_contrast(gray)
#gray_clahe = clahe(gray_norm)

#plt.imshow(gray_clahe, cmap='gray')
#plt.show()
    
fig, axes = plt.subplots(3, 6, figsize=(25,8))

fig.suptitle('Origin gray and pre-processed gray in DUNet', fontsize=20)
for i in range(6):
    img = cv2.imread(test_list[i])
    gray = rgb_2_gray(img)
    #gray_norm = normalize_contrast(gray)
    gray_he = histogram_equl(gray)
    gray_gaussian = gaussian(gray_he)
    axes[0, i].imshow(gray, cmap='gray')
    axes[1, i].imshow(gray_he, cmap='gray')
    axes[2, i].imshow(gray_gaussian, cmap='gray')
axes[0, 0].set_ylabel('Gray img', size='large')    
axes[1, 0].set_ylabel('Histo equal', size='large')
axes[2, 0].set_ylabel('Gaussian filter', size='large')
plt.savefig('hierarchical.png', format='png', dpi=300)
plt.show()
#plt.clf()


