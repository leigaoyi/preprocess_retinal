# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 17:03:07 2019

@author: kasy
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
import copy

test_path = '../dataset_transfer/DRIVE/train/images/'
test_list = []

for i in os.listdir(test_path):
    test_list.append(os.path.join(test_path, i))

def clahe(gray_img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgs_equalized = clahe.apply(np.array(gray_img, dtype=np.uint8))
    return imgs_equalized    


def adjust_gamma(img, gamma=1.0):
    #assert (len(imgs.shape) == 4)  # 4D arrays
    # assert (imgs.shape[1]==1)  #check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(img.shape)
    #for i in range(imgs.shape[0]):
    new_imgs = cv2.LUT(np.array(img, dtype=np.uint8), table)
    return new_imgs

def pre_process(rgb_img):
    green_channel = copy.deepcopy(rgb_img[..., 1])
    green_clahe = clahe(green_channel)
    
    green_gamma = adjust_gamma(green_channel)
    # if multiply 0.5, the shown image would destroy
    rgb_norm = np.stack([green_channel, green_clahe, green_gamma], axis=2)
    
    return rgb_norm


fig, axes = plt.subplots(2, 6, figsize=(25,8))

fig.suptitle('Origin rgb and pre-processed rgb in Multi', fontsize=20)
for i in range(6):
    img = cv2.imread(test_list[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #axes[0, i].imshow(img)
    img_norm = pre_process(img)
    img_test = 0.5*img_norm
    cv2.imwrite('test.png', img_test)
    axes[0, i].imshow(img)
    axes[1, i].imshow(img_norm)
    
axes[0,0].set_ylabel('Origin rgb', size='large')
axes[1,0].set_ylabel('Norm rgb', size='large')

plt.savefig('MultiLevel.png', format='png', dpi=300)
plt.show()
    
