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

test_path = '../dataset_transfer/STARE/train/images/'
test_list = []

for i in os.listdir(test_path):
    test_list.append(os.path.join(test_path, i))
    
def clahe(gray_img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgs_equalized = clahe.apply(np.array(gray_img, dtype=np.uint8))
    return imgs_equalized       
    
def LUV_contrast_norm(img):
    #BGR, cv2
    #assert img.shape[2] != 3
    L_channel = img[..., 0]
    L_clahe = clahe(L_channel)
    
    new_LUV = np.stack([L_clahe, img[..., 1], img[..., 2]], 2)
    new_RGB = cv2.cvtColor(new_LUV, cv2.COLOR_LUV2RGB)
    
    return new_RGB


fig, axes = plt.subplots(2, 6, figsize=(25,8))

fig.suptitle('Origin rgb and pre-processed rgb in LUV', fontsize=20)
for i in range(6):
    img = cv2.imread(test_list[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_LUV = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        
    img_norm = LUV_contrast_norm(img_LUV)
    
    axes[0, i].imshow(img)
    axes[1, i].imshow(img_norm)
    
axes[0,0].set_ylabel('Origin rgb', size='large')
axes[1,0].set_ylabel('Norm rgb', size='large')

plt.savefig('LUV.png', format='png', dpi=300)
plt.show()
    
