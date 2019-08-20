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

test_path = '../dataset_transfer/DRIVE/train/images/'
test_list = []

for i in os.listdir(test_path):
    test_list.append(os.path.join(test_path, i))
    
def rgb_contrast_norm(img):
    #BGR, cv2
    #assert img.shape[2] != 3
    img_contrast = np.zeros_like(img)
    for i in range(3):
        fig = img[..., i]
        mean = np.mean(fig)
        std = np.std(fig)
        fig = (fig-mean)/std
        max_fig = np.max(fig)
        min_fig = np.min(fig)
        
        fig = (fig-min_fig)/(max_fig-min_fig)*255.
        
        img_contrast[..., i] = fig
    return img_contrast


fig, axes = plt.subplots(2, 6, figsize=(25,8))

fig.suptitle('Origin rgb and pre-processed rgb in DeepRetinal', fontsize=20)
for i in range(6):
    img = cv2.imread(test_list[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_norm = rgb_contrast_norm(img)
    
    axes[0, i].imshow(img)
    axes[1, i].imshow(img_norm)
    
axes[0,0].set_ylabel('Origin rgb', size='large')
axes[1,0].set_ylabel('Norm rgb', size='large')

plt.savefig('DeepRetinal.png', format='png', dpi=300)
plt.show()
    
