#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 20:19:35 2019

@author: robot
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage

    
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

def my_preprocess(img):
    img_norm = rgb_contrast_norm(img)
    img_BGR = cv2.cvtColor(img_norm, cv2.COLOR_RGB2BGR)
    return img_BGR