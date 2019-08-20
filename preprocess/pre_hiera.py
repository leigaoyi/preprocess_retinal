#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 20:02:07 2019

@author: robot
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage

def rgb_2_gray(img, mode='green'):
    # img is read by opencv
    # so its channel sequence is B-G-R
    if mode == 'normal':
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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

def my_preprocess(img):
    #img = cv2.imread(img)
    gray = rgb_2_gray(img)
    #gray_norm = normalize_contrast(gray)
    gray_he = histogram_equl(gray)/255.
    #print(gray_he.max())
    gray_gaussian = gaussian(gray_he)*255.
    #print(gray_gaussian.max())
    return gray_gaussian