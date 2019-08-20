# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 21:02:37 2019

@author: kasy
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def rgb_2_gray(img):
    # img is read by opencv
    # transfered to RGB already
    #print(img.shape)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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

def my_preprocess(img):

    gray = rgb_2_gray(img)
    gray_norm = normalize_contrast(gray)
    gray_clahe = clahe(gray_norm)

    #img_patches_norm.append(gray_clahe)
        
    return gray_clahe


