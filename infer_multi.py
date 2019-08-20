#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 12:51:56 2019

@author: robot
"""

import torch
import cv2

import os
import numpy as np
from models.unet import UNet
from models.DUNet import Dense_UNet
from data_loader_multi import MyTestItem

DATASET_NAME = 'DRIVE'
PRE_NAME = 'Deep'
MODEL_NAME = 'Dense_UNet'
ckpt_num = 800

test_dir = './data/{0}/{1}/test/'.format(PRE_NAME, DATASET_NAME)

ckpt_path = './checkpoints/{0}-{1}-{2}-{3}.pt'.format(PRE_NAME, DATASET_NAME, MODEL_NAME, ckpt_num)

model = torch.load(ckpt_path)

dataset_test = MyTestItem(test_dir)
# print(len(dataset_train))
data_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for param in model.parameters():
    param.requires_grad = False

model.cuda()


def restore_img(pred_patches, w_size, h_size):
    w_num, h_num = w_size // 48, h_size // 48
    re_img = np.zeros([(w_num + 1) * 48, (h_num + 1) * 48])
    #print(pred_patches.shape)
    for i in range(w_num + 1):
        for j in range(h_num + 1):
            re_img[i * 48:(i + 1) * 48, j * 48:(j + 1) * 48] = pred_patches[i * (h_num + 1) + j, ...]
    return re_img[:w_size, :h_size]


for i, img_name in data_loader:
    #print(i.shape)
    in_patches = np.reshape(i, [-1, 3, 48, 48])
    in_patches = np.asarray(in_patches, np.float32)
    preds = torch.sigmoid(model(torch.tensor(in_patches).to(device)))
    #print(preds.shape)
    preds = preds.detach().cpu().numpy()[:, 0, :, :]
    #print(preds.shape)
    # (batch_size, 1, size, size) -> (batch_size, size, size)
    img_path = os.path.join(test_dir, 'images', img_name[0])
    img = cv2.imread(img_path)
    # print(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    w_size, h_size = img.shape[0], img.shape[1]
    #print(w_size, h_size)
    # pred_img = restore_img(in_patches[:,0,:,:], w_size, h_size)
    pred_img = restore_img(preds, w_size, h_size)

    if not os.path.exists('./result/{0}/{1}/{2}'.format(MODEL_NAME, PRE_NAME, DATASET_NAME)):
        os.makedirs('./result/{0}/{1}/{2}'.format(MODEL_NAME, PRE_NAME, DATASET_NAME))
    cv2.imwrite('./result/{0}/{1}/{2}/{3}'.format(MODEL_NAME, PRE_NAME, DATASET_NAME, img_name[0]), pred_img * 255)
    # break

