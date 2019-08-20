# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 08:06:13 2019

@author: kasy
"""


import torch
import numpy as np
import os
import torch.utils.data
import cv2
from models.unet import UNet
from models.DUNet import Dense_UNet
from models.losses import MixedLoss
from models.losses import dice_loss
from preprocess import MyImageItem
import glob

train_dir = './dataset_transfer/DRIVE/train/'

sample_img_path = glob.glob(os.path.join(train_dir, 'images','*.png'))
sample_img = cv2.imread(sample_img_path[0])
print(sample_img.shape)


def restore_img(pred_patches, w_size=584, h_size=565):
    # shape of pred_patches [bs, 48, 48]
    #print(pred_patches.shape)
    w_num, h_num = w_size // 48, h_size // 48
    re_img = np.zeros([(w_num+1)*48, (h_num+1)*48])

    for i in range(w_num+1):
        for j in range(h_num+1):
            re_img[i*48:(i+1)*48, j*48:(j+1)*48] = pred_patches[i*(h_num+1)+j]
    return re_img[:w_size, :h_size]

#model_unet = UNet(1)
#model_unet = Dense_UNet(1)
#model_unet.cuda()

dataset_train = MyImageItem(train_dir)
#print(len(dataset_train))
data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True)

#model_unet
#device = torch.device('cuda:0')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#for param in model_unet.parameters():
#    param.requires_grad = True
#    

num_epoch = 1
criterion = MixedLoss(10.0, 2.0)
ckpt_path = './checkpoints/'

if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
      
count_img = 0


    
for i in range(num_epoch):
    #model_unet.train()
    for imgs, targets in data_loader:
        imgs = np.array(imgs).astype(np.float32)
        imgs = np.reshape(imgs, [-1, 1, 48, 48])
        targets = np.reshape(targets, [-1, 1, 48, 48])
        
        re_img = restore_img(imgs[:,0,:,:])*255
        re_label = restore_img(targets[:,0,:,:])*255
        
        cv2.imwrite('./result/visual/{}_img.png'.format(count_img), re_img)
        cv2.imwrite('./result/visual/{}_label.png'.format(count_img), re_label)
        
        count_img += 1