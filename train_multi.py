# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 21:42:19 2019

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
from data_loader_multi import MyImageItem

DATASET_NAME = 'DRIVE'
PRE_NAME = 'Multi'

MODEL_NAME = 'Dense_UNet'

train_dir = './data/{0}/{1}/train/'.format(PRE_NAME, DATASET_NAME)

# model_unet = UNet(1)
model_unet = Dense_UNet(3)

dataset_train = MyImageItem(train_dir)
# print(len(dataset_train))
data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=20, shuffle=True)

# model_unet
# device = torch.device('cuda:0')
device = torch.device("cuda:1")
model_unet.to(device)

for param in model_unet.parameters():
    param.requires_grad = True

params = [p for p in model_unet.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=5,
                                               gamma=0.1)

num_epoch = 1000
criterion = MixedLoss(10.0, 2.0)
ckpt_path = './checkpoints/'

if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)

for i in range(num_epoch):
    model_unet.train()
    for imgs, targets in data_loader:
        imgs = np.array(imgs).astype(np.float32)
        imgs = np.reshape(imgs, [-1, 3, 48, 48])
        # normalize for multi-scale, multi-level
        mean = np.mean(imgs)
        std = np.std(imgs)
        imgs = (imgs-mean)/std

        targets = np.reshape(targets, [-1, 1, 48, 48])

        # visualize the imgs
        # cv2.imwrite('vi_1.png', imgs[0][0]*255)
        # cv2.imwrite('vi_2.png', imgs[10][0] * 255)
        imgs = torch.tensor(imgs).to(device)
        targets = torch.tensor(np.asarray(targets, np.float32)).to(device)

        outputs = model_unet(imgs)

        losses = criterion(outputs, targets)
        dice_score = dice_loss(outputs, targets).item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    # break
    if (i + 1) % 20 == 0:
        print('Epoch {0} loss {1:.4f}'.format(i + 1, losses.item()), ' dice score {0:.4f}'.format(dice_score))
    if (i + 1) % 200 == 0:
        torch.save(model_unet, ckpt_path + '{0}-{1}-{2}-{3}.pt'.format(PRE_NAME, DATASET_NAME, MODEL_NAME, i + 1))
        # break
print('Train Over!')
