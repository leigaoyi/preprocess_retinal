# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 10:08:19 2019

@author: kasy
"""

import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channel):
        super(UNet, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channel, 32, 3, 1, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, 3, 1, padding=1)
        
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.conv2_1 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, 3, 1, padding=1)
        
        self.pool2 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.conv3_1 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, 3, 1, padding=1)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1,
                                      output_padding=1)
        
        self.conv4_1 = nn.Conv2d(128, 64, 3, 1, padding=1)
        self.conv4_2 = nn.Conv2d(64, 64, 3, 1, padding=1)
        
        self.up2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1,
                                      output_padding=1)
        
        self.conv5_1 = nn.Conv2d(64, 32, 3, 1, padding=1)
        self.conv5_2 = nn.Conv2d(32, 32, 3, 1, padding=1)
        
        self.conv6 = nn.Conv2d(32, 1, 1, 1)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        conv1_1 = self.relu(self.conv1_1(x))
        conv1_2 = self.relu(self.conv1_2(conv1_1))
        
        pool1 = self.pool1(conv1_2)
        
        conv2_1 = self.relu(self.conv2_1(pool1))
        conv2_2 = self.relu(self.conv2_2(conv2_1))
        
        pool2 = self.pool2(conv2_2)
        
        conv3_1 = self.relu(self.conv3_1(pool2))
        conv3_2 = self.relu(self.conv3_2(conv3_1))
        
        up1 = self.relu(self.up1(conv3_2))
        cat1 = torch.cat([up1, conv2_2], 1)
        
        conv4_1 = self.relu(self.conv4_1(cat1))
        conv4_2 = self.relu(self.conv4_2(conv4_1))
        
        up2 = self.relu(self.up2(conv4_2))
        cat2 = torch.cat([up2, conv1_2], 1)
        
        conv5_1 = self.relu(self.conv5_1(cat2))
        conv5_2 = self.relu(self.conv5_2(conv5_1))
        
        conv6 = self.conv6(conv5_2)
        
        return conv6

if __name__ == '__main__':
    x = torch.ones([4,1, 96, 96])
    y = UNet(1)(x)
    print(y.shape)