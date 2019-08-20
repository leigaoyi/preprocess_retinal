import torch
import os
import cv2
import numpy as np
import torch.utils.data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# process single channel, like gray image
#from .pre_DUNet import my_preprocess
#from .pre_hiera import my_preprocess
#from .pre_anotation import my_preprocess

#train_dir_path = '../dataset_transfer/DRIVE/train/'
#test_dir_path = '../dataset_transfer/DRIVE/test'

class MyImageItem(torch.utils.data.Dataset):
    def __init__(self, train_dir_path):
        self.train_dir = train_dir_path
        self.train_imgs_basename = os.listdir(os.path.join(self.train_dir, 'images'))
        self.train_labels_basename = os.listdir(os.path.join(self.train_dir, 'labels'))

    def __len__(self):
        return len(self.train_imgs_basename)

    def __getitem__(self, item):
        #assert self.train_imgs_basename != self.train_labels_basename
        
        num_patches = 20
        patch_size = 48
        
        img_name = self.train_imgs_basename[item]
        label_name = self.train_imgs_basename[item]

        img = cv2.imread(os.path.join(self.train_dir,'images',img_name))[..., 0]
        #print(img_name)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)# transfer to RGB
        label = cv2.imread(os.path.join(self.train_dir,'labels',label_name))[..., 0]
        
        img_norm = img/255

        label[label>0] = 1.0

        return img_norm, label

def extract_whole_patches(img, patch_size=48):
    w_size, h_size = img.shape[0], img.shape[1]
    w_num, h_num = w_size//patch_size, h_size//patch_size
    
    img_zeros = np.zeros([(w_num+1)*patch_size, (h_num+1)*patch_size])
    img_zeros[:w_size, :h_size] = img
    
    img_patches = []
    for i in range(w_num+1):
        for j in range(h_num+1):
            crop_patch = img_zeros[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            img_patches.append(crop_patch)
    img_patches = np.reshape(img_patches, [-1, patch_size, patch_size])
    return img_patches
    

class MyTestItem(torch.utils.data.Dataset):    
    def __init__(self, test_dir_path):
        self.test_dir = test_dir_path
        self.test_imgs_basename = os.listdir(os.path.join(self.test_dir, 'images'))
        #self.test_labels_basename = os.listdir(os.path.join(self.train_dir, 'labels'))

    def __len__(self):
        return len(self.test_imgs_basename)

    def __getitem__(self, item):
        #assert self.train_imgs_basename != self.train_labels_basename
        img_name = self.test_imgs_basename[item]
        
        img = cv2.imread(os.path.join(self.test_dir, 'images', img_name))[..., 0]
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#transfer the RGB
        #print(img.shape)

        img_norm = img
        img_patches = extract_whole_patches(img_norm)

        img_patches = np.asarray(img_patches, np.float32)
        img_patches_norm = img_patches / 255.

        return img_patches_norm, img_name

        
        
    

#dataset_train = MyImageItem(train_dir_path)
#dataloader = DataLoader(dataset_train, batch_size=4, shuffle=True)
#print(os.path.exists(train_dir_path))
#print(os.listdir('../'))
