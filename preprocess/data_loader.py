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
from .pre_anotation import my_preprocess

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
        assert self.train_imgs_basename != self.train_labels_basename
        
        num_patches = 20
        patch_size = 48
        
        img_name = self.train_imgs_basename[item]
        label_name = self.train_labels_basename[item]

        img = cv2.imread(os.path.join(self.train_dir,'images',img_name))
        #print(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)# transfer to RGB
        label = cv2.imread(os.path.join(self.train_dir,'labels',label_name))[..., 0]
        #print(label)

        img_norm = my_preprocess(img)
        img_norm = np.asarray(img_norm)

        #img_patches, label_patches = extract_random_patches(img_norm, label, num_patches, patch_size)
        img_patches = extract_whole_patches(img_norm)
        label_patches = extract_whole_patches(label)

        img_patches_norm = img_patches/255.
        label_patches[label_patches>0] = 1
        
        img_patches_norm = np.reshape(img_patches_norm, [-1, patch_size, patch_size])
       # print('***********')
        label_patches = np.reshape(label_patches, [-1, patch_size, patch_size])

        return img_patches_norm, label_patches

def extract_random_patches(img, label, num_patches=20, patch_size=48):
    # extract num_pathches from one image
    w_size, h_size = img.shape[0], img.shape[1]
    img_patches = []
    label_patches = []
    for i in range(num_patches):
        x_loc = np.random.randint(0, w_size - patch_size) 
        y_loc = np.random.randint(0, h_size - patch_size)
        
        img_patches.append(img[x_loc:(x_loc+patch_size), y_loc:(y_loc+patch_size)])
        label_patches.append(label[x_loc:(x_loc+patch_size), y_loc:(y_loc+patch_size)])
        
    img_patches = np.reshape(img_patches, [num_patches, patch_size, patch_size])
    #print(label_patches[0].shape)
    label_patches = np.reshape(label_patches, [num_patches, patch_size, patch_size])
    
    return img_patches, label_patches

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
        
        img = cv2.imread(os.path.join(self.test_dir, 'images', img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#transfer the RGB
        #print(img.shape)

        img_norm = my_preprocess(img)
        img_patches = extract_whole_patches(img_norm)

        img_patches = np.asarray(img_patches, np.float32)
        img_patches_norm = img_patches / 255.

        return img_patches_norm, img_name

        
        
    

#dataset_train = MyImageItem(train_dir_path)
#dataloader = DataLoader(dataset_train, batch_size=4, shuffle=True)
#print(os.path.exists(train_dir_path))
#print(os.listdir('../'))
