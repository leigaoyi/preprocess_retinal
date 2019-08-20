import torch
import torch.nn as nn
import torchvision

class DenseBlock(nn.Module):
    def __init__(self, input_shape, outdim):
        super(DenseBlock, self).__init__()
        
        in_shape = input_shape
        self.outdim = outdim
        self.bn0 = nn.BatchNorm2d(in_shape)
        self.relu0 = nn.ReLU()
        
        self.conv1 = nn.Conv2d(in_shape, outdim, kernel_size=3, 
                               stride=1, padding=1)        
        self.bn1 = nn.BatchNorm2d(outdim*2)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(2*outdim, outdim, 3, 1, 1)
        self.relu2 = nn.ReLU()
        self.shortCNN = nn.Conv2d(in_shape, outdim, 3, 1, 1)
        #self.bn2 = nn.BatchNorm2d()
        
    def forward(self, x):
        pre_norm = self.relu0(self.bn0(x))
        conv1 = self.conv1(pre_norm)
        
        in_shape = x.shape[1]
        if in_shape != self.outdim:
            shortcut = self.shortCNN(x)
        else:
            shortcut = x
        
        result1 = torch.cat([conv1, shortcut], 1)
        
        bn1 = self.bn1(result1)
        relu1 = self.relu1(bn1)
        
        conv2 = self.conv2(relu1)
        result2 = torch.cat([conv2, result1, shortcut], 1)
        return self.relu2(result2)
    
class Dense_UNet(nn.Module):
    def __init__(self, in_shape):
        super(Dense_UNet, self).__init__()
        self.conv0 = nn.Conv2d(in_shape, 32, 3, 1, 1)
        self.bn0 = nn.BatchNorm2d(32)
        self.relu0 = nn.ReLU()
        
        self.Dense1 = DenseBlock(32, 32)#32*4
        self.max_pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.Dense2 = DenseBlock(32*4, 64)#64*4
        self.max_pool2 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.Dense3 = DenseBlock(64*4, 64)#64*4
        self.max_pool3 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.Dense4 = DenseBlock(64*4, 64)
        
        self.up1 = nn.ConvTranspose2d(64*4, 64, 3, stride=2, padding=1,
                                     output_padding=1)#cat1, conv3
        
        self.Dense5 = DenseBlock(64*4+64, 64)
        self.up2 = nn.ConvTranspose2d(64*4, 64, 3, stride=2, padding=1,
                                     output_padding=1)#cat2, conv2
        
        self.Dense6 = DenseBlock(64*4+64, 64)
        self.up3 = nn.ConvTranspose2d(64*4, 32, 3, stride=2, padding=1,
                                     output_padding=1)#cat3, conv1
        
        self.Dense7 = DenseBlock(32*4+32, 32)
        self.conv8 = nn.Conv2d(32*4, 1, kernel_size=1)
        self.relu8 = nn.ReLU()
        
    def forward(self, x):
        conv0 = self.conv0(x)
        conv0 = self.bn0(conv0)
        conv0 = self.relu0(conv0)
        
        conv1 = self.Dense1(conv0)
        pool1 = self.max_pool1(conv1)
        
        conv2 = self.Dense2(pool1)
        pool2 = self.max_pool2(conv2)
        
        conv3 = self.Dense3(pool2)
        pool3 = self.max_pool3(conv3)
        
        conv4 = self.Dense4(pool3)
        
        up1 = self.up1(conv4)
        cat1 = torch.cat([up1, conv3], 1)
        
        conv5 = self.Dense5(cat1)
        up2 = self.up2(conv5)
        
        cat2 = torch.cat([up2, conv2], 1)
        conv6 = self.Dense6(cat2)
        
        up3 = self.up3(conv6)
        cat3 = torch.cat([up3, conv1], 1)
        
        conv7 = self.Dense7(cat3)
        conv8 = self.conv8(conv7)
        
        return conv8