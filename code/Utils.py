import torch
import torch.nn.init
import torch.nn as nn
import cv2
import numpy as np
import random

image_size = 48

# resize image to size 32x32
cv2_scale = lambda x: cv2.resize(x, dsize=(image_size, image_size),
                                 interpolation=cv2.INTER_LINEAR)
# reshape image
np_reshape = lambda x: np.reshape(x, (image_size, image_size, 1))

np_reshape_color = lambda x: np.reshape(x, (image_size, image_size, 3))

centerCrop = lambda x: x[7:55,7:55,:]
#centerCrop = lambda x: x[15:47,15:47,:]

class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x= x / norm.unsqueeze(-1).expand_as(x)
        return x

class L1Norm(nn.Module):
    def __init__(self):
        super(L1Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sum(torch.abs(x), dim = 1) + self.eps
        x= x / norm.expand_as(x)
        return x


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
