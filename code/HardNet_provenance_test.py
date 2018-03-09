#!/usr/bin/python2 -utt
#-*- coding: utf-8 -*-
"""
This is HardNet local patch descriptor. The training code is based on PyTorch TFeat implementation
https://github.com/edgarriba/examples/tree/master/triplet
by Edgar Riba.

If you use this code, please cite 
@article{HardNet2017,
 author = {Anastasiya Mishchuk, Dmytro Mishkin, Filip Radenovic, Jiri Matas},
    title = "{Working hard to know your neighbor's margins:Local descriptor learning loss}",
     year = 2017}
(c) 2017 by Anastasiia Mishchuk, Dmytro Mishkin 
"""

from __future__ import division, print_function
import sys
from copy import deepcopy
import argparse
import torch
import torch.nn.init
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
import numpy as np
import random
import cv2
import copy
import synthesized_journal
from EvalMetrics import ErrorRateAt95Recall
from Utils import L2Norm, cv2_scale, np_reshape_color, centerCrop
from Utils import str2bool

from scipy.spatial import distance
import image_processing

# Training settings
parser = argparse.ArgumentParser(description='PyTorch HardNet')
# Model options

parser.add_argument('--dataroot', type=str,
                    default='../datasets/',
                    help='path to dataset')
parser.add_argument('--model-dir', default='../provenance_model/',
                    help='folder to output model checkpoints')
parser.add_argument('--suffix', default='synthesized_journals_2_train_min_triplet_margin_as_da_do',
                    help='folder to output model checkpoints')
parser.add_argument('--test-set', default= 'MSCOCO_synthesized',
                    help='Other options: notredame, yosemite')
parser.add_argument('--num-workers', default= 1,
                    help='Number of workers to be created')
parser.add_argument('--pin-memory',type=bool, default= True,
                    help='')
parser.add_argument('--anchorave', type=bool, default=False,
                    help='anchorave')
parser.add_argument('--imageSize', type=int, default=32,
                    help='the height / width of the input image to network')
parser.add_argument('--mean-image', type=float, default=0.443728476019,
                    help='mean of train dataset for normalization')
parser.add_argument('--std-image', type=float, default=0.20197947209,
                    help='std of train dataset for normalization')
parser.add_argument('--start-epoch', default=9, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='BST',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

suffix = args.suffix 
try:
    args.resume = '{}{}/checkpoint_{}.pth'.format(args.model_dir,suffix,args.start_epoch)
except:
    args.resume = '{}{}/checkpoint_12.pth'.format(args.model_dir,suffix)
# set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

args.cuda = not args.no_cuda and torch.cuda.is_available()
        
if args.cuda:
    cudnn.benchmark = True

class testDataset(data.Dataset):
    """From the PhotoTour Dataset it generates triplet samples
    note: a triplet is composed by a pair of matching images and one of
    different class.
    """
    def __init__(self, data, train=True, transform=None, batch_size = None, *arg, **kw):
        self.transform = transform
        self.train = train
        self.batch_size = batch_size
        self.data = data
        self.data_size = self.data.numpy().shape[0]
        
    def __getitem__(self,index):
        def transform_img(img):
            if self.transform is not None:
                img = self.transform(img.numpy())
            return img
        img1 = transform_img(self.data[index])
        img1 = torch.from_numpy(deepcopy(img1.numpy()[:,7:39,7:39]))
        return img1

    def __len__(self):
        return self.data_size


class HardNet(nn.Module):
    def __init__(self):
        super(HardNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2,padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(128, 128, kernel_size=8, bias = False),
            nn.BatchNorm2d(128, affine=False),
        )

    def forward(self, input):
        x_features = self.features(input)
        x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)


def main(model):
    # print the experiment configuration
    model.eval()
    root_dir = '/home/xuzhang/project/Medifor/data/' + args.test_set + '/'
    index_dir = '/home/xuzhang/project/Medifor/data/' + args.test_set + '/'
    save_dir = '/home/xuzhang/project/Medifor/code/provenance-filtering/data/' + args.test_set + '/'
    if 'MFC18' in args.test_set or 'NC2017' in args.test_set:
        ndataset_list = args.test_set.split('_')
        dbindex_name = ndataset_list[0]+'_'+ndataset_list[1]
    else:
        dbindex_name = args.test_set

    world_list = image_processing.load_csv(root_dir + 'indexes/' + dbindex_name + '-provenancefiltering-world.csv','|')
    world_file_list = world_list["WorldFileName"]
    world_id_list = world_list["WorldFileID"]
    subset = 'world'
    kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory} if args.cuda else {}

    if args.cuda:
        model.cuda()

    transform = transforms.Compose([
            #transforms.Lambda(cv2_scale),
            transforms.Lambda(centerCrop),
            transforms.Lambda(np_reshape_color),
            transforms.ToTensor(),
            transforms.Normalize((args.mean_image,args.mean_image,args.mean_image),
                (args.std_image,args.std_image,args.std_image))])

    test_batch_size = args.test_batch_size
    descriptor_dim = 128
    detector_name = 'DOG'
    
    save_dir = save_dir + '/' + detector_name + '_prov_desc_{}_{}/'.format(args.suffix,subset)
    try:
        os.stat(save_dir) 
    except:
        os.makedirs(save_dir)

    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print('=> no checkpoint found at {}'.format(args.resume))
            
    for idx,(filename,fileid) in tqdm(enumerate(zip(world_file_list,world_id_list))):
        meta = np.array([])
        features = np.array([])
        feature_file = save_dir+fileid+'.npz'
        
        try:
            gray_image, ratio = image_processing.read_image_from_name(root_dir,filename)
            color_image, ratio = image_processing.read_color_image_from_name(root_dir,filename)
        except:
            print(filename)
            np.savez(feature_file, meta, features)
        
        kp_list, _ = image_processing.extract_sift(gray_image)

        patch_number = len(kp_list)
        if patch_number == 0:
            np.savez(feature_file, meta=meta, features = features)
            continue

        patches_list = []
        
        for kp_idx, kp in enumerate(kp_list):
            tmp_patch = image_processing.extract_patch(color_image, kp)
            patches_list.append(tmp_patch)

        patches_list = np.array(patches_list)
        patches_list = torch.ByteTensor(patches_list)
        
        test_loader = torch.utils.data.DataLoader(
            testDataset(train=False,
                        data = patches_list,
                        batch_size=args.test_batch_size,
                        root=args.dataroot,
                        name=args.test_set,
                        download=False,
                        transform=transform),
                        batch_size=args.test_batch_size,
                        shuffle=False, **kwargs)

        patch_number = len(kp_list)
        if patch_number == 0:
            np.savez(feature_file, meta=meta, features = features)
            continue
         
        offset = 0
        meta = kp_list#np.zeros((patch_number,4))
        features = []
        for data_a in test_loader:
            if args.cuda:
                data_a = data_a.cuda()

            data_a = Variable(data_a, volatile=True)
            out_a = model(data_a)
            features.append(out_a.data.cpu().numpy())
        
        features = np.vstack(features).reshape(patch_number,descriptor_dim)
        try:
            os.stat(feature_file)
            os.remove(feature_file)
        except:
            pass
        np.savez(feature_file, meta=meta, features = features)


if __name__ == '__main__':

    model = HardNet()
    main(model)
