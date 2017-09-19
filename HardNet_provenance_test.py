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
from Losses import loss_margin_min
from Loggers import Logger, FileLogger
from Utils import L2Norm, cv2_scale, np_reshape, centerCrop
from Utils import str2bool

from scipy.spatial import distance
import image_processing

# Training settings
parser = argparse.ArgumentParser(description='PyTorch HardNet')
# Model options

parser.add_argument('--w1bsroot', type=str,
                    default='wxbs-descriptors-benchmark/code',
                    help='path to dataset')
parser.add_argument('--dataroot', type=str,
                    default='datasets/',
                    help='path to dataset')
parser.add_argument('--test-set', default= 'MSCOCO_synthesized',
                    help='Other options: notredame, yosemite')
parser.add_argument('--num-workers', default= 8,
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
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=10, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--anchorswap', type=bool, default=True,
                    help='turns on anchor swap')
parser.add_argument('--batch-size', type=int, default=128, metavar='BS',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='BST',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--n-triplets', type=int, default=5000000, metavar='N',
                    help='how many triplets will generate from the dataset')
parser.add_argument('--margin', type=float, default=1.0, metavar='MARGIN',
                    help='the margin value for the triplet loss function (default: 1.0')
parser.add_argument('--act-decay', type=float, default=0,
                    help='activity L2 decay, default 0')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--fliprot', type=str2bool, default=False,
                    help='turns on flip and 90deg rotation augmentation')
parser.add_argument('--lr-decay', default=1e-6, type=float, metavar='LRD',
                    help='learning rate decay ratio (default: 1e-6')
parser.add_argument('--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--optimizer', default='sgd', type=str,
                    metavar='OPT', help='The optimizer to use (default: SGD)')
# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='1', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')

args = parser.parse_args()
# check if path to w1bs dataset testing module exists
if os.path.isdir(args.w1bsroot):
    sys.path.insert(0, args.w1bsroot)
    import utils.w1bs as w1bs
    TEST_ON_W1BS = True

# set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(args.seed)

class testDataset(data.Dataset):
    """From the PhotoTour Dataset it generates triplet samples
    note: a triplet is composed by a pair of matching images and one of
    different class.
    """
    def __init__(self, data, train=True, transform=None, batch_size = None, *arg, **kw):
        self.transform = transform
        self.train = train
        self.n_triplets = args.n_triplets
        self.batch_size = batch_size
        self.data = data

        self.data_size = self.data.numpy().shape[0]
        #print(self.data_size)
        #if not self.train:
            #print(self.data[199].numpy().shape)
            #print(self.data[199].numpy()[:5,30,:])
        
    def __getitem__(self,index):
        def transform_img(img):
            if self.transform is not None:
                img = self.transform(img.numpy())
            return img
        #if index==0:
            #print(self.data[199].numpy().shape)
            #print(self.data[199].numpy()[:5,30,:])

        img1 = transform_img(self.data[index])

        #if index==199:
            #print(img1.numpy().shape)
            #print(img1.numpy()[:,:5,15])

        return img1

    def __len__(self):
        return self.data_size


class TNet(nn.Module):
    """TFeat model definition
    """
    def __init__(self):
        super(TNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2,padding=1),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(128, 128, kernel_size=8),
            nn.BatchNorm2d(128, affine=False),
            #nn.Conv2d(1, 32, kernel_size=7),
            ##nn.BatchNorm2d(32, affine=False),
            #nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2),
            #nn.Conv2d(32, 64, kernel_size=6),
            ##nn.BatchNorm2d(64, affine=False),
            #nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2),
            #nn.Conv2d(64, 128, kernel_size=5),
            ##nn.BatchNorm2d(128, affine=False),
            #nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2),
            #nn.Conv2d(128, 128, kernel_size=4),
            ##nn.BatchNorm2d(128, affine=False),
        )
        self.features.apply(weights_init)

    def forward(self, input):
        x_features = self.features(input)
        x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal(m.weight.data, gain=0.7)
        nn.init.constant(m.bias.data, 0.01)
    if isinstance(m, nn.Linear):
        nn.init.orthogonal(m.weight.data, gain=0.01)
        nn.init.constant(m.bias.data, 0.)


def main(model):
    # print the experiment configuration
    model.eval()
    root_dir = '/home/xuzhang/project/Medifor/data/' + args.test_set + '/'
    world_list = image_processing.load_csv(root_dir + 'indexes/' + args.test_set + '-provenancefiltering-world.csv','|')
    world_file_list = world_list["WorldFileName"]
    world_id_list = world_list["WorldFileID"]
    subset = 'world'
    kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory} if args.cuda else {}
    
    if args.cuda:
        model.cuda()

    transform = transforms.Compose([
        transforms.Lambda(centerCrop),
        #transforms.Lambda(cv2_scale),
        transforms.Lambda(np_reshape),
        transforms.ToTensor(),
        #transforms.Normalize((args.mean_image,), (args.std_image,))
        transforms.Normalize((args.mean_image,args.mean_image,args.mean_image),
                (args.std_image,args.std_image,args.std_image))
        ])

    test_batch_size = args.test_batch_size
    descriptor_dim = 128
    detector_name = 'DOG'

    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print('=> no checkpoint found at {}'.format(args.resume))
            
    for filename,fileid in tqdm(zip(world_file_list,world_id_list)):
        meta = np.array([])
        features = np.array([])
        feature_file = root_dir+'/'+detector_name+'_prov_desc_'+subset+'/'+fileid+'.npz'
        
        #if fileid != 'COCO_train2014_000000258035':
            #continue
            
        try:
            gray_image, ratio = image_processing.read_image_from_name(root_dir,filename)
            color_image, ratio = image_processing.read_color_image_from_name(root_dir,filename)
        except:
            print(filename)
            np.savez(feature_file, meta, features)

        kp_list, sift_desc = image_processing.extract_sift(gray_image)
        patches_list = []
        
        for kp_idx, kp in enumerate(kp_list):
            tmp_patch = image_processing.extract_patch(color_image, kp)
            #tmp_patch = transform(tmp_patch)
            #if fileid == 'COCO_train2014_000000258035' and kp_idx == 199:
            #    print(tmp_patch.numpy().shape)
            #    print(tmp_patch.numpy()[:,:10,30])
            patches_list.append(tmp_patch)
        #print(tmp_patch.numpy().shape)

        #if fileid == 'COCO_train2014_000000258035':
            #print(patches_list[199][:5,30,:])

        patches_list = np.array(patches_list)
        #print(patches_list.dtype)
        patches_list = torch.ByteTensor(patches_list)
        #patches_list = transform(patches_list)
        
        test_loader = torch.utils.data.DataLoader(
            testDataset(train=False,
                        data = patches_list,
                        batch_size=args.test_batch_size,
                        root=args.dataroot,
                        name=args.test_set,
                        download=False,
                        transform=transform),
                        batch_size=args.batch_size,
                        shuffle=False, **kwargs)

        patch_number = len(kp_list)
        #print(patch_number)
        if patch_number == 0:
            np.savez(feature_file, meta=meta, features = features)
            continue
         
        offset = 0
        meta = np.zeros((patch_number,4))
        features = []
        #pbar = tqdm(enumerate())
        for data_a in test_loader:
        ##if batch_idx == 0:
        #    #print(data_a.numpy().shape)
        #    #print(data_a[0,:,:10,30])
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
            feature_file = feature_file

        #if fileid == 'COCO_train2014_000000258035':
            #print(features[199,:10])
            #break
            
        np.savez(feature_file, meta=meta, features = features)


if __name__ == '__main__':

    model = TNet()

    main(model)
