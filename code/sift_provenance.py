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
import torch.optim as optim
import torchvision.datasets as dset
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
from Losses import loss_HardNet, loss_random_sampling, loss_L2Net, global_orthogonal_regularization
from W1BS import w1bs_extract_descs_and_save
from Utils import L2Norm, cv2_scale, np_reshape_color, centerCrop
from Utils import str2bool
import torch.nn as nn
import torch.nn.functional as F
import pdb
from Loggers import Logger, FileLogger

class CorrelationPenaltyLoss(nn.Module):
    def __init__(self):
        super(CorrelationPenaltyLoss, self).__init__()

    def forward(self, input):
        mean1 = torch.mean(input, dim=0)
        zeroed = input - mean1.expand_as(input)
        cor_mat = torch.bmm(torch.t(zeroed).unsqueeze(0), zeroed.unsqueeze(0)).squeeze(0)
        d = torch.diag(torch.diag(cor_mat))
        no_diag = cor_mat - d
        d_sq = no_diag * no_diag
        return torch.sqrt(d_sq.sum())/input.size(0)

# Training settings
parser = argparse.ArgumentParser(description='PyTorch HardNet')
# Model options

parser.add_argument('--dataroot', type=str,
                    default='../datasets/',
                    help='path to dataset')
parser.add_argument('--enable-logging',type=bool, default=False,
                    help='output to tensorlogger')
parser.add_argument('--log-dir', default='../provenance_log/',
                    help='folder to output log')
parser.add_argument('--model-dir', default='../provenance_model/',
                    help='folder to output model checkpoints')
parser.add_argument('--training-set', default= 'sift',
                    help='Other options: notredame, yosemite')
parser.add_argument('--imageSize', type=int, default=32,
                    help='the height / width of the input image to network')
parser.add_argument('--mean-image', type=float, default=0.443728476019,
                    help='mean of train dataset for normalization')
parser.add_argument('--std-image', type=float, default=0.20197947209,
                    help='std of train dataset for normalization')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=10, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--anchorswap', type=bool, default=True,
                    help='turns on anchor swap')
parser.add_argument('--batch-size', type=int, default=1024, metavar='BS',
                    help='input batch size for training (default: 1024)')
parser.add_argument('--test-batch-size', type=int, default=1024, metavar='BST',
                    help='input batch size for testing (default: 1024)')
parser.add_argument('--n-triplets', type=int, default=5000000, metavar='N',
                    help='how many triplets will generate from the dataset')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=10, metavar='LI',
                    help='how many batches to wait before logging training status')
parser.add_argument('--no-mask', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--bg', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--inner-product', action='store_true', default=False,
                    help='enables CUDA training')

args = parser.parse_args()

dataset_names = ['NC2017_Dev1_Beta4', 'NC2017_Dev2_Beta1',\
        'MFC18_Dev1_Ver2', 'MFC18_Dev2_Ver1', 'synthesized_journals_test_direct']

suffix = '{}'.format(args.training_set)

# create loggin directory
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

# set random seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)

class TripletPhotoTour(synthesized_journal.synthesized_journal):
    """
    From the PhotoTour Dataset it generates triplet samples
    note: a triplet is composed by a pair of matching images and one of
    different class.
    """
    def __init__(self, train=True, transform=None, batch_size = None, load_random_triplets = False,  *arg, **kw):
        super(TripletPhotoTour, self).__init__(*arg, **kw)
        self.transform = transform
        self.out_triplets = load_random_triplets
        self.n_triplets = args.n_triplets
        self.batch_size = batch_size

    def __getitem__(self, index):
        def transform_img(img):
            if self.transform is not None:
                img = self.transform(img.numpy())
            return img

        m = self.matches[index]
        descriptor1 = self.descriptor[m[0]]
        descriptor2 = self.descriptor[m[1]]
        return descriptor1, descriptor2, m[2]
        
    def __len__(self):
        return self.matches.size(0)

def create_loaders(load_random_triplets = False):

    test_dataset_names = copy.copy(dataset_names)
    #test_dataset_names.remove(args.training_set)

    kwargs = {}

    transform = transforms.Compose([
            #transforms.Lambda(cv2_scale),
            transforms.Lambda(centerCrop),
            transforms.Lambda(np_reshape_color),
            transforms.ToTensor(),
            transforms.Normalize((args.mean_image,args.mean_image,args.mean_image),
                (args.std_image,args.std_image,args.std_image))])

    test_loaders = [{'name': name,
                     'dataloader': torch.utils.data.DataLoader(
             TripletPhotoTour(train=False,
                     batch_size=args.test_batch_size,
                     root=args.dataroot,
                     name=name,
                     download=True,
                     descriptor_flag=True,
                     transform=transform),
                        batch_size=args.test_batch_size,
                        shuffle=False, **kwargs)}
                    for name in test_dataset_names]

    return test_loaders

def test(test_loader, epoch, logger, logger_test_name):
    labels, distances = [], []
    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:
        data_a = data_a.numpy().astype(float)
        data_p = data_p.numpy().astype(float)
        data_a_p = np.power(data_a,2)
        data_a /= np.sqrt(data_a_p.sum(axis=1, keepdims=True) + 1e-7)
        data_p_p = np.power(data_p,2)
        data_p /= np.sqrt(data_p_p.sum(axis=1, keepdims=True) + 1e-7)
        dists = np.sqrt(np.sum((data_a-data_p)**2, 1))  # euclidean distance
        distances.append(dists.reshape(-1,1))
        ll = label.numpy().reshape(-1, 1)
        labels.append(ll)

    num_tests = test_loader.dataset.matches.size(0)
    labels = np.vstack(labels).reshape(num_tests)
    distances = np.vstack(distances).reshape(num_tests)

    fpr95 = ErrorRateAt95Recall(labels, 1.0 / (distances + 1e-8))
    print(logger_test_name)
    print('\33[91mTest set: Accuracy(FPR95): {:.8f}\n\33[0m'.format(fpr95))
    pos_dis = distances[labels==1]
    neg_dis = distances[labels==0]
    if (args.enable_logging):
        logger.log_histogram(logger_test_name+' pos dis',  pos_dis, step=epoch)
        logger.log_histogram(logger_test_name+' neg dis',  neg_dis, step=epoch)
        logger.log_value(logger_test_name+'_fpr95', fpr95, step=epoch)
    return

def main(test_loaders, logger, file_logger):
    print('\nparsed options:\n{}\n'.format(vars(args)))

    for test_loader in test_loaders:
        test(test_loader['dataloader'], 0, logger, test_loader['name'])
        
if __name__ == '__main__':
    LOG_DIR = args.log_dir
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    LOG_DIR = args.log_dir + suffix
    logger, file_logger = None, None

    if(args.enable_logging):
        logger = Logger(LOG_DIR)
    test_loaders = create_loaders()
    main(test_loaders, logger, file_logger)
