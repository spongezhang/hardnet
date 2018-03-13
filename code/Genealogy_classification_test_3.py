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
import genealogy_journal_test_2
from EvalMetrics import ErrorRateAt95Recall
from Losses import loss_HardNet, loss_random_sampling, loss_L2Net, global_orthogonal_regularization
from Utils import L2Norm, cv2_scale, np_reshape_color, centerCrop
from Utils import str2bool
import torch.nn as nn
import torch.nn.functional as F
import pdb
import collections
import glob

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
parser = argparse.ArgumentParser(description='PyTorch Genealogy Classification')
# Model options

parser.add_argument('--dataroot', type=str,
                    default='../datasets/',
                    help='path to dataset')
parser.add_argument('--enable-logging',type=bool, default=False,
                    help='output to tensorlogger')
parser.add_argument('--log-dir', default='../genealogy_log/',
                    help='folder to output log')
parser.add_argument('--model-dir', default='../genealogy_model/',
                    help='folder to output model checkpoints')
parser.add_argument('--training-set', default= 'synthesized_journals_2_train',
                    help='Other options: notredame, yosemite')
parser.add_argument('--loss', default= 'triplet_margin',
                    help='Other options: softmax, contrastive')
parser.add_argument('--batch-reduce', default= 'min',
                    help='Other options: average, random, random_global, L2Net')
parser.add_argument('--num-workers', default= 1,
                    help='Number of workers to be created')
parser.add_argument('--pin-memory',type=bool, default= True,
                    help='')
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
parser.add_argument('--n-pairs', type=int, default=5000000, metavar='N',
                    help='how many triplets will generate from the dataset')
parser.add_argument('--alpha', type=float, default=1.0, metavar='ALPHA',
                    help='gor parameter')
parser.add_argument('--act-decay', type=float, default=0,
                    help='activity L2 decay, default 0')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
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
parser.add_argument('--data_augment', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=10, metavar='LI',
                    help='how many batches to wait before logging training status')
parser.add_argument('--donor', action='store_true', default=False,
                    help='enables CUDA training')

args = parser.parse_args()

suffix = '{}'.format(args.training_set)

if args.loss == 'classification':
    suffix = suffix + '_m_{}'.format(args.margin)

if args.data_augment:
    suffix = suffix + '_da'
if args.donor:
    suffix = suffix + '_do'

args.resume = '{}{}/checkpoint_17.pth'.format(args.model_dir,suffix)

#dataset_names = ['NC2017_Dev1_Beta4'] #, 'NC2017_Dev2_Beta1_bg'

test_dataset = 'NC2017_Dev1_Beta4'
# set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(args.seed)

# create loggin directory
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

# set random seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)

class PairPhotoTour(genealogy_journal_test_2.genealogy_journal_test_2):
    """
    From the PhotoTour Dataset it generates triplet samples
    note: a triplet is composed by a pair of matching images and one of
    different class.
    """
    def __init__(self, train=True, transform=None, batch_size = None, *arg, **kw):
        super(PairPhotoTour, self).__init__(*arg, **kw)
        self.transform = transform
        self.train = train
        self.n_pairs = args.n_pairs
        self.batch_size = batch_size

    def __getitem__(self, index):
        def transform_img(img):
            if self.transform is not None:
                img = self.transform(img.numpy())
            return img

        if index%2==0:
            img1 = transform_img(self.data[index])
            img2 = transform_img(self.data[index+1])
            label = int(self.image_index[index]<self.image_index[index+1])
        else:
            img1 = transform_img(self.data[index])
            img2 = transform_img(self.data[index-1])
            label = int(self.image_index[index]<self.image_index[index-1])
        img1 = deepcopy(img1.numpy()[:,7:39,7:39])
        img2 = deepcopy(img2.numpy()[:,7:39,7:39])
        img_pair = torch.from_numpy(np.concatenate((img1,img2), axis = 2))
        return img_pair, label
        
    def __len__(self):
        return self.image_index.size(0)

class HardNet(nn.Module):
    """HardNet model definition
    """
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
            nn.Conv2d(128, 128, kernel_size=(8,16)),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(128, 2, kernel_size=(1,1))
        )
        self.features.apply(weights_init)
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), 3, -1)
        mp = torch.sum(flat, dim=2) / (32. * 32.)
        sp = torch.std(flat, dim=2) + 1e-7
        return (x - mp.unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.unsqueeze(-1).unsqueeze(-1).expand_as(x)

    def forward(self, input):
        x_features = self.features(input)
        x = x_features.view(x_features.size(0), -1)
        return x

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal(m.weight.data, gain=0.7)
        try:
            nn.init.constant(m.bias.data, 0.0)
        except:
            pass
    return

def test(model):
    # switch to evaluate mode
    model.eval()

    transform = transforms.Compose([
        #transforms.Lambda(cv2_scale),
        transforms.Lambda(centerCrop),
        transforms.Lambda(np_reshape_color),
        transforms.ToTensor(),
        transforms.Normalize((args.mean_image,args.mean_image,args.mean_image),
            (args.std_image,args.std_image,args.std_image))])

    kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory} if args.cuda else {}
    for journal in glob.glob(args.dataroot + test_dataset + "/*_patch.dat"):
        journal = os.path.basename(journal)
        journal_name = journal.split('_')[0]
        print(journal_name)
        test_loader = torch.utils.data.DataLoader(
                 PairPhotoTour(train=False,
                         batch_size=args.test_batch_size,
                         root=args.dataroot,
                         name=test_dataset,
                         journal_name = journal_name,
                         download=True,
                         transform=transform),
                            batch_size=args.test_batch_size,
                            shuffle=False, **kwargs)

        labels, predicts = [], []
        pbar = tqdm(enumerate(test_loader))

        for batch_idx, (image_pair, label) in pbar:
            if args.cuda:
                image_pair = image_pair.cuda()

            image_pair, label = Variable(image_pair, volatile=True), Variable(label)

            out = model(image_pair)
            _, pred = torch.max(out,1)
            ll = label.data.cpu().numpy().reshape(-1, 1)
            pred = pred.data.cpu().numpy().reshape(-1, 1)
            labels.append(ll)
            predicts.append(pred)
        
        num_tests = int(test_loader.dataset.image_index.size(0))
        labels = np.vstack(labels).reshape(num_tests)
        predicts = np.vstack(predicts).reshape(num_tests)

        num_trial = np.max(test_loader.dataset.trial_index.numpy())+1
        trial_count = collections.Counter(test_loader.dataset.trial_index.numpy())

        journal_trial_dict = {} 
        right_count = 0
        total_number = 0
        for i in range(num_trial):
            num_samples = int(trial_count[i])
            start_point = int(test_loader.dataset.trial_fast_access[i])

            journal_id = test_loader.dataset.journal_index[start_point]
            inx_str = '{}'.format(journal_id)
            try:
                journal_trial_dict[inx_str] += 1
            except:
                journal_trial_dict[inx_str] = 1

            real_label = int(test_loader.dataset.image_index[start_point]<test_loader.dataset.image_index[start_point+1])
            all_one = np.sum(predicts[int(start_point):(int(start_point)+num_samples):2])
            final_predict = int(all_one/float(num_samples/2)>0.5)
            if final_predict == real_label:
                right_count = right_count+1

            real_label = int(test_loader.dataset.image_index[start_point+1]<test_loader.dataset.image_index[start_point])
            all_one = np.sum(predicts[int(start_point+1):(int(start_point)+num_samples):2])
            final_predict = int(all_one/float(num_samples/2)>0.5)
            if final_predict == real_label:
                right_count = right_count+1

        total_num_dict = {}
        correct_num_dict = {}
        acc_dict = {}
        for ind, label in enumerate(labels):
            if ind%2==0:
                index_0 = test_loader.dataset.image_index[ind]
                index_1 = test_loader.dataset.image_index[ind+1]
            else:
                index_0 = test_loader.dataset.image_index[ind]
                index_1 = test_loader.dataset.image_index[ind-1]

            inx_str = '{}_{}'.format(index_0,index_1)
            try:
                total_num_dict[inx_str] += 1
            except:
                total_num_dict[inx_str] = 1
                correct_num_dict[inx_str] = 0
            if label == predicts[ind]:
                correct_num_dict[inx_str] +=1

        for k,v in total_num_dict.items():
            acc_dict[k] = correct_num_dict[k]/float(v)
            sys.stdout.write('{}: {:.2f} '.format(k, correct_num_dict[k]/float(v)))
        sys.stdout.flush()
        print('')

        total_num_dict = {}
        correct_num_dict = {}
        acc_dict = {}
        for ind, label in enumerate(labels):
            journal_id = test_loader.dataset.journal_index[ind]
            inx_str = '{}'.format(journal_id)
            try:
                total_num_dict[inx_str] += 1
            except:
                total_num_dict[inx_str] = 1
                correct_num_dict[inx_str] = 0
            if label == predicts[ind]:
                correct_num_dict[inx_str] +=1

        for k,v in total_num_dict.items():
            acc_dict[k] = correct_num_dict[k]/float(v)

        print('Trial Acc: {:.2f}'.format(right_count/float(num_trial*2)))

        acc = np.sum(labels == predicts)/float(num_tests)
        print('\33[91mTest set: Accuracy: {:.8f}\n\33[0m'.format(acc))
    return


def main(model):
    print('\nparsed options:\n{}\n'.format(vars(args)))
    if args.cuda:
        model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print('=> no checkpoint found at {}'.format(args.resume))
            
    test(model)
        
if __name__ == '__main__':
    model = HardNet()
    main(model)
