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
import genealogy_journal
from EvalMetrics import ErrorRateAt95Recall
from Losses import loss_HardNet, loss_random_sampling, loss_L2Net, global_orthogonal_regularization
from Utils import L2Norm, cv2_scale, np_reshape_color, centerCrop
from Utils import str2bool
import torch.nn as nn
import torch.nn.functional as F
import pdb

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
parser.add_argument('--png', action='store_true', default=False,
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
dataset_names = ['NC2017_Dev1_Beta4_bg', 'NC2017_Dev2_Beta1_bg'] #

if args.png:
    args.training_set = args.training_set + '_png'
    for ind in range(len(dataset_names)):
        dataset_names[ind] = dataset_names[ind] + '_png'
print(dataset_names)

suffix = '{}'.format(args.training_set)

if args.loss == 'classification':
    suffix = suffix + '_m_{}'.format(args.margin)

if args.data_augment:
    suffix = suffix + '_da'
if args.donor:
    suffix = suffix + '_do'


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

class PairPhotoTour(genealogy_journal.genealogy_journal):
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

        if self.train:
            print('Generating {} pairs'.format(self.n_pairs))
            self.pairs, self.label = self.generate_pairs(self.labels, self.n_pairs)

    @staticmethod
    def generate_pairs(labels, num_pairs):
        def create_indices(_labels):
            inds = dict()
            for idx, ind in enumerate(_labels):
                if ind not in inds:
                    inds[ind] = []
                inds[ind].append(idx)
            return inds

        pairs = []
        label = []
        indices = create_indices(labels)
        unique_labels = np.unique(labels.numpy())
        n_classes = unique_labels.shape[0]
        # add only unique indices in batch
        already_idxs = set()

        for x in tqdm(range(num_pairs)):
            if len(already_idxs) >= args.batch_size:
                already_idxs = set()
            c1 = np.random.randint(0, n_classes)
            while c1 in already_idxs:
                c1 = np.random.randint(0, n_classes)
            already_idxs.add(c1)
            if len(indices[c1]) == 2:  # hack to speed up process
                n1, n2 = 0, 1
            else:
                if args.donor:
                    n1 = 0 
                    n2 = np.random.randint(1, len(indices[c1]))
                    tmp_label = np.random.randint(0, 2)
                    if tmp_label<1: 
                        n1 = n2
                        n2 = 0
                else:
                    n1 = np.random.randint(0, len(indices[c1]))
                    n2 = np.random.randint(0, len(indices[c1]))
                while n1 == n2:
                    n2 = np.random.randint(0, len(indices[c1]))

            tmp_label = np.random.randint(0, 2)
            if n1 > n2:
                tmp_swap = n1
                n1 = n2
                n2 = tmp_swap
            if tmp_label<1: 
                tmp_swap = n1
                n1 = n2
                n2 = tmp_swap
            #tmp_label = 1 if n1<n2 else 0 
            pairs.append([indices[c1][n1], indices[c1][n2]])
            label.append(tmp_label)
        return torch.LongTensor(np.array(pairs)), torch.LongTensor(np.array(label))

    def __getitem__(self, index):
        def transform_img(img):
            if self.transform is not None:
                img = self.transform(img.numpy())
            return img

        if not self.train:
            m = self.pc_pairs[index]
            img1 = transform_img(self.data[m[0]])
            img2 = transform_img(self.data[m[1]])
            img1 = deepcopy(img1.numpy()[:,7:39,7:39])
            img2 = deepcopy(img2.numpy()[:,7:39,7:39])
            img_pair = torch.from_numpy(np.concatenate((img1,img2), axis = 2))
            return img_pair, m[2]
        
        t = self.pairs[index]
        label = self.label[index]
        a, p = self.data[t[0]], self.data[t[1]]
        
        img_a = transform_img(a)
        img_p = transform_img(p)

        if not args.data_augment:
            #pass
            img_a = deepcopy(img_a.numpy()[:,7:39,7:39])
            img_p = deepcopy(img_p.numpy()[:,7:39,7:39])
        else:
            random_x = random.randint(0,8)
            random_y = random.randint(0,8)
            img_a = deepcopy(img_a.numpy()[:,random_y:(random_y+32),\
                    random_x:(random_x+32)])
            random_x = random.randint(0,8)
            random_y = random.randint(0,8)
            img_p = deepcopy(img_p.numpy()[:,random_y:(random_y+32),\
                    random_x:(random_x+32)])
            # transform images if required
            if args.fliprot:
                do_flip = random.random() > 0.5
                do_rot = random.random() > 0.5
                if do_rot:
                    img_a = img_a.permute(0,2,1)
                    img_p = img_p.permute(0,2,1)
                if do_flip:
                    img_a = deepcopy(img_a.numpy()[:,:,::-1])
                    img_p = deepcopy(img_p.numpy()[:,:,::-1])
        img_pair = torch.from_numpy(np.concatenate((img_a,img_p), axis = 2))
        return (img_pair, label)

    def __len__(self):
        if self.train:
            return self.pairs.size(0)
        else:
            return self.matches.size(0)

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
    
    def forward_2(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        return x

    def forward(self, input):
        #flat = input.view(input.size(0), -1)
        #mp = args.mean-image#torch.sum(flat, dim=1) / (32. * 32.)
        #sp = args.std-image#torch.std(flat, dim=1) + 1e-7
        #x_features = self.features(
            #(input - mp.unsqueeze(-1).unsqueeze(-1).expand_as(input)) / sp.unsqueeze(-1).unsqueeze(1).expand_as(input))
        x_features = self.features(input)
        x = x_features.view(x_features.size(0), -1)
        return x

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal(m.weight.data, gain=0.7)
        #nn.init.kaiming_normal(m.weight.data)
        try:
            nn.init.constant(m.bias.data, 0.0)
        except:
            pass
    return

def create_loaders():

    test_dataset_names = copy.copy(dataset_names)
    #test_dataset_names.remove(args.training_set)

    kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory} if args.cuda else {}

    transform = transforms.Compose([
            #transforms.Lambda(cv2_scale),
            transforms.Lambda(centerCrop),
            transforms.Lambda(np_reshape_color),
            transforms.ToTensor(),
            transforms.Normalize((args.mean_image,args.mean_image,args.mean_image),
                (args.std_image,args.std_image,args.std_image))])

    train_loader = torch.utils.data.DataLoader(
            PairPhotoTour(train=True,
                             batch_size=args.batch_size,
                             root=args.dataroot,
                             name=args.training_set,
                             download=True,
                             transform=transform),
                             batch_size=args.batch_size,
                             shuffle=True, **kwargs)

    test_loaders = [{'name': name,
                     'dataloader': torch.utils.data.DataLoader(
             PairPhotoTour(train=False,
                     batch_size=args.test_batch_size,
                     root=args.dataroot,
                     name=name,
                     download=True,
                     transform=transform),
                        batch_size=args.test_batch_size,
                        shuffle=False, **kwargs)}
                    for name in test_dataset_names]

    return train_loader, test_loaders

def train(train_loader, model, optimizer, criterion,  epoch, logger):
    # switch to train mode
    model.train()
    pbar = tqdm(enumerate(train_loader))
    for batch_idx, data in pbar:
        image_pair, label = data

        if args.cuda:
            image_pair, label  = image_pair.cuda(), label.cuda()
            image_pair, label = Variable(image_pair), Variable(label)
            out= model(image_pair)

        loss = criterion(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    adjust_learning_rate(optimizer)

    if (args.enable_logging):
        logger.log_value('loss', loss.data[0]).step()

    try:
        os.stat('{}{}'.format(args.model_dir,suffix))
    except:
        os.makedirs('{}{}'.format(args.model_dir,suffix))

    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
               '{}{}/checkpoint_{}.pth'.format(args.model_dir,suffix,epoch))

def test(test_loader, model, epoch, logger, logger_test_name):
    # switch to evaluate mode
    model.eval()

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

    num_tests = test_loader.dataset.matches.size(0)
    labels = np.vstack(labels).reshape(num_tests)
    predicts = np.vstack(predicts).reshape(num_tests)

    acc = np.sum(labels == predicts)/float(num_tests)
    print('\33[91mEpoch: {}, Test set: Accuracy: {:.8f}\n\33[0m'.format(epoch,acc))

    if (args.enable_logging):
        logger.log_value(logger_test_name+' acc', acc)
    return

def adjust_learning_rate(optimizer):
    """Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    for group in optimizer.param_groups:
        if 'step' not in group:
            group['step'] = 0.
        else:
            group['step'] += 1.
        group['lr'] = args.lr
        #group['lr'] = args.lr * (1.0 - float(group['step']) * \
        #        float(args.batch_size)/(args.n_pairs * float(args.epochs)))
    return

def create_optimizer(model, new_lr):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, dampening=0.9,
                              weight_decay=args.wd)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=args.wd)
    else:
        raise Exception('Not supported optimizer: {0}'.format(args.optimizer))
    return optimizer


def main(train_loader, test_loaders, model, logger, file_logger):
    print('\nparsed options:\n{}\n'.format(vars(args)))

    optimizer1 = create_optimizer(model.features, args.lr)
    criterion = nn.CrossEntropyLoss()
    if args.cuda:
        model.cuda()
        criterion.cuda()

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
            
    start = args.start_epoch
    end = start + args.epochs
    for test_loader in test_loaders:
        test(test_loader['dataloader'], model, 0, logger, test_loader['name'])
    for epoch in range(start, end):
        # iterate over test loaders and test results
        train(train_loader, model, optimizer1, criterion, epoch, logger)
        for test_loader in test_loaders:
            test(test_loader['dataloader'], model, epoch, logger, test_loader['name'])
        
if __name__ == '__main__':
    LOG_DIR = args.log_dir
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    LOG_DIR = args.log_dir + suffix
    logger, file_logger = None, None
    model = HardNet()
    if(args.enable_logging):
        from Loggers import Logger, FileLogger
        logger = Logger(LOG_DIR)
        #file_logger = FileLogger(./log/+suffix)
    train_loader, test_loaders = create_loaders()
    main(train_loader, test_loaders, model, logger, file_logger)
