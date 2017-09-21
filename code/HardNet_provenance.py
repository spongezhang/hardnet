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
from Loggers import Logger
from Utils import L2Norm, cv2_scale, np_reshape
from Utils import str2bool

from scipy.spatial import distance

import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import plotly.plotly as py

# Training settings
parser = argparse.ArgumentParser(description='PyTorch HardNet')
# Model options

parser.add_argument('--dataroot', type=str,
                    default='../datasets/',
                    help='path to dataset')
parser.add_argument('--enable-logging',type=bool, default=True,
                    help='folder to output model checkpoints')
parser.add_argument('--log-dir', default='../provenance_log/',
                    help='folder to output model checkpoints')
parser.add_argument('--model-dir', default='../provenance_model/',
                    help='folder to output model checkpoints')
parser.add_argument('--training-set', default= 'synthesized_journals_train',
                    help='Other options: notredame, yosemite')
parser.add_argument('--test-set', default= 'synthesized_journals_test_direct',
                    help='Other options: notredame, yosemite')
parser.add_argument('--num-workers', default= 1,
                    help='Number of workers to be created')
parser.add_argument('--pin-memory',type=bool, default= True,
                    help='')
parser.add_argument('--anchorave', type=bool, default=True,
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
parser.add_argument('--test-batch-size', type=int, default=1024, metavar='BST',
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
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')

parser.add_argument('--log-interval', type=int, default=10, metavar='LI',
                    help='how many batches to wait before logging training status')

parser.add_argument("--loss_type", nargs='?', type=int, default = 0,
                    help="Number of embedding dimemsion")

parser.add_argument("--alpha", nargs='?', type=float, default = 0.0,
                    help="alpha")

parser.add_argument("--beta", nargs='?', type=float, default = 0.0,
                    help="beta")

args = parser.parse_args()

suffix = 'alpha{:1.1e}_beta{:1.1e}_lossType_{}'\
        .format( args.alpha, args.beta, args.margin, args.loss_type)
        
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

class TripletPhotoTour(synthesized_journal.synthesized_journal):
    """From the PhotoTour Dataset it generates triplet samples
    note: a triplet is composed by a pair of matching images and one of
    different class.
    """
    def __init__(self, train=True, transform=None, batch_size = None, *arg, **kw):
        super(TripletPhotoTour, self).__init__(*arg, **kw)
        self.transform = transform

        self.train = train
        self.n_triplets = args.n_triplets
        self.batch_size = batch_size
        
        #if not self.train:
        #    print(self.data[0].numpy().shape)
        #    print(self.data[0].numpy()[:5,30,:])
        
        if self.train:
            print('Generating {} triplets'.format(self.n_triplets))
            self.triplets = self.generate_triplets(self.labels, self.n_triplets)

    @staticmethod
    def generate_triplets(labels, num_triplets):
        def create_indices(_labels):
            inds = dict()
            for idx, ind in enumerate(_labels):
                if ind not in inds:
                    inds[ind] = []
                inds[ind].append(idx)
            return inds

        triplets = []
        indices = create_indices(labels)
        unique_labels = np.unique(labels.numpy())
        n_classes = unique_labels.shape[0]
        # add only unique indices in batch
        already_idxs = set()

        for x in tqdm(range(num_triplets)):
            if len(already_idxs) >= args.batch_size:
                already_idxs = set()
            c1 = np.random.randint(0, n_classes - 1)
            while c1 in already_idxs:
                c1 = np.random.randint(0, n_classes - 1)
            already_idxs.add(c1)
            c2 = np.random.randint(0, n_classes - 1)
            while c1 == c2:
                c2 = np.random.randint(0, n_classes - 1)
            if len(indices[c1]) == 2:  # hack to speed up process
                n1, n2 = 0, 1
            else:
                n1 = 0#np.random.randint(0, len(indices[c1]) - 1)
                n2 = np.random.randint(1, len(indices[c1]) - 1)
                while n1 == n2:
                    n2 = np.random.randint(0, len(indices[c1]) - 1)
            n3 = np.random.randint(0, len(indices[c2]) - 1)
            triplets.append([indices[c1][n1], indices[c1][n2], indices[c2][n3]])
        return torch.LongTensor(np.array(triplets))

    def remove_easy_triplets(self, embeddings):

        old_triplets = self.triplets.numpy()
        new_triplets = []
        old_triplets_len = old_triplets.shape[0]
        good_num = 0

        for index in tqdm(xrange(old_triplets_len)):
            pos_distance = distance.euclidean(embeddings[old_triplets[index][0]], embeddings[old_triplets[index][1]])

            if pos_distance<0.4:
                prob = random.randint(0,9)
                good_num = good_num+1
                if prob<5:
                    continue
                
            new_triplets.append(old_triplets[index])

        print('good number: {}'.format(good_num))
        print('old number: {}'.format(old_triplets_len))
        print('new number: {}'.format(len(new_triplets)))
        self.n_triplets = len(new_triplets)
        self.triplets = torch.LongTensor(np.array(new_triplets))

    def __getitem__(self, index):
        def transform_img(img):
            if self.transform is not None:
                img = self.transform(img.numpy())
            return img

        if not self.train:
            m = self.matches[index]
            #if m[1]==0:
            #    print(self.data[m[1]].numpy().shape)
            #    print(self.data[m[1]].numpy()[:5,30,:])
            img1 = transform_img(self.data[m[0]])
            img2 = transform_img(self.data[m[1]])
            #if m[1]==0:
            #    print(img1.numpy().shape)
            #    print(img1.numpy()[:,:5,15])
            return img1, img2, m[2]
        
        t = self.triplets[index]
        #print(t)
        a, p, n = self.data[t[0]], self.data[t[1]], self.data[t[2]]
        #print(type(a))
        #print(a.size())
        #exit()
        img_a = transform_img(a)
        img_p = transform_img(p)

        # transform images if required
        if args.fliprot:
            do_flip = random.random() > 0.5
            do_rot = random.random() > 0.5

            if do_rot:
                img_a = img_a.permute(0,2,1)
                img_p = img_p.permute(0,2,1)

            if do_flip:
                img_a = torch.from_numpy(deepcopy(img_a.numpy()[:,:,::-1]))
                img_p = torch.from_numpy(deepcopy(img_p.numpy()[:,:,::-1]))
        return img_a, img_p

    def __len__(self):
        if self.train:
            return self.triplets.size(0)
        else:
            return self.matches.size(0)

class simpleDataLoader(synthesized_journal.synthesized_journal):
    """From the PhotoTour Dataset it generates triplet samples
    note: a triplet is composed by a pair of matching images and one of
    different class.
    """
    def __init__(self, train=True, transform=None, batch_size = None, *arg, **kw):
        super(simpleDataLoader, self).__init__(*arg, **kw)
        self.transform = transform

        self.train = train
        self.n_triplets = args.n_triplets
        self.batch_size = batch_size
        self.num_sample = self.data.numpy().shape[0]
        
    def __getitem__(self, index):
        def transform_img(img):
            if self.transform is not None:
                img = self.transform(img.numpy())
            return img

        #if not self.train:
        img = transform_img(self.data[index])
        return img

    def __len__(self):
        return self.num_sample

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
            #nn.Dropout(0.1),
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
        flat = input.view(input.size(0), -1)
        #mp = args.mean-image#torch.sum(flat, dim=1) / (32. * 32.)
        #sp = args.std-image#torch.std(flat, dim=1) + 1e-7
        #x_features = self.features(
            #(input - mp.unsqueeze(-1).unsqueeze(-1).expand_as(input)) / sp.unsqueeze(-1).unsqueeze(1).expand_as(input))
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

def create_loaders():
    kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory} if args.cuda else {}

    transform = transforms.Compose([
            transforms.Lambda(cv2_scale),
            transforms.Lambda(np_reshape),
            transforms.ToTensor(),
            #transforms.Normalize((args.mean_image,),(args.std_image,))
            transforms.Normalize((args.mean_image,args.mean_image,args.mean_image),
                (args.std_image,args.std_image,args.std_image))
            ])

    train_loader = torch.utils.data.DataLoader(
            TripletPhotoTour(train=True,
                             batch_size=args.batch_size,
                             root=args.dataroot,
                             name=args.training_set,
                             download=True,
                             transform=transform),
                             batch_size=args.batch_size,
                             shuffle=False, **kwargs)

    train_sample_loader = torch.utils.data.DataLoader(
            simpleDataLoader(train=True,
                             batch_size=args.batch_size,
                             root=args.dataroot,
                             name=args.training_set,
                             download=True,
                             transform=transform),
                             batch_size=args.test_batch_size,
                             shuffle=False, **kwargs)

    test_loader =  torch.utils.data.DataLoader(
                    TripletPhotoTour(train=False,
                     batch_size=args.test_batch_size,
                     root=args.dataroot,
                     name=args.test_set,
                     download=True,
                     transform=transform),
                     batch_size=args.test_batch_size,
                     shuffle=False, **kwargs)

    return train_loader, train_sample_loader, test_loader

def train(train_loader, model, optimizer, epoch, logger):
    # switch to train mode
    model.train()
    print('Number of Batches in training: {}'.format(len(train_loader)))
    print(len(train_loader.dataset))
    pbar = tqdm(enumerate(train_loader))
    for batch_idx, (data_a, data_p) in pbar:
        #print(data_a.shape())
        if args.cuda:
            data_a, data_p = data_a.cuda(), data_p.cuda()

        data_a, data_p = Variable(data_a), Variable(data_p)
        out_a, out_p = model(data_a), model(data_p)

        #hardnet loss
        if args.loss_type==0:
            loss = loss_margin_min(out_a, out_p, margin=args.margin, anchor_swap=args.anchorswap, alpha = args.alpha, anchor_ave=args.anchorave)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        adjust_learning_rate(optimizer)

        logger.log_value('loss', loss.data[0]).step()

    try:
        os.stat('{}{}'.format(args.model_dir,suffix))
    except:
        os.mkdir('{}{}'.format(args.model_dir,suffix))

    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
               '{}{}/checkpoint_{}.pth'.format(args.model_dir,suffix,epoch))

def remove_easy(train_loader, train_sample_loader, model, epoch, logger):
    # switch to evaluate mode
    model.eval()

    embedding = []
    sample_number = train_sample_loader.dataset.num_sample
    pbar = tqdm(enumerate(train_sample_loader))
    descriptor_dim = 128
    for batch_idx, data_a in pbar:
        if args.cuda:
            data_a = data_a.cuda()

        data_a = Variable(data_a, volatile=True)
        out_a = model(data_a)
        embedding.append(out_a.data.cpu().numpy())

    embedding = np.vstack(embedding).reshape(sample_number,descriptor_dim)
    train_loader.dataset.remove_easy_triplets(embedding)
    return

def test(test_loader, model, epoch, logger, logger_test_name):
    # switch to evaluate mode
    model.eval()

    labels, distances = [], []

    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:
        if args.cuda:
            data_a, data_p = data_a.cuda(), data_p.cuda()

        data_a, data_p, label = Variable(data_a, volatile=True), \
                                Variable(data_p, volatile=True), Variable(label)

        out_a, out_p = model(data_a), model(data_p)

            
        dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
        distances.append(dists.data.cpu().numpy().reshape(-1,1))
        ll = label.data.cpu().numpy().reshape(-1, 1)
        labels.append(ll)

    
    num_tests = test_loader.dataset.matches.size(0)
    labels = np.vstack(labels).reshape(num_tests)
    distances = np.vstack(distances).reshape(num_tests)

    fpr95 = ErrorRateAt95Recall(labels, distances)
    print('\33[91mTest set: Accuracy(FPR95): {:.8f}\n\33[0m'.format(fpr95))
    if (args.enable_logging):
        logger.log_value('fpr95', fpr95)
        
    if True:
        try: 
            os.stat('../histogram_map/'+suffix)
        except:
            os.mkdir('../histogram_map/'+suffix)

        bins = np.linspace(0, 2, 100)
        cos_dists = distances;
        pos_dists = cos_dists[labels==1]
        neg_dists = cos_dists[labels==0]
        plt.hist(pos_dists, bins, alpha = 0.5, label = 'Matched Pairs')
        plt.hist(neg_dists, bins, alpha = 0.5, label = 'Non-Matched Pairs')
        plt.legend(loc='upper left')
        plt.xlim(0, 2)
        plt.ylim(0, 7e4)
        plt.xlabel('l2')
        plt.ylabel('#Pairs')
        plt.savefig('../histogram_map/{}/iter_{}.png'.format(suffix,epoch), bbox_inches='tight')
        plt.clf()

    good_match_ratio = np.sum((distances<0.3)*labels)/np.sum(labels==1)
    print('Good match ratio for test: {}'.format(good_match_ratio))
    if (args.enable_logging):
        logger.log_value('good_match_ratio', good_match_ratio)

    good_mismatch_ratio = np.sum((distances>0.8)*(1-labels))/np.sum(labels==0)
    print('Good mismatch ratio for test: {}'.format(good_mismatch_ratio))
    if (args.enable_logging):
        logger.log_value('good_mismatch_ratio', good_mismatch_ratio)

    bad_mismatch_ratio = np.sum((distances<0.4)*(1-labels))/np.sum(labels==0)
    print('Bad mismatch ratio for test: {}'.format(bad_mismatch_ratio))
    if (args.enable_logging):
        logger.log_value('bad_mismatch_ratio', bad_mismatch_ratio)

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
        group['lr'] = args.lr * (
        1.0 - float(group['step']) * float(args.batch_size) / (args.n_triplets * float(args.epochs)))
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


def main(train_loader, train_sample_loader, test_loader, model, logger):
    # print the experiment configuration
    print('\nparsed options:\n{}\n'.format(vars(args)))
    if args.cuda:
        model.cuda()
    optimizer1 = create_optimizer(model.features, args.lr)

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
    for epoch in range(start, end):
        train(train_loader, model, optimizer1, epoch, logger)
        #remove_easy(train_loader, train_sample_loader, model, epoch, logger)
        test(test_loader, model, epoch, logger, args.test_set)

if __name__ == '__main__':
    LOG_DIR = args.log_dir + suffix
    logger = None
    model = TNet()
    if(args.enable_logging):
        logger = Logger(LOG_DIR)
        train_loader, train_sample_loader, test_loader = create_loaders()
    main(train_loader, train_sample_loader, test_loader, model, logger)
