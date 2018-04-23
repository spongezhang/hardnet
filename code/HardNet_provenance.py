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
import nmslib

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
parser.add_argument('--decor',type=str2bool, default = False,
                    help='L2Net decorrelation penalty')
parser.add_argument('--anchorave', type=str2bool, default=False,
                    help='anchorave')
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
parser.add_argument('--margin', type=float, default=1.0, metavar='MARGIN',
                    help='the margin value for the triplet loss function (default: 1.0')
parser.add_argument('--gor',type=str2bool, default=False,
                    help='use gor')
parser.add_argument('--alpha', type=float, default=1.0, metavar='ALPHA',
                    help='gor parameter')
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
parser.add_argument('--data_augment', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=10, metavar='LI',
                    help='how many batches to wait before logging training status')
parser.add_argument('--no-hinge', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--donor', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--hard_mining', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--input_norm', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--no-mask', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--bg', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--inner-product', action='store_true', default=False,
                    help='enables CUDA training')

args = parser.parse_args()

#dataset_names = ['NC2017_Dev1_Beta4', 'NC2017_Dev2_Beta1',\
#        'MFC18_Dev1_Ver2', 'MFC18_Dev2_Ver1', 'synthesized_journals_test_direct']

dataset_names = []#'NC2017_Dev2_Beta1', 'MFC18_Dev2_Ver1', 'synthesized_journals_test_direct'

if args.bg:
    dataset_names = [dataset + '_bg' for dataset in dataset_names]
    args.training_set = args.training_set + '_bg'

suffix = '{}_{}_{}'.format(args.training_set, args.batch_reduce, args.loss)

if args.loss == 'classification':
    suffix = suffix + '_m_{}'.format(args.margin)

if args.gor:
    suffix = suffix+'_gor_alpha{:1.1f}'.format(args.alpha)
if args.anchorswap:
    suffix = suffix + '_as'
if args.anchorave:
    suffix = suffix + '_av'
if args.no_hinge:
    suffix = suffix + '_nh'
if args.inner_product:
    suffix = suffix + '_ip'
if args.no_mask:
    suffix = suffix + '_nm'
if args.data_augment:
    suffix = suffix + '_da'
if args.donor:
    suffix = suffix + '_do'
if args.fliprot:
    suffix = suffix + '_fr'
if args.input_norm:
    suffix = suffix + '_ln'
if args.resume != '':
    suffix = suffix + '_re'
if args.hard_mining != '':
    suffix = suffix + '_hm'

da_offset = 4

triplet_flag = True#(args.batch_reduce == 'random_global') or args.gor 

#dataset_names = ['NC2017_Dev1_Beta4_bg']
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
    """
    From the PhotoTour Dataset it generates triplet samples
    note: a triplet is composed by a pair of matching images and one of
    different class.
    """
    def __init__(self, train=True, transform=None, hm=False, batch_size = None, load_random_triplets = False,  *arg, **kw):
        super(TripletPhotoTour, self).__init__(*arg, **kw)
        self.transform = transform
        self.out_triplets = load_random_triplets
        self.train = train
        self.hm = hm
        self.n_triplets = args.n_triplets
        self.batch_size = batch_size

        if self.train:
            print('Generating {} triplets'.format(self.n_triplets))
            #try:
            #    self.triplets = np.load('../data/{}_triplets.npz'.format(args.training_set))
            #    self.triplets = self.triplets['arr_0']
            #    self.triplets = torch.LongTensor(self.triplets)
            #except:
            self.triplets = self.generate_triplets(self.labels, self.n_triplets)
            print('Triplet shape: {}'.format(self.triplets.shape))

    def update_triplets(self):
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
                if args.donor:
                    n1 = 0 
                    n2 = np.random.randint(1, len(indices[c1]) - 1)
                else:
                    n1 = np.random.randint(0, len(indices[c1]) - 1)
                    n2 = np.random.randint(0, len(indices[c1]) - 1)
                while n1 == n2:
                    n2 = np.random.randint(0, len(indices[c1]) - 1)
            n3 = np.random.randint(0, len(indices[c2]) - 1)
            triplets.append([indices[c1][n1], indices[c1][n2], indices[c2][n3]])
        return torch.LongTensor(np.array(triplets))

    def __getitem__(self, index):
        def transform_img(img):
            if self.transform is not None:
                img = self.transform(img.numpy())
            return img

        if not self.train:
            m = self.matches[index]
            img1 = transform_img(self.data[m[0]])
            img2 = transform_img(self.data[m[1]])
            img1 = torch.from_numpy(deepcopy(img1.numpy()[:,8:40,8:40]))
            img2 = torch.from_numpy(deepcopy(img2.numpy()[:,8:40,8:40]))
            return img1, img2, m[2]

        if self.hm:
            img1 = transform_img(self.data[index])
            img1 = torch.from_numpy(deepcopy(img1.numpy()[:,8:40,8:40]))
            return img1

        t = self.triplets[index]
        a, p, n = self.data[t[0]], self.data[t[1]], self.data[t[2]]
        
        img_a = transform_img(a)
        img_p = transform_img(p)
        img_n = None
        if self.out_triplets:
            img_n = transform_img(n)
        if not args.data_augment:
            #pass
            img_a = torch.from_numpy(deepcopy(img_a.numpy()[:,8:40,8:40]))
            img_p = torch.from_numpy(deepcopy(img_p.numpy()[:,8:40,8:40]))
            if self.out_triplets:
                img_n = torch.from_numpy(deepcopy(img_n.numpy()[:,8:40,8:40]))
        else:
            random_x = random.randint(8-da_offset/2,8+da_offset/2)
            random_y = random.randint(8-da_offset/2,8+da_offset/2)
            img_a = torch.from_numpy(deepcopy(img_a.numpy()[:,random_y:(random_y+32),\
                    random_x:(random_x+32)]))
            random_x = random.randint(8-da_offset/2,8+da_offset/2)
            random_y = random.randint(8-da_offset/2,8+da_offset/2)
            img_p = torch.from_numpy(deepcopy(img_p.numpy()[:,random_y:(random_y+32),\
                    random_x:(random_x+32)]))
            if self.out_triplets:
                random_x = random.randint(8-da_offset/2,8+da_offset/2)
                random_y = random.randint(8-da_offset/2,8+da_offset/2)
                img_n = torch.from_numpy(deepcopy(img_n.numpy()[:,random_y:(random_y+32),\
                    random_x:(random_x+32)]))
            # transform images if required
            if args.fliprot:
                do_flip = random.random() > 0.5
                do_rot = random.random() > 0.5
                if do_rot:
                    img_a = img_a.permute(0,2,1)
                    #img_p = img_p.permute(0,2,1)
                    if self.out_triplets:
                        img_n = img_n.permute(0,2,1)
                if do_flip:
                    img_a = torch.from_numpy(deepcopy(img_a.numpy()[:,:,::-1]))
                    #img_p = torch.from_numpy(deepcopy(img_p.numpy()[:,:,::-1]))
                    if self.out_triplets:
                        img_n = torch.from_numpy(deepcopy(img_n.numpy()[:,:,::-1]))
        if self.out_triplets:
            return (img_a, img_p, img_n, self.labels[t[0]])
        else:
            return (img_a, img_p)

    def __len__(self):
        if self.hm:
            return self.data.shape[0]
        elif self.train:
            return self.triplets.size(0)
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
            nn.Dropout(0.1),
            nn.Conv2d(128, 128, kernel_size=8, bias = False),
            nn.BatchNorm2d(128, affine=False),
        )
        self.features.apply(weights_init)
        return
    
    def input_norm_2(self,x):
        flat = x.view(x.size(0), 3, -1)
        mp = torch.sum(flat, dim=2) / (32. * 32.)
        sp = torch.std(flat, dim=2) + 1e-7
        return (x - mp.unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.unsqueeze(-1).unsqueeze(-1).expand_as(x)

    def input_norm(self,x):
        flat = x.view(x.size(0),  -1)
        mp = torch.sum(flat, dim=1) / (32. * 32.)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
    
    def forward_2(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

    def forward(self, input):
        if args.input_norm:
            x_features = self.features(self.input_norm(input))
        else:
            x_features = self.features(input)
        x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal(m.weight.data, gain=0.7)
        try:
            nn.init.constant(m.bias.data, 0.01)
        except:
            pass
    return

def create_loaders(load_random_triplets = False):

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
            TripletPhotoTour(train=True,
                             load_random_triplets = load_random_triplets,
                             batch_size=args.batch_size,
                             root=args.dataroot,
                             name=args.training_set,
                             download=True,
                             transform=transform),
                             batch_size=args.batch_size,
                             shuffle=False, **kwargs)

    test_loaders = [{'name': name,
                     'dataloader': torch.utils.data.DataLoader(
                     TripletPhotoTour(train=False,
                     batch_size=args.test_batch_size,
                     root=args.dataroot,
                     name=name,
                     download=True,
                     transform=transform),
                        batch_size=args.test_batch_size,
                        shuffle=False, **kwargs)}
                    for name in test_dataset_names]

    return train_loader, test_loaders

def train(train_loader, model, optimizer, epoch, logger, load_triplets  = False):
    # switch to train mode
    model.train()

    triplet = train_loader.dataset.triplets.numpy()
    perm_idx = np.random.permutation(triplet.shape[0])
    triplet = triplet[perm_idx, :]
    train_loader.dataset.triplets = torch.LongTensor(triplet)
    pbar = tqdm(enumerate(train_loader))

    for batch_idx, data in pbar:
        if load_triplets:
            data_a, data_p, data_n, label = data
        else:
            data_a, data_p = data

        if args.cuda:
            data_a, data_p  = data_a.cuda(), data_p.cuda()
            data_a, data_p = Variable(data_a), Variable(data_p)
            out_a, out_p = model(data_a), model(data_p)

        if load_triplets:
            label = label.numpy()
            label = label.reshape(-1,1)
            label_matrix = np.repeat(label,label.shape[0],axis = 1)
            label_matrix = np.equal(label_matrix,label_matrix.T)
            label_matrix = label_matrix.astype(np.float32)
            label_matrix = torch.from_numpy(label_matrix)
            data_n  = data_n.cuda()
            data_n = Variable(data_n)
            label_matrix  = label_matrix.cuda()
            label_matrix = Variable(label_matrix)
            out_n = model(data_n)
        
        if args.batch_reduce == 'L2Net':
            loss = loss_L2Net(out_a, out_p, anchor_swap = args.anchorswap,
                    margin = args.margin, loss_type = args.loss)
        elif args.batch_reduce == 'random_global':
            loss = loss_random_sampling(out_a, out_p, out_n,
                margin=args.margin,
                anchor_swap=args.anchorswap,
                loss_type = args.loss)
        else:
            loss = loss_HardNet(out_a, out_p, out_n, label_matrix,
                            margin=args.margin,
                            anchor_swap=args.anchorswap,
                            anchor_ave=args.anchorave,
                            no_hinge = args.no_hinge, 
                            no_mask = args.no_mask, 
                            inner_product = args.inner_product,
                            batch_reduce = args.batch_reduce,
                            loss_type = args.loss)

        if args.decor:
            loss += CorrelationPenaltyLoss()(out_a)
            
        if args.gor:
            loss += args.alpha*global_orthogonal_regularization(out_a, out_n)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        adjust_learning_rate(optimizer)

    if (args.enable_logging):
        logger.log_value('loss', loss.data[0], step=epoch)

    try:
        os.stat('{}{}'.format(args.model_dir,suffix))
    except:
        os.makedirs('{}{}'.format(args.model_dir,suffix))

    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
               '{}{}/checkpoint_{}.pth'.format(args.model_dir,suffix,epoch))

def hm(train_loader, model, epoch, logger, update_flag=False):
    # switch to evaluate mode
    triplet = train_loader.dataset.triplets.numpy()
    triplet_list = []
    
    for i in range(triplet.shape[0]):
        triplet_list.append([triplet[i,0], triplet[i,1], triplet[i,2]])

    model.eval()
    expand = 3
    nn = 5
    num_sample = train_loader.dataset.data.shape[0]
    train_loader.dataset.hm = True

    descriptor = []
    pbar = tqdm(enumerate(train_loader))
    for batch_idx, data_a in pbar:
        if args.cuda:
            data_a = data_a.cuda()
        data_a  = Variable(data_a, volatile=True)
        out_a = model(data_a)
        descriptor.append(out_a.data.cpu().numpy())
    descriptor = np.vstack(descriptor)
    print(descriptor.shape)
    
    expand = 3
    nn = 5
    index = nmslib.init(space='l2', method='hnsw')
    index.addDataPointBatch(data=descriptor)
    index.createIndex(print_progress=True, index_params={"maxM": 32, "maxM0": 64, "indexThreadQty": 6})
    index.setQueryTimeParams(params={"ef": nn * expand})
    I = index.knnQueryBatch(queries=descriptor, k=nn, num_threads=6)
    I = np.array(I)
    D = I[:,1,:]
    I = I[:,0,:]
    I = I.astype(int)
    gt_label_mat = np.ones(I.shape, dtype=np.int32)
    label_mat = np.ones(I.shape, dtype=np.int32)
    bad_num = 0
    bad_distance = 0
    all_label = train_loader.dataset.labels.numpy()
    thres = 0.0
    if num_sample < 100000:
        thres = 0.0
    for i in range(num_sample):
        gt_label_mat[i,:] = all_label[i]
        retrieved_label = all_label[I[i,:]]
        label_mat[i,:] = retrieved_label
        bad_num += np.sum(np.absolute(retrieved_label-all_label[i])>100)
        bad_distance += np.sum(D[i,np.absolute(retrieved_label-all_label[i])>100])
        for j in range(retrieved_label.shape[0]):
            if abs(retrieved_label[j]-all_label[i])>100 and D[i,j]>thres:
                if i>0 and all_label[i] == all_label[i-1]:
                    triplet_list.append([i, i-1, I[i,j]]) 
                else:
                    triplet_list.append([i, i+1, I[i,j]])

    triplet_list = np.array(triplet_list)
    #np.savez('../data/{}_triplets.npz'.format(args.training_set),triplet_list)
    print(triplet_list[-5:,:])
    bad_distance_list = D[np.abs(gt_label_mat-label_mat)>100]
    if update_flag:
        train_loader.dataset.triplets = torch.LongTensor(triplet_list)
    print(len(triplet_list))
    print(bad_num)
    print(bad_distance/bad_num)

    train_loader.dataset.hm=False
    if (args.enable_logging):
        logger.log_histogram('Hard Distance',  bad_distance_list, step=epoch)
        logger.log_value('Bad Number', bad_num, step=epoch)
        logger.log_value('Bad Distance', bad_distance/bad_num, step=epoch)
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


def main(train_loader, test_loaders, model, logger, file_logger):
    print('\nparsed options:\n{}\n'.format(vars(args)))

    if args.cuda:
        model.cuda()

    optimizer1 = create_optimizer(model.features, args.lr)

    # optionally resume from a checkpoint
    if args.resume != '':
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
    #if args.hard_mining:
    #        hm(train_loader, model, 0, logger)
    for test_loader in test_loaders:
        test(test_loader['dataloader'], model, 0, logger, test_loader['name'])
    for epoch in range(start, end):
        # iterate over test loaders and test results
        train(train_loader, model, optimizer1, epoch, logger, triplet_flag)
        for test_loader in test_loaders:
            test(test_loader['dataloader'], model, epoch, logger, test_loader['name'])
        if args.hard_mining and (epoch)%5 == 0:
            hm(train_loader, model, epoch, logger, update_flag = True)
        else:
            hm(train_loader, model, epoch, logger, update_flag = False)

if __name__ == '__main__':
    LOG_DIR = args.log_dir
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    LOG_DIR = args.log_dir + suffix
    logger, file_logger = None, None
    model = HardNet()
    if(args.enable_logging):
        
        logger = Logger(LOG_DIR)
        #file_logger = FileLogger(./log/+suffix)
    train_loader, test_loaders = create_loaders(load_random_triplets = triplet_flag)
    main(train_loader, test_loaders, model, logger, file_logger)
