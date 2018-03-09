import os
import errno
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import hickle
import collections
from tqdm import tqdm
import random

#import pdb

class genealogy_journal_test(data.Dataset):
    """`Learning Local Image Descriptors Data <http://phototour.cs.washington.edu/patches/default.htm>`_ Dataset.


    Args:
        root (string): Root directory where images are.
        name (string): Name of the dataset to load.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    def __init__(self, root, name, train=True, transform=None, download=False):
        self.image_dir = '../datasets/original/' 
        self.root = os.path.expanduser(root)
        self.name = name
        self.data_dir = os.path.join(self.image_dir, name)
        self.data_file = os.path.join(self.root, '{}_genealogy_test.pt'.format(name))
        
        self.train = train
        self.transform = transform

        self.mean = 0.4854
        self.std = 0.1864

        self.download()

        if not self._check_datafile_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        # load the serialized data
        self.data, self.image_index, self.trial_index, \
                self.journal_index, self.trial_fast_access = torch.load(self.data_file)
        #print(type(self.data))
        print(self.data.shape)
        print('max_label: {}'.format(self.trial_index.max()))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (data1, data2, matches)
        """
        if self.train:
            data = self.data[index]
            if self.transform is not None:
                data = self.transform(data)
            return data
        m = self.matches[index]
        data1, data2 = self.data[m[0]], self.data[m[1]]
        if self.transform is not None:
            data1 = self.transform(data1)
            data2 = self.transform(data2)
        return data1, data2, m[2]

    def __len__(self):
        if self.train:
            return self.lens[self.name]
        return len(self.matches)

    def _check_datafile_exists(self):
        return os.path.exists(self.data_file)

    def _check_downloaded(self):
        return os.path.exists(self.data_dir)

    def download(self):
        if self._check_datafile_exists():
            print('# Found cached data {}'.format(self.data_file))
            return

        # process and save as torch files
        print('# Caching data {}'.format(self.data_file))

        trial_index, journal_index, image_index = read_info_file(self.image_dir, self.name)
        
        trial_fast_access = create_indices(trial_index)
        dataset = (
            read_image_file(self.image_dir, self.name),
            image_index, 
            trial_index,
            journal_index,
            trial_fast_access
        )

        with open(self.data_file, 'wb') as f:
            torch.save(dataset, f)

def read_image_file(data_dir, dataset_name):
    """Return a Tensor containing the patches
    """
    #patches = np.load(os.path.join(data_dir, dataset_name+'_patch.dat'))
    patches = hickle.load(os.path.join(data_dir, dataset_name+'_genealogy_2_patch.dat'))
    #print(patches.shape)
    return torch.ByteTensor(np.array(patches))


def read_info_file(data_dir, dataset_name):
    """Return a Tensor containing the list of labels
       Read the file and keep only the ID of the 3D point.
    """
    #labels = np.load(os.path.join(data_dir, dataset_name+'_label.dat'))
    #pdb.set_trace()
    #labels = hickle.load(os.path.join(data_dir, dataset_name+'_genealogy_label.dat'))
    trial_index, journal_index_list, image_index_list =\
            hickle.load(os.path.join(data_dir, dataset_name+'_genealogy_2_label.dat'))
    return torch.LongTensor(np.array(trial_index)),\
            torch.LongTensor(np.array(journal_index_list)),\
            torch.LongTensor(np.array(image_index_list))

def create_indices(labels):
    old = labels[0]
    indices = dict()
    indices[old] = 0
    for x in xrange(len(labels) - 1):
        new = labels[x + 1]
        if old != new:
            indices[new] = x + 1
        old = new
    return indices
