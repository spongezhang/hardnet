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

class synthesized_journal(data.Dataset):
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
        self.image_dir = '/dvmm-filer2/users/xuzhang/Medifor/data/all_journal/data/' 
        self.root = os.path.expanduser(root)
        self.name = name
        self.data_dir = os.path.join(self.image_dir, name)
        self.data_file = os.path.join(self.root, '{}.pt'.format(name))
        
        self.train = train
        self.transform = transform

        self.mean = 0.4854
        self.std = 0.1864

        self.download()

        if not self._check_datafile_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        # load the serialized data
        self.data, self.labels, self.matches = torch.load(self.data_file)
        #print(type(self.data))
        print(self.data.shape)
        print('max_label: {}'.format(self.labels.max()))

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

        dataset = (
            read_image_file(self.image_dir, self.name),
            read_info_file(self.image_dir, self.name),
            read_matches_files(self.image_dir, self.name)
        )

        with open(self.data_file, 'wb') as f:
            torch.save(dataset, f)

def read_image_file(data_dir, dataset_name):
    """Return a Tensor containing the patches
    """
    #patches = np.load(os.path.join(data_dir, dataset_name+'_patch.dat'))
    patches = hickle.load(os.path.join(data_dir, dataset_name+'_patch.dat'))
    #print(patches.shape)
    return torch.ByteTensor(np.array(patches))


def read_info_file(data_dir, dataset_name):
    """Return a Tensor containing the list of labels
       Read the file and keep only the ID of the 3D point.
    """
    labels = np.load(os.path.join(data_dir, dataset_name+'_label.dat'))
    return torch.LongTensor(labels)


def read_matches_files(data_dir, dataset_name):
    """Return a Tensor containing the ground truth matches
       Read the file and keep only 3D point ID.
       Matches are represented with a 1, non matches with a 0.
    """
    matches = []
    try:
        os.stat(data_dir, dataset_name + '_match.txt')
    except:
        labels = np.load(os.path.join(data_dir, dataset_name+'_label.dat'))
        generate_matches_one_by_one(labels, data_dir, dataset_name, 100000)

    with open(os.path.join(data_dir, dataset_name + '_match.txt'), 'r') as f:
        for line in f:
            l = line.split()
            matches.append([int(l[0]), int(l[2]), int(l[1] == l[3])])
    return torch.LongTensor(matches)

def generate_matches_one_by_one(labels, base_dir, dataset_name, n_samples):
    # group labels in order to have O(1) search
    count = collections.Counter(labels)
    # index the labels in order to have O(1) search
    indices = create_indices(labels)

    # range for the sampling
    labels_size = len(labels) - 1
    
    # generate the matches
    pbar = tqdm(xrange(len(count)))

    match_file = os.path.join(base_dir, dataset_name+'_match.txt')
    f = open(match_file, 'w')
    
    for x in pbar:
        pbar.set_description('Generating triplets')
        num_samples = count[labels[x]]
        begin_positives = indices[labels[x]]

        #offset_a, offset_p = random.sample(xrange(num_samples), 2)
        offset_a = 0
        offset_p = num_samples - 1
        
        idx_a = begin_positives + offset_a
        idx_p = begin_positives + offset_p

        idx_n = random.randint(0, labels_size)
        while labels[idx_n] == labels[idx_a] and \
                        labels[idx_n] == labels[idx_p]:
            idx_n = random.randint(0, labels_size)
        f.write('{:d} {:d} {:d} {:d}\n'.format(idx_p, labels[idx_p], idx_a, labels[idx_a]))
        f.write('{:d} {:d} {:d} {:d}\n'.format(idx_p, labels[idx_p], idx_n, labels[idx_n]))
    f.close()

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
