"""
Check the correctness of gor on HardNet loss using multiple GPUs
Usage: check_gor_HardNet.py

Author: Xu Zhang
Email: xu.zhang@columbia.edu.cn
"""

#! /usr/bin/env python2

import numpy as np
import scipy.io as sio
import time
import os
import sys
import pandas as pd
import subprocess
import shlex
import argparse
####################################################################
# Parse command line
####################################################################
def usage():
    print >> sys.stderr 
    sys.exit(1)

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

gpu_set = ['0']
parameter_set = [' --donor --data_augment ']#, ' --data_augment ', ' --donor ', ' '
number_gpu = len(gpu_set)

#datasets = ['notredame', 'yosemite', 'liberty']
datasets = ['MFC18_Dev1_Ver2_bg']
process_set = []


for dataset in datasets:
    for idx, parameter in enumerate(parameter_set):
        print('Test Parameter: {}'.format(parameter))
        command = 'python HardNet_provenance.py --training-set {} --fliprot=False --n-triplets=1000000 --lr=0.01 --resume=../provenance_model/synthesized_journals_2_train_min_triplet_margin_as_da_do/checkpoint_10.pth --batch-size=128 --epochs 10 --gor=False {} --w1bsroot=None --gpu-id {} --log-dir ../provenance_log/ --enable-logging=True --batch-reduce=min '\
                .format(dataset, parameter, gpu_set[idx%number_gpu])
    
        print(command)
        p = subprocess.Popen(shlex.split(command))
        process_set.append(p)
        
        if (idx+1)%number_gpu == 0:
            print('Wait for process end')
            for sub_process in process_set:
                sub_process.wait()
        
            process_set = []
    
        time.sleep(60)
    
    for sub_process in process_set:
        sub_process.wait()

