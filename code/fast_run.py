"""
Check the correctness of gor on triple loss using multiple GPUs
Usage: check_gor_triplet.py

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

gpu_set = ['0','1']
#parameter_set = ['--no-mask', '--no-hinge', '--inner-product', '--no-hinge, --no-mask']
number_gpu = len(gpu_set)

#parameter_set = [' --inner-product --loss=softmax', '--loss=classification']#, 
parameter_set = [1.0, 2.0, 3.0, 3.5, 4.0, 5.0]#, 
datasets = ['notredame']
process_set = []


for dataset in datasets:
    for idx, parameter in enumerate(parameter_set):
        print('Test Parameter: {}'.format(parameter))
        command = 'python HardNet.py --training-set {} --fliprot=False --n-triplets=1000000 --batch-size=128 --epochs 5 --w1bsroot=None --loss=classification --margin={} --log-dir ../ubc_log/ --enable-logging=True --batch-reduce=min --model-dir ../ubc_model/ --gpu-id {}' \
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

