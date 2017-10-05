
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

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
    print >> sys.stderr, """hahaha!"""
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

parser = argparse.ArgumentParser()

#gpu_set = ['0']
#parameter_set = ['0.01']
gpu_set = ['0','1','2','3']
parameter_set = ['0.0', '0.01', '0.1', '1.0']
#parameter_set = ['0.0','1.0','5.0','10.0','20.0','50.0','100.0','200']
number_gpu = len(gpu_set)

process_set = []
for idx, parameter in enumerate(parameter_set):
    print('Test Parameter: {}'.format(parameter))
     
    #command = 'python patch_network_train_triplet.py --training notredame --test liberty --learning_rate 0.1 --num_epoch 10 --loss_type 1 --alpha 0 --beta {} --gpu_ind {} --no_normalization'\
    #        .format(parameter,gpu_set[idx%number_gpu])
    
    command = 'python triplet.py --Loss_Type 3 --epochs 50 --margin 0.5 --alpha {} --gpu-id {}'\
            .format(parameter,gpu_set[idx%number_gpu])
    
    print(command)
    p = subprocess.Popen(shlex.split(command))#, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    #(stdoutput,erroutput) = p.communicate()
    process_set.append(p)
    #time.sleep(1)
    
    if (idx+1)%number_gpu == 0:
        print('Wait for process end')
        for sub_process in process_set:
            sub_process.wait()
    
        process_set = []

for sub_process in process_set:
    sub_process.wait()

