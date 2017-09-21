#! /usr/bin/env python2

import numpy as np
import time
import os
import pandas as pd
import sys
import scipy.io as sio
import subprocess
import json
import marshal as pickle
from copy import deepcopy
import cyvlfeat
import exifread
import cv2

#sift = cv2.SIFT(nfeatures=1000)
#sift = cv2.SIFT(contrastThreshold=0.132)
sift = None#cv2.xfeatures2d.SIFT_create(contrastThreshold=0.12)
standard_size = 64
num_channels = 3
feature_type = 1

def err_quit(msg, exit_status=1):
    print(msg)
    exit(exit_status)

def load_csv(csv_fn, sep="|"):
    try:
        return pd.read_csv(csv_fn, sep)
    except IOError as ioerr:
        err_quit("{}. Aborting!".format(ioerr))

def read_image_from_name(root_dir, file_name):
    """Return the factorial of n, an exact integer >= 0. Image rescaled to no larger than 1024*768
    
    Args:
        root_dir (str): Image directory
        filename (str): Name of the file

    Returns:
        np array: Image 
        float: Rescale ratio

    """
    img = cv2.imread(root_dir + file_name)
    if img.shape[2] == 4 :
        img = img[:,:,:3]
    if img.shape[2] == 3 :
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ftest = open(root_dir + file_name, 'rb')
    tags = exifread.process_file(ftest)
    
    try:
        if str(tags['Thumbnail Orientation']) == 'Rotated 90 CW':
            img = cv2.transpose(img)  
            img = cv2.flip(img, 1)
        elif str(tags['Thumbnail Orientation']) == 'Rotated 90 CCW':
            img = cv2.transpose(img)  
            img = cv2.flip(img, 0)
        elif str(tags['Thumbnail Orientation']) == 'Rotated 180':
            img = cv2.flip(img, -1)
    except:
        tags = tags

    ratio = 1.0
    if img.shape[0]*img.shape[1]>1024*768:
        ratio = (1024*768/float(img.shape[0]*img.shape[1]))**(0.5)
        img = cv2.resize(img,(int(img.shape[1]*ratio), int(img.shape[0]*ratio)),interpolation = cv2.INTER_CUBIC);
    return img, ratio

def read_color_image_from_name(root_dir,file_name):
    
    img = cv2.imread(root_dir + file_name)
    if img.shape[2] == 1 :
        img = np.repeat(img, 3, axis = 2)
    if img.shape[2] == 4 :
        img = img[:,:,:3]

    ftest = open(root_dir + file_name, 'rb')
    tags = exifread.process_file(ftest)
    
    try:
        if str(tags['Thumbnail Orientation']) == 'Rotated 90 CW':
            img = cv2.transpose(img)  
            img = cv2.flip(img, 1)
        elif str(tags['Thumbnail Orientation']) == 'Rotated 90 CCW':
            img = cv2.transpose(img)  
            img = cv2.flip(img, 0)
        elif str(tags['Thumbnail Orientation']) == 'Rotated 180':
            img = cv2.flip(img, -1)
    except:
        tags = tags
        
    ratio = 1.0
    if img.shape[0]*img.shape[1]>1024*768:
        ratio = (1024*768/float(img.shape[0]*img.shape[1]))**(0.5)
        img = cv2.resize(img,(int(img.shape[1]*ratio), int(img.shape[0]*ratio)),interpolation = cv2.INTER_CUBIC);

    return img, ratio

def match_and_insert(donor_kp_list, donor_desc_list, donor_patch_list,\
        base_image, base_kp, base_desc, transform_matrix):
    for kp,desc in zip(base_kp,base_desc):
        mapped_kp = map_kp(kp,transform_matrix)
        min_distance = 1000
        min_index = -1
        for idx, d_kp_list in enumerate(donor_kp_list):
            donor_kp = d_kp_list[0]
            match_flag,distance = match_two_kp(mapped_kp, donor_kp)
            if match_flag and distance<min_distance:
                min_index = idx
                min_distance = distance
                #print((donor_kp.pt,kp.pt))
        if min_index>=0:
            donor_kp_list[min_index].append(kp)
            donor_desc_list[min_index].append(desc)
            tmp_patch = extract_patch(base_image, kp)
            donor_patch_list[min_index].append(tmp_patch)

def extract_sift(img, kp_list = None):
    if kp_list == None:
        if feature_type == 0:
            kp, des = sift.detectAndCompute(img,None)
        elif feature_type == 1:
            kp, des = cyvlfeat.sift.sift(img, peak_thresh=3.0, compute_descriptor=True)
    else:
        if feature_type == 0:
            kp, des = sift.compute(img,kp_list)
        elif feature_type == 1:
            kp, des = cyvlfeat.sift.sift(img, peak_thresh=3.0,\
                    compute_descriptor=True, frames=np.array(kp_list))
            
    if feature_type>0 :
        kp = kp.tolist()
    return kp,des

#def extract_sift(img, kp_list = None):
#    if kp_list == None:
        #kp, des = sift.detectAndCompute(img,None)
#        kp, des = cyvlfeat.sift.sift(img, peak_thresh=3.0, compute_descriptor=True)
#    else:
#        kp, des = cyvlfeat.sift.sift(img, peak_thresh=3.0, compute_descriptor=True, frames=np.array(kp_list))
#    kp = kp.tolist()
#    return kp,des

    #if kp_list == None:
    #    kp, des = sift.detectAndCompute(img,None)
    #else:
    #    kp, des = sift.compute(img,kp_list)
    #    
    #return kp,des

def load_csv(csv_fn, sep="|"):
    try:
        return pd.read_csv(csv_fn, sep)
    except IOError as ioerr:
        err_quit("{}. Aborting!".format(ioerr))

def rectify_patch(img, kp, patch_sz):
    if feature_type == 0:
        scale = 1.0 #rotate in the patch
        M = cv2.getRotationMatrix2D((patch_sz/2,patch_sz/2),  -1*kp[3]*180/3.1415, scale )
    elif feature_type == 1:
        scale = 1.0 #rotate in the patch
        M = cv2.getRotationMatrix2D((patch_sz/2,patch_sz/2), -1*kp[3]*180/3.1415, scale)
       # print(M)

    rot = cv2.warpAffine(img, np.float32(M), (patch_sz, patch_sz), \
          flags = cv2.WARP_INVERSE_MAP + cv2.INTER_CUBIC) #+ cv2.WARP_FILL_OUTLIERS

    return rot

def extract_patch(img, kp):
    if feature_type == 0:
        sub = cv2.getRectSubPix(img, (int(kp.size/2*standard_size),\
            int(kp.size/2*standard_size)), kp.pt)
    elif feature_type == 1:            
        sub = cv2.getRectSubPix(img, (int(kp[2]/2*standard_size),\
                int(kp[2]/2*standard_size)), (kp[1],kp[0]))
    res = cv2.resize(sub, (standard_size, standard_size))
    res = rectify_patch(res, kp, standard_size)
    return np.asarray(res)
