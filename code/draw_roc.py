import numpy as np
import scipy.io as sio
import pickle
import operator
import math
import matplotlib as mpl
import os

if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import plotly.plotly as py

def ROC_Curve(labels, scores, fpr_point):
    recall_point = []

    # Sort label-score tuples by the score in descending order.
    sorted_scores = zip(labels, scores)
    sorted_scores.sort(key=operator.itemgetter(1), reverse=False)
    
    # Compute error rate
    n_match = sum(1 for x in sorted_scores if x[0] == 1)

    tp = 0
    count = 0
    index = 0
    all_N = sum(labels==0)
    for label, score in sorted_scores:
        count += 1
        if label == 1:
            tp += 1
        fpr = float(count - tp) / all_N

        if fpr >= fpr_point[index]:
            recall_point.append((float(tp)/n_match))
            index = index+1
            if index>=fpr_point.shape[0]:
                break
    return recall_point


if __name__ == "__main__":
    setting = 'notredame_yosemite_notredame_random_global'
    method_list = [setting+'_as', setting+'_gor_alpha1.0_as']
    method_label = ['Triplet','Triplet+GOR(Ours)']
    recall_list = []
    distance_dir = '../distance_mat/'
    file_list = os.listdir(distance_dir+method_list[0])
    fig = plt.figure()
    for file_t in file_list:
        if file_t.endswith('.mat'):
            #print(file_t)
            #file_t = 'epoch_01_iter_1000.mat' 
            recall_list = []
            epoch = int(file_t[6:8])
            iter_num = int(file_t[14:18])
            for method in method_list: 
                file_name = distance_dir + method + '/' + file_t
                #file_name = '../data/photoTour/result/dis_mat_{}/{}_{}_step_10.mat'.format(k,training_name,test_name)
                #file_name = './epoch_00_iter_0000.mat'.format(k,training_name,test_name)
                mat = sio.loadmat(file_name)
                mat = mat['save_object']
                dists = mat[0][0][0]
                labels = mat[0][1][0]
                #print(dists)
                #print(labels)
                fpr_point = np.arange(0.01,0.5,0.01)
                recall = ROC_Curve(labels,dists,fpr_point)
                recall_list.append(recall)
            SMALL_SIZE = 14
            MEDIUM_SIZE = 20
            BIGGER_SIZE = 24

            plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
            plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
            plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
            plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
            plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
            plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
            plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title 
            
            
            plt.plot(fpr_point, recall_list[0], 'b', fpr_point, recall_list[1], 'r')
            plt.suptitle('Epoch: {:02d} Iter: {:04d}'.format(epoch, iter_num), fontsize=24)
            plt.ylim([0.0,1.0])
            plt.xlim([0,0.5])
            plt.xlabel('FPR(%)')
            plt.ylabel('TPR(%)')
            plt.legend(method_label)
            plt.legend(loc='bottom right')
            try:
                os.stat('../roc/'+setting)
            except:
                os.makedirs('../roc/'+setting)

            plt.savefig('../roc/'+setting+'/'+file_t[0:-4]+'.png', bbox_inches='tight')
            plt.clf()
            #break
            #plt.show()
