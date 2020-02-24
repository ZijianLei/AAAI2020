import os
import subprocess
from read_dataset import *
import gc
from multiprocessing import Pool, Queue
import numpy as np
from sklearn.datasets import dump_svmlight_file, load_svmlight_file
import scipy.sparse
from functools import partial
from sklearn.datasets import dump_svmlight_file
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas
from collections import Counter
import sklearn
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dir1 = 'other_method/kmeans_linear/'
    data = ['gisette', 'news', 'rcv', 'real-sim', 'usps', 'mushrooms']
    # data0 = ['usps','gisette','rcv','news']
    # data = ['real-sim']
    method = ['gaussian', 'sparse', 'sparse_embedding', 'random', 'weight_sampling', 'norm', 'laplacian',]
    # method = ['random','laplacian']
    f_sub = [2 ** (-6), 2 ** (-5), 2 ** (-4), 2 ** (-3), 2 ** (-2), 2 ** (-1)]
    baseline = [97.31, 95.01, 95.42, 97.43, 91.13, 99.85]
    what = 1
    flag = 0  # 0 for plot the accuracy and 1 for thline relationship with the parameter
    for d_idx, d in enumerate(data):
        print(d)
        train_url = "/home/comp/cszjlei/svm/BudgetedSVM/original/%s/test_0" % (d)

        # # train_url = "/home/comp/cszjlei/svm/BudgetedSVM/original/%s/train" % d
        train1, tra_label = load_svmlight_file(train_url)
        n_number, n_feature = np.shape(train1)
        acc_max_of_each_round = np.zeros((8, 6))
        acc_average_of_each_iteration = np.zeros((2, 15))
        acc_average_of_each_parameter = np.zeros((8, 6))
        for idx, m in enumerate(method):
            result = np.load('other_method/kmeans_linear/%s/%s_result.npy' % (
            d, m))  # 3 dimension data sub dimension, round,accuracy 6*10*13
            # print(result)
            # print(m)
            # print(result)
            for i in range(6):
                temp = result[i, :5]
                # if m == 'weight_sampling':
                #     print(temp)
                # print(temp.shape)
                # print(temp)
                # print(np.amax(temp,axis = 1))
                mask = np.all(np.isnan(temp) | np.equal(temp, 0), axis=1)
                # print(np.amax(temp[~mask],axis=1))
                # print(temp[:,np.argmax(np.mean(temp[~mask],axis=0))])
                # exit()
                acc_max_of_each_round[idx, i] = np.std(np.amax(temp[~mask], axis=1))
                # acc_max_of_each_round[idx,i] = np.std(temp[:,np.argmax(np.mean(temp[~mask],axis=0))])
                acc_average_of_each_parameter[idx, i] = np.mean(np.amax(temp[~mask], axis=1))
                # acc_average_of_each_parameter[idx, i] = np.mean(temp[~mask],axis=0)

                # if i == 2:
                #     acc_average_of_each_iteration[idx] = np.amax(temp[~mask],axis=1)
                #     plt.plot([int(x ) for x in range(15)], acc_average_of_each_iteration[idx], linewidth=3,
                #              linestyle='--', marker='o', label='%s' % m)

            if m == 'laplacian':
                m = 'ISRHT-supervised'
                plt.plot([int(x * n_feature) for x in f_sub[:5]], acc_average_of_each_parameter[idx, :5], linewidth=3,
                         linestyle='--', marker='v', label='%s' % m, markersize=10)
            if m == 'norm':
                m = 'ISRHT-top-r'
                plt.plot([int(x * n_feature) for x in f_sub[:5]], acc_average_of_each_parameter[idx, :5], linewidth=3,
                         linestyle='--', marker='^', label='%s' % m, markersize=10)
            if m == 'sparse':
                m = 'Achliotas'
                plt.plot([int(x * n_feature) for x in f_sub[:5]], acc_average_of_each_parameter[idx, :5], linewidth=3,
                         linestyle='--', marker='.', label='%s' % m, markersize=10)
            if m == 'random':
                m = 'SRHT'
                plt.plot([int(x * n_feature) for x in f_sub[:5]], acc_average_of_each_parameter[idx, :5], linewidth=3,
                         linestyle='--', marker='o', label='%s' % m, markersize=10)
            if m == 'gaussian':
                m = 'Gaussian'
                plt.plot([int(x * n_feature) for x in f_sub[:5]], acc_average_of_each_parameter[idx, :5], linewidth=3,
                         linestyle='--', marker='.', label='%s' % m, markersize=10)
            if m == 'sparse_embedding':
                m = 'Sparse Embedding'
                plt.plot([int(x * n_feature) for x in f_sub[:5]], acc_average_of_each_parameter[idx, :5], linewidth=3,
                         linestyle='--', marker='.', label='%s' % m, markersize=10)
            if m == 'weight_sampling':
                m = 'ISRHT-nps'
                plt.plot([int(x * n_feature) for x in f_sub[:5]], acc_average_of_each_parameter[idx, :5], linewidth=3,
                         linestyle='--', marker='*', label='%s' % m, markersize=10)
            # plt.plot([int(x*n_feature) for x in f_sub[:5]],acc_average_of_each_parameter[idx,:5],linewidth= 3, linestyle = '--', marker = 'o', label='%s'%m)
            # else:
        if flag == 1:
            plt.plot([x for x in range(100)], [y for y in range(100)], linewidth=3)
            plt.scatter(acc_average_of_each_parameter[0], acc_average_of_each_parameter[1], marker='o')
            # plt.legend(fontsize=15)
            plt.title("%s" % d, fontsize=20)
            plt.xlabel("SRHT", fontsize=15)
            plt.ylabel("ISRHT-supervised", fontsize=15)
            plt.tight_layout()
            # if what == 1:
            # plt.savefig('other_method/%s_%s.png' %(d,'random_laplacian'))
            # else:
            #     plt.savefig('other_method/%s_%s.eps' % (d, 'average_of_each_parameter'))
            # plt.subplot(211)
            plt.show()

        if flag == 0:
            original = np.zeros((1, 5))
            a = baseline[d_idx]
            original[0,:] = a
            # plt.plot([x for x in range(100)], [y for y in range(100)],linewidth = 3)
            # plt.scatter(acc_average_of_each_parameter[0], acc_average_of_each_parameter[1], marker='o')
            # result = np.load('other_method/kmeans_linear/%s/%s_result.npy' % (
            #     d, 'pca'))
            # print (result)
            # plt.plot([int(x * n_feature) for x in f_sub[:5]], result[ :5], linewidth=3,
            #          linestyle='--', marker='.', label='%s' % 'pca', markersize=10)
            #plt.plot([int(x * n_feature) for x in f_sub[:5]], original[0,:], linewidth=2,
            #         linestyle='--', label='%s' % 'liblinear', markersize=10)
            plt.legend(fontsize=10)
            plt.title("%s" % d, fontsize=20)

            plt.xlabel("dimension", fontsize=15)
            plt.ylabel("accuracy", fontsize=15)
            plt.tight_layout()
            # if what == 1:
            plt.savefig('other_method/%s_%s.png' %(d,'accuracy'))
            # else:
            #     plt.savefig('other_method/%s_%s.eps' % (d, 'average_of_each_parameter'))
            plt.show()
            # m = d_idx +1
            # plt.subplot(2,3,m)
            # print(np.around(acc_max_of_each_round,decimals=4))
            print (np.around(acc_average_of_each_parameter, decimals=2))
        if flag == 2:
            plt.legend(fontsize=15)
            plt.title("%s" % d, fontsize=20)
            plt.xlabel("dimension", fontsize=15)
            plt.ylabel("std_error", fontsize=15)
            plt.tight_layout()
            # if what == 1:
            plt.savefig('other_method/%s_%s.png' % (d, 'different_round'))
            # else:
            #     plt.savefig('other_method/%s_%s.eps' % (d, 'average_of_each_parameter'))
            plt.show()
    # plt.show()



