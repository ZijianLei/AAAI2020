#ain__':
#     parser = argparse.ArgumentParser(description='choose the dataset and training parameter')
#     parser.add_argument('-d', action='store', dest='data', help='choose the dataset', default=False)
#     parser.add_argument('-m', action='store', dest='landmark', help='choose the landmark point', default=False)
#     parser.add_argument('-l', action='store', dest='lambda', help='paremeter for l2 regularzation', default=False)
#     parser.add_argument('-g', action='store', dest='gamma', help='gamma for the rbf kernel', default=False)
#     param = parser.parse_args()
#     np.set_printoptions(threshold=np.inf, suppress=True)
#     name = '%s' % (param.data)
#     train_url = "/home/comp/cszjlei/svm/BudgetedSVM/original/%s/train1" % (name)
#     train, tra_label = load_svmlight_file(train_url)
#     transformer = sklearn.random_projection.SparseRandomProjection(n_components=10)
#     train = transformer.fit_transform(train)
#     print(train.shape)
#     matrix = transformer.components_
#     matrix = matrix.todense()
#     print(matrix)
#     X_train, X_test, y_train, y_test = train_test_split(train, tra_label, test_size = 0.33)
    # dump_svmlight_file(X_train,y_train,'svm/BudgetedSVM/original/%s/%s'%(name,'train'),zero_based=False)
    # dump_svmlight_file(X_test, y_test, 'svm/BudgetedSVM/original/%s/%s' % (name, 'test'), zero_based=Fal
import os
import subprocess
from read_dataset import  *
import gc
from multiprocessing import Pool, Queue
import numpy as np
from sklearn.datasets import dump_svmlight_file,load_svmlight_file
import  scipy.sparse
from generate_data import generater
from functools import  partial
from sklearn.datasets import dump_svmlight_file
from sklearn.cluster import  KMeans
from sklearn import  preprocessing
import pandas
from collections import Counter
import  sklearn
import time
import  matplotlib.pyplot as plt
from scipy.sparse.linalg import  *
# def random(data, d_sub, name, what, R,n_label):
#     test_url = "/home/comp/cszjlei/svm/BudgetedSVM/original/%s/train_%d" % (name, 0)
#     # train_url = "/home/comp/cszjlei/svm/BudgetedSVM/original/%s/train" % name
#     data, tra_label = load_svmlight_file(test_url)
#     label = np.mat(tra_label).T
#     n_sample, d_feature = np.shape(data)
#     data = data.todense()
#     norm = np.linalg.norm(data, axis=0)
#     norm =  sorted(norm)
#     plt.legend(fontsize=15)
#     plt.title("%s" % d, fontsize=20)
#     plt.xlabel("dimension", fontsize=15)
#     plt.ylabel("accuracy", fontsize=15)

if __name__ == '__main__':
    dir1 = 'other_method/kmeans_linear/'
    data = ['usps','mushrooms','gisette']
    data0 = ['mushrooms','usps','rcv','a9a','real-sim']
    data1 = ['usps','mushrooms']
    method = ['gaussian','sparse','norm','sparse_embedding','random','laplacian']
    # method = ['random','norm','srht_leverage_score]
    f_sub = [2**(-6), 2**(-5), 2**(-4), 2**(-3), 2**(-2), 2**(-1)]
    what = 1
    flag = 0 # 0 for plot the accuracy and 1 for the relationship with the parameter
    for d_idx,d in enumerate(data):
        print(d)
        # test_url = "/home/comp/cszjlei/svm/BudgetedSVM/original/%s/test_%d" % (d, 0)
        test_url = "/home/comp/cszjlei/svm/BudgetedSVM/original/%s/test" % d
        data, tra_label = load_svmlight_file(test_url)
        data = data.todense()
        n,d_feature = np.shape(data)
        u,s,vt = svds(data,k=64)
        # print(vt)
        # score = np.sum(vt**2,axis=0)
        # norm = score
        # print(score)
        norm = np.linalg.norm(vt, axis=0)
        # print(norm.shape)
        norm = norm / sum(norm)
        norm = sorted(norm)
        # print(norm.shape)
        # print(norm)
        # R = norm**2/np.sum(norm**2)
        # norm = sorted(R)
        plt.plot([x for x in range(d_feature)], norm, linewidth=3, linestyle='--', marker='o', label='original')
        train, train_lable, test, test_lable = generater(d)
        # # label = np.mat(tra_label).T
        n_sample, d_feature = np.shape(test)
        # # print(test.shape)
        # # exit()
        # test = test.todense()
        u,s,vt = svds(test,k = 64)
        norm = np.linalg.norm(vt, axis=0)
        norm = norm/sum(norm)
        # # print(norm.shape)
        norm = sorted(norm)
        plt.plot([x for x in range(d_feature)],norm, linewidth=3, linestyle='--',marker='o',label = 'transform')


        # plt.hist(norm,50)
        plt.legend(fontsize=15)
        plt.title("%s" % d, fontsize=20)
        plt.xlabel("dimension", fontsize=15)
        plt.ylabel("norm", fontsize=15)
        plt.savefig('other_method/%s_%s.png' % (d, 'weight'))
        plt.show()
        # time.sleep(2)
        # plt.close()