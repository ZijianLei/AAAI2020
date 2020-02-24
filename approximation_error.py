import os
import subprocess
import multiprocessing
from AAAI2020.read_dataset  import *
import math
import gc
from multiprocessing import Pool, Queue,Manager
import numpy as np
import scipy.sparse
from functools import partial
from sklearn.datasets import dump_svmlight_file,load_svmlight_file
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.decomposition import PCA
import  matplotlib.pyplot as plt
from liblinear.python import liblinearutil
import argparse
from numba import jit
from scipy.sparse.linalg import *
import generate_data
def random(data, d_sub, name, what, array, R):
    np.set_printoptions(threshold=np.inf, suppress=True)
    # data = scipy.sparse.coo_matrix.tocsr(data)
    label = np.where(data[:, 0] != 1, -1, 1)
    # label = data[:,0]
    data = data[:, 1:]
    # scaler = preprocessing.StandardScaler(with_std=False).fit(data)
    n_sample, d_feature = np.shape(data)
    a = math.sqrt(float(d_feature / d_sub)) * data

    train2 = a[:, R]
    label = (np.asarray(label)).reshape(-1)
    dump_svmlight_file(np.around(train2, decimals=4), label, 'other_method/kmeans_linear/%s/%s_uniform_scale' % (name, what),
                       zero_based=False)
    n_sample, d_feature = np.shape(data)
    a =  data

    train2 = a[:, R]
    label = (np.asarray(label)).reshape(-1)
    dump_svmlight_file(np.around(train2, decimals=4), label,
                       'other_method/kmeans_linear/%s/%s_uniform_no_scale' % (name, what),
                       zero_based=False)
    #
    np.set_printoptions(threshold=np.inf, suppress=True)
    train_url = "/home/comp/cszjlei/svm/BudgetedSVM/original/%s/test_0" % (name)
    data, tra_label = load_svmlight_file(train_url)
    tra_label = np.mat(tra_label).T
    label = np.where(tra_label[:, 0] != 1, -1, 1)
    u, s, vt = svds(data)
    norm = np.linalg.norm(vt, axis=0)
    # p = norm ** 2 / np.sum(norm ** 2)
    # R_norm = np.random.choice(data.shape[1], d_sub, p=p, replace=False)
    topn = np.argsort(-norm, axis=0)
    topn = np.reshape(topn, (1, -1))
    topn = np.asarray(topn)
    R_norm = topn[0, :d_sub]
    R_norm = sorted(R_norm)
    train2 = data[:, R_norm]
    label = (np.asarray(label)).reshape(-1)
    dump_svmlight_file(np.around(train2, decimals=4), label,
                       'other_method/kmeans_linear/%s/%s_top_r' % (name, what), zero_based=False)

    np.set_printoptions(threshold=np.inf, suppress=True)
    train_url = "/home/comp/cszjlei/svm/BudgetedSVM/original/%s/test_0" % (name)
    data, tra_label = load_svmlight_file(train_url)
    tra_label = np.mat(tra_label).T
    label = np.where(tra_label[:, 0] != 1, -1, 1)
    u, s, vt = svds(data)
    norm = np.linalg.norm(vt, axis=0)
    p = norm ** 2 / np.sum(norm ** 2)
    R_norm = np.random.choice(data.shape[1],d_sub,p = p,replace=False)
    R_norm = sorted(R_norm)
    train2 = data[:, R_norm]
    label = (np.asarray(label)).reshape(-1)
    dump_svmlight_file(np.around(train2, decimals=4), label,
                       'other_method/kmeans_linear/%s/%s_NPS_no_scale' % (name, what), zero_based=False)



    np.set_printoptions(threshold=np.inf, suppress=True)
    train_url = "/home/comp/cszjlei/svm/BudgetedSVM/original/%s/test_0" % (name)
    data, tra_label = load_svmlight_file(train_url)
    te_label = np.mat(tra_label).T
    np.set_printoptions(threshold=np.inf, suppress=True)
    label = np.where(te_label[:, 0] != 1, -1, 1)
    u, s, vt = svds(data)
    norm = np.linalg.norm(vt, axis=0)
    p = norm ** 2 / np.sum(norm ** 2)
    R_norm = np.random.choice(data.shape[1],d_sub,p = p,replace=False)
    train2 = data[:, R_norm] / np.sqrt(d_sub * p[R_norm])
    label = (np.asarray(label)).reshape(-1)
    dump_svmlight_file(np.around(train2, decimals=4), label,
                       'other_method/kmeans_linear/%s/%s_NPS_scale' % (name, what), zero_based=False)



def read_original(name,n,f_s):
    train_url = "/home/comp/cszjlei/svm/BudgetedSVM/original/%s/test_0" % (name)
    train1,tra_label  = load_svmlight_file(train_url)
    tra_label = np.mat(tra_label).T
    n_data, n_feature = np.shape(train1)
    train = scipy.sparse.hstack((tra_label, train1))
    train = train.todense()
    sub = int(math.pow(2, f_s) * n_feature)
    array = np.mat(np.random.uniform(-1, 1, n_feature))  # -1 +1均匀采样
    R = np.random.choice(int(train.shape[1]) - 1, int(sub), replace=False)
    array[array < 0] = -1
    array[array > 0] = 1
    R = sorted(R)
    train1 = train
    random(data=train1, d_sub=sub,name=name, what='test', array=array, R=R)
    return n_feature
    # sampling(data = train1,d_sub = sub,name  = name, what =  'train',tradeoff = trad)
    # corr(data=train1, d_sub=sub, name=name, what='train', array=array, R=R)

def read_c(re):
    cj, cc, cn = re
    c_average[cj,cn] = cc
    # print(c,j,n)
def read_acc(rea):
    aidx, aa, an =rea
    acc[aidx,an] = aa
def norm_test(idx,meth):
    scaling_param = 1
    x0,y = load_svmlight_file("/home/comp/cszjlei/svm/BudgetedSVM/original/%s/test_0" % (name))
    x0= x0.todense()
    x1,y = load_svmlight_file("other_method/kmeans_linear/%s/test_%s" % (name, meth))
    x1 = x1.todense()
    if meth =='top_k':
        scaling_param = np.trace(np.dot(x0.T,x0))/np.trace(np.dot(x1.T,x1))
        # scaling_param = np.dot(x0[0],x0[0].T)/np.dot(x1[0],x1[0].T)
        print(scaling_param)
    acc[idx,f_idx,n] = np.linalg.norm(np.dot(x0,x0.T)-scaling_param*np.dot(x1,x1.T),ord='fro')/np.linalg.norm(np.dot(x0,x0.T),ord='fro')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='choose the dataset and training parameter')
    parser.add_argument('-d', action='store', dest='data', help='choose the dataset', default=False)
    parser.add_argument('-m', action='store', dest='landmark', help='choose the landmark point', default=False)
    parser.add_argument('-l', action='store', dest='lambda', help='paremeter for l2 regularzation', default=False)
    parser.add_argument('-g', action='store', dest='gamma', help='gamma for the rbf kernel', default=False)
    param = parser.parse_args()
    # method = ['random','srht_leverage_score','leverage_score']
    method = ['uniform_scale','uniform_no_scale','NPS_scale','NPS_no_scale','top_r']
    method = ['uniform_no_scale','NPS_no_scale','top_r']
    np.set_printoptions(threshold=np.inf, suppress=True)
    # method = ['corr','nonuniform']
    tradeoff = [0.2,0.4,0.6,0.8,1]
    trad = 0.5
    f_sub = [-6,-5,-4, -3, -2,-1]
    # f_sub = [-10,-9,-8,-7,-6]
    cost = [2**(-5),2**(-4),2**(-3),2**(-2),2**(-1),1,2,4,8,16,32]
    name = '%s'%(param.data)
    c_average = np.empty((5,5))
    acc = np.empty((5,len(f_sub),5))
    norm_average = np.empty((5,len(f_sub)))

    for f_idx,f_s in enumerate(f_sub):
        # print( 'the ensemble dimension space is 1/%d' %(math.pow(2,-f_s)))
        for n in range(5):
            n_feature = read_original(name, n, f_s)
            for idx, val in enumerate(method):
                norm_test(idx,val)

    for idx, val in enumerate(method):
        for f_idx,f_val in enumerate(f_sub):
            norm_average[idx,f_idx] = np.mean(acc[idx,f_idx])

        plt.plot([int(math.pow(2,x) * n_feature) for x in f_sub], norm_average[idx], linewidth=3, linestyle='--',marker='o', label='%s' % val)
    # print(norm_average)
    for m in method:
        os.remove('other_method/kmeans_linear/%s/test_%s'%(name,m))
    plt.legend(fontsize=15)
    plt.xlabel("dimension", fontsize=15)
    plt.ylabel("approximation error", fontsize=15)
    plt.savefig('other_method/%s_%s.png' %(name,'approximation_error'))
    plt.show()


