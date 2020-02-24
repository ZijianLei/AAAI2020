from __future__ import division
import os
import subprocess
import time
import multiprocessing
from read_dataset import *
import gc
from scipy.sparse.linalg import *
from multiprocessing import Pool, Queue,Manager
import numpy as np
import scipy.sparse
from scipy.sparse import linalg
from functools import partial
from sklearn.datasets import dump_svmlight_file,load_svmlight_file
from sklearn.cluster import KMeans
import sklearn
from sklearn import preprocessing
from sklearn.decomposition import PCA
from liblinear.python import liblinearutil
import argparse
from scipy.sparse.linalg import  *
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from numba import jit
def random( name, what):

    np.set_printoptions(threshold=np.inf, suppress=True)
    train_url = "/home/comp/cszjlei/svm/BudgetedSVM/original/%s/train" % (name)
    train, tra_label = load_svmlight_file(train_url)
    # train = train.todense()
    # scaler = preprocessing.StandardScaler(with_std=False).fit(train)
    # train = scaler.transform(train)
    b_num, n_feature = np.shape(train)
    power = math.ceil(np.log2(n_feature))
    if n_feature < 4000:
        d_sub = int(2**(power+f_s))
        # k = int(np.linalg.matrix_rank(train)*4/5)
        k = d_sub
        # print(k)
    else:
        d_sub = int(math.pow(2,12+f_s))
        # k =  int(math.pow(2,8))
        k =100
    tra_label = np.mat(tra_label).T
    np.set_printoptions(threshold=np.inf, suppress=True)
    label = np.where(tra_label[:, 0] != 1, -1, 1)
    start = time.time()
    u,s,vt = svds(train,k=k)
    # print(np.linalg.matrix_rank(train))
    # u,s,vt = np.linalg.svd(train,full_matrices=False)
    # norm = np.linalg.norm(train, axis=0)
    norm = scipy.sparse.linalg.norm(train,axis =0)
    topn = np.argsort(-norm, axis=0)
    topn = np.reshape(topn, (1, -1))
    topn = np.asarray(topn)
    R_norm = topn[0, :d_sub]
    R_norm = sorted(R_norm)
    train2 = train[:, R_norm]
    t1 = time.time()-start
    label = (np.asarray(label)).reshape(-1)
    dump_svmlight_file(np.around(train2,decimals=4), label, 'other_method/kmeans_linear/%s/%s_leverage_score' % (name, what),zero_based=False)
    test_url = "/home/comp/cszjlei/svm/BudgetedSVM/original/%s/test" % (name)
    data, te_label = load_svmlight_file(test_url)
    data = data.todense()
    # test = scaler.transform(data)
    te_label = np.mat(te_label).T
    np.set_printoptions(threshold=np.inf, suppress=True)
    label = np.where(te_label[:, 0] != 1, -1, 1)
    train2 = data[:, R_norm]
    # print(train2.shape[1])
    label = (np.asarray(label)).reshape(-1)
    dump_svmlight_file(np.around(train2, decimals=4), label,
                       'other_method/kmeans_linear/%s/%s_leverage_score' % (name, 'test'), zero_based=False)
    return t1


def read_original(name,f_s):
    t1 = random( name=name, what='train')
    acc[0,f_idx],t2=model_training()
    return t1+t2


def model_training():
    meth = method[0]
    y, x = liblinearutil.svm_read_problem("other_method/kmeans_linear/%s/train_%s" % (
        name, meth))
    y_test, x_test = liblinearutil.svm_read_problem("other_method/kmeans_linear/%s/test_%s" % (
        name, meth))

    prob = liblinearutil.problem(y, x)
    temp_result = np.zeros((12))
    # print(x.shape(1))

    for idx, val in enumerate(cost):
        start = time.time()
        param = liblinearutil.parameter(' -q -c %f' % (val))
        m = liblinearutil.train(prob, param)
        pred_labels, (temp_result[idx], MSE, SCC), pred_values = liblinearutil.predict(y_test, x_test, m)
    # print(temp_result)
    t2 = time.time()-start
    return np.max(temp_result),t2



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='choose the dataset and training parameter')
    parser.add_argument('-d', action='store', dest='data', help='choose the dataset', default=False)
    parser.add_argument('-m', action='store', dest='landmark', help='choose the landmark point', default=False)
    parser.add_argument('-l', action='store', dest='lambda', help='paremeter for l2 regularzation', default=False)
    parser.add_argument('-g', action='store', dest='gamma', help='gamma for the rbf kernel', default=False)
    param = parser.parse_args()
    method = ['leverage_score']
    np.set_printoptions(threshold=np.inf, suppress=True,precision=5)
    # tradeoff = [0.2,0.4,0.6,0.8,1]
    # f_sub = [-6,-5,-4, -3, -2, -1]
    f_sub = [-4]
    # cost = [2**(-8),2**(-6),2**(-5),2**(-4),2**(-3),2**(-2),2**(-1),1,2,4,8,16]
    cost = [1]
    name = '%s'%(param.data)
    sub = 0
    computation_time = np.empty((1,6))
    acc = np.empty((1,6)) #build the matrix to stall the result
    # transform_time = np.zeros(4,6)
    for f_idx,f_s in enumerate(f_sub):
        computation_time[0,f_idx] = read_original(name, f_s)
    # print(acc)


    for idx, val in enumerate(method):
        numpy.save('other_method/kmeans_linear/%s/%s_result'%(name,val),acc[idx])
        print(acc[idx])
        print(computation_time[idx])
    for m in method:
        os.remove('other_method/kmeans_linear/%s/train_%s'%(name,m))
        os.remove('other_method/kmeans_linear/%s/test_%s' % (name, m))
