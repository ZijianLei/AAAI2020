from __future__ import division
import os
import subprocess
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
import matplotlib.pyplot as plt
from numba import jit
import time
from sklearn.feature_extraction import FeatureHasher
def random( name, what):

    np.set_printoptions(threshold=np.inf, suppress=True)
    train_url = "/home/comp/cszjlei/svm/BudgetedSVM/original/%s/train" % (name)
    train, tra_label = load_svmlight_file(train_url)
    # train = train.todense()
    b_num, n_feature = np.shape(train)
    power = math.ceil(np.log2(n_feature))
    if n_feature < 4000:
        d_sub = int(2**(power+f_s))
    else:
        d_sub = int(math.pow(2,12+f_s))
    start = time.time()
    pca = sklearn.decomposition.PCA(n_components=d_sub,svd_solver = 'randomized')
    train2 = pca.fit_transform(train.toarray())
    t1 = time.time()-start
    tra_label = np.mat(tra_label).T
    np.set_printoptions(threshold=np.inf, suppress=True)
    label = np.where(tra_label[:, 0] != 1, -1, 1)
    # hasher = FeatureHasher(n_features =d_sub)
    # train2 = hasher.transform(train)
    label = (np.asarray(label)).reshape(-1)
    dump_svmlight_file(np.around(train2,decimals=4), label, 'other_method/kmeans_linear/%s/%s_pca' % (name, what),zero_based=False)
    test_url = "/home/comp/cszjlei/svm/BudgetedSVM/original/%s/test" % (name)
    data, te_label = load_svmlight_file(test_url)
    # data = data.todense()
    te_label = np.mat(te_label).T
    np.set_printoptions(threshold=np.inf, suppress=True)
    label = np.where(te_label[:, 0] != 1, -1, 1)
    train2 =pca.transform(data.toarray())
    label = (np.asarray(label)).reshape(-1)
    dump_svmlight_file(np.around(train2, decimals=4), label,
                       'other_method/kmeans_linear/%s/%s_pca' % (name, 'test'), zero_based=False)
    return t1



def read_original(name,f_s):
    t1 = random( name=name, what='train')
    acc[f_idx],t2=model_training()
    print(t1+t2)



def model_training():
    meth = method[0]
    y, x = liblinearutil.svm_read_problem("other_method/kmeans_linear/%s/train_%s" % (
        name, meth))
    y_test, x_test = liblinearutil.svm_read_problem("other_method/kmeans_linear/%s/test_%s" % (
        name, meth))

    prob = liblinearutil.problem(y, x)
    temp_result = np.zeros((9))
    for idx, val in enumerate(cost):
        start = time.time()
        param = liblinearutil.parameter(' -q -c %f' % (val))
        m = liblinearutil.train(prob, param)
        t2 = time.time()-start
        pred_labels, (temp_result[idx], MSE, SCC), pred_values = liblinearutil.predict(y_test, x_test, m)

    return np.max(temp_result),t2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='choose the dataset and training parameter')
    parser.add_argument('-d', action='store', dest='data', help='choose the dataset', default=False)
    parser.add_argument('-m', action='store', dest='landmark', help='choose the landmark point', default=False)
    parser.add_argument('-l', action='store', dest='lambda', help='paremeter for l2 regularzation', default=False)
    parser.add_argument('-g', action='store', dest='gamma', help='gamma for the rbf kernel', default=False)
    param = parser.parse_args()
    np.set_printoptions(threshold=np.inf, suppress=True)
    method = ['pca']
    np.set_printoptions(threshold=np.inf, suppress=True, precision=5)
    f_sub = [-4]
    # cost = [2 ** (-6), 2 ** (-5), 2 ** (-4), 2 ** (-3), 2 ** (-2), 2 ** (-1), 1, 2, 4]
    cost = [1]
    name = '%s' % (param.data)
    sub = 0
    acc = np.empty((6))  # build the matrix to stall the result
    # transform_time = np.zeros(4,6)
    for f_idx, f_s in enumerate(f_sub):
        read_original(name, f_s)
    print(acc)

    for idx, val in enumerate(method):
        numpy.save('other_method/kmeans_linear/%s/%s_result'%(name,'pca'),acc)
    # print(acc[idx])
    for m in method:
        os.remove('other_method/kmeans_linear/%s/train_%s' % (name, m))
        os.remove('other_method/kmeans_linear/%s/test_%s' % (name, m))