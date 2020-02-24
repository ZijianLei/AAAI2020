from __future__ import division
import os
import subprocess
import multiprocessing
import gc
from multiprocessing import Pool, Queue,Manager
import numpy as np
import scipy.sparse
from scipy.sparse import linalg
from functools import partial
from sklearn.datasets import dump_svmlight_file,load_svmlight_file
from sklearn.cluster import KMeans
from scipy.linalg import hadamard
import sklearn
from sklearn import preprocessing,random_projection
from sklearn.decomposition import PCA
from liblinear.python import liblinearutil
import argparse
from scipy._lib._util import check_random_state
import sklearn.utils
from scipy.sparse import csc_matrix
import time
from skfeature.function.similarity_based import lap_score
from skfeature.utility import construct_W
from skfeature.utility import unsupervised_evaluation
import ffht
from numba import jit
import sklearn.utils
import math

def ped(data):
    n_number, f_num = np.shape(data)
    power = math.ceil(np.log2(f_num))
    d = 2**power - f_num
    data = np.pad(data, ((0,0),(0, d)), 'constant', constant_values=(0, 0))
    return data,2**power

def cwt_matrix(sketch_size, original_feature, seed=None):
    rng = check_random_state(seed)
    rows = rng.randint(0, sketch_size, original_feature)
    cols = np.arange(original_feature + 1)
    signs = rng.choice([1, -1], original_feature)
    S = csc_matrix((signs, rows, cols), shape=(sketch_size, original_feature))
    return S

def random(f_s, name,data,test):
    test_url = "/home/comp/cszjlei/svm/BudgetedSVM/original/%s/train_%d" % (name, 0)
    # train_url = "/home/comp/cszjlei/svm/BudgetedSVM/original/%s/train" % name
    d, tra_label = load_svmlight_file(test_url)
    n_sample, d_feature = np.shape(d)
    d_sub = int(math.pow(2, f_s) * d_feature)


    train_url = "/home/comp/cszjlei/svm/BudgetedSVM/original/%s/train_%d" % (name, 0)
    test_url = "/home/comp/cszjlei/svm/BudgetedSVM/original/%s/test_%d" % (name, 0)
    a, tra_label = load_svmlight_file(train_url)
    label = np.mat(tra_label).T
    data = data
    b, te_label = load_svmlight_file(test_url)
    te_label = np.mat(te_label).T
    test_data = test
    start = time.process_time()
    R = np.random.choice(int(data.shape[1]) - 1, int(d_sub), replace=False)
    R = sorted(R)
    train2 = data[:, R]
    label = (np.asarray(label)).reshape(-1)
    transform_time[0, 3] += time.process_time() - start
    test_start = time.process_time()
    test = test_data[:, R]
    transform_time[2, 3] = time.process_time() - test_start
    transform_time[1, 3], transform_time[3, 3] = model_training(np.around(train2, decimals=2), label,
                                                                np.around(test, decimals=2), te_label)

    start = time.process_time()
    norm = scipy.linalg.norm(data, axis=0)
    p = norm ** 2 / np.sum(norm ** 2)
    R_sampling = np.random.choice(data.shape[1], d_sub, replace=False, p=p)
    R_sampling = sorted(R_sampling)
    R_sampling = np.asarray(R_sampling)
    train2 = data[:, R_sampling]
    label = (np.asarray(label)).reshape(-1)
    transform_time[0, 4] += time.process_time() - start
    test_start = time.process_time()
    test = test_data[:, R_sampling]
    transform_time[2, 4] = time.process_time() - test_start
    transform_time[1, 4], transform_time[3, 4] = model_training(np.around(train2, decimals=2), label,
                                                                np.around(test, decimals=2), te_label)

    start = time.process_time()
    norm = scipy.linalg.norm(data, axis=0)
    topn = np.argsort(norm, axis=0)
    topn = np.reshape(topn, (1, -1))
    topn = np.asarray(topn)
    R = topn[0, d_feature - d_sub:]
    R = sorted(R)
    R_norm = np.asarray(R)
    train2 = data[:, R_norm]
    transform_time[0, 5] += time.process_time() - start
    label = (np.asarray(label)).reshape(-1)
    test_start = time.process_time()
    test = test_data[:, R_norm]
    transform_time[2, 5] = time.process_time() - test_start
    transform_time[1, 5], transform_time[3, 5] = model_training(np.around(train2, decimals=2), label,
                                                                np.around(test, decimals=2), te_label)

    n_label = int(len(tra_label) / 20)
    label = np.mat(tra_label).T
    positive = np.where(label[:, 0] == 1)[0]
    one = np.ones((2 * n_label, 2 * n_label))

    negitive = np.where(label[:, 0] != 1)[0]
    # print(negitive,negitive.shape)
    sub = np.vstack((data[positive[:n_label]], data[negitive[:n_label]]))
    one[n_label:, 0:n_label] = -1
    one[0:n_label, n_label:] = -1
    # A = np.vstack((np.hstack((one, -one * tradeoff)), np.hstack((-tradeoff * one, one))))
    start = time.process_time()
    A = one
    np.fill_diagonal(A, 0)
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    trace = np.sum(np.multiply(np.dot(sub.T, L), sub.T), axis=1)
    topn = np.argsort(trace.T)
    topn = np.reshape(topn, (1, -1))
    topn = np.asarray(topn)
    # print(topn)
    R_nonuniform = sorted(topn[0, :d_sub])
    R_nonuniform = np.asarray(R_nonuniform)
    train2 = data[:, R_nonuniform]

    transform_time[0, 6] += time.process_time() - start
    label = (np.asarray(label)).reshape(-1)
    test_start = time.process_time()
    test = test_data[:, R_nonuniform]
    transform_time[2, 6] = time.process_time() - test_start
    transform_time[1, 6], transform_time[3, 6] = model_training(np.around(train2, decimals=2), label,
                                                                np.around(test, decimals=2), te_label)
    
    test_url = "/home/comp/cszjlei/svm/BudgetedSVM/original/%s/train" % (name)
    data, tra_label = load_svmlight_file(test_url)

    test_url = "/home/comp/cszjlei/svm/BudgetedSVM/original/%s/test" % (name)
    test_data, te_label = load_svmlight_file(test_url)

    n_num,f_num = np.shape(data)
    start = time.process_time()
    gaussian_transformer = random_projection.GaussianRandomProjection(n_components=d_sub)
    train2 = gaussian_transformer.fit_transform(data)
    transform_time[0, 0] += time.process_time() - start
    start_test = time.process_time()
    test = gaussian_transformer.fit_transform(test_data)
    transform_time[2, 0] += time.process_time() - start_test
    label = (np.asarray(tra_label)).reshape(-1)
    label = (np.asarray(label)).reshape(-1)
    te_label = (np.asarray(te_label)).reshape(-1)
    te_label = (np.asarray(te_label)).reshape(-1)
    transform_time[1, 0], transform_time[3, 0] = model_training(np.around(train2, decimals=2), label,
                                                                np.around(test, decimals=2), te_label)


    start = time.process_time()
    sparse_transformer = random_projection.SparseRandomProjection(
        n_components=d_sub,density=1/3.0)  # transformer attribute component (n_component,n_feature)
    train2 = sparse_transformer.fit_transform(data)
    transform_time[0, 1] += time.process_time() - start
    start_test = time.process_time()
    test = gaussian_transformer.fit_transform(test_data)
    transform_time[2, 1] += time.process_time() - start_test

    label = (np.asarray(tra_label)).reshape(-1)
    label = (np.asarray(label)).reshape(-1)
    te_label = (np.asarray(te_label)).reshape(-1)
    te_label = (np.asarray(te_label)).reshape(-1)
    transform_time[1, 1], transform_time[3, 1] = model_training(np.around(train2, decimals=2), label,
                                                                np.around(test, decimals=2), te_label)

    start = time.process_time()
    transformer = cwt_matrix(d_sub,f_num)
    train2 = scipy.sparse.csr_matrix.dot(data, transformer.T)
    label = (np.asarray(label)).reshape(-1)
    transform_time[0, 2] += time.process_time() - start
    test_start = time.process_time()
    test = scipy.sparse.csr_matrix.dot(test_data, transformer.T)
    transform_time[2,2] = time.process_time() - test_start
    transform_time[1, 2], transform_time[3, 2] = model_training(np.around(train2, decimals=2), label,
                                                                np.around(test, decimals=2), te_label)



    





def model_training(x,y,x_test,y_test):
    prob = liblinearutil.problem(y, x)
    start = time.process_time()
    # sklearn.svm.libsvm.cross_validation(x,y)
    # sklearn.svm.libsvm.fit(x,y)
    param = liblinearutil.parameter('-q')
    m = liblinearutil.train(prob, param)
    train_time = time.process_time() - start
    start2 = time.process_time()
    pred_labels, (acc, MSE, SCC), pred_values = liblinearutil.predict(y_test, x_test, m)
    predict_time = time.process_time() - start2
    # print(time.process_time() - start)
    return train_time,predict_time


def transform(name):
    np.set_printoptions(threshold=np.inf, suppress=True)
    train_url = "/home/comp/cszjlei/svm/BudgetedSVM/original/%s/train" % (name)
    # data = load_svmlight_file(train_url)
    train, tra_label = load_svmlight_file(train_url)
    train = train.todense()
    tra_label = np.mat(tra_label).T
    b_num,n_feature = np.shape(train)
    if n_feature < 4000:
        train, n_feature = ped(train)
        print(train.shape)
        start = time.process_time()
    else:
        start = time.process_time()
        sub = int(math.pow(2,12))
        transformer =  cwt_matrix(sub,n_feature)
        train = scipy.sparse.csr_matrix.dot(train, transformer.T)
        n_feature = sub
    rng = check_random_state(None)
    array = np.mat(rng.choice([1, -1], n_feature))
    n_sample, d_feature = np.shape(train)
    a = np.array(np.multiply(train, array))
    for i in range(n_sample):
        ffht.fht(a[i])
    transform_time = time.process_time() - start
    return transform_time,a,array

def test_transform(name,array):
    np.set_printoptions(threshold=np.inf, suppress=True)
    train_url = "/home/comp/cszjlei/svm/BudgetedSVM/original/%s/test" % (name)
    # data = load_svmlight_file(train_url)
    train, tra_label = load_svmlight_file(train_url)
    train = train.todense()
    tra_label = np.mat(tra_label).T
    b_num,n_feature = np.shape(train)

    if n_feature < 10000:
        train, n_feature = ped(train)
        start = time.process_time()
    else:
        start = time.process_time()
        sub = int(math.pow(2,12))
        transformer =  cwt_matrix(sub,n_feature)
        train = scipy.dot(train[:,:n_feature], transformer.T)
        n_feature = sub
    a = np.array(np.multiply(train, array))
    for i in range(b_num):
        ffht.fht(a[i])
    transform_time = time.process_time() - start
    return transform_time,a


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='choose the dataset and training parameter')
    parser.add_argument('-d', action='store', dest='data', help='choose the dataset', default=False)
    parser.add_argument('-m', action='store', dest='landmark', help='choose the landmark point', default=False)
    parser.add_argument('-l', action='store', dest='lambda', help='paremeter for l2 regularzation', default=False)
    parser.add_argument('-g', action='store', dest='gamma', help='gamma for the rbf kernel', default=False)
    param = parser.parse_args()
    method = ['gaussian','sparse','sparse_embedding','random','weight_sampling','norm','laplacian']
    np.set_printoptions(threshold=np.inf, suppress=True)
    # method = ['corr','nonuniform']
    tradeoff = [0.2,0.4,0.6,0.8,1]
    # f_sub = [-4]
    f_sub = [-4]
    # f_sub = [-10, -9, -8, -7, -6]
    cost = 1
    name = '%s'%(param.data)
    sub = 0
    transform_time = np.zeros((4,7)) # model training[0,1],model_prediction[2,3]
    for f_idx,f_s in enumerate(f_sub):
        for n in range(1):
            hadamard_time,train,array= transform(name)
            transform_time[0,3:] += hadamard_time
            # print(transform_time)
            transform_time[2, 3:], test = test_transform(name,array)
            random(f_s,name,train,test)

        # print(np.around(transform_time,decimals=3))
        print('training time',np.around(transform_time[0],decimals=2),np.around(transform_time[1],decimals=2))
        print('prediction time',np.around(transform_time[2] + transform_time[3], decimals=2))

