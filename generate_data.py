import os
import subprocess
import gc
import numpy as np
import scipy.sparse
from multiprocessing import Pool, Queue,Manager
from sklearn.datasets import dump_svmlight_file
from sklearn.cluster import KMeans
from sklearn import preprocessing
from liblinear.python import liblinearutil
from sklearn.model_selection import train_test_split
import argparse
from sklearn import preprocessing
import ffht
import time
from scipy.linalg import hadamard
from sklearn.datasets import load_svmlight_file
from scipy.sparse.linalg import  *
import argparse
from scipy._lib._util import check_random_state
from scipy.sparse import csc_matrix

def ped(data):
    n_number, f_num = np.shape(data)
    power = math.ceil(np.log2(f_num))
    np.pad(data, (0, 2**power - f_num), 'constant', constant_values=(0, 0))
    return data,2**power

def cwt_matrix(sketch_size, original_feature, seed=None):
    rng = check_random_state(seed)
    rows = rng.randint(0, sketch_size, original_feature)
    cols = np.arange(original_feature + 1)
    signs = rng.choice([1, -1], original_feature)
    S = csc_matrix((signs, rows, cols), shape=(sketch_size, original_feature))
    return S

def generater(name, what = 'train_0', what2 = 'test_0'):
    np.set_printoptions(threshold=np.inf, suppress=True)
    train_url = "/home/comp/cszjlei/svm/BudgetedSVM/original/%s/train" % (name)
    # data = load_svmlight_file(train_url)
    train, tra_label = load_svmlight_file(train_url)
    tra_label = np.mat(tra_label).T
    d_data,n_feature = np.shape(train)
    train_shape = n_feature
    # sub = int(math.pow(2,f_sub))
    if n_feature>10000:
        sub = int(math.pow(2,13))
        transformer =  cwt_matrix(sub,n_feature)
        train = scipy.sparse.csr_matrix.dot(train, transformer.T)
        n_feature = sub
    else:
        train, n_feature = ped(train)


    array = np.mat(np.random.uniform(-1, 1, n_feature))  # -1 +1均匀采样
    array[array < 0] = -1
    array[array > 0] = 1
    label = np.where(tra_label[:, 0] != 1, -1, 1)
    n_sample, d_feature = np.shape(train)
    train = train.todense()
    # scaler = preprocessing.StandardScaler(with_std=False).fit(train)
    scaler = preprocessing.StandardScaler(with_std=False).fit(train)
    train = scaler.transform(train)
    # print(0)
    a = np.asarray(np.multiply(np.asarray(train), array))
    for i in range(a.shape[0]):
        # print(a[i].shape)
        ffht.fht(a[i])
    train = np.around(a,decimals=4)
    train_lable = (np.asarray(label)).reshape(-1)
    # dump_svmlight_file(np.around(a, decimals=4),(np.asarray(label)).reshape(-1),'svm/BudgetedSVM/original/%s/%s'%(name,what),zero_based=False)
    del a
    gc.collect()
    test_url = "/home/comp/cszjlei/svm/BudgetedSVM/original/%s/test" % (name)
    test, te_label = load_svmlight_file(test_url)
    # test, n_feature = ped(test)
    te_label = np.mat(te_label).T
    label = np.where(te_label[:, 0] != 1, -1, 1)
    n_number, n_feature = np.shape(test)
    if n_feature > 10000:
        if n_feature>train_shape:
            # print(train_shape,n_feature)
            test = scipy.sparse.csr_matrix.dot(test[:,:train_shape], transformer.T)
        elif n_feature == train_shape:
            test = scipy.sparse.csr_matrix.dot(test, transformer.T)
        else:
            pad = np.zeros((n_number,train_shape-n_feature))
            test = scipy.sparse.hstack(test,pad)
            test = scipy.sparse.csr_matrix.dot(test, transformer.T)
    else:
        test, n_feature = ped(test)

        # n_feature = sub
    test = test.todense()
    test = scaler.transform(test)
    a = np.asarray(np.multiply(np.asarray(test), array))

    for i in range(a.shape[0]):
        ffht.fht(a[i])
    test = np.around(a, decimals=4)
    test_lable = (np.asarray(label)).reshape(-1)
    return train,train_lable,test,test_lable
    # dump_svmlight_file(np.around(a, decimals=4), (np.asarray(label)).reshape(-1), 'svm/BudgetedSVM/original/%s/%s' % (name, what2),
    #                    zero_based=False)

    # print(R_variace, R_means)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='choose the trainset and training parameter')
    parser.add_argument('-d', action='store', dest='train', help='choose the trainset', default=False)
    parser.add_argument('-m', action='store', dest='landmark', help='choose the landmark point', default=False)
    parser.add_argument('-l', action='store', dest='lambda', help='paremeter for l2 regularzation', default=False)
    parser.add_argument('-g', action='store', dest='gamma', help='gamma for the rbf kernel', default=False)
    param = parser.parse_args()
    np.set_printoptions(threshold=np.inf, suppress=True,precision=4)
    name = '%s' % (param.train)
    generater( name=name)

