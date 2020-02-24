from __future__ import division
import os
import subprocess
import multiprocessing
from read_dataset import *
import gc
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
from generate_data import generater
from numba import jit
def random( d_sub, name, what, R,data,tra_label):
    # test_url = "/home/comp/cszjlei/svm/BudgetedSVM/original/%s/train_%d" % (name, 0)
    # # train_url = "/home/comp/cszjlei/svm/BudgetedSVM/original/%s/train" % name
    # data, tra_label = load_svmlight_file(test_url)
    label = np.mat(tra_label).T
    n_sample, d_feature = np.shape(data)
    a = math.sqrt(float(d_feature / d_sub)) * data
    train2 = a[:,R]
    label = (np.asarray(label)).reshape(-1)
    dump_svmlight_file(np.around(train2, decimals=4), label, 'other_method/kmeans_linear/%s/%s_random' % (name, what),
                       zero_based=False)
    # data = data.todense()
    norm = np.linalg.norm(data, axis=0)
    topn = np.argsort(norm, axis=0)
    topn = np.reshape(topn, (1, -1))
    topn = np.asarray(topn)
    R= topn[0, d_feature - d_sub:]
    R = sorted(R)
    R_norm = np.asarray(R)
    train2 = data[:, R_norm]
    label = (np.asarray(label)).reshape(-1)
    # exit()
    dump_svmlight_file(np.around(train2, decimals=4), label, 'other_method/kmeans_linear/%s/%s_norm' % (name, what),
                       zero_based=False)

    p = norm**2/np.sum(norm**2)
    R_sampling = np.random.choice(data.shape[1],d_sub,replace=False,p = p)
    R_sampling = sorted(R_sampling)
    train2 = data[:,R_sampling]
    # print(train2.shape)
    label = (np.asarray(label)).reshape(-1)
    # exit()
    dump_svmlight_file(np.around(train2,decimals=4), label, 'other_method/kmeans_linear/%s/%s_weight_sampling' % (name, what),
                       zero_based=False)

    return R_norm,R_sampling,p

def test_transform(Rrandom,Rnorm,Rsampling,p,d_sub,name,what,data,tra_label):
    # test_url = "/home/comp/cszjlei/svm/BudgetedSVM/original/%s/test_%d" % (name, 0)
    # train_url = "/home/comp/cszjlei/svm/BudgetedSVM/original/%s/train" % name
    # data, tra_label = load_svmlight_file(test_url)
    te_label = np.mat(tra_label).T
    np.set_printoptions(threshold=np.inf, suppress=True)
    # data = scipy.sparse.coo_matrix.tocsr(data)
    n_number, d_feature = np.shape(data)
    label = np.where(te_label[:, 0] != 1, -1, 1)
    # a = math.sqrt(float(d_feature / d_sub)) * data
    train2 = data[:, Rrandom]
    label = (np.asarray(label)).reshape(-1)
    dump_svmlight_file(np.around(train2, decimals=4), label,
                       'other_method/kmeans_linear/%s/%s_random' % (name, what), zero_based=False)
    train2 = data[:, Rnorm]
    label = (np.asarray(label)).reshape(-1)
    dump_svmlight_file(np.around(train2, decimals=4), label,
                       'other_method/kmeans_linear/%s/%s_norm' % (name, what), zero_based=False)
    label = np.where(te_label[:, 0] != 1, -1, 1)

    train2 = data[:, Rsampling]
    # train2 = np.multiply(data[:,Rnorm],p)
    label = (np.asarray(label)).reshape(-1)
    dump_svmlight_file(np.around(train2, decimals=4), label,
                       'other_method/kmeans_linear/%s/%s_weight_sampling' % (name, what), zero_based=False)



def read_original(name,f_s,train,tra_label,test,te_label):
    tra_label = np.mat(tra_label).T
    n_bnumber, n_feature = np.shape(train)
    # train = scipy.sparse.hstack((tra_label, train1))
    # print(train.shape)
    # power = math.ceil(np.log2(n_feature))
    # sub = int(math.pow(2, -power-f_s+4) * n_feature)
    sub = int(math.pow(2, f_s) * n_feature)  #
    if n_feature > 1024:
        sub = int(math.pow(2, f_s) * 4096)
    R = np.random.choice(int(train.shape[1]) - 1, int(sub), replace=False)
    R = sorted(R)
    # train1 = train
    R_norm,Rsampling,p = random(d_sub=sub,name=name, what='train', R=R,data=train,tra_label=tra_label)
    test_transform(Rrandom=R,Rnorm=R_norm,Rsampling = Rsampling,p =p ,d_sub = sub,name=name, what='test',data=test,tra_label = te_label)



def model_training(n, i):
    meth = method[i]
    y, x = liblinearutil.svm_read_problem("other_method/kmeans_linear/%s/train_%s" % (
        name, meth))
    y_test, x_test = liblinearutil.svm_read_problem("other_method/kmeans_linear/%s/test_%s" % (
        name, meth))

    prob = liblinearutil.problem(y, x)
    temp_result = np.empty((14))

    for idx, val in enumerate(cost):
        param = liblinearutil.parameter(' -q -c %f' % (val))
        m = liblinearutil.train(prob, param)
        pred_labels, (temp_result[idx], MSE, SCC), pred_values = liblinearutil.predict(y_test, x_test, m)

    return (i, f_idx, n, temp_result)


def read_acc(rea):
    i, f_idx, n, accuracy = rea
    acc[i, f_idx, n, :] = accuracy[:]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='choose the dataset and training parameter')
    parser.add_argument('-d', action='store', dest='data', help='choose the dataset', default=False)
    parser.add_argument('-m', action='store', dest='landmark', help='choose the landmark point', default=False)
    parser.add_argument('-l', action='store', dest='lambda', help='paremeter for l2 regularzation', default=False)
    parser.add_argument('-g', action='store', dest='gamma', help='gamma for the rbf kernel', default=False)
    param = parser.parse_args()
    method = ['random','norm','weight_sampling']
    np.set_printoptions(threshold=np.inf, suppress=True)
    f_sub = [-6,-5,-4, -3, -2, -1]
    # f_sub = [-3]
    # f_sub = [-10, -9, -8, -7, -6]
    cost = [2**(-6),2**(-5),2**(-4),2**(-3),2**(-2),2**(-1),1,2,4,8,16,32,64,128]
    name = '%s'%(param.data)
    sub = 0
    acc = np.empty((3,len(f_sub),5,14)) #build the matrix to stall the result
    for f_idx,f_s in enumerate(f_sub):
        for n in range(5):
            train, train_lable, test, test_lable = generater(name)
            read_original(name, f_s,train, train_lable, test, test_lable)
            pool = Pool(3)
            f2 = partial(model_training,n)
            for j in range(len(method)):
                 pool.apply_async(f2,(j,),callback = read_acc )
            pool.close()
            pool.join()

    for idx, val in enumerate(method):
        np.save('other_method/kmeans_linear/%s/%s_result'%(name,val),acc[idx])
        # print(acc[idx])
    for m in method:
        os.remove('other_method/kmeans_linear/%s/train_%s'%(name,m))
        os.remove('other_method/kmeans_linear/%s/test_%s' % (name, m))
