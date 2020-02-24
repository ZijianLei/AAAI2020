import os
import subprocess
from read_dataset import *
import gc
from multiprocessing import Pool, Queue
import numpy as np
import scipy.sparse
from functools import partial
from sklearn.datasets import dump_svmlight_file,load_svmlight_file
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.decomposition import PCA
from liblinear.python import liblinearutil
import argparse
from numba import jit

@jit
def corr(data, d_sub, name, what, array, R,n_label):
    np.set_printoptions(threshold=np.inf, suppress=True)
    # label = np.where(data[:, 0]!=1,-1,1)
    label = data[:, 0]
    # label1 = data[:, 0]
    # data = data[:, 1:]
    n_sample, d_feature = np.shape(data)
    data = data[:, 1:]
    a = math.sqrt(float(d_feature / d_sub)) * data
    transform = a[:n_label]
    sub_lab = label[:n_label]
    matrix = np.hstack((sub_lab, transform))
    corr = np.corrcoef(matrix, rowvar=False)
    topn = np.argsort(-abs(corr[0]), axis=0) - 1
    topn = np.reshape(topn, (1, -1))
    topn = np.asarray(topn)
    R_corr = sorted(topn[0, 1:d_sub + 1])
    R_corr = np.asarray(R_corr)
    train2 = a[:, R_corr]
    label = (np.asarray(label)).reshape(-1)
    dump_svmlight_file(np.around(train2,decimals=4), label, 'other_method/kmeans_linear/%s/%s_corr' % (name, what),
                       zero_based=False)
def test_transform(Rlaplacian,d_sub,name,what):
    test_url = "/home/comp/cszjlei/svm/BudgetedSVM/original/%s/test_%d" % (name, 0)
    # train_url = "/home/comp/cszjlei/svm/BudgetedSVM/original/%s/train" % name
    data, tra_label = load_svmlight_file(test_url)
    te_label = np.mat(tra_label).T
    np.set_printoptions(threshold=np.inf, suppress=True)
    # data = scipy.sparse.coo_matrix.tocsr(data)
    n_number, d_feature = np.shape(data)
    label = np.where(te_label[:, 0] != 1, -1, 1)
    a = math.sqrt(float(d_feature / d_sub)) * data

    train2 = a[:,Rlaplacian]
    label = (np.asarray(label)).reshape(-1)
    dump_svmlight_file(np.around(train2, decimals=4), label,
                       'other_method/kmeans_linear/%s/%s_laplacian' % (name, what), zero_based=False)


def sampling(data, d_sub, name, what,tradeoff,n_label):
    np.set_printoptions(threshold=np.inf, suppress=True)
    n_sample, d_feature = np.shape(data)
    # label = np.where(data[:, 0]!=1,-1,1)
    label = data[:, 0]
    data = data[:, 1:]
    a = math.sqrt(float(d_feature / d_sub)) * data #a is without the feature
    # data = preprocessing.scale(a,with_std=False)
    positive = np.where(label[:,0] == 1)[0]
    one = np.ones((n_label,n_label))
    A = np.vstack((np.hstack((one,-one*tradeoff)),np.hstack((-tradeoff*one,one))))
    np.fill_diagonal(A,0)
    D = np.diag(np.sum(A,axis=1))
    L = D-A
    negitive = np.where(label[:,0] != 1 )[0]
    sub = np.vstack((a[positive[:n_label]],a[negitive[:n_label]]))[:,1:] #sub is feature 1600*d_feature
    trace = np.diag( np.dot(np.dot(sub.T,L),sub))
    topn = np.argsort(trace)
    topn = np.reshape(topn, (1, -1))
    topn = np.asarray(topn)
    R_nonuniform = sorted(topn[0,:d_sub])
    R_nonuniform = np.asarray(R_nonuniform)
    train2 = a[:, R_nonuniform]
    # label = data[:, 0]
    label = (np.asarray(label)).reshape(-1)
    dump_svmlight_file(np.around(train2,decimals=4), label, 'other_method/kmeans_linear/%s/%s_laplacian' % (name, what),
                       zero_based=False)
    return R_nonuniform

def read_original(name,n,f_s,trad):
    train_url = "/home/comp/cszjlei/svm/BudgetedSVM/original/%s/train_%d" % (name, 0)
    # train_url = "/home/comp/cszjlei/svm/BudgetedSVM/original/%s/train" % (name)
    train1,tra_label  = load_svmlight_file(train_url)
    tra_label = np.mat(tra_label).T
    train1, n_feature = ped(train1)
    train = scipy.sparse.hstack((tra_label, train1))
    train = train.todense()
    sub = int(math.pow(2, f_s) * n_feature)
    array = np.mat(np.random.uniform(-1, 1, n_feature))  #
    R = np.random.choice(int(train.shape[1]) - 1, int(sub), replace=False)
    R = sorted(R)
    # train1 =  maxabs_scale(train[:, 1:])
    # random(data=train1, d_sub=sub,name=name, what='train', array=array, R=R)
    R_laplacian = sampling(data = train,d_sub = sub,name  = name, what =  'train',tradeoff = trad, n_label = int(len(tra_label)/10))
    test_transform(Rlaplacian = R_laplacian,d_sub = sub,name  = name, what =  'test')
    # corr(data=train, d_sub=sub, name=name, what='train', array=array, R=R, n_label = int(len(tra_label)/10))


def model_training(n, i):
    meth = method[0]
    y, x = liblinearutil.svm_read_problem("other_method/kmeans_linear/%s/train_%s" % (
        name, meth))
    y_test, x_test = liblinearutil.svm_read_problem("other_method/kmeans_linear/%s/test_%s" % (
        name, meth))

    prob = liblinearutil.problem(y, x)
    temp_result = np.empty((18))

    for idx, val in enumerate(cost):
        param = liblinearutil.parameter(' -q -c %f' % (val))
        m = liblinearutil.train(prob, param)
        pred_labels, (temp_result[idx], MSE, SCC), pred_values = liblinearutil.predict(y_test, x_test, m)

    return (i, f_idx, n, temp_result)


def read_acc(rea):
    i, f_idx, n, accuracy = rea
    acc[f_idx, i, n,:] = accuracy[:]



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='choose the dataset and training parameter')
    parser.add_argument('-d', action='store', dest='data', help='choose the dataset', default=False)
    parser.add_argument('-m', action='store', dest='landmark', help='choose the landmark point', default=False)
    parser.add_argument('-l', action='store', dest='lambda', help='paremeter for l2 regularzation', default=False)
    parser.add_argument('-g', action='store', dest='gamma', help='gamma for the rbf kernel', default=False)
    param = parser.parse_args()
    # method = ['random','variance','norm','corr','nonuniform']
    np.set_printoptions(threshold=np.inf, suppress=True)
    method = ['laplacian']
    tradeoff = [0.2,0.4,0.6,0.8,1]
    # trad = 1.5
    # tradeoff = [0.8]
    f_sub = [-6,-5,-4,-3, -2, -1]
    # f_sub = [-3]
    # f_sub = [-10, -9, -8, -7, -6]
    cost = [2 ** (-10), 2 ** (-8), 2 ** (-6), 2 ** (-5), 2 ** (-4), 2 ** (-3), 2 ** (-2), 2 ** (-1), 1, 2, 4, 8, 16,32,64,128,256,512]
    name = '%s'%(param.data)
    # label_used = [100,500,1000,2000,4000]
    acc = np.empty(( len(f_sub), len(tradeoff), 15, 18))
    summary = np.zeros((1,len(f_sub),15,18))
    temp = np.zeros(5)
    for f_idx,f_s in enumerate(f_sub):
        for j,trad in enumerate(tradeoff):

            for n in range(15):
                read_original(name, n, f_s,trad)
                pool = Pool(2)
                f2 = partial(model_training,n)
                pool.apply_async(f2,(j,),callback = read_acc)
                pool.close()
                pool.join()
    # print(acc)
    for idx,val in enumerate(f_sub):
        for n in range(5):
            temp[n] = np.mean(np.amax(acc[idx,n],axis=1))
        max_trad_idx = np.argsort(-temp)
        summary[0,idx] = acc[idx,max_trad_idx[0]]

    for idx, val in enumerate(method):
        numpy.save('other_method/kmeans_linear/%s/%s_result'%(name,'laplacian'),summary[idx])
    for m in method:
        os.remove('other_method/kmeans_linear/%s/train_%s'%(name,m))


