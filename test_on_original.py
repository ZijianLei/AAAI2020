import os
import subprocess
import multiprocessing
from read_dataset import *
import gc
import sklearn
from multiprocessing import Pool, Queue,Manager
import numpy as np
import scipy.sparse
from functools import partial
from sklearn.datasets import dump_svmlight_file,load_svmlight_file
from scipy.sparse import linalg
from sklearn import preprocessing
from sklearn.decomposition import PCA
from liblinear.python import liblinearutil
import argparse
from numba import jit

def random(data, d_sub, name, what, array, R,n_label):
    np.set_printoptions(threshold=np.inf, suppress=True)
    data = scipy.sparse.coo_matrix.tocsr(data)
    label = data[:,0]
    data = sklearn.preprocessing.maxabs_scale(data[:, 1:])
    n_sample, d_feature = np.shape(data)
    a = math.sqrt(float(d_feature / d_sub)) * data
    #
    # pca = PCA(n_components=d_sub).fit_transform(a)
    #
    # label = (np.asarray(label)).reshape(-1)
    # dump_svmlight_file(pca, label, 'other_method/kmeans_linear/%s/%s_pca' % (name, what), zero_based=False)

    train2 = a[:,R]
    dump_svmlight_file(train2, label, 'other_method/kmeans_linear/%s/%s_random' % (name, what), zero_based=False)

    # func = KMeans(n_clusters=d_sub)
    # transform = a.T
    # kmeans = func.fit(transform)
    # klabel = kmeans.labels_
    # R_means = []
    # for i in range(d_sub):
    #     temp1 = []
    #     temp2 = []
    #     for idx, val in enumerate(klabel):
    #         if val == i:
    #             temp1.append(idx)
    #             distance = func.transform(transform[idx])
    #             temp2.append(distance[0, val])
    #
    #     idx_temp = temp2.index(min(temp2))
    #     R_means.append(temp1[idx_temp])
    # R_means = sorted(R_means)
    # train2 = a[:,R_means]
    # label = (np.asarray(label)).reshape(-1)
    # dump_svmlight_file(train2, label, 'other_method/kmeans_linear/%s/%s_means' % (name, what),
    #                    zero_based=False)
    # b = a
    # transform = b.T
    # var = np.var(transform, axis=1)
    # topn = np.argsort(var, axis=0)
    # topn = np.reshape(topn, (1, -1))
    # topn = np.asarray(topn)
    # R_variace = topn[0, d_feature - d_sub:]
    # R_variace = sorted(R_variace)
    # R_variace = np.asarray(R_variace)
    # train2 = a[:,R_variace]
    # label = (np.asarray(label)).reshape(-1)
    # dump_svmlight_file(train2, label, 'other_method/kmeans_linear/%s/%s_variance' % (name, what),
    #                    zero_based=False)
    b = a[:n_label]
    transform = b.T
    norm = linalg.norm(transform, axis=1)
    # print(norm)
    topn = np.argsort(norm, axis=0)
    # print(topn)
    topn = np.reshape(topn, (1, -1))
    topn = np.asarray(topn)
    R_norm = topn[0, d_feature - d_sub:]
    R_norm = sorted(R_norm)
    R_norm = np.asarray(R_norm)
    train2 = a[:,R_norm]
    # exit()
    dump_svmlight_file(train2, label, 'other_method/kmeans_linear/%s/%s_norm' % (name, what),
                       zero_based=False)
@jit
def corr(data, d_sub, name, what, array, R):
    np.set_printoptions(threshold=np.inf, suppress=True)
    label = data[:, 0]
    label1 = data[:, 0]
    data = data[:, 1:]
    n_sample, d_feature = np.shape(data)
    a = math.sqrt(float(d_feature / d_sub)) * data
    transform = a[:1000, :]
    sub_lab = label1[:1000,0]
    matrix = np.hstack((sub_lab, transform))
    corr = np.corrcoef(matrix, rowvar=False)
    topn = np.argsort(-abs(corr[0]), axis=0) - 1
    topn = np.reshape(topn, (1, -1))
    topn = np.asarray(topn)
    R_corr = sorted(topn[0, 1:d_sub + 1])
    R_corr = np.asarray(R_corr)
    train2 = a[:, R_corr]
    label = (np.asarray(label)).reshape(-1)
    dump_svmlight_file(train2, label, 'other_method/kmeans_linear/%s/%s_corr' % (name, what),
                       zero_based=False)

def sampling(data, d_sub, name, what,tradeoff):
    np.set_printoptions(threshold=np.inf, suppress=True)
    n_sample, d_feature = np.shape(data)
    a = math.sqrt(float(d_feature / d_sub)) * data[:,1:] #a is without the feature
    positive = np.where(data[:,0] == 1)[0]
    one = np.ones((500,500))
    A = np.vstack((np.hstack((one,-one*tradeoff)),np.hstack((-tradeoff*one,one))))
    np.fill_diagonal(A,0)
    D = np.diag(np.sum(A,axis=1))
    L = D-A
    negitive = np.where(data[:,0] != 1 )[0]
    sub = np.vstack((data[positive[:500]],data[negitive[:500]]))[:,1:] #sub is feature 1600*d_feature
    trace = np.diag( np.dot(np.dot(sub.T,L),sub))
    topn = np.argsort(trace)
    topn = np.reshape(topn, (1, -1))
    topn = np.asarray(topn)
    R_nonuniform = sorted(topn[0,:d_sub])
    R_nonuniform = np.asarray(R_nonuniform)
    train2 = a[:, R_nonuniform]
    label = data[:, 0]
    label = (np.asarray(label)).reshape(-1)
    dump_svmlight_file(train2, label, 'other_method/kmeans_linear/%s/%s_nonuniform' % (name, what),
                       zero_based=False)
def read_original(name,n,f_s):
    # train_url = "/home/comp/cszjlei/svm/BudgetedSVM/original/%s/train_%d" % (name, n)
    train_url = "/home/comp/cszjlei/svm/BudgetedSVM/original/%s/train" % name
    train1,tra_label  = load_svmlight_file(train_url)
    tra_label = np.mat(tra_label[:80000]).T
    train1 = train1[:80000]
    n_smaple,n_feature = np.shape(train1)
    # train1, n_feature = ped(train1)
    train = scipy.sparse.hstack((tra_label, train1))
    power = math.ceil(np.log2(n_feature))
    sub = int(math.pow(2, f_s) * n_feature)
    array = np.mat(np.random.uniform(-1, 1, n_feature))  # -1 +1均匀采样
    R = np.random.choice(int(train.shape[1]) - 1, int(sub), replace=False)
    R = sorted(R)
    train1 = train
    random(data=train1, d_sub=sub,name=name, what='original', array=array, R=R,n_label = int(len(tra_label)/2))
    # sampling(data = train1,d_sub = sub,name  = name, what =  'train',tradeoff = trad)
    # corr(data=train1, d_sub=sub, name=name, what='train', array=array, R=R)


def model_training(n,i):
    meth = method[i]
    y, x = liblinearutil.svm_read_problem("other_method/kmeans_linear/%s/original_%s" % (
        name,meth))
    prob = liblinearutil.problem(y, x)
    temp_result = np.empty((13))

    for idx, val in enumerate(cost):
        param = liblinearutil.parameter('-v 5 -q -c %f' % (val))
        temp_result[idx] = liblinearutil.train(prob, param)
    return (i, f_idx, n, temp_result)
    # print(i)


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
    method = ['random','norm']
    np.set_printoptions(threshold=np.inf, suppress=True)
    # method = ['corr','nonuniform']
    tradeoff = [0.2,0.4,0.6,0.8,1]
    f_sub = [-6,-5,-4, -3, -2, -1]
    # f_sub = [-3]
    # f_sub = [-10, -9, -8, -7, -6]
    cost = [2 ** (-10), 2 ** (-8), 2 ** (-6), 2 ** (-5), 2 ** (-4), 2 ** (-3), 2 ** (-2), 2 ** (-1), 1, 2, 4, 8, 16]
    name = '%s'%(param.data)
    acc = np.empty((2,len(f_sub),10,13))
    for f_idx, f_s in enumerate(f_sub):
        pool = Pool(10)
        # sub = int(1024 * math.pow(2, f_s))
        for n in range(10):
            read_original(name, n, f_s)
            f2 = partial(model_training,n)
            for j in range(len(method)):
                 pool.apply_async(f2,(j,),callback=read_acc)
        pool.close()
        pool.join()
    for idx, val in enumerate(method):
        numpy.save('other_method/kmeans_linear/%s/original_%s_result'%(name,val),np.around(acc[idx],decimals=2))

    for m in method:
        os.remove('other_method/kmeans_linear/%s/original_%s'%(name,m))
        #result matrix is an 4 dimension matrix (method,round,acc,cost)