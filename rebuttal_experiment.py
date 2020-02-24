'''
compute the training time, predicting time and accuracy of different dataset
'''
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
import time
from liblinear.python import liblinearutil
import argparse
from numba import jit
import matplotlib.pyplot as plt
from generate_data import generater
from sklearn.preprocessing import *

def model_training():
    x,y = load_svmlight_file("svm/BudgetedSVM/original/%s/train" % (
        name))
    x_test,y_test= load_svmlight_file("svm/BudgetedSVM/original/%s/test" % (
        name))
    # x = np.asarray(x)
    # x_test = np.asarray(x_test)
    # scaler = StandardScaler().fit(x)
    #
    # x = scaler.transform(x)
    prob = liblinearutil.problem(y, x)
    temp_result = np.empty((14))


    # x_test = scaler.transform(x_test)
    param = liblinearutil.parameter(' -q ')
    start = time.process_time()
    m = liblinearutil.train(prob, param)
    print(time.process_time()-start)
    start2 = time.process_time()
    pred_labels, (acc, MSE, SCC), pred_values = liblinearutil.predict(y_test, x_test, m)
    print(time.process_time()-start2,acc)
    exit()
        # print(temp_result[idx])


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
    cost = [ 2 ** (-6), 2 ** (-5), 2 ** (-4), 2 ** (-3), 2 ** (-2), 2 ** (-1), 1, 2, 4, 8, 16,32,64,128]
    name = '%s'%(param.data)
    model_training()
    exit()
    # label_used = [100,500,1000,2000,4000]
    acc = np.empty(( len(f_sub), len(tradeoff), 5, 14))
    summary = np.zeros((1,len(f_sub),5,14))
    temp = np.zeros(5)
    for f_idx,f_s in enumerate(f_sub):
        for j,trad in enumerate(tradeoff):
            for n in range(5):
                model_training()
    # print(acc)


