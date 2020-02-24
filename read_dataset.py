import pandas as pd #用于读取csv数据的一个数据包
import numpy as np
from sklearn.preprocessing import *
import math
import sys
import scipy
from scipy import sparse
import numpy.matlib
import re


def read_mnist(data_file_name,return_scipy = True):
    prob_y = []
    prob_x = []
    row_ptr = [0]
    col_idx = []
    for i, line in enumerate(open(data_file_name)):
        line = line.split(None, 1)
        # In case an instance with all zero features
        if len(line) == 1: line += ['']
        label, features = line
        prob_y += [float(label)]
        if scipy != None and return_scipy:
            nz = 0
            for e in features.split():
                ind, val = e.split(":")
                val = float(val)
                if val != 0:
                    col_idx += [int(ind) - 1]
                    prob_x += [val]
                    nz += 1
            row_ptr += [row_ptr[-1] + nz]
        else:
            xi = {}
            for e in features.split():
                ind, val = e.split(":")
                xi[int(ind)] = float(val)
            prob_x += [xi]
    if scipy != None and return_scipy:
        prob_y = scipy.array(prob_y)
        prob_x = scipy.array(prob_x)
        col_idx = scipy.array(col_idx)
        row_ptr = scipy.array(row_ptr)
        prob_x = sparse.csr_matrix((prob_x, col_idx, row_ptr))
    return prob_y, prob_x


def pedding(data):
    n_number , f_num = np.shape(data)
    power = math.ceil(np.log2(f_num))
    for i in range(f_num - 2, 2 ** power):
        data = np.insert(i, 'pad%d' % i, 0)
    return data

def ped(data):
    n_number, f_num = np.shape(data)
    # data  = scipy.sparse.coo_matrix.todense(data)
    power = math.ceil(np.log2(f_num))
    np.pad(data, (0, 2**power - f_num), 'constant', constant_values=(0, 0))
    return data,2**power


