# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
#import scikit_image-0.12.3.dist-info
import argparse
import sys
import ffht
import math
import time
from sklearn.metrics import *
from sklearn.datasets import load_svmlight_file
import scipy
import numpy as np
import pandas
from scipy.stats import chi

from scipy._lib._util import _asarray_validated


def get_data(name,what):
  data = load_svmlight_file("./svm/BudgetedSVM/original/%s/%s" %(name,what))
  return data[1],data[0]



def logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
  a = _asarray_validated(a, check_finite=False)
  if b is not None:
    a, b = np.broadcast_arrays(a, b)
    if np.any(b == 0):
      a = a + 0.  # promote to at least float
      a[b == 0] = -np.inf

  a_max = np.amax(a, axis=axis, keepdims=True)

  if a_max.ndim > 0:
    a_max[~np.isfinite(a_max)] = 0
  elif not np.isfinite(a_max):
    a_max = 0

  if b is not None:
    b = np.asarray(b)
    tmp = b * np.exp(a - a_max)
  else:
    tmp = np.exp(a - a_max)

  # suppress warnings about log of zero
  with np.errstate(divide='ignore'):
    s = np.sum(tmp, axis=axis, keepdims=keepdims)
    if return_sign:
      sgn = np.sign(s)
      s *= sgn  # /= makes more sense but we need zero -> zero
    out = np.log(s)

  if not keepdims:
    a_max = np.squeeze(a_max, axis=axis)
  out += a_max

  if return_sign:
    return out, sgn
  else:
    return out

def softmax(x):
  return np.exp(x - logsumexp(x, axis=1, keepdims=True))


def hadamard(d,f_num,batch,G,B,PI_value,S):
    T = FLAGS.T
    x_ = batch
    x_ = np.pad(x_, ((0,0),(0, d - f_num)), 'constant', constant_values=(0, 0))  # x.shape [batch,d]
    x_ = np.tile(x_, (1, T))
    x_i = np.multiply(x_, S)
    x_ = x_i.reshape(FLAGS.BATCHSIZE, T, d)
    h = 1
    # while h < d:
    #     for i in range(0, d, h * 2):
    #         for j in range(i, i + h):
    #             a = x_[:, :, j]
    #             b = x_[:, :, j + h]
    #             temp = a - b
    #             x_[:, :, j] = a + b
    #             x_[:, :, j + h] = temp
    #     h *= 2
    for i in range(x_.shape[0]):
        for j in range(T):
            ffht.fht(x_[i,j])
    x_transformed = np.multiply(x_.reshape(FLAGS.BATCHSIZE, d * T), G)
    x_transformed = np.reshape(x_transformed, (FLAGS.BATCHSIZE, T, d))
    x_permutation = x_transformed[:, :, PI_value]
    x_ = x_permutation
    h = 1

    while h < d:
        for i in range(0, d, h * 2):
            for j in range(i, i + h):
                a = x_[:, :, j]
                b = x_[:, :, j + h]
                temp = a - b
                x_[:, :, j] = a + b
                x_[:, :, j + h] = temp
        h *= 2



    # for i in range(x_.shape[0]):
    #     for j in range(T):
    #         ffht.fht(x_[i,j])

    x_ = x_.reshape(FLAGS.BATCHSIZE, T * d)
    x_value = np.multiply(x_, B)
    x_value = np.sign(x_value)
    return x_value


def main(name ):
    y, x = get_data(name, 'train')
    n_number, f_num = np.shape(x)
    d = 2 ** math.ceil(np.log2(f_num))
    T = FLAGS.T
    G = np.random.randn(T * d)
    B = np.random.uniform(-1, 1, T * d)
    B[B > 0] = 1
    B[B < 0] = -1
    PI_value = np.random.permutation(d)
    G_fro = G.reshape(T, d)
    s_i = chi.rvs(d, size=(T, d))
    S = np.multiply(s_i, np.array(np.linalg.norm(G_fro, axis=1) ** (-0.5)).reshape(T, -1))
    S = S.reshape(1, -1)
    print('Training LSH')
    W_fcP = np.asmatrix(-np.ones((T*d,1)))
    x = x.todense()
    y = np.where(y[:] != 1, -1, 1)
    n_number, f_num = np.shape(x)
    FLAGS.BATCHSIZE = n_number
    d = 2 ** math.ceil(np.log2(f_num))
    y = y.reshape(n_number,1)

    # x = sklearn.preprocessing.scale(x)
    x_value = np.asmatrix(hadamard(d, f_num, x, G, B, PI_value, S))


    '''
    #regression initialize
    reg = LinearRegression(fit_intercept=False).fit(x_value,y)
    predict = reg.predict(x_value)
    print(sklearn.metrics.mean_squared_error(y_temp,reg.predict(x_value) ))
    print(np.hstack((y_temp[:20], predict.reshape(n_number,1)[:20])))
    W_fcP = (np.sign(reg.coef_)).reshape(T*d,1)
    '''

    # print(x_value.shape)
    predict = np.dot(x_value, W_fcP) # classification
    print(predict.shape)
    #loss_init = sklearn.metrics.mean_squared_error(y, predict)
    hinge_loss = np.max(np.hstack((np.zeros((n_number,1)),1-np.multiply(y,predict))),axis = 1)
    loss_init = np.sum(hinge_loss)
    print(loss_init,0)
    loss_0 = 0
    init = np.dot(x_value,W_fcP)
    j = 1
    while loss_0 != loss_init:
        loss_0 = loss_init
        for i in range(T*d):
            derta = init-np.multiply(W_fcP[i],x_value[:,i])*2
            hinge_loss = np.max(np.hstack((np.zeros((n_number, 1)),  1- np.multiply(y, derta))), axis=1)
            loss = np.sum(hinge_loss)
            if loss < loss_init:
                loss_init = loss
                init = derta
                W_fcP[i] = -W_fcP[i]
        print(loss_init,j)
        j+=1
    predict = np.sign(np.array(np.dot(x_value, W_fcP)))
    acc = accuracy_score(np.array(y), np.array(predict))
    print(acc,'train')

    y, x = get_data(name, 'test')
    y = np.where(y[:] != 1, -1, 1)
    n_number, f_num = np.shape(x)
    FLAGS.BATCHSIZE = n_number
    x = x.todense()
    x_value = np.asmatrix(hadamard(d, f_num, x, G, B, PI_value, S))
    predict = np.sign(np.array(np.dot(x_value, W_fcP)))
    acc = accuracy_score(np.array(y), np.array(predict))
    print(acc,'test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-T', type=int,
                        default=1,
                        help='Directory for storing input data')
    parser.add_argument('--BATCHSIZE', type=int,
                        default=128,
                        help='Directory for storing input data')
    parser.add_argument('-d', type=str,
                        default='a9a',
                        help='Directory for storing input data')
    np.set_printoptions(threshold=np.inf, suppress=True)
    FLAGS, unparsed = parser.parse_known_args()
    name = FLAGS.d
    main(name)
