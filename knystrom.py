import matplotlib
import os
import subprocess
import gc
from  read_dataset import  *
# import fht

from multiprocessing import Pool, Queue
import  multiprocessing
import numpy as np
import  scipy.sparse
from functools import  partial
from sklearn.datasets import dump_svmlight_file
if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf,suppress=True)
    lamda = [ -2]
    gamma = [ -2]
    landmark = [500, 1000, 3000]
    name ='a9a'

    test_url = "/home/comp/cszjlei/svm/BudgetedSVM/original/%s/test" %(name)
    train_url = "/home/comp/cszjlei/svm/BudgetedSVM/original/%s/train" %(name)

    tra_label, train = read_mnist(train_url)
    te_label, test = read_mnist(test_url)
    for land in landmark:
        for lamb in lamda:
            for gam in gamma:
                print('landmark = %d  lambda = %f gamma = %f  ' % (land, math.pow(2, lamb), math.pow(2, gam)))
                for time in range(30):
                    subprocess.run(
                        "other_method/BudgetedSVM/bin/budgetedsvm-train -A 3 -L %f -K 0 -g %f -B %d -m 1  -v 0  svm/BudgetedSVM/original/%s/train svm/BudgetedSVM/original/%s/model.txt" % (
                            math.pow(2, lamb), math.pow(2, gam), land, name, name), shell=True)
                    subprocess.run(
                        "other_method/BudgetedSVM/bin/budgetedsvm-predict -v 1 svm/BudgetedSVM/original/%s/test svm/BudgetedSVM/original/%s/model.txt svm/BudgetedSVM/original/%s/predictions2.txt" % (
                            name, name, name), shell=True)