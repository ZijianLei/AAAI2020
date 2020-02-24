import numpy as np
import numpy.matlib
import gc
from read_dataset import *
import scipy.sparse
from sklearn.datasets import dump_svmlight_file
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
from scipy.sparse.linalg import *
from numpy.linalg import *
if __name__ == '__main__':
    d_feature = 2
    n_sample = 400
    array = np.mat(np.random.uniform(-1,1,n_sample))
    array[array > 0] = 1
    array[array < 0] = -1
    array = array.reshape(-1,1)
    feature = np.mat(np.empty((n_sample,d_feature)))
    h = hadamard(2)
    i = 0
    for val in array:
        if val == 1:
            feature[i,:] = np.random.multivariate_normal([3,3],[[1, 1.5], [1.75, 1]])
            # feature[i, 1] = 0.5*np.matlib.randn(1, 1)
            i = i+1
        elif val == -1:
            # feature[i, :] = np.matlib.randn(1, 1)+3
            # feature[i, 1] = 0.5*np.matlib.randn(1, 1)+3
            feature[i, :] = np.random.multivariate_normal([6, 6], [[1, 1.75], [1.5, 1]])
            i = i+1
    # data = np.vstack(feature)
    # print(data.shape)
    u, s, vt = svd(feature)
    data = np.dot(feature,h)
    print(vt.shape)
    print(u.shape,s.shape,vt.shape)
    norm = np.linalg.norm(feature, axis=0)
    print(norm)
    feature =  np.hstack((array,feature))
    positive = feature[np.where(feature[:,0] == 1)[0],:]
    negitive = feature[np.where(feature[:,0] == -1)[0],:]
    plt.scatter(positive[:,1].tolist(),positive[:,2].tolist(),marker='P',s = 80,label = 'c+' )
    plt.scatter(negitive[:,1].tolist(),negitive[:,2].tolist(),marker='o',s = 80,label = 'c-' )

    plt.legend(fontsize=15)
    # plt.title("%s" % d, fontsize=20)
    plt.xlabel("feature 1", fontsize=15)
    plt.ylabel("feature 2", fontsize=15)
    plt.savefig('other_method/synthetic.png')
    plt.show()
    array = [1,1]
    positive = np.dot(positive[:,1:],h)
    negitive = np.dot(negitive[:,1:],h)
    # data = np.dot(data,h)
    u, s, vt = svd(data)
    norm = np.linalg.norm(data, axis=0)
    print(u, s, vt)
    print(norm)
    plt.scatter(positive[:, 0].tolist(), positive[:, 1].tolist(),marker='P',s = 80,label = 'c+')
    plt.scatter(negitive[:, 0].tolist(), negitive[:, 1].tolist(),marker='o',s = 80,label = 'c-' )
    plt.legend(fontsize=15)
    # plt.title("%s" % d, fontsize=20)
    plt.xlabel("feature 1", fontsize=15)

    plt.ylabel("feature 2", fontsize=15)
    plt.ylim(-7,7)
    plt.savefig('other_method/transformed.png')
    plt.show()