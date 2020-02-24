import sklearn
import time
from sklearn.svm import LinearSVC
from read_dataset import  *
from sklearn.tree import DecisionTreeClassifier
import fht
import nystrom
from pandas.tools.plotting import parallel_coordinates
import matplotlib.pyplot as plt
# load the dataset
test_url = "\adult.test"
train_url = "\adult.data"

#'''
#使用第一种数值变换的方式进行数据处理
train, test,tra_num,te_num,f_num = read_data1(train_url, test_url)
test, te_label= cleandata(test,f_num)
train, tra_label = cleandata(train,f_num)

tra_label = np.array(tra_label).reshape(-1,1)
te_label = np.array(te_label).reshape(-1,1)
test = freature_encode(test)
train = freature_encode(train)

plt.figure()
parallel_coordinates(train,'target-labels')
plt.show()