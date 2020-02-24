import subprocess
import os
# data1=['a9a','webspam']
# data2 = ['rcv','usps']
# data3 = ['phishing']
split =['a9a','gisette','usps','real-sim','rcv','mushrooms']
data = ['gisette','rcv']
data1 = ['mushrooms','news']
data2 = ['news']

data3 = ['usps','mushrooms'] # for nomuniform not finished
method='nonuniform'
for d in data2:
    # cmd = 'python python/generate_data.py -d %s &' %d
    # os.system(cmd)
    cmd = 'nohup python ./AAAI2020/laplacian.py -d %s &'%(d) #linear_test  for supervised learning
    os.system(cmd)
    # cmd = 'nohup python ./python/weight_sampling.py -d %s &'%(d)
    # os.system(cmd)
    cmd = 'nohup python ./AAAI2020/sparse_embedding.py -d %s &' % (d)
    os.system(cmd)
    # cmd = 'nohup python ./python/leverage_score_sampling.py -d %s &' % (d)
    # os.system(cmd)
    cmd ='nohup python ./AAAI2020/srht_linear.py -d %s &'%(d) #srht for unsupervised and random
    os.system(cmd)
    cmd = 'nohup python ./AAAI2020/gaussian.py -d %s &' % (d)  # linear_test  for supervised learning
    os.system(cmd)