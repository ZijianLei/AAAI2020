import fht
from scipy.linalg import hadamard
import numpy as np
import time
from math import  log
import multiprocessing
from multiprocessing import Pool, Queue,Manager
import ffht

def f(i):
    for j in range(i*1000,(i+1)*1000):
        ffht.fht(b[j])
    return b[i*1000:(i+1)*1000],i
def read(rea):
    data,i = rea
    b[i*1000:(i+1)*1000] = data

a = np.random.randn(40000,8192)
# a = np.ones((4,4))
b = np.random.randn(40000,8192)
# pool = Pool(20)
start = time.time()

for i in range(a.shape[0]):
    # pool.apply_async(fht.fht, (a[i],))
    a[i] = fht.fht(a[i])
# pool.close()
# pool.join()
print(time.time()-start)
print(b[0])
start = time.time()
pool = Pool(40)
for i in range(40):
    pool.apply_async(f, (i,),callback = read)
pool.close()
pool.join()
print(time.time()-start)
print(b[0])