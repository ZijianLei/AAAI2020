import numpy as np
import ffht
import timeit
import  time
import sys
print(ffht.__file__)
reps = 50000
n = 2**12
chunk_size = 1024

a = np.random.randn(n).astype(np.float32)
print(a.shape)
t1 = timeit.default_timer()
start = time.time()
for i in range(reps):
    ffht.fht(a)
print(start-time.time())
t2 = timeit.default_timer()

if sys.version_info[0] == 2:
    print (t2 - t1 + 0.0) / (reps + 0.0)
if sys.version_info[0] == 3:
    print('{}'.format((t2 - t1 + 0.0)))
