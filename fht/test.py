from fht import *
from scipy.linalg import hadamard
import numpy as np
import time
h = hadamard(4)
a = np.random.randn((10.4))
c = np.dot(a,h)
start = time.time()
a = fht.fht(a)
print(c -a)
