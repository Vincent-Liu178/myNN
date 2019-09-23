#函数只能处理numpy.array？

import numpy as np

a = [[1,2],
     [3,4],
     [5,6]]
#a = np.array(a)

def fun(x):
    return x**2

print(fun(a))
