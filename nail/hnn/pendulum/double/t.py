import numpy as np

l1 = np.asarray([7.30044127e-01, 1.26510811e+00 ,9.68099514e-04 ,1.97869278e+02 ,5.52428501e+00])
l2 = np.asarray([0.36081859 ,0.00470767 ,0.72634476 ,4.50331972 ,0.06235374])
c = np.concatenate((l1,l2))
print(f'c: {c}')

a = np.array([[1,1,2],[1,2,3],[3,3,4]] )
print(a)
s = np.array([1,2,3])
idx = np.flatnonzero((a == s).all(1))
print(idx)
print(a[idx][0])

import uuid
print(uuid.uuid1())
