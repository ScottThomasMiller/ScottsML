import numpy as np
from glob import glob

filenames = glob('/share/nail/stmille3/save/Hypers.archive/oldpop_*.npy')
filenames.sort()
oldpop = np.load(filenames[0])
for i in range(len(filenames)):
  print(f'pop {i}: {filenames[i]}')
print(' ')
for i in range(1,len(filenames)):
  newpop = np.load(filenames[i])
  if np.array_equal(oldpop, newpop):
    print(f'pop {i-1} and pop {i} are equal')
  else:
    equals = np.equal(oldpop, newpop)
    ones = np.ones(oldpop.shape)
    nsame = len(ones[equals])
    print(f'pop {i-1} and pop {i} are NOT equal.  num diff: {oldpop.size - nsame}')
  oldpop = newpop.copy()
  
