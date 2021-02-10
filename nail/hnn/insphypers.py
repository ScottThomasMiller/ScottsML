import numpy as np

from hypers import Hypers
H = Hypers()
Z = H.best()
print(f'min Z: {Z}')
H.show_z(Z)
print(f'lowest cost: {H.cache[tuple(Z)]:1.2E}')

# min Z: (0, 1, 4)
# Z = np.array([0,1,4], dtype=int)
#for bs in range(H.cache.shape[0]):
#  Z = np.array([bs, 1, 4], dtype=int)
#  print(f"bs: {H.Hparams['batch_size'][bs]} cost: {H.cache[tuple(Z)]:1.2E}")

