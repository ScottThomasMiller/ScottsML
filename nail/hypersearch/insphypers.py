import numpy as np

from hypers import Hypers
H = Hypers()
Z = H.best()
print(f'min Z: {Z}')
H.show_z(Z)
print(f'lowest cost: {H.cache[tuple(Z)]}')

Z = np.array([5,2,3], dtype=int)
print(f'convergence Z: {Z}')
H.show_z(Z)
print(f'cost(Z): {H.cache[tuple(Z)]}')

