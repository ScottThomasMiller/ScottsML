from hypers import Hypers
H = Hypers()
nfiles = H.collect_all()
Z = H.best()
print(f'Z: {Z}')
H.show_z(Z)
print(f'cost(Z): {H.cache[tuple(Z)]}')

