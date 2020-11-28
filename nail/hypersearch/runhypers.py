from hypers import Hypers
H = Hypers()
H.search()

Z = H.best()
print(f'Z: {Z}')
H.show_z(Z)
print(f'cost(Z): {H.cache[tuple(Z)]}')

