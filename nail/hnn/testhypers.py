from hypers import Hypers
H = Hypers()
H.search()

Z = H.best()
H.show_z(Z)
print(f'lowest cost: {H.cache[tuple(Z)]}')

