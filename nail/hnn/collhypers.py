from hypers import Hypers
from utils import logmsg, get_args

args = get_args()

H = Hypers(args.config_filename)
nfiles = H.collect_all()
Z = H.best()
print(f'Z: {Z}')
H.show_z(Z)
print(f'cost(Z): {H.cache[tuple(Z)]}')

