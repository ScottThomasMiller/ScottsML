from hypers import Hypers
from utils import logmsg, get_args
import torch
import numpy as np

args = get_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
H = Hypers(args.config_filename)
H.search()
Z = H.best()
print('\nbest of cache:')
print(f'Z: {Z}')
H.show_z(Z)
print(f'cost(Z): {H.cache[tuple(Z)]:1.2E}')

