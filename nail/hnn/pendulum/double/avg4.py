
import argparse
import torch

#NAIL modules:
from ddp.blnn import BLNN
import ddp.gpuspawn as gpu
from ddp.hnn import HNN
import ddp.gpuspawn as gpu
from ddp.utils import from_pickle, to_pickle

parser = argparse.ArgumentParser(description=None)
parser.add_argument('-base',
                    type=str, help='base filename for model')
parser.add_argument('--clip', type=float, default=0,
                    help='gradient clipping')
parser.add_argument('--num_gpus', default=4,
                    type=int, help='number of GPUs')
args = parser.parse_args()
models = []
#for i in range(args.num_gpus):
for i in range(4):
  saved_model = from_pickle(args.base.format(i))
  for k, v in saved_model.items():
    print("k: {}".format(k))
  if saved_model['args'].model == 'baseline':
    models.append(BLNN(6, saved_model['args'].hidden_dim, 6, 'Tanh'))
  else:
    models.append(HNN(6, saved_model['args'].hidden_dim, 1, 'Tanh'))
  models[i].load_state_dict(saved_model['model'])
avg_model = gpu.average_models(models)
saved_model['model'] = avg_model.state_dict()
to_pickle(saved_model, args.base.format("-avg"))

