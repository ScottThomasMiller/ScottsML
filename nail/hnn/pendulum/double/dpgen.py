
import torch
import numpy as np
import nail.hnn.utils as utils
from dpdata import DoublePendulumDS

args = utils.get_args()
args.state_symbols = ['q1','q2','p1','p2']
label = utils.get_label(args)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
utils.logmsg("args:\n{}".format(args))
utils.logmsg("label: {}".format(label))
dynsys = DoublePendulumDS(args)
data = dynsys.get_dataset(args.name+".pkl", args.save_dir)
