
#NAIL modules:
import torch
import numpy as np
import ddp.gpuspawn as gpu
import ddp.utils as utils
import ddp.dpdata as dpdata

args = utils.get_args()
label = utils.get_label(args)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
utils.logmsg("args:\n{}".format(args))
utils.logmsg("label: {}".format(label))
data = gpu.load_data(args)
