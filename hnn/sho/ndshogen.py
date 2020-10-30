
import torch
import numpy as np
import nail.hnn.utils as utils
from ndshodata import SimpleHarmonicOscillatorDS 

args = utils.get_args()
args.state_symbols = ['q','p']
label = utils.get_label(args)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
utils.logmsg("args:\n{}".format(args))
utils.logmsg("label: {}".format(label))
dynsys = SimpleHarmonicOscillatorDS(args)
data = dynsys.get_dataset(args.name+".pkl", args.save_dir)
