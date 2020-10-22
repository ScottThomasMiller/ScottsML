
''' This script is used for GPU training of the simple pendulum. '''

#nail.hnn.modules:
from   nail.hnn.blnn import BLNN
from   sphnn import HNN
import nail.hnn.gpu as gpu
import nail.hnn.utils as utils

if __name__ == "__main__":
    args = utils.get_args()
    utils.logmsg("args:\n{}".format(args))
    label = utils.get_label(args)
    in_dim = 3
    if args.model == 'baseline':
       out_dim = in_dim
       model = BLNN(in_dim, args.hidden_dim, out_dim, args.activation_fn)
    else:
       out_dim = 1
       model = SimplePendulumHNN(in_dim, args.hidden_dim, out_dim, args.activation_fn)
    model.set_label(label)
    gpu.spawn_replicas(gpu.run_model, model, args.num_gpus, args)

