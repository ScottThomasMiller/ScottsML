
#NAIL modules:
from   ddp.blnn import BLNN
import ddp.gpuspawn as gpu
from   ddp.hnn import HNN
import ddp.utils as utils

if __name__ == "__main__":
    args = utils.get_args()
    utils.logmsg("args:\n{}".format(args))
    label = utils.get_label(args)
    in_dim = 6
    if args.model == 'baseline':
       out_dim = in_dim
       model = BLNN(in_dim, args.hidden_dim, out_dim, args.activation_fn)
    else:
       out_dim = 1
       model = HNN(in_dim, args.hidden_dim, out_dim, args.activation_fn)
    model.set_label(label)
    gpu.spawn_replicas(gpu.run_model, model, args.num_gpus, args)

