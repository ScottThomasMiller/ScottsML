
#NAIL modules:
from   blnn import BLNN
import train2
from   hnn2 import HNN
import utils as utils

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
    model.run_label = utils.get_label(args) 
    train2.run_model(model, args)

