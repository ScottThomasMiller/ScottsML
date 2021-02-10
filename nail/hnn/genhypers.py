from utils import logmsg, get_args
from hypers import Hypers

args = get_args()
H = Hypers(args.config_filename)
H.prep_all()

