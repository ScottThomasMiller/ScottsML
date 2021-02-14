import argparse
import gc
import pickle
import numpy as np
import datetime
import os
import smtplib
import uuid 

#from __future__ import print_function
from email.message import EmailMessage
from sys import getsizeof, stderr
from itertools import chain
from collections import deque
try:
    from reprlib import repr
except ImportError:
    pass

def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)



THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def get_label(args):
    id = uuid.uuid1()
    tstamp = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    label = f"{tstamp}_{id}"

    return label
    
def to_pickle(thing, path, mode='wb'):  # save something
    with open(path, mode) as handle:
        pickle.dump(thing, handle, protocol=pickle.HIGHEST_PROTOCOL)

def from_pickle(path):  # load something
    thing = None
    with open(path, 'rb') as handle:
        thing = pickle.load(handle)
    return thing

def L2_loss(u, v):
    return (u - v).pow(2).mean()

def logmsg(vmsg):
  tstamp = "["+str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))+"]"
  print("{} {}".format(tstamp, vmsg), flush=True)

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--config_filename', 
                        type=str, help='config filename')
    parser.add_argument('--hamiltonian',
                        type=str, help='Hamiltonian of the system')
    parser.add_argument('--clip', type=float, default=0,
                        help='gradient clipping')
    parser.add_argument('--num_bodies', default=1,
                        type=int, help='number of bodies/particles in the system')
    parser.add_argument('--exponent', default=2,
                        type=int, help='exponent of the Hamiltonian for NDSHO')
    parser.add_argument('--cid', 
                        type=int, help='contestant ID for hyperparameter search')
    parser.add_argument('--gid', 
                        type=int, help='group ID for hyperparameter search')
    parser.add_argument('--num_forecasts', default=32,
                        type=int, help='number of forecasts per experiment')
    parser.add_argument('--num_gpus', default=4,
                        type=int, help='number of GPUs')
    parser.add_argument('--epochs', default=32,
                        type=int, help='number of training epochs')
    parser.add_argument('--input_dim', default=4,
                        type=int, help='Input dimension')
    parser.add_argument('--hidden_dim',nargs="*", default=[200, 200],
                        type=int, help='hidden layers dimension')
    parser.add_argument('--beta', default=0.5,
                        type=float, help='beta for custom loss')
    parser.add_argument('--momentum', default=0,
                        type=float, help='momentum for SGD')
    parser.add_argument('--learn_rate', default=1e-03,
                        type=float, help='learning rate')
    parser.add_argument('--eps', default=1e-08,
                        type=float, help='epsilon for numerical stability')
    parser.add_argument('--weight_decay', default=0,
                        type=float, help='weight decay')
    parser.add_argument('--batch_size', default=500,
                        type=int, help='batch size'),
    parser.add_argument('--train_pct', default=100,
                        type=int, help='use pct of total available training data'),
    parser.add_argument('--train_pts', default=0,
                        type=int, help='use number of total available training data'),
    parser.add_argument('--npower', default=16,
                        type=int, help='power of 2 for determining number of training points')
    parser.add_argument('--master_port', default=10101,
                        type=int, help='TCP port number for master process'),
    parser.add_argument('--optim', default='Adam',
                        type=str, help='optimizer')
    parser.add_argument('--input_noise', default='',
                        type=str, help='noise strength added to the inputs')
    parser.add_argument('--print_every', default=200,
                        type=int, help='printing interval')
    parser.add_argument('--integrator_scheme', default='RK45',
                        type=str, help='name of the integration scheme [RK4, RK45, Symplectic]')
    parser.add_argument('--activation_fn', default='Tanh', type=str,
                        help='which activation function to use [Tanh, ReLU]')
    parser.add_argument('--name', default='Henon-Heiles',
                        type=str, help='Name of the system')
    parser.add_argument('--model', default='baseline',
                        type=str, help='baseline or hamiltonian')
    parser.add_argument('--early_stop', default=0, type=float,
                        help='validation loss value for early stopping')
    parser.add_argument('--energy', default=None, type=float,
                        help='fixed energy level')
    parser.add_argument('--split_ratio', default=None, type=float,
                        help='test data split ratio')
    parser.add_argument('--dsr', default=0.01, type=float,
                        help='data sampling rate')
    parser.add_argument('--save_stats', default=False, dest='save_stats', action='store_true',
                        help='save stats.  USES DOUBLE MEMORY')
    parser.add_argument('--verbose', default=False, dest='verbose', action='store_true',
                        help='Verbose output or not')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str,
                        help='where to save the trained model')
    parser.add_argument('--load_model', default='', type=str,
                        help='name of the previously saved model to resume')
    parser.add_argument('--test_pct', default=10,
                        type=int, help='percentage of total train data to use for test set')
    parser.add_argument('--gpu', default=-1,
                        type=int, help='target GPU device #')
    parser.add_argument('--trajectories', default=10000,
                        type=int, help='number of trajectories to generate')
    parser.add_argument('--seed', default=42,
                        type=int, help='random generator seed')
    parser.add_argument('--total_steps', default=4000,
                        type=int, help='No. of steps for gradient descent')
    parser.add_argument('--tspan',nargs="*", default=[0, 1000],
                        type=int, help='timespan')
    parser.add_argument('--state_symbols', nargs="+", default="q p",
                        help='state symbols')
    parser.set_defaults(feature=True)

    return parser.parse_args()

#def showmem(msg):
#    process = psutil.Process(os.getpid())
#    gb = process.memory_info().rss / (1024**3)
#    logmsg(f"{msg}. memory: {gb:.2f}G")

def printmem():
    logmsg("all gc objects:")
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                logmsg(f"type: {type(obj)}, size: {obj.size()}")
        except:
            pass

def printargs(args):
    logmsg("Runtime Args:")
    vargs = vars(args)
    for key in sorted(vargs.keys()):
        logmsg(f"  {key}: {vargs[key]}")

def send_email(to, subject, body):
    # Create a text/plain message
    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = 'stmille3@ncsu.edu'
    msg['To'] = to
    s = smtplib.SMTP('localhost')
    s.send_message(msg)
    s.quit()
