''' Methods for running NAIL models across distributed GPUs via PyTorch's Distributed Data Parallel (DDP).

    With guidance from https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
'''

from .blnn import BLNN
from .data import DataStream, DynamicalSystem
from .utils import logmsg, to_pickle, from_pickle

import copy
import datetime
import numpy as np
import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import signal

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
state_symbols = ['q', 'p']

def sighandler(signum, frame):
    ''' experimental  
    '''
    logmsg('Signal handler called with signal'.format(signum))
    cleanup()
    exit -1 * signum

def average_stats(estats):
    ''' Update the epoch stats of all GPUs with the average of the
        epoch stats from all GPUs.  '''
    size = float(dist.get_world_size())
    dist.all_reduce(estats, op=dist.ReduceOp.SUM)
    estats /= size

def average_model(model):
    ''' All-reduce the model's parameters with averages of the parameters.
    '''
    size = float(dist.get_world_size())
    for name, param in model.named_parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= size

def average_models(models):
    ''' Update and return the first model's parameters with averages of all models' parameters, 
        from the given list.
    '''
    for name, param in models[0].named_parameters():
        for i in range(1,len(models)):
            param.data += models[i].state_dict()[name]
        param.data /= 4

    return models[0]
    
def setup(rank, args, world_size):
    ''' Initialize and activate the network of GPUs, and set the random seed.
    '''
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(args.master_port)

    # initialize the process group
    # gloo is only 1/30th slower than nccl but uses half the GPU and less CPU:
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    #dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

def cleanup():
    dist.destroy_process_group()

def spawn_replicas(run_fn, model, world_size, args):
    ''' Run replicas of the model on each GPU.
    '''
    mp.spawn(run_fn,
             args=(model,world_size,args,),
             nprocs=world_size,
             join=True)

def load_model(model, model_path, device=None):
    if (model_path is None) or (model_path == ''):
        return False
    try:
        saved_model = from_pickle(model_path)
        model.load_state_dict(saved_model['model'])
        if (device is None) or (device == 0):
            logmsg("HELLO! loaded model: {}".format(model_path))
        return True
    except Exception as ex:
        if (device is None) or (device == 0):
           logmsg("ERROR: exception occurred trying to load model {} "+str(model_path))
           logmsg("EXCEPTION: "+str(ex))
        return False

def append_stats(stats_list, stats):
    train_loss = stats['training'][-1]
    test_loss = stats['testing'][-1]
    grad_norm = stats['grad_norms'][-1]
    grad_std = stats['grad_stds'][-1]
    stats_list['training'].append(train_loss)
    stats_list['testing'].append(test_loss)
    stats_list['grad_norms'].append(grad_norm)
    stats_list['grad_stds'].append(grad_std)
    estats = torch.Tensor([train_loss, test_loss, grad_norm, grad_std])

    return estats

def to_gpu(input_data, device, world_size):
    ''' Copy input_data from CPU to GPU: '''
    len_train = input_data['coords'].shape[0]
    chunk_train = int(len_train / world_size)
    start_train = int(device * chunk_train)
    stop_train = start_train + chunk_train
    stop_train = len_train if stop_train > len_train else stop_train
    chunk_coords = input_data['coords'][start_train:stop_train]
    chunk_dcoords = input_data['dcoords'][start_train:stop_train]
    input_data = {
        'coords': torch.tensor(chunk_coords, dtype=torch.float32, requires_grad=True).to(device),
        'dcoords': torch.tensor(chunk_dcoords, dtype=torch.float32, requires_grad=True).to(device),
        'test_coords': torch.tensor(input_data['test_coords'], dtype=torch.float32, requires_grad=True).to(device),
        'test_dcoords': torch.tensor(input_data['test_dcoords'], dtype=torch.float32, requires_grad=True).to(device)}

    return input_data

def prep_data(args, world_size, device):
    input_data = load_data(args)
    data = to_gpu(input_data, device, world_size)
    train_data, test_data = to_dataset(args, data)
    if device == 0:
        logmsg("total data->training: {:,}, testing: {:,}".format(input_data['coords'].shape[0], input_data['test_coords'].shape[0]))
  
    return train_data, test_data

def prep_paths(args, run_label, device):
    paths = {}
    save_model_base = '{}/{}_model_{}-gpu{}.trch'
    load_path = "{}/{}".format(args.save_dir, args.load_model)
    paths['load_model'] = load_path if args.load_model != '' else ''
    paths['save_model'] = save_model_base.format(args.save_dir, args.model, run_label, device)
    paths['save_lowest'] = f"{args.save_dir}/lowest_{args.model}_model_{run_label}_gpu{device}.trch"
    paths['stats_list'] = '{}/{}_stats_{}-gpu{}.trch'.format(args.save_dir, args.model, run_label, device)

    return paths

def early_stop(args, device):
    stop = torch.Tensor([1]).to(device)
    logmsg("gpu[{}] Early stopping at test threshold {}".format(device, args.early_stop))
    dist.all_reduce(stop, op=dist.ReduceOp.SUM)

def run_model(rank, cpu_model, world_size, args):
    ''' Setup and run the model:  
	- derive the device ID from the given rank
	- register with signal handlers
	- convert the model to DDP
	- set the optimizer
        - create the save dir 
	- load optional saved model
        - load the input data
        - train the model
        - checkpoint the model and the stats
    '''
    setup(rank, args, world_size)
    n = torch.cuda.device_count() // world_size
    if args.gpu >= 0:
      device = args.gpu
      device_ids = [args.gpu]
    else:
      device_ids = list(range(rank * n, (rank + 1) * n))
      device = device_ids[0]
    signal.signal(signal.SIGTERM, sighandler)
    signal.signal(signal.SIGINT, sighandler)
    if rank == 0:
        os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    model = cpu_model.to(device)
    del cpu_model
    ddp_model = DDP(model, device_ids=device_ids)
    del ddp_model
    save_model = {'args': args, 'model': model.state_dict()}
    optimizer = get_optimizer(model, args)
    if args.save_stats:
        stats_list = copy.deepcopy(BLNN.stats_dict)
    paths = prep_paths(args, model.run_label, device)
    loaded = load_model(model, paths['load_model'], device)
    train_data, test_data = prep_data(args, world_size, device)
    if device == 0:
        logmsg("start model {}, file label: {}".format(args.model, model.run_label))

    lowest_loss = 100.0
    model.init_device()
    for e in range(args.epochs):
        stats = model.etrain(args, train_data, optimizer)
        stats['testing'] = [model.validate(args, test_data, device)]
        train_loss = stats['training'][-1]
        test_loss = stats['testing'][-1]
        grad_norm = stats['grad_norms'][-1]
        grad_std = stats['grad_stds'][-1]
        if args.save_stats:
            estats = append_stats(stats_list, stats).to(device)
            to_pickle(stats_list, paths['stats_list'])
            #average_stats(estats)
        average_model(model)
        save_model['model'] = model.state_dict()
        to_pickle(save_model, paths['save_model'])
        if device == 0:
            logmsg(f"epoch {e} (avgs) loss->train:{train_loss:.4e} test:{test_loss:.4e} " + \
                   f"grads->norm:{grad_norm:.4e} std:{grad_std:.4e}")
        if test_loss < lowest_loss:
            lowest_loss = test_loss
            to_pickle(save_model, paths['save_lowest'])
        if test_loss <= args.early_stop:
            early_stop(args, device)
            break

    cleanup()

def load_data(args):
    ''' Return the dataset from the given DyanimcalSystem, for the given or current GPU.
    '''
    tspan = []
    for t in args.tspan:
        tspan.append(t)
    tpoints = int((1.0 / args.dsr) * tspan[1])
    path = args.name+".pkl"
    data = DynamicalSystem.read_dataset(path, args.save_dir)
    if args.train_pts == 0:
      len_train = int(data['coords'].shape[0] * (args.train_pct/100.0))
    else:
      len_train = args.train_pts
    len_test = int(data['test_coords'].shape[0] * (args.test_pct/100.0))
    shuffle = torch.randperm(data['coords'].shape[0])
    data['coords'] = data['coords'][shuffle][:len_train]
    data['dcoords'] = data['dcoords'][shuffle][:len_train]
    shuffle = torch.randperm(data['test_coords'].shape[0])
    data['test_coords'] = data['test_coords'][shuffle][:len_test]
    data['test_dcoords'] = data['test_dcoords'][shuffle][:len_test]

    return data

def to_dataset(args, data):
    train_len = data['coords'].shape[0]
    test_len = data['test_coords'].shape[0]
    device = dist.get_rank()
    training = {'x': data['coords'], 'dxdt': data['dcoords']}
    train_data = DataLoader(dataset=DataStream(training), batch_size=args.batch_size, shuffle=True, pin_memory=False)
    testing = {'x': data['test_coords'], 'dxdt': data['test_dcoords']}
    test_data = DataLoader(dataset=DataStream(testing), batch_size=test_len, shuffle=True, pin_memory=False)
    if device == 0:
        logmsg("gpu dataset sizes-> train: {:,}, test: {:,}".format(train_len, test_len))

    return train_data, test_data

def get_optimizer(model, args):
  name = args.optim.lower()
  if name ==  'adadelta': 
    optim = torch.optim.Adadelta(model.parameters(), args.learn_rate, weight_decay=args.weight_decay, eps=args.eps)
  elif name ==  'adagrad': 
    optim = torch.optim.Adagrad(model.parameters(), args.learn_rate, weight_decay=args.weight_decay, eps=args.eps)
  elif name ==  'adam': 
    optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=args.weight_decay, eps=args.eps)
  elif name ==  'adamax': 
    optim = torch.optim.Adamax(model.parameters(), args.learn_rate, weight_decay=args.weight_decay, eps=args.eps)
  elif name ==  'adamw': 
    optim = torch.optim.AdamW(model.parameters(), args.learn_rate, weight_decay=args.weight_decay, eps=args.eps)
  elif name ==  'asgd': 
    optim = torch.optim.ASGD(model.parameters(), args.learn_rate, weight_decay=args.weight_decay,
                             momentum=args.momentum)
  elif name ==  'rmsprop': 
    optim = torch.optim.RMSprop(model.parameters(), args.learn_rate, weight_decay=args.weight_decay, eps=args.eps)
  elif name ==  'rprop': 
    optim = torch.optim.Rprop(model.parameters(), args.learn_rate, weight_decay=args.weight_decay)
  elif name ==  'sgd': 
    optim = torch.optim.SGD(model.parameters(), args.learn_rate, weight_decay=args.weight_decay,
                            momentum=args.momentum, eps=args.eps)
  elif name ==  'sparseadam': 
    optim = torch.optim.SparseAdam(model.parameters(), args.learn_rate, weight_decay=args.weight_decay, eps=args.eps)

  return optim
