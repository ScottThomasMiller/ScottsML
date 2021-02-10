''' Methods for running NAIL models across distributed GPUs via PyTorch's Distributed Data Parallel (DDP).

    With guidance from https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
'''

from nail.hnn.blnn import BLNN
from nail.hnn.data import DataStream, DynamicalSystem
from nail.hnn.utils import logmsg, to_pickle, from_pickle

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

def to_tensor(input_data):
    ''' Copy input_data from CPU to GPU: '''
    len_train = input_data['coords'].shape[0]
    coords = torch.tensor(input_data['coords'], dtype=torch.float32, requires_grad=True)
    dcoords = torch.tensor(input_data['dcoords'], dtype=torch.float32, requires_grad=True)
    tcoords = torch.tensor(input_data['tcoords'], dtype=torch.float32, requires_grad=True)
    test_coords = torch.tensor(input_data['test_coords'], dtype=torch.float32, requires_grad=True)
    test_dcoords = torch.tensor(input_data['test_dcoords'], dtype=torch.float32, requires_grad=True)
    test_tcoords = torch.tensor(input_data['test_tcoords'], dtype=torch.float32, requires_grad=True)
    input_data = {
        'coords': coords,
        'dcoords': dcoords,
        'tcoords': tcoords,
        'test_coords': test_coords,
        'test_dcoords': test_dcoords,
        'test_tcoords': test_tcoords}

    return input_data

def prep_data(args):
    input_data = load_data(args)
    data = to_tensor(input_data)
    train_data, test_data = to_dataset(args, data)
    logmsg(f"prep_data. sizes->training: {input_data['coords'].shape[0]}, testing: {input_data['test_coords'].shape[0]}")
  
    return train_data, test_data

def prep_paths(args, run_label, device):
    paths = {}
    save_model_base = '{}/{}_model_{}.trch'
    load_path = "{}/{}".format(args.save_dir, args.load_model)
    paths['load_model'] = load_path if args.load_model != '' else ''
    paths['save_model'] = save_model_base.format(args.save_dir, args.model, run_label)
    paths['save_lowest'] = f"{args.save_dir}/lowest_{args.model}_model_{run_label}.trch"
    paths['stats_list'] = '{}/{}_stats_{}.trch'.format(args.save_dir, args.model, run_label)

    return paths

def early_stop(args, device):
    stop = torch.Tensor([1]).to(device)
    logmsg("device[{}] Early stopping at test threshold {}".format(device, args.early_stop))
    dist.all_reduce(stop, op=dist.ReduceOp.SUM)

def dev0log(device, msg):
    if device <= 0:
        logmsg(msg)

def rank0log(rank, msg):
    if rank == 0:
        logmsg(msg)

def load_data(args):
    ''' Return the dataset from the given DyanimcalSystem, for the given or current GPU.
    '''
    logmsg(f'load_data->train_pts: {args.train_pts} train_pct: {args.train_pct} test_pct: {args.test_pct}')
    tspan = []
    for t in args.tspan:
        tspan.append(t)
    tpoints = int((1.0 / args.dsr) * tspan[1])
    path = args.name+".pkl"
    data = DynamicalSystem.read_dataset(path, args.save_dir)
    ''' create the time-shifted versions of coords and test_coords: '''
    data['tcoords'] = np.array(data['coords'][1:])
    data['test_tcoords'] = np.array(data['test_coords'][1:])
    ''' trim the last item from coords and test_coords because they have no time-shifted values: '''
    data['coords']  = data['coords'][:-1] 
    data['test_coords']  = data['test_coords'][:-1] 
    ''' adjust the len_train if necessary: '''
    if args.train_pts == 0:
      len_train = int(data['coords'].shape[0] * (args.train_pct/100.0))
    else:
      len_train = args.train_pts
    len_test = int(data['test_coords'].shape[0] * (args.test_pct/100.0))
    ''' construct the results: '''
    shuffle = torch.randperm(data['coords'].shape[0])
    data['coords'] = data['coords'][shuffle][:len_train]
    data['tcoords'] = data['tcoords'][shuffle][:len_train]
    data['dcoords'] = data['dcoords'][shuffle][:len_train]
    shuffle = torch.randperm(data['test_coords'].shape[0])
    data['test_coords'] = data['test_coords'][shuffle][:len_test]
    data['test_tcoords'] = data['test_tcoords'][shuffle][:len_test]
    data['test_dcoords'] = data['test_dcoords'][shuffle][:len_test]
    logmsg(f"load_data.  args.train_pts: {args.train_pts}, len_train: {len_train}, len_test: {len_test}")

    return data

def to_dataset(args, data):
    train_len = data['coords'].shape[0]
    test_len = data['test_coords'].shape[0]
    #device = dist.get_rank()
    logmsg(f'data keys: {data.keys()}')
    training = {'x': data['coords'], 'dxdt': data['dcoords'], 'tx': data['tcoords']}
    train_data = DataLoader(dataset=DataStream(training), batch_size=args.batch_size, shuffle=True, pin_memory=False)
    testing = {'x': data['test_coords'], 'dxdt': data['test_dcoords'], 'tx': data['test_tcoords']}
    test_data = DataLoader(dataset=DataStream(testing), batch_size=test_len, shuffle=True, pin_memory=False)

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
    optim = torch.optim.ASGD(model.parameters(), args.learn_rate, weight_decay=args.weight_decay)
  elif name ==  'rmsprop': 
    optim = torch.optim.RMSprop(model.parameters(), args.learn_rate, weight_decay=args.weight_decay, eps=args.eps)
  elif name ==  'rprop': 
    optim = torch.optim.Rprop(model.parameters(), args.learn_rate)
  elif name ==  'sgd': 
    optim = torch.optim.SGD(model.parameters(), args.learn_rate, weight_decay=args.weight_decay,
                            momentum=args.momentum)
  elif name ==  'sparseadam': 
    optim = torch.optim.SparseAdam(model.parameters(), args.learn_rate, eps=args.eps)

  return optim

def train(model, args):
  ''' Train the given model using the given hyperparameters. Save its best model as well as its
      final model.  Return training stats. '''
  paths = prep_paths(args, model.run_label, device=-1)
  train_data, test_data = prep_data(args)
  lowest_loss = 100.0
  optimizer = get_optimizer(model, args)
  save_model = {'args': args, 'model': model.state_dict()}
  (epochs, early_stop) = (args.epochs, 0) if args.early_stop is None else (sys.maxsize, args.early_stop)
  for e in range(epochs):
    stats = model.etrain(args, train_data, optimizer)
    stats['testing'] = [model.validate(args, test_data, device=-1)]
    train_loss = stats['training'][-1]
    test_loss = stats['testing'][-1]
    grad_norm = stats['grad_norms'][-1]
    grad_std = stats['grad_stds'][-1]
    save_model['model'] = model.state_dict()
    #to_pickle(save_model, paths['save_model'])
    torch.save(save_model, paths['save_model'])
    if test_loss < lowest_loss:
      lowest_loss = test_loss
      #to_pickle(save_model, paths['save_lowest'])
      torch.save(save_model, paths['save_lowest'])
    if test_loss <= early_stop:
      logmsg(f"early stop at loss {early_stop}")
      break
    logmsg(f"epoch {e} loss->train:{train_loss:.4e} test:{test_loss:.4e} " + \
           f"grads->norm:{grad_norm:.4e} std:{grad_std:.4e}")

  return stats

