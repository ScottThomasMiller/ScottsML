
from blnn import BLNN
from hnn import HNN
from dpdata import DoublePendulumDS, DataStream
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
from torch.nn.parallel import DataParallel 
from torch.utils.data import DataLoader
from utils import logmsg, to_pickle, from_pickle

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
state_symbols = ['q', 'p']

# epoch train:
def etrain(model, args, train_data, optimizer):
    model.train()
    loss_fn = torch.nn.MSELoss().to("cuda")
    stats = copy.deepcopy(BLNN.stats_dict)
    addnoise = False
    if args.input_noise != '':
      addnoise = True
      npnoise = np.array(args.input_noise, dtype="float")
      noise = torch.tensor(npnoise).to("cuda")

    batches = 0
    for x, dxdt in train_data:
        optimizer.zero_grad()
        dxdt_hat = model.module.time_derivative(x)
        if addnoise:
          dxdt_hat += noise * torch.randn(*x.shape).to("cuda")  # add noise, maybe
        loss = loss_fn(dxdt_hat, dxdt)
        loss.backward()
        optimizer.step()

        grad = torch.cat([p.grad.flatten() for p in model.parameters()])
        stats['training'].append(loss.item())
        stats['grad_norms'].append((grad @ grad).item())
        stats['grad_stds'].append(grad.std().item())

        if args.verbose and (batches % args.print_every == 0):
          grad_norm = grad @ grad
          logmsg("batch[{}] train loss {:.4e}, grad norm: {:.4e}, grad std: {:.4e}"
                .format(batches, loss.item(), grad_norm, grad.std()))
        batches += 1

    return stats

def validate(model, args, data):
    model.eval()
    loss_fn = torch.nn.MSELoss().to("cuda")
    if args.input_noise != '': 
      npnoise = np.array(args.input_noise, dtype="float")
      noise = torch.tensor(npnoise).to("cuda")

    # the validation data stream is configured to return the entire set in a 
    # single, randomized batch:
    if model.module.model == 'baseline':
        with torch.no_grad():
            for x, dxdt in data:
                dxdt_hat = model.module.time_derivative(x)
                if args.input_noise != '': 
                  dxdt_hat += noise * torch.randn(*x.shape).to("cuda")  
    else:
        # the HNN needs the gradients for computing the dH values:
        for x, dxdt in data:
            dxdt_hat = model.module.time_derivative(x)
            if args.input_noise != '':
                dxdt_hat += noise * torch.randn(*x.shape).to("cuda")  

    return loss_fn(dxdt, dxdt_hat).item()

def load_model(model, model_path):
    if (model_path is None) or (model_path == ''):
        return False
    try:
        saved_model = from_pickle(model_path)
        model.load_state_dict(saved_model['model'])
        logmsg("HELLO! loaded model: {}".format(model_path))
        return True
    except Exception as ex:
        logmsg("ERROR: exception occurred trying to load model {} "+str(model_path))
        logmsg("EXCEPTION: "+str(ex))
        return False

def append_stats(save_stats, stats):
    train_loss = stats['training'][-1]
    test_loss = stats['testing'][-1]
    grad_norm = stats['grad_norms'][-1]
    grad_std = stats['grad_stds'][-1]
    save_stats['training'].append(train_loss)
    save_stats['testing'].append(test_loss)
    save_stats['grad_norms'].append(grad_norm)
    save_stats['grad_stds'].append(grad_std)
    estats = torch.Tensor([train_loss, test_loss, grad_norm, grad_std])

    return estats

def prep_data(args):
    input_data = load_data(args)
    input_data = {
        'coords': torch.tensor(input_data['coords'], dtype=torch.float32, requires_grad=True).to("cuda"),
        'dcoords': torch.tensor(input_data['dcoords'], dtype=torch.float32, requires_grad=True).to("cuda"),
        'test_coords': torch.tensor(input_data['test_coords'], dtype=torch.float32, requires_grad=True).to("cuda"),
        'test_dcoords': torch.tensor(input_data['test_dcoords'], dtype=torch.float32, requires_grad=True).to("cuda")}
    train_data, test_data = to_dataset(args, input_data)
    logmsg("total data->training: {:,}, testing: {:,}".format(input_data['coords'].shape[0], input_data['test_coords'].shape[0]))
  
    return train_data, test_data

def prep_paths(args, run_label):
    paths = {}
    save_model_base = '{}/{}_model_{}.trch'
    load_path = "{}/{}".format(args.save_dir, args.load_model)
    paths['load_model'] = load_path if args.load_model != '' else ''
    paths['save_model'] = save_model_base.format(args.save_dir, args.model, run_label)
    paths['save_lowest'] = "{}/lowest_model_{}.trch".format(args.save_dir, run_label)
    paths['save_stats'] = '{}/{}_stats_{}.trch'.format(args.save_dir, args.model, run_label)

    return paths

def load_data(args):
    ''' Return the dataset from the given DyanimcalSystem, for the given or current GPU.
    '''
    tspan = []
    for t in args.tspan:
        tspan.append(t)
    tpoints = int((1.0 / args.dsr) * tspan[1])
    dynsys = DoublePendulumDS(sys_hamiltonian=args.hamiltonian, tspan=tspan, timesteps=tpoints,
                      integrator=args.integrator_scheme, state_symbols=args.state_symbols, symplectic_order=4, 
                      energy=args.energy, no_trajectories=args.trajectories, split_ratio=0.2)
    path = args.name+".pkl"
    data = dynsys.get_dataset(path, args.save_dir)
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
    training = {'x': data['coords'], 'dxdt': data['dcoords']}
    train_data = DataLoader(dataset=DataStream(training), batch_size=args.batch_size, shuffle=True, pin_memory=False)
    testing = {'x': data['test_coords'], 'dxdt': data['test_dcoords']}
    test_data = DataLoader(dataset=DataStream(testing), batch_size=test_len, shuffle=True, pin_memory=False)
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

def run_model(model, args):
    ''' Setup and run the model:  
	- convert the model to DataParallel
	- set the optimizer
        - create the save dir 
	- load optional saved model
        - load the input data
        - train the model
        - checkpoint the model and the stats
    '''
    save_model = {'args': args, 'model': model.state_dict()}
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    if torch.cuda.device_count() > 1:
      model = nn.DataParallel(model)
      model.to("cuda")
    optimizer = get_optimizer(model, args)
    save_stats = copy.deepcopy(BLNN.stats_dict)
    paths = prep_paths(args, model.module.run_label)
    loaded = load_model(model, paths['load_model'])
    train_data, test_data = prep_data(args)
    logmsg("start model {}, file label: {}".format(args.model, model.module.run_label))
    lowest_loss = 100.0

    '''
    for epoch in range(args.epochs):
      logmsg("epoch {} ".format(epoch))
      for x,dx in train_data:
        output = model(x)
        #output = model.module.time_derivative(x)
    '''
    for epoch in range(args.epochs):
        stats = etrain(model, args, train_data, optimizer)
        stats['testing'] = [validate(model, args, test_data)]
        estats = append_stats(save_stats, stats).to("cuda")
        to_pickle(save_stats, paths['save_stats'])
        save_model['model'] = model.state_dict()
        to_pickle(save_model, paths['save_model'])
        logmsg("epoch {} (avgs) loss->train:{:.4e} test:{:.4e} grads->norm:{:.4e} std:{:.4e}"
                   .format(epoch, estats[0], estats[1], estats[2], estats[3]))
        if estats[1] < lowest_loss:
            lowest_loss = estats[1]
            to_pickle(save_model, paths['save_lowest'])
        if estats[1] <= args.early_stop:
            break


