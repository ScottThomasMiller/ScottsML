
import math 
import torch
import numpy as np
import pandas as pd

#NAIL modules:
#nail.hnn.modules:
from nail.hnn.blnn import BLNN
from dphnn import DoublePendulumHNN
from dpdata import DoublePendulumDS
from nail.hnn.utils import *
import nail.hnn.run as run
from dpforecast import DPforecast

def train(model, args, train_data, test_data):
  ''' Train the given model using the given hyperparameters. Save its best model as well as its
      final model.  Return training stats. '''
  paths = run.prep_paths(args, model.run_label, device=-1)
  lowest_loss = 100.0
  optimizer = run.get_optimizer(model, args)
  save_model = {'args': args, 'model': model.state_dict()}
  for e in range(args.epochs):
    stats = model.etrain(args, train_data, optimizer)
    stats['testing'] = [model.validate(args, test_data, device=-1)]
    train_loss = stats['training'][-1]
    test_loss = stats['testing'][-1]
    grad_norm = stats['grad_norms'][-1]
    grad_std = stats['grad_stds'][-1]
    save_model['model'] = model.state_dict()
    to_pickle(save_model, paths['save_model'])
    #torch.save(save_model, paths['save_model'])
    if test_loss < lowest_loss:
      lowest_loss = test_loss
      to_pickle(save_model, paths['save_lowest'])
      #torch.save(save_model, paths['save_lowest'])
    logmsg(f"epoch {e} loss->train:{train_loss:.4e} test:{test_loss:.4e} " + \
           f"grads->norm:{grad_norm:.4e} std:{grad_std:.4e}")
  logmsg(f"lowest model: {paths['save_lowest']}")

  return stats

def xy2q(state_xy):
    x1 = state_xy[0]
    y1 = state_xy[1]
    x2 = state_xy[2]
    y2 = state_xy[3]
    p1 = state_xy[4]
    p2 = state_xy[5]
    q1 = np.arctan2(y1, x1)
    q2 = np.arctan2(y2, x2)

    return np.row_stack((q1, q2, p1, p2))

if __name__ == "__main__":
  in_dim = 6
  args = get_args()
  args.state_symbols = ['q1','q2','p1','p2']
  args.name = "dp-dataset-dsr1e-04-tspan0_10-traj100-xy-p1pi"
  args.hamiltonian="-(p1**2 + 2*p2**2 - 2*p1*p2*cos(q1 - q2) + (-3 + cos(2*(q1 - q2)))*(2*cos(q1) + cos(q2))) / (-3 + cos(2*(q1 - q2)))"
  args.test_pct = 0.5
  args.train_pct = 0 
  args.master_port = 11571
  args.activation_fn = 'Tanh'
  args.learn_rate = 1e-03 
  args.tspan = [0, 10]
  args.dsr = 0.0001
  args.hidden_dim = [32, 32] 
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  printargs(args)
  npoints = 2**args.npower
  args.train_pts = npoints
  save_label = get_label(args)
  train_data, test_data = run.prep_data(args)
  bout_dim = in_dim
  hout_dim = 1

  args.model = 'baseline'
  bmodel = BLNN(in_dim, args.hidden_dim, bout_dim, args.activation_fn, beta=args.beta)
  bmodel.set_label(save_label)
  logmsg('training baseline')
  train(bmodel, args, train_data, test_data)
  bforecast = DPforecast(args=args, model=bmodel)

  args.model = 'hnn'
  hmodel = DoublePendulumHNN(in_dim, args.hidden_dim, hout_dim, args.activation_fn, beta=args.beta)
  hmodel.set_label(save_label)
  logmsg('training HNN')
  train(hmodel, args, train_data, test_data)
  hforecast = DPforecast(args=args, model=hmodel)

  logmsg('forecasting orbits')
  for i in range(args.num_forecasts):
      state0 = bforecast.dpsys.random_config()
      truth = bforecast.true_orbit(state0)
      baseline = bforecast.model_orbit(state0, numpts=truth.shape[1])
      hnn = hforecast.model_orbit(state0, numpts=truth.shape[1])
      logmsg(f'shapes truth: {truth.shape} baseline: {baseline.shape} hnn: {hnn.shape}')
      fname = f'{args.save_dir}/orbit_truth_{i}.csv'
      bforecast.write_orbit(truth.T, fname)
      fname = f'{args.save_dir}/orbit_baseline_{i}.csv'
      bforecast.write_orbit(xy2q(baseline.T).T, fname)
      fname = f'{args.save_dir}/orbit_hnn_{i}.csv'
      hforecast.write_orbit(xy2q(hnn.T).T, fname)
  logmsg('done.')
