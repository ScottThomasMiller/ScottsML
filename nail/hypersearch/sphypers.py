
import gc
import math 
import torch
import numpy as np

#nail.hnn.modules:
from nail.hnn.blnn import BLNN
from sphnn import SimplePendulumHNN
from spdata import SimplePendulumDS
from nail.hnn.utils import *
import nail.hnn.run as run

def gen_dynsys(args):
    ''' Create and return a DynamicalSystem object, for generating orbits and ICs. '''
    spsys = SimplePendulumDS(args)
    spsys.tspan = args.tspan
    spsys.time_points = abs(int((1.0 / args.dsr) * (spsys.tspan[1]-spsys.tspan[0])))
    spsys.integrator = "RK45"

    return spsys

def load_hnn_model(args, path):
    ''' Load the trained PyTorch model for HNN. '''
    saved_model = from_pickle(path)
    args = saved_model['args']
    #dim = 2 * args.num_bodies
    dim = 3
    model = SimplePendulumHNN(d_in=dim, d_hidden=args.hidden_dim, d_out=1, activation_fn=args.activation_fn)
    model.load_state_dict(saved_model['model'])
    model.eval()

    return (args, model)
    
def load_models(args, label):
    ''' Load both NN and HNN trained models. Use the lowest-loss saved copies, for 
        max. accuracy. '''
    base_args, base_model, hnn_args, hnn_model = (None, None, None, None)
    base_path = f"{args.save_dir}/lowest_baseline_model_{label}.trch"
    hnn_path = f"{args.save_dir}/lowest_hnn_model_{label}.trch"
    base_args, base_model = load_base_model(args, base_path)
    hnn_args, hnn_model = load_hnn_model(args, hnn_path)
    models = (base_args, base_model, hnn_args, hnn_model)
    
    return models

def train(model, args, train_data, test_data):
  ''' Train the given model using the given hyperparameters. Save its best model as well as its
      final model.  Return training stats. '''
  paths = run.prep_paths(args, model.run_label, device=-1)
  lowest_loss = 100.0
  optimizer = run.get_optimizer(model, args)
  save_model = {'args': args, 'model': model.state_dict()}
  if args.gpu >= 0:
    model.to(args.gpu)
    model.init_device()
  for e in range(args.epochs):
    stats = model.etrain(args, train_data, optimizer)
    stats['testing'] = [model.validate(args, test_data, device=-1)]
    train_loss = stats['training'][-1]
    test_loss = stats['testing'][-1]
    grad_norm = stats['grad_norms'][-1]
    grad_std = stats['grad_stds'][-1]
    save_model['model'] = model.state_dict()
    #to_pickle(save_model, paths['save_model'])
    #torch.save(save_model, paths['save_model'])
    if test_loss < lowest_loss:
      lowest_loss = test_loss
      #to_pickle(save_model, paths['save_lowest'])
      torch.save(save_model, paths['save_lowest'])
    logmsg(f"epoch {e} loss->train:{train_loss:.4e} test:{test_loss:.4e} " + \
           f"grads->norm:{grad_norm:.4e} std:{grad_std:.4e}")

  return stats
   
def write_orbit(ofile, orbit):
    for i in range(orbit.shape[1]):
      rec = ""
      for j in range(orbit.shape[0]):
          rec += f"{orbit[j,i]:.8f}\t"
      ofile.write(f"{rec}\n")

def write_orbits(args, ground_orbit, base_orbit, hnn_orbit):
    with open(f"{args.save_dir}/ndsqo_{args.num_bodies}D_orbits_seed-{args.seed}.tsv", "a") as ofile:
        ofile.write("ground\n")
        write_orbit(ofile, ground_orbit)
        ofile.write("NN\n")
        write_orbit(ofile, base_orbit)
        ofile.write("HNN\n")
        write_orbit(ofile, hnn_orbit)

def forecast(models, spsys, args, npoints):
    ''' Generate multiple orbits, each with its own IC, for both NN and HNN.  
        Concatenate the NN orbits together, and the HNN orbits together, and
        then calculate and write the stats of the concatenated orbits. ''' 
    base_args, base_model, hnn_args, hnn_model = models
    state0 = spsys.random_config()
    orbits = gen_orbits(spsys, base_model, hnn_model, state0)
    gorbits, gsettings, borbits, bsettings, horbits, hsettings = orbits
    for i in range(args.num_forecasts):
        state0 = spsys.random_config()
        orbits = gen_orbits(spsys, base_model, hnn_model, state0)
        ground_orbit, ground_settings, base_orbit, base_settings, hnn_orbit, hnn_settings = orbits
        if npoints == 2**15:
            write_orbits(args, ground_orbit, base_orbit, hnn_orbit)
        gorbits = np.concatenate((gorbits, ground_orbit), axis=1)
        borbits = np.concatenate((borbits, base_orbit), axis=1)
        horbits = np.concatenate((horbits, hnn_orbit), axis=1)
    genergy = spsys.get_energy(gorbits)
    #benergy = spsys.get_energy(borbits)
    #henergy = spsys.get_energy(horbits)
    x = borbits[0,:]
    p = borbits[2,:]
    benergy = p**2/2.0 + (1 - x)
    x = horbits[0,:]
    p = horbits[2,:]
    henergy = p**2/2.0 + (1 - x)
    bdEfinal = abs(genergy[-1] - benergy[-1])
    bdEavg = np.absolute(genergy - benergy).mean()
    hdEfinal = abs(genergy[-1] - henergy[-1])
    hdEavg = np.absolute(genergy - henergy).mean()
    E = genergy[-1]
    f = open(f"sp_{args.num_bodies}D_dE_seed-{args.seed}.tsv", "a")
    f.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(npoints, E, bdEfinal, bdEavg, hdEfinal, hdEavg))
    f.close()

if __name__ == "__main__":
  power = 13
  args = get_args()
  printargs(args)
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  spsys = gen_dynsys(args)
  npoints = 0
  in_dim = 3
  npoints = 2**power
  args.train_pts = npoints
  train_data, test_data = run.prep_data(args)
  save_label = get_label(args)
  args.train_pts = npoints
  bout_dim = in_dim
  hout_dim = 1

  args.model = 'hnn'
  hmodel = SimplePendulumHNN(in_dim, args.hidden_dim, hout_dim, args.activation_fn)
  hmodel.set_label(save_label)
  logmsg("train HNN")
  stats = train(hmodel, args, train_data, test_data)
  test_loss = stats['testing'][-1]
  print(f'CID: {args.cid}') 
  print(f'GID: {args.gid}') 
  print(f'LOSS: {test_loss}') 
  stream = os.popen('hostname')
  hostname = stream.read().rstrip('\n')
  print(f'HOSTNAME: {hostname}')
  print('OK')

