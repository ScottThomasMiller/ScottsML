
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

def model_update(t, state, model):
    state = state.reshape(-1,6)
    deriv = np.zeros_like(state)
    np_x = state 
    x = torch.tensor( np_x, requires_grad=True, dtype=torch.float32)
    dx_hat = model.time_derivative(x)
    deriv = dx_hat.detach().data.numpy() * 100.0
    
    return deriv.reshape(-1)

def gen_dynsys(args):
    ''' Create and return a DoublePendulumDS object, for generating orbits and ICs. '''
    dpsys = DoublePendulumDS(args)
    dpsys.tspan = args.tspan
    dpsys.time_points = abs(int((1.0 / args.dsr) * (dpsys.tspan[1]-dpsys.tspan[0])))
    dpsys.integrator = "RK45"

    return dpsys

def load_hnn_model(path):
    saved_model = from_pickle(path)
    args = saved_model['args']
    model = DoublePendulumHNN(d_in=6, d_hidden=args.hidden_dim, d_out=1, activation_fn=args.activation_fn, beta=args.beta)
    model.load_state_dict(saved_model['model'])
    model.eval()

    return (args, model)
    
def load_base_model(path):
    saved_model = from_pickle(path)
    args = saved_model['args']
    model = BLNN(d_in=6, d_hidden=args.hidden_dim, d_out=6, activation_fn=args.activation_fn, beta=args.beta)
    model.load_state_dict(saved_model['model'])
    model.eval()

    return (args, model)

def load_models(args, label):
    base_args, base_model, hnn_args, hnn_model = (None, None, None, None)
    base_path = f"{args.save_dir}/baseline_model_{label}.trch"
    hnn_path = f"{args.save_dir}/hnn_model_{label}.trch"
    base_args, base_model = load_base_model(base_path)
    hnn_args, hnn_model = load_hnn_model(hnn_path)
    models = (base_args, base_model, hnn_args, hnn_model)
    
    return models

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

def gen_orbits(dpsys, base_model, hnn_model, state0):
    save_state_xyp = np.zeros(6)
    save_state_xyp[0] = np.cos(state0[0])
    save_state_xyp[1] = np.sin(state0[0])
    save_state_xyp[2] = np.cos(state0[1])
    save_state_xyp[3] = np.sin(state0[1])
    save_state_xyp[4] = state0[2]
    save_state_xyp[5] = state0[3]
    save_state_qp = xy2q(save_state_xyp)
    base_orbit, base_settings = (None, None)
    if base_model is not None:
        base_model.eval()
    hnn_model.eval()
    
    dpsys.external_update_fn = None
    state = save_state_qp
    ground_orbit, ground_settings = dpsys.get_orbit(state)

    update_fn = lambda t, y0: model_update(t, y0, hnn_model)
    dpsys.external_update_fn = update_fn
    state = save_state_xyp
    hnn_orbit, hnn_settings = dpsys.get_orbit(state)

    update_fn = lambda t, y0: model_update(t, y0, base_model)
    dpsys.external_update_fn = update_fn
    state = save_state_xyp
    base_orbit, base_settings = dpsys.get_orbit(state)
    
    orbits = (ground_orbit, ground_settings, base_orbit, base_settings, hnn_orbit, hnn_settings)
    return orbits

def orbit_qp(orbit_xyp):
    ''' Convert an x/y/p orbit into a q/p orbit. '''
    x1 = orbit_xyp[0,:]
    y1 = orbit_xyp[1,:]
    x2 = orbit_xyp[2,:]
    y2 = orbit_xyp[3,:]
    p1 = orbit_xyp[4,:]
    p2 = orbit_xyp[5,:]
    q1 = np.arctan2(y1, x1)
    q2 = np.arctan2(y2, x2)
    orbit_qp = np.row_stack((q1, q2, p1, p2))
    
    return orbit_qp

def write_qpHfile(orbit_qp, energy):
  ''' Write the forecasted canonical coordinates and their respective energies to a txt file.
       {q1, q2, p1, p2, H}  '''
  with open('dp-HNN-q1q2p1p2H.tsv','w') as ofile:
    for i in range(len(orbit_qp)):
      q1 = orbit_qp[i][0]
      q2 = orbit_qp[i][1]
      p1 = orbit_qp[i][2]
      p2 = orbit_qp[i][3]
      H  = energy[i]
      ofile.write(f'{q1}\t{q2}\t{p1}\t{p2}\t{H}\n')

def write_orbits(args, ground_orbit, base_orbit, hnn_orbit, npoints):
    ''' Write the ground-truth and forecasted orbits to separate .csv files. '''
    colnames = ['q1','q2','p1','p2']
    df = pd.DataFrame(ground_orbit.T, columns=colnames)
    df.to_csv(f'{args.save_dir}/orbits_xyp_ground_npts{npoints}.csv', index=False)
    df = pd.DataFrame(xy2q(base_orbit).T, columns=colnames)
    df.to_csv(f'{args.save_dir}/orbits_xyp_base_npts{npoints}.csv', index=False)
    df = pd.DataFrame(xy2q(hnn_orbit).T, columns=colnames)
    df.to_csv(f'{args.save_dir}/orbits_xyp_hnn_npts{npoints}.csv', index=False)

def forecast(models, dpsys, args, npoints):
    ''' Generate multiple orbits, each with its own IC, for both NN and HNN.  
        Concatenate the NN orbits together, and the HNN orbits together, and
        then calculate and write the stats of the concatenated orbits. ''' 
    base_args, base_model, hnn_args, hnn_model = models
    state0 = dpsys.random_config()
    orbits = gen_orbits(dpsys, base_model, hnn_model, state0)
    gorbits, gsettings, borbits, bsettings, horbits, hsettings = orbits
    for i in range(args.num_forecasts):
        state0 = dpsys.random_config()
        orbits = gen_orbits(dpsys, base_model, hnn_model, state0)
        ground_orbit, ground_settings, base_orbit, base_settings, hnn_orbit, hnn_settings = orbits
        write_orbits(args, ground_orbit, base_orbit, hnn_orbit, npoints)
        gorbits = np.concatenate((gorbits, ground_orbit), axis=1)
        borbits = np.concatenate((borbits, base_orbit), axis=1)
        horbits = np.concatenate((horbits, hnn_orbit), axis=1)
    logmsg(f'shape gorbits: {gorbits.shape} borbits: {borbits.shape} horbits: {horbits.shape}')
    genergy = dpsys.get_energy(gorbits)
    benergy = dpsys.get_energy(xy2q(borbits))
    henergy = dpsys.get_energy(xy2q(horbits))
    x = borbits[0,:]
    p = borbits[2,:]
    x = horbits[0,:]
    p = horbits[2,:]
    bdEfinal = abs(genergy[-1] - benergy[-1])
    bdEavg = np.absolute(genergy - benergy).mean()
    hdEfinal = abs(genergy[-1] - henergy[-1])
    hdEavg = np.absolute(genergy - henergy).mean()
    E = genergy[-1]
    f = open(f"sp_{args.num_bodies}D_dE_seed-{args.seed}.tsv", "a")
    f.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(npoints, E, bdEfinal, bdEavg, hdEfinal, hdEavg))
    f.close()
    horbits_qp = orbit_qp(horbits)
    write_qpHfile(horbits_qp, henergy)

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

if __name__ == "__main__":
  in_dim = 6
  args = get_args()
  #args.epochs = 16
  args.epochs = 32
  args.state_symbols = ['q1','q2','p1','p2']
  args.name = "dp-dataset-dsr1e-02-tspan0_100-traj125-xy-p1pi"
  args.hamiltonian="-(p1**2 + 2*p2**2 - 2*p1*p2*cos(q1 - q2) + (-3 + cos(2*(q1 - q2)))*(2*cos(q1) + cos(q2))) / (-3 + cos(2*(q1 - q2)))"
  args.test_pct = 1
  args.master_port = 11571
  args.activation_fn = 'Tanh'
  args.learn_rate = 1e-03 
  args.tspan = [0, 100]
  args.dsr = 0.01
  args.batch_size = 1 
  args.hidden_dim = [32, 32] 
  args.train_pct = 0 
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  dpsys = gen_dynsys(args)
  logmsg('args:\n{}'.format(args))

  #for power in range(7,16):
  for power in range(14,15):
    save_label = get_label(args)
    npoints = 2**power
    args.train_pts = npoints
    train_data, test_data = run.prep_data(args)
    bout_dim = in_dim
    hout_dim = 1

    args.model = 'baseline'
    bmodel = BLNN(in_dim, args.hidden_dim, bout_dim, args.activation_fn, beta=args.beta)
    bmodel.set_label(save_label)
    logmsg('training baseline')
    train(bmodel, args, train_data, test_data)

    args.model = 'hnn'
    hmodel = DoublePendulumHNN(in_dim, args.hidden_dim, hout_dim, args.activation_fn, beta=args.beta)
    hmodel.set_label(save_label)
    logmsg('training HNN')
    train(hmodel, args, train_data, test_data)

    logmsg('forecasting orbits');
    models = load_models(args, save_label)
    forecast(models, dpsys, args, npoints)
