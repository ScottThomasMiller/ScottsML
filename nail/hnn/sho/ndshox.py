
''' 
    Module: ndshox.py
    Author: Scott Miller, Non-Linear A.I. Lab (NAIL), NCSU
    
    The ndshox.py module runs the delta energy (dE) experiments, comparing baseline, Multi-Layer Perceptron
    (MLP) Neural Network (NN) performance to Hamiltonian NN (HNN) for the N-Dimensional Simple Harmonic
    Oscillator (NDSHO).  It trains the networks, forecasts orbits via inference mode, calculates 
    the energies of those orbits, and then computes values for the average relative mean errors of those forecasts. 
 
    The experiments look at relative performance between baseline NN and HNN across a narrow range of input
    training sizes, from N=2^7 instances to N=2^15, in steps of powers of two.

    The script produces a tab-separated .tsv file with 9 lines, one line for each power of two in the test range.  
    The format of each line is:

    N	IC-Energy    NN-dE-Final	NN-dE-Avg	HNN-dE-Final	HNN-dE-Avg
        
'''
import gc
import math 
import torch
import numpy as np

#nail.hnn.modules:
from nail.hnn.blnn import BLNN
#from nail.hnn.sho.hnn import HNN
from nail.hnn.hnn import HNN
from nail.hnn.sho.ndshodata import SimpleHarmonicOscillatorDS
from nail.hnn.utils import *
import nail.hnn.run as run

def model_update(t, state, model):
    ''' This function is called by a DynamicalSystem object, for calculating differentials,
        to assist with generating an orbit. '''
    state = state.reshape(-1, model.d_in)
    deriv = np.zeros_like(state)
    np_x = state 
    x = torch.tensor( np_x, requires_grad=True, dtype=torch.float32)
    dx_hat = model.time_derivative(x)
    deriv= dx_hat.detach().data.numpy()
    
    return deriv.reshape(-1)

def gen_dynsys(args):
    ''' Create and return a DynamicalSystem object, for generating orbits and ICs. '''
    shosys = SimpleHarmonicOscillatorDS(args)
    shosys.tspan = args.tspan
    shosys.time_points = abs(int((1.0 / args.dsr) * (shosys.tspan[1]-shosys.tspan[0])))
    shosys.integrator = "RK45"

    return shosys

def load_hnn_model(args, path):
    ''' Load the trained PyTorch model for HNN. '''
    saved_model = from_pickle(path)
    args = saved_model['args']
    dim = 2 * args.num_bodies
    model = HNN(d_in=dim, d_hidden=args.hidden_dim, d_out=1, activation_fn=args.activation_fn)
    model.load_state_dict(saved_model['model'])
    model.eval()

    return (args, model)
    
def load_base_model(args, path):
    ''' Load the trained PyTorch model for NN. '''
    saved_model = from_pickle(path)
    args = saved_model['args']
    dim = 2 * args.num_bodies
    model = BLNN(d_in=dim, d_hidden=args.hidden_dim, d_out=dim, activation_fn=args.activation_fn)
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

def gen_orbits(shosys, base_model, hnn_model, state0):
    ''' Generate orbits based on the given trained models, and the IC state0. '''
    base_orbit, base_settings = (None, None)
    if base_model is not None:
        base_model.eval()
    hnn_model.eval()
    
    shosys.external_update_fn = None
    state = state0
    ground_orbit, ground_settings = shosys.get_orbit(state)

    update_fn = lambda t, y0: model_update(t, y0, hnn_model)
    shosys.external_update_fn = update_fn
    state = state0
    hnn_orbit, hnn_settings = shosys.get_orbit(state)

    update_fn = lambda t, y0: model_update(t, y0, base_model)
    shosys.external_update_fn = update_fn
    state = state0
    base_orbit, base_settings = shosys.get_orbit(state)
    
    orbits = (ground_orbit, ground_settings, base_orbit, base_settings, hnn_orbit, hnn_settings)
    return orbits

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
    to_pickle(save_model, paths['save_model'])
    #torch.save(save_model, paths['save_model'])
    if test_loss < lowest_loss:
      lowest_loss = test_loss
      to_pickle(save_model, paths['save_lowest'])
      #torch.save(save_model, paths['save_lowest'])
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

def forecast(models, shosys, args, npoints):
    ''' Generate multiple orbits, each with its own IC, for both NN and HNN.  
        Concatenate the NN orbits together, and the HNN orbits together, and
        then calculate and write the stats of the concatenated orbits. ''' 
    base_args, base_model, hnn_args, hnn_model = models
    state0 = shosys.random_config()
    orbits = gen_orbits(shosys, base_model, hnn_model, state0)
    gorbits, gsettings, borbits, bsettings, horbits, hsettings = orbits
    for i in range(args.num_forecasts):
        state0 = shosys.random_config()
        orbits = gen_orbits(shosys, base_model, hnn_model, state0)
        ground_orbit, ground_settings, base_orbit, base_settings, hnn_orbit, hnn_settings = orbits
        if npoints == 2**15:
            write_orbits(args, ground_orbit, base_orbit, hnn_orbit)
        gorbits = np.concatenate((gorbits, ground_orbit), axis=1)
        borbits = np.concatenate((borbits, base_orbit), axis=1)
        horbits = np.concatenate((horbits, hnn_orbit), axis=1)
    genergy = shosys.get_energy(gorbits)
    benergy = shosys.get_energy(borbits)
    henergy = shosys.get_energy(horbits)
    bdEfinal = abs(genergy[-1] - benergy[-1])
    bdEavg = np.absolute(genergy - benergy).mean()
    hdEfinal = abs(genergy[-1] - henergy[-1])
    hdEavg = np.absolute(genergy - henergy).mean()
    E = genergy[-1]
    f = open(f"ndsho_{args.num_bodies}D_dE_seed-{args.seed}.tsv", "a")
    f.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(npoints, E, bdEfinal, bdEavg, hdEfinal, hdEavg))
    f.close()

if __name__ == "__main__":
  args = get_args()
  printargs(args)
  #logmsg(f"env: {os.environ['OMP_NUM_THREADS']}")
  #nthreads = int(os.environ['OMP_NUM_THREADS']) if os.environ['OMP_NUM_THREADS'] is not None else 1
  #logmsg(f"nthreads: {nthreads}")
  #torch.set_num_threads(nthreads)
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  shosys = gen_dynsys(args)
  npoints = 0
  in_dim = 2 * args.num_bodies 
  for power in range(7,16):
    gc.collect()
    npoints = 2**power
    args.train_pts = npoints
    train_data, test_data = run.prep_data(args)
    save_label = get_label(args)
    args.train_pts = npoints
    bout_dim = in_dim
    hout_dim = 1

    args.model = 'baseline'
    bmodel = BLNN(in_dim, args.hidden_dim, bout_dim, args.activation_fn)
    bmodel.set_label(save_label)
    train(bmodel, args, train_data, test_data)

    args.model = 'hnn'
    hmodel = HNN(in_dim, args.hidden_dim, hout_dim, args.activation_fn)
    hmodel.set_label(save_label)
    train(hmodel, args, train_data, test_data)
    models = load_models(args, save_label)

    forecast(models, shosys, args, npoints)
  logmsg("done!")

