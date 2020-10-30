
import gc
import math 
import torch
import numpy as np

#nail.hnn.modules:
from nail.hnn.blnn import BLNN
from nail.hnn.sho.hnn import HNN
from nail.hnn.sho.ndshodata import SimpleHarmonicOscillatorDS
from nail.hnn.utils import *
import nail.hnn.gpu as gpu

def model_update(t, state, model):
    state = state.reshape(-1, model.d_in)
    deriv = np.zeros_like(state)
    np_x = state 
    x = torch.tensor( np_x, requires_grad=True, dtype=torch.float32)
    dx_hat = model.time_derivative(x)
    deriv= dx_hat.detach().data.numpy()
    
    return deriv.reshape(-1)

def gen_dynsys(args):
    shosys = SimpleHarmonicOscillatorDS(args)
    shosys.tspan = args.tspan
    shosys.time_points = abs(int((1.0 / args.dsr) * (shosys.tspan[1]-shosys.tspan[0])))
    shosys.integrator = "RK45"

    return shosys

def load_hnn_model(args, path):
    saved_model = from_pickle(path)
    args = saved_model['args']
    dim = 2 * args.num_bodies
    model = HNN(d_in=dim, d_hidden=args.hidden_dim, d_out=1, activation_fn=args.activation_fn)
    model.load_state_dict(saved_model['model'])
    model.eval()

    return (args, model)
    
def load_base_model(args, path):
    saved_model = from_pickle(path)
    args = saved_model['args']
    dim = 2 * args.num_bodies
    model = BLNN(d_in=dim, d_hidden=args.hidden_dim, d_out=dim, activation_fn=args.activation_fn)
    model.load_state_dict(saved_model['model'])
    model.eval()

    return (args, model)

def load_models(args, label):
    base_args, base_model, hnn_args, hnn_model = (None, None, None, None)
    base_path = f"{args.save_dir}/baseline_model_{label}-avg.trch"
    hnn_path = f"{args.save_dir}/hnn_model_{label}-avg.trch"
    base_args, base_model = load_base_model(args, base_path)
    hnn_args, hnn_model = load_hnn_model(args, hnn_path)
    models = (base_args, base_model, hnn_args, hnn_model)
    
    return models

def avg_models(args, label):
  models = []
  avg_model_name = f"{args.save_dir}/{args.model}_model_{label}-avg.trch"
  dim = 2 * args.num_bodies
  n = 0
  for i in range(-1,4):
      try:
          model_name = f"{args.save_dir}/lowest_{args.model}_model_{label}_gpu{i}.trch"
          saved_model = from_pickle(model_name)
          if saved_model['args'].model == 'baseline':
            models.append(BLNN(dim, saved_model['args'].hidden_dim, dim, args.activation_fn))
          else:
            models.append(HNN(dim, saved_model['args'].hidden_dim, 1, args.activation_fn))
          models[n].load_state_dict(saved_model['model'])
          n += 1
      except FileNotFoundError:
          #logmsg(f"avg_models.  FileNotFoundError for model_name {model_name}")
          pass
  avg_model = gpu.average_models(models)
  saved_model['model'] = avg_model.state_dict()
  to_pickle(saved_model, avg_model_name)

def gen_orbits(shosys, base_model, hnn_model, state0):
    logmsg(f"gen_orbits. state0: {state0}")
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
    
    logmsg(f"gen_orbits.  ground: {ground_orbit.shape}, base: {base_orbit.shape}, hnn: {hnn_orbit.shape}")
    orbits = (ground_orbit, ground_settings, base_orbit, base_settings, hnn_orbit, hnn_settings)
    return orbits

if __name__ == "__main__":
  args = get_args()
  printargs(args)
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  shosys = gen_dynsys(args)
  state0 = shosys.random_config()
  logmsg(f'IC vector: {state0}')
  logmsg(f'IC energy: {shosys.get_energy(state0)}')
  npoints = 0
  in_dim = 2 * args.num_bodies 
  for power in range(7,16):
    gc.collect()
    npoints = 2**power
    save_label = get_label(args)
    args.train_pts = npoints
    bout_dim = in_dim
    hout_dim = 1

    args.model = 'baseline'
    bmodel = BLNN(in_dim, args.hidden_dim, bout_dim, args.activation_fn)
    bmodel.set_label(save_label)
    gpu.spawn_replicas(gpu.run_model, bmodel, args.num_gpus, args)
    avg_models(args, save_label)

    args.model = 'hnn'
    hmodel = HNN(in_dim, args.hidden_dim, hout_dim, args.activation_fn)
    hmodel.set_label(save_label)
    gpu.spawn_replicas(gpu.run_model, hmodel, args.num_gpus, args)
    avg_models(args, save_label)

    models = load_models(args, save_label)
    base_args, base_model, hnn_args, hnn_model = models
    orbits = gen_orbits(shosys, base_model, hnn_model, state0)
    ground_orbit, ground_settings, base_orbit, base_settings, hnn_orbit, hnn_settings = orbits
    genergy = shosys.get_energy(ground_orbit)
    benergy = shosys.get_energy(base_orbit)
    henergy = shosys.get_energy(hnn_orbit)
    bdEfinal = abs(genergy[-1] - benergy[-1])
    bdEavg = np.absolute(genergy - benergy).mean()
    hdEfinal = abs(genergy[-1] - henergy[-1])
    hdEavg = np.absolute(genergy - henergy).mean()
    E = genergy[-1]
    f = open(f"ndsho_{args.num_bodies}D_dE_seed-{args.seed}.tsv", "a")
    f.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(npoints, E, bdEfinal, bdEavg, hdEfinal, hdEavg))
    f.close()

  logmsg("done!")

