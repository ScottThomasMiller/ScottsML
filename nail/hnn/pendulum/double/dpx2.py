
import math 
import torch
import numpy as np

#NAIL modules:
from   ddp.blnn import BLNN
import ddp.gpuspawn as gpu
from   ddp.hnn import HNN
import ddp.utils as utils
import ddp.dpdata as dpdata
from ddp.dpdata import DoublePendulumDS
from ddp.utils import from_pickle, to_pickle, logmsg

def model_update(t, state, model):
    state = state.reshape(-1,6)
    deriv = np.zeros_like(state)
    np_x = state 
    x = torch.tensor( np_x, requires_grad=True, dtype=torch.float32)
    dx_hat = model.time_derivative(x)
    deriv= dx_hat.detach().data.numpy()
    
    return deriv.reshape(-1)

def gen_dynsys(args):
    tpoints = abs(int((1.0 / args.dsr) * (args.tspan[1]-args.tspan[0])))
    spsys = DoublePendulumDS(sys_hamiltonian=args.hamiltonian,state_symbols=args.state_symbols,
                          tspan=args.tspan, timesteps=tpoints, integrator=args.integrator_scheme,
                          symplectic_order=4)
    spsys.integrator = "RK45"

    return spsys

def load_hnn_model(path):
    saved_model = from_pickle(path)
    args = saved_model['args']
    model = HNN(d_in=6, d_hidden=args.hidden_dim, d_out=1, activation_fn=args.activation_fn)
    model.load_state_dict(saved_model['model'])
    model.eval()

    return (args, model)
    
def load_base_model(path):
    saved_model = from_pickle(path)
    args = saved_model['args']
    model = BLNN(d_in=6, d_hidden=args.hidden_dim, d_out=6, activation_fn=args.activation_fn)
    model.load_state_dict(saved_model['model'])
    model.eval()

    return (args, model)

def load_models(label):
    base_args, base_model, hnn_args, hnn_model = (None, None, None, None)
    base_path = "ddp/baseline_model_{}-gpu-avg.trch".format(label)
    hnn_path = "ddp/hnn_model_{}-gpu-avg.trch".format(label)
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

def gen_orbits(spsys, base_model, hnn_model):
    # chaotic orbit #0 from dp-dataset-dsr1e-02-tspan0_100-traj125-xy-piover2.pkl:
    save_state_xyp = np.asarray([ 0.54108224, -0.84096968,  0.42370142, -0.90580191,  1.54895556, -0.03376365])
    save_state_qp = xy2q(save_state_xyp)
    base_orbit, base_settings = (None, None)
    if base_model is not None:
        base_model.eval()
    hnn_model.eval()
    
    logmsg("calculating ground truth orbit")
    spsys.external_update_fn = None
    state = save_state_qp
    ground_orbit, ground_settings = spsys.get_orbit(state)
    logmsg("ground shape: {}".format(ground_orbit.shape))

    logmsg("calculating HNN orbit")
    update_fn = lambda t, y0: model_update(t, y0, hnn_model)
    spsys.external_update_fn = update_fn
    state = save_state_xyp
    hnn_orbit, hnn_settings = spsys.get_orbit(state)
    logmsg("hnn shape: {}".format(hnn_orbit.shape))

    logmsg("calculating baseline orbit")
    update_fn = lambda t, y0: model_update(t, y0, base_model)
    spsys.external_update_fn = update_fn
    state = save_state_xyp
    base_orbit, base_settings = spsys.get_orbit(state)
    logmsg("base shape: {}".format(base_orbit.shape))
    
    orbits = (ground_orbit, ground_settings, base_orbit, base_settings, hnn_orbit, hnn_settings)
    return orbits

def avg_models(args, label):
  models = []
  avg_model_name = "ddp/{}_model_{}-gpu-avg.trch".format(args.model, label)
  for i in range(4):
    model_name = "ddp/{}_model_{}-gpu{}.trch".format(args.model, label, i)
    saved_model = from_pickle(model_name)
    if saved_model['args'].model == 'baseline':
      models.append(BLNN(6, saved_model['args'].hidden_dim, 6, args.activation_fn))
    else:
      models.append(HNN(6, saved_model['args'].hidden_dim, 1, args.activation_fn))
    models[i].load_state_dict(saved_model['model'])
    avg_model = gpu.average_models(models)
  saved_model['model'] = avg_model.state_dict()
  to_pickle(saved_model, avg_model_name)

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

if __name__ == "__main__":
  in_dim = 6
  args = utils.get_args()
  args.epochs = 16
  args.state_symbols = ['q1','q2','p1','p2']
  args.name = "dp-dataset-dsr1e-02-tspan0_100-traj125-xy-p1pi"
  args.hamiltonian="-(p1**2 + 2*p2**2 - 2*p1*p2*cos(q1 - q2) + (-3 + cos(2*(q1 - q2)))*(2*cos(q1) + cos(q2))) / (-3 + cos(2*(q1 - q2)))"
  args.test_pct = 1
  args.master_port = 11572
  args.activation_fn = 'Tanh'
  args.learn_rate = 1e-03 
  args.tspan = [0, 7]  
  args.dsr = 0.0546875  
  args.batch_size = 1 
  args.hidden_dim = [32, 32] 
  args.train_pct = 0 
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  spsys = gen_dynsys(args)
  logmsg('args:\n{}'.format(args))

  for power in range(7,16):
    save_label = utils.get_label(args)
    npoints = 2**power
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

    logmsg('generating orbits')
    models = load_models(save_label)
    base_args, base_model, hnn_args, hnn_model = models
    spsys.tspan = [0, 100]
    spsys.time_points = abs(int((1.0 / args.dsr) * (spsys.tspan[1]-spsys.tspan[0])))
    orbits = gen_orbits(spsys, base_model, hnn_model)
    ground_orbit, ground_settings, base_orbit, base_settings, hnn_orbit, hnn_settings = orbits
    genergy = spsys.get_energy(ground_orbit)
    base_orbit_qp = orbit_qp(base_orbit)
    benergy = spsys.get_energy(base_orbit_qp)
    hnn_orbit_qp = orbit_qp(hnn_orbit)
    henergy = spsys.get_energy(hnn_orbit_qp)
    bdEfinal = abs(genergy[-1] - benergy[-1])
    bdEavg = np.absolute(genergy - benergy).mean()
    hdEfinal = abs(genergy[-1] - henergy[-1])
    hdEavg = np.absolute(genergy - henergy).mean()
    E = genergy[-1]
    f = open("nn_vs_hnn_energies_seed-{}.tsv".format(args.seed), "a")
    f.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(npoints, E, bdEfinal, bdEavg, hdEfinal, hdEavg))
    f.close()
