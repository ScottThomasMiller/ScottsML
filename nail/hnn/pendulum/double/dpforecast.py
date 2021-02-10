
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
    x = torch.Tensor( np_x, requires_grad=True, dtype=torch.float32)
    dx_hat = model.time_derivative(x)
    deriv= dx_hat.detach().data.numpy() * 100.0
    
    return deriv.reshape(-1)

def gen_dynsys(args):
    ''' Create and return a DoublePendulumDS object, for generating orbits and ICs. '''
    dpsys = DoublePendulumDS(args)
    dpsys.tspan = args.tspan
    dpsys.time_points = abs(int((1.0 / args.dsr) * (dpsys.tspan[1]-dpsys.tspan[0])))
    dpsys.integrator = "RK45"

    return dpsys

def load_model(clargs):
    fname = f'{clargs.save_dir}/{clargs.load_model}'
    model_dict = from_pickle(fname)
    margs = model_dict['args']
    if margs.model == 'baseline':
        model = BLNN(d_in=6, d_hidden=margs.hidden_dim, d_out=6, activation_fn=margs.activation_fn)
    else:
        model = DoublePendulumHNN(d_in=6, d_hidden=margs.hidden_dim, d_out=1, activation_fn=margs.activation_fn)
    model.load_state_dict(model_dict['model'])
    model.eval()

    return (margs, model)

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

def gen_orbits(dpsys, model, state0):
    save_state_xyp = np.zeros(6)
    save_state_xyp[0] = np.cos(state0[0])
    save_state_xyp[1] = np.sin(state0[0])
    save_state_xyp[2] = np.cos(state0[1])
    save_state_xyp[3] = np.sin(state0[1])
    save_state_xyp[4] = state0[2]
    save_state_xyp[5] = state0[3]
    save_state_qp = xy2q(save_state_xyp)
    model.eval()
    dpsys.external_update_fn = None
    state = save_state_qp
    ground_orbit, ground_settings = dpsys.get_orbit(state)

    state = save_state_xyp
    update_fn = lambda t, y0: model_update(t, y0, model)
    dpsys.external_update_fn = update_fn
    model_orbit, model_settings = dpsys.get_orbit(state)
    '''
    state = save_state_xyp.reshape(-1,6)
    model_orbit = np.array(state)
    x = torch.tensor(state, requires_grad=True, dtype=torch.float32)
    for i in range(ground_orbit.shape[1]-1):
        dxdt_hat = model.time_derivative(x)
        np_dxdt_hat = dxdt_hat.detach().data.numpy()
        np_x = x.detach().data.numpy()
        next = np_x + np_dxdt_hat
        #model_orbit = np.row_stack((model_orbit, next))
        x = torch.tensor(next, requires_grad=True, dtype=torch.float32)
    '''
         
    return (ground_orbit, model_orbit.T)

def write_orbits(args, orbits, i):
    ''' Write the ground-truth and forecasted orbits to separate .csv files. '''
    colnames = ['q1','q2','p1','p2']
    ground_orbit, model_orbit = orbits
    logmsg(f'ground shape: {ground_orbit.shape}, model shape: {model_orbit.shape}')

    fname = f'{args.save_dir}/orbit_ground_{i}.csv'
    logmsg(f'writing orbit file {fname}')
    df = pd.DataFrame(ground_orbit.T, columns=colnames)
    df.to_csv(fname, index=False)

    fname = f'{args.save_dir}/orbit_model_{i}.csv'
    logmsg(f'writing orbit file {fname}')
    df = pd.DataFrame(xy2q(model_orbit).T, columns=colnames)
    df.to_csv(fname, index=False)

def forecast(args, model, dpsys):
    ''' Generate multiple orbits, each with its own IC, for the given model.
        Write each orbit to separate csv files. '''
    for i in range(args.num_forecasts):
        state0 = dpsys.random_config()
        orbits = gen_orbits(dpsys, model, state0)
        write_orbits(args, orbits, i)

if __name__ == "__main__":
  ''' models are in x/y/p format, for nonlinear pendulum '''
  in_dim = 6
  clargs = get_args()
  torch.manual_seed(clargs.seed)
  np.random.seed(clargs.seed)
  dpsys = gen_dynsys(clargs)
  logmsg('args:\n{}'.format(clargs))
  logmsg('generating orbits')
  margs, model = load_model(clargs)
  logmsg('forecasting orbits');
  forecast(clargs, model, dpsys)
