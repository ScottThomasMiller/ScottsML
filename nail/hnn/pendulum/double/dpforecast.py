
import math 
import torch
import numpy as np
import pandas as pd

#NAIL modules:
from nail.hnn.blnn import BLNN
from dphnn import DoublePendulumHNN
from dpdata import DoublePendulumDS
from nail.hnn.utils import *
import nail.hnn.run as run

class DPforecast():
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.dpsys = self.gen_dynsys()

    def model_update(t, state, model):
        state = state.reshape(-1,6)
        deriv = np.zeros_like(state)
        np_x = state
        x = torch.tensor( np_x, requires_grad=True, dtype=torch.float32)
        dx_hat = model.time_derivative(x) 
        deriv= dx_hat.detach().data.numpy() 
    
        return deriv.reshape(-1)
    
    def gen_dynsys(self):
        ''' Create and return a DoublePendulumDS object, for generating orbits and ICs. '''
        dpsys = DoublePendulumDS(self.args)
        dpsys.tspan = self.args.tspan
        dpsys.time_points = abs(int((1.0 / self.args.dsr) * (dpsys.tspan[1]-dpsys.tspan[0])))
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
    
    def model_orbit(self, state0, numpts):
        ''' Generate and return the forecasted orbit using an Euler update. '''
        state0_xyp = np.zeros(6)
        state0_xyp[0] = np.cos(state0[0])
        state0_xyp[1] = np.sin(state0[0])
        state0_xyp[2] = np.cos(state0[1])
        state0_xyp[3] = np.sin(state0[1])
        state0_xyp[4] = state0[2]
        state0_xyp[5] = state0[3]
        self.model.eval()
        state = state0_xyp.reshape(-1,6)
        model_orbit = np.array(state)
        x = torch.tensor(state, requires_grad=True, dtype=torch.float32)
        for i in range(numpts-1):
            dxdt_hat = self.model.time_derivative(x)
            np_dxdt_hat = dxdt_hat.detach().data.numpy()
            np_x = x.detach().data.numpy()
            next = np_x + (np_dxdt_hat * 0.0001)
            model_orbit = np.row_stack((model_orbit, next))
            x = torch.tensor(next, requires_grad=True, dtype=torch.float32)
             
        return model_orbit
    
    def true_orbit(self, state0):
        ''' Generate and return the ground-truth orbit.'''
        self.dpsys.external_update_fn = None
        ground_orbit, ground_settings = self.dpsys.get_orbit(state0)
        self.orbit_len = ground_orbit.shape[1]
        
        return ground_orbit
    
    def write_orbit(self, orbit, fname):
        colnames = ['q1','q2','p1','p2']
        logmsg(f'writing orbit file {fname} shape {orbit.shape}')
        df = pd.DataFrame(orbit, columns=colnames)
        df.to_csv(fname, index=False)
    
