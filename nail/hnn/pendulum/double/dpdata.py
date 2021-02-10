'''
File: dpdata.py
File Created: Sep 2019
Author: Scott Miller
'''

import math
import numpy as np
import torch
from torch.utils.data import Dataset
import sympy as sp
from nail.hnn.utils import logmsg
from nail.hnn.data import DynamicalSystem

class DoublePendulumDS(DynamicalSystem):
    def __init__(self, args):
        tspan = []
        for t in args.tspan:
            tspan.append(t)
        tpoints = int((1.0 / args.dsr) * tspan[1])
        super().__init__(
                 sys_hamiltonian=args.hamiltonian, 
                 split_ratio=args.split_ratio, 
                 state_symbols=args.state_symbols, 
                 no_trajectories=args.trajectories, 
                 timesteps=tpoints, 
                 tspan=tspan, 
                 verbose=args.verbose, 
                 integrator=args.integrator_scheme, 
                 symplectic_order=8)
        self.energy = args.energy
        sympy_symbols = sp.symbols([tuple(args.state_symbols)])

    def sample_orbits(self):
        data, settings = super().sample_orbits()
        # Convert the default state vector into cylindrical components:
        # Before:
        # coords = [q1,q2,p1,p2]
        # dcoords = [q1dot,q2dot,p1dot,p2dot]
        # After:
        # coords = [x1,y1,x2,y2,p1,p2]
        # dcoords = [x1dot,y1dot,x2dot,y2dot,p1dot,p2dot]
        q1 = data['coords'][:,0]
        q2 = data['coords'][:,1]
        p1 = data['coords'][:,2]
        p2 = data['coords'][:,3]
        q1dot = data['dcoords'][:,0]
        q2dot = data['dcoords'][:,1]
        p1dot = data['dcoords'][:,2]
        p2dot = data['dcoords'][:,3]
        x1 = np.cos(q1)
        x2 = np.cos(q2)
        y1 = np.sin(q1)
        y2 = np.sin(q2)
        x1dot = -np.sin(q1)*q1dot
        x2dot = -np.sin(q2)*q2dot
        y1dot = np.cos(q1)*q1dot
        y2dot = np.cos(q2)*q2dot
        data['coords'] = np.column_stack((x1,y1,x2,y2,p1,p2))
        data['dcoords'] = np.column_stack((x1dot,y1dot,x2dot,y2dot,p1dot,p2dot))

        return data, settings

    def random_config(self):
        '''
        generate and return a random state vector, customized for double pendulum.    
        '''
        energy = self.energy if self.energy else (0.02 + 0.11 * np.random.random())
        result = False
        while not result:
            with np.errstate(invalid='raise'):
                try:
                    q1,q2,p2 = (math.pi/2) * (1.0 - 2.0 * np.random.random(3))
                    p1 = (math.pi) * (1.0 - 2.0 * np.random.random())
                    result = True
                except FloatingPointError:
                    logmsg("FP ERROR! q1: {}, q2: {}, p1: {}, e: {}".format(q1, q2, p1, energy))
                    continue

        state = np.array([q1,q2,p1,p2])
        cal_energy = self.get_energy(state)

        return state

class DataStream(Dataset):
    def __init__(self, data):
        self.x = data['x']
        self.dxdt = data['dxdt']
        self.len = len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.dxdt[index]

    def __len__(self):
        return self.len


