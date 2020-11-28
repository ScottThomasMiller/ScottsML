'''
File: dpdata.py
File Created: Sep 2019
Author: Scott Miller
'''

import math
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import logmsg
from data import DynamicalSystem
import sympy as sp

class DoublePendulumDS(DynamicalSystem):
    def __init__(self, sys_hamiltonian, split_ratio=0.01, state_symbols=['q1','q2','p1','p2'],
                 no_trajectories=10000, timesteps=1000, tspan=[0, 1000], verbose=False, 
                 integrator="RK45", symplectic_order=2, energy=None):
        super().__init__(
                 sys_hamiltonian= sys_hamiltonian, 
                 split_ratio=split_ratio, 
                 state_symbols=state_symbols, 
                 no_trajectories=no_trajectories, 
                 timesteps=timesteps, 
                 tspan=tspan, 
                 verbose=verbose, 
                 integrator=integrator, 
                 symplectic_order=symplectic_order)
        self.energy = energy
        sympy_symbols = sp.symbols([tuple(state_symbols)])
        self.sym_energy = sp.Symbol('energy')
        #--------- Start of Changes for calculating p --------------
        solutions = sp.solve(self.sys_fn - self.sym_energy, sympy_symbols[0][-1])
        self.expr_p2 = solutions[0]
        self.expr_2_lam = sp.lambdify((self.sym_energy,) + sympy_symbols[0][:-1], self.expr_p2, 'numpy')
        #print("-----------------------------------------")
        #for i in range(len(solutions)):
        #  print("\tp2 solution[{}] = {}".format(i, solutions[i]))
        # && --------- End of Changes for calculating p2 --------------

    def sample_orbits(self):
        data, settings = super().sample_orbits()
        # Convert the default state vector into Cartesian components:
        # Before:
        # coords = [q, p]
        # dcoords = [qdot, pdot]
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
        # After:
        # coords = [x1, y1, x2, y2, p1, p2]
        # dcoords = [x1dot, y1dot, x2dot, y2dot, p1dot, p2dot]
        data['coords'] = np.column_stack((x1, y1, x2, y2, p1, p2))
        data['dcoords'] = np.column_stack((x1dot, y1dot, x2dot, y2dot, p1dot, p2dot))

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
                    p1 = math.pi * (1.0 - 2.0 * np.random.random())
                    #p2 = self.expr_2_lam(energy, q1, q2, p1)
                    result = True
                except FloatingPointError:
                    logmsg("FP ERROR! q1: {}, q2: {}, p1: {}, e: {}".format(q1, q2, p1, energy))
                    continue

        state = np.array([q1, q2, p1, p2])
        cal_energy = self.get_energy(state)
        #logmsg("q = {:.4e}, p = {:.4e}, sampled energy = {:.4e}, calculated energy = {:.4e}".format(q, p, energy, cal_energy))
        logmsg("state: {}, energy = {}, cal_energy: {}".format(state, energy, cal_energy))

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


