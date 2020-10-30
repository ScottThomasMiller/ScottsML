import math
import numpy as np
import torch
import sympy as sp
from nail.hnn.utils import logmsg
from nail.hnn.data import DynamicalSystem

class SimplePendulumDS(DynamicalSystem):
    ''' This class extends the DynamicalSystem (data.py) module, customizing it for the simple, 
        nonlinear pendulum. '''
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
                 symplectic_order=4)
        self.energy = args.energy
        sympy_symbols = sp.symbols([tuple(args.state_symbols)])
        # && --------- Start of Changes for calculating p --------------
        self.sym_energy = sp.Symbol('energy')
        solutions = sp.solve(self.sys_fn - self.sym_energy, sympy_symbols[0][-1])
        q_sym = sympy_symbols[0][:-1]
        self.expr_p = solutions[1]
        self.expr_p_lam = sp.lambdify((self.sym_energy,) + q_sym, self.expr_p, 'numpy')
        # && --------- End of Changes for calculating p2 --------------

    def sample_orbits(self):
        ''' Extend the parent by converting both the input data ('coords') and the labeled training 
            data ('dcoords') position values, which represent angles in radians, into x/y coordinates. 
            HNN will use the x/y coordinates to map the data onto a cylinder. '''
        data, settings = super().sample_orbits()
        q = data['coords'][:,0]
        p = data['coords'][:,1]
        qdot = data['dcoords'][:,0]
        pdot = data['dcoords'][:,1]
        x = np.cos(q)
        y = np.sin(q)
        xdot = -np.sin(q)*qdot
        ydot = np.cos(q)*qdot
        data['coords'] = np.column_stack((x, y, p))
        data['dcoords'] = np.column_stack((xdot, ydot, pdot))

        return data, settings

    def random_config(self):
        ''' Generate and return a random state vector, customized for simple pendulum. '''
        energy = self.energy if self.energy else (2.0 + 0.11 * np.random.random()) 
        result = False
        while not result:
            with np.errstate(invalid='raise'):
                try:
                    # initial angles between -pi and pi:
                    q = math.pi * (1.0 - 2.0 * np.random.random())
                    p = 3.0 * (1.0 - 2.0 * np.random.random())
                    energy = p**2/2.0 + (1.0 - np.cos(q))
                    result = True
                except FloatingPointError:
                    logmsg("FP ERROR! q: {}, e: {}".format(q, energy))
                    continue

        state = np.array([q, p])
        cal_energy = self.get_energy(state)
        #logmsg("q = {:.4e}, p = {:.4e}, sampled energy = {:.4e}, calculated energy = {:.4e}".format(q, p, energy, cal_energy))

        return state
