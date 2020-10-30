import math
import numpy as np
from nail.hnn.utils import logmsg
from nail.hnn.data import DynamicalSystem
import sympy as sp

class SimpleHarmonicOscillatorDS(DynamicalSystem):
    ''' creates a dynamical system for simulating a simple harmonic oscillator '''
    def __init__(self, args):
        self.init_time(args)
        super().__init__(
                 sys_hamiltonian=args.hamiltonian, 
                 split_ratio=args.split_ratio, 
                 state_symbols=args.state_symbols, 
                 no_trajectories=args.trajectories, 
                 timesteps=self.tpoints, 
                 tspan=self.tspan, 
                 verbose=args.verbose, 
                 integrator=args.integrator_scheme, 
                 symplectic_order=2)
        self.energy = args.energy
        sympy_symbols = sp.symbols([tuple(self.state_symbols)])
        # && --------- Start of Changes for calculating pN --------------
        self.sym_energy = sp.Symbol('energy')
        solutions = sp.solve(self.sys_fn - self.sym_energy, sympy_symbols[0][-1])
        q_sym = sympy_symbols[0][:-1]
        self.expr_p = solutions[1]
        self.expr_p_lam = sp.lambdify((self.sym_energy,) + q_sym, self.expr_p, 'numpy')
        # && --------- End of Changes for calculating p2 --------------

    def init_time(self, args):
        ''' initializes the temporal attributes '''
        self.tspan = []
        for t in args.tspan:
            self.tspan.append(t)
        self.tpoints = int((1.0 / args.dsr) * self.tspan[1])

    def random_config(self):
        '''
        generate and return a random state vector, customized for simple pendulum.    
        '''
        energy = self.energy if self.energy else (2.0 + 0.11 * np.random.random()) 
        result = False
        while not result:
            with np.errstate(invalid='raise'):
                try:
                    # initial angles between -pi and pi:
                    q = math.pi * (1.0 - 2.0 * np.random.random())
                    p = self.expr_p_lam(energy, q)
                    result = True
                except FloatingPointError:
                    continue

        state = np.array([q, p])
        cal_energy = self.get_energy(state)
        logmsg("q = {:.4e}, p = {:.4e}, sampled energy = {:.4e}, calculated energy = {:.4e}".format(q, p, energy, cal_energy))

        return state
