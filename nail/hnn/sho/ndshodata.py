import math
import numpy as np
from nail.hnn.utils import logmsg
from nail.hnn.data import DynamicalSystem
import sympy as sp

class SimpleHarmonicOscillatorDS(DynamicalSystem):
    ''' creates a dynamical system for simulating an N-body
        simple harmonic oscillator '''
    def __init__(self, args):
        self.init_time(args)
        self.init_sym(args)
        super().__init__(
                 sys_hamiltonian=self.hamiltonian, 
                 split_ratio=args.split_ratio, 
                 state_symbols=self.state_symbols, 
                 no_trajectories=args.trajectories, 
                 timesteps=self.tpoints, 
                 tspan=self.tspan, 
                 verbose=args.verbose, 
                 integrator=args.integrator_scheme, 
                 symplectic_order=2*args.num_bodies)
        self.energy = args.energy
        # && --------- Start of Changes for calculating pN --------------
        self.sym_energy = sp.Symbol('energy')
        solutions = sp.solve(self.sys_fn - self.sym_energy, self.sympy_symbols[0][-1])
        rightmost = ['energy']
        for q in self.state_symbols[:-1]:
          rightmost.append(q)
        sym_p_lam = sp.symbols([tuple(rightmost)])
        #self.expr_p = solutions[1]
        self.expr_p = solutions[0]
        self.expr_p_lam = sp.lambdify(sym_p_lam, self.expr_p, 'numpy')
        # && --------- End of Changes for calculating p2 --------------

    def init_time(self, args):
        ''' initializes the temporal attributes '''
        self.tspan = []
        for t in args.tspan:
            self.tspan.append(t)
        self.tpoints = int((1.0 / args.dsr) * self.tspan[1])

    def init_sym(self, args):
        ''' initializes the state symbols and Hamiltonian based on num bodies and 
            (p**2 + q**2) / 2 '''
        self.num_bodies = args.num_bodies
        fullham = ""
        qsyms = []
        psyms = []
        self.state_symbols = []
        exp = args.exponent if 'exponent' in args.__dict__.keys() else 2
        for n in range(args.num_bodies):
          qsym = "q"+str(n+1)
          psym = "p"+str(n+1)
          subham = f"+(({psym}**{exp}+{qsym}**{exp})/2)"
          fullham += subham
          qsyms.append(qsym)
          psyms.append(psym)
        self.hamiltonian = fullham
        self.state_symbols = np.hstack((qsyms, psyms))

    def random_config(self):
        '''
        generate and return a random initial state vector
        '''
        energy = -1.0
        ntries = 0
        while (energy < 0) or (energy > 1.0):
            ntries += 1
            state0 = []
            for s in self.state_symbols:
                q = 1.0 - 2.0 * np.random.random()
                state0.append(q)
            energy = self.get_energy(state0)
        state0 = np.asarray(state0)
        #logmsg(f"energy: {energy} num tries: {ntries} state0: {state0}")

        return state0
