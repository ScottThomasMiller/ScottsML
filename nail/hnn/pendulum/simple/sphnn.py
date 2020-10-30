from nail.hnn.sho.hnn import HNN 
from nail.hnn.utils import *
import numpy as np
import torch
import torch.distributed as dist
import sys

class SimplePendulumHNN(HNN):
    ''' Extend the simple harmonic oscillator (SHO) HNN class by overriding time_derivative(), to perform the cylinder mapping. '''
    def __init__(self,  d_in, d_hidden, d_out, activation_fn):
        super(SimplePendulumHNN, self).__init__(d_in, d_hidden, d_out, activation_fn)

    def time_derivative(self, vinput, t=None):
        H, = self.forward(vinput)
        dH = torch.autograd.grad(H.sum(), vinput, create_graph=True)[0]
        dHdx = dH[:,0]
        dHdy = dH[:,1]
        dHdp = dH[:,2]
        x = vinput[:,0]
        y = vinput[:,1]
        p = vinput[:,2]
        xhdot = -y * dHdp
        yhdot = x * dHdp
        phdot = x.clone()
        phdot = torch.where(abs(x) < abs(y), dHdx/y, -dHdy/x)
        voutput = torch.stack((xhdot, yhdot, phdot), dim=1)

        return voutput

