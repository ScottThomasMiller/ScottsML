import numpy as np
import torch
import torch.distributed as dist
import sys

#NAIL modules:
from nail.hnn.sho.hnn import HNN 
from nail.hnn.utils import *

class DoublePendulumHNN(HNN):
    def __init__(self,  d_in, d_hidden, d_out, activation_fn, beta=0.5):
        super(HNN, self).__init__(d_in, d_hidden, d_out, activation_fn, beta)
        self.Mt = self.permutation_tensor(d_in)

    def time_derivative(self, vinput, t=None):
        H, = self.forward(vinput)
        dH = torch.autograd.grad(H.sum(), vinput, create_graph=True)[0]
        dHdx1 = dH[:,0]
        dHdy1 = dH[:,1]
        dHdx2 = dH[:,2]
        dHdy2 = dH[:,3]
        dHdp1 = dH[:,4]
        dHdp2 = dH[:,5]
        x1 = vinput[:,0]
        y1 = vinput[:,1]
        x2 = vinput[:,2]
        y2 = vinput[:,3]
        p1 = vinput[:,4]
        p2 = vinput[:,5]
        x1hdot = -y1 * dHdp1
        y1hdot = x1 * dHdp1
        x2hdot = -y2 * dHdp2
        y2hdot = x2 * dHdp2
        p1hdot = x1.clone()
        p2hdot = x2.clone()
        p1hdot = torch.where(abs(x1) < abs(y1), dHdx1/y1, -dHdy1/x1)
        p2hdot = torch.where(abs(x2) < abs(y2), dHdx2/y2, -dHdy2/x2)
        voutput = torch.stack((x1hdot, y1hdot, x2hdot, y2hdot, p1hdot, p2hdot), dim=1)

        return voutput

    def permutation_tensor(self, n):
        M = None
        M = torch.eye(n)
        M = torch.cat([M[n // 2:], -M[:n // 2]])
        Mt = M.t().type(torch.Tensor)

        return Mt

    def validate(self, args, test_data, device):
        self.eval()
        self.device = device
        if device >= 0:
          loss_fn = loss_fn.to(device)
        if args.input_noise != '':
          npnoise = np.array(args.input_noise, dtype=np.float32)
          noise = torch.tensor(npnoise).to(device)
        for x, dxdt, nextx in test_data:
            dxdt_hat = self.time_derivative(x)
            if args.input_noise != '':
                dxdt_hat += noise * torch.randn(*x.shape).to(device)  # add noise, maybe
            #return loss_fn(dxdt, dxdt_hat).item()
            #nextx_hat = x + dxdt_hat 
            #return self.loss_fn(nextx_hat, nextx).item()
            return self.custom_loss(x, nextx, dxdt, dxdt_hat)


