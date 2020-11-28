from blnn import BLNN
import numpy as np
import torch
import torch.distributed as dist
from utils import logmsg
import sys

class HNN(BLNN):
    def __init__(self,  d_in, d_hidden, d_out, activation_fn):
        super(HNN, self).__init__(d_in, d_hidden, d_out, activation_fn)
        self.Mt = self.permutation_tensor(d_in)

    def forward(self, input):
        H, = super().forward(input).split(1,1)
        dH = torch.autograd.grad(H.sum(), input, create_graph=True)[0]
        dHdx1 = dH[:,0]
        dHdy1 = dH[:,1]
        dHdx2 = dH[:,2]
        dHdy2 = dH[:,3]
        dHdp1 = dH[:,4]
        dHdp2 = dH[:,5]
        x1 = input[:,0]
        y1 = input[:,1]
        x2 = input[:,2]
        y2 = input[:,3]
        p1 = input[:,4]
        p2 = input[:,5]
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

    def validate(self, args, test_data):
        self.eval()
        loss_fn = torch.nn.MSELoss().to("cuda")
        if args.input_noise != '':
          npnoise = np.array(args.input_noise, dtype=np.float32)
          noise = torch.tensor(npnoise).to("cuda")
        for x, dxdt in test_data:
            dxdt_hat = self.time_derivative(x)
            if args.input_noise != '':
                dxdt_hat += noise * torch.randn(*x.shape).to("cuda")  # add noise, maybe

            return loss_fn(dxdt, dxdt_hat).item()


