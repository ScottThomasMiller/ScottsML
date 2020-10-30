from nail.hnn.blnn import BLNN
import numpy as np
import torch
import torch.distributed as dist
from nail.hnn.utils import logmsg

class HNN(BLNN):
    ''' This class is an extension of the Baseline Neural Network (BLNN) class.  The "special sauce" of
        HNN lies within time_derivative(), where the symplectic magic happens. '''

    def __init__(self,  d_in, d_hidden, d_out, activation_fn):
        super(HNN, self).__init__(d_in, d_hidden, d_out, activation_fn)
        self.device = None

    def forward(self, x):
        ''' Call the BLNN forward(), and then check to make sure the network output is 2D. '''
        y = super().forward(x)
        assert y.dim() == 2 and y.shape[1] == self.d_out
        ysplit = y.split(1, 1)

        return ysplit

    def time_derivative(self, x):
        ''' Return the symplectic gradients of the Hamiltonian, for each pair of
            coordinates.'''
        H, = self.forward(x)
        dHdx = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        dxdH = torch.autograd.grad(x.sum(), H, create_graph=True)[0]
        num_pairs = x.shape[1] // 2
        dHdx = torch.cat((dHdx[:,num_pairs:], -dHdx[:,:num_pairs]), dim=1)
        dxdH = torch.cat((dxdH[:,num_pairs:], -dxdH[:,:num_pairs]), dim=1)

        return (dHdx, dxdH)

    def validate(self, args, test_data, device):
        ''' Run the test input through the model, to calculate and return its loss. '''
        self.eval()
        if args.input_noise != '':
          npnoise = np.array(args.input_noise, dtype=np.float32)
          noise = torch.tensor(npnoise).to(device)
        for x, dHdx in test_data:
            dHdx_hat, dxdH_hat = self.time_derivative(x)
            if args.input_noise != '':
                dHdx_hat += noise * torch.randn(*x.shape).to(device)  # add noise, maybe
            return self.loss_fn(args, dHdx, dHdx_hat, x, dxdH_hat).item()

