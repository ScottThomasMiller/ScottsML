from nail.hnn.blnn import BLNN
import numpy as np
import torch
import torch.distributed as dist
from nail.hnn.utils import logmsg

class HNN(BLNN):
    def __init__(self,  d_in, d_hidden, d_out, activation_fn):
        super(HNN, self).__init__(d_in, d_hidden, d_out, activation_fn)
        self.device = None

    def forward(self, x):
        y = super().forward(x)
        assert y.dim() == 2 and y.shape[1] == self.d_out
        ysplit = y.split(1, 1)

        return ysplit

    def time_derivative(self, x):
        ''' Return the symplectic gradients of the Hamiltonian, for each pair of
            coordinates.'''
        H, = self.forward(x)
        dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        num_pairs = x.shape[1] // 2
        vector_field = torch.cat((dH[:,num_pairs:], -dH[:,:num_pairs]), dim=1)

        return vector_field

    def validate(self, args, test_data, device):
        self.eval()
        #loss_fn = torch.nn.MSELoss().to(device)
        if args.input_noise != '':
          npnoise = np.array(args.input_noise, dtype=np.float32)
          noise = torch.tensor(npnoise).to(device)
        for x, dxdt in test_data:
            dxdt_hat = self.time_derivative(x)
            if args.input_noise != '':
                dxdt_hat += noise * torch.randn(*x.shape).to(device)  # add noise, maybe

            return self.loss_fn(dxdt, dxdt_hat).item()


