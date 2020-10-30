import copy 
import datetime
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from .utils import logmsg

''' module: blnn.py
    Authors: Anshul Choudary and Scott Miller, Nonlinear A.I. Lab (NAIL).  Scott wrote validate() and etrain().
             Anshul wrote the constructor, forward() and time_derivative() functions.
'''
class BLNN(torch.nn.Module):
    ''' MLP network implementation which serves as a Baseline Neural Network (BLNN), for comparison
        against the more advanced Hamiltonian Neural Network (HNN). 
        
        The outputs of the network represent the time derivatives of the canonical input coordinates (q, p). 
    '''

    stats_dict = {'training': [],
                  'testing': [],
                  'grad_norms': [],
                  'grad_stds': []}

    def __init__(self, d_in, d_hidden, d_out, activation_fn):
        ''' Construct the network:
            d_in: input dimension
            d_hidden: hidden layer dimension
            d_out: output dimension
        '''
        super(BLNN, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.nonlinearity = []
        self.d_in = d_in
        self.d_out = d_out
        self.d_hidden = d_hidden
        self.loss_fn = torch.nn.MSELoss()

        if activation_fn == 'Tanh':
            nonlinear_fn = torch.nn.Tanh()
        elif activation_fn == 'ReLU':
            nonlinear_fn = torch.nn.ReLU()

        self.layers.append(torch.nn.Linear(d_in, d_hidden[0]))
        self.nonlinearity.append(nonlinear_fn)

        for i in range(len(d_hidden) - 1):
            self.layers.append(torch.nn.Linear(d_hidden[i], d_hidden[i + 1]))
            self.nonlinearity.append(nonlinear_fn)

        self.last_layer = torch.nn.Linear(d_hidden[-1], d_out, bias=None)

        for i in range(len(self.layers)):
            torch.nn.init.orthogonal_(self.layers[i].weight)

        torch.nn.init.orthogonal_(self.last_layer.weight)

    def init_device(self):
        ''' Initialize the CPU/GPU device attributes, and move the loss function onto the
            selected device. '''
        self.device = self.state_dict()['layers.0.weight'].get_device()
        self.sdevice = torch.device(f"cuda:{self.device}" if self.device >= 0 else "cpu")
        self.loss_fn = self.loss_fn.to(self.sdevice)

        return self.device

    def forward(self, x):
        ''' Make one forward propagation of input tensor x through the entire network. Return
            the output layer.'''
        dict_layers = dict(zip(self.layers, self.nonlinearity))
        for layer, nonlinear_transform in dict_layers.items():
            out = nonlinear_transform(layer(x))
            x = out
        return self.last_layer(out)

    def time_derivative(self, x):
        ''' This function by default calls the forward() function.  It will be overridden by
            HNN, which will extend the functionality via Hamilton's symplectic gradient. '''
        return self(x)

    def validate(self, args, data_stream, device):
        ''' Test the model against the validation (holdout) data.  The loss will be used to
            determine how accurately the network is learning the time derivatives. '''
        self.eval()
        device = self.state_dict()['layers.0.weight'].get_device()
        if args.input_noise != '':
          npnoise = np.array(args.input_noise, dtype="float")
          noise = torch.tensor(npnoise).to(device)

        ''' The validation data stream is configured to return the entire set in a 
            single, randomized batch, therefore there will only ever be a single
            iteration. '''
        with torch.no_grad():
            for x, dxdt in data_stream:
                dxdt_hat = self.time_derivative(x)
                if args.input_noise != '':
                  dxdt_hat += noise * torch.randn(*x.shape).to(device)  # add noise, maybe
                return self.loss_fn(dxdt_hat, dxdt).item()

    def etrain(self, args, data_stream, optimizer):
        ''' Train one entire epoch of the input data stream. Return the statistics for the epoch:
            loss, gradient norm, and gradient standard deviation. '''
        self.train()
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        stats = copy.deepcopy(BLNN.stats_dict)
        addnoise = False
        if args.input_noise != '':
          addnoise = True
          npnoise = np.array(args.input_noise, dtype="float")
          noise = torch.tensor(npnoise).to(torchdev)

        batches = 0
        for x, dxdt in data_stream:
            optimizer.zero_grad()
            dxdt_hat = self.time_derivative(x)
            if addnoise:
              dxdt_hat += noise * torch.randn(*x.shape).to(torchdev)  # add noise, maybe
            loss = self.loss_fn(dxdt_hat, dxdt)
            loss.backward()
            if args.clip != 0:
              clip_grad_norm_(self.parameters(), args.clip)
            optimizer.step()

            grad = torch.cat([p.grad.flatten() for p in self.parameters()])
            stats['training'].append(loss.item())
            stats['grad_norms'].append((grad @ grad).item())
            stats['grad_stds'].append(grad.std().item())

            if args.verbose and ((batches % args.print_every == 0) and self.device <= 0):
              grad_norm = grad @ grad
              logmsg("batch[{}] train loss {:.4e}, grad norm: {:.4e}, grad std: {:.4e}"
                    .format(batches, loss.item(), grad_norm, grad.std()))
              logmsg('\tgrad: {}'.format(grad))
              for name, param in self.named_parameters():
                logmsg('\t{}: {}'.format(name, param.data))
              logmsg('x: {}'.format(x))
              logmsg('dxdt: {}'.format(dxdt))
            batches += 1

        return stats

    def set_label(self, label):
        ''' Helper function to set the label used to identify the current experiment. '''
        self.run_label = label

