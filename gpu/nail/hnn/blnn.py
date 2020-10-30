import copy 
import datetime
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from .utils import logmsg

class BLNN(torch.nn.Module):
    ''' Classic neural network implementation which serves as a baseline NN model '''

    stats_dict = {'training': [],
                  'testing': [],
                  'grad_norms': [],
                  'grad_stds': []}

    def __init__(self, d_in, d_hidden, d_out, activation_fn):
        super(BLNN, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.nonlinearity = []
        self.d_in = d_in
        self.d_out = d_out
        self.d_hidden = d_hidden
        #self.loss_fn  = BLNN.xy_loss
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
        #logmsg("model: {}".format(self))

    def init_device(self):
        self.device = self.state_dict()['layers.0.weight'].get_device()
        self.sdevice = torch.device(f"cuda:{self.device}" if self.device >= 0 else "cpu")
        self.loss_fn = self.loss_fn.to(self.sdevice)

        return self.device

    def forward(self, x):
        dict_layers = dict(zip(self.layers, self.nonlinearity))
        for layer, nonlinear_transform in dict_layers.items():
            out = nonlinear_transform(layer(x))
            x = out
        return self.last_layer(out)

    def time_derivative(self, x):
        return self(x)

    def average_gradients(self):
        size = float(dist.get_world_size())
        for param in self.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size

    def validate(self, args, data, device):
        self.eval()
        #loss_fn = torch.nn.MSELoss().to(device)
        device = self.state_dict()['layers.0.weight'].get_device()
        if args.input_noise != '':
          npnoise = np.array(args.input_noise, dtype="float")
          noise = torch.tensor(npnoise).to(device)

        # the validation data stream is configured to return the entire set in a 
        # single, randomized batch:
        with torch.no_grad():
            for x, dxdt in data:
                dxdt_hat = self.time_derivative(x)
                if args.input_noise != '':
                  dxdt_hat += noise * torch.randn(*x.shape).to(device)  # add noise, maybe
                return self.loss_fn(dxdt_hat, dxdt).item()
                #return loss_fn(dxdt_hat, dxdt).item()

    # epoch train:
    def etrain(self, args, train_data, optimizer):
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
        for x, dxdt in train_data:
            optimizer.zero_grad()
            dxdt_hat = self.time_derivative(x)
            if addnoise:
              dxdt_hat += noise * torch.randn(*x.shape).to(torchdev)  # add noise, maybe
            loss = self.loss_fn(dxdt_hat, dxdt)
            loss.backward()
            self.average_gradients()
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
        self.run_label = label

    def xy_loss(dxhat, dx):
        loss = ((dxhat - dx)**2).sum()
        return loss
