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

    def custom_loss(self, x, nextx, dxdt, dxdt_hat):
        nextx_hat = x + (dxdt_hat * 0.0001)
        #loss = (self.beta * torch.mean((dxdt - dxdt_hat)**2)) + \
        #       ((1.0 - self.beta) * torch.mean((nextx - nextx_hat)**2))
        loss = torch.mean((self.beta * (dxdt - dxdt_hat)**2) + \
                         ((1.0 - self.beta) * (nextx - nextx_hat)**2))
 
        return loss

    def custom_loss_old(outputs, targets, coefficients=None):
        ''' Function custom_loss returns a weighted sum of MSE losses across 
            multiple sets of output/target pairs. '''
        total_loss = torch.zeros((outputs.size[0]))
        for o in range(outputs.size[0]):
            #loss = torch.mean((outputs[o] - targets[o])**2)
            loss = torch.nn.MSELoss(outputs[o], targets[o])
            if coefficients is not None:
                loss *= coefficients[o]
            total_loss += loss
        
        return total_loss

    def __init__(self, d_in, d_hidden, d_out, activation_fn, beta=0.5):
        super(BLNN, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.nonlinearity = []
        self.d_in = d_in
        self.d_out = d_out
        self.d_hidden = d_hidden
        self.loss_fn = torch.nn.MSELoss()
        self.beta = beta

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

    def validate(self, args, data, device):
        self.eval()
        device = self.state_dict()['layers.0.weight'].get_device()
        #loss_fn = torch.nn.MSELoss().to(device)
        if args.input_noise != '':
          npnoise = np.array(args.input_noise, dtype="float")
          noise = torch.tensor(npnoise).to(device)

        # the validation data stream is configured to return the entire set in a 
        # single, randomized batch:
        with torch.no_grad():
            for x, dxdt, nextx in data:
                dxdt_hat = self.time_derivative(x)
                if args.input_noise != '':
                  dxdt_hat += noise * torch.randn(*x.shape).to(device)  # add noise, maybe
                #return self.loss_fn(dxdt_hat, dxdt).item()
                #nextx_hat = x + dxdt_hat 
                #return self.loss_fn(nextx_hat, nextx).item()
                return self.custom_loss(x, nextx, dxdt, dxdt_hat)

    # epoch train:
    def etrain(self, args, train_data, optimizer):
        self.train()
        stats = copy.deepcopy(BLNN.stats_dict)
        addnoise = False
        if args.input_noise != '':
          addnoise = True
          npnoise = np.array(args.input_noise, dtype="float")
          noise = torch.tensor(npnoise).to(torchdev)

        batches = 0
        for x, dxdt, nextx in train_data:
            optimizer.zero_grad()
            dxdt_hat = self.time_derivative(x)
            if addnoise:
              dxdt_hat += noise * torch.randn(*x.shape).to(torchdev)  # add noise, maybe
            #loss = self.loss_fn(dxdt_hat, dxdt)
            #nextx_hat = x + dxdt_hat 
            #loss = self.loss_fn(nextx_hat, nextx)
            loss = self.custom_loss(x, nextx, dxdt, dxdt_hat)
            loss.backward()
            if args.clip != 0:
              clip_grad_norm_(self.parameters(), args.clip)
            optimizer.step()

            grad = torch.cat([p.grad.flatten() for p in self.parameters()])
            stats['training'].append(loss.item())
            stats['grad_norms'].append((grad @ grad).item())
            stats['grad_stds'].append(grad.std().item())

            if args.verbose and (batches % args.print_every == 0): 
              grad_norm = grad @ grad
              logmsg(f'batch size: {x.shape[0]}')
              logmsg("batch[{}] train loss {:.4e}, grad norm: {:.4e}, grad std: {:.4e}"
                    .format(batches, loss.item(), grad_norm, grad.std()))
            batches += 1

        return stats

    def set_label(self, label):
        self.run_label = label

    def xy_loss(dxhat, dx):
        loss = ((dxhat - dx)**2).sum()
        return loss
