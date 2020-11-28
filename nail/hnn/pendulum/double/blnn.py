import copy 
import datetime
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from utils import logmsg

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

    def forward(self, x):
        dict_layers = dict(zip(self.layers, self.nonlinearity))
        for layer, nonlinear_transform in dict_layers.items():
            out = nonlinear_transform(layer(x))
            x = out
        return self.last_layer(out)

    def time_derivative(self, x):
        return self(x)

