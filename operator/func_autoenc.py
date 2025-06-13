"""
Function Autoencoder 
Source: https://github.com/hanCi422/basis_operator_net/blob/main/src/kdv_burgers/model.py
"""
import torch  
import torch.nn as nn 
import torch.nn.functional as F 


class NeuralBasis(nn.Module):
    def __init__(self, dim_in=1, hidden=[4,4,4], n_base=4, activation=None):
        super().__init__()
        self.sigma = activation
        dim = [dim_in] + hidden + [n_base]
        self.layers = nn.ModuleList([nn.Linear(dim[i-1], dim[i]) for i in range(1, len(dim))])
        self.t_in = torch.tensor(grid_in).to(device)

    def forward(self, t):
        for i in range(len(self.layers)-1):
            t = self.sigma(self.layers[i](t))
        # linear activation at the last layer
        return self.layers[-1](t)

class FuncAutoenc(nn.Module):
    def __init__(self, hidden=[4,4,4], n_base=4, activation=None, device=None):
        super().__init__()
        self.n_base = n_base
        self.device = device 
        self.dim = 1
        self.BL = NeuralBasis(self.dim, hidden=hidden, n_base=n_base, activation=activation)
    
    def forward(self, x):
        """
        x: (batch_size, dim_in)
        """
        x = x.to(self.device)
        B = self.BL(x)
