"""
Deep Learning for Functional Data Analysis with Adaptive Basis Layers
"""
#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tqdm import trange
import matplotlib.pyplot as plt
device = "cuda" if torch.cuda.is_available() else "cpu"

class LayerNorm(nn.Module):

    def __init__(self, d, eps=1e-6):
        super().__init__()
        # d is the normalization dimension
        self.d = d
        self.eps = eps
        self.alpha = nn.Parameter(torch.randn(d))
        self.beta = nn.Parameter(torch.randn(d))

    def forward(self, x):
        # x is a torch.Tensor
        # avg is the mean value of a layer
        avg = x.mean(dim=-1, keepdim=True)
        # std is the standard deviation of a layer (eps is added to prevent dividing by zero)
        std = x.std(dim=-1, keepdim=True) + self.eps
        return (x - avg) / std * self.alpha + self.beta

class NeuralBasis(nn.Module):
    def __init__(self, dim_in=1, hidden=[4,4,4], n_base=4, activation=None):
        super().__init__()
        self.sigma = activation
        dim = [dim_in] + hidden + [n_base]
        self.layers = nn.ModuleList([nn.Linear(dim[i-1], dim[i]) for i in range(1, len(dim))])

    def forward(self, t):
        for i in range(len(self.layers)-1):
            t = self.sigma(self.layers[i](t))
        # linear activation at the last layer
        return self.layers[-1](t)

def _inner_product(f1, f2, h):
    """    
    f1 - (J) : B functions, observed at J time points,
    f2 - (J) : same as f1
    h  - (J-1,1): weights used in the trapezoidal rule
    pay attention to dimension
    <f1, f2> = sum (h/2) (f1(t{j}) + f2(t{j+1}))
    """
    prod = f1 * f2 # (B, J = len(h) + 1)
    return torch.matmul((prod[:-1] + prod[1:]), h)/2

class AdaFNN(nn.Module):

    def __init__(self, n_base=4, base_hidden=[64, 64, 64], activation=torch.nn.ReLU()):
        """
        n_base      : number of basis nodes, integer
        base_hidden : hidden layers used in each basis node, array of integers
        grid        : observation time grid, array of sorted floats including 0.0 and 1.0
        sub_hidden  : hidden layers in the subsequent network, array of integers
        dropout     : dropout probability
        lambda1     : penalty of L1 regularization, a positive real number
        lambda2     : penalty of L2 regularization, a positive real number
        device      : device for the training
        """
        super().__init__()
        self.n_base = n_base
        self.device = device
        self.n_base = n_base
        self.dim_in = 1
        # instantiate each basis node in the basis layer
        self.BL = NeuralBasis(dim_in=self.dim_in, hidden=base_hidden, n_base = n_base, activation=activation)


    def forward(self, xs, us):
        scores = self.encode(xs, us)
        us_restore = self.decode(xs, scores)
        return us_restore
    
    def encode(self, xs, us):
        """
        xs: (B, J) : B functions, observed at J time points
        us: (B, J) : B functions, observed at J time points
        """
        B, J = xs.size()

        # send the time grid tensor to device
        hs = xs[:, 1:] - xs[:, :-1]# (B, J-1):  grid size
        # evaluate the current basis nodes at time grid
        basess = self.BL(xs.reshape(-1,1)).reshape(B,self.n_base,J) # (B, n_base, J)


        scores = torch.vmap(torch.vmap(_inner_product, in_dims=(0, None, None)), in_dims=(0, 0, 0))(basess, us, hs)
        # (B, n_bases, J), (B, J), (B, J-1)

        assert scores.size() == (B, self.n_base), f"Expected shape (B, n_base), got {scores.size()}"

        return scores
    
    def decode(self, xs, scores):
        """
        xs: (B, J) : B functions, observed at J time points 
        scores: (B, n_base) : B functions, encoded into n_base scores
        """
        B, J = xs.size()
        basess = self.BL(xs.reshape(-1,1)).reshape(B,J,self.n_base) # (B, J, n_base)
        us_restore = torch.bmm(basess, scores.unsqueeze(-1)).squeeze(-1)  # (B, J, n_base) x (B, n_base, 1) -> (B, J, 1)
        return us_restore

"""
Data Generation
"""
grids = torch.linspace(0, 1, 100)

def _phi(k, t):
    """
    basis functions
    k: mode 
    t: sensor point
    """
    return 2**0.5 * torch.cos((k-1)*torch.pi*t)

def generate_data(zs, xs, n_batch=100):
    """
    Generate data based on the given weights of basis functions and random coefficients.

    Args:
        zs: weight of basis functions 
        xs: sensor points (batch_size, J)
        n_batch (int, optional): number of batch. Defaults to 100.
    Return: 
        us: function response at sensor points (n_batch, J)
    """
    n_basis = zs.size(0)
    rs = torch.Tensor.uniform_(torch.empty(n_batch, n_basis), -3**0.5, 3**0.5)
    cs = zs * rs 
    ks = torch.arange(1, n_basis + 1)
    phis = torch.vmap(torch.vmap(_phi, in_dims=(0, None)), in_dims=(None,0))(ks, xs)

    us = torch.bmm(phis.transpose(1,2), cs.unsqueeze(-1))  # (n_batch, J)
    return xs, us.squeeze(dim=-1)  # (n_batch, J)

def train(model, nstep, optimizer, data_gen):
    """
    Train the model for nstep steps.
    """
    model.train()
    xs, us = data_gen()
    xs = xs.to(device)
    us = us.to(device)
    tstep = trange(nstep, desc="Training")
    for step in tstep:
        # Generate new data for each step
        optimizer.zero_grad()
        # Forward pass
        us_restore = model(xs, us)
        # Compute loss (mean squared error)
        loss = F.mse_loss(us_restore, us)
        # Backward pass
        loss.backward()
        optimizer.step()

        tstep.set_postfix(loss=loss.item())
    return model

if __name__ == "__main__":
    """
    Data
    """
    # Data Generation
    n_batch = 64
    # Weights of basis functions
    zs = torch.asarray([20, 5, 5] + [1] * 47)
    # Sensor Points
    n_xs = 100
    xs = torch.linspace(0, 1, n_xs).repeat(n_batch, 1)  # (n_batch, J)
    # Function response
    xs, Us = generate_data(zs, xs, n_batch=n_batch) # (n_batch, J)
    data_gen = lambda: generate_data(zs, xs, n_batch=n_batch)  
    """
    Model
    """
    n_base = 9
    # Create Function Autoencoder Model 
    model = AdaFNN(n_base=n_base, base_hidden=[64,64,64]).to(device)  # Change 'cpu' to 'cuda' if using GPU
    # Encode function into scores
    scores = model.encode(xs.to(device), Us.to(device))
    # Decode scores back to function
    us_restore = model.decode(xs.to(device), scores)
    # Autoencoder
    us_restore2 = model(xs.to(device), Us.to(device))

    # Sanity checks
    assert xs.size() == Us.size()
    assert scores.size() == (n_batch, n_base)   
    assert us_restore.size() == (n_batch, n_xs)
    assert torch.allclose(us_restore, us_restore2, atol=1e-6)

    # Train the model
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model_opt = train(model, nstep=60000, optimizer=opt, data_gen=data_gen)


# %%
