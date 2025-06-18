"""
Deep Learning for Functional Data Analysis with Adaptive Basis Layers

Reference:
1. https://github.com/jwyyy/AdaFNN
2. https://github.com/hanCi422/basis_operator_net/tree/main
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

class NeuralBasis(nn.Module):
    def __init__(self, dim_in=1, hidden=[4,4,4], n_base=4, activation=None, fourier_feature=None):
        super().__init__()
        self.sigma = activation
        self.fourier_feature = fourier_feature
        if fourier_feature is not None:
            m=100
            dim_in = 2*m
            sigma = 5
            B1 = torch.rand(m, 1) * sigma
            B2 = torch.rand(m, 1) * sigma
            self.B1= nn.Parameter(B1, requires_grad=True)
            self.B2 = nn.Parameter(B2, requires_grad=True)

        dim = [dim_in] + hidden + [n_base]
        self.layers = nn.ModuleList([nn.Linear(dim[i-1], dim[i]) for i in range(1, len(dim))])

    def forward(self, t):
        t = self.get_fourier_feature(t)  # (B, J, 2M) if fourier_feature is not None else (B, J)
        for i in range(len(self.layers)-1):
            t = self.sigma(self.layers[i](t))
        # linear activation at the last layer
        return self.layers[-1](t)

    def get_fourier_feature(self, xs):
        _xs = xs # (B, 1)
        if self.fourier_feature is not None:
            Bxs = torch.einsum("BD, MD-> BM", _xs, self.B1)
            sin_xs = torch.sin(Bxs)  # (B, J, M)
            cos_xs = torch.cos(Bxs)  # (B, J, M)
            feature = torch.cat([sin_xs, cos_xs], dim=-1)  # (B, J, 2M)
            return feature
        else: 
            return xs

def _inner_product(f1, f2, h):
    """    
    f1 - (J) : observed at J time points,
    f2 - (J) : same as f1
    h  - (J-1,1): weights used in the trapezoidal rule
    pay attention to dimension
    <f1, f2> = sum (h/2) (f1(t{j}) + f2(t{j+1}))
    """
    prod = f1 * f2 # (B, J = len(h) + 1)
    return torch.matmul((prod[:-1] + prod[1:]), h)/2

def _l1(f, h):
    # f dimension : ( B bases, J )
    B, J = f.size()
    return _inner_product(torch.abs(f), torch.ones((B, J)), h)


def _l2(f, h):
    # f dimension : ( B bases, J )
    # output dimension - ( B bases, 1 )
    return torch.sqrt(_inner_product(f, f, h)) 

class AdaFNN(nn.Module):

    def __init__(self, n_base=4, base_hidden=[64, 64, 64], activation=torch.nn.ReLU(), fourier_feature=None):
        """
        labda1      : penalty of L1 regularization, a positive real number
        lambda2     : penalty of L2 regularization, a positive real number
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
        self.fourier_feature = fourier_feature
        # instantiate each basis node in the basis layer

        dim_in = 1
        self.BL = NeuralBasis(dim_in=dim_in, hidden=base_hidden, n_base = n_base, activation=activation, fourier_feature=fourier_feature)

    def forward(self, xs, us):
        scores, basess = self.encode(xs, us)
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

        return scores, basess
    
    def decode(self, xs, scores):
        """
        xs: (B, J) : B functions, observed at J time points 
        scores: (B, n_base) : B functions, encoded into n_base scores
        """
        B, J = xs.size()
        basess = self.BL(xs.reshape(-1,1)).reshape(B,J,self.n_base) # (B, J, n_base)
        us_restore = torch.bmm(basess, scores.unsqueeze(-1)).squeeze(-1)  # (B, J, n_base) x (B, n_base, 1) -> (B, J, 1)
        return us_restore

    def get_bases_products(self, xs, basess, n_choice=None):
        hs = xs[:, 1:] - xs[:, :-1]# (B, J-1):  grid size
        # Noramalization: compute each basis node's L2 norm normalize basis nodes
        n_choice = n_choice if n_choice is not None else self.n_base
        ids = np.random.choice(self.n_base, n_choice, replace=False)  # Randomly select n_choice basis nodes
        _basess = basess[:, ids, :]  # (B, n_choice, J)
        # Create scores matrix
        # [<u1,u1>, <u1,u2>, ..., <u1,un>]
        # [<u2,u1>, <u2,u2>, ..., <u2,un>]
        # ...
        # [<un,u1>, <un,u2>, ..., <un,un>
        m_scores = torch.vmap(
                    torch.vmap(
                        torch.vmap(_inner_product,  # (J,), (J,), (J-1)
                                       in_dims=(None, 0, None)), # (J,), (n_base,J), (J-1)
                                       in_dims=(0, None, None)), # (n_base, J), (n_base, J), (J-1)
                                       in_dims=(0, 0, 0))(_basess, _basess, hs) # (B, n_base, J), (B, n_base, J), (B, J-1)
        
        return m_scores

    def get_R1_R2(self, xs, bases, n_choice=None):
        m_scores = self.get_bases_products(xs, bases, n_choice=n_choice)
        r1 = self.R1(m_scores)
        r2 = self.R2(m_scores)
        return r1, r2
    
    @staticmethod
    def R1(m_scores):
        # sample l1_k basis nodes to regularize
        norm2 = torch.diagonal(m_scores) # <u1,u1>, <u2,u2>, ..., <un,un>
        ideal_norm2 = torch.ones_like(norm2)  # Ideal norm is 1 for each basis node
        return F.mse_loss(norm2, ideal_norm2)  # Mean squared error loss between actual and ideal norms
    
    @staticmethod
    def R2(m_scores):
        cross_norm = torch.triu(m_scores, diagonal=1)
        idea_cross_norm = torch.zeros_like(cross_norm)  # Ideal cross norm is 0 for each pair
        return F.mse_loss(cross_norm, idea_cross_norm)  # Mean squared error
"""
Data Generation
"""
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


def poly_sin_func(coeffs:torch.Tensor, x:torch.Tensor.float)->torch.Tensor.float:
    """
    The function to compute the polynomial sine function based on the coefficients and sensor points.

    $$
    f(x) = \sum_{n=1}^{N} c_n \sin(n \pi x)
    $$

    coeffs: [n,]. c_1,...,c_n are the coefficients of the polynomial sine function
    x: single sensor point  
    """
    ws = torch.arange(1, len(coeffs) + 1) # [n,]
    pi_ws = 2*torch.pi * ws

    out = torch.dot(coeffs, torch.sin(pi_ws * x))

    return out 

poly_sin_func_batched = torch.vmap(poly_sin_func, in_dims=(None, 0))

def generate_data_poly_sin(high_freq:int, nx:int, n_batch:int)->torch.Tensor:
    """
    Generate data based on the polynomial sine function with random coefficients.

    high_freq: highest frequency used in the polynomial sine function
    n_x: number of sensor points
    n_batch: number of batch
    """
    xs = torch.linspace(0, 1, nx).repeat(n_batch, 1)# Random sensor points (n_batch, nx)
    coeffs = torch.randn(n_batch, high_freq)  # Random coefficients for the polynomial sine function
    us = torch.vmap(poly_sin_func_batched, in_dims=(0, 0))(coeffs, xs)  # Apply the function to each batch
    return xs, us


"""
Training
"""
def loss_fn(model, xs, us, n_choose:int=5, lambda1:float=1., lambda2:float=1.):
    # ENCODE
    scores, bases = model.encode(xs, us)  # Encode the function into scores
    # DECODE
    us_restore = model.decode(xs, scores)  # Decode the scores back to function

    loss_res = F.mse_loss(us_restore, us)  # Mean squared error loss

    if (n_choose > 0)  and  (lambda1>0 or lambda2>0):
        loss_r1, loss_r2 = model.get_R1_R2(xs, bases, n_choice=n_choose)  # Get R1 and R2 regularization losses
    else: 
        loss_r1, loss_r2 = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

    return loss_res + lambda1*loss_r1 + lambda2*loss_r2, (loss_res, loss_r1, loss_r2)
    

def train(model, nstep, optimizer, data_gen, scheduler=None):
    """
    Train the model for nstep steps.
    """
    model.train()
    xs, us = data_gen()
    xs = xs.to(device)
    us = us.to(device)
    tstep = trange(nstep, desc="Training")
    losses = []
    for step in tstep:
        # Generate new data for each step
        optimizer.zero_grad()
        # Forward pass
        us_restore = model(xs, us)
        # Compute loss (mean squared error)
        loss, (loss_res, loss_r1, loss_r2) = loss_fn(model, xs, us, lambda1=1.0, lambda2=1.0)
        # Backward pass
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        # Logging
        losses.append(loss.detach().item())
        tstep.set_description(f"loss: {loss:.2e} / loss_res: {loss_res:.2e} / loss_r1: {loss_r1:.2e} / loss_r2: {loss_r2:.2e}")
    return model, losses

"""
Evaluation
"""
def error_over_freqs(model:nn.Module, n_xs:int, n_batch:int):
    is_train = model.training
    model.eval()
    with torch.no_grad():
        error_means = []
        error_stds = []
        # Evalution on multiple frequency 
        freqs = np.array(list(range(1,n_xs//2 - 1,1))) ## Less than half of the sampling frequency
        for freq in freqs:
            data_gen_test = lambda: generate_data_poly_sin(high_freq=freq, nx=n_xs, n_batch=n_batch)
            xs_test, us_test = data_gen_test()
            us_pred = model(xs_test.to(device), us_test.to(device))

            errors = torch.abs(us_pred - us_test.to(device)).cpu()
            error_means.append(errors.mean().item())
            error_stds.append(errors.std().item())
        error_means = np.array(error_means)
        error_stds = np.array(error_stds)
    model.train() if is_train else None # Restore the mode
    return freqs, error_means, error_stds

def plot_error_over_freqs(max_freq, model:nn.Module, n_xs:int, n_batch:int):
    freqs, error_means, error_stds  = error_over_freqs(model, n_xs, n_batch)
    fig, ax = plt.subplots()
    ax.plot(freqs, error_means, marker='o', linestyle='-', color='blue')
    ax.fill_between(freqs, error_means - error_stds, error_means + error_stds, color='blue', alpha=0.2)
    ax.set_xlabel('Max Frequency Mode')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Error vs Max Frequency')
    ax.axvline(x=max_freq, color='red', linestyle='--', label='Training Frequency')
    ax.legend()
    ax.set_yscale('log')
    return fig, ax


if __name__ == "__main__":
    """
    Data
    """
    # Data Generation
    n_batch = 64
    # Weights of basis functions
    zs = torch.asarray([1, 1, 1] + [1] * 2)
    # Sensor Points
    n_xs = 100
    xs = torch.linspace(0, 1, n_xs).repeat(n_batch, 1)  # (n_batch, J)
    # Function response
    xs, Us = generate_data(zs, xs, n_batch=n_batch) # (n_batch, J)
    #data_gen = lambda: generate_data(zs, xs, n_batch=n_batch)  
    high_freq = 8
    n_xs = max(n_xs, high_freq * 2 + 1)  # Ensure n_xs is at least twice the highest frequency
    data_gen = lambda: generate_data_poly_sin(high_freq=high_freq, nx=n_xs, n_batch=n_batch)  # Generate data for the polynomial sine function
    """
    Model
    """
    n_base = 45
    # Create Function Autoencoder Model 
    model = AdaFNN(n_base=n_base, base_hidden=[256,256,256,256], fourier_feature=True).to(device)  # Change 'cpu' to 'cuda' if using GPU
    # Encode function into scores
    scores,_ = model.encode(xs.to(device), Us.to(device))
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
    nstep = 100_000
    lr=1e-3
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, 1000, gamma=0.9, last_epoch=-1)
    model_opt, losses = train(model, nstep=nstep, optimizer=opt, data_gen=data_gen, scheduler=scheduler)

    # Plotting the training loss
    fig, ax = plt.subplots()
    ax.plot(losses, label='Training Loss', color='blue')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    # Evaluation
    with torch.no_grad():
        model_opt.eval()
        fig1, ax1 = plot_error_over_freqs(high_freq, model_opt, n_xs=n_xs, n_batch=n_batch)
    
        model_opt.eval()
        for freq in [4, 16, 30]:
            data_gen2 = lambda: generate_data_poly_sin(high_freq=freq, nx=n_xs, n_batch=n_batch)
            xs, us = data_gen2()  # Generate new data for evaluation
            us_restore = model_opt(xs.to(device), us.to(device))
            scores, bases = model_opt.encode(xs.to(device), us.to(device))
            us_restore = us_restore.cpu().numpy()
            # Convert to numpy
            scores = scores.cpu().numpy()
            bases = bases.cpu().numpy()
            xs = xs.cpu().numpy()
            us = us.cpu().numpy()
            # Plotting 
            for i in [0]:
                fig, ax = plt.subplots()
                ax.plot(xs[i], us[i], label='Original Function', color='blue')
                ax.plot(xs[i], us_restore[i], label='Restored Function', color='red')
                ax.legend()
                ax.set_xlabel("x")
                ax.set_ylabel("u(x)")
                ax.set_title(f"Max frequency Mode {freq}")

                fig1, ax1 = plt.subplots()
                js = np.argsort(np.abs(scores[i]))
                ax1.plot(xs[i], bases[i, js[-1]], label='Basis Functions with highest score')
                ax1.plot(xs[i], bases[i, js[-2]], label='Basis Functions with 2nd score')
                ax1.set_xlabel("x")
                ax1.set_ylabel("Basis Function Value")
                ax1.set_title(f"Max frequency Mode {freq}")
                ax1.legend()



    

# %%
