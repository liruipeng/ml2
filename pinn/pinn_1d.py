# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
# ---

# %%
# jupytext --set-formats ipynb,py pinn_1d.py
# jupytext --sync pinn_1d.py

# %%
# Define modules and device
"""
1D PINN model to solve the problem:
            -u_xx + r*u = f
and homogeneous boundary conditions (BC).
The analytical solution is
   u(x) = sum_k c_k * sin(w_k * pi * x)
and
   f = sum_k c_k * (w_k^2 * pi^2 + r) * sin(w_k * pi * x).
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt

# torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
# Define PDE
class PDE:
    def __init__(self, w=None, c=None, r=0):
        self.w = w if w is not None else [1]
        self.c = c if c is not None else [1]
        self.r = r

    # Source term
    def f(self, x):
        y = torch.zeros_like(x)
        for w, c in zip(self.w, self.c):
            pi_w = torch.pi * w
            y += c * (pi_w ** 2 + self.r) * torch.sin(pi_w * x)
        return y

    # Analytical solution
    def u_ex(self, x):
        y = torch.zeros_like(x)
        for w, c in zip(self.w, self.c):
            y += c * torch.sin(w * torch.pi * x)
        return y

# %%
# Define mesh
class Mesh:
    def __init__(self, ntrain, neval, ax, bx):
        self.ntrain = ntrain
        self.neval = neval
        self.ax = ax
        self.bx = bx
        # training sample points (excluding the two points on the boundaries)
        self.x_train = torch.linspace(self.ax, self.bx, self.ntrain, device=device).unsqueeze(-1)
        self.x_eval = torch.linspace(self.ax, self.bx, self.neval, device=device).unsqueeze(-1)
        self.pde = None
        self.f = None
        self.u_ex = None

    def set_pde(self, pde: PDE):
        self.pde = pde
        # source term
        self.f = pde.f(self.x_train)
        # analytical solution
        self.u_ex = pde.u_ex(self.x_train)

# %%
# Define one level NN
class Level(nn.Module):
    def __init__(self, dim_inputs, dim_outputs, dim_hidden: list,
                 act: nn.Module = nn.Tanh()) -> None:
        """Simple neural network with linear layers and non-linear activation function
        This class is used as universal function approximate for the solution of
        partial differential equations using PINNs
        """
        super().__init__()
        self.dim_inputs = dim_inputs
        self.dim_outputs = dim_outputs
        # multi-layer MLP
        layer_dim = [dim_inputs] + dim_hidden + [dim_outputs]
        self.linear = nn.ModuleList([nn.Linear(layer_dim[i], layer_dim[i + 1])
                                     for i in range(len(layer_dim) - 1)])
        # activation function
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.linear):
            x = layer(x)
            # not applying nonlinear activation in the last layer
            if i < len(self.linear) - 1:
                x = self.act(x)
        return x

# %%
# Define multilevel NN
class MultiLevelNN(nn.Module):
    def __init__(self, num_levels: int, dim_inputs, dim_outputs, dim_hidden: list,
                 act: nn.Module = nn.ReLU()) -> None:
        super().__init__()
        # currently the same model on each level
        self.models = nn.ModuleList([
            Level(dim_inputs=dim_inputs, dim_outputs=dim_outputs, dim_hidden=dim_hidden, act=act)
            for _ in range(num_levels)
            ])
        self.dim_inputs = dim_inputs
        self.dim_outputs = dim_outputs

    def num_levels(self):
        return len(self.models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ys = []
        for _, lvl in enumerate(self.models):
            y = lvl.forward(x=x)
            ys.append(y)
        # Concatenate along the column (feature) dimension
        out = torch.cat(ys, dim=1)
        assert(out.shape[1] == self.num_levels() * self.dim_outputs)
        return out

    def get_solution(self, x: torch.Tensor) -> torch.Tensor:
        y = self.forward(x)
        y = y.view(-1, self.num_levels(), self.dim_outputs)
        out = y.sum(dim=1)  # shape: (n, dim_outputs)
        return out

    def freeze_levels(self, freeze_mask: list[bool]):
        for i, m in enumerate(self.models):
            for param in m.parameters():
                param.requires_grad = not freeze_mask[i]

    # def _init_weights(self, m):
    #    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #        nn.init.ones_(m.weight)
    #        m.bias.data.fill_(0.01)
    #    if type(m) == nn.Linear:
    #        torch.nn.init.xavier_uniform(m.weight)  #


# %%
# Define Loss
class Loss:
    def __init__(self, loss_type):
        self.type = loss_type
        if self.type == -1:
            self.name = "Super Loss"
        elif self.type == 0:
            self.name = "PINN Loss"
        else:
            raise ValueError(f"Unknown loss type: {self.type}")

    # "supervised" loss against the analytical solution
    def super_loss(self, model, mesh):
        x = mesh.x_train
        u = model(x)[:, 0].unsqueeze(-1)
        loss_func = nn.MSELoss()
        loss = loss_func(u, mesh.u_ex)
        return loss

    # "PINN" loss
    def pinn_loss(self, model, mesh):
        x = mesh.x_train
        x_int = x[1:-1].clone().detach().requires_grad_(True)
        x_bc = torch.vstack([x[0], x[-1]])

        u = model(x_int)[:, 0].unsqueeze(-1)
        du_dx, = torch.autograd.grad(u, x_int, grad_outputs=torch.ones_like(u), create_graph=True)
        d2u_dx2, = torch.autograd.grad(du_dx, x_int, grad_outputs=torch.ones_like(du_dx), create_graph=True)

        u_bc = model(x_bc)
        u_ex_bc = torch.vstack([mesh.u_ex[0], mesh.u_ex[-1]])

        loss_func = nn.MSELoss()
        pde = mesh.pde
        loss_i = loss_func(d2u_dx2 + mesh.f[1:-1], pde.r * u)
        loss_b = loss_func(u_bc, u_ex_bc)
        loss = loss_i + loss_b

        # print(f"loss {loss:.4e}, loss i {loss_i:.4e}, loss_b {loss_b:.4e}")
        return loss

    def loss(self, model, mesh):
        if self.type == -1:
            loss = self.super_loss(model=model, mesh=mesh)
        elif self.type == 0:
            loss = self.pinn_loss(model=model, mesh=mesh)
        else:
            raise ValueError(f"Unknown loss type: {self.type}")
        return loss


# %%
# Define the training loop
def train(model, mesh, criterion, iterations, learning_rate, num_check, num_plots):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=5000, gamma=0.1)
    check_freq = (iterations + num_check - 1) // num_check
    plot_freq = (iterations + num_plots - 1) // num_plots
    fig, ax = plt.subplots(num_plots + 1)
    fig2, ax2 = plt.subplots(num_plots)
    def to_np(t): return t.detach().cpu().numpy()

    u_analytic = mesh.pde.u_ex(mesh.x_eval)
    ax[0].plot(to_np(mesh.x_eval), to_np(u_analytic), linestyle='-', color="black")
    ax[-1].plot(to_np(mesh.x_eval), to_np(u_analytic), linestyle='-', color="black", alpha=0.5)
    ax_i = 1

    for i in range(iterations):
        # we need to set to zero the gradients of all model parameters (PyTorch accumulates grad by default)
        optimizer.zero_grad()
        # compute the loss value for the current batch of data
        loss = criterion.loss(model=model, mesh=mesh)
        # backpropagation to compute gradients of model param respect to the loss. computes dloss/dx
        # for every parameter x which has requires_grad=True.
        loss.backward()
        # update the model param doing an optim step using the computed gradients and learning rate
        optimizer.step()
        #
        scheduler.step()

        if np.remainder(i + 1, check_freq) == 0 or i == iterations - 1:
            model.eval()
            with torch.no_grad():
                u_eval = model.get_solution(mesh.x_eval)[:, 0].unsqueeze(-1)
                error = u_analytic - u_eval.to(u_analytic.device)
                print(f"Iteration {i:6d}/{iterations:6d}, {criterion.name}: {loss.item():.4e}, "
                      f"Err 2-norm: {torch.norm(error): .4e}, "
                      f"inf-norm: {torch.max(torch.abs(error)):.4e}")
            model.train()

        if np.remainder(i + 1, plot_freq) == 0 or i == iterations - 1:
            model.eval()
            with torch.no_grad():
                u_train = model.get_solution(mesh.x_train)[:, 0].unsqueeze(-1)
                u_eval = model.get_solution(mesh.x_eval)[:, 0].unsqueeze(-1)
                error = u_analytic - u_eval.to(u_analytic.device)
                # plot
                ax[ax_i].scatter(to_np(mesh.x_train), to_np(u_train), color="red", label="Sample training points")
                ax[ax_i].plot(to_np(mesh.x_eval), to_np(u_eval), linestyle='-', marker=',', alpha=0.5)
                ax2[ax_i - 1].plot(to_np(mesh.x_eval), to_np(error))
                ax_i += 1
            model.train()

    for axis in ax[:-1]:
        axis.get_xaxis().set_visible(False)
    for axis in ax2[:-1]:
        axis.get_xaxis().set_visible(False)

# %%
# Define the main function
def main():
    torch.manual_seed(0)
    eval_resolution = 256
    # Generate training data
    # number of point for training
    nx = 128
    num_check = 20
    num_plots = 4
    iterations = 10000
    # Domain is interval [ax, bx] along the x-axis
    ax = 0.0
    bx = 1.0
    # PDE coeff
    h = 16 # highest frequency
    # w = list(range(1, h + 1))
    w = list(range(2, h + 1, 2))
    c = [1] * len(w)
    r = 0
    pde = PDE(w=w, c=c, r=r)
    #
    dim_inputs = 1
    dim_outputs = 1
    loss_type = 0
    # ** RELU does NOT work! **
    act = nn.Tanh() # SiLU, Tanh, SoftPlus, GELU
    # 1-D mesh
    mesh = Mesh(ntrain=nx, neval=eval_resolution, ax=ax, bx=bx)
    mesh.set_pde(pde=pde)
    # Create an instance of multilevel model
    model = MultiLevelNN(num_levels=1, dim_inputs=dim_inputs, dim_outputs=dim_outputs, dim_hidden=[64, 64], act=act)
    print(model)
    model.to(device)
    # Train the PINN model
    loss = Loss(loss_type=loss_type)
    print(f"Using loss: {loss.name}")
    #
    train(model=model, mesh=mesh, criterion=loss, iterations=iterations, learning_rate=1e-3,
          num_check=num_check, num_plots=num_plots)

    return 0

# %%
# can run it like normal: python filename.py
if __name__ == "__main__":
    err = main()
    plt.show()
    try:
        import sys
        sys.exit(err)
    except SystemExit:
        pass  # Prevent traceback in Jupyter or VS Code
