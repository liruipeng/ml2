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
from enum import Enum

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
# Define level status
class LevelStatus(Enum):
    OFF = "off"
    TRAIN = "train"
    FROZEN = "frozen"

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

        # All levels start as "off"
        self.level_status = [LevelStatus.OFF] * num_levels

        # No gradients are tracked initially
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False

        # Scale factor
        self.scales = [1.0] * num_levels

    def set_status(self, level_idx: int, status: LevelStatus):
        assert isinstance(status, LevelStatus), f"Invalid status: {status}"
        if level_idx < 0 or level_idx >= self.num_levels():
            raise IndexError(f"Level index {level_idx} is out of range")
        self.level_status[level_idx] = status
        requires_grad = status == LevelStatus.TRAIN
        for param in self.models[level_idx].parameters():
            param.requires_grad = requires_grad

    def set_all_status(self, status_list: list[LevelStatus]):
        assert len(status_list) == len(self.models), "Length mismatch in status list"
        for i, status in enumerate(status_list):
            self.set_status(i, status)

    def print_status(self):
        for i, status in enumerate(self.level_status):
            print(f"Level {i}: {status.name}")

    def num_levels(self):
        return len(self.models)

    def num_active_levels(self) -> int:
        """Returns the number of levels currently active (train or frozen)"""
        return sum(status != LevelStatus.OFF for status in self.level_status)

    def set_scale(self, level_idx: int, scale: float):
        if level_idx < 0 or level_idx >= self.num_levels():
            raise IndexError(f"Level index {level_idx} is out of range")
        self.scales[level_idx] = scale

    def set_all_scales(self, scale_list: list[float]):
        assert len(scale_list) == len(self.models), "Length mismatch in scales"
        for i, scale in enumerate(scale_list):
            self.set_scale(i, scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ys = []
        for i, model in enumerate(self.models):
            if self.level_status[i] != LevelStatus.OFF:
                x_scale = self.scales[i] * x
                y = model.forward(x=x_scale)
                ys.append(y)
        if not ys:
            # No active levels, return zeros with correct shape
            return torch.zeros((x.shape[0], self.dim_outputs), device=x.device)
        # Concatenate along the column (feature) dimension
        out = torch.cat(ys, dim=1)
        assert(out.shape[1] == self.num_active_levels() * self.dim_outputs)
        return out

    def get_solution(self, x: torch.Tensor) -> torch.Tensor:
        y = self.forward(x)
        n_active = self.num_active_levels()
        # reshape to [batch_size, num_levels, dim_outputs]
        # and sum over levels
        if n_active > 1:
            y = y.view(-1, n_active, self.dim_outputs)
            return y.sum(dim=1)  # shape: (n, dim_outputs)
        return y

    # def _init_weights(self, m):
    #    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #        nn.init.ones_(m.weight)
    #        m.bias.data.fill_(0.01)
    #    if type(m) == nn.Linear:
    #        torch.nn.init.xavier_uniform(m.weight)  #


# %%
# Define Loss
class Loss:
    def __init__(self, loss_type, loss_func=nn.MSELoss()):
        self.loss_func = loss_func
        self.type = loss_type
        if self.type == -1:
            self.name = "Super Loss"
        elif self.type == 0:
            self.name = "PINN Loss"
        else:
            raise ValueError(f"Unknown loss type: {self.type}")

    # "supervised" loss against the analytical solution
    def super_loss(self, model, mesh, loss_func):
        x = mesh.x_train
        u = model.get_solution(x)
        loss = loss_func(u, mesh.u_ex)
        return loss

    # "PINN" loss
    def pinn_loss(self, model, mesh, loss_func):
        x = mesh.x_train.clone().detach().requires_grad_(True)
        u = model.get_solution(x)

        du_dx, = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)
        d2u_dx2, = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True)

        u_int = u[1:-1]
        u_bc = u[[0, -1]]
        u_ex_bc = mesh.u_ex[[0, -1]]

        # Losses
        pde = mesh.pde
        loss_i = loss_func(d2u_dx2[1:-1] + mesh.f[1:-1], pde.r * u_int)
        loss_b = loss_func(u_bc, u_ex_bc)
        loss = loss_i + loss_b

        # print(f"loss {loss:.4e}, loss i {loss_i:.4e}, loss_b {loss_b:.4e}")
        return loss

    def loss(self, model, mesh):
        if self.type == -1:
            loss = self.super_loss(model=model, mesh=mesh, loss_func=self.loss_func)
        elif self.type == 0:
            loss = self.pinn_loss(model=model, mesh=mesh, loss_func=self.loss_func)
        else:
            raise ValueError(f"Unknown loss type: {self.type}")
        return loss


# %%
# Define the training loop
def train(model, mesh, criterion, iterations, learning_rate, num_check, num_plots, axs1, axs2, level_idx):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=5000, gamma=0.1)
    check_freq = (iterations + num_check - 1) // num_check
    plot_freq = (iterations + num_plots - 1) // num_plots

    def to_np(t): return t.detach().cpu().numpy()

    u_analytic = mesh.pde.u_ex(mesh.x_eval)
    axs1[0, level_idx].plot(to_np(mesh.x_eval), to_np(u_analytic), linestyle='-', color="black")
    axs1[-1, level_idx].plot(to_np(mesh.x_eval), to_np(u_analytic), linestyle='-', color="black", alpha=0.5)
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
                axs1[ax_i, level_idx].scatter(to_np(mesh.x_train), to_np(u_train), color="red", label="Sample training points")
                axs1[ax_i, level_idx].plot(to_np(mesh.x_eval), to_np(u_eval), linestyle='-', marker=',', alpha=0.5)
                axs2[ax_i - 1, level_idx].plot(to_np(mesh.x_eval), to_np(error))
                ax_i += 1
            model.train()

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
    h = 32 # highest frequency
    w = [h]
    # w = list(range(1, h + 1))
    # w = list(range(2, h + 1, 2))
    # w = [2**i for i in range(h.bit_length()) if 2**i <= h]
    c = [1] * len(w)
    r = 0
    pde = PDE(w=w, c=c, r=r)
    # input and output dimension: x -> u(x)
    dim_inputs = 1
    dim_outputs = 1
    # loss function [supervised with analytical solution (-1) or PDE loss (0)]
    loss_type = 0
    loss = Loss(loss_type=loss_type)
    print(f"Using loss: {loss.name}")
    # number of levels
    num_levels = 2

    # ** RELU does NOT work! **
    act = nn.Tanh() # SiLU, Tanh, SoftPlus, GELU
    # 1-D mesh
    mesh = Mesh(ntrain=nx, neval=eval_resolution, ax=ax, bx=bx)
    mesh.set_pde(pde=pde)
    # Create an instance of multilevel model
    model = MultiLevelNN(num_levels=num_levels, dim_inputs=dim_inputs, dim_outputs=dim_outputs,
                         dim_hidden=[64, 64], act=act)
    print(model)
    model.to(device)
    # Plotting
    fig1, axs1 = plt.subplots(num_plots + 1, num_levels)
    axs1 = axs1.reshape(num_plots + 1, num_levels)
    fig2, axs2 = plt.subplots(num_plots, num_levels)
    axs2 = axs2.reshape(num_plots, num_levels)
    fig1.suptitle("Model Outputs")
    fig2.suptitle("Error Plots")
    # Train the model
    for l in range(num_levels):
        # Turn level l-1 to "frozen"
        if l > 0:
            model.set_status(level_idx=l-1, status=LevelStatus.FROZEN)
        # Turn level l to "train"
        model.set_status(level_idx=l, status=LevelStatus.TRAIN)
        print("\nTraining Level", l)
        model.print_status()
        # set scale
        if l == 0:
            model.set_scale(level_idx=l, scale=1.0)
        else:
            model.set_scale(level_idx=l, scale=32.0)
        # Crank that !@#$ up
        train(model=model, mesh=mesh, criterion=loss, iterations=iterations, learning_rate=1e-3,
              num_check=num_check, num_plots=num_plots, axs1=axs1, axs2=axs2, level_idx=l)
    # Plotting
    for axis_row in axs1[:-1]:
        for axis in axis_row:
            axis.get_xaxis().set_visible(False)
    for axis_row in axs2[:-1]:
        for axis in axis_row:
            axis.get_xaxis().set_visible(False)
    fig1.tight_layout()
    fig2.tight_layout()

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
