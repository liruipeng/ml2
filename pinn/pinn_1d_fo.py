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
"""
1D PINN model to solve the problem:
            -u_xx + r*u = f
and homogeneous boundary conditions (BC).
The analytical solution is
   1. u(x) = exp(-2x^2) + 1 / 2
   2. u(x) = sum_k c_k * sin(w_k * pi * x)
and so,
   1. f = (4 - 16x^2) exp(-2x^2)
   2. f = sum_k c_k * (w_k^2 * pi^2 + r) * sin(w_k * pi * x)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys

# %% [markdown]
# torch.set_default_dtype(torch.float64)

# %%
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


# %%
class ProbParam:
    def __init__(self):
        self.problem = None
        self.w = None
        self.c = None
        self.r = None
        #
        self.dtype = torch.get_default_dtype()
        self.np_dtype = np.float32
        if self.dtype == torch.float64:
            self.np_dtype = np.float64
        elif self.dtype == torch.float32:
            self.np_dtype = np.float32


# %%
prob = ProbParam()


# %%
class PINN(nn.Module):
    def __init__(self, dim_inputs, dim_outputs, dim_hidden: list,
                 num_lev,
                 act_u: nn.Module = nn.ReLU(),
                 act_du: nn.Module = nn.ReLU()) -> None:
        """Simple neural network with linear layers and non-linear activation function
        This class is used as universal function approximate for the solution of
        partial differential equations using PINNs
        Args:
        """
        super().__init__()

        self.num_lev = num_lev
        self.dim_inputs = dim_inputs
        self.dim_outputs = dim_outputs

        layer_dim = [dim_inputs] + dim_hidden + [dim_outputs]
        self.linears_u = nn.ModuleList(
            [nn.Linear(layer_dim[i], layer_dim[i+1]) for i in range(len(layer_dim) - 1)],
        )

        self.linears_du = nn.ModuleList(
            [nn.Linear(layer_dim[i], layer_dim[i+1]) for i in range(len(layer_dim) - 1)],
        )

        self.act_u = act_u
        self.act_du = act_du

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_layers = len(self.linears_u)
        out_u = x
        out_du = x

        for i in range(num_layers):
            if i < num_layers - 1:
                out_u = self.act_u(self.linears_u[i](out_u))
                out_du = self.act_du(self.linears_du[i](out_du))
            else:
                out_u = self.linears_u[i](out_u)
                out_du = self.linears_du[i](out_du)

        out = torch.cat((out_u, out_du), dim=1)

        return out

    def get_solution(self, x: torch.Tensor) -> torch.Tensor:
        y = self.forward(x)
        # out = torch.zeros((x.shape[0], self.dim_outputs), device=device)
        out = y
        # for i in range(self.num_lev):
        #     out += y[:, i * self.dim_outputs: (i + 1) * self.dim_outputs]
        return out

    # def _init_weights(self, m):
    #    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #        nn.init.ones_(m.weight)
    #        m.bias.data.fill_(0.01)
    #    if type(m) == nn.Linear:
    #        torch.nn.init.xavier_uniform(m.weight)  #


# %%
class Mesh:
    def __init__(self, ntrain, neval, ax, bx):
        self.ntrain = ntrain
        self.neval = neval
        self.ax = ax
        self.bx = bx
        # training sample points (excluding the two points on the boundaries)
        x_train_np = np.linspace(self.ax, self.bx, self.ntrain)[:, None][1:-1].astype(prob.np_dtype)
        x_eval_np = np.linspace(self.ax, self.bx, self.neval)[:, None].astype(prob.np_dtype)
        self.x_train = torch.tensor(x_train_np, requires_grad=True).to(device)
        self.x_train_bc = torch.tensor([[ax], [bx]], requires_grad=True).to(device)
        self.x_eval = torch.tensor(x_eval_np, requires_grad=False).to(device)


# %%
# Source term
def f1(x):
    y = (4 - 16 * x ** 2) * torch.exp(-2 * x ** 2)
    return y


# %%
def f2(x):
    lw = len(prob.w)
    r = prob.r
    y = torch.zeros_like(x)
    for i in range(lw):
        w = prob.w[i]
        c = prob.c[i]
        y += c * (w * w * np.pi * np.pi + r) * torch.sin(w * np.pi * x)
    return y


# %%
def f(x):
    if prob.problem == 1:
        return f1(x)
    elif prob.problem == 2:
        return f2(x)


# %%
def u_ex1(x):
    y = torch.exp(-2 * x ** 2) + 1 / 2
    return y


# %%
# Exact solution
def u_ex2(x):
    lw = len(prob.w)
    y = torch.zeros_like(x)
    for i in range(lw):
        w = prob.w[i]
        c = prob.c[i]
        y += c * torch.sin(w * np.pi * x)
    return y


# %%
def u_ex(x):
    if prob.problem == 1:
        return u_ex1(x)
    elif prob.problem == 2:
        return u_ex2(x)


# %%
def pinn_loss_fo(model, mesh):
    x = mesh.x_train
    x_bc = mesh.x_train_bc
    z = model(x)
    z_bc = model(x_bc)
    loss_func = nn.MSELoss()

    interior2 = None
    boundary = None
    loss = 0

    for i in range(model.num_lev):
        u = z[:, 2 * i].unsqueeze(-1)
        s = z[:, 2 * i + 1].unsqueeze(-1)
        du_dx, = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)
        ds_dx, = torch.autograd.grad(s, x, grad_outputs=torch.ones_like(s), create_graph=True)
        u_bc = z_bc[:, 2 * i].unsqueeze(-1)
        # interior/boundary loss
        if i == 0:
            interior1 = du_dx + s
            interior2 = ds_dx + prob.r * u - f(x)
            boundary = u_bc - u_ex(x_bc)
        else:
            interior1 = du_dx + s
            interior2 = interior2.detach().clone() + ds_dx + prob.r * u
            boundary = boundary.detach().clone() + u_bc
        # total loss
        loss_i1 = loss_func(interior1, torch.zeros_like(interior1))
        loss_i2 = loss_func(interior2, torch.zeros_like(interior2))
        loss_b = loss_func(boundary, torch.zeros_like(boundary))
        if i == 0:
            loss = loss_i1 + loss_i2 + loss_b
        else:
            loss = loss + loss_i1 + loss_i2 + loss_b

    return loss


# %%
# Define the loss function for the PINN
class Loss:
    def __init__(self, model, mesh, t):
        self.model = model
        self.mesh = mesh
        self.type = t

    def loss(self):
        # x_train_np = np.random.uniform(self.mesh.ax, self.mesh.bx, self.mesh.ntrain)[:, None].astype(prob.np_dtype)
        # self.mesh.x_train = torch.tensor(x_train_np, requires_grad=True)

        loss = pinn_loss_fo(model=self.model, mesh=self.mesh)

        return loss


# %%
# Define the training loop
def train(model, mesh, criterion, iterations, learning_rate, num_check, num_plots):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    check_freq = (iterations + num_check - 1) // num_check
    plot_freq = (iterations + num_plots - 1) // num_plots
    fig, ax = plt.subplots(num_plots + 1)
    fig2, ax2 = plt.subplots(num_plots)

    u_analytic = u_ex(mesh.x_eval)
    ax[0].plot(mesh.x_eval.cpu().detach().numpy(), u_analytic.cpu().detach().numpy(), linestyle='-',
               color="black")
    ax[-1].plot(mesh.x_eval.cpu().detach().numpy(), u_analytic.cpu().detach().numpy(), linestyle='-',
                color="black", alpha=0.5)
    ax_i = 1

    for i in range(iterations):
        # we need to set to zero the gradients of all model parameters (PyTorch accumulates grad by default)
        optimizer.zero_grad()
        # compute the loss value for the current batch of data
        loss = criterion.loss()
        # backpropagation to compute gradients of model param respect to the loss. computes dloss/dx
        # for every parameter x which has requires_grad=True.
        loss.backward()
        # update the model param doing an optim step using the computed gradients and learning rate
        optimizer.step()

        if np.remainder(i + 1, check_freq) == 0 or i == iterations - 1:
            model.eval()
            with torch.no_grad():
                u_eval = model.get_solution(mesh.x_eval)[:, 0].unsqueeze(-1)
                error = u_analytic - u_eval
                print(f"Iteration {i:6d}/{iterations:6d}, PINN Loss: {loss.item():.4e}, "
                      f"Err 2-norm: {torch.norm(error): .4e}, "
                      f"inf-norm: {torch.max(torch.abs(error)):.4e}")
            model.train()

        if np.remainder(i + 1, plot_freq) == 0 or i == iterations - 1:
            model.eval()
            with torch.no_grad():
                u_train = model.get_solution(mesh.x_train)[:, 0].unsqueeze(-1)
                u_eval = model.get_solution(mesh.x_eval)[:, 0].unsqueeze(-1)
                error = u_analytic - u_eval
                # plot
                ax[ax_i].scatter(mesh.x_train.cpu().detach().numpy(), u_train.cpu().detach().numpy(), color="red",
                                 label="Sample training points")
                ax[ax_i].plot(mesh.x_eval.cpu().detach().numpy(), u_eval.cpu().detach().numpy(), linestyle='-',
                              marker=',', alpha=0.5)
                ax2[ax_i - 1].plot(mesh.x_eval.cpu().detach().numpy(), error.cpu().detach().numpy())
                ax_i += 1
            model.train()

    for i in range(len(ax)):
        if i < len(ax) - 1:
            ax[i].get_xaxis().set_visible(False)
        # ax[i].set_yscale('symlog')
    for i in range(len(ax2) - 1):
        if i < len(ax2) - 1:
            ax2[i].get_xaxis().set_visible(False)
        # ax2[i].set_yscale('symlog')


# %%
def main():
    torch.manual_seed(0)
    eval_resolution = 256
    # Generate training data
    # number of point for training
    nx = 128
    prob.problem = 2
    prob.w = [4]  # [1, 2, 4, 8, 16]
    prob.c = [1., 1., 1., 1., 1.]
    prob.r = 0
    num_check = 100
    num_plots = 4
    iterations = 10000
    # Domain is interval [ax, bx] along the x-axis
    ax = 0.0
    bx = 1.0
    #
    loss_type = 0
    #
    mesh = Mesh(ntrain=nx, neval=eval_resolution, ax=ax, bx=bx)
    # Create an instance of the PINN model
    model = PINN(dim_inputs=1, dim_outputs=1, dim_hidden=[32, 32], num_lev=1, act_u=nn.Tanh(),
                 act_du=nn.ReLU())
    print(model)
    model.to(device)
    # Exact solution
    # u_analytic = u_ex(mesh.x_eval)
    # Train the PINN model
    loss = Loss(mesh=mesh, model=model, t=loss_type)
    #
    train(model=model, mesh=mesh, criterion=loss, iterations=iterations, learning_rate=1e-3,
          num_check=num_check, num_plots=num_plots)

    return 0


# %%
if __name__ == "__main__":
    err = main()
    plt.show()
    sys.exit(err)
