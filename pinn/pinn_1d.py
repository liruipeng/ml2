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
import sys

# torch.set_default_dtype(torch.float64)

if torch.cuda.is_available():
   dev = "cuda:0"
else:
   dev = "cpu"
device = torch.device(dev)


class ProbParam:
    def __init__(self):
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


prob = ProbParam()


class PINN(nn.Module):
    def __init__(self, dim_inputs, dim_outputs, dim_hidden: list, num_lev,
                 act: nn.Module = nn.ReLU()) -> None:
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
        self.linears = nn.ModuleList(
            [nn.Linear(layer_dim[i], layer_dim[i+1]) for i in range(len(layer_dim) - 1)],
        )

        if self.num_lev > 1:
            self.linears2 = nn.ModuleList(
                [nn.Linear(layer_dim[i], layer_dim[i+1]) for i in range(len(layer_dim) - 1)],
            )

        if self.num_lev > 2:
            self.linears3 = nn.ModuleList(
                [nn.Linear(layer_dim[i], layer_dim[i+1]) for i in range(len(layer_dim) - 1)],
            )

        # for _ in range(num_hidden):
        #    self.middle_layers.apply(self._init_weights)
        # self.apply(init_weights)

        self.act = act  # activation function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_layers = len(self.linears)
        out = None
        out1 = x
        if self.num_lev == 1:
            for i in range(num_layers):
                if i < num_layers - 1:
                    out1 = self.act(self.linears[i](out1))
                else:
                    out1 = self.linears[i](out1)
            out = out1
        elif self.num_lev == 2:
            out2 = x
            for i in range(num_layers):
                if i < num_layers - 1:
                    out1 = self.act(self.linears[i](out1))
                    out2 = self.act(self.linears2[i](out2))
                else:
                    out1 = self.linears[i](out1)
                    out2 = self.linears2[i](out2)
            out = torch.cat((out1, out2), dim=1)
        elif self.num_lev == 3:
            out2 = x
            out3 = x
            for i in range(num_layers):
                if i < num_layers - 1:
                    out1 = self.act(self.linears[i](out1))
                    out2 = self.act(self.linears2[i](out2))
                    out3 = self.act(self.linears3[i](out3))
                else:
                    out1 = self.linears[i](out1)
                    out2 = self.linears2[i](out2)
                    out3 = self.linears3[i](out3)
            out = torch.cat((out1, out2, out3), dim=1)

        assert(out.shape[1] == self.num_lev * self.dim_outputs)
        return out

    def get_solution(self, x: torch.Tensor) -> torch.Tensor:
        y = self.forward(x)
        out = torch.zeros((x.shape[0], self.dim_outputs))
        for i in range(self.num_lev):
            out += y[:, i * self.dim_outputs: (i + 1) * self.dim_outputs].to(out.device)
        return out

    # def _init_weights(self, m):
    #    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #        nn.init.ones_(m.weight)
    #        m.bias.data.fill_(0.01)
    #    if type(m) == nn.Linear:
    #        torch.nn.init.xavier_uniform(m.weight)  #


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


# Source term
def f(x):
    lw = len(prob.w)
    r = prob.r
    y = torch.zeros_like(x)
    for i in range(lw):
        w = prob.w[i]
        c = prob.c[i]
        y += c * (w * w * np.pi * np.pi + r) * torch.sin(w * np.pi * x)
    return y


# Exact solution
def u_ex(x):
    lw = len(prob.w)
    y = torch.zeros_like(x)
    for i in range(lw):
        w = prob.w[i]
        c = prob.c[i]
        y += c * torch.sin(w * np.pi * x)
    return y


def super_loss(model, mesh):
    x = mesh.x_train
    x_bc = mesh.x_train_bc
    u = model(x)[:, 0].unsqueeze(-1)
    u_bc = model(x_bc)[:, 0].unsqueeze(-1)
    # interior loss
    interior = u - u_ex(x)
    # boundary loss
    boundary = u_bc - u_ex(x_bc)
    loss_func = nn.MSELoss()
    loss = loss_func(interior, torch.zeros_like(interior)) + loss_func(boundary, torch.zeros_like(boundary))

    return loss


def pinn_loss(model, mesh):
    x = mesh.x_train
    x_bc = mesh.x_train_bc
    z = model(x)
    z_bc = model(x_bc)
    loss_func = nn.MSELoss()

    interior = None
    boundary = None
    loss = 0

    for i in range(model.num_lev):
        u = z[:, i].unsqueeze(-1)
        du_dx, = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)
        d2u_dx2, = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True)
        u_bc = z_bc[:, i].unsqueeze(-1)
        # interior/boundary loss
        if i == 0:
            interior = d2u_dx2 - prob.r * u + f(x)
            boundary = u_bc - u_ex(x_bc)
        else:
            interior = interior.detach().clone() + d2u_dx2 - prob.r * u
            boundary = boundary.detach().clone() + u_bc
        # total loss
        loss_i = loss_func(interior, torch.zeros_like(interior))
        loss_b = loss_func(boundary, torch.zeros_like(boundary))
        loss += loss_i + loss_b

    # print(f"loss {loss:.4e}, loss i {loss_i:.4e}, loss_b {loss_b:.4e}")
    return loss


# Define the loss function for the PINN
class Loss:
    def __init__(self, model, mesh, t):
        self.model = model
        self.mesh = mesh
        self.type = t

    def loss(self):
        if self.type == -1:
            loss = super_loss(model=self.model, mesh=self.mesh)
        elif self.type == 0:
            loss = pinn_loss(model=self.model, mesh=self.mesh)
        else:
            loss = torch.nan
        return loss


# Define the training loop
def train(model, mesh, criterion, iterations, learning_rate, num_check, num_plots):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=5000, gamma=0.1)
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
        #
        scheduler.step()

        if np.remainder(i + 1, check_freq) == 0 or i == iterations - 1:
            model.eval()
            with torch.no_grad():
                u_eval = model.get_solution(mesh.x_eval)[:, 0].unsqueeze(-1)
                error = u_analytic - u_eval.to(u_analytic.device)
                print(f"Iteration {i:6d}/{iterations:6d}, PINN Loss: {loss.item():.4e}, "
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
                ax[ax_i].scatter(mesh.x_train.cpu().detach().numpy(), u_train.cpu().detach().numpy(), color="red",
                                 label="Sample training points")
                ax[ax_i].plot(mesh.x_eval.cpu().detach().numpy(), u_eval.cpu().detach().numpy(), linestyle='-',
                              marker=',', alpha=0.5)
                ax2[ax_i - 1].plot(mesh.x_eval.cpu().detach().numpy(), error.cpu().detach().numpy())
                ax_i += 1
            model.train()

    for i, axis in enumerate(ax):
        if i < len(ax) - 1:
            axis.get_xaxis().set_visible(False)
    for axis in ax2[:-1]:
        axis.get_xaxis().set_visible(False)


def main():
    torch.manual_seed(0)
    eval_resolution = 256
    # Generate training data
    # number of point for training
    nx = 128
    prob.w = [10] # [1, 2, 4, 8, 16]
    prob.c = [1] # [1., 1., 1., 1., 1.]
    prob.r = 0
    num_check = 20
    num_plots = 4
    iterations = 10000
    # Domain is interval [ax, bx] along the x-axis
    ax = 0.0
    bx = 1.0
    #
    dim_outputs = 1
    loss_type = 0
    #
    mesh = Mesh(ntrain=nx, neval=eval_resolution, ax=ax, bx=bx)
    # Create an instance of the PINN model
    model = PINN(dim_inputs=1, dim_outputs=dim_outputs, dim_hidden=[64, 64], num_lev=1, act=nn.ReLU())
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


if __name__ == "__main__":
    err = main()
    plt.show()
    sys.exit(err)

# %%
