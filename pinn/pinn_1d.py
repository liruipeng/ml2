"""
1D PINN model to solve the problem:
            -u_xx + r*u = f
and homogeneous boundary conditions (BC).
The analytical solution is
   u(x) = sum_k c_k * sin(w_k * pi * x)
and
   f = sum_k c_k * (w_k^2 * pi^2 + r) * sin(w_k * pi * x).
"""
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch.nn.functional as F

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


class MLPINN(nn.Module):
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
            out += y[:, i * self.dim_outputs: (i + 1) * self.dim_outputs]
        return out

    # def _init_weights(self, m):
    #    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #        nn.init.ones_(m.weight)
    #        m.bias.data.fill_(0.01)
    #    if type(m) == nn.Linear:
    #        torch.nn.init.xavier_uniform(m.weight)  #

class PINN(nn.Module):
    def __init__(self, dim_inputs, dim_outputs, dim_hidden: list, grid,
                 act: nn.Module = nn.ReLU()) -> None:
        """Simple neural network with linear layers and non-linear activation function
        This class is used as universal function approximate for the solution of
        partial differential equations using PINNs
        Args:
        """
        super().__init__()

        self.dim_inputs = dim_inputs
        self.dim_outputs = dim_outputs
        self.ntrain = grid.ntrain
        self.neval = grid.neval

        layer_dim = [dim_inputs] + dim_hidden + [dim_outputs]
        self.linears = nn.ModuleList(
            [nn.Linear(layer_dim[i], layer_dim[i+1]) for i in range(len(layer_dim) - 1)],
        )

        self.act = act  # activation function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_layers = len(self.linears)
        out = x

        for i in range(num_layers):
                if i < num_layers - 1:
                    out = self.act(self.linears[i](out))
                else:
                    out = self.linears[i](out)
        return out


                
class Mesh:
    def __init__(self, ntrain, neval, ax, bx):
        self.ntrain = ntrain
        self.neval = neval
        self.ax = ax
        self.bx = bx

        x_err_tr_np = np.linspace(self.ax, self.bx, (self.ntrain*2-1))[1:-1].astype(prob.np_dtype)
        x_err_train = torch.tensor(x_err_tr_np, requires_grad=True).to(device)
        self.x_err_train = x_err_train.unsqueeze(1)

        x_tr_np = np.linspace(self.ax, self.bx, self.ntrain)[1:-1].astype(prob.np_dtype)
        x_train = torch.tensor(x_tr_np, requires_grad=True).to(device)
        self.x_train = x_train.unsqueeze(1)

        x_eval_np = np.linspace(self.ax, self.bx, self.neval)[:, None].astype(prob.np_dtype)
        self.x_eval = torch.tensor(x_eval_np, requires_grad=False).to(device)

        self.x_train_bc = torch.tensor([[ax], [bx]], requires_grad=True).to(device)
       
class Solution:
    def __init__(self, models, mesh):
        self.x = mesh.x_train
        self.x_bc = mesh.x_train_bc
        self.u = models[0](self.x)
        self.u_bc = models[0](self.x_bc)
        self.load = f(self.x)
        self.load_bc = u_ex(self.x_bc)
        self.err = torch.zeros_like(mesh.x_err_train, requires_grad=False)
    def update(self, models, mesh, loss, flag):
        if flag: # error mesh
            self.x = mesh.x_err_train
            self.u = models[1](self.x)
            self.u_bc = models[1](self.x_bc)
            tar_size = mesh.ntrain*2-3
            r_loss1 = loss[0:-2].view(1,1,-1)
            load_interp = F.interpolate(r_loss1, size=tar_size, mode='linear', align_corners= False)
            self.load = load_interp.view(tar_size,-1)
            self.load_bc = loss[-2:]
            self.err = (self.u).detach().clone()
        else:
            self.x = mesh.x_train
            r = torch.arange(mesh.ntrain-2)
            c = torch.arange(1,(mesh.ntrain*2-3), step =2)
            d = torch.zeros(mesh.ntrain-2,(mesh.ntrain*2-3))
            d[r,c] = 1
            self.u = torch.mm(d,self.err) + models[0](self.x)
            self.u_bc = models[0](self.x_bc)
            self.load = f(self.x)
            self.load_bc = u_ex(self.x_bc)

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


def get_solution(models, x: torch.Tensor, n_mod) -> torch.Tensor:
    u = models[0].forward(x)
    if n_mod > 1:
        e = models[1].forward(x)
        out = u + e
    else:
        out = u
    return out


def fake_loss(model, mesh):
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
            # NOTE this is different from the original version on git - stil this loss does not work
            interior1 = du_dx + s #interior1.detach().clone()
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


def mpinn_loss(sol):
    x = sol.x
    u = sol.u
    u_bc = sol.u_bc

    interior = None
    boundary = None
    du_dx, = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)
    d2u_dx2, = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True)
    # interior/boundary loss
    interior = d2u_dx2 - prob.r * u + sol.load
    boundary = u_bc - sol.load_bc
    loss = torch.cat((interior,boundary), dim=0)
    
    return loss


# Define the loss function for the PINN
class Loss:
    def __init__(self, mesh, t):
        self.mesh = mesh
        self.type = t
        self.fler = False

    def loss(self, model, sol):
        if self.type == -1:
            loss = fake_loss(model=model, mesh=self.mesh)
        elif self.type == 0:
            loss = 0
            loss = mpinn_loss(sol)
        elif self.type == 1:
            loss = pinn_loss_fo(model=model, mesh=self.mesh)
        else:
            loss = torch.nan
        return loss


# Define the training loop
def train(models, mesh, criterion, sol, iterations, learning_rate, num_check, num_plots):
    # Plot setting 
    check_freq = (iterations + num_check - 1) // num_check
    plot_freq = (iterations + num_plots - 1) // num_plots
    fig, ax = plt.subplots(num_plots + 1)
    plt.subplots_adjust(hspace = 0.5)
    fig2, ax2 = plt.subplots(num_plots)
    plt.subplots_adjust(hspace = 0.5)

    u_analytic = u_ex(mesh.x_eval)
    ax[0].plot(mesh.x_eval.cpu().detach().numpy(), u_analytic.cpu().detach().numpy(), linestyle='-',
               color="black")
    ax[0].set_title("Ground truth", fontsize = 10)
    ax[-1].plot(mesh.x_eval.cpu().detach().numpy(), u_analytic.cpu().detach().numpy(), linestyle='-',
                color="black", alpha=0.5)
    ax_i = 1

    n_mod = len(models)
    k = 0
    loss_func = nn.MSELoss()
    optimizers = [optim.Adam(model.parameters(), lr=learning_rate) for model in models]
    loss1 = torch.tensor([0])
    for i in range(iterations):
        if n_mod > 1 and np.remainder(i, 10) == 0 and i > 0:
            criterion.fler = not criterion.fler
            k = k ^ 1  # changes 0 to 1 and viceversa
        sol.update(models, mesh, loss1.detach().clone(), criterion.fler)

        optimizers[k].zero_grad()
        # change the flag so that you switch model on the loss. Maybe not needed
        loss1 = criterion.loss(models[k], sol)
        if n_mod > 1:
            loss_i = loss_func(loss1[0:-2], torch.zeros_like(loss1[0:-2]))
            loss_b = loss_func(loss1[-2:], torch.zeros_like(loss1[-2:]))
            loss = loss_b + loss_i
            # print(f"loss {loss:.4e}, loss i {loss_i:.4e}, loss_b {loss_b:.4e}")
        else:
            loss = loss1
        # backpropagation to compute gradients of model param respect to the loss. computes dloss/dx
        # for every parameter x which has requires_grad = True.
        loss.backward()
        # update the model param doing an optim step using the computed gradients and learning rate
        optimizers[k].step()

        if np.remainder(i + 1, check_freq) == 0 or i == iterations - 1:
            for model in models:
                model.eval()
            with torch.no_grad():
                u_eval = get_solution(models, mesh.x_eval, n_mod)[:,0].unsqueeze(-1)
                error = u_analytic - u_eval
                print(f"Iteration {i:6d}/{iterations:6d}, PINN Loss: {loss.item():.4e}, "
                      f"Err 2-norm: {torch.norm(error): .4e}, "
                      f"inf-norm: {torch.max(torch.abs(error)):.4e}")
            for model in models:
                model.train()

        if np.remainder(i + 1, plot_freq) == 0 or i == iterations - 1:
            for model in models:
                model.eval()
            with torch.no_grad():
                u_train = get_solution(models, mesh.x_train, n_mod)[:,0].unsqueeze(-1)
                u_eval = get_solution(models, mesh.x_eval, n_mod)[:,0].unsqueeze(-1)
                error = u_analytic - u_eval
                # plot
                ax[ax_i].scatter(mesh.x_train.cpu().detach().numpy(), u_train.cpu().detach().numpy(), color="red",
                                 label="Sample training points")
                ax[ax_i].plot(mesh.x_eval.cpu().detach().numpy(), u_eval.cpu().detach().numpy(), linestyle='-',
                              marker=',', alpha=0.5)
                ax[ax_i].set_title(f"Iteration {i:6d}/{iterations:6d}", fontsize = 10)
                ax2[ax_i - 1].plot(mesh.x_eval.cpu().detach().numpy(), error.cpu().detach().numpy())
                ax_i += 1
            for model in models:
                model.train()

    for i in range(len(ax)):
        if i < len(ax) - 1:
            ax[i].get_xaxis().set_visible(False)
    fig.savefig("output1.png")
    for i in range(len(ax2) - 1):
        if i < len(ax2) - 1:
            ax2[i].get_xaxis().set_visible(False)
    fig2.savefig("output2.png")


def main():
    torch.manual_seed(0)
    eval_resolution = 256
    # Generate training data
    # number of point for training
    nx = 15
    prob.w = [4] #[1, 2, 4, 8, 16]
    prob.c = [1.]# [1., 1., 1., 1., 1.]
    prob.r = 0
    num_check = 10
    num_plots = 4
    iterations = 15000
    # Domain is interval [ax, bx] along the x-axis
    ax = 0.0
    bx = 1.0
    #
    dim_outputs = 1
    loss_type = 0
    mesh = Mesh(ntrain=nx, neval=eval_resolution, ax=ax, bx=bx)
    # Create an instance of the PINN model
    model1 = PINN(dim_inputs=1, dim_outputs=dim_outputs, dim_hidden=[32, 32], grid=mesh, act=nn.Tanh())
    model2 = PINN(dim_inputs=1, dim_outputs=dim_outputs, dim_hidden=[32, 32], grid=mesh, act=nn.Tanh())

    models = [model1, model2]
    print(models)
    for model in models:
        model.to(device)
    # Train the PINN model
    loss = Loss(mesh=mesh, t=loss_type)
    
    pb_sol = Solution(models, mesh)
    train(models=models, mesh=mesh, criterion=loss, sol=pb_sol, iterations=iterations, learning_rate=1e-2,
          num_check=num_check, num_plots=num_plots)

    return 0


if __name__ == "__main__":
    err = main()
    plt.show()
    sys.exit(err)

# to see if the error is oscillarory or not we can maybe derive the 
# same measure that we have for linear systems, lambda = e^tAe/e^te, 
# and change the matrix vect multiplications with the actual operator, 
# so int(e,L(e))/int(e^2). If the lambda max is high it means the error is very oscillatory.
