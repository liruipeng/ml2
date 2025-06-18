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
#   kernelspec:
#     display_name: pymfem
#     language: python
#     name: python3
# ---

# %% [markdown]
# ### 1D PDE problem:
#
# $-u_{xx} + \gamma u = f$
#
# and homogeneous boundary conditions (BC)
#
# #### Problem 1
# The analytical solution is
#
# $u(x) = \sum_k c_k  \sin(2 w_k  \pi  x)$
#
# and
#
# $f = \sum_k c_k  (4 w_k^2  \pi^2 + \gamma)  \sin(2 w_k  \pi  x)$
#
# #### Problem 2 (from [MscaleDNN](https://arxiv.org/abs/2007.11207))
# The analytical solution is
#
# $u(x) = e^{-x^2} \sin(\mu x)$
#
# and
#
# $f(x) = e^{-x^2} [(r + 4 \mu^2 x^2 - 4 x^2 + 2) \sin(\mu x^2) + (8 \mu x^2 - 2 μ) \cos(\mu x^2)]$

# %%
# Define modules and device
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from enum import Enum
from utils import parse_args, get_activation, print_args, save_frame, make_video_from_frames, is_notebook, cleanfiles, fourier_analysis
from SOAP.soap import SOAP
# torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# Define PDE
class PDE:
    def __init__(self, high=None, mu=70, r=0, problem=1, device=device):
        # omega = [high]
        omega = list(range(1, high + 1, 2))
        # omega += [i + 50 for i in omega]
        # omega = list(range(2, high + 1, 2))
        # omega = [2**i for i in range(high.bit_length()) if 2**i <= high]
        coeff = [1] * len(omega)

        self.w = torch.asarray(omega, device=device)
        self.c = torch.asarray(coeff, device=device)
        self.mu = mu
        self.r = r
        if problem == 1:
            self.f = self.f_1
            self.u_ex = self.u_ex_1
        else:
            self.f = self.f_2
            self.u_ex = self.u_ex_2

    # Source term
    @staticmethod
    def sin_series(w:float, c:float, x:float, r:float)->float:
        """
        return c  (4 w^2  \pi^2 + r)  \sin(2 w  \pi  x)$
        x: shape (nx,) or scalar 
        return: shape (nx,) or scalar
        """
        pi_w = 2*w*torch.pi
        sin_term = c * (pi_w ** 2 + r) * torch.sin(pi_w * x)
        return sin_term
    def f_1(self, x):
        """
        x: shape (nx)
        """
        y = torch.zeros_like(x)
        sin_terms = torch.func.vmap(self.sin_series, in_dims=(0,0,None,None))(self.w, self.c, x, self.r) # shape (len(w), nx)
        y = torch.sum(sin_terms, dim=0)
        return y

    def f_2(self, x):
        z = x ** 2
        a = self.r + 4 * z * (self.mu ** 2 - 1) + 2
        b = self.mu * z
        c = 8 * b - 2 * self.mu
        return torch.exp(-z) * (a * torch.sin(b) + c * torch.cos(b))

    # Analytical solution
    def u_ex_1(self, x):
        y = torch.zeros_like(x)
        for w, c in zip(self.w, self.c):
            y += c * torch.sin(2 * w * torch.pi * x)
        return y

    def u_ex_2(self, x):
        return torch.exp(-x**2) * torch.sin(self.mu * x ** 2)

# %%
# Define mesh
class Mesh:
    def __init__(self, ntrain, neval, ax, bx):
        self.ntrain = ntrain
        self.neval = neval
        self.ax = ax
        self.bx = bx
        # training sample points (excluding the two points on the boundaries)
        self.x_train = torch.linspace(self.ax, self.bx, self.ntrain + 1, device=device)[:-1].unsqueeze(-1)
        self.x_eval = torch.linspace(self.ax, self.bx, self.neval + 1, device=device)[:-1].unsqueeze(-1)
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
    def __init__(self, mesh: Mesh, num_levels: int, dim_inputs, dim_outputs, dim_hidden: list,
                 act: nn.Module = nn.ReLU(), enforce_bc: bool = False) -> None:
        super().__init__()
        self.mesh = mesh
        # currently the same model on each level
        self.models = nn.ModuleList([
            Level(dim_inputs=dim_inputs, dim_outputs=dim_outputs, dim_hidden=dim_hidden, act=act)
            for _ in range(num_levels)
            ])
        self.dim_inputs = dim_inputs
        self.dim_outputs = dim_outputs
        self.enforce_bc = enforce_bc

        # All levels start as "off"
        self.level_status = [LevelStatus.OFF] * num_levels

        # No gradients are tracked initially
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False

        # Scale factor
        self.scales = [1.0] * num_levels

    def get_status(self, level_idx: int):
        if level_idx < 0 or level_idx >= self.num_levels():
            raise IndexError(f"Level index {level_idx} is out of range")
        return self.level_status[level_idx]

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
            y = y.sum(dim=1)  # shape: (n, dim_outputs)
        #
        if self.enforce_bc:
            g0 = self.mesh.u_ex[0].item()
            g1 = self.mesh.u_ex[-1].item()
            # in domain x in [0, 1]
            y = g0 * (1 - x) + g1 * x + x * (1 - x) * y
            # y = g0 + (x-0)/(1-0)*(g1 - g0) + (1 - torch.exp(0-x)) * (1 - torch.exp(x-1)) * y
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
    def __init__(self, loss_type, loss_func=nn.MSELoss(), bc_weight=1.0):
        self.loss_func = loss_func
        self.type = loss_type
        if self.type == -1:
            self.name = "Super Loss"
        elif self.type == 0:
            self.name = "PINN Loss"
        elif self.type == 1:
            self.name = "DRM Loss"
        else:
            raise ValueError(f"Unknown loss type: {self.type}")
        self.bc_weight = bc_weight

    # "Supervised" loss against the analytical solution
    def super_loss(self, model, mesh, loss_func):
        x = mesh.x_train
        u = model.get_solution(x)
        loss = loss_func(u, mesh.u_ex)
        return loss

    # "PINN" loss
    def pinn_loss(self, model, mesh, loss_func):
        x = mesh.x_train.requires_grad_(True)
        u = model.get_solution(x)

        du_dx, = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)
        d2u_dx2, = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True)

        # Internal loss
        pde = mesh.pde
        loss = loss_func(d2u_dx2[1:-1] + mesh.f[1:-1], pde.r * u[1:-1])
        # Boundary loss
        if not model.enforce_bc:
            u_bc = u[[0, -1]]
            u_ex_bc = mesh.u_ex[[0, -1]]
            loss_b = loss_func(u_bc, u_ex_bc)
            loss += self.bc_weight * loss_b

        return loss

    def drm_loss(self, model, mesh: Mesh):
        """Deep Ritz Method loss"""
        xs = mesh.x_train.requires_grad_(True)
        u = model(xs)

        grad_u_pred = torch.autograd.grad(u, xs, 
                                        grad_outputs=torch.ones_like(u), 
                                        create_graph=True)[0]
        
        u_pred_sq = torch.sum(u**2, dim=1, keepdim=True)
        grad_u_pred_sq = torch.sum(grad_u_pred**2, dim=1, keepdim=True)

        f_val = mesh.pde.f(xs)
        fu_prod = f_val * u

        integrand_values = 0.5 * grad_u_pred_sq[1:-1] + 0.5 * mesh.pde.r * u_pred_sq[1:-1] - fu_prod[1:-1]
        loss = torch.mean(integrand_values)

        # Boundary loss
        u_bc = u[[0,-1]] 
        u_ex_bc = mesh.u_ex[[0,-1]]
        loss_b = self.loss_func(u_bc, u_ex_bc)
        loss += self.bc_weight * loss_b


        xs.requires_grad_(False)  # Disable gradient tracking for x
        return loss

    def loss(self, model, mesh):
        if self.type == -1:
            loss_value = self.super_loss(model=model, mesh=mesh, loss_func=self.loss_func)
        elif self.type == 0:
            loss_value = self.pinn_loss(model=model, mesh=mesh, loss_func=self.loss_func)
        elif self.type == 1: 
            loss_value = self.drm_loss(model=model, mesh=mesh)
        else:
            raise ValueError(f"Unknown loss type: {self.type}")
        return loss_value


# %%
# Define the training loop
def train(model, mesh, criterion, iterations, adam_iterations, learning_rate,
          num_check, num_plots, sweep_idx, level_idx, frame_dir):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = SOAP(model.parameters(), lr = 3e-3, betas=(.95, .95), weight_decay=.01,
    #                  precondition_frequency=10)
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.9)
    use_lbfgs = False

    def to_np(t): return t.detach().cpu().numpy()

    u_analytic = mesh.pde.u_ex(mesh.x_eval)
    _, uf_analytic, _, _ = fourier_analysis(to_np(mesh.x_eval), to_np(u_analytic))
    check_freq = (iterations + num_check - 1) // num_check
    plot_freq = (iterations + num_plots - 1) // num_plots if num_plots > 0 else 0

    for i in range(iterations):
        if i == adam_iterations:
            use_lbfgs = True
            optimizer = optim.LBFGS(model.parameters(), lr=learning_rate,
                                    max_iter=20, tolerance_grad=1e-7, history_size=100)

        def closure():
            optimizer.zero_grad()
            loss = criterion.loss(model=model, mesh=mesh)
            loss.backward()
            return loss

        if use_lbfgs:
            loss = optimizer.step(closure)
        else:
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

        if plot_freq > 0 and (np.remainder(i + 1, plot_freq) == 0 or i == iterations - 1):
            model.eval()
            with torch.no_grad():
                u_train = model.get_solution(mesh.x_train)[:, 0].unsqueeze(-1)
                u_eval = model.get_solution(mesh.x_eval)[:, 0].unsqueeze(-1)
                error = u_analytic - u_eval.to(u_analytic.device)
                xf_eval, uf_eval, _, _ = fourier_analysis(to_np(mesh.x_eval), to_np(u_eval))
                save_frame(x=xf_eval, t=uf_analytic, y=uf_eval, xs=None,  ys=None,
                           iteration=[sweep_idx, level_idx, i], title="Model_Frequencies", frame_dir=frame_dir)
                save_frame(x=to_np(mesh.x_eval), t=to_np(u_analytic), y=to_np(u_eval),
                           xs=to_np(mesh.x_train), ys=to_np(u_train),
                           iteration=[sweep_idx, level_idx, i], title="Model_Outputs", frame_dir=frame_dir)
                save_frame(x=to_np(mesh.x_eval), t=None, y=to_np(error), xs=None, ys=None,
                           iteration=[sweep_idx, level_idx, i], title="Model_Errors", frame_dir=frame_dir)
            model.train()

# %%
# Define the main function
def main(args=None):
    # For reproducibility
    torch.manual_seed(0)
    # Parse args
    args = parse_args(args=args)
    print_args(args=args)
    # PDE
    pde = PDE(high=args.high_freq, mu=args.mu, r=args.gamma,
              problem=args.problem_id)
    # Loss function [supervised with analytical solution (-1) or PINN loss (0)]
    loss = Loss(loss_type=args.loss_type, bc_weight=args.bc_weight)
    print(f"Using loss: {loss.name}")
    # 1-D mesh
    mesh = Mesh(ntrain=args.nx, neval=args.nx_eval, ax=args.ax, bx=args.bx)
    mesh.set_pde(pde=pde)
    # Create an instance of multilevel model
    # Input and output dimension: x -> u(x)
    dim_inputs = 1
    dim_outputs = 1
    model = MultiLevelNN(mesh=mesh,
                         num_levels=args.levels,
                         dim_inputs=dim_inputs, dim_outputs=dim_outputs,
                         dim_hidden=args.hidden_dims,
                         act=get_activation(args.activation),
                         enforce_bc=args.enforce_bc)
    print(model)
    model.to(device)
    # Plotting
    frame_dir = "./frames"
    os.makedirs(frame_dir, exist_ok=True)
    if args.clear:
        cleanfiles(frame_dir)
    num_plots = args.num_plots if args.plot else 0
    # Train the model
    for i in range(args.sweeps):
        print("\nTraining Sweep", i)
        # train each level at a time
        for l in range(args.levels):
            # Turn all levels to "frozen" if they are not off
            for k in range(args.levels):
                if model.get_status(level_idx=k) != LevelStatus.OFF:
                    model.set_status(level_idx=k, status=LevelStatus.FROZEN)
            # Turn level l to "train"
            model.set_status(level_idx=l, status=LevelStatus.TRAIN)
            print("\nTraining Level", l)
            model.print_status()
            # set scale
            scale = l + 1
            model.set_scale(level_idx=l, scale=scale)
            # Crank that !@#$ up
            train(model=model, mesh=mesh, criterion=loss, iterations=args.epochs,
                  adam_iterations=args.adam_epochs,
                  learning_rate=args.lr, num_check=args.num_checks, num_plots=num_plots,
                  sweep_idx=i, level_idx=l, frame_dir=frame_dir)
    # Turn PNGs into a video using OpenCV
    if args.plot:
        make_video_from_frames(frame_dir=frame_dir, name_prefix="Model_Outputs",
                               output_file="Solution.mp4")
        make_video_from_frames(frame_dir=frame_dir, name_prefix="Model_Errors",
                               output_file="Errors.mp4")
        make_video_from_frames(frame_dir=frame_dir, name_prefix="Model_Frequencies",
                               output_file="Frequencies.mp4")
    return 0

# %%
# can run it like normal: python filename.py
if __name__ == "__main__":
    if is_notebook():
        err = main(['--levels', '4', '--epochs', '10000', '--sweeps', '1', '--plot'])
    else:
        err = main()
    try:
        import sys
        sys.exit(err)
    except SystemExit:
        pass  # Prevent traceback in Jupyter or VS Code
