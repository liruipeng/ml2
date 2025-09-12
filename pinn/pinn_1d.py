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
import numpy as np
import itertools
from enum import Enum
from typing import Union, Tuple, Callable
from utils import parse_args, get_activation, print_args, save_frame, make_video_from_frames
from utils import is_notebook, cleanfiles, fourier_analysis, get_scheduler_generator, scheduler_step
from cheby import chebyshev_transformed_features, chebyshev_transformed_features2 # noqa F401
# from SOAP.soap import SOAP
# torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
# Helper functions from the new BC implementation
def _calculate_laplacian_1d(func: Callable[[torch.Tensor], torch.Tensor], x_val: float) -> torch.Tensor:
    x_tensor = torch.tensor([[x_val]], dtype=torch.float32, requires_grad=True)
    u = func(x_tensor)
    grad_u = torch.autograd.grad(u, x_tensor, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    laplacian_u = torch.autograd.grad(grad_u, x_tensor, grad_outputs=torch.ones_like(grad_u), create_graph=False, retain_graph=False)[0]
    return laplacian_u

def get_g0_func(
    u_exact_func: Callable[[torch.Tensor], torch.Tensor],
    domain_dim: int,
    domain_bounds: Union[Tuple[float, float], Tuple[Tuple[float, float], ...]],
    g0_type: str = "multilinear"
) -> Callable[[torch.Tensor], torch.Tensor]:
    domain_bounds_tuple = domain_bounds
    if domain_dim == 1 and not isinstance(domain_bounds[0], (tuple, list)):
        domain_bounds_tuple = (domain_bounds,)
    min_bounds = torch.tensor([b[0] for b in domain_bounds_tuple], dtype=torch.float32)
    max_bounds = torch.tensor([b[1] for b in domain_bounds_tuple], dtype=torch.float32)

    if g0_type == "hermite_cubic_2nd_deriv":
        if domain_dim != 1: raise ValueError("Hermite cubic interpolation with 2nd derivatives is only supported for 1D problems.")
        x0, x1 = min_bounds.item(), max_bounds.item()
        h = x1 - x0
        u_x0 = u_exact_func(torch.tensor([[x0]], dtype=torch.float32)).item()
        u_x1 = u_exact_func(torch.tensor([[x1]], dtype=torch.float32)).item()
        u_prime_prime_x0 = _calculate_laplacian_1d(u_exact_func, x0).item()
        u_prime_prime_x1 = _calculate_laplacian_1d(u_exact_func, x1).item()
        a3 = (u_prime_prime_x1 - u_prime_prime_x0) / (6 * h)
        a2 = u_prime_prime_x0 / 2 - 3 * a3 * x0
        a1 = (u_x1 - u_x0) / h - a2 * (x1 + x0) - a3 * (x1**2 + x1 * x0 + x0**2)
        a0 = u_x0 - a1 * x0 - a2 * x0**2 - a3 * x0**3
        coeffs = torch.tensor([a0, a1, a2, a3], dtype=torch.float32)

        def g0_hermite_cubic_val(x: torch.Tensor) -> torch.Tensor:
            x_flat = x[:, 0]
            g0_vals = coeffs[0] + coeffs[1] * x_flat + coeffs[2] * (x_flat**2) + coeffs[3] * (x_flat**3)
            return g0_vals.unsqueeze(1)
        return g0_hermite_cubic_val

    elif g0_type == "multilinear":
        boundary_values = {}
        dim_ranges = [[min_bounds[d].item(), max_bounds[d].item()] for d in range(domain_dim)]
        for corner_coords in itertools.product(*dim_ranges):
            corner_coords_tensor = torch.tensor(corner_coords, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                boundary_values[corner_coords] = u_exact_func(corner_coords_tensor).item()

        def g0_multilinear_val(x: torch.Tensor) -> torch.Tensor:
            num_points = x.shape[0]
            xi = (x - min_bounds.to(x.device)) / (max_bounds.to(x.device) - min_bounds.to(x.device))
            xi = torch.clamp(xi, 0.0, 1.0)
            g0_vals = torch.zeros((num_points, 1), device=x.device)
            for corner_label in itertools.product([0, 1], repeat=domain_dim):
                current_corner_key_list = []
                weight_factors = torch.ones((num_points, 1), device=x.device)
                for d in range(domain_dim):
                    if corner_label[d] == 0:
                        current_corner_key_list.append(min_bounds[d].item())
                        weight_factors *= (1 - xi[:, d]).unsqueeze(1)
                    else:
                        current_corner_key_list.append(max_bounds[d].item())
                        weight_factors *= xi[:, d].unsqueeze(1)
                corner_key_tuple = tuple(current_corner_key_list)
                corner_value = boundary_values[corner_key_tuple]
                g0_vals += corner_value * weight_factors
            return g0_vals
        return g0_multilinear_val

    else:
        raise ValueError(f"Unknown g0_type: {g0_type}. Choose 'multilinear' or 'hermite_cubic_2nd_deriv'.")

def _psi_tensor(t: torch.Tensor) -> torch.Tensor:
    return torch.where(t <= 0, torch.tensor(0.0, dtype=t.dtype, device=t.device), torch.exp(-1.0 / t))

def get_d_func(domain_dim: int, domain_bounds: Union[Tuple[float, float], Tuple[Tuple[float, float], ...]], 
              d_type: str = "quadratic_bubble") -> Callable[[torch.Tensor], torch.Tensor]:
    domain_bounds_tuple = domain_bounds
    if domain_dim == 1 and not isinstance(domain_bounds[0], (tuple, list)):
        domain_bounds_tuple = (domain_bounds,)
    min_bounds = torch.tensor([b[0] for b in domain_bounds_tuple], dtype=torch.float32)
    max_bounds = torch.tensor([b[1] for b in domain_bounds_tuple], dtype=torch.float32)
    domain_length = (max_bounds[0] - min_bounds[0]).item() if domain_dim == 1 else None

    if d_type == "quadratic_bubble":
        def d_func_val(x: torch.Tensor) -> torch.Tensor:
            d_vals = torch.ones_like(x[:, 0], dtype=torch.float32, device=x.device)
            for i in range(domain_dim):
                x_i = x[:, i]
                min_val, max_val = domain_bounds_tuple[i]
                d_vals *= (x_i - min_val) * (max_val - x_i)
            return d_vals.unsqueeze(1)
        return d_func_val

    elif d_type == "inf_smooth_bump":
        def d_inf_smooth_bump_val(x: torch.Tensor) -> torch.Tensor:
            product_terms = torch.ones((x.shape[0],), dtype=x.dtype, device=x.device)
            for i in range(domain_dim):
                x_i = x[:, i]
                min_val_i = min_bounds[i]
                max_val_i = max_bounds[i]
                x_c_i = (min_val_i + max_val_i) / 2.0
                R_i = (max_val_i - min_val_i) / 2.0
                R_i_squared = R_i**2
                arg_for_psi = R_i_squared - (x_i - x_c_i)**2
                product_terms *= _psi_tensor(arg_for_psi)
            return product_terms.unsqueeze(1)
        return d_inf_smooth_bump_val

    elif d_type == "abs_dist_complement":
        if domain_dim != 1: raise ValueError(f"d_type '{d_type}' is only supported for 1D problems.")
        def d_abs_dist_complement_val(x: torch.Tensor) -> torch.Tensor:
            x_val = x[:, 0]
            x_norm = (x_val - min_bounds[0]) / domain_length
            sqrt_term = torch.sqrt(x_norm**2 + (1.0 - x_norm)**2)
            return (1.0 - sqrt_term).unsqueeze(1)
        return d_abs_dist_complement_val

    elif d_type == "ratio_bubble_dist":
        if domain_dim != 1: raise ValueError(f"d_type '{d_type}' is only supported for 1D problems.")
        def d_ratio_bubble_dist_val(x: torch.Tensor) -> torch.Tensor:
            x_val = x[:, 0]
            x_norm = (x_val - min_bounds[0]) / domain_length
            numerator = x_norm * (1.0 - x_norm)
            denominator = torch.sqrt(x_norm**2 + (1.0 - x_norm)**2)
            return (numerator / denominator).unsqueeze(1)
        return d_ratio_bubble_dist_val

    elif d_type == "sin_half_period":
        if domain_dim != 1: raise ValueError(f"d_type '{d_type}' is only supported for 1D problems.")
        if domain_length is None: raise ValueError("Domain length must be defined for 'sin_half_period' d_type.")
        def d_sin_half_period_val(x: torch.Tensor) -> torch.Tensor:
            x_val = x[:, 0]
            argument = (torch.pi / domain_length) * (x_val - min_bounds[0])
            return torch.sin(argument).unsqueeze(1)
        return d_sin_half_period_val

    else:
        raise ValueError(f"Unknown d_type: {d_type}. Choose from 'quadratic_bubble', 'inf_smooth_bump', 'abs_dist_complement', 'ratio_bubble_dist', or 'sin_half_period'.")

# %%
# Define PDE
class PDE:
    def __init__(self, high=None, mu=70, r=0, problem=1):
        # omega = [high]
        omega = list(range(1, high + 1, 2))
        # omega += [i + 50 for i in omega]
        # omega = list(range(2, high + 1, 2))
        # omega = [2**i for i in range(high.bit_length()) if 2**i <= high]
        coeff = [1] * len(omega)

        self.w = omega
        self.c = coeff
        self.mu = mu
        self.r = r
        if problem == 1:
            self.f = self.f_1
            self.u_ex = self.u_ex_1
        else:
            self.f = self.f_2
            self.u_ex = self.u_ex_2

    # Source term
    def f_1(self, x):
        y = torch.zeros_like(x)
        for w, c in zip(self.w, self.c):
            pi_w = 2 * torch.pi * w
            y += c * (pi_w ** 2 + self.r) * torch.sin(pi_w * x)
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
        # source term
        self.f = None
        # analytical solution
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
                 act: nn.Module = nn.Tanh(),
                 use_chebyshev_basis: bool = False,
                 chebyshev_freq_min: int = 0,
                 chebyshev_freq_max: int = 0) -> None:
        """Simple neural network with linear layers and non-linear activation function
        This class is used as universal function approximate for the solution of
        partial differential equations using PINNs
        """
        super().__init__()
        self.dim_inputs = dim_inputs
        self.dim_outputs = dim_outputs
        self.use_chebyshev_basis = use_chebyshev_basis
        self.chebyshev_freq_min = chebyshev_freq_min
        self.chebyshev_freq_max = chebyshev_freq_max
        # multi-layer MLP
        layer_dim = [dim_inputs] + dim_hidden + [dim_outputs]
        # Adjust input dimension if using Chebyshev basis for the first layer
        if self.use_chebyshev_basis:
            num_chebyshev_features = self.chebyshev_freq_max - self.chebyshev_freq_min + 1
            layer_dim[0] = num_chebyshev_features

        self.linear = nn.ModuleList([nn.Linear(layer_dim[i], layer_dim[i + 1])
                                     for i in range(len(layer_dim) - 1)])
        # activation function
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_chebyshev_basis:
            x_features = chebyshev_transformed_features(x, self.chebyshev_freq_min, self.chebyshev_freq_max)
        else:
            x_features = x

        for i, layer in enumerate(self.linear):
            x_features = layer(x_features)
            # not applying nonlinear activation in the last layer
            if i < len(self.linear) - 1:
                x_features = self.act(x_features)
        return x_features


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
                 act: nn.Module = nn.ReLU(), enforce_bc: bool = False,
                 g0_type: str = "multilinear", d_type: str = "quadratic_bubble",
                 use_chebyshev_basis: bool = False,
                 chebyshev_freq_min: int = 0,
                 chebyshev_freq_max: int = 0) -> None:
        super().__init__()
        self.mesh = mesh
        # currently the same model on each level
        self.dim_inputs = dim_inputs
        self.dim_outputs = dim_outputs
        self.enforce_bc = enforce_bc

        self.g0_func = None
        self.d_func = None
        if self.enforce_bc:
            self.g0_func = get_g0_func(
                u_exact_func=self.mesh.pde.u_ex,
                domain_dim=1,
                domain_bounds=(self.mesh.ax, self.mesh.bx),
                g0_type=g0_type
            )
            self.d_func = get_d_func(
                domain_dim=1,
                domain_bounds=(self.mesh.ax, self.mesh.bx),
                d_type=d_type
            )
            print(f"BCs will be enforced using g0_type: {g0_type} and d_type: {d_type}")

        self.models = nn.ModuleList([
            Level(dim_inputs=dim_inputs, dim_outputs=dim_outputs, dim_hidden=dim_hidden, act=act,
                  use_chebyshev_basis=use_chebyshev_basis,
                  chebyshev_freq_min=chebyshev_freq_min,
                  chebyshev_freq_max=chebyshev_freq_max)
            for _ in range(num_levels)
            ])
        
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
        assert out.shape[1] == self.num_active_levels() * self.dim_outputs
        return out

    def get_solution(self, x: torch.Tensor) -> torch.Tensor:
        raw_nn_output = self.forward(x)
        
        n_active = self.num_active_levels()
        # reshape to [batch_size, num_levels, dim_outputs]
        # and sum over levels
        if n_active > 1:
            raw_nn_output = raw_nn_output.view(-1, n_active, self.dim_outputs).sum(dim=1)
        
        if self.enforce_bc:
            g0_vals = self.g0_func(x)
            d_vals = self.d_func(x)
            y = g0_vals + d_vals * raw_nn_output
        else:
            y = raw_nn_output
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
        u = model.get_solution(xs)
        grad_u_pred = torch.autograd.grad(u, xs,
                                          grad_outputs=torch.ones_like(u),
                                          create_graph=True)[0]
        u_pred_sq = torch.sum(u**2, dim=1, keepdim=True)
        grad_u_pred_sq = torch.sum(grad_u_pred**2, dim=1, keepdim=True)

        f_val = mesh.pde.f(xs)
        fu_prod = f_val * u

        integrand_values = 0.5 * grad_u_pred_sq[1:-1] + 0.5 * mesh.pde.r * u_pred_sq[1:-1] - fu_prod[1:-1]
        loss = torch.mean(integrand_values)

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
def train(model, mesh, criterion, iterations, adam_iterations, learning_rate, num_check, num_plots, sweep_idx,
          level_idx, frame_dir, scheduler_gen):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = SOAP(model.parameters(), lr = 3e-3, betas=(.95, .95), weight_decay=.01,
    #                  precondition_frequency=10)
    scheduler = scheduler_gen(optimizer)
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
            scheduler_step(scheduler, loss)

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
    # Ensure chebyshev_freq_max is at least chebyshev_freq_min for range to be valid
    if args.use_chebyshev_basis and args.chebyshev_freq_max < args.chebyshev_freq_min:
        raise ValueError("chebyshev_freq_max must be >= chebyshev_freq_min when using Chebyshev basis.")
    print_args(args=args)
    # PDE
    pde = PDE(high=args.high_freq, mu=args.mu, r=args.gamma,
              problem=args.problem_id)
    # Loss function [supervised with analytical solution (-1) or PINN loss (0)]
    loss = Loss(loss_type=args.loss_type, bc_weight=args.bc_weight)
    print(f"Using loss: {loss.name}")
    # scheduler gen takes optimizer to return scheduler
    scheduler_gen = get_scheduler_generator(args)
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
                         enforce_bc=args.enforce_bc,
                         g0_type=args.bc_extension,
                         d_type=args.distance,
                         use_chebyshev_basis=args.use_chebyshev_basis,
                         chebyshev_freq_min=args.chebyshev_freq_min,
                         chebyshev_freq_max=args.chebyshev_freq_max)
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
        for lev in range(args.levels):
            # Turn all levels to "frozen" if they are not off
            for k in range(args.levels):
                if model.get_status(level_idx=k) != LevelStatus.OFF:
                    model.set_status(level_idx=k, status=LevelStatus.FROZEN)
            # Turn level l to "train"
            model.set_status(level_idx=lev, status=LevelStatus.TRAIN)
            print("\nTraining Level", lev)
            model.print_status()
            # set scale
            scale = lev + 1
            model.set_scale(level_idx=lev, scale=scale)
            # Crank that !@#$ up
            train(model=model, mesh=mesh, criterion=loss, iterations=args.epochs,
                  adam_iterations=args.adam_epochs,
                  learning_rate=args.lr, num_check=args.num_checks, num_plots=num_plots,
                  sweep_idx=i, level_idx=lev, frame_dir=frame_dir, scheduler_gen=scheduler_gen)
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
        err = main(['--levels', '4', '--epochs', '10000', '--sweeps', '1', '--plot', '--enforce_bc', '--g0_type', 'hermite_cubic_2nd_deriv', '--d_type', 'quadratic_bubble'])
    else:
        err = main()
    try:
        import sys
        sys.exit(err)
    except SystemExit:
        pass  # Prevent traceback in Jupyter or VS Code
