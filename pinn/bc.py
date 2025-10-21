import torch
from typing import Union, Tuple, Callable
import itertools

# %% [markdown]
# Helper functions from the new BC implementation

# %%
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
        if domain_dim != 1:
            raise ValueError("Hermite cubic interpolation with 2nd derivatives is only supported for 1D problems.")
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

    if g0_type == "multilinear":
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

    raise ValueError(f"Unknown g0_type: {g0_type}. Choose 'multilinear' or 'hermite_cubic_2nd_deriv'.")


def _psi_tensor(t: torch.Tensor) -> torch.Tensor:
    return torch.where(t <= 0, torch.tensor(0.0, dtype=t.dtype, device=t.device), torch.exp(-1.0 / t))


def get_d_func(domain_dim: int, domain_bounds: Union[Tuple[float, float], Tuple[Tuple[float, float], ...]],
              d_type: str = "sin_half_period") -> Callable[[torch.Tensor], torch.Tensor]:
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

    if d_type == "inf_smooth_bump":
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

    if d_type == "abs_dist_complement":
        if domain_dim != 1: raise ValueError(f"d_type '{d_type}' is only supported for 1D problems.")
        def d_abs_dist_complement_val(x: torch.Tensor) -> torch.Tensor:
            x_val = x[:, 0]
            x_norm = (x_val - min_bounds[0]) / domain_length
            sqrt_term = torch.sqrt(x_norm**2 + (1.0 - x_norm)**2)
            return (1.0 - sqrt_term).unsqueeze(1)
        return d_abs_dist_complement_val

    if d_type == "ratio_bubble_dist":
        if domain_dim != 1: raise ValueError(f"d_type '{d_type}' is only supported for 1D problems.")
        def d_ratio_bubble_dist_val(x: torch.Tensor) -> torch.Tensor:
            x_val = x[:, 0]
            x_norm = (x_val - min_bounds[0]) / domain_length
            numerator = x_norm * (1.0 - x_norm)
            denominator = torch.sqrt(x_norm**2 + (1.0 - x_norm)**2)
            return (numerator / denominator).unsqueeze(1)
        return d_ratio_bubble_dist_val

    if d_type == "sin_half_period":
        if domain_dim != 1: raise ValueError(f"d_type '{d_type}' is only supported for 1D problems.")
        if domain_length is None: raise ValueError("Domain length must be defined for 'sin_half_period' d_type.")
        def d_sin_half_period_val(x: torch.Tensor) -> torch.Tensor:
            x_val = x[:, 0]
            argument = (torch.pi / domain_length) * (x_val - min_bounds[0])
            return torch.sin(argument).unsqueeze(1)
        return d_sin_half_period_val

    raise ValueError(f"Unknown d_type: {d_type}. Choose from 'quadratic_bubble', 'inf_smooth_bump', 'abs_dist_complement', 'ratio_bubble_dist', or 'sin_half_period'.")
