import torch
import numpy as np
import itertools
from typing import Union, Tuple, Callable

# Helper function for calculating 1D Laplacian (second derivative) using autograd
def _calculate_laplacian_1d(func: Callable[[torch.Tensor], torch.Tensor], x_val: float) -> torch.Tensor:
    """
    Calculates the 1D Laplacian (second spatial derivative) of a function at a specific point.
    Assumes func takes input of shape (N, 1) and returns (N, 1).
    """
    x_tensor = torch.tensor([[x_val]], dtype=torch.float32, requires_grad=True)
    u = func(x_tensor)

    # First derivative
    # Using create_graph=True to allow for second derivative computation
    grad_u = torch.autograd.grad(u, x_tensor,
                                 grad_outputs=torch.ones_like(u),
                                 create_graph=True, retain_graph=True)[0]

    # Second derivative
    # create_graph=False as we only need the value here for g0 coefficients
    laplacian_u = torch.autograd.grad(grad_u, x_tensor,
                                       grad_outputs=torch.ones_like(grad_u),
                                       create_graph=False, retain_graph=False)[0]
    return laplacian_u


def get_g0_func(
    u_exact_func: Callable[[torch.Tensor], torch.Tensor],
    domain_dim: int,
    domain_bounds: Union[Tuple[float, float], Tuple[Tuple[float, float], ...]],
    interpolation_type: str = "linear_multidim" # "linear_multidim" or "hermite_cubic_2nd_deriv"
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Returns g0(x) that satisfies g0|_boundary = u_exact.

    Args:
        u_exact_func (callable): The exact solution function u(x) which takes
                                  a torch.Tensor of shape (num_points, domain_dim)
                                  and returns a torch.Tensor of shape (num_points, 1).
                                  Must be differentiable if Hermite interpolation is used.
        domain_dim (int): The dimension of the problem (e.g., 1, 2, 3).
        domain_bounds (tuple or list): Defines the spatial extent.
            For 1D: (min_val, max_val)
            For ND: ((min_x, max_x), (min_y, max_y), ...)
        interpolation_type (str): Type of interpolation to use for g0(x).
                                  "linear_multidim" for multi-linear interpolation (default).
                                  "hermite_cubic_2nd_deriv" for 1D cubic Hermite interpolation
                                  using boundary values and second derivatives.

    Returns:
        callable: A function g0(x) that takes a torch.Tensor of shape (num_points, domain_dim)
                  and returns a torch.Tensor of shape (num_points, 1), representing
                  the boundary-satisfying part of the solution.
    """

    # Ensure domain_bounds is consistently a tuple of (min, max) for each dimension
    domain_bounds_tuple = domain_bounds
    if domain_dim == 1 and not isinstance(domain_bounds[0], (tuple, list)):
        domain_bounds_tuple = (domain_bounds,)

    min_bounds = torch.tensor([b[0] for b in domain_bounds_tuple], dtype=torch.float32)
    max_bounds = torch.tensor([b[1] for b in domain_bounds_tuple], dtype=torch.float32)

    if interpolation_type == "hermite_cubic_2nd_deriv":
        if domain_dim != 1:
            raise ValueError("Hermite cubic interpolation with 2nd derivatives is only supported for 1D problems.")

        x0, x1 = min_bounds.item(), max_bounds.item()
        h = x1 - x0

        # Get boundary values
        u_x0 = u_exact_func(torch.tensor([[x0]], dtype=torch.float32)).item()
        u_x1 = u_exact_func(torch.tensor([[x1]], dtype=torch.float32)).item()

        # Get second derivatives (Laplacian in 1D) at boundaries
        u_prime_prime_x0 = _calculate_laplacian_1d(u_exact_func, x0).item()
        u_prime_prime_x1 = _calculate_laplacian_1d(u_exact_func, x1).item()

        # Calculate coefficients for P(x) = a0 + a1*x + a2*x^2 + a3*x^3
        # Using formulas derived from P(x0)=u_x0, P(x1)=u_x1, P''(x0)=u''_x0, P''(x1)=u''_x1
        a3 = (u_prime_prime_x1 - u_prime_prime_x0) / (6 * h)
        a2 = u_prime_prime_x0 / 2 - 3 * a3 * x0
        a1 = (u_x1 - u_x0) / h - a2 * (x1 + x0) - a3 * (x1**2 + x1 * x0 + x0**2)
        a0 = u_x0 - a1 * x0 - a2 * x0**2 - a3 * x0**3
        
        # Store coefficients in a detached tensor to avoid tracking gradients during g0_func_val execution
        coeffs = torch.tensor([a0, a1, a2, a3], dtype=torch.float32)

        def g0_hermite_cubic_val(x: torch.Tensor) -> torch.Tensor:
            """
            Calculates the 1D cubic Hermite interpolation.
            Args:
                x (torch.Tensor): Input tensor of shape (num_points, 1).
            Returns:
                torch.Tensor: g0(x) values of shape (num_points, 1).
            """
            # Ensure x is (num_points,) for polynomial evaluation, then reshape to (num_points, 1)
            x_flat = x[:, 0]
            # Evaluate polynomial: a0 + a1*x + a2*x^2 + a3*x^3
            g0_vals = coeffs[0] + coeffs[1] * x_flat + coeffs[2] * (x_flat**2) + coeffs[3] * (x_flat**3)
            return g0_vals.unsqueeze(1) # Reshape to (num_points, 1)

        return g0_hermite_cubic_val

    elif interpolation_type == "linear_multidim":
        boundary_values = {}
        # Generate all 2^D corner coordinates
        dim_ranges = [[min_bounds[d].item(), max_bounds[d].item()] for d in range(domain_dim)]
        
        for corner_coords in itertools.product(*dim_ranges):
            # corner coordinates as a tuple, e.g., (0.0,) or (0.0, 0.0)
            corner_coords_tensor = torch.tensor(corner_coords, dtype=torch.float32).unsqueeze(0) # unsqueeze to (1, domain_dim)
            with torch.no_grad(): # Don't track gradients for boundary value lookups
                boundary_values[corner_coords] = u_exact_func(corner_coords_tensor).item() # tuple 

        def g0_linear_multidim_val(x: torch.Tensor) -> torch.Tensor:
            """
            Calculates the multilinear interpolation satisfying the boundary conditions.
            Args:
                x (torch.Tensor): Input tensor of shape (num_points, domain_dim).
            Returns:
                torch.Tensor: g0(x) values of shape (num_points, 1).
            """
            num_points = x.shape[0]
            
            # Calculate normalized coordinates xi for each dimension and each point
            xi = (x - min_bounds.to(x.device)) / (max_bounds.to(x.device) - min_bounds.to(x.device))
            
            # Clamp xi to [0, 1] to handle potential floating point errors near boundaries
            xi = torch.clamp(xi, 0.0, 1.0)

            g0_vals = torch.zeros((num_points, 1), device=x.device)

            # Iterate through all 2^D corners based on their min/max configuration
            for corner_label in itertools.product([0, 1], repeat=domain_dim):
                current_corner_key_list = []
                weight_factors = torch.ones((num_points, 1), device=x.device) # Initialize product of weights

                for d in range(domain_dim):
                    if corner_label[d] == 0: # min_bound
                        current_corner_key_list.append(min_bounds[d].item())
                        weight_factors *= (1 - xi[:, d]).unsqueeze(1)
                    else: # max_bound 
                        current_corner_key_list.append(max_bounds[d].item())
                        weight_factors *= xi[:, d].unsqueeze(1)
                    
                # Convert list to tuple to use as a dictionary key, ensuring consistency
                corner_key_tuple = tuple(current_corner_key_list)

                # Get the pre-calculated u_exact value at this corner
                corner_value = boundary_values[corner_key_tuple]
                
                g0_vals += corner_value * weight_factors
                
            return g0_vals

        return g0_linear_multidim_val

    else:
        raise ValueError(f"Unknown interpolation_type: {interpolation_type}. Choose 'linear_multidim' or 'hermite_cubic_2nd_deriv'.")


# Helper for the infinitely smooth bump function (psi(t) = exp(-1/t) for t > 0, and 0 for t <= 0)
def _psi_tensor(t: torch.Tensor) -> torch.Tensor:
    """
    Helper function for the C-infinity bump function construction.
    psi(t) = exp(-1/t) for t > 0, and 0 for t <= 0.
    """
    return torch.where(t <= 0, torch.tensor(0.0, dtype=t.dtype, device=t.device), torch.exp(-1.0 / t))


def get_d_func(domain_dim: int, domain_bounds: Union[Tuple[float, float], Tuple[Tuple[float, float], ...]], 
               d_type: str = "quadratic_bubble") -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Returns d(x), a function that is zero on the boundary and positive in the interior.

    Args:
        domain_dim (int): The dimension of the problem (e.g., 1, 2, 3).
        domain_bounds (tuple or list): Defines the spatial extent.
            For 1D: (min_val, max_val)
            For ND: ((min_x, max_x), (min_y, max_y), ...)
        d_type (str): Type of d(x) function.
                      "quadratic_bubble" for a simple polynomial (default).
                      "inf_smooth_bump" for a generalized C-infinity bump function (product of 1D bumps).
                      "abs_dist_complement" for 1D: 1 - sqrt(x_norm^2 + (1-x_norm)^2).
                      "ratio_bubble_dist" for 1D: x_norm*(1-x_norm) / sqrt(x_norm^2 + (1-x_norm)^2).

    Returns:
        callable: A function d(x) that takes a torch.Tensor of shape (num_points, domain_dim)
                  and returns a torch.Tensor of shape (num_points, 1).
    """
    # Ensure domain_bounds is consistently a tuple of (min, max) for each dimension
    domain_bounds_tuple = domain_bounds
    if domain_dim == 1 and not isinstance(domain_bounds[0], (tuple, list)):
        domain_bounds_tuple = (domain_bounds,)

    min_bounds = torch.tensor([b[0] for b in domain_bounds_tuple], dtype=torch.float32)
    max_bounds = torch.tensor([b[1] for b in domain_bounds_tuple], dtype=torch.float32)
    
    # Calculate domain length for 1D normalization, only applicable for 1D specific d_types
    domain_length = (max_bounds[0] - min_bounds[0]).item() if domain_dim == 1 else None

    if d_type == "quadratic_bubble":
        def d_func_val(x: torch.Tensor) -> torch.Tensor:
            d_vals = torch.ones_like(x[:, 0], dtype=torch.float32, device=x.device) 

            for i in range(domain_dim):
                x_i = x[:, i]
                min_val, max_val = domain_bounds_tuple[i]
                d_vals *= (x_i - min_val) * (max_val - x_i)
            return d_vals.unsqueeze(1) # Return shape (num_points, 1)
        return d_func_val

    elif d_type == "inf_smooth_bump":
        # Generalization to N-D for an axis-aligned hyper-rectangle
        # d(x) = product_i( exp(-1/(R_i^2 - (x_i - x_c_i)^2)) )
        # as a product of _psi_tensor calls for each dimension.

        def d_inf_smooth_bump_val(x: torch.Tensor) -> torch.Tensor:
            # x is (num_points, domain_dim)
            
            # Initialize product_terms with ones, to multiply contributions from each dimension
            product_terms = torch.ones((x.shape[0],), dtype=x.dtype, device=x.device)

            for i in range(domain_dim):
                x_i = x[:, i] # Current dimension's coordinates for all points
                min_val_i = min_bounds[i]
                max_val_i = max_bounds[i]
                
                x_c_i = (min_val_i + max_val_i) / 2.0 # Center of domain in this dimension
                R_i = (max_val_i - min_val_i) / 2.0 # Half-width of domain in this dimension
                R_i_squared = R_i**2 
                
                # The argument for _psi_tensor: R_i^2 - (x_i - x_c_i)^2
                arg_for_psi = R_i_squared - (x_i - x_c_i)**2
                
                # Multiply the _psi_tensor output for each dimension
                product_terms *= _psi_tensor(arg_for_psi)
            
            return product_terms.unsqueeze(1) # Reshape to (num_points, 1)
            
        return d_inf_smooth_bump_val

    elif d_type == "abs_dist_complement":
        if domain_dim != 1:
            raise ValueError(f"d_type '{d_type}' is only supported for 1D problems.")
        
        def d_abs_dist_complement_val(x: torch.Tensor) -> torch.Tensor:
            x_val = x[:, 0] # Extract the 1D coordinate

            # Normalize x to [0,1]
            x_norm = (x_val - min_bounds[0]) / domain_length
            
            # Calculate sqrt(x_norm^2 + (1-x_norm)^2)
            sqrt_term = torch.sqrt(x_norm**2 + (1.0 - x_norm)**2)
            
            # d(x) = 1 - sqrt_term
            return (1.0 - sqrt_term).unsqueeze(1)
            
        return d_abs_dist_complement_val
        
    elif d_type == "ratio_bubble_dist":
        if domain_dim != 1:
            raise ValueError(f"d_type '{d_type}' is only supported for 1D problems.")
        
        def d_ratio_bubble_dist_val(x: torch.Tensor) -> torch.Tensor:
            x_val = x[:, 0] # Extract the 1D coordinate

            # Normalize x to [0,1]
            x_norm = (x_val - min_bounds[0]) / domain_length
            
            numerator = x_norm * (1.0 - x_norm)
            denominator = torch.sqrt(x_norm**2 + (1.0 - x_norm)**2)
            
            return (numerator / denominator).unsqueeze(1)
            
        return d_ratio_bubble_dist_val

    else:
        raise ValueError(f"Unknown d_type: {d_type}. Choose from 'quadratic_bubble', 'inf_smooth_bump', 'abs_dist_complement', or 'ratio_bubble_dist'.")
