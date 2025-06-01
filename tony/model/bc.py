import torch
import numpy as np
import itertools
from typing import Union, Tuple, Callable

def get_g0_func(u_exact_func: Callable[[torch.Tensor], torch.Tensor], domain_dim: int, domain_bounds: Union[Tuple[float, float], Tuple[Tuple[float, float], ...]]) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Returns g0(x) that satisfies g0|_boundary = u_exact.
    g0(x) is a multilinear interpolation of u_exact values at the domain boundaries.

    Args:
        u_exact_func (callable): The exact solution function u(x) which takes
                                 a torch.Tensor of shape (num_points, domain_dim)
                                 and returns a torch.Tensor of shape (num_points, 1).
        domain_dim (int): The dimension of the problem (e.g., 1, 2, 3).
        domain_bounds (tuple or list): Defines the spatial extent.
            For 1D: (min_val, max_val)
            For ND: ((min_x, max_x), (min_y, max_y), ...)

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

    # Cache boundary values of u_exact_func at all 2^D corners
    boundary_values = {}
    
    # Generate all 2^D corner coordinates
    dim_ranges = [[min_bounds[d].item(), max_bounds[d].item()] for d in range(domain_dim)]
    
    for corner_coords in itertools.product(*dim_ranges):
        # corner coordinates as a tuple, e.g., (0.0,) or (0.0, 0.0)
        corner_coords_tensor = torch.tensor(corner_coords, dtype=torch.float32).unsqueeze(0) # unsqueeze to (1, domain_dim)
        with torch.no_grad():
            boundary_values[corner_coords] = u_exact_func(corner_coords_tensor).item() # tuple 

    def g0_func_val(x: torch.Tensor) -> torch.Tensor:
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

    return g0_func_val

def get_d_func(domain_dim, domain_bounds):
    """
    Returns d(x), a function that is zero on the boundary and positive in the interior.
    For a unit hypercube [min, max]^dim, this is (x_1-min)(max-x_1)...(x_d-min)(max-x_d).
    """
    def d_func_val(x):
        d_vals = torch.ones_like(x[:, 0], dtype=torch.float32) 

        for i in range(domain_dim):
            x_i = x[:, i]
            min_val, max_val = domain_bounds[i]
            d_vals *= (x_i - min_val) * (max_val - x_i)
        return d_vals.unsqueeze(1) # Return shape (batch_size, 1)

    return d_func_val
