import numpy as np
import torch

def generate_uniform_grid_points(domain_bounds, num_uniform_partition):
    """
    Generates a uniform grid of interior points within the specified domain bounds,
    representing the midpoints of `num_uniform_partition` subintervals.
    These points are spaced such that they are h/2 from the boundaries and then h apart.

    Args:
        domain_bounds (tuple or list): Defines the spatial extent.
            For 1D: (min_val, max_val)
            For ND: ((min_x, max_x), (min_y, max_y), ...)
        num_uniform_partition (int): The number of subintervals along each dimension.
                                     This defines the mesh size h = (max-min)/num_uniform_partition.
                                     The number of generated interior points along each dimension
                                     will be num_uniform_partition (the midpoints of each interval).

    Returns:
        torch.Tensor: A tensor of shape (num_points, domain_dim) representing
                      the uniformly distributed interior grid points.
    """
    # Determine domain dimension based on the structure of domain_bounds
    if isinstance(domain_bounds[0], (tuple, list)):
        domain_dim = len(domain_bounds)
    else:
        domain_dim = 1
        domain_bounds = (domain_bounds,) # tuple

    points = []
    for i in range(domain_dim):
        min_val, max_val = domain_bounds[i]
        h = (max_val - min_val) / num_uniform_partition
        current_dim_points = np.linspace(min_val + h/2, max_val - h/2, num_uniform_partition)
        points.append(current_dim_points)

    if domain_dim == 1:
        return torch.from_numpy(points[0].reshape(-1, 1).astype(np.float32))
    else:
        # Tensorial grid 
        mesh = np.meshgrid(*points, indexing='ij')
        stacked_points = np.stack([m.flatten() for m in mesh], axis=-1)
        return torch.from_numpy(stacked_points.astype(np.float32))
