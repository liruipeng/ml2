import torch
from typing import Tuple, List, Optional

class Mesh:
    """
    Handles the generation and storage of collocation points for training and testing.
    """
    def __init__(self, 
                 dim: int, 
                 bounds: List[Tuple[float, float]], 
                 device: str = "cpu"):
        self.dim = dim
        self.bounds = bounds
        self.device = device
        
        # Convert bounds to tensors for easy scaling
        self.min_b = torch.tensor([b[0] for b in bounds], device=device, dtype=torch.float32)
        self.max_b = torch.tensor([b[1] for b in bounds], device=device, dtype=torch.float32)

    def _scale_points(self, points: torch.Tensor) -> torch.Tensor:
        """Scales points from [0, 1] to [min_b, max_b]."""
        return self.min_b + points * (self.max_b - self.min_b)

    def generate_interior_points(self, n: int, method: str = "uniform") -> torch.Tensor:
        """
        Generates n points inside the domain.
        Methods: 'uniform', 'random'
        """
        if method == "uniform":
            if self.dim == 1:
                x = torch.linspace(0, 1, n, device=self.device).reshape(-1, 1)
            else:
                # For nD uniform, n is points per dimension
                ticks = [torch.linspace(0, 1, n, device=self.device) for _ in range(self.dim)]
                grid = torch.meshgrid(*ticks, indexing='ij')
                x = torch.stack(grid, dim=-1).reshape(-1, self.dim)
        elif method == "random":
            x = torch.rand(n, self.dim, device=self.device)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        x = self._scale_points(x)
        x.requires_grad_(True)
        return x

    def generate_boundary_points(self, n_per_face: int) -> torch.Tensor:
        """
        Generates points on the boundaries. 
        In 1D, this is just the two endpoints.
        In nD, this samples the hyperplanes.
        """
        if self.dim == 1:
            return self.min_b.reshape(-1, 1), self.max_b.reshape(-1, 1)
        
        # Implementation for nD would generate points for each face
        # (e.g., x=0, x=1, y=0, y=1, etc.)
        boundary_points = []
        for d in range(self.dim):
            for val in [0.0, 1.0]:
                pts = torch.rand(n_per_face, self.dim, device=self.device)
                pts[:, d] = val
                boundary_points.append(self._scale_points(pts))
        
        return torch.cat(boundary_points, dim=0)

    def get_test_grid(self, n_test: int) -> torch.Tensor:
        """High-resolution uniform grid for error analysis."""
        return self.generate_interior_points(n_test, method="uniform")
