import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Union, Optional

class Problem(ABC):
    """
    Abstract Base Class for a Manufactured Solution / PDE problem.
    """
    def __init__(self, 
                 dim: int, 
                 domain_bounds: Union[Tuple[float, float], Tuple[Tuple[float, float], ...]],
                 device: str = "cpu"):
        self.dim = dim
        self.device = device
        
        # Standardize bounds to a list of tuples
        if dim == 1 and isinstance(domain_bounds[0], (float, int)):
            self.domain_bounds = (domain_bounds,)
        else:
            self.domain_bounds = domain_bounds
            
        self.min_bounds = torch.tensor([b[0] for b in self.domain_bounds], dtype=torch.float32, device=device)
        self.max_bounds = torch.tensor([b[1] for b in self.domain_bounds], dtype=torch.float32, device=device)

    @abstractmethod
    def u_exact(self, x: torch.Tensor) -> torch.Tensor:
        """The analytical solution u(x)."""
        pass

    @abstractmethod
    def source_term(self, x: torch.Tensor) -> torch.Tensor:
        """The right-hand side f(x) of the PDE."""
        pass

    @abstractmethod
    def operator(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        The differential operator L[u]. 
        For example, in -u_xx + r*u = f, this returns -u_xx + r*u.
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(dim={self.dim}, bounds={self.domain_bounds})"

class LinearEllipticProblem(Problem):
    """
    A specialized base class for linear problems like Poisson or Helmholtz.
    Contains shared parameters like gamma (reaction coefficient).
    """
    def __init__(self, dim, domain_bounds, gamma=0.0, device="cpu"):
        super().__init__(dim, domain_bounds, device)
        self.gamma = gamma

    def operator(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Default implementation for -Laplacian(u) + gamma * u.
        Note: Requires autograd-ready x.
        """
        # Calculate gradients
        grad_u = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        
        # Calculate Laplacian
        laplacian = torch.zeros_like(u)
        for i in range(self.dim):
            grad_u_i = grad_u[:, i:i+1]
            laplacian += torch.autograd.grad(grad_u_i, x, grad_outputs=torch.ones_like(grad_u_i), 
                                            create_graph=True)[0][:, i:i+1]
            
        return -laplacian + self.gamma * u
