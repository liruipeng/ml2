import torch
import numpy as np
from .base import LinearEllipticProblem, Problem

class Problem1(LinearEllipticProblem):
    """
    Analytical solution: u(x) = sum(c_k * sin(2 * w_k * pi * x))
    PDE: -u_xx + gamma * u = f
    """
    def __init__(self, high_freq, gamma=0.0, device="cpu"):
        # Problem 1 is defined on [0, 1]
        super().__init__(dim=1, domain_bounds=(0.0, 1.0), gamma=gamma, device=device)
        
        # Setup frequencies and coefficients
        self.w = torch.arange(1, high_freq + 1, 2, device=device)
        self.c = torch.ones_like(self.w, device=device)

    def u_exact(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.zeros_like(x)
        for w, c in zip(self.w, self.c):
            y += c * torch.sin(2 * w * torch.pi * x)
        return y

    def source_term(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.zeros_like(x)
        for w, c in zip(self.w, self.c):
            pi_w = 2 * torch.pi * w
            y += c * (pi_w**2 + self.gamma) * torch.sin(pi_w * x)
        return y

class MScaleProblem(LinearEllipticProblem):
    """
    Problem 2 from MscaleDNN: u(x) = exp(-x^2) * sin(mu * x^2)
    PDE: -u_xx + gamma * u = f
    """
    def __init__(self, mu=70.0, gamma=0.0, device="cpu"):
        # MScale problems usually focus on [0, 1] or [-1, 1]
        super().__init__(dim=1, domain_bounds=(0.0, 1.0), gamma=gamma, device=device)
        self.mu = mu

    def u_exact(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-x**2) * torch.sin(self.mu * x**2)

    def source_term(self, x: torch.Tensor) -> torch.Tensor:
        z = x**2
        a = self.gamma + 4 * z * (self.mu**2 - 1) + 2
        b = self.mu * z
        c = 8 * b - 2 * self.mu
        return torch.exp(-z) * (a * torch.sin(b) + c * torch.cos(b))

class NonlinearReactionDiffusion1D(Problem):
    """
    Example of a non-linear elliptic PDE: -u_xx + u^3 = f
    """
    def __init__(self, device="cpu"):
        super().__init__(dim=1, domain_bounds=(0.0, 1.0), device=device)

    def u_exact(self, x: torch.Tensor) -> torch.Tensor:
        # Chosen to satisfy homogeneous BCs: u(0)=u(1)=0
        return torch.sin(torch.pi * x)

    def operator(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Compute -u_xx
        grad_u = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(grad_u, x, grad_outputs=torch.ones_like(grad_u), create_graph=True)[0]
        # Return -u_xx + u^3
        return -u_xx + u**3

    def source_term(self, x: torch.Tensor) -> torch.Tensor:
        u = self.u_exact(x)
        # f = pi^2 * sin(pi*x) + sin^3(pi*x)
        return (torch.pi**2) * torch.sin(torch.pi * x) + u**3
