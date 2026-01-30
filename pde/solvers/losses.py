import torch
import torch.nn as nn
from physics.base import Problem

class LossFactory:
    """
    A collection of loss functionals for training Neural PDE solvers.
    """
    
    @staticmethod
    def pinn_loss(model: nn.Module, problem: Problem, x: torch.Tensor) -> torch.Tensor:
        """
        Standard PINN loss (Strong Form).
        Calculates the Mean Squared Error of the residual: L[u] - f = 0
        """
        u = model(x)
        # problem.operator computes the PDE residual (e.g., -u_xx + gamma*u)
        residual = problem.operator(u, x) - problem.source_term(x)
        return torch.mean(residual**2)

    @staticmethod
    def ritz_loss(model: nn.Module, problem: Problem, x: torch.Tensor) -> torch.Tensor:
        """
        Variational/Energy loss (Deep Ritz Method).
        Minimizes the energy functional: J(u) = \int [ 0.5*|\nabla u|^2 + 0.5*gamma*u^2 - f*u ] dx
        Only applicable to self-adjoint operators (like Poisson).
        """
        from physics.operators import gradient # Helper from our operators.py
        
        u = model(x)
        grad_u = gradient(u, x)
        
        # Energy density for -Laplacian(u) + gamma*u = f
        # Note: This assumes gamma is part of the problem's energy definition
        # For Poisson: 0.5 * |grad u|^2 - f*u
        kinetic_energy = 0.5 * torch.sum(grad_u**2, dim=1, keepdim=True)
        potential_energy = -problem.source_term(x) * u
        
        if hasattr(problem, 'gamma'):
            potential_energy += 0.5 * problem.gamma * (u**2)
            
        return torch.mean(kinetic_energy + potential_energy)

    @staticmethod
    def supervised_loss(model: nn.Module, problem: Problem, x: torch.Tensor) -> torch.Tensor:
        """
        L2 Error loss against the exact solution.
        Used primarily for debugging or 'Pre-training' levels.
        """
        u_pred = model(x)
        u_true = problem.u_exact(x)
        return torch.mean((u_pred - u_true)**2)

    @staticmethod
    def get_loss(loss_type: str):
        """Returns the requested loss function by name."""
        losses = {
            "pinn": LossFactory.pinn_loss,
            "ritz": LossFactory.ritz_loss,
            "supervised": LossFactory.supervised_loss
        }
        return losses.get(loss_type.lower(), LossFactory.pinn_loss)
