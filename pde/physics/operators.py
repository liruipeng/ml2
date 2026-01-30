import torch
from typing import Union, Tuple

def gradient(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Computes the gradient of u with respect to x.
    If u is a scalar field, returns a vector [du/dx1, du/dx2, ...].
    """
    return torch.autograd.grad(
        u, x, 
        grad_outputs=torch.ones_like(u), 
        create_graph=True, 
        retain_graph=True
    )[0]

def laplacian(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Computes the Laplacian (Δu) for any dimension of x.
    Δu = div(grad(u)) = sum(d^2u / dxi^2)
    """
    grad_u = gradient(u, x)
    dims = x.shape[1]
    lap = torch.zeros_like(u)
    
    for i in range(dims):
        # Compute d/dxi of the i-th component of the gradient
        grad_u_i = grad_u[:, i:i+1]
        second_deriv = torch.autograd.grad(
            grad_u_i, x, 
            grad_outputs=torch.ones_like(grad_u_i),
            create_graph=True,
            retain_graph=True
        )[0][:, i:i+1]
        lap += second_deriv
        
    return lap

def divergence(f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Computes the divergence of a vector field f with respect to x.
    div(f) = sum(dfi / dxi)
    """
    dims = x.shape[1]
    div = torch.zeros((f.shape[0], 1), device=f.device)
    
    for i in range(dims):
        fi = f[:, i:i+1]
        div += torch.autograd.grad(
            fi, x, 
            grad_outputs=torch.ones_like(fi),
            create_graph=True,
            retain_graph=True
        )[0][:, i:i+1]
        
    return div

def directional_derivative(u: torch.Tensor, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Computes the directional derivative of u in direction v.
    (v · grad)u
    """
    grad_u = gradient(u, x)
    return torch.sum(v * grad_u, dim=1, keepdim=True)
