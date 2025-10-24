import torch
import numpy as np

def get_manufactured_solution(case_number, domain_dim):
    """
    Generates the manufactured solution u_exact, the diffusion coefficient a_exact,
    the reaction coefficient c_exact, and the source term f_exact for
    the Reaction-Diffusion equation: -div(a(x) grad u) + c(x)u = f(x).
    """

    def u_exact(x): # x : (batch_size, domain_dim)
        if domain_dim == 1:
            x0 = x[:, 0]
            if case_number == 1:
                return (torch.sin(torch.pi * x0) + torch.sin(2 * torch.pi * x0) + torch.sin(3 * torch.pi * x0) + torch.sin(4 * torch.pi * x0) + torch.sin(5 * torch.pi * x0) + torch.sin(6 * torch.pi * x0) + torch.sin(7 * torch.pi * x0) + torch.sin(8 * torch.pi * x0)).unsqueeze(1)
            elif case_number == 2:
                return (torch.sin(2 * torch.pi * x0) + torch.sin(20 * torch.pi * x0)).unsqueeze(1)
            elif case_number == 3:
                return (torch.sin(torch.pi * x0) + 0.1 * torch.sin(40 * torch.pi * x0)).unsqueeze(1)
            else:
                raise ValueError(f"Manufactured solution case {case_number} not defined for 1D.")
        else:
            raise ValueError(f"Manufactured solution case {case_number} not defined for dimension {domain_dim}")

    def a_exact(x): # diffusion coefficient
        if domain_dim == 1:
            x0 = x[:, 0]
            if case_number == 1:
                return (1.0 + 0.5 * torch.sin(2 * torch.pi * x0)).unsqueeze(1)
            elif case_number == 2:
                return (1.0 + 0.5 * torch.sin(2 * torch.pi * x0)).unsqueeze(1)
            elif case_number == 3:
                return (1.0 + 0.05 * torch.sin(100 * torch.pi * x0)).unsqueeze(1)
            else:
                raise ValueError(f"Diffusion coefficient case {case_number} not defined for 1D.")
        else:
            raise ValueError(f"Diffusion coefficient case {case_number} not defined for dimension {domain_dim}")

    def c_exact(x): # reaction coefficient
        if domain_dim == 1:
            x0 = x[:, 0]
            if case_number == 1:
                return (1.0 + 0.5 * torch.cos(torch.pi * x0)).unsqueeze(1)
            elif case_number == 2:
                return (1.0 + 0.5 * torch.cos(torch.pi * x0)).unsqueeze(1)
            elif case_number == 3:
                return 0.0 * torch.zeros_like(x0).unsqueeze(1)
            else:
                raise ValueError(f"Reaction coefficient case {case_number} not defined for 1D.")
        else:
            raise ValueError(f"Reaction coefficient case {case_number} not defined for dimension {domain_dim}")

    def f_exact(x):
        x_clone = x.clone().detach().requires_grad_(True)
        
        u_val = u_exact(x_clone) # (batch_size, 1)
        a_val = a_exact(x_clone) # (batch_size, 1) - assuming scalar a(x)
        c_val = c_exact(x_clone) # (batch_size, 1)

        # Compute grad u
        grad_u = torch.autograd.grad(u_val, x_clone, grad_outputs=torch.ones_like(u_val), create_graph=True)[0] # (batch_size, domain_dim)

        # Compute grad(a * grad u)
        a_grad_u = a_val * grad_u # Element-wise product: (batch_size, domain_dim)

        # Compute divergence of (a * grad u)
        div_a_grad_u = torch.zeros_like(u_val) # (batch_size, 1)
        
        for i in range(domain_dim):
            # Compute d/dxi (a * du/dxi)
            d_a_grad_u_dxi = torch.autograd.grad(a_grad_u[:, i], x_clone,
                                                  grad_outputs=torch.ones_like(a_grad_u[:, i]),
                                                  create_graph=True)[0][:, i]
            div_a_grad_u += d_a_grad_u_dxi.unsqueeze(1)
        
        # f = -div(a grad u) + c*u
        f_val = -div_a_grad_u + c_val * u_val
            
        x_clone.requires_grad_(False)
        
        return f_val

    return u_exact, f_exact, a_exact, c_exact
