import torch
import numpy as np

def get_manufactured_solution(case_number, domain_dim):
    """
    Generates the manufactured solution u_exact, its Laplacian f, and boundary condition g.
    Uses pre-defined functions based on `case_number` for safety and clarity.
    """

    def u_exact(x): # x : (batch_size, domain_dim)
        if domain_dim == 1:
            x0 = x[:, 0]
            if case_number == 1:
                return torch.sin(torch.pi * x0).unsqueeze(1)
            elif case_number == 2:
                return torch.sin(3 * torch.pi * x0).unsqueeze(1)
            elif case_number == 3:
                return torch.cos(3 * torch.pi * x0).unsqueeze(1)
            elif case_number == 4:
                return (torch.sin(3 * torch.pi * x0) + torch.sin(7 * torch.pi * x0)).unsqueeze(1)
            elif case_number == 5:
                return (torch.sin(3 * torch.pi * x0) * torch.sin(7 * torch.pi * x0)).unsqueeze(1)
            elif case_number == 6:
                return (torch.sin(torch.pi * x0) + torch.sin(2 * torch.pi * x0) + torch.sin(3 * torch.pi * x0) + torch.sin(4 * torch.pi * x0) + torch.sin(5 * torch.pi * x0) + torch.sin(6 * torch.pi * x0) + torch.sin(7 * torch.pi * x0) + torch.sin(8 * torch.pi * x0)).unsqueeze(1)
            elif case_number == 7:
                return (torch.sin(2 * torch.pi * x0) + torch.sin(20 * torch.pi * x0)).unsqueeze(1)
            else:
                raise ValueError(f"Manufactured solution case {case_number} not defined for 1D.")
        elif domain_dim == 2:
            x0 = x[:, 0]
            x1 = x[:, 1]
            if case_number == 1:
                return (torch.sin(torch.pi * x0) * torch.sin(torch.pi * x1)).unsqueeze(1)
            else:
                raise ValueError(f"Manufactured solution case {case_number} not defined for 2D.")
        elif domain_dim == 3:
            x0 = x[:, 0]
            x1 = x[:, 1]
            x2 = x[:, 2]
            if case_number == 1:
                return (torch.sin(torch.pi * x0) * torch.sin(torch.pi * x1) * torch.sin(torch.pi * x2)).unsqueeze(1)
            else:
                raise ValueError(f"Manufactured solution case {case_number} not defined for 3D.")
            
        raise ValueError(f"Manufactured solution case {case_number} not defined for dimension {domain_dim}")

    def f_exact(x):
        x_clone = x.clone().detach().requires_grad_(True) # Use a clone to avoid modifying original 'x'
        u_val = u_exact(x_clone) # (batch_size, 1)
        grad_u = torch.autograd.grad(u_val, x_clone, grad_outputs=torch.ones_like(u_val), create_graph=True)[0] # (batch_size, domain_dim)
        laplacian_u = torch.zeros_like(u_val) # (batch_size, 1)
        
        for i in range(domain_dim):
            # Compute gradient of grad_u[:, i] with respect to x_clone[:, i]
            d2u_dxi2 = torch.autograd.grad(grad_u[:, i], x_clone, 
                                            grad_outputs=torch.ones_like(grad_u[:, i]), 
                                            create_graph=True)[0][:, i]
            laplacian_u += d2u_dxi2.unsqueeze(1) 
            
        x_clone.requires_grad_(False)
        return -laplacian_u

    return u_exact, f_exact
