import torch

def calculate_drm_loss(u_nn_model, f_exact_func, a_exact_func, c_exact_func, domain_points):
    """
    Calculates the DRM (Domain Regularization Method) loss for the Reaction-Diffusion equation.
    The integral form is often derived from the weak form.
    For -div(a grad u) + c*u = f, a common energy functional (or part of it) is:
    Integral over domain [ 0.5 * a * |grad u|^2 + 0.5 * c * u^2 - f * u ] dX
    We'll assume the DRM loss aims to minimize this energy, or a similar variational form.
    """
    domain_points.requires_grad_(True)
    u_pred = u_nn_model(domain_points)
    
    # Get a(x) and c(x) values at domain_points
    a_val = a_exact_func(domain_points) # (batch_size, 1) or (batch_size, dim)
    c_val = c_exact_func(domain_points) # (batch_size, 1)

    grad_u_pred = torch.autograd.grad(u_pred, domain_points, 
                                       grad_outputs=torch.ones_like(u_pred), 
                                       create_graph=True)[0]
    
    # Calculate |grad u|^2
    grad_u_pred_sq = torch.sum(grad_u_pred**2, dim=1, keepdim=True)

    f_val = f_exact_func(domain_points)

    # The integrand for DRM based on the variational formulation of the PDE
    # For -div(a grad u) + c*u = f, the energy functional typically involves:
    # 0.5 * a * |grad u|^2 + 0.5 * c * u^2 - f * u
    # We will use this form.
    integrand_values = 0.5 * a_val * grad_u_pred_sq + 0.5 * c_val * u_pred**2 - f_val * u_pred
    loss = torch.mean(integrand_values) # Mean over the batch as an approximation of integral
    
    domain_points.requires_grad_(False)
    return loss

def calculate_pinn_loss(u_nn_model, f_exact_func, a_exact_func, c_exact_func, domain_points, domain_dim):
    """
    Calculates the PINN (Physics-Informed Neural Network) loss for the Reaction-Diffusion equation:
    -div(a(x) grad u) + c(x)u = f(x)
    The residual is: R(x) = -div(a(x) grad u_nn) + c(x)u_nn - f(x)
    """
    domain_points.requires_grad_(True)
    u_pred = u_nn_model(domain_points)
    
    # Get a(x) and c(x) values at domain_points
    a_val = a_exact_func(domain_points) # (batch_size, 1)
    c_val = c_exact_func(domain_points) # (batch_size, 1)

    # First derivatives (gradients)
    grad_u_pred = torch.autograd.grad(u_pred, domain_points, 
                                       grad_outputs=torch.ones_like(u_pred), 
                                       create_graph=True)[0] 
    
    # Compute a * grad u_pred
    # Assuming a_val is (batch_size, 1) or broadcastable (batch_size, domain_dim)
    a_grad_u_pred = a_val * grad_u_pred # Element-wise product

    # Compute divergence of (a * grad u_pred)
    # div(V) = dV0/dx0 + dV1/dx1 + ...
    div_a_grad_u_pred = torch.zeros_like(u_pred) # (batch_size, 1)
    
    for i in range(domain_dim):
        # Compute gradient of (a_grad_u_pred[:, i]) with respect to x_clone[:, i]
        d_a_grad_u_pred_dxi = torch.autograd.grad(a_grad_u_pred[:, i], domain_points,
                                                   grad_outputs=torch.ones_like(a_grad_u_pred[:, i]),
                                                   create_graph=True)[0][:, i]
        div_a_grad_u_pred += d_a_grad_u_pred_dxi.unsqueeze(1)
        
    # Get the exact source term f(x)
    f_val = f_exact_func(domain_points)

    # Residual for the Reaction-Diffusion equation: R(x) = -div(a grad u_nn) + c*u_nn - f
    residual = -div_a_grad_u_pred + c_val * u_pred - f_val

    loss = torch.mean(residual**2)
    
    domain_points.requires_grad_(False) 
    return loss
