import torch

def calculate_drm_loss(u_nn_model, f_exact_func, domain_points):
    domain_points.requires_grad_(True)
    u_pred = u_nn_model(domain_points)
    
    grad_u_pred = torch.autograd.grad(u_pred, domain_points, 
                                      grad_outputs=torch.ones_like(u_pred), 
                                      create_graph=True)[0]
    
    grad_u_pred_sq = torch.sum(grad_u_pred**2, dim=1, keepdim=True)

    f_val = f_exact_func(domain_points)
    fu_prod = f_val * u_pred

    integrand_values = 0.5 * grad_u_pred_sq - fu_prod
    loss = torch.mean(integrand_values)
    
    domain_points.requires_grad_(False)
    return loss

def calculate_pinn_loss(u_nn_model, f_exact_func, domain_points, domain_dim):
    domain_points.requires_grad_(True)
    u_pred = u_nn_model(domain_points)
    
    # First derivatives (gradients)
    grad_u_pred = torch.autograd.grad(u_pred, domain_points, 
                                     grad_outputs=torch.ones_like(u_pred), 
                                     create_graph=True)[0] 

    # Second derivatives (Laplacian)
    laplacian_u_pred = torch.zeros_like(u_pred)
    for i in range(domain_dim):
        d2u_dxi2 = torch.autograd.grad(grad_u_pred[:, i], domain_points, 
                                        grad_outputs=torch.ones_like(grad_u_pred[:, i]), 
                                        create_graph=True)[0][:, i]
        laplacian_u_pred += d2u_dxi2.unsqueeze(1) 
    
    f_val = f_exact_func(domain_points)
    residual = -laplacian_u_pred - f_val

    loss = torch.mean(residual**2)
    
    domain_points.requires_grad_(False) 
    return loss
