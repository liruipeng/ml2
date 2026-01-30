# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
# ---

# %%
from scipy.fft import rfft, rfftfreq, dst
import numpy as np
import torch

def fourier_analysis(x, y, sine_series: bool = False):
    """
    Compute the magnitude spectrum using the Fast Fourier Transform (FFT).
    Ref: https://docs.scipy.org/doc/scipy/tutorial/fft.html
    """
    x = x.flatten()
    y = y.flatten()

    if not np.isrealobj(y):
        raise ValueError("Input y must be real for rfft.")

    # Check uniform sampling
    dx = np.diff(x.flatten())
    if not np.allclose(dx, dx[0], rtol=1e-5, atol=1e-7):
        raise ValueError("x must be uniformly sampled.")

    N = len(x)
    # Sampling interval
    Ts = dx[0]

    if sine_series:
        yf = dst(y, type=1)
        yf /= (N + 1)
        yf = yf[:N-1]
        L = N * Ts
        xf = np.arange(1, N + 1) * (np.pi / L)
        xf = xf[:N-1]
        yf_imag = np.zeros_like(yf)
        return xf, np.abs(yf), yf, yf_imag
    else:
        yf = rfft(y)
        xf = rfftfreq(N, Ts)
        yf *= 2.0 / N
        # Correct scaling for DC and Nyquist (they should not be doubled)
        yf[0] /= 2
        if N % 2 == 0:
            yf[-1] /= 2
        return xf, np.abs(yf), np.real(yf), -np.imag(yf)

def error_analysis(x: torch.Tensor, u_true: torch.Tensor, model: torch.nn.Module) -> dict:
    """
    Calculates the L2, H1, and H2 relative errors of the NN solution and its derivatives 
    against the true solution and its derivatives. Derivatives are calculated within 
    the routine using finite differences for the true solution and automatic 
    differentiation for the NN solution.

    Args:
        x (torch.Tensor): The 1D mesh points (must be uniformly spaced).
        u_true (torch.Tensor): True solution values at x.
        model (torch.nn.Module): Neural Network solution.
        
    Returns:
        dict: Dictionary containing L2, H1, and H2 relative errors.
    """
    
    # Ensure all inputs are column vectors (N, 1) and on the same device/dtype
    x = x.flatten().unsqueeze(-1).clone().detach().requires_grad_(True)
    u_true = u_true.flatten().unsqueeze(-1)
    u_nn = model.get_solution(x)[:, 0].unsqueeze(-1)
    
    # Compute first derivative (u'_nn)
    u_prime_nn_and_rest = torch.autograd.grad(
        outputs=u_nn, 
        inputs=x, 
        grad_outputs=torch.ones_like(u_nn), 
        create_graph=True, 
        retain_graph=True
    )
    u_prime_nn = u_prime_nn_and_rest[0]
    
    # Compute second derivative (u''_nn)
    u_double_prime_nn_and_rest = torch.autograd.grad(
        outputs=u_prime_nn, 
        inputs=x, 
        grad_outputs=torch.ones_like(u_prime_nn), 
        create_graph=False
    )
    u_double_prime_nn = u_double_prime_nn_and_rest[0]
    
    # Convert to NumPy for finite difference calculation
    x_np = x.detach().cpu().numpy().flatten()
    u_true_np = u_true.detach().cpu().numpy().flatten()
    
    # Use central finite difference (or second-order difference)
    # The domain is assumed to be uniformly sampled based on the existing script's fourier_analysis.
    
    # First derivative (u'_true): gradient is a simple NumPy finite difference
    u_prime_true_np = np.gradient(u_true_np, x_np, edge_order=2)
    
    # Second derivative (u''_true): gradient of the first derivative
    u_double_prime_true_np = np.gradient(u_prime_true_np, x_np, edge_order=2)
    
    # Convert back to Torch Tensors
    u_prime_true = torch.from_numpy(u_prime_true_np).float().to(u_true.device).unsqueeze(-1)
    u_double_prime_true = torch.from_numpy(u_double_prime_true_np).float().to(u_true.device).unsqueeze(-1)
    
    # Relative L2 Error (u)
    L2_error_num = torch.linalg.norm(u_true - u_nn, ord=2)
    L2_error_den = torch.linalg.norm(u_true, ord=2)
    L2_relative_error = (L2_error_num / L2_error_den).item()
    
    # Relative H1 Error (u and u')
    H1_error_num = torch.linalg.norm(u_prime_true - u_prime_nn, ord=2)
    H1_error_den = torch.linalg.norm(u_prime_true, ord=2)
    H1_relative_error = (H1_error_num/ H1_error_den).item()
    
    # Relative H2 Error (u, u', and u'')
    H2_error_den = torch.linalg.norm(u_double_prime_true, ord=2)
    H2_error_num = torch.linalg.norm(u_double_prime_true - u_double_prime_nn, ord=2)
    H2_relative_error = (H2_error_num / H2_error_den).item()

    return L2_relative_error, H1_relative_error, H2_relative_error
