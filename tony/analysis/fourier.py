import torch
import numpy as np
import scipy.fft as fft
import math

def calculate_fourier_coefficients(config, model, u_exact_func):
    """
    Calculates the Fourier coefficients of u_NN, u_exact, and their error based on config.

    This function is strictly designed for 1D problems.

    Returns:
    - Dict with magnitudes of selected low-frequency coeffs for u_NN, u_exact, and error (or empty dict).
    - True solution grid data for plotting (only meaningful for 1D).
    - NN solution grid data for plotting (only meaningful for 1D).
    - Evaluation points for plotting (only meaningful for 1D).
    """

    if config.domain_dim != 1:
        print(f"Fourier coefficient calculation for real basis is only supported for 1D. "
              f"Current domain dimension is {config.domain_dim}. Skipping calculation.")
        config.log_fourier_coefficients = False
        return {}, np.array([]), np.array([]), np.array([[]])

    # Ensure domain_bounds is consistently a tuple of (min, max) for 1D
    domain_bounds = config.domain_bounds
    if not isinstance(config.domain_bounds[0], (tuple, list)):
        domain_bounds = (config.domain_bounds,)

    # Generate 1D evaluation points for model inference and plotting
    grid_coords = [np.linspace(b[0], b[1], config.plot_resolution) for b in domain_bounds]
    eval_points_for_vis = grid_coords[0][:, np.newaxis]
    eval_points = torch.from_numpy(eval_points_for_vis.astype(np.float32)).to(config.device)

    # Evaluate the neural network and true solution
    with torch.no_grad():
        u_pred_vec = model(eval_points).cpu().numpy().flatten()
        u_exact_vec = u_exact_func(eval_points).cpu().numpy().flatten()
    error_vec = u_pred_vec - u_exact_vec

    # Initialize fourier_data to be empty by default
    fourier_data = {
        'nn_coeffs': [],
        'true_coeffs': [],
        'error_coeffs': [],
        'frequencies': []
    }

    if config.log_fourier_coefficients:
        # Reshape to grid for FFT (for 1D, this is just a 1D array)
        u_pred_grid = u_pred_vec.reshape(config.plot_resolution)
        u_exact_grid = u_exact_vec.reshape(config.plot_resolution)
        error_grid = error_vec.reshape(config.plot_resolution)

        N_points = config.plot_resolution

        fft_u_pred_complex = fft.fft(u_pred_grid)
        fft_u_exact_complex = fft.fft(u_exact_grid)
        fft_error_complex = fft.fft(error_grid)

        all_fourier_entries = []

        # A_0 (constant): A_0 = X_0.real / N
        A0_nn = fft_u_pred_complex[0].real / N_points
        A0_exact = fft_u_exact_complex[0].real / N_points
        A0_error = fft_error_complex[0].real / N_points
        all_fourier_entries.append({
            'k_tuple': (0,), 'type': 'cos',
            'nn_val': np.abs(A0_nn), 'exact_val': np.abs(A0_exact), 'error_val': np.abs(A0_error)
        })

        for k in config.fourier_freq:
            if k == 0:
                pass

            # A_k (cosine component): Ak = 2 * Re(Xk) / N
            Ak_nn = 2 * fft_u_pred_complex[k].real / N_points
            Ak_exact = 2 * fft_u_exact_complex[k].real / N_points
            Ak_error = 2 * fft_error_complex[k].real / N_points
            all_fourier_entries.append({
                'k_tuple': (k,), 'type': 'cos',
                'nn_val': np.abs(Ak_nn), 'exact_val': np.abs(Ak_exact), 'error_val': np.abs(Ak_error)
            })

            # B_k (sine component): Bk = -2 * Im(Xk) / N
            Bk_nn = -2 * fft_u_pred_complex[k].imag / N_points
            Bk_exact = -2 * fft_u_exact_complex[k].imag / N_points
            Bk_error = -2 * fft_error_complex[k].imag / N_points
            all_fourier_entries.append({
                'k_tuple': (k,), 'type': 'sin',
                'nn_val': np.abs(Bk_nn), 'exact_val': np.abs(Bk_exact), 'error_val': np.abs(Bk_error)
            })
        
        all_fourier_entries.sort(key=lambda x: (x['k_tuple'][0], x['type']))

        fourier_data['nn_coeffs'] = [entry['nn_val'] for entry in all_fourier_entries]
        fourier_data['true_coeffs'] = [entry['exact_val'] for entry in all_fourier_entries]
        fourier_data['error_coeffs'] = [entry['error_val'] for entry in all_fourier_entries]
        fourier_data['frequencies'] = [(entry['k_tuple'][0], entry['type']) for entry in all_fourier_entries]

    return fourier_data, u_exact_vec, u_pred_vec, eval_points_for_vis
