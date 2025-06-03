import os
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import Path
import torch
import scipy.fft as fft
import shutil


def cleanfiles(dir_name):
    dir_path = Path(dir_name)
    if dir_path.exists() and dir_path.is_dir():
        for item in dir_path.iterdir():
            if item.is_file():
                item.unlink()
            # elif item.is_dir():
            #    shutil.rmtree(item)

def is_notebook():
    try:
        from IPython import get_ipython
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except:
        return False

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Train a PINN model.")

    parser.add_argument('--nx', type=int, default=128,
                        help="Number of training points in the 1D mesh.")
    parser.add_argument('--nx_eval', type=int, default=256,
                        help="Number of evaluation points in the 1D mesh.")
    parser.add_argument('--num_checks', type=int, default=20,
                        help="Number of evaluation checkpoints during training.")
    parser.add_argument('--num_plots', type=int, default=10,
                        help="Number of plotting points during training.")
    parser.add_argument('--epochs', type=int, default=10000,
                        help="Number of training epochs per sweep.")
    parser.add_argument('--sweeps', type=int, default=3,
                        help="Number of multilevel outer sweeps.")
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[64, 64],
                        help="List of hidden layer dimensions (e.g., --hidden_dims 64 64)")
    parser.add_argument('--ax', type=float, default=0.0,
                        help="Lower bound of the 1D domain.")
    parser.add_argument('--bx', type=float, default=1.0,
                        help="Upper bound of the 1D domain.")
    parser.add_argument('--high_freq', type=int, default=16,
                        help="Highest frequency used in the PDE solution (PDE 1).")
    parser.add_argument('--gamma', type=float, default=0,
                        help="Coefficient \gamma in the PDE: -u_xx + \gamma u = f.")
    parser.add_argument('--mu', type=float, default=70,
                        help="Oscillation parameter in the solution (PDE 2).")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Learning rate for the optimizer.")
    parser.add_argument('--levels', type=int, default=4,
                        help="Number of levels in multilevel training.")
    parser.add_argument('--loss_type', type=int, default=0, choices=[-1, 0],
                        help="Loss type: -1 for supervised (true solution), 0 for PINN loss.")
    parser.add_argument('--activation', type=str, default='tanh',
                        choices=['tanh', 'silu', 'relu', 'gelu', 'softmax'],
                        help="Activation function to use.")
    parser.add_argument('--plot', action='store_true',
                        help="If set, generate plots during or after training.")
    parser.add_argument('--no-clear', action='store_false', dest='clear',
                        help="If set, do not remove plot files generated before.")
    parser.add_argument('--problem_id', type=int, default=1, choices=[1, 2],
                        help="PDE problem to solve: 1 or 2.")
    parser.add_argument('--enforce_bc', action='store_true',
                        help="If set, enforce the BC in solution.")
    parser.add_argument('--bc_weight', type=float, default=1.0,
                        help="Weight for the loss of BC.")

    return parser.parse_args(args)


def print_args(args):
    print("Options used:")
    for key, value in vars(args).items():
        print(f"   --{key}: {value}")


def get_activation(name: str):
    name = name.lower()
    activations = {
        'tanh': nn.Tanh,
        'relu': nn.ReLU,
        'silu': nn.SiLU,
        'gelu': nn.GELU,
        'softmax': lambda: nn.Softmax(dim=1),  # safer default
    }
    if name not in activations:
        raise ValueError(f"Unknown activation function: {name}")
    return activations[name]()

def save_frame(x, t, y, xs, ys, iteration, title, frame_dir):
    """_summary_

    Args:
        x (_type_): points in x to plot
        t (_type_): true solution to plot
        xs: pointx in x to scatter
        ys (_type_): values to scatter
        y (_type_): solution to plot
        iteration (int): _description_
        title (str): _description_
        frame_dir (str): _description_
    """
    fig, ax = plt.subplots()
    if t is not None:
        ax.plot(x, t, label="Exact", linestyle='-', color="black")
    if y is not None:
        ax.plot(x, y, label=f"NN: step {iteration}", color="blue")
    if xs is not None and ys is not None:
        ax.scatter(xs, ys, color="red", label="Sample training points")
    ax.set_title(title)
    ax.legend(loc='upper right')
    iters_str = "_".join(f"{i:04d}" for i in iteration)
    frame_path = os.path.join(frame_dir, f"{title}_{iters_str}.png")
    fig.savefig(frame_path)
    plt.close(fig)


def make_video_from_frames(frame_dir, name_prefix, output_file, fps=10):
    frame_paths = sorted([
        os.path.join(frame_dir, fname)
        for fname in os.listdir(frame_dir)
        if fname.endswith(".png") and fname.startswith(name_prefix)
    ])
    if not frame_paths:
        print("No frames found.")
        return
    frame = cv2.imread(frame_paths[0])
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_file_path = os.path.join(frame_dir, output_file)
    video = cv2.VideoWriter(output_file_path, fourcc, fps, (width, height))
    for path in frame_paths:
        img = cv2.imread(path)
        video.write(img)
    video.release()
    print(f"  Video saved as {output_file_path}")

def calculate_fourier_coefficients_1d(model, mesh, plot_resolution:int=100, fourier_freq:list[int]=[1, 4, 9], log_fourier_coefficients=True, device='cpu'):
    """
    Calculates the Fourier coefficients of u_NN, u_exact, and their error based on config.

    This function is strictly designed for 1D problems.

    Returns:
    - Dict with magnitudes of selected low-frequency coeffs for u_NN, u_exact, and error (or empty dict).
    - True solution grid data for plotting (only meaningful for 1D).
    - NN solution grid data for plotting (only meaningful for 1D).
    - Evaluation points for plotting (only meaningful for 1D).
    Ref: 
        - https://github.com/liruipeng/ml2/blob/5956d7a54badf8ea966053c511e23f5fc8fcc9f2/tony/analysis/fourier.py
    """
    domain_bounds = (mesh.ax, mesh.bx)
    # Generate 1D evaluation points for model inference and plotting
    grid_coords = np.linspace(*domain_bounds, plot_resolution)
    eval_points_for_vis = grid_coords[:, np.newaxis]
    eval_points = torch.from_numpy(eval_points_for_vis.astype(np.float32)).to(device)

    # Evaluate the neural network and true solution
    model.eval()
    with torch.no_grad():
        u_pred_vec = model(eval_points).cpu().numpy().flatten()
        u_exact_vec = mesh.pde.u_ex(eval_points).cpu().numpy().flatten()
    error_vec = u_pred_vec - u_exact_vec

    # Initialize fourier_data to be empty by default
    fourier_data = {
        'nn_coeffs': [],
        'true_coeffs': [],
        'error_coeffs': [],
        'frequencies': []
    }

    if log_fourier_coefficients:
            # Reshape to grid for FFT (for 1D, this is just a 1D array)
            u_pred_grid = u_pred_vec.reshape(plot_resolution)
            u_exact_grid = u_exact_vec.reshape(plot_resolution)
            error_grid = error_vec.reshape(plot_resolution)

            N_points = plot_resolution

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

            for k in fourier_freq:
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