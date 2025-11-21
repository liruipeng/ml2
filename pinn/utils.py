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
import os
import argparse
import inspect
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from scipy.fft import rfft, rfftfreq, dst
import numpy as np
import torch
import ast


# %%
def cleanfiles(dir_name):
    dir_path = Path(dir_name)
    if dir_path.exists() and dir_path.is_dir():
        for item in dir_path.iterdir():
            if item.is_file():
                item.unlink()
            # elif item.is_dir():
            #    shutil.rmtree(item)


# %%
def is_notebook():
    try:
        from IPython import get_ipython
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except (ImportError, AttributeError):
        return False


# %%
def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Train a PINN model.")

    parser.add_argument('--nx', type=int, nargs='+', default=128,
                        help="Number of training points in the 1D mesh.")
    parser.add_argument('--nx_eval', type=int, default=256,
                        help="Number of evaluation points in the 1D mesh.")
    parser.add_argument('--num_checks', type=int, default=20,
                        help="Number of evaluation checkpoints during training.")
    parser.add_argument('--num_plots', type=int, default=10,
                        help="Number of plotting points during training.")
    parser.add_argument('--epochs', type=int, nargs='+', default=10000,
                        help="Number of training epochs per sweep.")
    parser.add_argument('--adam_epochs', type=int, default=None,
                        help="Number of training epochs using Adam per sweep. Defaults to --epochs if not set.")
    parser.add_argument('--sweeps', type=int, default=2,
                        help="Number of multilevel outer sweeps.")
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[64, 64],
                        help="List of hidden layer dimensions (e.g., --hidden_dims 64 64)")
    parser.add_argument('--ax', type=float, default=0.0,
                        help="Lower bound of the 1D domain.")
    parser.add_argument('--bx', type=float, default=1.0,
                        help="Upper bound of the 1D domain.")
    parser.add_argument('--high_freq', type=int, default=8,
                        help="Highest frequency used in the PDE solution (PDE 1).")
    parser.add_argument('--gamma', type=float, default=0,
                        help="Coefficient γ in the PDE: -uₓₓ + γ u = f.")
    parser.add_argument('--mu', type=float, default=70,
                        help="Oscillation parameter in the solution (PDE 2).")
    parser.add_argument('--lr', type=float, nargs='+', default=1e-3,
                        help="Learning rate for the optimizer.")
    parser.add_argument('--levels', type=int, default=4,
                        help="Number of levels in multilevel training.")
    parser.add_argument('--loss_type', type=int, default=0, choices=[-1, 0, 1, 2],
                        help="Loss type: -1 for supervised (true solution), 0 for PINN loss, 1 for DRM loss, 2 for mixed.")
    parser.add_argument('--activation', type=str, default='tanh',
                        choices=['tanh', 'silu', 'relu', 'gelu', 'softmax'],
                        help="Activation function to use.")
    parser.add_argument('--enforce_bc', action='store_true',
                        help="If set, enforce the BC in solution.")
    parser.add_argument('--bc_extension', type=str, default='hermite_cubic_2nd_deriv', 
                        choices=['multilinear', 'hermite_cubic_2nd_deriv'],
                        help='Boundary value extension function.')
    parser.add_argument('--distance', type=str, default='sin_half_period', 
                        choices=['quadratic_bubble', 'inf_smooth_bump', 'abs_dist_complement', 'ratio_bubble_dist', 'sin_half_period'],
                        help='Distance function.')
    parser.add_argument('--use_chebyshev_basis', action='store_true',
                        help="If set, use Chebyshev features.")
    parser.add_argument('--chebyshev_freq_min', type=int, nargs='+',
                        help='Minimum frequency for Chebyshev polynomials.')
    parser.add_argument('--chebyshev_freq_max', type=int, nargs='+',
                        help='Maximum frequency for Chebyshev polynomials.')
    parser.add_argument('--track_freqs', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7],
                    help="Integer array of frequencies (modes) whose coefficients will be tracked and plotted over epochs.")
    parser.add_argument('--plot', action='store_true',
                        help="If set, generate plots during or after training.")
    parser.add_argument('--no-clear', action='store_false', dest='clear',
                        help="If set, do not remove plot files generated before.")
    parser.add_argument('--problem_id', type=int, default=1, choices=[1, 2],
                        help="PDE problem to solve: 1 or 2.")
    parser.add_argument('--bc_weight', type=float, default=1.0,
                        help="Weight for the loss of BC.")
    parser.add_argument("--scheduler", type=str, default="StepLR",
                        help="Learning rate scheduler to use. "
                        "See https://docs.pytorch.org/docs/stable/optim.html for full list of schedulers")
    parser.add_argument("--scheduler_config", type=str, nargs='+',
                        default=["step_size", "1000", "gamma", "0.9"],
                        help="Configuration for learning rate scheduler. "
                        "Follow https://docs.pytorch.org/docs/stable/optim.html for full list of schedulers. "
                        "The setting is corresponding to `--scheduler` setting.")

    args = parser.parse_args(args)

    # Set adam_epochs to epochs if not provided
    if args.adam_epochs is None:
        args.adam_epochs = args.epochs

    return args


# %%
def str2arg(str_info: str):
    """
    Convert string to argument with ast.literal_eval.
    Noted that 'min' functions can cause `ValueError: malformed node or string on`.
    This return original string if conversion fails.
    """
    try:
        arg = ast.literal_eval(str_info)
    except (ValueError):
        arg = str_info
    return arg


# %%
def get_scheduler_generator(args):
    """
    Return scheduler generator by argument
    """
    scheduler_name = args.scheduler
    scheduler_kargs = {k: str2arg(v) for k, v in zip(args.scheduler_config[::2], args.scheduler_config[1::2])}
    scheduler_attr = getattr(torch.optim.lr_scheduler, scheduler_name)

    def scheduler_generator(optimizer: torch.optim) -> torch.optim.lr_scheduler:
        """
        Return scheduler from optimizer
        Args:
            optimizer (torch.optim): Optimizer to attach the scheduler to.
        """
        return scheduler_attr(optimizer, **scheduler_kargs)
    return scheduler_generator


# %%
def scheduler_step(scheduler, loss, epoch=None):
    """
    Wrapper function for step the learning rate scheduler.
    """
    func_args = inspect.getfullargspec(scheduler.step).args
    # ReduceLROnPlateau scheduler requires metrics on stepping
    if "metrics" in func_args:
        scheduler.step(metrics=loss, epoch=epoch)
    else:
        # Otherwise, call step without metrics
        scheduler.step(epoch=epoch)


# %%
def print_args(args, output_file=None):
    if output_file:
        f = open(output_file, 'w')
    else:
        f = None
    print("Options used:")
    for key, value in vars(args).items():
        print(f"   --{key}: {value}")
        if f:
            print(f"   --{key}: {value}", file=f)

# %%
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

def plot_coefficient_evolution(data: list, freqs: list, sweep_idx: int, level_idx: int, frame_dir: str, analytic_freqs: np.ndarray):
    """
    Plots the evolution of specific Fourier/Sine coefficients over training epochs.
    """
    if not data:
        print("No coefficient data collected for plotting.")
        return

    # Extract all data into a structured format
    epochs = np.array([d['epoch'] for d in data])
    true_coeffs = np.array([d['true_coeffs'] for d in data])
    nn_coeffs = np.array([d['nn_coeffs'] for d in data])
    
    num_freqs = len(freqs)
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_title(f"Sweep {sweep_idx}, Level {level_idx}: Fourier Coefficient Evolution")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Coefficient Magnitude")

    for i in range(num_freqs):
        # Plot true coefficient (should be constant)
        ax1.plot(epochs, true_coeffs[:, i], 
                 label=f"True (Freq = {freqs[i]} pi)", 
                 linestyle='--', alpha=0.7)
        # Plot NN coefficient evolution
        ax1.plot(epochs, nn_coeffs[:, i], 
                 label=f"NN (Freq = {freqs[i]} pi)", 
                 linestyle='-')

    ax1.legend(loc='best')
    
    iters_str = f"Sweep{sweep_idx:02d}_Lvl{level_idx:02d}"
    filename1 = os.path.join(frame_dir, f"Coeffs_Evolution_{iters_str}.png")
    fig1.savefig(filename1)
    plt.close(fig1)

    print(f"  Coefficient evolution plot saved to {filename1}")

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.set_title(f"Sweep {sweep_idx}, Level {level_idx}: Coefficient Error Evolution")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Absolute Error (|True - NN|)")

    error_coeffs = np.abs(true_coeffs - nn_coeffs)

    for i in range(num_freqs):
        ax2.plot(epochs, error_coeffs[:, i], 
                 label=f"Freq = {freqs[i]} pi", 
                 linestyle='-')
        
    ax2.legend(loc='best')
    ax2.set_yscale('log')
    
    filename2 = os.path.join(frame_dir, f"Coeffs_Error_Evolution_{iters_str}.png")
    fig2.savefig(filename2)
    plt.close(fig2)

    print(f"  Coefficient error plot saved to {filename2}")

# %%
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
    iters_str = "_".join(f"{i:08d}" for i in iteration)
    frame_path = os.path.join(frame_dir, f"{title}_{iters_str}.png")
    fig.savefig(frame_path)
    plt.close(fig)


# %%
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


# %%
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

# %%
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

# %%
def plot_error_evolution(data: list, sweep_idx: int, level_idx: int, frame_dir: str):
    """
    Plots the evolution of L2, H1, and H2 relative errors over training epochs.

    Args:
        data (list): List of dictionaries, each containing 'epoch' and error metrics.
        sweep_idx (int): Current sweep index for file naming.
        level_idx (int): Current level index for file naming.
        frame_dir (str): Directory to save the plot.
    """
    if not data:
        print("No error data collected for plotting.")
        return

    # Extract all data into structured numpy arrays
    epochs = np.array([d['epoch'] for d in data])
    l2_errors = np.array([d['L2_error'] for d in data])
    h1_errors = np.array([d['H1_error'] for d in data])
    h2_errors = np.array([d['H2_error'] for d in data])

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the three error metrics
    ax.plot(epochs, l2_errors, label="L2 Relative Error", linestyle='-', marker='o', markevery=len(epochs)//10)
    ax.plot(epochs, h1_errors, label="H1 Relative Error", linestyle='--', marker='s', markevery=len(epochs)//10)
    ax.plot(epochs, h2_errors, label="H2 Relative Error", linestyle=':', marker='^', markevery=len(epochs)//10)
    
    ax.set_title(f"Sweep {sweep_idx}, Level {level_idx}: Solution Error Evolution")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Relative Error (Log Scale)")
    
    # Set the y-axis to a logarithmic scale, as errors typically span several orders of magnitude
    ax.set_yscale('log')
    ax.legend(loc='best')
    ax.grid(True, which="both", ls="--", linewidth=0.5)

    iters_str = f"Sweep{sweep_idx:02d}_Lvl{level_idx:02d}"
    filename = os.path.join(frame_dir, f"Error_Evolution_{iters_str}.png")
    fig.savefig(filename)
    plt.close(fig)

    print(f"  Error evolution plot saved to {filename}")
