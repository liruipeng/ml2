import os
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
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
    parser.add_argument('--adam_epochs', type=int, default=None,
                        help="Number of training epochs using Adam per sweep. Defaults to --epochs if not set.")
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

    args = parser.parse_args(args)

    # Set adam_epochs to epochs if not provided
    if args.adam_epochs is None:
        args.adam_epochs = args.epochs

    return args

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

def calculate_fourier_coefficients(u_pred, u_exact):
    pass