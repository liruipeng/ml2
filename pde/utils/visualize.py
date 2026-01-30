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
import matplotlib.pyplot as plt
import cv2
import numpy as np

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
    ax.plot(epochs, l2_errors, label="L2 Relative Error", linestyle='-', marker='o')
    ax.plot(epochs, h1_errors, label="H1 Relative Error", linestyle='--', marker='s')
    ax.plot(epochs, h2_errors, label="H2 Relative Error", linestyle=':', marker='^')
    
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
