import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def load_plot_data(plot_data_dir):
    loaded_data = {}
    
    true_sol_files = glob.glob(os.path.join(plot_data_dir, 'true_solution_data_epoch_*.npz'))
    if true_sol_files:
        true_sol_data = np.load(true_sol_files[0])
        loaded_data['eval_points'] = true_sol_data['eval_points']
        loaded_data['u_exact_data'] = true_sol_data['u_exact_data']
    else:
        print(f"Warning: No true solution data found in {plot_data_dir}")
        return None

    nn_sol_files = sorted(glob.glob(os.path.join(plot_data_dir, 'nn_solution_data_epoch_*.npz')))
    loaded_data['nn_solution_snapshots'] = []
    loaded_data['snapshot_epochs'] = []
    for filepath in nn_sol_files:
        epoch_str = filepath.split('_epoch_')[-1].replace('.npz', '')
        epoch = int(epoch_str)
        data = np.load(filepath)['u_nn_data']
        loaded_data['nn_solution_snapshots'].append(data)
        loaded_data['snapshot_epochs'].append(epoch)
    
    return loaded_data

def plot_solution_video(base_log_dir, config_obj, rank_val, output_filename='solution_evolution.mp4'):
    """
    1. Video of the NN solution y = NN(x) (solid line) as training goes on.
       The true solution y = u(x) should be plotted (dashed line, static) as a reference.
    Supports 1D, 2D, and 3D (via slice).
    """
    plot_data_dir = os.path.join(base_log_dir, f'plots_run{rank_val+1}')
    data = load_plot_data(plot_data_dir)
    if data is None: return

    eval_points_np = data['eval_points']
    u_exact_data_flat = data['u_exact_data']
    nn_solution_snapshots = data['nn_solution_snapshots']
    snapshot_epochs = data['snapshot_epochs']
    
    dim = config_obj.domain_dim
    resolution = config_obj.plot_resolution
    
    fig = plt.figure(figsize=(10, 8))
    ax = None

    all_u_values = np.concatenate([u_exact_data_flat] + nn_solution_snapshots)
    u_min, u_max = all_u_values.min(), all_u_values.max()

    if dim == 1:
        ax = fig.add_subplot(111)
        x_coords = eval_points_np.flatten()
        
        ax.plot(x_coords, u_exact_data_flat, 'k--', label='True Solution')
        line, = ax.plot([], [], 'r-', label='NN Approximation')
        ax.set_title(f'NN Solution Evolution (1D Poisson - Case {config_obj.case_number})')
        ax.set_xlabel('x')
        ax.set_ylabel('u(x)')
        ax.legend()
        ax.grid(True)
        ax.set_ylim(u_min, u_max)
        epoch_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10)

        def animate_1d(i):
            epoch = snapshot_epochs[i]
            u_nn = nn_solution_snapshots[i]
            line.set_data(x_coords, u_nn)
            epoch_text.set_text(f'Epoch: {epoch}')
            return line, epoch_text

        ani = animation.FuncAnimation(fig, animate_1d, frames=len(snapshot_epochs), interval=100, blit=False)

    elif dim == 2:
        ax = fig.add_subplot(111, projection='3d')
        x_coords = eval_points_np[:, 0].reshape(resolution, resolution)
        y_coords = eval_points_np[:, 1].reshape(resolution, resolution)
        u_exact_reshaped = u_exact_data_flat.reshape(resolution, resolution)

        ax.plot_surface(x_coords, y_coords, u_exact_reshaped, cmap='viridis', alpha=0.5, label='True Solution')
        
        surf = ax.plot_surface(x_coords, y_coords, np.zeros_like(u_exact_reshaped), cmap='plasma', alpha=0.9)
        
        ax.set_title(f'NN Solution Evolution (2D Poisson - Case {config_obj.case_number})')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u(x,y)')
        ax.set_zlim(u_min, u_max)
        
        epoch_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=10)

        def animate_2d(i):
            nonlocal surf 
            epoch = snapshot_epochs[i]
            u_nn_reshaped = nn_solution_snapshots[i].reshape(resolution, resolution)
            
            surf.remove()
            surf = ax.plot_surface(x_coords, y_coords, u_nn_reshaped, cmap='plasma', alpha=0.9)
            
            epoch_text.set_text(f'Epoch: {epoch}')
            return [surf, epoch_text]

        ani = animation.FuncAnimation(fig, animate_2d, frames=len(snapshot_epochs), interval=100, blit=False)
    
    elif dim == 3:
        ax = fig.add_subplot(111)
        slice_dim = config_obj.slice_plane_dim
        slice_val = config_obj.slice_plane_val

        grid_coords_for_slice = [np.linspace(b[0], b[1], resolution) for b in config_obj.domain_bounds]
        slice_idx = np.argmin(np.abs(grid_coords_for_slice[slice_dim] - slice_val))
        actual_slice_val = grid_coords_for_slice[slice_dim][slice_idx]

        plot_dims = [d for d in range(dim) if d != slice_dim]
        
        if len(plot_dims) == 2:
            X_plot, Y_plot = np.meshgrid(grid_coords_for_slice[plot_dims[0]], grid_coords_for_slice[plot_dims[1]])
        else:
            raise ValueError("Unexpected dimensions for 3D slice plotting.")

        u_exact_reshaped_3d = u_exact_data_flat.reshape(resolution, resolution, resolution)
        
        if slice_dim == 0:
            u_exact_slice = u_exact_reshaped_3d[slice_idx, :, :]
        elif slice_dim == 1:
            u_exact_slice = u_exact_reshaped_3d[:, slice_idx, :]
        else:
            u_exact_slice = u_exact_reshaped_3d[:, :, slice_idx]

        c_min, c_max = u_min, u_max

        img_true = ax.imshow(u_exact_slice.T, origin='lower', extent=[config_obj.domain_bounds[plot_dims[0]][0], config_obj.domain_bounds[plot_dims[0]][1],
                                                               config_obj.domain_bounds[plot_dims[1]][0], config_obj.domain_bounds[plot_dims[1]][1]],
                             cmap='viridis', vmin=c_min, vmax=c_max, alpha=0.5)
        
        img_nn = ax.imshow(np.zeros_like(u_exact_slice.T), origin='lower', extent=[config_obj.domain_bounds[plot_dims[0]][0], config_obj.domain_bounds[plot_dims[0]][1],
                                                               config.domain_bounds[plot_dims[1]][0], config_obj.domain_bounds[plot_dims[1]][1]],
                             cmap='plasma', vmin=c_min, vmax=c_max)

        ax.set_title(f'NN Solution Evolution (3D Poisson - Case {config_obj.case_number})\nSlice at {chr(ord("x")+slice_dim)}={actual_slice_val:.2f}')
        ax.set_xlabel(f'{chr(ord("x")+plot_dims[0])}')
        ax.set_ylabel(f'{chr(ord("x")+plot_dims[1])}')
        plt.colorbar(img_nn, ax=ax, label='u value')
        epoch_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10, color='white', 
                             bbox=dict(facecolor='black', alpha=0.5))

        def animate_3d_slice(i):
            epoch = snapshot_epochs[i]
            u_nn_reshaped_3d = nn_solution_snapshots[i].reshape(resolution, resolution, resolution)
            
            if slice_dim == 0:
                u_nn_slice = u_nn_reshaped_3d[slice_idx, :, :]
            elif slice_dim == 1:
                u_nn_slice = u_nn_reshaped_3d[:, slice_idx, :]
            else:
                u_nn_slice = u_nn_reshaped_3d[:, :, slice_idx]
            
            img_nn.set_array(u_nn_slice.T)
            epoch_text.set_text(f'Epoch: {epoch}')
            return [img_nn, epoch_text]

        ani = animation.FuncAnimation(fig, animate_3d_slice, frames=len(snapshot_epochs), interval=100, blit=False)

    else:
        print(f"Solution visualization not supported for dimension {dim}")
        plt.close(fig)
        return

    output_path = os.path.join(base_log_dir, output_filename)
    print(f"Saving animation to {output_path}...")
    try:
        ani.save(output_path, writer='ffmpeg', dpi=150)
    except Exception as e:
        print(f"Error saving video: {e}. Make sure ffmpeg is installed and in your PATH.")
    
    plt.close(fig)
    print("Animation saved.")
    
def plot_fourier_coefficients(base_log_dir, history_data, config_obj, output_filename='fourier_coefficients_evolution.png'):
    """
    2. Plot of the Fourier coefficients (solid line) as a function of epoch.
       Use different colors for different frequency. The true coefficients should be plotted (dashed line)
       with corresponding color.
    """
    epochs_logged = history_data['epochs_logged']
    frequencies_logged = history_data['fourier_frequencies_logged']
    
    fig = plt.figure(figsize=(12, 7))
    cmap = plt.get_cmap('tab10')

    plot_count = 0

    for i, freq_idx_tuple in enumerate(frequencies_logged):
        freq_key = str(freq_idx_tuple)
        nn_coeffs = history_data['fourier_coeffs_nn_magnitudes'].get(freq_key, [])
        true_coeffs = history_data['fourier_coeffs_true_magnitudes'].get(freq_key, [])

        if not nn_coeffs or not true_coeffs:
            continue

        color = cmap(i % cmap.N)
        
        if freq_idx_tuple[0] == 0:
            label_nn = 'NN Coeff for $A_0$ (cos)'
            label_true = 'True Coeff for $A_0$ (cos)'
        elif freq_idx_tuple[1] == 'cos':
            label_nn = f'NN Coeff for $A_{{{freq_idx_tuple[0]}}}$ (cos)'
            label_true = f'True Coeff for $A_{{{freq_idx_tuple[0]}}}$ (cos)'
        elif freq_idx_tuple[1] == 'sin':
            label_nn = f'NN Coeff for $B_{{{freq_idx_tuple[0]}}}$ (sin)'
            label_true = f'True Coeff for $B_{{{freq_idx_tuple[0]}}}$ (sin)'
        else:
            label_nn = f'NN Coeff for Freq {freq_idx_tuple}'
            label_true = f'True Coeff for Freq {freq_idx_tuple}'

        plt.plot(epochs_logged, nn_coeffs, solid_capstyle='projecting', 
                 color=color, label=label_nn)
        plt.plot(epochs_logged, true_coeffs, linestyle='--', color=color, 
                 label=label_true)
        plot_count += 1

    plt.title(f'Fourier Coefficient Magnitudes vs. Epoch (Dim: {config_obj.domain_dim}, Case: {config_obj.case_number})')
    plt.xlabel('Epoch')
    plt.ylabel('Coefficient Magnitude')
    plt.yscale('log')
    
    if plot_count > 0:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    else:
        print("Warning: No Fourier coefficients plotted for legend in plot_fourier_coefficients.")

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(base_log_dir, output_filename))
    plt.close(fig)


def plot_fourier_coefficient_errors(base_log_dir, history_data, config_obj, output_filename='fourier_coefficient_errors_evolution.png'):
    """
    3. Plot of the error of Fourier coefficients (solid line) as a function of epoch.
       Use different colors for different frequency.
    """
    epochs_logged = history_data['epochs_logged']
    frequencies_logged = history_data['fourier_frequencies_logged']
    
    fig = plt.figure(figsize=(12, 7))
    cmap = plt.get_cmap('tab10')

    plot_count = 0

    for i, freq_idx_tuple in enumerate(frequencies_logged):
        freq_key = str(freq_idx_tuple)
        error_coeffs = history_data['fourier_coeffs_error_magnitudes'].get(freq_key, [])
        
        if not error_coeffs:
            continue

        color = cmap(i % cmap.N)
        
        if freq_idx_tuple[0] == 0:
            label_error = f'Error Coeff for $A_0$ (cos)'
        elif freq_idx_tuple[1] == 'cos':
            label_error = f'Error Coeff for $A_{{{freq_idx_tuple[0]}}}$ (cos)'
        elif freq_idx_tuple[1] == 'sin':
            label_error = f'Error Coeff for $B_{{{freq_idx_tuple[0]}}}$ (sin)'
        else:
            label_error = f'Error Coeff for Freq {freq_idx_tuple}'

        plt.plot(epochs_logged, error_coeffs, solid_capstyle='projecting', 
                 color=color, label=label_error)
        plot_count += 1

    plt.title(f'Fourier Coefficient Error Magnitudes vs. Epoch (Dim: {config_obj.domain_dim}, Case: {config_obj.case_number})')
    plt.xlabel('Epoch')
    plt.ylabel('Error Magnitude')
    plt.yscale('log')
    
    if plot_count > 0:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    else:
        print("Warning: No Fourier coefficient errors plotted for legend in plot_fourier_coefficient_errors.")

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(base_log_dir, output_filename))
    plt.close(fig)


def plot_norm_errors(base_log_dir, history_data, config_obj, output_filename='norm_errors_evolution.png'):
    """
    4. Plot of the L2, H1, and H2 norm of the NN approximation error.
       Plot each of these as a function of epoch.
    """
    epochs_logged = history_data['epochs_logged']
    
    fig = plt.figure(figsize=(12, 7))
    plot_count = 0

    if 'l2_error_u' in history_data and history_data['l2_error_u']:
        plt.plot(epochs_logged, history_data['l2_error_u'], label='$L^2$ Error', color='blue')
        plot_count += 1
    if 'h1_seminorm_error_u' in history_data and history_data['h1_seminorm_error_u']:
        plt.plot(epochs_logged, history_data['h1_seminorm_error_u'], label='$H^1$ Seminorm Error', color='green')
        plot_count += 1
    if 'h2_seminorm_error_u' in history_data and history_data['h2_seminorm_error_u']:
        plt.plot(epochs_logged, history_data['h2_seminorm_error_u'], label='$H^2$ Seminorm Error', color='red')
        plot_count += 1

    plt.title(f'Approximation Norm Errors vs. Epoch (Dim: {config_obj.domain_dim}, Case: {config_obj.case_number})')
    plt.xlabel('Epoch')
    plt.ylabel('Error Magnitude')
    plt.yscale('log')
    
    if plot_count > 0:
        plt.legend()
    else:
        print("Warning: No norm errors plotted for legend in plot_norm_errors.")

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(base_log_dir, output_filename))
    plt.close(fig)


def plot_ensemble_norm_errors(base_log_dir, ensemble_histories, config_obj, output_filename='ensemble_norm_errors.png'):
    """
    Plots the L2, H1, H2 norm errors for an ensemble of runs, showing median and interquartile range (IQR).
    """
    fig = plt.figure(figsize=(14, 8))
    
    colors = ['blue', 'green', 'red']
    labels = ['$L^2$ Error', '$H^1$ Seminorm Error', '$H^2$ Seminorm Error']
    error_keys = ['l2_error_u', 'h1_seminorm_error_u', 'h2_seminorm_error_u']

    if not ensemble_histories or not ensemble_histories[0]['epochs_logged']:
        print("No logged epochs or ensemble histories found for plotting ensemble errors.")
        plt.close(fig)
        return
    epochs_logged = ensemble_histories[0]['epochs_logged']
    
    agg_data = {key: {'median': [], 'q25': [], 'q75': []} for key in error_keys}

    for epoch_idx in range(len(epochs_logged)):
        for key in error_keys:
            current_epoch_values = []
            for history_data in ensemble_histories:
                if key in history_data and len(history_data[key]) > epoch_idx:
                    current_epoch_values.append(history_data[key][epoch_idx])
            
            if current_epoch_values:
                values_array = np.array(current_epoch_values)
                agg_data[key]['median'].append(np.percentile(values_array, 50))
                agg_data[key]['q25'].append(np.percentile(values_array, 25))
                agg_data[key]['q75'].append(np.percentile(values_array, 75))
            else:
                agg_data[key]['median'].append(np.nan)
                agg_data[key]['q25'].append(np.nan)
                agg_data[key]['q75'].append(np.nan)

    plot_count = 0

    for i, key in enumerate(error_keys):
        median_vals = np.array(agg_data[key]['median'])
        q25_vals = np.array(agg_data[key]['q25'])
        q75_vals = np.array(agg_data[key]['q75'])

        if not np.all(np.isnan(median_vals)):
            plt.plot(epochs_logged, median_vals, 
                     color=colors[i], linestyle='-', linewidth=2, 
                     label=f'{labels[i]} (Median)')
            
            plt.fill_between(epochs_logged, q25_vals, q75_vals, 
                             color=colors[i], alpha=0.2, 
                             label=f'{labels[i]} (25th-75th Pctl Range)')
            plot_count += 1

    plt.title(f'Ensemble Approximation Norm Errors vs. Epoch (Dim: {config_obj.domain_dim}, Case: {config_obj.case_number})')
    plt.xlabel('Epoch')
    plt.ylabel('Error Magnitude')
    plt.yscale('log')
    
    if plot_count > 0:
        plt.legend(loc='upper right')
    else:
        print("Warning: No ensemble norm errors plotted for legend in plot_ensemble_norm_errors.")

    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(os.path.join(base_log_dir, output_filename))
    plt.close(fig)
