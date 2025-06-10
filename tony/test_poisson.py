import argparse
import datetime
import os
import json
import time
import torch
import numpy as np
import torch.optim as optim
from mpi4py import MPI

from config.poisson import PoissonSolverConfig
from pde.poisson import get_manufactured_solution
from model.nn import NNModel
from model.bc import get_g0_func, get_d_func
from loss.poisson import calculate_drm_loss, calculate_pinn_loss
from data.samplers import generate_uniform_grid_points
from analysis.fourier import calculate_fourier_coefficients
from utils.log import ExperimentLogger
from utils.visualize import plot_ensemble_norm_errors, plot_solution_video, plot_fourier_coefficients, plot_fourier_coefficient_errors, plot_norm_errors

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Helper function for evaluation, logging, and saving
def evaluate_and_log(epoch, u_nn_model, history, config, f_exact_func, u_exact_func, logger, rank_val,
                     eval_points_for_plot_np, u_exact_plot_data_flat,
                     full_uniform_grid_points):
    u_nn_model.eval()

    eval_points_for_errors = full_uniform_grid_points.requires_grad_(True).to(config.device)

    # Calculate losses
    drm_loss = calculate_drm_loss(u_nn_model, f_exact_func, eval_points_for_errors)
    pinn_loss = calculate_pinn_loss(u_nn_model, f_exact_func, eval_points_for_errors, config.domain_dim)
    total_loss = config.drm_weight * drm_loss + config.pinn_weight * pinn_loss

    if epoch == 0:
        history['total_loss'].append(total_loss.item())
        history['drm_loss'].append(drm_loss.item())
        history['pinn_loss'].append(pinn_loss.item())

    logger.log_scalar('Loss/Total_Loss', total_loss.item(), step=epoch)
    logger.log_scalar('Loss/DRM_Loss', drm_loss.item(), step=epoch)
    logger.log_scalar('Loss/PINN_Loss', pinn_loss.item(), step=epoch)

    # L2 norm error
    u_pred_eval_l2 = u_nn_model(eval_points_for_errors)
    u_exact_eval_l2 = u_exact_func(eval_points_for_errors).detach()
    l2_error_u = torch.mean((u_pred_eval_l2 - u_exact_eval_l2)**2)

    # H1 seminorm error
    eval_points_for_derivs = eval_points_for_errors.clone().detach().requires_grad_(True)
    u_pred_for_derivs = u_nn_model(eval_points_for_derivs)
    u_exact_for_derivs = u_exact_func(eval_points_for_derivs)
    grad_u_pred_eval = torch.autograd.grad(u_pred_for_derivs, eval_points_for_derivs,
                                            grad_outputs=torch.ones_like(u_pred_for_derivs),
                                            create_graph=True, allow_unused=True)[0]
    grad_u_exact_eval = torch.autograd.grad(u_exact_for_derivs, eval_points_for_derivs,
                                            grad_outputs=torch.ones_like(u_exact_for_derivs),
                                            create_graph=True, allow_unused=True)[0]
    if grad_u_pred_eval is None: grad_u_pred_eval = torch.zeros_like(eval_points_for_derivs)
    if grad_u_exact_eval is None: grad_u_exact_eval = torch.zeros_like(eval_points_for_derivs)

    h1_seminorm_error_u = torch.mean(torch.sum((grad_u_pred_eval - grad_u_exact_eval)**2, dim=1))

    # H2 seminorm error
    laplacian_u_pred_eval = torch.zeros_like(u_pred_for_derivs, device=config.device)
    for i in range(config.domain_dim):
        d2u_dxi2 = torch.autograd.grad(grad_u_pred_eval[:, i], eval_points_for_derivs,
                                        grad_outputs=torch.ones_like(grad_u_pred_eval[:, i]),
                                        create_graph=True, allow_unused=True)[0][:, i]
        laplacian_u_pred_eval += d2u_dxi2.unsqueeze(1)

    laplacian_u_exact_eval = torch.zeros_like(u_exact_for_derivs, device=config.device)
    for i in range(config.domain_dim):
        d2u_dxi2_star = torch.autograd.grad(grad_u_exact_eval[:, i], eval_points_for_derivs,
                                            grad_outputs=torch.ones_like(grad_u_exact_eval[:, i]),
                                            create_graph=False, allow_unused=True)[0][:, i]
        laplacian_u_exact_eval += d2u_dxi2_star.unsqueeze(1)

    h2_seminorm_error_u = torch.mean((laplacian_u_pred_eval - laplacian_u_exact_eval)**2)

    eval_points_for_errors.requires_grad_(False)
    eval_points_for_derivs.requires_grad_(False)

    history['epochs_logged'].append(epoch)
    history['l2_error_u'].append(l2_error_u.item())
    history['h1_seminorm_error_u'].append(h1_seminorm_error_u.item())
    history['h2_seminorm_error_u'].append(h2_seminorm_error_u.item())

    # Fourier coefficient
    if config.domain_dim != 1:
        print(f"Fourier coefficient calculation for real basis is only supported for 1D. "
              f"Current domain dimension is {config.domain_dim}. Skipping calculation.")
        config.log_fourier_coefficients = False

    if config.log_fourier_coefficients:
        fourier_data, u_exact_plot_data_flat_current, u_pred_plot_data_flat_current, eval_points_for_plot_np_current = \
            calculate_fourier_coefficients(config, u_nn_model, u_exact_func)

        if history['fourier_frequencies_logged'] is None:
            history['fourier_frequencies_logged'] = fourier_data['frequencies']
            for freq_idx_tuple in history['fourier_frequencies_logged']:
                freq_key = str(freq_idx_tuple)
                history['fourier_coeffs_nn_magnitudes'][freq_key] = []
                history['fourier_coeffs_true_magnitudes'][freq_key] = []
                history['fourier_coeffs_error_magnitudes'][freq_key] = []

        for i, freq_idx_tuple in enumerate(history['fourier_frequencies_logged']):
            freq_key = str(freq_idx_tuple)
            history['fourier_coeffs_nn_magnitudes'][freq_key].append(fourier_data['nn_coeffs'][i])
            history['fourier_coeffs_true_magnitudes'][freq_key].append(fourier_data['true_coeffs'][i])
            history['fourier_coeffs_error_magnitudes'][freq_key].append(fourier_data['error_coeffs'][i])

            logger.log_scalar(f'Fourier/NN_Coeff_Magnitude_Freq_{freq_key}', fourier_data['nn_coeffs'][i], step=epoch)
            logger.log_scalar(f'Fourier/True_Coeff_Magnitude_Freq_{freq_key}', fourier_data['true_coeffs'][i], step=epoch)
            logger.log_scalar(f'Fourier/Error_Coeff_Magnitude_Freq_{freq_key}', fourier_data['error_coeffs'][i], step=epoch)

        # Save true solution data ONCE per run
        if eval_points_for_plot_np is None:
            eval_points_for_plot_np = eval_points_for_plot_np_current
            u_exact_plot_data_flat = u_exact_plot_data_flat_current
            logger.save_plot_data({'eval_points': eval_points_for_plot_np, 'u_exact_data': u_exact_plot_data_flat},
                                   'true_solution_data', epoch)
            logger.log_message(f"Saved true solution and evaluation points for plotting.")

        # Save NN solution snapshots
        if u_pred_plot_data_flat_current is not None and u_pred_plot_data_flat_current.size > 0:
            logger.save_plot_data({'u_nn_data': u_pred_plot_data_flat_current}, 'nn_solution_data', epoch)
            history['solution_snapshots_epochs'].append(epoch)
        else:
            logger.log_message(f"Warning: u_pred_plot_data_flat_current is empty/None, skipping saving nn_solution_data for epoch {epoch}.")

    current_lr = None
    if epoch == 0:
        # For initial eval, use the initial learning rate from config
        current_lr = config.learning_rate
        logger.log_scalar('Learning_Rate', current_lr, step=epoch)
    elif hasattr(u_nn_model, 'optimizer') and u_nn_model.optimizer is not None and u_nn_model.optimizer.param_groups:
        # Only get current LR if optimizer is initialized
        current_lr = u_nn_model.optimizer.param_groups[0]['lr']
        logger.log_scalar('Learning_Rate', current_lr, step=epoch)

    # Print to console (each rank prints its own status)
    print_str = f"Rank {rank_val}, Epoch {epoch}/{config.num_epochs}, " \
                f"L2 Error: {l2_error_u.item():.6e}, " \
                f"H1 Error: {h1_seminorm_error_u.item():.6e}, " \
                f"H2 Error: {h2_seminorm_error_u.item():.6e}, " \
                f"Total Loss: {total_loss.item():.6e}, " \
                f"DRM Loss: {drm_loss.item():.6e}, " \
                f"PINN Loss: {pinn_loss.item():.6e}"
    if current_lr is not None:
        print_str += f", LR: {current_lr:.2e}"
    print(print_str)

    return eval_points_for_plot_np, u_exact_plot_data_flat


# --- Main Training Loop for a Single Experiment Run ---
def run_experiment(config: PoissonSolverConfig, rank_val):
    print(f"Rank {rank_val}: Initializing experiment")

    # Each rank has its own random seed, config file, and model sub-directory
    config.random_seed += rank_val
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)

    config_dict = config.to_dict()
    logger = ExperimentLogger(config.base_log_dir, rank_val)
    logger.log_message(f"\n--- Experiment Configuration (Rank {rank_val}/{size-1}) ---")
    for key, value in config_dict.items():
        logger.log_message(f"  {key}: {value}")
    logger.log_message("------------------------------\n")
    logger.save_config(config_dict)

    rank_model_path = os.path.join(config.base_model_save_dir, f"run{rank_val+1}")
    os.makedirs(rank_model_path, exist_ok=True)

    # PDE & Boundary Functions
    u_exact_func, f_exact_func = get_manufactured_solution(config.case_number, config.domain_dim)
    g0_func = get_g0_func(u_exact_func, config.domain_dim, config.domain_bounds)
    d_func = get_d_func(config.domain_dim, config.domain_bounds)

    # Uniform grid
    full_uniform_grid_points = generate_uniform_grid_points(
        config.domain_bounds, config.num_uniform_partition
    ).to(config.device)
    total_domain_points_in_grid = full_uniform_grid_points.shape[0]
    logger.log_message(f"Total domain points in grid: {total_domain_points_in_grid}")

    # NN Model
    u_nn_model = NNModel(
        input_dim=config.domain_dim,
        output_dim=1,
        hidden_neurons=config.hidden_neurons,
        activation=config.activation,
        g0_func=g0_func,
        d_func=d_func
    ).to(config.device)

    # Optimizer and Learning Rate Scheduler
    optimizer_class = getattr(optim, config.optimizer_type)
    optimizer = optimizer_class(u_nn_model.parameters(), lr=config.learning_rate)

    # Attach optimizer to model for easy access in evaluate_and_log
    u_nn_model.optimizer = optimizer

    if config.lr_scheduler_type == 'ExponentialLR':
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_decay_gamma)
    elif config.lr_scheduler_type == 'MultiStepLR':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.lr_step_milestones, gamma=config.lr_step_gamma)
    else:
        lr_scheduler = None

    # Training Loop
    logger.log_message(f"Starting training for {config.num_epochs} epochs...")

    history = {
        'epochs_logged': [], 'total_loss': [], 'drm_loss': [], 'pinn_loss': [],
        'l2_error_u': [], 'h1_seminorm_error_u': [], 'h2_seminorm_error_u': [],
        'fourier_coeffs_nn_magnitudes': {}, 'fourier_coeffs_true_magnitudes': {},
        'fourier_coeffs_error_magnitudes': {}, 'fourier_frequencies_logged': None,
        'solution_snapshots_epochs': [],
    }
    eval_points_for_plot_np = None
    u_exact_plot_data_flat = None

    start_total_time = time.time()

    # Initial evaluation and checkpoint
    logger.log_message(f"--- Epoch 0 ---")
    eval_points_for_plot_np, u_exact_plot_data_flat = evaluate_and_log(
        epoch=0,
        u_nn_model=u_nn_model,
        history=history,
        config=config,
        f_exact_func=f_exact_func,
        u_exact_func=u_exact_func,
        logger=logger,
        rank_val=rank_val,
        eval_points_for_plot_np=eval_points_for_plot_np,
        u_exact_plot_data_flat=u_exact_plot_data_flat,
        full_uniform_grid_points=full_uniform_grid_points
    )
    torch.save(u_nn_model.state_dict(), os.path.join(rank_model_path, f'model_epoch_0.pth'))
    logger.log_message(f"Checkpoint saved at epoch 0")

    for epoch in range(1, config.num_epochs + 1):
        u_nn_model.train()
        optimizer.zero_grad()

        # Mini-batching for domain points from the pre-generated grid
        train_batch_size = min(config.batch_size, total_domain_points_in_grid)
        indices = torch.randperm(total_domain_points_in_grid)[:train_batch_size]
        sampled_domain_points_batch = full_uniform_grid_points[indices].requires_grad_(True).to(config.device)

        # Calculate losses using the sampled batches
        drm_loss = calculate_drm_loss(u_nn_model, f_exact_func, sampled_domain_points_batch)
        pinn_loss = calculate_pinn_loss(u_nn_model, f_exact_func, sampled_domain_points_batch, config.domain_dim)
        
        total_loss = config.drm_weight * drm_loss + config.pinn_weight * pinn_loss

        total_loss.backward()
        optimizer.step()

        if lr_scheduler:
            lr_scheduler.step()

        # Log losses for current epoch
        history['total_loss'].append(total_loss.item())
        history['drm_loss'].append(drm_loss.item())
        history['pinn_loss'].append(pinn_loss.item())

        if epoch % config.logging_freq == 0: # Check for logging frequency
            logger.log_message(f"--- Epoch {epoch} ---")
            eval_points_for_plot_np, u_exact_plot_data_flat = evaluate_and_log(
                epoch=epoch,
                u_nn_model=u_nn_model,
                history=history,
                config=config,
                f_exact_func=f_exact_func,
                u_exact_func=u_exact_func,
                logger=logger,
                rank_val=rank_val,
                eval_points_for_plot_np=eval_points_for_plot_np,
                u_exact_plot_data_flat=u_exact_plot_data_flat,
                full_uniform_grid_points=full_uniform_grid_points
            )

        if epoch % config.checkpoint_freq == 0: # Check for checkpoint frequency
            torch.save(u_nn_model.state_dict(), os.path.join(rank_model_path, f'model_epoch_{epoch}.pth'))
            logger.log_message(f"Checkpoint saved at epoch {epoch}")

    end_total_time = time.time()
    total_duration = end_total_time - start_total_time
    logger.log_message(f"Training finished in {total_duration:.2f} seconds.")
    torch.save(u_nn_model.state_dict(), os.path.join(rank_model_path, 'model_final.pth'))
    
    # Return the log directory for plotting (which is unique for this rank)
    rank_log_dir_for_plotting = logger.log_dir
    logger.close() 

    # Return state_dict and history for gathering by rank 0
    return u_nn_model.state_dict(), history


# --- Main Execution Block with Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Poisson PINN experiments with configurable parameters.')

    # Core experiment parameters
    parser.add_argument('--dim', type=int, default=1, choices=[1, 2, 3],
                        help='Dimension of the Poisson problem (1, 2, or 3).')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed for reproducibility; actual seed will be seed + rank.')
    parser.add_argument('--case', type=int, default=1,
                        help='Manufactured solution case number.')

    # Neural Network Architecture
    parser.add_argument('--hidden_neurons', type=int, nargs='+', default=[30, 30, 30],
                        help='List of integers for the number of neurons in each hidden layer.')
    parser.add_argument('--activation', type=str, nargs='+', default=['tanh'],
                        help='Activation function(s). Can be a single string (e.g., "relu") or a list (e.g., "tanh" "relu").')

    # Loss weights
    parser.add_argument('--drm_weight', type=float, default=0.0,
                        help='Weight for the DRM loss term.')
    parser.add_argument('--pinn_weight', type=float, default=1.0,
                        help='Weight for the PINN (PDE residual) loss term.')

    # Sampling parameters
    parser.add_argument('--num_uniform_partition', type=int, default=64,
                        help='Number of subintervals along each dimension for uniform partitioning.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Number of points in each batch.')
    
    # Optimization
    parser.add_argument('--optimizer_type', type=str, default='Adam', choices=['Adam', 'SGD', 'LBFGS'],
                        help='Type of optimizer to use (e.g., Adam, SGD, LBFGS).')
    parser.add_argument('--lr_scheduler_type', type=str, default='MultiStepLR', choices=['ExponentialLR', 'MultiStepLR', 'None'],
                        help='Type of learning rate scheduler.')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.999,
                        help='Gamma for ExponentialLR (decay rate per epoch).')
    parser.add_argument('--lr_step_milestones', type=int, nargs='+', default=[2000, 4000],
                        help='List of epochs when learning rate should drop for MultiStepLR.')
    parser.add_argument('--lr_step_gamma', type=float, default=0.1,
                        help='Factor by which to multiply the learning rate at milestones for MultiStepLR.')

    # Logging and plotting
    parser.add_argument('--logging_freq', type=int, default=50,
                        help='Frequency (in epochs) to log metrics and save snapshots.')
    parser.add_argument('--plot_resolution', type=int, default=256,
                        help='Resolution for plotting grids')
    parser.add_argument('--log_fourier_coeffs', type=bool, default=True,
                    help='Whether to log Fourier coefficients in the real basis.')
    parser.add_argument('--fourier_freq', type=int, nargs='+', default=[1, 4, 9],
                        help='Frequency of Fourier coefficients to log.')
    parser.add_argument('--base_log_dir', type=str, default='logs',
                        help='Base directory for saving experiment logs.')
    parser.add_argument('--base_model_save_dir', type=str, default='models',
                        help='Base directory for saving model checkpoints.')
    parser.add_argument('--checkpoint_freq', type=int, default=1000,
                        help='Frequency (in epochs) to save model checkpoints.')

    # 3D specific plotting (if dim=3)
    parser.add_argument('--slice_plane_dim', type=int, default=2, choices=[0, 1, 2],
                        help='For 3D, which dimension to slice (0 for x, 1 for y, 2 for z).')
    parser.add_argument('--slice_plane_val', type=float, default=0.5,
                        help='For 3D, value along the sliced dimension (e.g., z=0.5).')

    args = parser.parse_args()

    # Special handling for activation argument if it's a single string vs. a list
    if len(args.activation) == 1:
        parsed_activation = args.activation[0]
    else:
        parsed_activation = args.activation

    # Define common configuration arguments, including device
    base_config_kwargs = {
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'domain_dim': args.dim,
        'case_number': args.case,
        'hidden_neurons': args.hidden_neurons,
        'activation': parsed_activation,
        'drm_weight': args.drm_weight,
        'pinn_weight': args.pinn_weight,
        'num_epochs': args.epochs,
        'learning_rate': args.lr,
        'num_uniform_partition': args.num_uniform_partition,
        'batch_size': args.batch_size,
        'random_seed': args.seed,
        'optimizer_type': args.optimizer_type,
        'lr_scheduler_type': args.lr_scheduler_type,
        'lr_decay_gamma': args.lr_decay_gamma,
        'lr_step_milestones': args.lr_step_milestones,
        'lr_step_gamma': args.lr_step_gamma,
        'plot_resolution': args.plot_resolution,
        'log_fourier_coefficients': args.log_fourier_coeffs,
        'fourier_freq': args.fourier_freq,
        'base_log_dir': args.base_log_dir,
        'base_model_save_dir': args.base_model_save_dir,
        'logging_freq': args.logging_freq,
        'checkpoint_freq': args.checkpoint_freq,
        'slice_plane_dim': args.slice_plane_dim,
        'slice_plane_val': args.slice_plane_val
    }

    # Set domain bounds based on dimension
    if args.dim == 1:
        base_config_kwargs['domain_bounds'] = (0.0, 1.0)
    elif args.dim == 2:
        base_config_kwargs['domain_bounds'] = ((0.0, 1.0), (0.0, 1.0))
    elif args.dim == 3:
        base_config_kwargs['domain_bounds'] = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))

    # Create the config object for THIS rank
    rank_config = PoissonSolverConfig(**base_config_kwargs)

    print(f"Rank {rank}: Starting its experiment run{rank+1}.")
    model_state_dict, history = run_experiment(rank_config, rank)
    print(f"Rank {rank}: Finished its experiment run{rank+1}.")

    # Each rank will now trigger its own plotting functions
    print(f"Rank {rank}: Generating individual plots for run{rank+1}.")
    plot_solution_video(args.base_log_dir, rank_config, rank,
                        output_filename=f'solution_evolution_run{rank+1}.mp4')
    plot_norm_errors(args.base_log_dir, history, rank_config,
                    output_filename=f'norm_errors_evolution_run{rank+1}.png')
    if args.log_fourier_coeffs:
        plot_fourier_coefficients(args.base_log_dir, history, rank_config,
                                  output_filename=f'fourier_coefficients_evolution_run{rank+1}.png')
        plot_fourier_coefficient_errors(args.base_log_dir, history, rank_config,
                                        output_filename=f'fourier_coefficient_errors_evolution_run{rank+1}.png')
    print(f"Rank {rank}: Finished generating individual plots for run{rank+1}.")


    # MPI Synchronization and Gathering
    if size > 1:
        print(f"Rank {rank}: Gathering results for ensemble visualization")
        all_histories = comm.gather(history, root=0)

        if rank == 0:
            print(f"Rank 0: Generating Ensemble Visualizations for {size} runs in total)")
            plot_ensemble_norm_errors(args.base_log_dir, all_histories, rank_config,
                                      output_filename=f'ensemble_norm_errors.png')
            print("Rank 0: Ensemble plots complete.")

    comm.Barrier()
    print(f"Rank {rank}: Exiting.")
