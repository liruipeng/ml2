import os
import torch
import numpy as np
from datetime import datetime

from physics.library_1d import Problem1, MScaleProblem
from solvers.mesh import Mesh
from solvers.losses import LossFactory
from solvers.trainer import Trainer
from model.arch import MultiLevelNN, LevelStatus
from model.bc import BoundaryEnforcedModel
from utils.control import parse_args, get_activation, print_args, clean_files
from utils.analyze import fourier_analysis, error_analysis
from utils.visualize import save_frame, make_video_from_frames, plot_error_evolution, plot_coefficient_evolution

def main():
    # Initialize Environment and Parse Arguments
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    # Register run by timestamp and setup directories
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"results_poisson_1d_{ts}"
    frame_dir = os.path.join(run_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)
    
    if args.clear:
        cleanfiles(frame_dir)
    
    print_args(args, output_file=f"{run_dir}/args.txt")

    # Define Physics Problem
    if args.problem_id == 1:
        problem = Problem1(high_freq=args.high_freq, gamma=args.gamma, device=device)
    else:
        problem = MScaleProblem(mu=args.mu, gamma=args.gamma, device=device)

    # Build Multi-Level Model
    if args.use_chebyshev_basis:
        cheby_min = np.array(args.chebyshev_freq_min) if len(args.chebyshev_freq_min) > 1 \
                    else np.full(args.levels, args.chebyshev_freq_min[0])
        cheby_max = np.array(args.chebyshev_freq_max) if len(args.chebyshev_freq_max) > 1 \
                    else np.full(args.levels, args.chebyshev_freq_max[0])
    else:
        cheby_min = cheby_max = np.full(args.levels, -1)

    base_nn = MultiLevelNN(
        num_levels=args.levels,
        dim_inputs=1,
        dim_outputs=1,
        dim_hidden=args.hidden_dims,
        act=get_activation(args.activation),
        use_chebyshev_basis=args.use_chebyshev_basis,
        chebyshev_freq_min=cheby_min,
        chebyshev_freq_max=cheby_max,
        init_frozen=args.init_frozen
    )

    # Wrap for Exact BC enforcement
    model = BoundaryEnforcedModel(
        base_model=base_nn,
        problem=problem,
        g0_type=args.bc_extension,
        d_type=args.distance
    ).to(device)

    # Initialize Trainer and Loss
    loss_map = {0: "pinn", 1: "ritz", -1: "supervised"}
    loss_fn = LossFactory.get_loss(loss_map.get(args.loss_type, "pinn"))
    
    # Mesh for training/evaluation
    mesh = Mesh(dim=1, bounds=problem.domain_bounds, device=device)
    
    trainer = Trainer(model, problem, mesh, loss_fn, lr=args.lr[0], device=device)

    # Training Loop with Analysis & Visualization
    u_analytic = problem.u_exact(mesh.get_test_grid(args.nx_eval))
    xf_analytic, uf_analytic, _, _ = fourier_analysis(
        mesh.get_test_grid(args.nx_eval).cpu().numpy(), 
        u_analytic.cpu().numpy(), 
        args.enforce_bc
    )

    tracked_data = []

    for s in range(args.sweeps):
        for lev in range(args.levels):
            print(f"\n--- Sweep {s} | Training Level {lev} ---")
            model.base_model.scales[lev] = lev + 1 # Input scaling for frequencies
            
            # Helper to run analysis during training
            def analysis_callback(epoch, loss_val):
                model.eval()
                with torch.no_grad():
                    x_test = mesh.get_test_grid(args.nx_eval)
                    u_pred = model(x_test)
                    u_train = model(mesh.generate_interior_points(args.nx[-1]))
                    
                    # Error Metrics
                    L2_err, H1_err, H2_err = error_analysis(x_test, u_analytic, model)
                    
                    # Fourier Analysis
                    xf, uf, _, _ = fourier_analysis(x_test.cpu().numpy(), u_pred.cpu().numpy(), args.enforce_bc)
                    
                    # Store data for evolution plots
                    tracked_data.append({
                        'epoch': epoch, 'L2_error': L2_err, 'H1_error': H1_err, 'H2_error': H2_err,
                        'nn_coeffs': uf[args.track_freqs] if args.track_freqs else None,
                        'true_coeffs': uf_analytic[args.track_freqs] if args.track_freqs else None
                    })

                    # Save Visualization Frames
                    save_frame(x=x_test.cpu().numpy(), t=u_analytic.cpu().numpy(), y=u_pred.cpu().numpy(),
                               xs=None, ys=None, iteration=[s, lev, epoch], 
                               title="Model_Outputs", frame_dir=frame_dir)
                model.train()

            # Execute training for the level
            epochs = args.epochs[lev] if len(args.epochs) > 1 else args.epochs[0]
            trainer.train_single_level(
                level_idx=lev, 
                epochs=epochs, 
                n_points=args.nx[-1],
                callback=analysis_callback if args.plot else None
            )

    # Final Analysis and Video Generation
    if args.plot:
        print("\nGenerating Videos...")
        make_video_from_frames(frame_dir=frame_dir, name_prefix="Model_Outputs", output_file=f"{run_dir}/Solution.mp4")
        
        if args.track_freqs:
            plot_error_evolution(tracked_data, sweep_idx=s, level_idx=lev, frame_dir=run_dir)
            plot_coefficient_evolution(tracked_data, freqs=args.track_freqs, sweep_idx=s, 
                                        level_idx=lev, frame_dir=run_dir, analytic_freqs=xf_analytic[args.track_freqs])

    metrics = trainer.evaluate(n_test=args.nx_eval)
    print(f"\nFinal L2 Relative Error: {metrics['l2']:.6e}")

if __name__ == "__main__":
    main()
