import numpy as np
import torch

class PoissonSolverConfig:
    def __init__(self, **kwargs):
        device_arg = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        if isinstance(device_arg, str):
            self.device = torch.device(device_arg)
        elif isinstance(device_arg, torch.device):
            self.device = device_arg
        else:
            raise TypeError(f"Device must be a string or torch.device object, got {type(device_arg)}")

        # Domain parameters
        self.domain_dim = kwargs.get('domain_dim', 1)  # 1, 2, or 3
        self.domain_bounds = kwargs.get('domain_bounds', (0.0, 1.0))
        if isinstance(self.domain_bounds[0], (int, float)):
            self.domain_bounds = tuple((self.domain_bounds[0], self.domain_bounds[1]) for _ in range(self.domain_dim))

        # Manufactured Solution (integer case number)
        self.case_number = kwargs.get('case_number', 1)

        # Neural Network Architecture
        self.hidden_neurons = kwargs.get('hidden_neurons', [20, 20, 20]) # Array of numbers of intermediate neurons
        self.activation = kwargs.get('activation', 'tanh') # e.g., 'tanh', 'relu', 'sigmoid'

        # Loss Function Weights
        self.drm_weight = kwargs.get('drm_weight', 1.0)
        self.pinn_weight = kwargs.get('pinn_weight', 1.0)

        # Training Parameters
        self.num_epochs = kwargs.get('num_epochs', 5000)
        self.batch_size = kwargs.get('batch_size', 256)
        self.learning_rate = kwargs.get('learning_rate', 1e-3)
        self.optimizer_type = kwargs.get('optimizer_type', 'Adam')
        self.lr_scheduler_type = kwargs.get('lr_scheduler_type', 'ExponentialLR')
        self.lr_decay_gamma = kwargs.get('lr_decay_gamma', 0.9)
        self.lr_step_milestones = kwargs.get('lr_step_milestones', [2000, 4000])
        self.lr_step_gamma = kwargs.get('lr_step_gamma', 0.1)

        # Sampling Parameters
        self.num_uniform_partition = kwargs.get('num_uniform_partition', 10000)

        # Experiment Management
        self.random_seed = kwargs.get('random_seed', 42)
        self.base_log_dir = kwargs.get('base_log_dir', 'logs/poisson_solver')
        self.base_model_save_dir = kwargs.get('base_model_save_dir', 'models/poisson_solver')
        self.checkpoint_freq = kwargs.get('checkpoint_freq', 100)
        self.logging_freq = kwargs.get('logging_freq', 10)

        # Metric Logging for Fourier Coefficients
        self.log_fourier_coefficients = kwargs.get('log_fourier_coefficients', True)
        self.fourier_freq = kwargs.get('fourier_freq', [1, 4, 9]) # Array of frequencies to compute Fourier coefficients

        # Visualization parameters
        self.plot_resolution = kwargs.get('plot_resolution', 100)
        self.slice_plane_dim = kwargs.get('slice_plane_dim', 2)
        self.slice_plane_val = kwargs.get('slice_plane_val', 0.5)

    def to_dict(self):
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'): # Exclude private/protected attributes
                # Convert torch.device objects to string for JSON serialization
                if key == 'device' and isinstance(value, torch.device):
                    config_dict[key] = str(value)
                else:
                    config_dict[key] = value
        return config_dict
