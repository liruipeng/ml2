import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional
from enum import Enum
from .cheby import generate_chebyshev_features # Assuming relative import

class LevelStatus(Enum):
    OFF = "off"
    TRAIN = "train"
    FROZEN = "frozen"

class Level(nn.Module):
    def __init__(self, dim_inputs: int, dim_outputs: int, dim_hidden: List[int],
                 act: nn.Module = nn.Tanh(),
                 use_chebyshev_basis: bool = False,
                 chebyshev_freq_min: int = -1,
                 chebyshev_freq_max: int = -1) -> None:
        """
        A single MLP level within the multi-level architecture.
        """
        super().__init__()
        self.use_chebyshev_basis = use_chebyshev_basis
        self.chebyshev_freq_min = chebyshev_freq_min
        self.chebyshev_freq_max = chebyshev_freq_max
        
        # Determine input dimension for the first layer
        if self.use_chebyshev_basis and chebyshev_freq_max >= 0:
            current_dim = chebyshev_freq_max - chebyshev_freq_min + 1
        else:
            current_dim = dim_inputs

        layers = []
        for h_dim in dim_hidden:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(act)
            current_dim = h_dim
        
        layers.append(nn.Linear(current_dim, dim_outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_chebyshev_basis and self.chebyshev_freq_max >= 0:
            x = generate_chebyshev_features(x, self.chebyshev_freq_min, self.chebyshev_freq_max)
        return self.net(x)

class MultiLevelNN(nn.Module):
    def __init__(self, 
                 num_levels: int,
                 dim_inputs: int, 
                 dim_outputs: int, 
                 dim_hidden: List[int],
                 act: nn.Module = nn.ReLU(),
                 use_chebyshev_basis: bool = False,
                 chebyshev_freq_min: Optional[np.ndarray] = None,
                 chebyshev_freq_max: Optional[np.ndarray] = None,
                 init_frozen: bool = False) -> None:
        """
        Container for multiple Level objects with gating logic.
        Note: BC enforcement and Mesh logic are moved to higher-level handlers.
        """
        super().__init__()
        self.init_frozen = init_frozen
        self.num_levels = num_levels
        self.dim_outputs = dim_outputs
        
        # Build levels
        self.levels = nn.ModuleList([
            Level(dim_inputs, dim_outputs, dim_hidden, act,
                  use_chebyshev_basis,
                  chebyshev_freq_min[i] if chebyshev_freq_min is not None else -1,
                  chebyshev_freq_max[i] if chebyshev_freq_max is not None else -1)
            for i in range(num_levels)
        ])

        # Initialize gates and status
        self.gates = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0 if (not init_frozen or i == 0) else 0.0), 
                         requires_grad=False)
            for i in range(num_levels)
        ])
        
        self.level_status = [LevelStatus.FROZEN if init_frozen else LevelStatus.OFF] * num_levels
        self.scales = [1.0] * num_levels

    def set_status(self, level_idx: int, status: LevelStatus):
        """Sets training status and toggles gradient requirements."""
        self.level_status[level_idx] = status
        is_training = (status == LevelStatus.TRAIN)
        
        # Toggle level weights
        for param in self.levels[level_idx].parameters():
            param.requires_grad = is_training
            
        # Toggle gate training (only if init_frozen was requested)
        if self.init_frozen:
            self.gates[level_idx].requires_grad = is_training

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the weighted sum of all active levels.
        """
        total_output = torch.zeros((x.shape[0], self.dim_outputs), device=x.device)
        
        for i, level in enumerate(self.levels):
            if self.level_status[i] != LevelStatus.OFF:
                # Apply per-level input scaling
                x_scaled = x * self.scales[i]
                
                # Weighted contribution
                total_output += self.gates[i] * level(x_scaled)
                
        return total_output
