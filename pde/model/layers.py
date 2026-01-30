import torch
import torch.nn as nn
from cheby import generate_chebyshev_features

class ChebyshevLayer(nn.Module):
    """
    A layer that transforms input coordinates into Chebyshev polynomial 
    features of the second kind (U_k) using the logic in cheby.py.
    """
    def __init__(self, freq_min: int, freq_max: int):
        super().__init__()
        self.freq_min = freq_min
        self.freq_max = freq_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If frequencies are not set (e.g., -1), return raw input
        if self.freq_max < 0:
            return x
        return generate_chebyshev_features(x, self.freq_min, self.freq_max)

class GatedLevel(nn.Module):
    """
    Handles the scalar gating for each level. 
    Matches the logic in your pinn_1d.py.
    """
    def __init__(self, init_value: float = 1.0, trainable: bool = True):
        super().__init__()
        self.gate = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32), 
            requires_grad=trainable
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate * x

class AdaptiveScale(nn.Module):
    """
    Scales inputs (useful for MscaleDNN frequency shifting).
    """
    def __init__(self, init_scale: float = 1.0):
        super().__init__()
        self.scale = init_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale
