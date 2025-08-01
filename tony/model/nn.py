import torch
import torch.nn as nn
import math

class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class NNModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_neurons, activation='tanh',
                 g0_func=None, d_func=None,
                 use_chebyshev_basis=False, chebyshev_freq_min=-1, chebyshev_freq_max=-1):
        super(NNModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.g0_func = g0_func
        self.d_func = d_func

        self.use_chebyshev_basis = use_chebyshev_basis
        self.chebyshev_freq_min = chebyshev_freq_min
        self.chebyshev_freq_max = chebyshev_freq_max

        self.activation_fns = []
        if isinstance(activation, str):
            # All layers have the same activation function
            self.activation_fns = [self._get_activation_fn(activation)] * len(hidden_neurons)
        elif isinstance(activation, (list, tuple)):
            # Each layer has its own activation function
            if len(activation) == len(hidden_neurons):
                self.activation_fns = [self._get_activation_fn(act_str) for act_str in activation]
            elif len(activation) == len(hidden_neurons) + 1:
                print("Warning: Activation list length is one longer than hidden layers. "
                      "Assuming the last activation is intended for an output layer, "
                      "but for regression problems, the output layer typically has no activation in this model.")
                self.activation_fns = [self._get_activation_fn(act_str) for act_str in activation[:-1]]
            else:
                raise ValueError(f"Activation list length ({len(activation)}) must match "
                                 f"hidden_neurons length ({len(hidden_neurons)}), or be one longer.")
        else:
            raise ValueError("Activation must be a string or a list/tuple of strings.")

        # Determine the effective input dimension for the trainable layers
        effective_input_dim = input_dim
        if self.use_chebyshev_basis and self.input_dim >= 1: # Can be applied to the first dimension for multi-D
            # Calculate total features: Chebyshev features for dim 0 + remaining raw dimensions
            num_chebyshev_features = self.chebyshev_freq_max - self.chebyshev_freq_min + 1
            effective_input_dim = num_chebyshev_features + (self.input_dim - 1 if self.input_dim > 1 else 0)

        layers = []
        prev_neurons = effective_input_dim
        for i, num_neurons in enumerate(hidden_neurons):
            layers.append(nn.Linear(prev_neurons, num_neurons))
            layers.append(self.activation_fns[i])
            prev_neurons = num_neurons
            
        layers.append(nn.Linear(prev_neurons, output_dim)) # Output layer
        self.layers = nn.Sequential(*layers)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _get_activation_fn(self, activation_str):
        if activation_str == 'relu':
            return nn.ReLU()
        elif activation_str == 'sin':
            return SinActivation()
        elif activation_str == 'tanh':
            return nn.Tanh()
        elif activation_str == 'sigmoid':
            return nn.Sigmoid()
        elif activation_str == 'elu':
            return nn.ELU()
        elif activation_str == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation_str == 'linear' or activation_str is None:
            return nn.Identity()
        else:
            raise ValueError(f"Unknown activation function: {activation_str}")
    
    def forward(self, x_raw):
        x = x_raw.float()
        if self.use_chebyshev_basis:
            chebyshev_arg = torch.cos(math.pi * x[:, 0]) # Argument for U_k(x)
            chebyshev_features = []
            # Initialize for recurrence
            uk_minus_2 = torch.ones_like(chebyshev_arg) # U_0(x_mapped)
            uk_minus_1 = 2 * chebyshev_arg             # U_1(x_mapped)
            for k in range(self.chebyshev_freq_min, self.chebyshev_freq_max + 1):
                if k == 0:
                    current_uk = uk_minus_2 # This is U_0
                elif k == 1:
                    current_uk = uk_minus_1 # This is U_1
                else:
                    # Compute U_k using the recurrence relation
                    current_uk = 2 * chebyshev_arg * uk_minus_1 - uk_minus_2
                    # Update for next iteration
                    uk_minus_2 = uk_minus_1
                    uk_minus_1 = current_uk
                chebyshev_features.append(current_uk.unsqueeze(1))
            processed_input = torch.cat(chebyshev_features, dim=1)
            if self.input_dim > 1: # Append other raw dimensions if multi-D
                processed_input = torch.cat([processed_input, x[:, 1:]], dim=1)
        else:
            processed_input = x
        raw_nn_output = self.layers(processed_input)

        # Incorporate boundary conditions if g0_func and d_func are provided
        if self.g0_func is None or self.d_func is None:
            return raw_nn_output
        else:
            return self.g0_func(x_raw) + self.d_func(x_raw) * raw_nn_output
