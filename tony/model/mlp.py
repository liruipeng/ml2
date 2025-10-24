import torch
import torch.nn as nn
import math

class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class CosActivation(nn.Module):
    def forward(self, x):
        return torch.cos(x)

class NNModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_neurons, activation='tanh',
                 g0_func=None, d_func=None,
                 use_positional_encoding=False, pe_freq_min=0, pe_freq_max=6,
                 use_chebyshev_basis=False, chebyshev_freq_min=-1, chebyshev_freq_max=-1):
        super(NNModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.g0_func = g0_func
        self.d_func = d_func

        self.use_positional_encoding = use_positional_encoding
        self.pe_freq_min = pe_freq_min 
        self.pe_freq_max = pe_freq_max

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
        if self.use_positional_encoding:
            trainable_input_dim = input_dim * (self.pe_freq_max - self.pe_freq_min + 1)
        elif self.use_chebyshev_basis:
            trainable_input_dim = self.chebyshev_freq_max - self.chebyshev_freq_min + 1
        else:
            trainable_input_dim = input_dim

        if self.use_chebyshev_basis:
            trainable_output_dim = self.chebyshev_freq_max - self.chebyshev_freq_min + 1
        else:
            trainable_output_dim = output_dim

        layers = []
        prev_neurons = trainable_input_dim
        for i, num_neurons in enumerate(hidden_neurons):
            layers.append(nn.Linear(prev_neurons, num_neurons))
            layers.append(self.activation_fns[i])
            prev_neurons = num_neurons
        layers.append(nn.Linear(prev_neurons, trainable_output_dim)) # Output layer
        self.layers = nn.Sequential(*layers)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialize weights for the final combiner
        if use_chebyshev_basis:
            self.final_combiner = nn.Linear(trainable_output_dim, output_dim, bias=False)
            nn.init.xavier_normal_(self.final_combiner.weight)

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
        elif activation_str == 'silu':
            return nn.SiLU()
        elif activation_str == 'linear' or activation_str is None:
            return nn.Identity()
        else:
            raise ValueError(f"Unknown activation function: {activation_str}")
    
    def _positional_encode(self, x):
        """
        Applies positional encoding to the input tensor x.
        Encodes each dimension of x by concatenating sin and cos features at
        frequencies 2^0*pi, 2^1*pi, ..., 2^(pe_freq_max-1)*pi.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            
        Returns:
            torch.Tensor: Encoded tensor of shape (batch_size, input_dim * pe_freq_max * 2).
        """
        pe_features = []
        for i in range(self.pe_freq_min, self.pe_freq_max + 1):
            freq_scale = 2**i * math.pi 
            pe_features.append(torch.cos(freq_scale * x))
        
        # Concatenate features along the last dimension (for each input_dim)
        return torch.cat(pe_features, dim=-1)

    def _generate_chebyshev_features(self, x):
        chebyshev_features = []
        theta = math.pi * x
        dx = torch.sin(theta)
        left_end = torch.abs(theta) < 1e-8
        right_end = torch.abs(theta - math.pi) < 1e-8
        chebyshev_arg = torch.cos(math.pi * x)
        uk_minus_2 = torch.sin((self.chebyshev_freq_min + 1) * math.pi * x) / dx 
        uk_minus_2[left_end] = self.chebyshev_freq_min + 1
        uk_minus_2[right_end] = (self.chebyshev_freq_min + 1) * (-1)**self.chebyshev_freq_min
        uk_minus_1 = torch.sin((self.chebyshev_freq_min + 2) * math.pi * x) / dx 
        uk_minus_1[left_end] = self.chebyshev_freq_min + 2
        uk_minus_1[right_end] = (self.chebyshev_freq_min + 2) * (-1)**(self.chebyshev_freq_min + 1)
        for k in range(self.chebyshev_freq_min, self.chebyshev_freq_max + 1):
            if k == self.chebyshev_freq_min:
                current_uk = uk_minus_2
            elif k == self.chebyshev_freq_min + 1:
                current_uk = uk_minus_1
            else:
                current_uk = 2 * chebyshev_arg * uk_minus_1 - uk_minus_2
                uk_minus_2 = uk_minus_1
                uk_minus_1 = current_uk
            chebyshev_features.append(current_uk.unsqueeze(1))
        return torch.cat(chebyshev_features, dim=1)

    def forward(self, x_raw):
        x = x_raw.float()
        if self.use_positional_encoding:
            x = self._positional_encode(x)
        elif self.use_chebyshev_basis:
            chebyshev_features_s = self._generate_chebyshev_features(x[:, 0])
            y = self.layers(chebyshev_features_s)
            if y.shape[1] != chebyshev_features_s.shape[1]:
                raise RuntimeError("Mismatch between learned features and Chebyshev features dimensions.")
            combined_features = chebyshev_features_s + y
            raw_nn_output = self.final_combiner(combined_features)
        else:
            raw_nn_output = self.layers(x)

        # Incorporate boundary conditions if g0_func and d_func are provided
        if self.g0_func is None or self.d_func is None:
            return raw_nn_output
        else:
            return self.g0_func(x_raw) + self.d_func(x_raw) * raw_nn_output
