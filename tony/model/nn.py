import torch
import torch.nn as nn

class NNModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_neurons, activation='tanh', g0_func=None, d_func=None):
        super(NNModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.g0_func = g0_func
        self.d_func = d_func

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

        layers = []

        prev_neurons = input_dim
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
        if activation_str == 'sin':
            return nn.sin()
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

        raw_nn_output = self.layers(x)

        # Incorporate boundary conditions
        if self.g0_func is None or self.d_func is None:
            return raw_nn_output
        else:
            return self.g0_func(x) + self.d_func(x) * raw_nn_output
