import torch
from torch import nn
import numpy as np


class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first, omega_0=30):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.omega_0 = omega_0
        self.in_features = in_features

        if is_first:
            nn.init.uniform_(self.linear.weight, -1 / in_features, 1 / in_features)
        else:
            nn.init.uniform_(self.linear.weight, -np.sqrt(6 / self.in_features) / self.omega_0,
                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, x):
        return torch.sin(self.linear(self.omega_0 * x))


class Siren(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, hidden_layers, omega_0=30):
        super().__init__()
        self.hidden_features = hidden_features
        self.omega_0 = omega_0
        self.first_layer = SirenLayer(in_features=in_features, out_features=hidden_features, is_first=True,
                                      omega_0=omega_0)
        self.hidden_layers = nn.ModuleList(
            [SirenLayer(in_features=hidden_features, out_features=hidden_features, is_first=False, omega_0=omega_0) for
             _ in range(hidden_layers)]
        )
        self.final_layer = nn.Linear(hidden_features, out_features)

        nn.init.uniform_(self.final_layer.weight, -np.sqrt(6 / self.hidden_features) / self.omega_0,
                         np.sqrt(6 / self.hidden_features) / self.omega_0)

    def forward(self, model_input):
        coords = model_input['coords'].clone().detach().requires_grad_(True)
        inputs = self.first_layer(coords)
        for layer in self.hidden_layers:
            inputs = layer(inputs)
        return {'model_out': self.final_layer(inputs), 'model_in':coords}


if __name__ == "__main__":
    model = Siren(2, 1, 256, 3, True)
    print(model)
