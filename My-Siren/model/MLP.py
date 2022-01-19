import torch
from torch import nn


class Layer(nn.Module):
    def __init__(self, in_features, out_features, nonlinearity):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

        if nonlinearity == 'relu':
            nn.init.kaiming_normal_(self.linear.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        else:
            nn.init.xavier_normal_(self.linear.weight)

        activation_map = {
            'relu':nn.ReLU(),
            'tanh':nn.Tanh(),
            'sigmoid':nn.Sigmoid()
        }
        self.activation = activation_map[nonlinearity]

    def forward(self, x):
        return self.activation(self.linear(x))


class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, hidden_layers, nonlinearity):
        super().__init__()
        self.hidden_features = hidden_features
        self.first_layer = Layer(in_features, hidden_features, nonlinearity)
        self.hidden_layers = nn.ModuleList(
            [Layer(hidden_features, hidden_features, nonlinearity) for
             _ in range(hidden_layers)]
        )
        self.final_layer = nn.Linear(hidden_features, out_features)

        if nonlinearity == 'relu':
            nn.init.kaiming_normal_(self.final_layer.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        else:
            nn.init.xavier_normal_(self.final_layer.weight)

    def forward(self, model_input):
        coords = model_input['coords'].clone().detach().requires_grad_(True)
        inputs = self.first_layer(coords)
        for layer in self.hidden_layers:
            inputs = layer(inputs)
        return {'model_out': self.final_layer(inputs), 'model_in':coords}

if __name__ == "__main__":
    model = MLP(in_features=2, out_features=1, hidden_features=256, hidden_layers=3, nonlinearity="relu")
    print(model)

