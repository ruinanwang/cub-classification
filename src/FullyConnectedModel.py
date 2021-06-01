import torch
import torch.nn as nn

class FullyConnectedModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2):
        super(FullyConnectedModel, self).__init__()
        
        if num_layers == 1:
            self.model = nn.Linear(input_size, num_classes)
        elif num_layers == 2:
            self.model = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_classes)
            )
        elif num_layers == 3:
            self.model = nn.Sequential(
                if type(hidden_size) == int:
                    hidden_size = [hidden_size, hidden_size]
                nn.Linear(input_size, hidden_size[0]),
                nn.ReLU(),
                nn.Linear(hidden_size[0], hidden_size[1]),
                nn.ReLU(),
                nn.Linear(hidden_size[1], num_classes)
            )
        
    def forward(self, x):
        out = self.model(x)
        return out