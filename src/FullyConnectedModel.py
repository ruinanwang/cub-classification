import torch
import torch.nn as nn

class FullyConnectedModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FullyConnectedModel, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        out = self.model(x)
        return out