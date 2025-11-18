"""Neural network model for sine function approximation."""

import torch
import torch.nn as nn
from . import config


class SineNet(nn.Module):
    """Simple neural network to approximate the sine function."""
    
    def __init__(
        self,
        hidden_size_1=config.HIDDEN_SIZE_1,
        hidden_size_2=config.HIDDEN_SIZE_2,
        hidden_size_3=config.HIDDEN_SIZE_3,
    ):
        super(SineNet, self).__init__()
        self.fc1 = nn.Linear(1, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, hidden_size_3)
        self.fc4 = nn.Linear(hidden_size_3, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
