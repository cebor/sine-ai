"""Data generation utilities for training and testing."""

import numpy as np
import torch


def generate_training_data(n_samples=1000):
    """Generates training and test data for the sine function.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Tuple of (x_tensor, y_tensor) as PyTorch tensors
    """
    x = np.linspace(0, 2 * np.pi, n_samples)
    y = np.sin(x)
    
    # Convert to PyTorch tensors
    x_tensor = torch.FloatTensor(x).reshape(-1, 1)
    y_tensor = torch.FloatTensor(y).reshape(-1, 1)
    
    return x_tensor, y_tensor
