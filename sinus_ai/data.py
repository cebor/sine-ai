"""Data generation utilities for training and testing."""

import numpy as np
import torch
from . import config


def generate_training_data(
    n_samples: int | None = None,
    x_min: float | None = None,
    x_max: float | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates training and test data for the sine function.
    
    Args:
        n_samples: Number of samples to generate (default from config)
        x_min: Minimum x value (default from config)
        x_max: Maximum x value (default from config)
        
    Returns:
        Tuple of (x_tensor, y_tensor) as PyTorch tensors
    """
    if n_samples is None:
        n_samples = config.N_SAMPLES_TRAIN
    if x_min is None:
        x_min = config.X_MIN
    if x_max is None:
        x_max = config.X_MAX
    
    x = np.linspace(x_min, x_max, n_samples)
    y = np.sin(x)
    
    # Convert to PyTorch tensors
    x_tensor = torch.FloatTensor(x).reshape(-1, 1).to(config.DEVICE)
    y_tensor = torch.FloatTensor(y).reshape(-1, 1).to(config.DEVICE)
    
    return x_tensor, y_tensor
